"""Warm-core radiance measurements from native pixels at several spatial scales."""

from functools import cache, lru_cache

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# Nominal AHI band wavelengths, used only to linearize brightness-temperature mixtures.
WAVELENGTH_UM = np.array([6.24, 6.94, 7.35, 13.28])
WIDTHS = (1.0, 1.7, 2.8, 4.6, 7.5, 12.0, 21.0)
EARTH_TEMPERATURE_K = 270.0
HALF = 62


def radiance_ratio(kelvin):
    exponent = 14_387.77 / WAVELENGTH_UM[:, None, None]
    return np.expm1(exponent / EARTH_TEMPERATURE_K) / np.expm1(exponent / kelvin)


def masked_gaussian(values, valid, width):
    support = gaussian_filter(valid.astype(np.float32), width, mode="constant")
    smooth = gaussian_filter(np.where(valid, values, 0), (0, width, width), mode="constant")
    return np.divide(smooth, support, out=np.full_like(values, np.nan), where=support > 0.8)


@lru_cache(maxsize=1)
def local_coordinates():
    y, x = np.mgrid[-HALF : HALF + 1, -HALF : HALF + 1]
    return y, x, x**2 + y**2


@cache
def ring_offsets(low, high, step):
    axis = np.arange(-high, high + 1, step)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    inside = (x**2 + y**2 >= low**2) & (x**2 + y**2 < high**2)
    return y[inside], x[inside]


def patch_at(image, mask, y, x):
    yc = np.clip(y, 0, mask.shape[0] - 1)
    xc = np.clip(x, 0, mask.shape[1] - 1)
    valid = mask[yc, xc] & (y == yc) & (x == xc)
    temperature = image[:, yc, xc].astype(np.float32) * 0.01 + 150.0
    return temperature, valid


def warm_core(image, mask, xy):
    """Find a compact warm patch and measure its scale response in linearized radiance."""
    x, y = np.rint(np.asarray(xy) - 0.5).astype(int)
    dy, dx, radius2 = local_coordinates()
    temperature, valid = patch_at(image, mask, y + dy, x + dx)
    radiance = radiance_ratio(temperature.clip(150, 340))
    smoothed = np.stack([masked_gaussian(radiance, valid, w) for w in WIDTHS])
    # Response of a small core relative to the wider cloud field. The divisor
    # balances bands of disparate optical depths as a fraction of local radiance.
    relative = (smoothed[:-1] - smoothed[-1]) / np.maximum(smoothed[-1], 0.03)
    combined = np.einsum("b,sbxy->sxy", (0.14, 0.23, 0.37, 0.26), relative)
    # Penalize displacement from the supplied storm center and isolated hot pixels.
    score = np.where(radius2[None] <= 26**2, combined - 0.00040 * radius2, -np.inf)
    score = np.where(np.isfinite(score), score, -np.inf)
    if not np.isfinite(score).any():
        return (y, x), np.full(56, np.nan)
    scale, j, i = np.unravel_index(score.argmax(), score.shape)
    spectrum = smoothed[:, :, j, i]
    response = relative[:, :, j, i]
    features = np.r_[spectrum.ravel(), response.ravel(), dx[j, i], dy[j, i],
                     WIDTHS[scale], combined[scale, j, i]]
    return (y + dy[j, i], x + dx[j, i]), features


def measurements(image, mask, xy):
    center, core = warm_core(image, mask, xy)
    inner, wall, environment = [], [], []
    for low, high, step in ((0, 2, 1), (0, 4, 1), (0, 8, 1), (4, 12, 1), (8, 18, 1),
                            (14, 26, 1), (24, 42, 2), (40, 72, 3), (68, 113, 4),
                            (105, 165, 5), (155, 236, 6)):
        dy, dx = ring_offsets(low, high, step)
        patch, ok = patch_at(image, mask, center[0] + dy, center[1] + dx)
        stats = np.full(35, np.nan)
        if ok.sum() >= max(3, 0.55 * len(ok)):
            pixels = patch[:, ok]
            spectra = np.vstack((pixels, np.diff(pixels, axis=0)))
            quantiles = np.quantile(spectra, (0.03, 0.25, 0.5, 0.75, 0.97), axis=1)
            stats = quantiles.ravel()
        group = inner if low < 14 else wall if low < 68 else environment
        group.append(stats)
    eye = np.asarray(inner).reshape(-1, 5, 7)
    wall = np.asarray(wall).reshape(-1, 5, 7)
    outer = np.asarray(environment)
    contrast = (eye[:, 3:, :4] - wall[1:2, :2, :4]).ravel()
    return {"warm_radiance": core, "eye": eye.ravel(), "wall": wall.ravel(),
            "contrast": contrast, "environment": outer.ravel()}


def sequence(image, mask, centers, times):
    valid = mask.any(axis=(-1, -2)) & np.isfinite(times)
    slots = np.flatnonzero(valid)
    if not slots.size:
        raise ValueError("W1 requires at least one available scan")
    frames = [measurements(image[i], mask[i], centers[i]) for i in slots]
    values = {}
    for key, example in frames[0].items():
        values[key] = np.full((len(times), example.size), np.nan, dtype=np.float32)
        values[key][valid] = np.stack([frame[key] for frame in frames])
    return values


def common_kernel(log_mass, basis):
    return torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8.0


def quality_covariance(common, nuisance, settings, quality):
    # Fair and Poor can have separate systematic measurement errors. A new
    # record's prediction uses only the shared image-to-pressure relationship.
    needs_bias = quality > 0
    grouped = (quality[:, None] == quality[None, :]) & needs_bias[:, None]
    return (common + grouped * (settings[3].exp() * nuisance + settings[4].exp())
            + torch.diag(settings[:3].exp()[quality] + 0.002))


def latent_loo(common, inverse, dual):
    full_signal = common @ dual
    leverage = (common * inverse).sum(dim=1)
    return full_signal - leverage * dual / inverse.diag()


def objective(log_mass, settings, basis, nuisance, y, quality, weights, prior):
    common = common_kernel(log_mass, basis)
    combined = quality_covariance(common, nuisance, settings, quality)
    factor, _ = torch.linalg.cholesky_ex(combined)
    inverse = torch.cholesky_inverse(factor)
    dual = inverse @ y
    loss = (weights * (y - latent_loo(common, inverse, dual)).square()).mean()
    evidence = (y @ dual + 2 * factor.diag().log().sum()) / len(y)
    return loss + 0.14 * evidence + 0.005 * (log_mass - prior).square().mean()
