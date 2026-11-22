"""Image views for U1. No record labels are inputs to these transformations."""

import numpy as np
import torch

from h9dp.a1 import translated_codes
from h9dp.r1 import _normalize, summarize_time

WAVELENGTH_UM = np.array([6.2, 6.9, 7.3, 13.3])
RADII = (5, 10, 18)  # cells in the 2-km ground grid


def relative_radiance(kelvin):
    """Planck radiance divided by each band's radiance at 240 K (nominal wavelengths)."""
    c = 14387.77 / WAVELENGTH_UM[:, None]
    return np.expm1(c / 240.0) / np.expm1(c / kelvin)


def mixing_profile(patch, radius):
    """Co-located warm/cold endmembers; do not extrapolate to an unobserved clear eye."""
    kelvin = patch[:4].astype(np.float64) * 0.01 + 240.0
    valid = patch[4].astype(bool) & np.isfinite(kelvin).all(axis=0)
    profile = np.full((len(RADII), 17), np.nan)
    for i, edge in enumerate(RADII):
        disk = radius <= edge
        keep = valid & disk
        if keep.sum() < 0.8 * disk.sum():
            continue
        bt = kelvin[:, keep]
        rad = relative_radiance(bt)
        order = np.argsort(rad[3], kind="stable")
        size = len(order)
        # Use the same pixels in all channels, including for a very small warm cap.
        cold_idx, warm_idx = order[:max(3, size // 5)], order[-max(3, size // 20):]
        cold, warm = rad[:, cold_idx].mean(1), rad[:, warm_idx].mean(1)
        spread = max(warm[3] - cold[3], 0.025)
        slope = (warm[:3] - cold[:3]) / spread
        residual = rad[:3] - (cold[:3, None] + slope[:, None] * (rad[3] - cold[3]))
        fraction = ((rad[3] - cold[3]) / spread).clip(0, 1)
        profile[i] = np.r_[cold, warm, slope, np.sqrt(np.mean(residual**2, axis=1)),
                           fraction.mean(), np.mean(fraction > 0.75), fraction @ radius[keep] / max(fraction.sum(), 1e-8)]
    return profile.ravel()


def air_mass(field):
    """Colocated broad-cloud and warm-background spectra, not a 12–13 degree reference."""
    y, x = np.indices(field.shape[-2:]) - (np.array(field.shape[-2:]) - 1)[:, None, None] / 2
    radius = np.hypot(x, y)
    out = np.full((2, 5, 7), np.nan)
    ok = np.isfinite(field).all(axis=0)
    for i, region in enumerate(((radius >= 2) & (radius < 6), radius >= 6)):
        if (ok & region).sum() < max(12, 0.4 * region.sum()):
            continue
        bands = field[:, ok & region]
        spectra = np.concatenate((bands, np.diff(bands, axis=0)))
        for j, indices in enumerate(np.array_split(np.argsort(bands[3], kind="stable"), 5)):
            out[i, j] = spectra[:, indices].mean(axis=1)
    return out.ravel()


def image_views(ground, field, times, latitude):
    eye_map = ground["eye_map"]
    valid = (eye_map[:, :, 4] != 0).any(axis=(-1, -2)) & np.isfinite(times)
    yy, xx = np.indices(eye_map.shape[-2:]) - (np.array(eye_map.shape[-2:]) // 2)[:, None, None]
    radius = np.hypot(yy, xx)
    mix = np.full((*times.shape, 17 * len(RADII)), np.nan)
    air = np.full((*times.shape, 70), np.nan)
    for i, j in zip(*np.nonzero(valid), strict=True):
        mix[i, j] = mixing_profile(eye_map[i, j], radius)
    for i, j in zip(*np.nonzero(np.isfinite(field).any(axis=(2, 3, 4))), strict=True):
        air[i, j] = air_mass(field[i, j])
    blocks = {
        "eye": ground["eye"], "relief": ground["relief"], "strata": ground["eye_spectral_strata"],
        "mixing": mix, "wall": ground["wall"], "overcast": np.concatenate(
            [ground[k] for k in ("ring", "ring_sectors", "cloud_cover", "eye_cdo_relief")], axis=2),
        "waves": ground["waves"], "sector": ground["sector"], "background": air,
    }
    stats = {k: summarize_time(z, times, valid) for k, z in blocks.items()}
    views = {k: v["recent"] for k, v in stats.items()}
    for key, names in (("core", ("mixing", "strata", "eye")), ("cloud", ("overcast", "wall", "background"))):
        for mode in ("latest", "trend"):
            views[f"{key}_{mode}"] = np.column_stack([stats[k][mode] for k in names])
    views["latitude"] = np.asarray(latitude)[:, None]
    return views


def project_view(values, train, held, rank):
    """Training-only PCA, partial whitening and padding to a common rank."""
    a, b = _normalize(values[train], values[held])
    result = np.zeros((len(a) + len(b), rank), dtype=np.float64)
    if not a.shape[1]:
        return result
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean = a.mean(axis=0)
    _, singular, directions = np.linalg.svd(a - mean, full_matrices=False)
    size = min(rank, np.count_nonzero(singular > 1e-6))
    if size:
        sd = singular[:size] / np.sqrt(len(a))
        pcs = (np.concatenate((a, b)) - mean) @ directions[:size].T
        # Coordinate variance is proportional to sd rather than sd²; modest PCs survive.
        result[:, :size] = (pcs / np.sqrt(sd * sd.sum())).clip(-5., 5.)
    return result


def scene_kernels(maps, times, device):
    """PSD similarities between distributions of oriented synoptic textures across scans."""
    codes = translated_codes(maps).mean(0)
    x = torch.as_tensor(codes, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, dim=1)
    similarity = ((x @ x.T + 1) * 0.5).clamp(0, 1)
    valid = np.isfinite(maps).any(axis=(2, 3, 4)) & np.isfinite(times)
    if not valid.any(1).all():
        raise ValueError("Every record needs a timed field")
    t = np.where(valid, times, -np.inf)
    weights = np.exp((t - t.max(axis=1, keepdims=True)) / 90)
    w = x.new_tensor(weights / weights.sum(axis=1, keepdims=True))
    n, frames = times.shape
    kernels = []
    for power in (2, 8, 32):
        gram = torch.einsum("if,ifjh,jh->ij", w, similarity.pow(power).reshape(n, frames, n, frames), w)
        scale = gram.diag().clamp_min(1e-6).rsqrt()
        kernels.append((scale[:, None] * gram * scale).double())
    return torch.stack(kernels)
