"""D1 multiscale spectral distributions on six native Himawari scans."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter


@lru_cache
def annulus(lo, hi, stride):
    y, x = np.mgrid[-hi : hi + 1 : stride, -hi : hi + 1 : stride]
    ring = (x * x + y * y >= lo**2) & (x * x + y * y < hi**2)
    return y[ring], x[ring]


def locate(image, mask, center):
    """Choose a warm-core location by multiscale matched filtering within 22 pixels."""
    y, x = center
    offsets = np.arange(-64, 65)
    yy, xx = y + offsets[:, None], x + offsets
    inside = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    yy, xx = yy.clip(0, mask.shape[0] - 1), xx.clip(0, mask.shape[1] - 1)
    ok = inside & mask[yy, xx]
    # Centikelvin offsets cancel in differences between spatial scales.
    warm = (image[2, yy, xx].astype(float) + image[3, yy, xx]) * 0.005
    filtered = []
    support = np.ones_like(ok)
    for sigma in (1.5, 4, 9, 18):
        weight = gaussian_filter(ok.astype(float), sigma)
        filtered.append(gaussian_filter(np.where(ok, warm, 0), sigma) / weight.clip(1e-8))
        support &= weight > 0.85
    dy, dx = np.mgrid[-64:65, -64:65]
    r2 = dx * dx + dy * dy
    support &= (r2 <= 22**2) & ok
    if not support.any():
        return center, np.full(4, np.nan)
    responses = np.stack(filtered[:3]) - filtered[-1]
    score = responses.mean(0) - 0.009 * r2
    i, j = np.unravel_index(np.argmax(np.where(support, score, -np.inf)), score.shape)
    return (y + i - 64, x + j - 64), np.r_[np.sqrt(r2[i, j]), responses[:, i, j]]


def samples(image, mask, center, ring):
    dy, dx = ring
    yy, xx = center[0] + dy, center[1] + dx
    inside = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    yy, xx = yy[inside], xx[inside]
    ok = mask[yy, xx]
    if ok.sum() < max(3, 0.45 * len(dy)):
        return None
    bt = image[:, yy[ok], xx[ok]].astype(np.float32) * 0.01 + 150
    return np.vstack((bt, bt[1:] - bt[:-1], bt[3] - bt[0]))


def frame_signature(image, mask, xy):
    center = tuple(np.rint(np.asarray(xy)[::-1] - 0.5).astype(int))
    eye, morphology = locate(image, mask, center)
    core, cloud = [], []
    for lo, hi, step in (
        (0, 3, 1),
        (0, 7, 1),
        (0, 14, 1),
        (0, 24, 1),
        (14, 30, 1),
        (30, 55, 2),
        (55, 92, 3),
        (92, 148, 4),
        (148, 230, 5),
    ):
        values = samples(image, mask, eye, annulus(lo, hi, step))
        quantiles = np.full((5, 8), np.nan)
        spectrum = np.full((3, 8), np.nan)
        if values is not None:
            quantiles = np.quantile(values, (0.02, 0.15, 0.5, 0.85, 0.98), axis=1)
            # Co-located spectra of the cold, middle and warm B16 partitions.
            for i, (lower, upper) in enumerate(((0, 1), (1, 3), (3, 4))):
                select = (values[3] >= quantiles[lower, 3]) & (values[3] <= quantiles[upper, 3])
                if select.any():
                    spectrum[i] = values[:, select].mean(axis=1)
        (core if lo == 0 else cloud).append(np.vstack((quantiles, spectrum)).ravel())
    # Keep cloud orientation at scales well outside the eye for environmental context.
    scene = []
    for lo, hi in ((18, 60), (60, 130), (130, 220)):
        dy, dx = annulus(lo, hi, 4)
        sector = np.floor((np.arctan2(dy, dx) + np.pi) * 4 / np.pi).astype(int) % 8
        for k in range(8):
            values = samples(image, mask, center, (dy[sector == k], dx[sector == k]))
            scene.append(values.mean(1) if values is not None else np.full(8, np.nan))
    core = np.asarray(core)
    cloud = np.asarray(cloud)
    # Signed core-to-cloud contrasts retain band pairing and the edge intensity.
    contrast = np.stack((core[0] - cloud[0], core[2] - cloud[2], cloud[0] - cloud[-1]))
    return [np.r_[morphology, core.ravel()], cloud.ravel(), contrast.ravel(), np.ravel(scene)]
