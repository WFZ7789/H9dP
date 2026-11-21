"""Band-specific isotherm geometry and environmental references in Himawari imagery."""

import numpy as np
from scipy.ndimage import gaussian_filter, label

from h9dp.d1 import locate

GRID = np.arange(-38, 39)
DY, DX = GRID[:, None], GRID[None, :]
RADIUS = np.hypot(DY, DX)


def isotherms(image, mask, xy):
    center = tuple(np.rint(np.asarray(xy)[::-1] - 0.5).astype(int))
    eye, _ = locate(image, mask, center)
    y, x = eye[0] + DY, eye[1] + DX
    yn, xn = y.clip(0, mask.shape[0] - 1), x.clip(0, mask.shape[1] - 1)
    valid = mask[yn, xn] & (y == yn) & (x == xn)
    patch = image[:, yn, xn].astype(float) * 0.01 + 150
    surround = valid & (RADIUS >= 18) & (RADIUS <= 35)
    center_ok = valid & (RADIUS <= 14)
    shape = np.full((4, 4, 7), np.nan)
    spectrum = np.full((4, 4, 4), np.nan)
    peaks = np.full((4, 4), np.nan)
    if center_ok.any() and surround.sum() >= 0.75 * np.count_nonzero((RADIUS >= 18) & (RADIUS <= 35)):
        for b, data in enumerate(patch):
            support = gaussian_filter(valid.astype(float), 1.1)
            smoothed = gaussian_filter(np.where(valid, data, 0), 1.1) / support.clip(1e-8)
            candidates = np.where(center_ok & (support > 0.95), smoothed - 0.018 * RADIUS**2, -np.inf)
            if not np.isfinite(candidates).any():
                continue
            i, j = np.unravel_index(np.argmax(candidates), candidates.shape)
            bg = np.quantile(data[surround], 0.35)
            amplitude = smoothed[i, j] - bg
            peaks[b] = [smoothed[i, j], amplitude, i - 38, j - 38]
            if amplitude < 0.65:
                continue
            for k, fraction in enumerate((0.28, 0.50, 0.72, 0.90)):
                components, _ = label((smoothed >= bg + fraction * amplitude) & valid & (support >= 0.9))
                component = components == components[i, j]
                if not components[i, j] or (component & (RADIUS >= 35)).any():
                    continue
                ys, xs = np.nonzero(component)
                pos = np.stack([ys - ys.mean(), xs - xs.mean()])
                axes = np.linalg.eigvalsh(pos @ pos.T / len(xs)).clip(0)
                anomaly = (data[component] - bg).mean()
                shape[b, k] = [
                    np.log1p(len(xs)),
                    np.sqrt(axes[0]),
                    np.sqrt(axes[1]),
                    ys.mean() - 38,
                    xs.mean() - 38,
                    anomaly,
                    np.log1p(len(xs)) * anomaly,
                ]
                spectrum[b, k] = patch[:, component].mean(1)
    # Retrieve local warm-background spectra across a much wider range of cloud conditions.
    wide = image[:, ::4, ::4].astype(np.float32) * 0.01 + 150
    wy, wx = np.ogrid[: mask.shape[0] : 4, : mask.shape[1] : 4]
    radii = np.hypot(wy - eye[0], wx - eye[1])
    env = np.full((3, 3, 4), np.nan)
    for j, (lo, hi) in enumerate(((26, 60), (60, 125), (125, 240))):
        area = mask[::4, ::4] & (radii >= lo) & (radii < hi)
        if area.sum() < 64:
            continue
        values = wide[:, area]
        for k, quant in enumerate((0.08, 0.7, 0.95)):
            ref = np.quantile(values[3], quant)
            select = values[3] <= ref if quant < 0.5 else values[3] >= ref
            env[j, k] = values[:, select].mean(1)
    excess = peaks[:, 0][None, None, :] - env
    return {
        "shape": shape.ravel(),
        "spectrum": spectrum.ravel(),
        "peaks": peaks.ravel(),
        "environment": env.ravel(),
        "excess": excess.ravel(),
    }


def sequence(image, mask, centers, times):
    usable = np.flatnonzero(mask.any(axis=(-1, -2)) & np.isfinite(times))
    frames = [isotherms(image[j], mask[j], centers[j]) for j in usable]
    result = {}
    for k, v in frames[0].items():
        result[k] = np.full((6, len(v)), np.nan, dtype=np.float32)
        result[k][usable] = np.stack([f[k] for f in frames])
    return result
