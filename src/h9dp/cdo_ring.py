"""Estimate the coldest continuous cloud ring around a Himawari warm eye."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter1d, minimum_filter1d

from h9dp.d1 import locate

ANGLES = 48
RADII = np.arange(5, 118, 2, dtype=float)
WIDTHS = (1, 3, 7, 13)


@lru_cache(maxsize=1)
def offsets():
    angle = np.arange(ANGLES) * (2 * np.pi / ANGLES)
    return np.stack((np.sin(angle)[:, None] * RADII, np.cos(angle)[:, None] * RADII))


def coldest_ring(cost: np.ndarray) -> np.ndarray:
    """Minimize the warmest occupied cell on a closed path around the eye.

    Costs are [bands and widths, angles, radii]. A path may move two radial
    samples per angle. Returning to the starting radius enforces seam closure.
    """
    bands, angles, radii = cost.shape
    best = np.full((bands, radii, radii), np.inf, dtype=cost.dtype)
    diagonal = np.arange(radii)
    best[:, diagonal, diagonal] = cost[:, 0]
    for k in range(1, angles):
        prior = np.pad(best, ((0, 0), (0, 0), (2, 2)), constant_values=np.inf)
        best = np.minimum.reduce([prior[:, :, i : i + radii] for i in range(5)])
        np.maximum(best, cost[:, None, k, :], out=best)
    last = np.abs(diagonal[:, None] - diagonal[None, :]) <= 2
    return np.min(best[:, last], axis=1)


def measure(image, mask, xy):
    x, y = np.rint(np.asarray(xy) - 0.5).astype(int)
    eye, _ = locate(image, mask, (y, x))
    points = offsets() + np.asarray(eye)[:, None, None]
    support = map_coordinates(mask.astype(np.float32), points, order=1) > 0.999
    bt = np.stack([map_coordinates(v, points, order=1, output=np.float32) for v in image]) * 0.01 + 150
    bt = np.where(support[None], bt, np.nan)
    return describe_cloud(bt, support)


def describe_cloud(bt, support):
    """Describe cloud-ring connectivity and coverage from a polar sample grid."""
    cold_maxima = []
    sector_metrics = []
    for width in WIDTHS:
        # A wide embedded ring needs its entire radial strip to be cold.
        cost = maximum_filter1d(np.where(np.isfinite(bt), bt, np.inf), width, axis=-1,
                                mode="constant", cval=np.inf)
        cold_maxima.append(cost)
        per_angle = np.min(cost, axis=-1)
        for band in per_angle:
            samples = band[np.isfinite(band)]
            sector_metrics.append(np.quantile(samples, (0.1, 0.5, 0.9)) if samples.size >= 36 else [np.nan] * 3)
    ring = coldest_ring(np.concatenate(cold_maxima)).reshape(len(WIDTHS), 4)
    ring = np.where(np.isfinite(ring), ring, np.nan)

    # Separate widespread cold cloud from the continuous eyewall ring.
    fractions = []
    for low, high in ((10, 30), (30, 60), (60, 105)):
        area = (RADII >= low) & (RADII < high)
        vals = bt[3][:, area]
        found = np.isfinite(vals)
        if found.mean() < 0.75:
            fractions.extend([np.nan] * 5)
            continue
        ir = vals[found]
        fractions.extend([np.quantile(ir, 0.1), np.mean(ir < 200), np.mean(ir < 210),
                          np.mean(ir < 220), np.mean(ir < 230)])
    eye_bt = minimum_filter1d(support.astype(np.uint8), size=2, axis=-1)
    near = bt[:, :, :4]
    near = np.where(eye_bt[None, :, :4] > 0, near, np.nan)
    peaks = np.max(np.where(np.isfinite(near), near, -np.inf), axis=(1, 2))
    peaks = np.where(np.isfinite(peaks), peaks, np.nan)
    return {
        "ring": ring.ravel(),
        "ring_sectors": np.ravel(sector_metrics),
        "cloud_cover": np.array(fractions),
        "eye_cdo_relief": (peaks - ring).ravel(),
    }


def sequence(image, mask, centers, times):
    valid = mask.any(axis=(-1, -2)) & np.isfinite(times)
    frames = [measure(image[i], mask[i], centers[i]) for i in np.flatnonzero(valid)]
    if not frames:
        raise ValueError("CDO extraction requires at least one available scan")
    groups = {}
    for group, example in frames[0].items():
        values = np.full((len(times), example.size), np.nan, dtype=np.float32)
        values[valid] = np.stack([frame[group] for frame in frames])
        groups[group] = values
    return groups
