"""Polar cloud structure and angular displacement in native Himawari imagery."""

import itertools
from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates

from h9dp.d1 import locate

RADII = np.array([1, 2, 3, 5, 7, 10, 14, 19, 25, 33, 44, 58, 76, 98, 126, 160, 198, 228])
ANGLES = 96


@lru_cache
def offsets():
    angle = np.arange(ANGLES) * (2 * np.pi / ANGLES)
    return np.stack((RADII[:, None] * np.sin(angle), RADII[:, None] * np.cos(angle)))


def polar_scan(image, mask, xy):
    center = tuple(np.rint(np.asarray(xy)[::-1] - 0.5).astype(int))
    eye, _ = locate(image, mask, center)
    points = offsets() + np.asarray(eye)[:, None, None]
    support = map_coordinates(mask.astype(np.float32), points, order=1)
    values = np.stack([map_coordinates(band, points, order=1, output=np.float32) for band in image])
    return np.where(support[None] >= 0.999, values * 0.01 + 150.0, np.nan)


def angular_motion(early, late, minutes):
    """Correlate three neighboring rings, restricting the shift by elapsed scan time."""
    angle = np.arange(-ANGLES // 2 + 1, ANGLES // 2)
    result = []
    for i in range(2, len(RADII) - 1):
        a, b = early[:, i - 1 : i + 2], late[:, i - 1 : i + 2]
        ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        a = np.where(ok[..., None], a, 0)
        b = np.where(ok[..., None], b, 0)
        a = a - a.mean(-1, keepdims=True)
        b = b - b.mean(-1, keepdims=True)
        norm = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
        ok &= norm > ANGLES * 0.35**2
        if not ok.any():
            result.append([np.nan] * 4)
            continue
        corr = np.fft.irfft(np.fft.rfft(b) * np.fft.rfft(a).conj(), n=ANGLES)
        corr = (corr[ok] / norm[ok, None]).mean(0)
        shifts = angle[
            np.abs(angle) <= np.clip(0.55 * minutes * ANGLES / (2 * np.pi * RADII[i]), 2, ANGLES / 3)
        ]
        idx = shifts[np.argmax(corr[shifts])]
        change = late[:, i] - early[:, i]
        change = change[np.isfinite(change)]
        result.append(
            [
                idx * (2 * np.pi / ANGLES) * RADII[i] / minutes,
                corr[idx],
                corr[idx] - corr[0],
                np.median(change) * 60 / minutes if change.size else np.nan,
            ]
        )
    return np.ravel(result)


def sequence(image, mask, centers, times):
    usable = np.flatnonzero(mask.any(axis=(-1, -2)) & np.isfinite(times))
    ring = np.full((6, 4, len(RADII), ANGLES), np.nan)
    for j in usable:
        ring[j] = polar_scan(image[j], mask[j], centers[j])
    spectra = np.concatenate((ring, np.diff(ring, axis=1), ring[:, 3:4] - ring[:, :1]), axis=1)
    # Partial rings may supply radiometry but never a spurious Fourier/rotation signal.
    support = np.isfinite(spectra).sum(-1) >= ANGLES * 0.65
    filled = np.where(support[..., None], spectra, 0)
    quant = np.nanquantile(filled, (0.03, 0.2, 0.5, 0.8, 0.97), axis=-1)
    quant = np.where(support[None], quant, np.nan).transpose(1, 2, 3, 0)
    ft = np.fft.rfft(ring) / ANGLES
    harmonics = np.stack(
        [np.sqrt((abs(ft[..., a:b]) ** 2).sum(-1)) for a, b in ((1, 2), (2, 4), (4, 8), (8, 16))], -1
    )
    # Raw broad-sector absolute temperatures preserve the orientation of asymmetric clouds.
    sector = ring[:, :, 4:].reshape(6, 4, -1, 12, 8).mean(-1)
    motion = np.full((6, (len(RADII) - 3) * 4), np.nan)
    for a, b in itertools.pairwise(usable):
        minutes = times[b] - times[a]
        if 0 < minutes <= 130:
            motion[b] = angular_motion(ring[a], ring[b], minutes)
    delta = quant[:, :, :8, :] - quant[:, :, 9:12, 0].mean(2)[:, :, None, None]
    return {
        key: v.reshape(6, -1).astype(np.float32)
        for key, v in {
            "eye": quant[:, :, :8],
            "wall": quant[:, :, 8:],
            "relief": delta,
            "waves": harmonics,
            "sector": sector,
            "motion": motion,
        }.items()
    }
