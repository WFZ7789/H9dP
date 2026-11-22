"""Radiometric views along an AHI line of sight, not retrieved profile heights."""

from datetime import UTC

import numpy as np
from scipy.ndimage import gaussian_filter

from h9dp.r1 import summarize_time


def geometry(lat, lon, utc, projection):
    """Horizontal apparent displacement (south, east) per km of height, and smooth context."""
    sublon, _, _, _, _, rs, major, minor = projection
    phi, delta = np.deg2rad([lat, sublon - lon])
    sp, cp = np.sin(phi), np.cos(phi)
    e2 = 1 - (minor / major)**2
    n = major / np.sqrt(1 - e2 * sp**2)
    east = rs * np.sin(delta)
    north = -rs * sp * np.cos(delta) + n * e2 * sp * cp
    up = rs * cp * np.cos(delta) - n * (1 - e2 * sp**2)
    if up <= 0:
        raise ValueError("Center not visible from the geostationary satellite")
    # Toward the ground intercept of the line through the elevated emitter: away from nadir.
    tangent = np.array([north, -east]) / up
    utc = utc.astimezone(UTC)
    day = utc.timetuple().tm_yday
    decl = 0.409 * np.sin(2 * np.pi * (day - 81) / 365.2422)
    angle = np.deg2rad(lon + 15 * (utc.hour + utc.minute / 60 + utc.second / 3600 - 12))
    a, b = sp * np.sin(decl), cp * np.cos(decl)
    sunset = np.arccos(np.clip(-a / max(b, 1e-12), -1, 1))
    daily = (sunset * a + b * np.sin(sunset)) / np.pi
    cosine = max(0., a + b * np.cos(angle))
    return tangent, np.array([sp, np.sqrt(1 + tangent @ tangent), daily, cosine])


HEIGHT_KM = (0., 5., 11., -8.)
RELATIVE_HEIGHT = np.array([1., 0.72, 0.42, 0.])
SIGMA = (1.25, 3.0)


def column_profile(patch, tangent):
    """Read warm-eye spectra with and against parallax; reject unsupported apertures."""
    bands = patch[:4].astype(np.float64) * 0.01 + 240
    valid = (patch[4] > 0) & np.isfinite(bands).all(axis=0)
    y, x = np.indices(valid.shape) - (np.array(valid.shape) // 2)[:, None, None]
    radius2 = y*y + x*x
    support = gaussian_filter(valid.astype(float), 1.4, mode="constant")
    smooth = gaussian_filter(np.where(valid, bands[3], 0), 1.4, mode="constant") / support.clip(1e-10)
    anchor = (radius2 <= 8**2) & valid & (support > 0.9)
    read = np.full((len(HEIGHT_KM), len(SIGMA), 4), np.nan)
    if anchor.any():
        idx = np.unravel_index(np.where(anchor, smooth - 0.035 * radius2, -np.inf).argmax(), valid.shape)
        for i, height in enumerate(HEIGHT_KM):
            # G1 grid spacing is 2 km. B16 selects the common reference pixel.
            shift = RELATIVE_HEIGHT[:, None] * tangent * height / 2
            d2 = ((y - y[idx] - shift[:, 0, None, None])**2 + (x - x[idx] - shift[:, 1, None, None])**2)
            for j, sigma in enumerate(SIGMA):
                w = np.exp(d2 / (-2 * sigma**2)) * valid
                mass = w.sum(axis=(1, 2))
                average = (w * np.where(valid, bands, 0)).sum(axis=(1, 2)) / mass.clip(1e-10)
                read[i, j] = np.where(mass >= 0.90 * 2 * np.pi * sigma**2, average, np.nan)
    return np.r_[read.ravel(), np.diff(read, axis=2).ravel(), (read[1:, :, :3] - read[0, :, :3]).ravel()]


def column_views(pixels, tangent, times):
    valid = (pixels[:, :, 4] > 0).any(axis=(2, 3)) & np.isfinite(times)
    columns = np.stack([[column_profile(p, t) for p in row] for t, row in zip(tangent, pixels, strict=True)])
    stats = summarize_time(columns, times, valid)
    return {"column_recent": stats["recent"], "column_latest": stats["latest"]}
