"""Image-only warm-column descriptors on the cached R1 eye and wide AHI views."""

from __future__ import annotations

import numpy as np


def _warm_spectrum(pixels: np.ndarray, band: int, count: int) -> np.ndarray:
    """Read all four channels at the same sites; include ties at the cutoff."""
    rank = pixels[band]
    cutoff = np.partition(rank, -count)[-count]
    return pixels[:, rank >= cutoff].mean(axis=1)


def warm_columns(views: np.ndarray) -> dict[str, np.ndarray]:
    """Extract co-located spectra, warm-cap size and eye-minus-neighborhood contrasts.

    Inputs are (record, frame, channel, y, x) with the units and channel layout
    of r1._view. Operate one frame at a time, including on memory-mapped views.
    The outer reference is a *local* warm cloud-field statistic, not clear-sky BT.
    """
    if views.ndim != 5 or views.shape[2:] != (10, 64, 64):
        raise ValueError("Expected [records,frames,10,64,64] R1 views")
    y, x = np.ogrid[-32:32, -32:32]
    r2 = y * y + x * x
    core = r2 <= 12 ** 2
    # The wide view has stride four; this annulus is approx. 80–124 native pixels out.
    outer = (r2 >= 20 ** 2) & (r2 <= 31 ** 2)
    shape = views.shape[:2]
    column = np.full((*shape, 26), np.nan, dtype=np.float32)
    anomaly = np.full((*shape, 15), np.nan, dtype=np.float32)
    for i, j in np.ndindex(shape):
        view = views[i, j]
        mask = core & (view[4] == 100)
        if mask.sum() < 0.75 * core.sum():
            continue
        inner = view[:4, mask].astype(np.float32) * 0.01  # Kelvin relative to 240 K
        peaks = []
        size = []
        for band in (3, 0):  # CO2 and upper water vapor: need not peak in the same pixel
            peaks.extend(_warm_spectrum(inner, band, count) for count in (3, 20))
            drop = inner[band].max() - inner[band]
            size.extend(1.0 / np.sqrt(max(1, np.count_nonzero(drop <= width))) for width in (1, 3))
        top = peaks[0]
        column[i, j] = np.concatenate((*peaks, np.diff(top), np.diff(peaks[2]), size))
        usable = outer & (view[9] >= 75)
        if usable.sum() < 0.5 * outer.sum():
            continue
        env = view[5:9, usable].astype(np.float32) * 0.01
        ref = _warm_spectrum(env, 3, max(1, env.shape[1] // 10))
        contrast = top - ref
        anomaly[i, j] = np.concatenate((ref, contrast, peaks[2] - ref, np.diff(contrast)))
    return {"column": column, "column_anomaly": anomaly}
