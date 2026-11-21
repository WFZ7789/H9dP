"""Multi-pixel measurements of the warm anomaly in an R1 eye-candidate view."""

import numpy as np


def aperture_features(views: np.ndarray) -> np.ndarray:
    """Measure peak, positive integral, effective area and RMS radius per band.

    Both apertures share a local annular median. Require complete aperture
    coverage so a hole cannot masquerade as a smaller eye. The radius is
    measured about the anomaly's own centroid in native image pixels.
    """
    if views.ndim != 5 or views.shape[2:] != (10, 64, 64):
        raise ValueError("Expected [records,frames,10,64,64] R1 views")
    y, x = np.mgrid[-32:32, -32:32]
    radius2 = x**2 + y**2
    ring = (radius2 >= 18**2) & (radius2 <= 26**2)
    apertures = [radius2 <= radius**2 for radius in (6, 12)]
    features = np.full((*views.shape[:2], 2, 4, 4), np.nan, dtype=np.float32)
    for i, j in np.ndindex(views.shape[:2]):
        view = views[i, j]
        valid = view[4] == 100
        reference = ring & valid
        if reference.sum() < 0.8 * ring.sum():
            continue
        background = np.median(view[:4, reference], axis=1)
        for k, aperture in enumerate(apertures):
            if not valid[aperture].all():
                continue
            contrast = (view[:4, aperture] - background[:, None]) * 0.01
            weights = contrast.clip(min=0)
            total = weights.sum(axis=1)
            features[i, j, k, :, 0] = contrast.max(axis=1)
            features[i, j, k, :, 1] = total
            supported = total > 0
            if supported.any():
                fraction = weights[supported] / total[supported, None]
                features[i, j, k, supported, 2] = 1 / (fraction**2).sum(axis=1)
                dx = x[aperture] - (fraction @ x[aperture])[:, None]
                dy = y[aperture] - (fraction @ y[aperture])[:, None]
                features[i, j, k, supported, 3] = np.sqrt((fraction * (dx**2 + dy**2)).sum(axis=1))
    return features.reshape(*features.shape[:2], 32)
