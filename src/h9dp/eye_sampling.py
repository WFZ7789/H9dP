"""Fit pixel-integrated warm-eye templates across the available Himawari scans."""

from functools import lru_cache

import numpy as np
from scipy.special import erf

from h9dp.d1 import locate

HALF = 10
WIDTHS = np.array([0.55, 0.82, 1.22, 1.8, 2.7, 4.0, 5.9, 8.6])
PHASES = np.arange(-3, 3.1, 0.75)


@lru_cache(maxsize=1)
def template_bank():
    axis = np.arange(-HALF, HALF + 1, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    # An orthogonal plane absorbs a background level and a linear cloud gradient.
    plane, _ = np.linalg.qr(np.column_stack((np.ones(x.size), x.ravel(), y.ravel())))
    delta = axis - PHASES[:, None]
    width = WIDTHS[:, None, None]
    denominator = np.sqrt(2) * width
    projected = width * np.sqrt(np.pi / 2) * (
        erf((delta + 0.5) / denominator) - erf((delta - 0.5) / denominator)
    )
    templates = (projected[:, :, None, :, None] * projected[:, None, :, None, :]).reshape(
        len(width), PHASES.size**2, x.size
    )
    mean = templates.mean(-1)
    residual = templates - (templates @ plane) @ plane.T
    energy = np.sum(residual**2, axis=-1)
    return (
        plane.astype(np.float32),
        (residual / np.sqrt(energy[:, :, None])).astype(np.float32).reshape(-1, x.size),
        energy.astype(np.float32),
        mean.astype(np.float32),
    )


def core_patch(image, mask, xy):
    x, y = np.rint(np.asarray(xy) - 0.5).astype(int)
    y, x = locate(image, mask, (y, x))[0]
    if not (HALF <= y < mask.shape[0] - HALF and HALF <= x < mask.shape[1] - HALF):
        return None
    ys = slice(y - HALF, y + HALF + 1)
    xs = slice(x - HALF, x + HALF + 1)
    if not mask[ys, xs].all():
        return None
    return image[:, ys, xs].astype(np.float32).reshape(4, -1) * 0.01 + 150.0


def fit_sequence(patches, times):
    """Share the apparent width across frames and fit each frame's phase and amplitude."""
    plane, templates, energy, mean = template_bank()
    value = patches - (patches @ plane) @ plane.T
    # A common frame center and width apply to all four measured bands.
    correlation = (value @ templates.T).reshape(len(patches), 4, len(WIDTHS), -1)
    total = (value**2).sum(-1).clip(1.0)[:, :, None, None]
    positive = correlation.clip(min=0)
    fit_fraction = (positive**2 / total).clip(0, 1)
    spectral_weight = np.array([0.12, 0.16, 0.29, 0.43], dtype=np.float32)
    score = (fit_fraction * spectral_weight[None, :, None, None]).sum(1)
    spatial_shift = (PHASES[:, None]**2 + PHASES[None, :]**2).ravel()
    score -= 0.0012 * spatial_shift
    phase = score.argmax(-1)
    best = np.take_along_axis(score, phase[:, :, None], axis=2).squeeze(-1)
    recent = np.exp((times - times.max()) / 38)
    consensus = recent @ best / recent.sum()
    selected = consensus.argmax()
    phases = phase[:, selected]
    i = np.arange(len(patches))
    amplitude = positive[i, :, selected, phases] / np.sqrt(energy[selected, phases])[:, None]
    baseline = patches.mean(-1) - amplitude * mean[selected, phases][:, None]
    # Near-resolution widths are weakly identified. Preserve the score curve and
    # cap fitted amplitudes at a broad finite range to bound extrapolation.
    amplitude = amplitude.clip(0, 100)
    peak = baseline + amplitude
    shift = np.column_stack((PHASES[phases // PHASES.size], PHASES[phases % PHASES.size]))
    fit = np.column_stack((np.full(len(patches), WIDTHS[selected]), best[:, selected], shift,
                           np.broadcast_to(consensus, best.shape), fit_fraction[i, :, selected, phases]))
    radiometry = np.column_stack((peak, baseline, amplitude, np.diff(peak, axis=1),
                                 np.quantile(patches, 0.98, axis=-1)))
    return radiometry.astype(np.float32), fit.astype(np.float32)


def sequence(images, masks, centers, times):
    valid = masks.any((-1, -2)) & np.isfinite(times)
    patches, frames = [], []
    for i in np.flatnonzero(valid):
        patch = core_patch(images[i], masks[i], centers[i])
        if patch is not None:
            patches.append(patch)
            frames.append(i)
    radiometry = np.full((len(times), 19), np.nan, dtype=np.float32)
    sampling = np.full((len(times), 16), np.nan, dtype=np.float32)
    if patches:
        radiometry[frames], sampling[frames] = fit_sequence(np.stack(patches), times[frames])
    return {"integrated_eye": radiometry, "eye_sampling": sampling}
