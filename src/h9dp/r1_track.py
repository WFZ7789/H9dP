"""Image-only registration of earlier Himawari frames to a storm-centered analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from h9dp.r1 import _patch


@dataclass(frozen=True)
class Registration:
    center: np.ndarray
    correlation: float
    accepted: bool


def _match(template: np.ndarray, weight: np.ndarray, candidate: np.ndarray,
           valid: np.ndarray) -> np.ndarray:
    """Gaussian-weighted normalized cross-correlation, ignoring invalid overlap."""
    th, tw = weight.shape
    h, w = valid.shape
    axes = (-2, -1)
    # The valid convolution block does not reach the circularly wrapped portion.
    fft = np.fft.rfft2
    inv = np.fft.irfft2
    scale = (h, w)
    candidate_fft = fft(candidate, s=scale, axes=axes)
    valid_fft = fft(valid.astype(np.float64), s=scale, axes=axes)
    template_weighted = template * weight
    weight_fft = fft(weight[::-1, ::-1], s=scale, axes=axes)
    template_fft = fft(template_weighted[:, ::-1, ::-1], s=scale, axes=axes)

    def valid_block(product: np.ndarray):
        return inv(product, s=scale, axes=axes)[..., th - 1:h, tw - 1:w]

    mass = valid_block(valid_fft * weight_fft)
    norm_mass = np.maximum(mass, 1e-8)
    a = valid_block(valid_fft * template_fft)
    b = valid_block(candidate_fft * weight_fft)
    aa = valid_block(valid_fft * fft((template_weighted * template)[:, ::-1, ::-1], s=scale, axes=axes))
    bb = valid_block(fft(candidate ** 2, s=scale, axes=axes) * weight_fft)
    ab = valid_block(candidate_fft * template_fft)
    scatter_a = np.maximum(aa - a ** 2 / norm_mass, 0)
    scatter_b = np.maximum(bb - b ** 2 / norm_mass, 0)
    scatter = np.sqrt(scatter_a * scatter_b)
    corr = (ab - a * b / norm_mass) / np.maximum(scatter, 1e-8)
    supported = ((mass >= 0.82 * weight.sum()) & (scatter_a.min(axis=0) >= norm_mass)
                 & (scatter_b.min(axis=0) >= norm_mass))
    return np.where(supported, np.clip(corr, -1, 1).mean(axis=0), -np.inf)


def register_earlier(early_image: np.ndarray, early_valid: np.ndarray, predicted_center: np.ndarray,
                     late_image: np.ndarray, late_valid: np.ndarray, later_center: np.ndarray,
                     lag_minutes: float) -> Registration:
    """Bound translation by elapsed time; preserve the prior for weak/ambiguous matches."""
    if not np.isfinite(lag_minutes) or lag_minutes <= 0:
        return Registration(predicted_center, np.nan, False)
    step = 2
    radius = min(28, max(3, math.ceil(lag_minutes * 0.33)))
    half = 38
    template, usable, delta = _patch(late_image, late_valid, later_center[::-1], half, step)
    if usable.mean() < 0.60:
        return Registration(predicted_center, np.nan, False)
    # B10/B16 are sensitive to coherent inner-core structure in different layers.
    template = template[[2, 3]].astype(np.float64) - 220.0
    previous, valid, _ = _patch(early_image, early_valid, predicted_center[::-1], half + radius * step, step)
    values = np.where(valid, previous[[2, 3]].astype(np.float64) - 220.0, 0.0)
    window = np.exp(-(delta[:, None] ** 2 + delta[None, :] ** 2) / (2 * 23.0 ** 2)) * usable
    similarity = _match(template, window, values, valid)
    grid = np.arange(-radius, radius + 1)
    distance2 = (grid[:, None] ** 2 + grid[None, :] ** 2) / radius ** 2
    penalized = np.where(distance2 <= 1, similarity - 0.065 * distance2, -np.inf)
    yy, xx = np.unravel_index(penalized.argmax(), penalized.shape)
    score = float(similarity[yy, xx])
    if not np.isfinite(score) or score < 0.43 or penalized[yy, xx] < penalized[radius, radius] + 0.016:
        return Registration(predicted_center, score if np.isfinite(score) else np.nan, False)
    shift = step * np.array([grid[xx], grid[yy]])
    return Registration(predicted_center + shift, score, True)


def track_centers(images: np.ndarray, masks: np.ndarray, centers: np.ndarray,
                  times: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Walk backward from the final image, with no use of label, name, or other records."""
    aligned = np.array(centers, dtype=np.float64, copy=True)
    trace = np.full((len(times), 3), np.nan, dtype=np.float32)
    indices = np.flatnonzero(available)
    if not len(indices):
        return aligned, trace
    order = indices[np.argsort(times[indices], kind="stable")][::-1]
    trace[order[0]] = (0, np.nan, 0)
    for later, earlier in pairwise(order):
        lag = float(times[later] - times[earlier])
        prior = centers[earlier] + (aligned[later] - centers[later]) * np.exp(-max(0.0, lag) / 120)
        fit = register_earlier(images[earlier], masks[earlier], prior,
                               images[later], masks[later], aligned[later], lag)
        aligned[earlier] = fit.center
        trace[earlier] = (np.linalg.norm(fit.center - centers[earlier]), fit.correlation, fit.accepted)
    return aligned, trace
