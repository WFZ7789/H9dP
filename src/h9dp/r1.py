"""Label-free Himawari descriptors and record-level evaluation tools for R1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

FRAME_COUNT = 6
RING_EDGES = (0, 3, 6, 10, 16, 24, 36, 52, 76, 108, 152, 200)
FEATURE_SCHEMA = "r1-features/v2"
QUALITY_WEIGHT = {"Good": 2.0, "Fair": 1.0, "Poor": 0.5}


@dataclass(frozen=True)
class FrameFeatures:
    profile: np.ndarray
    structure: np.ndarray
    views: np.ndarray


def _patch(image: np.ndarray, valid: np.ndarray, center: np.ndarray, half: int, step: int = 1):
    """Sample a window in line/column coordinates without wrapping at image boundaries."""
    height, width = valid.shape
    y, x = np.rint(center - 0.5).astype(int)
    dy = np.arange(-half, half, step)
    yy, xx = y + dy, x + dy
    inside = (yy[:, None] >= 0) & (yy[:, None] < height) & (xx >= 0) & (xx < width)
    yy, xx = np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)
    ok = valid[yy[:, None], xx] & inside
    values = image[:, yy[:, None], xx].astype(np.float32) * 0.01 + 150.0
    return np.where(ok, values, 0.0), ok, dy


def _box_sum(integral: np.ndarray, y: np.ndarray, x: np.ndarray, half: int) -> np.ndarray:
    return (integral[..., y + half + 1, x + half + 1] - integral[..., y - half, x + half + 1]
            - integral[..., y + half + 1, x - half] + integral[..., y - half, x - half])


def eye_candidate(image: np.ndarray, valid: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a local warm-core candidate; not every record has a resolved eye."""
    half, search = 43, 18
    patch, ok, _ = _patch(image, valid, center, half)
    data = np.concatenate((patch, ok[None]), axis=0)
    integral = np.pad(data.cumsum(axis=1, dtype=np.float64).cumsum(axis=2), ((0, 0), (1, 0), (1, 0)))
    yy, xx = np.mgrid[-search:search + 1, -search:search + 1]
    r2 = yy ** 2 + xx ** 2
    inner = _box_sum(integral, half + yy, half + xx, 3)
    outer = _box_sum(integral, half + yy, half + xx, 21) - inner
    core = inner[:4] / np.maximum(inner[4], 1.0)
    surround = outer[:4] / np.maximum(outer[4], 1.0)
    contrast = core - surround
    # B10 is usually sensitive to a clear warm core; neighboring bands stabilize weak signals.
    score = 0.45 * contrast[2] + 0.25 * contrast[1] + 0.15 * (contrast[0] + contrast[3])
    score -= 0.010 * r2
    supported = (inner[4] >= 42) & (outer[4] >= 1440) & (r2 <= search ** 2)
    if not supported.any():
        return center, np.full(13, np.nan, dtype=np.float32)
    y, x = np.unravel_index(np.argmax(np.where(supported, score, -np.inf)), score.shape)
    offset = np.array([yy[y, x], xx[y, x]], dtype=float)
    middle = (search, search)
    at_center = contrast[:, middle[0], middle[1]] if supported[middle] else np.full(4, np.nan)
    properties = np.r_[np.linalg.norm(offset), core[:, y, x], contrast[:, y, x], at_center]
    return center + offset, properties.astype(np.float32)


def _spectral(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values, np.diff(values, axis=0)), axis=0)


def _annuli(image: np.ndarray, valid: np.ndarray, center: np.ndarray):
    values, ok, delta = _patch(image, valid, center, RING_EDGES[-1], 2)
    values = _spectral(values)
    yy, xx = np.meshgrid(delta, delta, indexing="ij")
    radius2 = yy ** 2 + xx ** 2
    angle_bin = np.floor((np.arctan2(yy, xx) + np.pi) * (4.0 / np.pi)).astype(int) % 8
    profiles = np.full((len(RING_EDGES) - 1, 5, len(values)), np.nan, dtype=np.float32)
    asymmetry = np.full((len(profiles), len(values)), np.nan, dtype=np.float32)
    for i, (lo, hi) in enumerate(pairwise(RING_EDGES)):
        region = (radius2 >= lo ** 2) & (radius2 < hi ** 2)
        selected = region & ok
        if selected.sum() < max(3, math.ceil(0.35 * region.sum())):
            continue
        samples = values[:, selected]
        profiles[i, 0] = samples.mean(axis=1)
        profiles[i, 1:4] = np.quantile(samples, (0.1, 0.5, 0.9), axis=1)
        profiles[i, 4] = samples.std(axis=1)
        sectors = [values[:, group].mean(axis=1) for a in range(8)
                   if (group := (selected & (angle_bin == a))).sum() >= 3]
        if len(sectors) >= 5:
            asymmetry[i] = np.std(sectors, axis=0)
    return profiles, asymmetry


def _view(image: np.ndarray, valid: np.ndarray, center: np.ndarray, stride: int) -> np.ndarray:
    values, ok, _ = _patch(image, valid, center, 32 * stride)
    coverage = ok.reshape(64, stride, 64, stride).sum(axis=(1, 3))
    mean = values.reshape(4, 64, stride, 64, stride).sum(axis=(2, 4)) / np.maximum(coverage, 1)
    # Save signed centikelvin relative to 240 K and coverage in percent.
    return np.concatenate((np.where(coverage, np.rint((mean - 240) * 100), 0),
                           np.rint(100 * coverage[None] / stride ** 2))).astype(np.int16)


def frame_features(image: np.ndarray, valid: np.ndarray, center_xy: np.ndarray) -> FrameFeatures:
    """Extract structure at the (column, line) center, in edge-origin pixel coordinates."""
    if image.shape != (4, *valid.shape) or np.shape(center_xy) != (2,) or not np.all(np.isfinite(center_xy)):
        raise ValueError("Expected [4,H,W] temperatures, matching validity and finite (x,y) center")
    center = np.asarray(center_xy)[::-1]
    eye, properties = eye_candidate(image, valid, center)
    nominal, nominal_asymmetry = _annuli(image, valid, center)
    core, asymmetry = _annuli(image, valid, eye)
    # Local warm/cold contrasts at different scales and angular variability.
    contrast = core[:4, :4, None, :] - core[4:8, :4].transpose(1, 0, 2)[None]
    structure = np.concatenate((properties, core[:8].ravel(), contrast.ravel(),
                                asymmetry[2:9].ravel()))
    views = np.concatenate((_view(image, valid, eye, 1), _view(image, valid, center, 4)))
    return FrameFeatures(np.r_[nominal.ravel(), nominal_asymmetry[2:].ravel()], structure, views)


def summarize_time(values: np.ndarray, times: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Use only available frames; all-missing columns remain explicitly NaN."""
    if values.ndim != 3 or values.shape[:2] != times.shape or times.shape != valid.shape:
        raise ValueError("Mismatched record/frame dimensions")
    if not np.all((valid & np.isfinite(times)).any(axis=1)):
        raise ValueError("Every record needs at least one timed, available frame")
    ok = np.isfinite(values) & (valid & np.isfinite(times))[..., None]
    lag = np.where(valid & np.isfinite(times), times, -np.inf)
    elapsed = lag - lag.max(axis=1, keepdims=True)
    recent_weights = np.exp(elapsed[..., None] / 45.0)

    def average(weights):
        w = np.where(ok, weights, 0.0)
        mass = w.sum(axis=1)
        moment = (np.where(ok, values, 0.0) * w).sum(axis=1)
        return np.divide(moment, mass, out=np.full_like(moment, np.nan), where=mass > 0)

    nearest = np.argmax(np.where(ok, lag[..., None], -np.inf), axis=1)
    latest = np.take_along_axis(values, nearest[:, None, :], axis=1)[:, 0]
    latest = np.where(ok.any(axis=1), latest, np.nan)
    recent = average(recent_weights)
    past = average((elapsed <= -60)[..., None])
    return {"recent": recent, "latest": latest, "trend": recent - past}


def record_folds(size: int, folds: int, seed: int) -> np.ndarray:
    """Shuffle at record level; metadata and labels do not affect assignment."""
    if not 2 <= folds <= size:
        raise ValueError("Need 2 <= folds <= size")
    assignment = np.empty(size, dtype=np.int64)
    assignment[np.random.default_rng(seed).permutation(size)] = np.arange(size) % folds
    return assignment


def scores(y: np.ndarray, prediction: np.ndarray, quality: np.ndarray) -> dict:
    if y.shape != prediction.shape or y.shape != quality.shape or not np.isfinite(prediction).all():
        raise ValueError("Metrics need one finite prediction per target")
    error = np.asarray(prediction, dtype=np.float64) - y
    result = {}
    for name in ("all", *QUALITY_WEIGHT):
        selected = error if name == "all" else error[quality == name]
        if len(selected):
            result[name] = {"n": len(selected), "rmse": float(np.sqrt(np.mean(selected ** 2))),
                            "mae": float(np.abs(selected).mean()), "bias": float(selected.mean())}
    return result


def _normalization_state(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-only column indices, means and scales."""
    finite = np.isfinite(train)
    count = finite.sum(axis=0)
    keep = count >= max(3, math.ceil(len(train) * 0.15))
    train, finite, count = train[:, keep], finite[:, keep], count[keep]
    mean = np.where(finite, train, 0).sum(axis=0) / count
    residual = np.where(finite, train - mean, 0)
    scale = np.sqrt((residual ** 2).sum(axis=0) / count).clip(0.05)
    return np.flatnonzero(keep), mean, scale


def _apply_normalization(values, columns, mean, scale):
    x = values[:, columns]
    return np.where(np.isfinite(x), np.clip((x - mean) / scale, -5.0, 5.0), 0.0) if columns.size else x


def _normalize(train: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-only scaling; zero means missing after centering, not a physical temperature."""
    state = _normalization_state(train)
    return _apply_normalization(train, *state), _apply_normalization(evaluate, *state)


def kernel_components(groups: dict[str, np.ndarray], train: np.ndarray, evaluate: np.ndarray) -> dict:
    """Compute normalized dot products and squared distances for each feature group."""
    result = {}
    for name, data in groups.items():
        a, b = _normalize(data[train], data[evaluate])
        if a.shape[1] == 0:
            continue
        x = np.concatenate((a, b), axis=0)
        xx = (x @ x.T) / x.shape[1]
        norm = xx.diagonal()
        dist = (norm[:, None] + norm[None, :] - 2 * xx).clip(0.0)
        result[name] = xx, dist
    return result


def rbf_kernel(components: dict, mixture: dict[str, float], width: float, linear: float) -> np.ndarray:
    effective = {k: w for k, w in mixture.items() if k in components and w > 0}
    if not effective or width <= 0 or linear < 0:
        raise ValueError("Need a positive feature mixture, width and nonnegative linear term")
    mass = sum(effective.values())
    dot = sum((v / mass) * components[k][0] for k, v in effective.items())
    dist = sum((v / mass) * components[k][1] for k, v in effective.items())
    # Multi-scale smooth and Matérn-3/2 components plus an extrapolating linear path.
    r = np.sqrt(3 * dist / width)
    return 0.55 * np.exp(-dist / (2 * width)) + 0.45 * (1 + r) * np.exp(-r) + linear * dot


def ridge_path(kernel: np.ndarray, y: np.ndarray, weight: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """Kernel ridge with an unpenalized intercept; one eigensolve for the penalty path."""
    n = len(y)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or len(kernel) < n:
        raise ValueError("Kernel needs square train+evaluation blocks")
    w = weight / weight.sum()
    root = np.sqrt(weight)
    train_means = kernel[:, :n] @ w
    centered = kernel[:, :n] - train_means[:, None] - train_means[None, :n] + w @ train_means[:n]
    average = w @ y
    gram = centered[:n] * root[:, None] * root[None, :]
    e, v = np.linalg.eigh(gram)
    coeff = v.T @ ((y - average) * root)
    return average + ((centered[n:] * root) @ v) @ (coeff[:, None] / (e.clip(0)[:, None] + penalty))

