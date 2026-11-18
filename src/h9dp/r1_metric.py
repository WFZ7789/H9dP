"""Pressure-aware morphology and thermal-scene kernels for small training sets."""

from __future__ import annotations

import numpy as np

from h9dp.r1 import RING_EDGES, _normalize, summarize_time


def _nan_extreme(x: np.ndarray, axis: int, maximum: bool) -> np.ndarray:
    finite = np.isfinite(x)
    if maximum:
        result = np.max(np.where(finite, x, -np.inf), axis=axis)
    else:
        result = np.min(np.where(finite, x, np.inf), axis=axis)
    return np.where(finite.any(axis=axis), result, np.nan)


def morphology(data: dict[str, np.ndarray]) -> np.ndarray:
    """Temperature contrasts and extremum radii in a candidate eye and its cloud ring."""
    structure = data["structure"]
    nominal = data["profile"][..., : (len(RING_EDGES) - 1) * 5 * 7]
    nominal = nominal.reshape(*nominal.shape[:2], -1, 5, 7)
    core = structure[..., 13:13 + 8 * 5 * 7].reshape(*structure.shape[:2], 8, 5, 7)
    warm_eye = _nan_extreme(core[..., :5, 3, :4], 2, True)
    thermals = [structure[..., :13], warm_eye]
    for source in (nominal, core):
        cloud = _nan_extreme(source[..., 3:8, 1, :4], 2, False)
        thermals.extend((warm_eye - cloud, warm_eye - source[..., 4:7, 2, :4].mean(axis=2),
                         source[..., 4:7, 3, :4].mean(axis=2) - cloud))
    for band in (1, 2, 3):
        radial = core[..., :, 3, band]
        good = np.isfinite(radial).any(axis=2)
        warm = np.argmax(np.where(np.isfinite(radial), radial, -np.inf), axis=2)
        cold = 3 + np.argmin(np.where(np.isfinite(radial[..., 3:]), radial[..., 3:], np.inf), axis=2)
        radius = np.array(RING_EDGES[1:9], dtype=np.float32)
        thermals.append(np.stack((np.where(good, radius[warm], np.nan), np.where(good, radius[cold], np.nan)), -1))
    return np.concatenate(thermals, axis=-1)


def image_fingerprint(views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masked, low-spatial-frequency scene fingerprint; no location or absolute clock input."""
    view = views[..., 5:, :, :]
    shape = (*view.shape[:2], 5, 16, 4, 16, 4)
    patch = view.reshape(shape)
    weight = patch[:, :, 4:5].astype(np.float32) * 0.01
    mass = weight.sum(axis=(4, 6))
    temp = (patch[:, :, :4] * weight).sum(axis=(4, 6)) / np.maximum(mass, 1e-6) * 0.01
    temp = np.where(mass >= 8, temp, np.nan)
    color = np.concatenate((temp[..., 0:1, :, :], temp[..., 2:4, :, :], temp[..., 2:3, :, :] - temp[..., :1, :, :]), 2)
    return color.reshape(*color.shape[:2], -1), color[..., 4:12, 4:12].reshape(*color.shape[:2], -1)


def contour_features(views: np.ndarray) -> np.ndarray:
    """Measure azimuthal eye closure, warm-region size, and warm-to-cold radial relief."""
    direction = np.arange(48)[:, None] * (2 * np.pi / 48)
    radius = np.arange(1, 30)[None]
    row = 32 + np.rint(np.sin(direction) * radius).astype(int)
    col = 32 + np.rint(np.cos(direction) * radius).astype(int)
    temp = views[..., :4, row, col].astype(np.float32) * 0.01 + 240
    valid = views[..., 4:5, row, col] == 100
    temp = np.where(valid, temp, np.nan)

    def mean(x, axis, support: int = 1):
        ok = np.isfinite(x)
        n = ok.sum(axis=axis)
        return np.where(n >= support, np.where(ok, x, 0).sum(axis=axis) / n.clip(1), np.nan)

    warm = mean(mean(temp[..., :4], -1, 2), -1, 24)
    cold_ray = _nan_extreme(temp[..., 8:25], -1, False)
    cold = mean(cold_ray, -1, 24)
    relief = warm - cold
    features = [warm, cold, relief, relief * ((308.15 - cold) / 100)]
    scaled = (temp - cold_ray[..., None]) / np.maximum(warm[..., None, None] - cold_ray[..., None], 1.5)
    for threshold in (0.25, 0.5, 0.75):
        inside = (scaled >= threshold) & valid
        # The first sustained crossing is robust to isolated pixellation and holes.
        boundary = ~inside[..., :-1] & ~inside[..., 1:]
        crossing = np.where(boundary.any(axis=-1), boundary.argmax(axis=-1) + 1.0, float(radius.max()))
        crossing = np.where((valid.sum(axis=-1) >= 24) & np.isfinite(cold_ray), crossing, np.nan)
        effective_radius = np.sqrt(mean(crossing ** 2, -1, 24))
        average = mean(crossing, -1, 24)
        elongation = np.sqrt(mean((crossing - average[..., None]) ** 2, -1, 24))
        coherence = mean(np.where(np.isfinite(crossing), (crossing < 24).astype(float), np.nan), -1, 24)
        features.extend((effective_radius, elongation, coherence, relief / (effective_radius + 4)))
    return np.concatenate(features, axis=-1)


def spectral_blocks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Distinct wavelength/shape summaries so one family cannot swamp the others by size."""
    shape = data["structure"].shape[:2]
    core = data["structure"][..., 13:13 + 8 * 5 * 7].reshape(*shape, 8, 5, 7)
    broad = data["regions"].reshape(*shape, 4, 3, 3)
    cloud = _nan_extreme(core[..., 3:, 1, :4], 2, False)
    lift = np.clip((308.15 - cloud) / 100.0, 0.50, 1.35)
    central = core[..., :4, 3, :4]
    shock = (central - cloud[..., None, :]) * lift[..., None, :]
    # Approximate vertical thermal contrasts; neither these nor the contour is a pressure label.
    return {
        "thermal_lift": shock.reshape(*shape, -1),
        "warm_brightness": central.reshape(*shape, -1),
        "outer_cloud": np.concatenate((cloud, broad[..., 1:, :2].reshape(*shape, -1)), -1),
        "spectral_edge": core[..., :6, :4, 4:].reshape(*shape, -1),
        "radial_slope": np.diff(core[..., :8, 3, :4], axis=2).reshape(*shape, -1),
    }


def extra_groups(data: dict[str, np.ndarray], views: np.ndarray, contour: bool = False,
                 spectral: bool = False) -> dict[str, np.ndarray]:
    scene, inner = image_fingerprint(views)
    inputs = {"thermal": morphology(data), "scene": scene, "scene_core": inner}
    if contour:
        inputs["contour"] = contour_features(views)
    if spectral:
        inputs.update(spectral_blocks(data))
    return {f"{name}_{stat}": value for name, frames in inputs.items()
            for stat, value in summarize_time(frames, data["time"], data["valid"]).items()}


def relevance_components(groups: dict[str, np.ndarray], train: np.ndarray, evaluate: np.ndarray,
                         y: np.ndarray, quality: np.ndarray) -> tuple[dict, dict]:
    """Two metrics, uniform and softly supervised by fold-local pressure dependence."""
    uniform, supervised = {}, {}
    sample_weights = np.where(quality[train] == "Good", 2.0, 1.0)
    w = sample_weights / sample_weights.sum()
    label = y[train] - w @ y[train]
    label = label / max(float(np.sqrt(w @ (label ** 2))), 1e-6)
    for name, values in groups.items():
        a, b = _normalize(values[train], values[evaluate])
        if not a.shape[1]:
            continue
        x = np.concatenate((a, b), axis=0)
        feature_covariance = (label * w) @ a
        relevance = 0.35 + feature_covariance ** 2 / max(np.mean(feature_covariance ** 2), 0.015)
        uniform[name] = _component(x, np.ones(a.shape[1]))
        supervised[name] = _component(x, relevance.clip(0.35, 6.0))
    return uniform, supervised


def _component(x: np.ndarray, relevance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = x * np.sqrt(relevance / relevance.sum())
    dot = z @ z.T
    norm = dot.diagonal()
    return dot, (norm[:, None] + norm[None, :] - 2 * dot).clip(0)


def analog_kernel(components: dict, width: float) -> np.ndarray:
    """Narrow visual-neighborhood prior, additive to a broader morphology prior."""
    _, latest = components["scene_latest"]
    _, recent = components["scene_recent"]
    _, core = components["scene_core_recent"]
    _, trend = components["scene_trend"]
    return np.exp(-(0.15 * latest + 0.4 * recent + 0.35 * core + 0.1 * trend) / (2 * width))
