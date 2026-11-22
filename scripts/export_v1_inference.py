"""Export the trained V1 GP and pack inputs for the standalone C++ predictor."""

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import (
    _apply_normalization,
    _normalization_state,
    record_folds,
    summarize_time,
)

# Binary layout must agree with inference/inference_best.cpp.
GROUPS = {"eye": 320, "wall": 400, "relief": 320, "waves": 288, "motion": 60, "sector": 672}
FRAMES = 6
FOLDS = 5
SEED = 1
MAX_TRAIN = 4096
SEQUENCE_WIDTHS = np.array([0.035, 0.14, 0.55])
SPLINE_WIDTHS = (0.7, 1.8)


def _array(stream, values, dtype):
    stream.write(np.asarray(values, dtype=dtype, order="C").tobytes())


def _frame_validity(groups, times):
    if times.ndim != 2 or times.shape[1] != FRAMES or not len(times):
        raise ValueError("Expected one or more six-frame minute sequences")
    for name, size in GROUPS.items():
        if groups[name].shape != (*times.shape, size) or np.isinf(groups[name]).any():
            raise ValueError(f"Invalid V1 feature group: {name}")
    valid = np.isfinite(groups["eye"]).any(axis=2)
    if not valid.any(axis=1).all() or np.any(valid & ~np.isfinite(times)) or np.isinf(times).any():
        raise ValueError("Every record needs finite times for all eye-valid frames, and at least one such frame")
    return valid


def _write_normalizer(stream, values):
    columns, mean, scale = state = _normalization_state(values)
    stream.write(struct.pack("<I", len(columns)))
    _array(stream, columns, "<u4")
    _array(stream, mean, "<f8")
    _array(stream, scale, "<f8")
    return _apply_normalization(values, *state)


def _write_stat(stream, name, stat, values, dual, weight):
    x = _write_normalizer(stream, values)
    if not x.shape[1]:
        return
    d = x.shape[1]
    _array(stream, weight.pop(f"{name}_{stat}_matern"), "<f8")
    _array(stream, (dual @ x) * (weight.pop(f"{name}_{stat}_linear") / d), "<f8")
    _array(stream, x, "<f8")
    if name in ("eye", "wall", "relief") and stat == "recent":
        for width in SPLINE_WIDTHS:
            phi = np.exp(-0.5 * ((x[..., None] - np.linspace(-3, 3, 9)) / width) ** 2)
            phi /= np.linalg.norm(phi, axis=-1, keepdims=True)
            projected = np.einsum("n,ndk->dk", dual, phi) * (weight.pop(f"{name}_spline_{width}") / d)
            _array(stream, projected, "<f8")


def _write_sequence(stream, name, frames, times, valid, weight):
    n = len(frames)
    flat = _write_normalizer(stream, frames.reshape(n * FRAMES, -1))
    d = flat.shape[1]
    if not d:
        raise ValueError(f"No trained {name} sequence columns")
    x = flat.astype(np.float32).reshape(n, FRAMES, d)
    elapsed = times - np.max(np.where(valid, times, -np.inf), axis=1, keepdims=True)
    w = np.exp(np.where(valid, elapsed / 180.0, -np.inf))
    w = (w / w.sum(axis=1, keepdims=True)).astype(np.float32).astype(np.float64)
    z = x.astype(np.float64)
    diff = z[:, :, None] - z[:, None, :]
    dist = np.einsum("nijd,nijd->nij", diff, diff) / d
    scales = []
    for width in SEQUENCE_WIDTHS:
        diag = np.einsum("ni,nij,nj->n", w, np.exp(-dist / (2 * width)), w)
        scales.append(1 / np.sqrt(np.maximum(diag, 1e-6)))
    _array(stream, [weight.pop(f"{name}_sequence_{i}") for i in range(len(scales))], "<f8")
    _array(stream, w, "<f8")
    _array(stream, x, "<f4")
    _array(stream, np.stack(scales, axis=1), "<f8")


def write_model(path, groups, times, names, masses, dual, target_mean, target_std):
    """Serialize one fitted posterior; the inputs are this fold's training data only."""
    times = np.asarray(times, dtype=np.float64)
    valid = _frame_validity(groups, times)
    n = len(times)
    masses, dual = np.asarray(masses), np.asarray(dual, dtype=np.float64).reshape(-1) * target_std
    if (
        not 0 < n <= MAX_TRAIN or masses.shape != (len(names),) or len(set(names)) != len(names)
        or not np.isfinite(masses).all() or np.any(masses < 0) or dual.shape != (n,)
        or not np.isfinite(dual).all() or not np.isfinite([target_mean, target_std]).all() or target_std <= 0
    ):
        raise ValueError("Invalid V1 posterior coefficients or dimensions")
    weight = dict(zip(names, masses, strict=True))
    payload = io.BytesIO()
    payload.write(struct.pack("<8sId", b"H9V1MOD1", n, target_mean + 8.0 * dual.sum()))
    _array(payload, dual, "<f8")
    try:
        for name in GROUPS:
            x = groups[name]
            for stat, values in summarize_time(x, times, valid).items():
                _write_stat(payload, name, stat, values, dual, weight)
            if name in ("waves", "sector"):
                _write_sequence(payload, name, x, times, valid, weight)
    except KeyError as exc:
        raise ValueError(f"Checkpoint missing V1 kernel {exc}") from exc
    if weight:
        raise ValueError(f"Unexpected checkpoint kernels: {sorted(weight)}")
    with path.open("xb") as stream:
        stream.write(payload.getbuffer())


def _read_features(path):
    with np.load(path) as data:
        if "mode" in data and data["mode"].item() != "rotation":
            raise ValueError("V1 inference requires rotation descriptors")
        ids, times = data["record"], data["time"]
        if ids.shape != (len(times),) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid or repeated feature record IDs")
        return ids, times, {key: data[key] for key in GROUPS}, data["source_sha256"].item()


def export(features, source, formal, output):
    import torch

    try:
        rows_raw = (source / "all_meta.json").read_bytes()
        rows = json.loads(rows_raw)["records"]
        metrics = json.loads((formal / "metrics.json").read_bytes())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read V1 source or run metadata: {exc}") from exc
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    ids, times, groups, source_digest = _read_features(features)
    if (
        source_digest != hashlib.sha256(rows_raw).hexdigest() or metrics["source_sha256"] != source_digest
        or metrics["seed"] != SEED or metrics.get("catalog_sha256", catalog_digest) != catalog_digest
        or not np.array_equal(ids, [row["record"] for row in rows])
        or not np.array_equal(times, [row["times"] for row in rows], equal_nan=True)
    ):
        raise ValueError("V1 source, feature and checkpoint provenance differs")
    fold = record_folds(len(ids), FOLDS, SEED)
    output.mkdir(parents=True, exist_ok=False)
    for k in range(FOLDS):
        train = fold != k
        state = torch.load(formal / f"fold{k}.pt", map_location="cpu", weights_only=True)
        if not np.array_equal(state["train_ids"], ids[train]):
            raise ValueError(f"Fold {k} training IDs differ from V1's record split")
        target = np.array([row["dP"] for row, keep in zip(rows, train, strict=True) if keep])
        if not np.allclose([state["target_mean"], state["target_std"]], [target.mean(), target.std()],
                           atol=1e-10, rtol=0):
            raise ValueError(f"Fold {k} target scaling differs from the root catalog")
        write_model(output / f"fold{k}.v1m", {g: x[train] for g, x in groups.items()}, times[train],
                    state["names"], state["mass"].exp().numpy(), state["dual"].numpy(),
                    state["target_mean"], state["target_std"])


def pack(path, groups, times):
    """Write timed V1 frame descriptors in batch order, without labels or IDs."""
    times = np.asarray(times, dtype=np.float64)
    groups = {key: np.asarray(groups[key], dtype=np.float32) for key in GROUPS}
    _frame_validity(groups, times)
    with path.open("xb") as stream:
        stream.write(struct.pack("<8sI", b"H9V1REQ1", len(times)))
        for i, minutes in enumerate(times):
            _array(stream, minutes, "<f8")
            for name in GROUPS:
                _array(stream, groups[name][i], "<f4")


def pack_features(features, output, fold=None):
    ids, times, groups, _ = _read_features(features)
    if fold is not None:
        held = record_folds(len(ids), FOLDS, SEED) == fold
        times, groups = times[held], {key: x[held] for key, x in groups.items()}
    pack(output, groups, times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, args in (("export", ("features", "source", "formal")), ("pack-features", ("features",))):
        command = sub.add_parser(name)
        for key in (*args, "output"):
            command.add_argument(f"--{key}", type=Path, required=True)
        if name == "pack-features":
            command.add_argument("--fold", type=int, choices=range(FOLDS), help="Pack only this fold's held records")
    a = parser.parse_args()
    try:
        if a.command == "export":
            export(a.features, a.source, a.formal, a.output)
        else:
            pack_features(a.features, a.output, a.fold)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
