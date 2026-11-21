#!/usr/bin/env python3
"""Nested record-level development comparison of closed-form R1 kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from h9dp.r1 import (
    FEATURE_SCHEMA,
    QUALITY_WEIGHT,
    kernel_components,
    rbf_kernel,
    record_folds,
    ridge_path,
    scores,
    summarize_time,
)

REGIONS = {"regions_latest": 1.0, "regions_recent": 1.0, "regions_trend": 0.20}
PROFILE = {"profile_latest": 0.4, "profile_recent": 1.0}
STRUCTURE = {"structure_latest": 0.6, "structure_recent": 1.0}
CORE = {"core_latest": 0.6, "core_recent": 1.0}
WIDE = {**PROFILE, **STRUCTURE, **CORE, "regions_recent": 0.35}
MIXTURES = {
    "coarse": REGIONS,
    "profile": {**PROFILE, "profile_trend": 0.2},
    "eye": {**STRUCTURE, **CORE},
    "wide": WIDE,
    "motion": {**WIDE, "profile_trend": 0.40, "structure_trend": 0.40, "core_trend": 0.30},
    "latest": {"profile_latest": 0.6, "structure_latest": 1.0, "core_latest": 1.0},
}
PENALTIES = np.array([0.001, 0.006, 0.025, 0.10, 0.40], dtype=np.float64)


@dataclass(frozen=True)
class KernelSpec:
    mixture: str
    width: float
    linear: float


@dataclass(frozen=True)
class Cache:
    root: Path
    source_digest: str
    rows: list[dict]
    data: dict[str, np.ndarray]
    y: np.ndarray
    quality: np.ndarray


def read_json(path: Path):
    try:
        raw = path.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}") from exc


def load_cache(root: Path) -> Cache:
    manifest, _ = read_json(root / "manifest.json")
    if manifest.get("schema") != FEATURE_SCHEMA:
        raise ValueError("Unsupported feature cache")
    source, digest = read_json(Path(manifest["source"]) / "all_meta.json")
    if digest != manifest["source_meta_sha256"]:
        raise ValueError("Source metadata changed after feature extraction")
    rows = source["records"]
    with np.load(root / "features.npz", allow_pickle=False) as loaded:
        data = dict(loaded.items())
    if "native_r3_index_sha256" in manifest:
        # This cache uses only rows with native, uncropped R3 frames; preserve source order.
        selected = set(data["record"])
        rows = [row for row in rows if row["record"] in selected]
    if not np.array_equal([r["record"] for r in rows], data["record"]):
        raise ValueError("Feature order differs from source metadata")
    targets = np.array([r["dP"] for r in rows], dtype=np.float64)
    quality = np.array([r["quality"] for r in rows])
    if not np.isfinite(targets).all() or not np.isin(quality, list(QUALITY_WEIGHT)).all():
        raise ValueError("Invalid pressure target or quality tag")
    return Cache(root, digest, rows, data, targets, quality)


def feature_groups(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Compact local descriptors: candidate properties, the inner four annuli and no large-field extent.
    inputs = {k: data[k] for k in ("regions", "profile", "structure")}
    inputs["core"] = data["structure"][:, :, :13 + 4 * 5 * 7]
    return {f"{name}_{stat}": value for name, features in inputs.items()
            for stat, value in summarize_time(features, data["time"], data["valid"]).items()}


def kernel_predictions(groups: dict[str, np.ndarray], y: np.ndarray, weights: np.ndarray,
                       train: np.ndarray, evaluate: np.ndarray, specs: list[KernelSpec]) -> np.ndarray:
    components = kernel_components(groups, train, evaluate)
    return np.stack([ridge_path(rbf_kernel(components, MIXTURES[spec.mixture], spec.width, spec.linear),
                                y[train], weights[train], PENALTIES) for spec in specs], axis=1)


def nested_predict(groups: dict[str, np.ndarray], y: np.ndarray, quality: np.ndarray,
                   train: np.ndarray, evaluate: np.ndarray, seed: int, inner_folds: int):
    """Tune and blend only with outer-training records, then predict outer-evaluation rows."""
    specs = [KernelSpec(*args) for args in product(MIXTURES, (0.3, 1.0, 3.0), (0.0, 0.16))]
    weights = np.array([QUALITY_WEIGHT[k] for k in quality])
    inner = record_folds(len(train), inner_folds, seed)
    oof = np.empty((len(train), len(specs), len(PENALTIES)))
    for fold in range(inner_folds):
        held = inner == fold
        oof[held] = kernel_predictions(groups, y, weights, train[~held], train[held], specs)
    error = (oof - y[train, None, None]) ** 2
    good = quality[train] == "Good"
    all_mse = error.mean(axis=0)
    objective = 0.75 * error[good].mean(axis=0) + 0.25 * all_mse if good.any() else all_mse
    selected = {}
    for tag, mixtures in (("coarse", {"coarse"}), ("structural", set(MIXTURES) - {"coarse"})):
        candidates = [(i, j) for i, spec in enumerate(specs) if spec.mixture in mixtures
                      for j in range(len(PENALTIES))]
        candidates.sort(key=lambda ij: objective[ij])
        # Blend the three best distinct kernels, not three penalties of the same kernel.
        seen: set[int] = set()
        keep = []
        for i, j in candidates:
            if i not in seen:
                keep.append((i, j))
                seen.add(i)
            if len(keep) == 3:
                break
        selected[tag] = keep
    needed = sorted({i for group in selected.values() for i, _ in group})
    blocks = kernel_predictions(groups, y, weights, train, evaluate, [specs[i] for i in needed])
    predictions = {tag: np.mean([blocks[:, needed.index(i), j] for i, j in group], axis=0)
                   for tag, group in selected.items()}
    train_oof = {tag: np.mean([oof[:, i, j] for i, j in group], axis=0) for tag, group in selected.items()}
    trace = {tag: [{**asdict(specs[i]), "penalty": PENALTIES[j], "inner_objective": objective[i, j]}
                   for i, j in group] for tag, group in selected.items()}
    return predictions, train_oof, trace


def evaluate(cache: Cache, output: Path, seeds: list[int], folds: int, inner_folds: int):
    groups = feature_groups(cache.data)
    output.mkdir(exist_ok=False)
    res = {}
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "coarse", "structural"])
        for seed in seeds:
            assignment = record_folds(len(cache.rows), folds, seed)
            predictions = {tag: np.full(len(assignment), np.nan) for tag in ("coarse", "structural")}
            trace = []
            for fold in range(folds):
                started = time.monotonic()
                train = np.flatnonzero(assignment != fold)
                held = np.flatnonzero(assignment == fold)
                fold_pred, _, report = nested_predict(groups, cache.y, cache.quality, train, held,
                                                       seed, inner_folds)
                for tag, values in fold_pred.items():
                    predictions[tag][held] = values
                for i in held:
                    writer.writerow([seed, fold, cache.data["record"][i], cache.quality[i], cache.y[i],
                                     predictions["coarse"][i], predictions["structural"][i]])
                stream.flush()
                trace.append(report)
                print(f"seed={seed} fold={fold} {time.monotonic()-started:.2f}s "
                      f"Good structural={scores(cache.y[held], fold_pred['structural'], cache.quality[held]).get('Good')}",
                      flush=True)
            res[str(seed)] = {"trace": trace, "metrics": {tag: scores(cache.y, values, cache.quality)
                                                         for tag, values in predictions.items()}}
            print(f"seed={seed} {json.dumps(res[str(seed)]['metrics'])}", flush=True)
    report = {"schema": "r1-kernel-development/v1", "features": str(cache.root.resolve()),
              "source_meta_sha256": cache.source_digest, "folds": folds, "inner_folds": inner_folds,
              "all_records": len(cache.rows), "execution": "CPU NumPy closed-form diagnostic", "results": res}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs=1, default=[1])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    evaluate(load_cache(args.features), args.output, args.seeds, args.folds, args.inner_folds)


if __name__ == "__main__":
    main()
