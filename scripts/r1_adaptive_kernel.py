#!/usr/bin/env python3
"""Nested diagnostic of relevance-weighted morphology and additive visual analogs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from h9dp.r1 import rbf_kernel, record_folds, ridge_path, scores
from h9dp.r1_metric import analog_kernel, extra_groups, relevance_components
from scripts.r1_kernel import MIXTURES, Cache, feature_groups, load_cache

BASE = {
    "coarse": MIXTURES["coarse"],
    "morphology": {"regions_recent": 1.0, "regions_latest": 0.30, "thermal_recent": 1.2,
                   "thermal_latest": 0.45, "core_recent": 0.7, "thermal_trend": 0.25},
    "motion": {**MIXTURES["motion"], "regions_recent": 0.7, "thermal_recent": 0.8},
}
LEVELS = np.array([0.0003, 0.002, 0.01, 0.04, 0.16], dtype=np.float64)


@dataclass(frozen=True)
class Settings:
    family: str
    width: float
    linear: float
    supervised: bool
    analog_width: float
    analog_mass: float
    good_priority: bool


def proposals() -> list[Settings]:
    choices = []
    for family, width, linear, priority in product(BASE, (0.6, 2.5, 8.0), (0.16, 0.8), (False, True)):
        choices.extend((Settings(family, width, linear, False, 0, 0, priority),
                        Settings(family, width, linear, True, 0, 0, priority)))
    for family, width, analog, mass, priority in product(
            ("coarse", "morphology"), (1.5, 5.0), (0.08, 0.28), (0.18, 0.75), (False, True)):
        choices.append(Settings(family, width, 0.45, True, analog, mass, priority))
    return choices


def pressure_weights(y: np.ndarray, quality: np.ndarray, priority: bool) -> np.ndarray:
    if priority:
        w = np.select((quality == "Good", quality == "Fair"), (1.0, 0.27), default=0.12)
    else:
        w = np.select((quality == "Good", quality == "Fair"), (1.0, 0.6), default=0.32)
    # A modest continuous tail weight improves focus without removing any record.
    w = w * (1 + 0.5 * np.clip((y - 90) / 45, 0, 1))
    return w / w.mean()


def pool(groups, y, quality, train, held, settings):
    components = relevance_components(groups, train, held, y, quality)
    main_cache = {}
    analog_cache = {}
    weights = {priority: pressure_weights(y[train], quality[train], priority) for priority in (False, True)}
    estimate = []
    for spec in settings:
        key = (spec.family, spec.width, spec.linear, spec.supervised)
        if key not in main_cache:
            main_cache[key] = rbf_kernel(components[spec.supervised], BASE[spec.family], spec.width, spec.linear)
        kernel = main_cache[key]
        if spec.analog_mass:
            if spec.analog_width not in analog_cache:
                analog_cache[spec.analog_width] = analog_kernel(components[1], spec.analog_width)
            kernel = kernel + spec.analog_mass * analog_cache[spec.analog_width]
        estimate.append(ridge_path(kernel, y[train], weights[spec.good_priority], LEVELS))
    return np.stack(estimate, axis=1)


def adaptive_predictions(groups, y, quality, train, held, seed: int, inner_folds: int = 3):
    settings = proposals()
    assignment = record_folds(len(train), inner_folds, seed)
    oof = np.empty((len(train), len(settings), len(LEVELS)))
    for fold in range(inner_folds):
        mask = assignment == fold
        oof[mask] = pool(groups, y, quality, train[~mask], train[mask], settings)
    error = oof - y[train, None, None]
    mse = (error ** 2).mean(axis=0)
    good = quality[train] == "Good"
    cost = (0.76 * (error[good] ** 2).mean(axis=0) + 0.24 * mse) if good.any() else mse
    selected = {}
    tags = {
        "relevance": [i for i, s in enumerate(settings) if not s.analog_mass],
        "analog": [i for i, s in enumerate(settings) if s.analog_mass],
        "adaptive": list(range(len(settings))),
    }
    for tag, candidates in tags.items():
        per_model = [(i, int(np.argmin(cost[i]))) for i in candidates]
        selected[tag] = sorted(per_model, key=lambda ij: cost[ij])[:5]
    needed = sorted({i for subset in selected.values() for i, _ in subset})
    final = pool(groups, y, quality, train, held, [settings[i] for i in needed])
    estimates = {name: np.mean([final[:, needed.index(i), j] for i, j in subset], axis=0)
                 for name, subset in selected.items()}
    trace = {name: [{**asdict(settings[i]), "penalty": LEVELS[j], "inner_objective": cost[i, j]} for i, j in subset]
             for name, subset in selected.items()}
    return estimates, trace


def latest_snapshot(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """For a one-scan model, choose the latest valid frame once per record."""
    known = data["valid"] & np.isfinite(data["time"])
    if not known.any(axis=1).all():
        raise ValueError("Every snapshot needs a timed valid frame")
    latest = np.argmax(np.where(known, data["time"], -np.inf), axis=1)
    return {**data, "valid": (np.arange(known.shape[1]) == latest[:, None])}


def assemble(cache: Cache, other: dict[str, np.ndarray] | None = None, views: np.ndarray | None = None,
             *, contour: bool = False, spectral: bool = False, snapshot: bool = False, warm_column: bool = False,
             aperture: bool = False):
    def extract(data, view):
        data = latest_snapshot(data) if snapshot else data
        return {**feature_groups(data), **extra_groups(data, view, contour, spectral, warm_column, aperture)}

    stored_views = np.load(cache.root / "views.npy", allow_pickle=False, mmap_mode="r")
    groups = extract(cache.data, stored_views)
    if (other is None) != (views is None):
        raise ValueError("External descriptors and external pixel views must be supplied together")
    if other is not None and views is not None:
        additional = extract(other, views)
        groups = {name: np.concatenate((data, additional[name])) for name, data in groups.items()}
    return groups


def evaluate(cache: Cache, output: Path, seeds: list[int], folds: int):
    groups = assemble(cache)
    output.mkdir(exist_ok=False)
    res = {}
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        tags = ["relevance", "analog", "adaptive"]
        writer.writerow(["seed", "fold", "record", "quality", "target", *tags])
        for seed in seeds:
            assignment = record_folds(len(cache.rows), folds, seed)
            predicted = {tag: np.full(len(assignment), np.nan) for tag in tags}
            trace = []
            for fold in range(folds):
                t = time.monotonic()
                train, held = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                forecast, report = adaptive_predictions(groups, cache.y, cache.quality, train, held, seed)
                for tag, rows in forecast.items():
                    predicted[tag][held] = rows
                trace.append(report)
                for i in held:
                    writer.writerow([seed, fold, cache.data["record"][i], cache.quality[i], cache.y[i],
                                     *(predicted[name][i] for name in tags)])
                stream.flush()
                metrics = {name: scores(cache.y[held], forecast[name], cache.quality[held])["Good"]["rmse"] for name in tags}
                print(f"seed={seed} fold={fold} {time.monotonic()-t:.2f}s Good RMSE {metrics}", flush=True)
            res[str(seed)] = {"trace": trace, "metrics": {tag: scores(cache.y, p, cache.quality) for tag, p in predicted.items()}}
            print(json.dumps(res[str(seed)]["metrics"]), flush=True)
    report = {"schema": "r1-adaptive-kernel-development/v1", "features": str(cache.root.resolve()),
              "source_meta_sha256": cache.source_digest, "folds": folds, "inner_folds": 3,
              "all_records": len(cache.rows), "candidate_models": len(proposals()),
              "execution": "CPU NumPy closed-form diagnostic", "results": res}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs=1, default=[1])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    evaluate(load_cache(args.features), args.output, args.seeds, args.folds)


if __name__ == "__main__":
    main()
