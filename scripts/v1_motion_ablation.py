"""Refit V1 with and without near-10-minute angular motion on its original folds."""

import argparse
import csv
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from h9dp.r1 import record_folds, scores
from h9dp.v1 import RADII
from scripts.v1 import bank, fit
from scripts.v1_why import load, paired_bootstrap

SHORT_RANGE = (8, 12)
VARIANTS = (
    "full", "no_motion", "no_10min", "only_10min", "only_10min_shift", "only_10min_context", "no_10min_shift",
)


def pair_minutes(times, usable):
    """Match V1's pairwise traversal over usable frames, skipping empty slots."""
    result = np.full(times.shape, np.nan)
    for j, (row, mask) in enumerate(zip(times, usable, strict=True)):
        slots = np.flatnonzero(mask & np.isfinite(row))
        result[j, slots[1:]] = np.diff(row[slots])
    return result


def motion_variants(motion, minutes):
    """Return cached motion with selected pairs/components set to missing."""
    short = (minutes >= SHORT_RANGE[0]) & (minutes <= SHORT_RANGE[1])
    rings = motion.reshape(*minutes.shape, len(RADII) - 3, 4)
    shift = np.array([True, False, False, False])
    short = short[..., None, None]
    keep = {
        "no_10min": ~short,
        "only_10min": short,
        "only_10min_shift": short & shift,
        "only_10min_context": short & ~shift,
        "no_10min_shift": ~(short & shift),
    }
    return {key: np.where(mask, rings, np.nan).reshape(motion.shape) for key, mask in keep.items()}


def describe_pairs(motion, minutes):
    values = motion.reshape(*minutes.shape, len(RADII) - 3, 4)
    usable = np.isfinite(values[..., 0]).any(-1)
    if np.any(usable & (~np.isfinite(minutes) | (minutes <= 0) | (minutes > 130))):
        raise ValueError("Cached motion cannot be matched to a usable V1 frame pair")
    summary = {}
    for name, lo, hi in (("short", *SHORT_RANGE), ("20min", 18, 22), ("30min", 28, 32), ("120min", 118, 122)):
        selected = (minutes >= lo) & (minutes <= hi) & usable
        speed, peak, gain, _ = values[selected].reshape(-1, 4).T
        ok = np.isfinite(speed)
        n = ok.sum().item()
        summary[name] = {
            "pairs": selected.sum().item(), "records": selected.any(1).sum().item(),
            "actual_minutes": [minutes[selected].min().item(), minutes[selected].max().item()] if n else None,
            "finite_rings": n, "total_ring_slots": len(speed),
            "nonzero_ring_fraction": np.count_nonzero(speed[ok]) / n if n else None,
            "median_peak": np.median(peak[ok]).item() if n else None,
            "median_gain_vs_zero": np.median(gain[ok]).item() if n else None,
        }
    summary["other_pairs"] = usable.sum().item() - sum(v["pairs"] for v in summary.values())
    return summary


def read_reference(path, ids, target, quality, fold):
    lookup = {record: i for i, record in enumerate(ids)}
    pred = np.full_like(target, np.nan, dtype=float)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                index = lookup.pop(int(row["record"]))
                value = float(row["prediction"])
                target_value = float(row["target"])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Invalid reference at {path}: {row}") from error
            if (row["seed"] != "1" or row["fold"] != str(fold[index]) or row["quality"] != quality[index]
                    or target_value != target[index] or not np.isfinite(value)):
                raise ValueError(f"Mismatched reference prediction at {path}: {row}")
            pred[index] = value
    if lookup:
        raise ValueError("Missing published reference predictions")
    return pred


def fit_variants(groups, motions, times, target, quality, fold):
    valid = np.isfinite(groups["eye"]).any(-1)
    predictions = {key: np.full_like(target, np.nan, dtype=float) for key in VARIANTS}
    timing = []
    for k in range(5):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        full, names = bank(groups, times, train, held, valid=valid)
        motion_index = {name: i for i, name in enumerate(names) if name.startswith("motion_")}
        static = full.clone()
        static[list(motion_index.values())] = 0
        for key, prediction in predictions.items():
            if key == "full":
                basis = full
            elif key == "no_motion":
                basis = static
            else:
                rows, changed = bank({"motion": motions[key]}, times, train, held, valid=valid)
                basis = static.clone()
                basis[[motion_index[name] for name in changed]] = rows
            prediction[held], _, duration = fit(basis, names, target[train], quality[train])
            timing.append({"fold": k, "variant": key, "first_step_s": duration[0], "steady_step_s": duration[1]})
        print(f"Completed fold {k}", flush=True)
    if not all(np.isfinite(pred).all() for pred in predictions.values()):
        raise ValueError("Incomplete or non-finite fold predictions")
    return predictions, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "features", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output / "triton"))
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(1)
    start = monotonic()
    ids, times, groups, target, quality, digest = load(args.source, args.features)
    masks = np.load(args.source / "all_masks.npy", mmap_mode="r", allow_pickle=False)
    usable = masks[:, :, 0].any(axis=(-1, -2))
    minutes = pair_minutes(times, usable)
    pairs = describe_pairs(groups["motion"], minutes)
    motions = motion_variants(groups["motion"], minutes)
    fold = record_folds(len(ids), 5, 1)
    reference = read_reference(args.reference, ids, target, quality, fold)
    pred, timing = fit_variants(groups, motions, times, target, quality, fold)
    max_reference_error = np.abs(pred["full"] - reference).max().item()
    if max_reference_error > 1e-6:
        raise ValueError(f"Full refit differs from the published predictions by {max_reference_error:g} hPa")
    contrasts = {
        ("no_motion", "full"), ("no_10min", "full"), ("only_10min", "no_motion"),
        ("only_10min_shift", "no_motion"), ("only_10min_context", "no_motion"),
        ("no_10min_shift", "full"), ("only_10min", "only_10min_context"),
    }
    metrics = {k: scores(v, target, quality) for k, v in pred.items()}
    result = {
        "seed": 1, "n": len(ids), "catalog_sha256": digest, "short_range_minutes": SHORT_RANGE,
        "features": str(args.features.resolve()), "reference": str(args.reference.resolve()),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "pairs": pairs,
        "reference_max_abs_diff_hpa": max_reference_error, "timing": timing, "wall_s": monotonic() - start,
        "metrics": metrics, "paired_bootstrap": {
            f"{a}_vs_{b}": paired_bootstrap(pred[a], pred[b], target) for a, b in sorted(contrasts)
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "fold", "record", "quality", "target", *VARIANTS))
        writer.writerows((1, fold[i], record, quality[i], target[i], *(pred[k][i] for k in VARIANTS))
                         for i, record in enumerate(ids))
    print(json.dumps({k: v["all"]["rmse"] for k, v in metrics.items()}, indent=2))
    print(json.dumps(result["paired_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
