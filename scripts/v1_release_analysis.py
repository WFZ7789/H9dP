"""Recompute the V1 release analysis from existing seed-1 out-of-fold predictions.

Every number here is derived from published per-record OOF tables; no model is
refit and no prediction is regenerated. The report that quotes these values is
docs/V1/RELEASE_ANALYSIS.md, which records the protocol and the seed.
"""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

# Run directory (relative to --runs) and prediction column of each published model.
# R1 quotes the fixed-center half-rotation variant, as in docs/EVALUATION.md.
MODELS = {
    "V1": ("v1/rotation_seed1/formal", "prediction"),
    "A1": ("a1/analogs_seed1/formal", "prediction"),
    "R1": ("r1/center_seed1_20261121/paired", "half"),
    "E1": ("e1/isotherms_seed1/formal", "prediction"),
    "C1": ("c1/cdo_ring_seed1/formal", "prediction"),
    "F1": ("f1/temporal_seed1/formal", "prediction"),
    "H1": ("h1/column_seed1/formal", "prediction"),
    "D1": ("d1/signature_seed1/folds", "prediction"),
    "W1": ("w1/radiance_seed1/formal", "prediction"),
    "EC1": ("ec1/pixels_seed1/formal", "prediction"),
}
REGIMES = ((0, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, math.inf))


def read_model(runs, directory, column):
    path = runs / directory / "predictions.csv"
    rows = {}
    for row in csv.DictReader(path.open()):
        if row["seed"] != "1" or row.get(column) in (None, ""):
            continue
        rows[int(row["record"])] = (float(row["target"]), float(row[column]), row["quality"], row["fold"])
    return rows


def rmse(errors):
    return math.sqrt(sum(x * x for x in errors) / len(errors))


def median_interval(values, low=0.025, high=0.975):
    ordered = sorted(values)
    return ordered[int(low * len(ordered))], ordered[int(high * len(ordered)) - 1]


def bootstrap_mse_difference(reference, other, resamples, seed):
    rng = random.Random(seed)
    n = len(reference)
    draws = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            total += other[j] ** 2 - reference[j] ** 2
        draws.append(total / n)
    return median_interval(draws)


def precision_curve(errors, targets):
    out = []
    for low, high in REGIMES:
        index = [i for i, value in enumerate(targets) if low <= value < high]
        if not index:
            continue
        picked = [errors[i] for i in index]
        out.append(
            {
                "range_hpa": f"{low:g}-{high:g}" if math.isfinite(high) else f">={low:g}",
                "n": len(picked),
                "rmse_hpa": rmse(picked),
                "mae_hpa": sum(abs(x) for x in picked) / len(picked),
                "bias_hpa": sum(picked) / len(picked),
            }
        )
    return out


def blend(records, keys, targets):
    return rmse([sum(records[key][record][1] for key in keys) / len(keys) - targets[record] for record in records["V1"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("/mnt/data/H9dP/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()

    records = {name: read_model(args.runs, *spec) for name, spec in MODELS.items()}
    common = set.intersection(*(set(rows) for rows in records.values()))
    if len(common) != 546:
        raise SystemExit(f"Expected the published 546 record set, found {len(common)}")
    for name, rows in records.items():
        records[name] = {key: rows[key] for key in common}

    keys = sorted(common)
    targets = {key: records["V1"][key][0] for key in keys}
    for name in records:
        if any(abs(records[name][key][0] - targets[key]) > 1e-9 for key in keys):
            raise SystemExit(f"{name} does not share the V1 target column")
    reference = [records["V1"][key][1] - targets[key] for key in keys]

    report = {"seed": 1, "records": len(keys), "models": {}, "comparisons": {}, "regimes": {}, "blends": {}}
    for name, rows in records.items():
        errors = [rows[key][1] - targets[key] for key in keys]
        report["models"][name] = {
            "rmse_hpa": rmse(errors),
            "mae_hpa": sum(abs(x) for x in errors) / len(errors),
            "bias_hpa": sum(errors) / len(errors),
        }
    for name in MODELS:
        if name == "V1":
            continue
        errors = [records[name][key][1] - targets[key] for key in keys]
        low, high = bootstrap_mse_difference(reference, errors, args.resamples, 1)
        report["comparisons"][name] = {
            "rmse_delta_hpa": rmse(errors) - rmse(reference),
            "mse_delta_ci_hpa2": [low, high],
            "separated": low > 0 or high < 0,
            "error_correlation": statistics.correlation(reference, errors),
        }

    report["regimes"]["V1"] = precision_curve(reference, [targets[key] for key in keys])
    magnitudes = sorted(abs(x) for x in reference)
    report["v1_tail"] = {
        "abs_error_median_hpa": statistics.median(magnitudes),
        "abs_error_p90_hpa": magnitudes[int(0.9 * len(magnitudes)) - 1],
        "abs_error_max_hpa": magnitudes[-1],
    }
    groups = defaultdict(list)
    folds = defaultdict(list)
    for key, error in zip(keys, reference, strict=True):
        groups[records["V1"][key][2]].append(error)
        folds[records["V1"][key][3]].append(error)
    report["v1_quality"] = {name: {"n": len(v), "rmse_hpa": rmse(v)} for name, v in groups.items()}
    report["v1_folds"] = {
        name: {"n": len(v), "rmse_hpa": rmse(v), "bias_hpa": sum(v) / len(v)} for name, v in sorted(folds.items())
    }

    top = ("V1", "A1", "R1", "E1", "C1")
    report["blends"] = {
        "note": "Post-hoc equal-weight averages over the same OOF table; optimistic, not released models.",
        "V1+A1": blend(records, ("V1", "A1"), targets),
        "V1+A1+R1": blend(records, ("V1", "A1", "R1"), targets),
        "top5": blend(records, top, targets),
        "all10": blend(records, tuple(MODELS), targets),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("| 模型 | RMSE | ΔRMSE | MSE 差 95% 区间 | 可分 | 误差相关 |")
    print("|---|---:|---:|---|---|---:|")
    for name in MODELS:
        entry = report["models"][name]
        if name == "V1":
            print(f"| {name} | {entry['rmse_hpa']:.3f} | — | — | — | 1.000 |")
            continue
        diff = report["comparisons"][name]
        low, high = diff["mse_delta_ci_hpa2"]
        print(
            f"| {name} | {entry['rmse_hpa']:.3f} | {diff['rmse_delta_hpa']:+.3f} | "
            f"[{low:+.1f}, {high:+.1f}] | {'是' if diff['separated'] else '否'} | "
            f"{diff['error_correlation']:.3f} |"
        )
    print()
    print("| 目标 dP (hPa) | 记录 | RMSE | MAE | 偏差 |")
    print("|---|---:|---:|---:|---:|")
    for row in report["regimes"]["V1"]:
        print(
            f"| {row['range_hpa']} | {row['n']} | {row['rmse_hpa']:.3f} | "
            f"{row['mae_hpa']:.3f} | {row['bias_hpa']:+.3f} |"
        )
    print()
    for name, value in report["blends"].items():
        if name != "note":
            print(f"{name}: {value:.4f}")
    print(json.dumps({k: v for k, v in report.items() if k in ("v1_tail", "v1_folds")}, indent=2))


if __name__ == "__main__":
    main()
