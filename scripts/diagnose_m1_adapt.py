"""Read-only diagnosis of a completed M1 adaptive OOF stage; never fit experts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from h9dp.m1 import EXPERTS, EYE_SIZE_MAX, M5, metrics
from scripts.m1 import read_json, write_json


def error_decomposition(contributions: np.ndarray, oracle_error: np.ndarray) -> dict:
    independent = float(np.mean(contributions ** 2, axis=0).sum())
    total_delta = contributions.sum(axis=1)
    return {
        "oracle_mse": float(np.mean(oracle_error ** 2)),
        "individual_parameter_delta_mse_sum": independent,
        "cross_parameter_terms": float(np.mean(total_delta ** 2) - independent),
        "oracle_interaction": float(2 * np.mean(oracle_error * total_delta)),
        "raw_mse": float(np.mean((oracle_error + total_delta) ** 2)),
        "parameter_delta_correlation": (np.corrcoef(contributions.T).tolist()
                                        if np.all(contributions.std(axis=0) > 0) else None),
    }


def diagnose(run: Path, epoch: int, data: Path) -> dict:
    meta = read_json(data / "all_meta.json")
    lookup = {r["record"]: r for r in meta["records"]}
    with (run / f"predictions_epoch{epoch:03d}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = [lookup[int(row["record"])] for row in rows]
    target = np.asarray([float(row["dP"]) for row in rows])
    raw = np.asarray([float(row["raw_m5"]) for row in rows])
    calibrated = np.asarray([float(row["calibrated_m5"]) for row in rows])
    predicted = np.asarray([[float(row[f"pred_{spec.parameter}"]) for spec in EXPERTS] for row in rows])
    true = np.asarray([r["params"] if r["params"] is not None else [np.nan] * 5 for r in records])
    paired = np.isfinite(true).all(axis=1)
    good = np.asarray([r["quality"] == "Good" for r in records], dtype=np.bool_)
    folds = np.asarray([int(row["fold"]) for row in rows])
    oracle = np.full(len(rows), np.nan)
    replacements = np.full((len(rows), 5), np.nan)
    contributions = np.full((len(rows), 5), np.nan)
    for fold in np.unique(folds):
        payload = read_json(run / f"outer{fold}" / f"calibration_epoch{epoch:03d}.json")
        teacher = M5(np.asarray(payload["teacher_coefficients"]))
        chosen = paired & (folds == fold)
        oracle[chosen] = teacher.predict(true[chosen])
        predicted_scale = (35 - np.minimum(predicted[chosen, 4], -50)) / 100
        true_scale = (35 - np.minimum(true[chosen, 4], -50)) / 100
        contributions[chosen, :4] = predicted_scale[:, None] * (predicted[chosen, :4] - true[chosen, :4]) * teacher.coefficients[1:]
        contributions[chosen, 4] = (predicted_scale - true_scale) * (true[chosen, :4] @ teacher.coefficients[1:])
        for j in range(5):
            values = true[chosen].copy()
            values[:, j] = predicted[chosen, j]
            replacements[chosen, j] = teacher.predict(values)
    result = {"epoch": epoch, "protocol": "read-only development diagnosis, not a blind test",
              "groups": {}, "one_replacement_good": {}, "parameters_good": {}}
    groups = {"all": np.ones(len(rows), dtype=np.bool_), "Good": good,
              "paired_Good": good & paired,
              "old_Good": good & np.asarray([r["origin"] == "old" for r in records], dtype=np.bool_),
              "new_Good": good & np.asarray([r["origin"] == "new" for r in records], dtype=np.bool_),
              "missing_Good": good & np.asarray([r["missing_frames"] > 0 for r in records], dtype=np.bool_),
              "complete_Good": good & np.asarray([r["missing_frames"] == 0 for r in records], dtype=np.bool_)}
    for name, chosen in groups.items():
        if not chosen.any():
            continue
        report = {"raw": metrics(raw[chosen], target[chosen]),
                  "calibrated": metrics(calibrated[chosen], target[chosen])}
        labeled = chosen & paired
        if labeled.any():
            report["oracle_paired_only"] = metrics(oracle[labeled], target[labeled])
        result["groups"][name] = report
    labeled = paired & good
    result["error_decomposition_Good"] = error_decomposition(contributions[labeled], oracle[labeled] - target[labeled])
    for j, spec in enumerate(EXPERTS):
        result["one_replacement_good"][spec.name] = metrics(replacements[labeled, j], target[labeled])
        result["parameters_good"][spec.name] = {
            **metrics(predicted[labeled, j], true[labeled, j]),
            "target_std": float(true[labeled, j].std()),
            "predicted_std": float(predicted[labeled, j].std()),
            "target_mean": float(true[labeled, j].mean()),
            "predicted_mean": float(predicted[labeled, j].mean()),
        }
        values = predicted[labeled, j]
        if spec.activation == "sigmoid":
            result["parameters_good"][spec.name]["boundary_rate"] = float(np.mean(
                (values < 0.01 * EYE_SIZE_MAX) | (values > 0.99 * EYE_SIZE_MAX)))
        if spec.activation == "softplus":
            result["parameters_good"][spec.name]["near_zero_rate"] = float(np.mean(values < 0.01))
    chosen = np.flatnonzero(good)
    order = chosen[np.argsort(np.abs(calibrated[chosen] - target[chosen]))[::-1]][:10]
    result["worst_good"] = [{"record": records[i]["record"], "storm_id": records[i]["storm_id"],
                             "origin": records[i]["origin"], "target": float(target[i]),
                             "prediction": float(calibrated[i]),
                             "error": float(calibrated[i] - target[i])} for i in order]
    epoch_times, first_steps, steady_times = [], [], []
    for log in run.glob("outer*/*/e*.jsonl"):
        try:
            with log.open() as stream:
                entries = [json.loads(line) for line in stream if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read training log {log}: {error}") from error
        for entry in entries:
            if entry["epoch"] > epoch:
                continue
            epoch_times.append(entry["seconds"])
            if entry["epoch"] == 1:
                first_steps.append(entry["first_step_seconds"])
            else:
                steady_times.append(entry["seconds"])
    result["timing"] = {"training_epoch_seconds_sum": float(sum(epoch_times)),
                        "first_steps_including_compile_seconds_sum": float(sum(first_steps)),
                        "steady_epoch_seconds_median": float(np.median(steady_times)) if steady_times else None}
    write_json(run / f"diagnosis_epoch{epoch:03d}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path("/mnt/data/H9dP/runs/m1_adapt/data_v1"))
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run_dir, args.epoch, args.data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
