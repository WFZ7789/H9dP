"""Audit MOTIS intermediate labels without loading images or training a network."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from h9dp.pressure_labels import read_catalog, verify_catalog_labels

COLUMNS = (
    "B24 dT", "B36 dT", "Upper dif", "Mid dif", "B27 dT", "Eye size",
    "B31-B32", "B33 dT tr.", "Aqua Bias", "Terra Bias",
)
MINIMAL_COLUMNS = (0, 1, 5, 7)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON input {path}: {error}") from error


def observation_key(name, year, date, time, satellite):
    """Normalize formatting only; never fuzzy-match names, dates, or times."""
    year, date, time = int(year), int(date), int(time)
    datetime(year, date // 100, date % 100, time // 100, time % 100, tzinfo=UTC)
    return str(name).strip().casefold(), year, date, time, str(satellite).strip().upper()


def csv_key(row):
    return observation_key(*(row[k] for k in ("Name", "Year", "Date", "Time", "Satellite")))


def record_key(row):
    time = row["time"]
    return observation_key(
        row["name"], time["year"], time["date_mmdd"], time["time_utc_hhmm"], row["satellite"],
    )


def unique_lookup(rows):
    lookup = {}
    for position, row in enumerate(rows):
        key = csv_key(row)
        if key in lookup:
            raise ValueError(f"Ambiguous duplicate MOTIS key: {key}")
        lookup[key] = position
    return lookup


def read_thermal_rows(path: Path) -> list[dict]:
    """Read only the pass identifiers and intermediate parameters used by M1."""
    required = ("Name", "Year", "Date", "Time", "Satellite", "T CDO", *COLUMNS)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if missing := set(required) - fields:
            raise ValueError(f"MOTIS table {path} lacks required columns: {sorted(missing)}")
        columns = (*required, "npix") if "npix" in fields else required
        return [{column: row[column] for column in columns} for row in reader]


def design_matrix(rows):
    values = np.array([[float(row[column]) for column in COLUMNS] for row in rows])
    temperature = np.array([float(row["T CDO"]) for row in rows])
    if not np.isfinite(values).all() or not np.isfinite(temperature).all():
        raise ValueError("MOTIS predictors must be finite")
    scale = (35.0 - np.minimum(temperature, -50.0)) / 100.0
    return scale[:, None] * values


def error_metrics(prediction, target):
    errors = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Metrics require nonempty finite inputs")
    return {
        "count": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias_prediction_minus_target": float(np.mean(errors)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }


def oof_predictions(features, target, seed=1, fold_count=5):
    """Record-uniform folds; each fit sees only its own training labels."""
    features, target = np.asarray(features), np.asarray(target)
    if len(features) != len(target) or len(target) < fold_count or fold_count < 2:
        raise ValueError("OOF requires matching inputs and at least one record per fold")
    folds = np.array_split(np.random.default_rng(seed).permutation(len(target)), fold_count)
    prediction = np.empty(len(target), dtype=np.float64)
    fold_ids = np.empty(len(target), dtype=np.int64)
    for fold_id, held_out in enumerate(folds):
        training = np.setdiff1d(np.arange(len(target)), held_out)
        matrix = np.column_stack([np.ones(len(training)), features[training]])
        coefficients = np.linalg.lstsq(matrix, target[training], rcond=None)[0]
        prediction[held_out] = coefficients[0] + features[held_out] @ coefficients[1:]
        fold_ids[held_out] = fold_id
    return prediction, fold_ids


def audit(csv_path, records_path, cache_path, seed=1):
    rows = read_thermal_rows(csv_path)
    lookup = unique_lookup(rows)
    records, catalog_digest = read_catalog(records_path)
    features = design_matrix(rows)
    result = {
        "status": "real tabular audit only; no images loaded; no neural training",
        "seed": seed,
        "inputs": {},
        "source_rows": len(rows),
        "source_years": dict(sorted(Counter(int(row["Year"]) for row in rows).items())),
        "catalog_sha256": catalog_digest,
        "splits": {},
    }
    inputs = {"csv": csv_path, "records": records_path}
    for split in ("train", "validation"):
        inputs[split] = cache_path / f"{split}.json"
    for name, path in inputs.items():
        result["inputs"][name] = {
            "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    missing_raw = [i for i, record in enumerate(records) if record_key(record) not in lookup]
    result["raw_records"] = {
        "count": len(records), "matched": len(records) - len(missing_raw),
        "missing_by_year": dict(sorted(Counter(records[i]["time"]["year"] for i in missing_raw).items())),
    }
    for split in ("train", "validation"):
        samples = read_json(inputs[split])
        verify_catalog_labels(samples, records)
        matched, missing = [], []
        for sample in sorted(samples, key=lambda item: item["record"]):
            record_id = sample["record"]
            record = records[record_id]
            position = lookup.get(record_key(record))
            if position is None:
                missing.append({"record": record_id, "key": record_key(record)})
            else:
                row = rows[position]
                matched.append({
                    "record": record_id, "source_row_zero_based": position,
                    "quality": sample["quality"], "current_dP": float(sample["dP"]),
                    "parameters": {key: float(row[key]) for key in (*COLUMNS, "T CDO")},
                })
        summary = {
            "count": len(samples), "matched": len(matched), "missing": missing,
            "years": dict(sorted(Counter(sample["year"] for sample in samples).items())),
            "pairs": matched,
            "non_null_current_field_counts": {
                field: sum(records[sample["record"]].get(field) is not None for sample in samples)
                for field in ("t_cdo", "pixels", "b27", "b31", "poci", "roci")
            },
        }
        for source_field, current_field in (("T CDO", "t_cdo"), ("npix", "pixels")):
            comparisons = [
                (float(rows[pair["source_row_zero_based"]][source_field]),
                 float(records[pair["record"]][current_field]))
                for pair in matched
                if source_field in rows[pair["source_row_zero_based"]]
                and records[pair["record"]].get(current_field) is not None
            ]
            if comparisons:
                old_values, current_values = np.asarray(comparisons).T
                summary[f"source_{source_field}_vs_current_{current_field}"] = error_metrics(
                    old_values, current_values,
                )
        result["splits"][split] = summary
    train_ids = {pair["record"] for pair in result["splits"]["train"]["pairs"]}
    validation_ids = {pair["record"] for pair in result["splits"]["validation"]["pairs"]}
    if train_ids & validation_ids:
        raise ValueError("Matched records overlap between train and validation")
    pairs = result["splits"]["train"]["pairs"]
    if len(pairs) >= 5:
        indices = [pair["source_row_zero_based"] for pair in pairs]
        matrix = features[indices]
        result["oracle_oof"] = {}
        targets = np.array([pair["current_dP"] for pair in pairs])
        for name, columns in (("M5", MINIMAL_COLUMNS), ("M11", tuple(range(10)))):
            prediction, folds = oof_predictions(matrix[:, columns], targets, seed)
            result["oracle_oof"][name] = {
                "current_dP": error_metrics(prediction, targets),
                "predicted_parameter_count": len(columns) + 1,
            }
        result["diagnostic_folds"] = [
            {"record": pair["record"], "fold": int(fold)} for pair, fold in zip(pairs, folds, strict=True)
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Radiometric parameter table")
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/h9dp/native500_v2_seed1"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="Store the report outside Git")
    args = parser.parse_args()
    try:
        result = audit(args.csv, args.records, args.cache, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "source_rows": result["source_rows"],
        "matched": {key: value["matched"] for key, value in result["splits"].items()},
        "oracle_oof": result.get("oracle_oof", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
