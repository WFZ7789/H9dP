"""Check V1 C++/Python parity and conservatively time their local DAT paths."""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from h9dp.hsd import (
    BANDS,
    brightness_temperature,
    crop_window,
    decode_counts,
    earth_mask,
    lonlat_to_pixel,
)
from h9dp.pressure_labels import read_catalog
from h9dp.r1 import record_folds
from h9dp.v1 import sequence
from scripts.export_v1_inference import FOLDS, SEED, pack_features

MAX_ACCURACY_LOSS_HPA = 0.35
MAX_RUNTIME_RATIO = 1.5


def _csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _predict(exe, *args):
    response = subprocess.check_output([exe, *args], text=True)
    return np.array([float(line) for line in response.splitlines()])


def _metrics(pred, rows):
    ref = np.array([float(row["prediction"]) for row in rows])
    target = np.array([float(row["target"]) for row in rows])
    if pred.shape != ref.shape or not len(pred) or not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Expected one finite C++ and Python prediction per OOF record")
    cpp = float(np.sqrt(np.mean((pred - target) ** 2)))
    python = float(np.sqrt(np.mean((ref - target) ** 2)))
    return {
        "n": len(pred), "cpp_rmse_hpa": cpp, "python_rmse_hpa": python, "rmse_loss_hpa": cpp - python,
        "max_abs_difference_hpa": float(np.max(np.abs(pred - ref))),
    }


def _descriptors(scans, lat, lon):
    """Reference V1 DAT preprocessing, excluding training-only archival checksums."""
    crops = []
    for files in scans:
        channels = sorted((decode_counts(path.read_bytes(), path) for path in files), key=lambda item: item[0].band)
        first = channels[0][0]
        x, y = crop_window(first, lat, lon)
        col, row = lonlat_to_pixel(first, lat, lon)
        earth = earth_mask(first, (x, y), 500)
        bt, valid = zip(*(brightness_temperature(counts[y:y + 500, x:x + 500], header, earth)
                          for header, counts in channels), strict=True)
        mask = np.logical_and.reduce(valid)
        image = np.rint(np.where(mask, bt, 250.0) * 100.0 - 15000.0).astype(np.int16)
        time = np.mean([(header.observation_start_mjd + header.observation_end_mjd) * 720
                        for header, _ in channels])
        crops.append((time, image, mask, (col - x, row - y)))
    times, images, masks, centers = zip(*sorted(crops, key=lambda frame: frame[0]), strict=True)
    return sequence(np.stack(images), np.stack(masks), centers, np.array(times))


def _paths(dataset, values):
    out = []
    for value in values.split(";"):
        path = dataset / Path(value).relative_to("HIMAWARI_data")
        if not path.is_file() and path.suffix in (".bz2", ".gz"):
            path = path.with_suffix("")
        if not path.is_file():
            raise FileNotFoundError(path)
        out.append(path)
    if len(out) != len(BANDS):
        raise ValueError("Expected four band paths per scan")
    return out


def check(exe, models, features, formal, dataset, output):
    catalog, _ = read_catalog()
    rows = _csv(formal / "predictions.csv")
    reference = {int(row["record"]): row for row in rows}
    with np.load(features) as f:
        ids = f["record"]
    fold = record_folds(len(ids), FOLDS, SEED)
    if len(reference) != len(rows) or set(reference) != set(ids) or any(
        int(reference[r]["seed"]) != SEED or int(reference[r]["fold"]) != k
        or float(reference[r]["target"]) != catalog[r]["dP"]
        or reference[r]["quality"] != catalog[r]["quality"]
        for r, k in zip(ids, fold, strict=True)
    ):
        raise ValueError("Python OOF predictions do not match the features, root catalog and seed-1 splits")
    with TemporaryDirectory(dir=output.parent) as tmp:
        predictions, held_rows = [], []
        for k in range(FOLDS):
            request = Path(tmp) / f"held{k}.v1q"
            pack_features(features, request, k)
            predictions.extend(_predict(exe, "--features", request, models / f"fold{k}.v1m"))
            held_rows.extend(reference[r] for r in ids[fold == k])
    cached = _metrics(np.array(predictions), held_rows)
    print("Cached OOF", cached, flush=True)

    scans = defaultdict(list)
    for row in _csv(dataset / "manifest.csv"):
        if row["status"] != "OK":
            raise ValueError(f"Non-OK scan for record {row['record']}")
        scans[int(row["record"])].append(_paths(dataset, row["files"]))
    dat_rows = _csv(dataset / "records.csv")
    dat_ids = [int(row["record"]) for row in dat_rows]
    if len(set(dat_ids)) != len(dat_ids) or set(dat_ids) != set(scans):
        raise ValueError("DAT records and manifest differ")
    result_rows = []
    for row, rid in zip(dat_rows, dat_ids, strict=True):
        ref = reference[rid]
        if len(scans[rid]) != 6:
            raise ValueError(f"Record {rid} does not have six usable DAT groups")
        model = models / f"fold{ref['fold']}.v1m"
        start = perf_counter()
        pred, = _predict(exe, row["lat"], row["lon"], model, "--", *(p for scan in scans[rid] for p in scan))
        cpp_s = perf_counter() - start
        start = perf_counter()
        _descriptors(scans[rid], float(row["lat"]), float(row["lon"]))
        py_s = perf_counter() - start
        result_rows.append((rid, int(ref["fold"]), float(ref["target"]), float(ref["prediction"]), pred, cpp_s, py_s))
        if len(result_rows) % 50 == 0:
            print(f"DAT: {len(result_rows)}/{len(dat_ids)}", flush=True)
    values = np.array(result_rows)
    raw = _metrics(values[:, 4], [reference[r] for r in dat_ids])
    cpp, py = values[:, 5], values[:, 6]
    ratios = cpp / py
    timing = {
        "cpp_total_s": float(cpp.sum()), "python_preprocessing_total_s": float(py.sum()),
        "ratio": float(cpp.sum() / py.sum()),
        "cpp_median_ms": float(1000 * np.median(cpp)), "python_preprocessing_median_ms": float(1000 * np.median(py)),
        "p95_record_ratio": float(np.quantile(ratios, 0.95)), "worst_record_ratio": float(ratios.max()),
    }
    result = {
        "seed": SEED, "cached_oof": cached, "dat_oof": raw, "timing": timing,
        "max_accuracy_loss_hpa": MAX_ACCURACY_LOSS_HPA, "max_runtime_ratio": MAX_RUNTIME_RATIO,
        "passed": cached["rmse_loss_hpa"] <= MAX_ACCURACY_LOSS_HPA
        and raw["rmse_loss_hpa"] <= MAX_ACCURACY_LOSS_HPA and timing["ratio"] <= MAX_RUNTIME_RATIO,
    }
    with output.with_suffix(".csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record", "fold", "target", "python_dp", "cpp_dp", "cpp_seconds", "python_preprocess_seconds"])
        writer.writerows(result_rows)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "models", "features", "formal", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    try:
        passed = check(**vars(parser.parse_args()))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
