"""Prepare one fixed train/validation split and a native 500-pixel cache."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import check_space, fingerprint, write_json
from .hsd import BANDS, brightness_temperature, crop_window, earth_mask, lonlat_to_pixel, read_counts, read_header

SCHEMA = "h9dp-native500-cache/v2"
LEGACY_SCHEMA = "h9dp-native500-cache/v1"


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def resolve_source(root: Path, value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_file():
        return path.resolve()
    if path.suffix == ".bz2" and path.with_suffix("").is_file():
        return path.with_suffix("").resolve()
    raise FileNotFoundError(path)


def split_specification(config: dict) -> dict:
    data = config["data"]
    return {
        "unit": "record", "weighting": "uniform_without_replacement",
        "first_training_year": data["first_training_year"],
        "validation_year": data["validation_year"],
        "validation_fraction": data["validation_fraction"],
        "split_seed": data["split_seed"],
    }


def select_validation_records(records: list[tuple[int, int]], config: dict) -> set[int]:
    """Only record IDs and years participate; cyclone identity never participates."""
    data = config["data"]
    if len({record for record, _ in records}) != len(records):
        raise ValueError("Duplicate record IDs are not separate records")
    if not 0 < data["validation_fraction"] < 1:
        raise ValueError("Validation fraction must be strictly between zero and one")
    if any(not data["first_training_year"] <= year <= data["validation_year"] for _, year in records):
        raise ValueError("Record outside the configured year range")
    eligible = sorted(record for record, year in records if year == data["validation_year"])
    if len(eligible) < 2:
        raise ValueError("At least two validation-year records are required")
    count = min(len(eligible) - 1, math.ceil(len(eligible) * data["validation_fraction"]))
    generator = np.random.default_rng(data["split_seed"])
    return set(map(int, generator.choice(eligible, size=count, replace=False)))


def split_for(record: int, year: int, config: dict, validation_records: set[int]) -> str:
    if not config["data"]["first_training_year"] <= year <= config["data"]["validation_year"]:
        raise ValueError("Record outside the configured year range")
    return "validation" if record in validation_records else "train"


def _source_snapshot(config: dict) -> dict:
    return {
        name: hashlib.sha256(Path(config["data"][name]).read_bytes()).hexdigest()
        for name in ("manifest", "records", "labels")
    }


def build_index(config: dict) -> dict:
    """Header-only pass: resolve labels, native crops, timing and record IDs."""
    root = Path(config["repository_root"])
    if config["data"]["crop_size"] != 500:
        raise ValueError("The selected model/data contract requires 500×500 inputs")
    with open(config["data"]["records"], newline="") as stream:
        records = list(csv.DictReader(stream))
    validation_records = select_validation_records(
        [(int(row["record"]), parse_time(row["tp_utc"]).year) for row in records], config
    )
    with open(config["data"]["manifest"], newline="") as stream:
        manifest = list(csv.DictReader(stream))
    labels = json.loads(Path(config["data"]["labels"]).read_text())["records"]
    by_record = defaultdict(list)
    for row in manifest:
        if row["status"] != "OK":
            raise ValueError(f"Non-OK observation: record {row['record']}")
        by_record[int(row["record"])].append(row)
    headers = {}
    file_stats = {}
    frames = []
    frame_lookup = {}
    samples = []
    seen_records = set()
    for record in records:
        index = int(record["record"])
        if index in seen_records or record["status"] != "OK":
            raise ValueError(f"Duplicate or invalid record: {index}")
        seen_records.add(index)
        if not 0 <= index < len(labels):
            raise ValueError(f"Record index outside zero-based label array: {index}")
        label = labels[index]
        target_time = parse_time(record["tp_utc"])
        if label["name"] != record["name"] or parse_time(label["time"]["iso_utc"]) != target_time:
            raise ValueError(f"MOTIS name/time mismatch for zero-based record {index}")
        latitude, longitude = float(record["lat"]), float(record["lon"])
        if max(abs(latitude - label["location"]["lat"]), abs(longitude - label["location"]["lon"])) > 1e-5:
            raise ValueError(f"MOTIS coordinate mismatch for record {index}")
        quality = label["quality"]
        target = float(label["dP"])
        if not np.isfinite(target) or target < 0:
            raise ValueError(f"Invalid dP/quality for record {index}")
        split = split_for(index, target_time.year, config, validation_records)
        observations = sorted(by_record[index], key=lambda row: parse_time(row["time"]))
        if len(observations) != 6:
            raise ValueError(f"Record {index} has {len(observations)} frames instead of 6")
        sample_frames, offsets, centers, sources = [], [], [], []
        for observation in observations:
            if observation["name"] != record["name"] or parse_time(observation["tp"]) != target_time:
                raise ValueError(f"Observation-to-record mismatch: {index}")
            paths = [resolve_source(root, value) for value in observation["files"].split(";")]
            if len(paths) != 4:
                raise ValueError(f"Record {index}: incomplete four-band frame")
            for path in paths:
                key = str(path)
                if key not in headers:
                    headers[key] = read_header(path)
                    stat = path.stat()
                    file_stats[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            paths.sort(key=lambda path: headers[str(path)].band)
            hs = [headers[str(path)] for path in paths]
            if tuple(header.band for header in hs) != BANDS:
                raise ValueError(f"Record {index}: incorrect or duplicate bands")
            first = hs[0]
            geometry = lambda header: (header.height, header.width, header.projection, header.segment_first_line)
            if any(geometry(header) != geometry(first) or header.satellite != first.satellite for header in hs):
                raise ValueError(f"Record {index}: four-band geometry/satellite mismatch")
            if max(abs((header.observation_time - first.observation_time).total_seconds()) for header in hs) > 60:
                raise ValueError(f"Record {index}: band observation times differ by more than one minute")
            if observation["source"] == "R3" and (first.height, first.width) != (500, 500):
                raise ValueError("Unexpected R3 source shape")
            if observation["source"] == "FLDK" and (first.height, first.width) != (550, 5500):
                raise ValueError("Unexpected FLDK source shape")
            if observation["source"] not in ("R3", "FLDK"):
                raise ValueError("Unsupported source area")
            window = crop_window(first, latitude, longitude, 500)
            col, row = lonlat_to_pixel(first, latitude, longitude)
            seconds = np.mean([(header.observation_time - target_time).total_seconds() for header in hs])
            offset = float(seconds / 60)
            nominal = (parse_time(observation["time"]) - parse_time(observation["T"])).total_seconds() / 60
            if abs(offset - nominal) > 10:
                raise ValueError(f"Observation outside ten-minute pairing tolerance: record {index}")
            frame_key = (tuple(map(str, paths)), window)
            if frame_key not in frame_lookup:
                frame_lookup[frame_key] = len(frames)
                frames.append({
                    "files": list(map(str, paths)),
                    "window": list(window),
                    "source": observation["source"],
                    "satellite": first.satellite,
                    "source_shape": [first.height, first.width],
                    "observation_time": first.observation_time.isoformat(),
                    "quality_flags": [list(header.quality_flags) for header in hs],
                })
            sample_frames.append(frame_lookup[frame_key])
            offsets.append(offset)
            centers.append([col - window[0], row - window[1]])
            sources.append(observation["source"])
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise ValueError(f"Non-increasing observation sequence for record {index}")
        samples.append({
            "record": index, "name": record["name"], "year": target_time.year,
            "storm_id": f"{target_time.year}:{record['name']}", "target_time": target_time.isoformat(),
            "split": split, "dP": target, "quality": quality,
            "frames": sample_frames, "time_offsets_minutes": offsets, "centers": centers,
            "sources": sources,
        })
    if seen_records != set(by_record):
        raise ValueError("Manifest and record table have different record sets")
    return {
        "schema": SCHEMA, "bands": list(BANDS), "crop_size": 500,
        "source_snapshot": _source_snapshot(config), "source_files": file_stats,
        "split_specification": split_specification(config),
        "frames": frames, "samples": samples,
    }


def index_fingerprint(index: dict) -> str:
    keys = ["schema", "source_snapshot", "source_files", "split_specification", "frames", "samples", "normalization"]
    if index["schema"] == SCHEMA:
        keys.append("target_normalization")
    return fingerprint({key: index[key] for key in keys})


def validate_cache_files(directory: Path, index: dict, config: dict) -> None:
    if index.get("schema") not in (SCHEMA, LEGACY_SCHEMA) or index.get("status") != "complete":
        raise ValueError("Cache is incomplete or has an incompatible schema")
    if index["source_snapshot"] != _source_snapshot(config):
        raise ValueError("Source manifest/records/labels changed: prepare a new cache")
    if index["fingerprint"] != index_fingerprint(index):
        raise ValueError("Cache metadata fingerprint mismatch")
    for name, expected in index["source_files"].items():
        stat = Path(name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Raw source changed since preparation: {name}")
    for name, expected in index["arrays"].items():
        array = np.load(directory / name, mmap_mode="r")
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError(f"Cache array shape/dtype mismatch: {name}")
        if (directory / name).stat().st_size != expected["file_bytes"]:
            raise ValueError(f"Cache array size mismatch: {name}")


def load_cache(config: dict) -> dict:
    directory = Path(config["data"]["cache_dir"])
    index = json.loads((directory / "index.json").read_text())
    validate_cache_files(directory, index, config)
    if index["schema"] != SCHEMA or index["split_specification"] != split_specification(config):
        raise ValueError("Prepared cache does not match the record-level split configuration")
    records = [(sample["record"], sample["year"]) for sample in index["samples"]]
    validation = select_validation_records(records, config)
    if any(sample["split"] != split_for(sample["record"], sample["year"], config, validation) for sample in index["samples"]):
        raise ValueError("Cached record split does not match the fixed record selection")
    return index


def update_training_statistics(directory: Path, index: dict) -> None:
    """Each record and each of its six frames has equal normalization weight."""
    training = [sample for sample in index["samples"] if sample["split"] == "train"]
    counts = Counter(frame for sample in training for frame in sample["frames"])
    images = np.load(directory / "brightness.npy", mmap_mode="r")
    packed = np.load(directory / "validity.npy", mmap_mode="r")
    means, seconds = np.zeros(4), np.zeros(4)
    for position, (frame, weight) in enumerate(sorted(counts.items())):
        masks = np.unpackbits(packed[frame], axis=-1, count=500 * 500, bitorder="little").reshape(4, 500, 500).astype(bool)
        for channel in range(4):
            values = images[frame, channel][masks[channel]].astype(np.float64)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid training brightness values in frame {frame}")
            means[channel] += weight * values.mean()
            seconds[channel] += weight * np.mean(values * values)
        if (position + 1) % 400 == 0:
            print(f"Training-record normalization: {position+1}/{len(counts)} unique frames", flush=True)
    occurrences = sum(counts.values())
    mean = means / occurrences
    variance = seconds / occurrences - mean * mean
    if np.any(variance <= 0):
        raise ValueError("Invalid training-only normalization variance")
    index["normalization"] = {
        "mean_kelvin": mean.tolist(), "std_kelvin": np.sqrt(variance).tolist(),
        "unique_training_frames": len(counts), "training_frame_occurrences": occurrences,
        "weighting": "equal_record_equal_frame",
    }
    targets = np.asarray([sample["dP"] for sample in training], dtype=np.float64)
    index["target_normalization"] = {
        "mean_hpa": float(targets.mean()), "std_hpa": max(1.0, float(targets.std())),
        "weighting": "equal_record", "count": len(targets),
    }
    index.pop("training_target_median", None)
    index["summary"]["split_counts"] = dict(Counter(sample["split"] for sample in index["samples"]))
    train_frames = set(counts)
    validation_frames = {frame for sample in index["samples"] if sample["split"] == "validation" for frame in sample["frames"]}
    index["summary"]["shared_frames_between_distinct_records"] = len(train_frames & validation_frames)


def publish_index(staging: Path, directory: Path, index: dict) -> dict:
    index["fingerprint"] = index_fingerprint(index)
    for split in ("train", "validation"):
        write_json(staging / f"{split}.json", [sample for sample in index["samples"] if sample["split"] == split])
    index["status"] = "complete"
    write_json(staging / "index.json", index)
    os.rename(staging, directory)
    print(json.dumps({"cache": str(directory), "fingerprint": index["fingerprint"], **index["summary"]}, ensure_ascii=False), flush=True)
    return index


def reuse_prepared_arrays(config: dict, source: Path, directory: Path) -> dict:
    started = time.monotonic()
    original = json.loads((source / "index.json").read_text())
    validate_cache_files(source, original, config)
    index = copy.deepcopy(original)
    index["schema"] = SCHEMA
    index["split_specification"] = split_specification(config)
    selected = select_validation_records([(s["record"], s["year"]) for s in index["samples"]], config)
    for sample in index["samples"]:
        sample["split"] = split_for(sample["record"], sample["year"], config, selected)
        sample.pop("sigma", None)
    index["storage_check"] = check_space(directory, 20_000_000)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + f".partial-{os.getpid()}-{time.time_ns()}")
    staging.mkdir()
    index["status"] = "preparing"
    index["reused_arrays"] = {"source": str(source), "source_fingerprint": original["fingerprint"], "method": "hardlink_read_only_arrays"}
    write_json(staging / "index.json", index)
    for name in index["arrays"]:
        os.link(source / name, staging / name)
    update_training_statistics(staging, index)
    index["summary"]["source_prepare_elapsed_seconds"] = original["summary"].get("elapsed_seconds")
    index["summary"]["elapsed_seconds"] = time.monotonic() - started
    return publish_index(staging, directory, index)


def prepare(config: dict) -> dict:
    directory = Path(config["data"]["cache_dir"])
    if directory.exists():
        result = load_cache(config)
        print(f"Existing complete cache: {directory}", flush=True)
        return result
    reuse = config["data"].get("reuse_cache")
    if reuse and Path(reuse).is_dir():
        return reuse_prepared_arrays(config, Path(reuse), directory)
    print("Checking source headers, label joins and the fixed split...", flush=True)
    index = build_index(config)
    number = len(index["frames"])
    projected = number * (4 * 500 * 500 * 4 + 4 * 31250) + 20_000_000
    index["storage_check"] = check_space(directory, projected)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + f".partial-{os.getpid()}-{time.time_ns()}")
    staging.mkdir()
    index["status"] = "preparing"
    write_json(staging / "index.json", index)
    images = np.lib.format.open_memmap(staging / "brightness.npy", mode="w+", dtype="float32", shape=(number, 4, 500, 500))
    packed = np.lib.format.open_memmap(staging / "validity.npy", mode="w+", dtype="uint8", shape=(number, 4, 31250))
    invalid_fractions = []
    start = time.monotonic()
    for frame_id, frame in enumerate(index["frames"]):
        x0, y0 = frame["window"]
        frame_masks = []
        on_earth = None
        for channel, path in enumerate(frame["files"]):
            header, counts, checksum = read_counts(path)
            stat = Path(path).stat()
            source = index["source_files"][path]
            if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(f"Source modified during preparation: {path}")
            if "sha256" in source and source["sha256"] != checksum:
                raise ValueError(f"Source content changed during preparation: {path}")
            source["sha256"] = checksum
            if on_earth is None:
                on_earth = earth_mask(header, (x0, y0), 500)
            temperature, valid = brightness_temperature(counts[y0:y0+500, x0:x0+500], header, on_earth)
            if not valid.any():
                raise ValueError(f"No valid earth pixels in frame {frame_id}, band {header.band}")
            images[frame_id, channel] = temperature
            packed[frame_id, channel] = np.packbits(valid.reshape(-1), bitorder="little")
            frame_masks.append(valid)
        if not np.logical_and.reduce(frame_masks).any():
            raise ValueError(f"No shared four-band validity in frame {frame_id}")
        invalid_fractions.append(float(1 - np.logical_and.reduce(frame_masks).mean()))
        if (frame_id + 1) % 50 == 0 or frame_id + 1 == number:
            check_space(staging)
            print(f"Prepared {frame_id+1}/{number} unique native500 frames ({time.monotonic()-start:.1f}s)", flush=True)
    index["summary"] = {
        "samples": len(index["samples"]), "unique_frames": number,
        "split_counts": dict(Counter(sample["split"] for sample in index["samples"])),
        "quality_counts": dict(Counter(sample["quality"] for sample in index["samples"])),
        "maximum_invalid_fraction": max(invalid_fractions),
        "elapsed_seconds": time.monotonic() - start,
    }
    index["arrays"] = {}
    for filename, array in (("brightness.npy", images), ("validity.npy", packed)):
        array.flush()
        index["arrays"][filename] = {"shape": list(array.shape), "dtype": str(array.dtype), "file_bytes": (staging / filename).stat().st_size}
    del images, packed
    update_training_statistics(staging, index)
    return publish_index(staging, directory, index)
