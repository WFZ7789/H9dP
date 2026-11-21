"""M1 adaptive short runs on all available years, without a permanent holdout.

Use prepare once, then probe --epochs N repeatedly with an increasing N and
one run directory. Each outer model evaluates storms absent from both its
experts and calibrator. Inner OOF predictions train the calibrator.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from h9dp.common import load_config
from h9dp.engine import compile_model, initialize_device, seed_everything
from h9dp.m1 import EXPERTS, M5, metrics
from h9dp.m1_adapt import (
    CalibratedM5,
    grouped_folds,
    learning_rate,
    normalization,
    quality_weights,
)
from scripts.audit_motis_inputs import record_key, unique_lookup
from scripts.m1 import (
    MOTIS_COLUMNS,
    M1Dataset,
    build_expert,
    move,
    read_json,
    record_arrays,
    resolve,
    target_statistics,
    write_json,
)


def prepare(config: dict) -> dict:
    data = config["data"]
    work = resolve(config, data["work_dir"])
    if (work / "all_meta.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {work}")
    cache = resolve(config, data["cache_dir"])
    new_root = resolve(config, data["new_cache_dir"])
    source_records = read_json(resolve(config, data["records"]))["records"]
    with resolve(config, data["motis_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    lookup = unique_lookup(source_rows)
    old = cast(list[dict], read_json(cache / "train.json")) + cast(list[dict], read_json(cache / "validation.json"))
    new = read_json(new_root / "index.json")
    if new["bands"] != [8, 9, 10, 16] or new["crop_size"] != 1000:
        raise ValueError("New data bands/crop do not match the declared adapter")
    entries = [("old", item) for item in old] + [("new", item) for item in new["records"]]
    entries.sort(key=lambda pair: pair[1]["record"])
    ids = [item["record"] for _, item in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record IDs across source caches")
    work.mkdir(parents=True, exist_ok=True)
    # 24 int16 images + six boolean masks per 500x500 record, plus metadata.
    estimated = len(entries) * 6 * 500 * 500 * 9 + 10_000_000
    root_used = shutil.disk_usage("/").used
    if root_used >= 375_000_000_000:
        raise RuntimeError(f"Root filesystem already exceeds the limit: {root_used}")
    if work.stat().st_dev == Path("/").stat().st_dev and root_used + estimated >= 375_000_000_000:
        raise RuntimeError("Projected preparation would exceed the root storage limit")
    if shutil.disk_usage(work).free < estimated:
        raise RuntimeError("Insufficient space for derived M1 arrays")
    count = len(entries)
    images = np.lib.format.open_memmap(work / "all_images.npy", mode="w+", dtype=np.int16,
                                      shape=(count, 6, 4, 500, 500))
    masks = np.lib.format.open_memmap(work / "all_masks.npy", mode="w+", dtype=np.bool_,
                                     shape=(count, 6, 1, 500, 500))
    numeric = np.lib.format.open_memmap(work / "all_numeric.npy", mode="w+", dtype=np.float32,
                                       shape=(count, 6, 4, 3, 3))
    coverage = np.lib.format.open_memmap(work / "all_coverage.npy", mode="w+", dtype=np.int32,
                                        shape=(count, 6, 3))
    old_images = np.load(cache / "brightness.npy", mmap_mode="r")
    old_masks = np.load(cache / "validity.npy", mmap_mode="r")
    records = []
    for position, (origin, sample) in enumerate(entries):
        record = source_records[sample["record"]]
        if origin == "old":
            frames = np.asarray(sample["frames"])
            image = np.asarray(old_images[frames], dtype=np.float32)
            mask = np.unpackbits(old_masks[frames], axis=-1, count=250000, bitorder="little")
            mask = mask.reshape(6, 4, 500, 500).all(axis=1, keepdims=True)
            centers = sample["centers"]
            offsets = sample["time_offsets_minutes"]
            time_basis = "actual_observation_midpoint"
        else:
            stem = f"{sample['position']:04d}.npy"
            image = np.array(np.load(new_root / "brightness" / stem, mmap_mode="r")[:, :, 250:750, 250:750])
            packed = np.load(new_root / "validity" / stem, mmap_mode="r")
            valid = np.unpackbits(packed, axis=-1, count=1000000, bitorder="little")
            mask = valid.reshape(6, 4, 1000, 1000)[:, :, 250:750, 250:750].all(axis=1, keepdims=True)
            centers = [[float(value) - 250.0 for value in sample["center"]]] * 6
            target_time = datetime.fromisoformat(sample["target_time"])
            offsets = [(datetime.fromisoformat(frame["nominal"]) - target_time).total_seconds() / 60
                       for frame in sample["frames"]]
            time_basis = "nominal_scan_start; fixed_target_center_in_all_frames"
        image = np.where(mask, image, 250.0)
        if not np.isfinite(image).all() or not mask.any():
            raise ValueError(f"Invalid record {sample['record']}")
        quantized, raw, area, stats = record_arrays(
            {"centers": centers}, image, mask, tuple(config["model"]["radii"]), allow_missing=True,
        )
        images[position], masks[position] = quantized, mask
        numeric[position], coverage[position] = raw, area
        matched = lookup.get(record_key(record))
        params = None if matched is None else [float(source_rows[matched][key]) for key in MOTIS_COLUMNS]
        year = int(record["time"]["year"])
        records.append({
            "record": int(sample["record"]), "name": record["name"], "year": year,
            "storm_id": f"{year}:{record['name']}", "quality": record["quality"],
            "dP": float(record["dP"]), "params": params, "times": offsets, "centers": centers,
            "stats": stats.tolist(), "origin": origin, "time_basis": time_basis,
            "missing_frames": int((area.sum(axis=1) == 0).sum()),
        })
        if position % 25 == 0 or position + 1 == count:
            print(json.dumps({"prepare": position + 1, "total": count}), flush=True)
    for array in (images, masks, numeric, coverage):
        array.flush()
    payload = {
        "schema": "m1-adapt-data/v1", "records": records, "config": config,
        "protocol": "all available old/new records; no permanent holdout; new center 500 crop",
        "counts": {
            "all": count, "paired": sum(r["params"] is not None for r in records),
            "quality": dict(Counter(r["quality"] for r in records)),
            "missing_frames": sum(r["missing_frames"] for r in records),
        },
    }
    write_json(work / "all_meta.json", payload)
    return payload["counts"]


def padded_batch(batch: dict, device: torch.device, size: int) -> tuple[dict, int]:
    """Fixed compiled batch shape; padded examples have zero loss weight."""
    batch = move(batch, device)
    count = len(batch["images"])
    if count < size:
        for key, value in batch.items():
            repeats = (size - count,) + (1,) * (value.ndim - 1)
            batch[key] = torch.cat([value, value[:1].repeat(repeats)], dim=0)
    return batch, count


def loader(dataset, config: dict, generator=None, shuffle=False):
    return DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=shuffle,
                      generator=generator, num_workers=int(config["training"]["num_workers"]),
                      pin_memory=False)


def train_one(spec, indices, predict_indices, config, meta, directory, epochs, seed, device):
    work = resolve(config, config["data"]["work_dir"])
    paired = [i for i in indices if meta["records"][i]["params"] is not None]
    if len(paired) < 8:
        raise ValueError("Too few paired records in a training fold")
    mean, std = normalization(meta["records"], indices)
    column = [item.parameter for item in EXPERTS].index(spec.parameter)
    tm, ts = target_statistics(meta, paired, column)
    seed_everything(seed)  # Must precede construction, including initial weights.
    model = build_expert(spec, config, mean, std, tm, ts, device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                 weight_decay=float(training["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"{spec.name}.pt"
    start = 1
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["indices"] != indices or state["seed"] != seed:
            raise ValueError("Resume fold/seed mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["epoch"] + 1
        if start > epochs + 1:
            raise ValueError("Cannot rewind an existing run to an earlier epoch")
    train_loader = loader(M1Dataset(work, "all", paired), config, generator, True)
    eval_loader = loader(M1Dataset(work, "all", predict_indices), config)
    weights = torch.tensor(quality_weights(meta["records"], training["quality_weights"]),
                           dtype=torch.float32, device=device)
    compiled = compile_model(model, config)  # No fallback if compilation fails.
    batch_size = int(training["batch_size"])
    first_step = None
    log = directory / f"{spec.name}.jsonl"
    for epoch in range(start, epochs + 1):
        model.train()
        lr = learning_rate(epoch, float(training["learning_rate"]), training["lr_milestones"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        total, seen = 0.0, 0
        torch.cuda.synchronize(device)
        before = time.perf_counter()
        for batch in train_loader:
            batch, count = padded_batch(batch, device, batch_size)
            cold_start = time.perf_counter() if first_step is None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = compiled(batch["images"], batch["mask"], batch["numeric"],
                                  batch["coverage"], batch["times"], batch["centers"])
                error = (output[:count, 1].float() - batch["params"][:count, column]) / ts
                sample_weights = weights[batch["record"][:count]]
                loss = (error.square() * sample_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            if cold_start is not None:
                torch.cuda.synchronize(device)
                first_step = time.perf_counter() - cold_start
            total += float(loss.detach()) * count
            seen += count
        torch.cuda.synchronize(device)
        entry = {"epoch": epoch, "weighted_normalized_mse": total / seen, "lr_used": lr,
                 "seconds": time.perf_counter() - before, "first_step_seconds": first_step,
                 "paired_train": len(paired), "seed": seed}
        with log.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "generator": generator.get_state(), "indices": indices, "seed": seed}
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        print(json.dumps({"model": str(directory.relative_to(directory.parents[1])) + "/" + spec.name,
                          **entry}), flush=True)
    model.eval()
    predictions = []
    # Eager evaluation is declared, as in historical M1; training is compiled.
    with torch.no_grad():
        for batch in eval_loader:
            batch, count = padded_batch(batch, device, batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["images"], batch["mask"], batch["numeric"],
                               batch["coverage"], batch["times"], batch["centers"])
            predictions.extend(output[:count, 1].float().cpu().tolist())
    return np.asarray(predictions)


def train_pack(indices, predict_indices, config, meta, directory, epochs, seed, device):
    return np.column_stack([
        train_one(spec, indices, predict_indices, config, meta, directory, epochs, seed, device)
        for spec in EXPERTS
    ])


def summarize(predictions, target, records):
    result = {"all": metrics(predictions, target)}
    for quality in ("Good", "Fair", "Poor"):
        chosen = np.asarray([r["quality"] == quality for r in records])
        if chosen.any():
            result[quality] = metrics(predictions[chosen], target[chosen])
    return result


def probe(config: dict, directory: Path, epochs: int) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    cache = resolve(config, config["data"]["work_dir"])
    meta = read_json(cache / "all_meta.json")
    records = meta["records"]
    identity = hashlib.sha256(json.dumps({"config": config, "meta": meta}, sort_keys=True).encode()).hexdigest()
    manifest = directory / "run.json"
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("Config/data changed: use a new run directory")
    device, environment = initialize_device(config)
    seed = int(config["training"]["seed"])
    folds = grouped_folds(records, int(config["crossfit"]["outer_folds"]), seed)
    write_json(manifest, {"identity": identity, "config": config, "environment": environment,
                          "status": "running", "target_epoch": epochs, "folds": folds.tolist(),
                          "no_permanent_holdout": True, "evaluation": "eager BF16; training compiled BF16"})
    target = np.asarray([r["dP"] for r in records])
    raw, calibrated = np.full(len(records), np.nan), np.full(len(records), np.nan)
    expert_predictions = np.full((len(records), 5), np.nan)
    details = []
    start = time.perf_counter()
    for outer in range(int(config["crossfit"]["outer_folds"])):
        training = np.flatnonzero(folds != outer).tolist()
        evaluation = np.flatnonzero(folds == outer).tolist()
        subset = [records[i] for i in training]
        inner_folds = grouped_folds(subset, int(config["crossfit"]["inner_folds"]), seed)
        oof = np.full((len(training), 5), np.nan)
        for inner in range(int(config["crossfit"]["inner_folds"])):
            inner_train = [training[i] for i in np.flatnonzero(inner_folds != inner)]
            local_eval = np.flatnonzero(inner_folds == inner)
            inner_eval = [training[i] for i in local_eval]
            oof[local_eval] = train_pack(inner_train, inner_eval, config, meta,
                                         directory / f"outer{outer}" / f"inner{inner}", epochs,
                                         seed, device)
        predicted = train_pack(training, evaluation, config, meta, directory / f"outer{outer}" / "base",
                               epochs, seed, device)
        paired = [i for i in training if records[i]["params"] is not None]
        teacher = M5.fit(np.asarray([records[i]["params"] for i in paired]), target[paired])
        calibration = CalibratedM5.fit(oof, target[training],
                                       quality_weights(subset, config["training"]["quality_weights"]),
                                       teacher, float(config["crossfit"]["ridge_alpha"]))
        raw[evaluation] = teacher.predict(predicted)
        calibrated[evaluation] = calibration.predict(predicted)
        expert_predictions[evaluation] = predicted
        detail = {"outer": outer, "training": len(training), "evaluation": len(evaluation),
                  "teacher_coefficients": teacher.coefficients.tolist(),
                  "calibrated_coefficients": calibration.coefficients.tolist(),
                  "training_records": [records[i]["record"] for i in training],
                  "evaluation_records": [records[i]["record"] for i in evaluation]}
        write_json(directory / f"outer{outer}" / f"calibration_epoch{epochs:03d}.json", detail)
        details.append(detail)
    if not np.isfinite(expert_predictions).all() or not np.isfinite(calibrated).all():
        raise RuntimeError("Incomplete OOF coverage")
    result = {
        "status": "completed short adaptive development round; NOT blind-test performance",
        "epochs": epochs, "records": meta["counts"], "elapsed_seconds": time.perf_counter() - start,
        "protocol": "nested storm-group crossfit; every record rotates through training and evaluation",
        "raw_m5": summarize(raw, target, records), "calibrated_m5": summarize(calibrated, target, records),
        "experts": {}, "folds": details,
    }
    for j, spec in enumerate(EXPERTS):
        chosen = [i for i, r in enumerate(records) if r["params"] is not None]
        truth = np.asarray([records[i]["params"][j] for i in chosen])
        report = summarize(expert_predictions[chosen, j], truth, [records[i] for i in chosen])
        for quality, group in report.items():
            selected = np.asarray([quality == "all" or records[i]["quality"] == quality for i in chosen], dtype=np.bool_)
            group["target_std"] = float(truth[selected].std())
            group["predicted_std"] = float(expert_predictions[np.asarray(chosen)[selected], j].std())
        result["experts"][spec.name] = report
    output = directory / f"predictions_epoch{epochs:03d}.csv"
    with output.open("w", newline="") as stream:
        fields = ["record", "storm_id", "quality", "origin", "fold", "dP", "raw_m5", "calibrated_m5"]
        fields += [f"pred_{spec.parameter}" for spec in EXPERTS]
        fields += [f"real_{spec.parameter}" for spec in EXPERTS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            row = {key: record[key] for key in ("record", "storm_id", "quality", "origin", "dP")}
            row.update(fold=int(folds[i]), raw_m5=float(raw[i]), calibrated_m5=float(calibrated[i]))
            for j, spec in enumerate(EXPERTS):
                row[f"pred_{spec.parameter}"] = float(expert_predictions[i, j])
                row[f"real_{spec.parameter}"] = "" if record["params"] is None else record["params"][j]
            writer.writerow(row)
    write_json(directory / f"summary_epoch{epochs:03d}.json", result)
    run = read_json(manifest)
    run.update(status="paused_for_diagnosis", completed_epoch=epochs)
    write_json(manifest, run)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "probe"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1_adapt.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    compiler_cache = resolve(config, config["compiler_cache"])
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(compiler_cache / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(compiler_cache / "triton"))
    if args.command == "prepare":
        print(json.dumps(prepare(config)), flush=True)
    else:
        if args.run_dir is None:
            parser.error("probe requires an external --run-dir")
        result = probe(config, args.run_dir.resolve(), args.epochs)
        print(json.dumps({key: result[key] for key in ("epochs", "raw_m5", "calibrated_m5")}), flush=True)


if __name__ == "__main__":
    main()
