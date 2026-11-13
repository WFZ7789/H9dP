"""M1 test phase: prepare paired data, train independent experts, evaluate M5.

Commands run from the repository root with the fy3-aivis-v1 environment:

    PYTHONPATH=src:. python -m scripts.m1 prepare --config configs/m1.yaml
    PYTHONPATH=src:. python -m scripts.m1 dev --config configs/m1.yaml
    PYTHONPATH=src:. python -m scripts.m1 final --config configs/m1.yaml

The five experts never share weights or an optimizer.  Formal training uses
torch.compile and CUDA BF16 through the existing engine policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from h9dp.common import load_config
from h9dp.engine import compile_model, initialize_device, seed_everything
from h9dp.m1 import (
    EXPERTS,
    EYE_SIZE_MAX,
    M5,
    PARAMETER_NAMES,
    M1Direct,
    M1Expert,
    metrics,
)
from scripts.audit_motis_inputs import audit

MOTIS_COLUMNS = ("B24 dT", "B36 dT", "Eye size", "B33 dT tr.", "T CDO")


def resolve(config: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["repository_root"]) / path


def write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except OSError as error:
        raise OSError(f"Cannot write {path}: {error}") from error


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON {path}: {error}") from error


# ---------------------------------------------------------------------------
# Data preparation


def region_map(center: np.ndarray, size: int, radii: tuple[int, int]) -> np.ndarray:
    """Native-pixel region ids from a (x, y) center; 0 inner, 1 mid, 2 outer."""
    coords = np.arange(size, dtype=np.float32) + 0.5
    dy = coords[:, None] - float(center[1])
    dx = coords[None, :] - float(center[0])
    distance = dy ** 2 + dx ** 2
    inner, outer = radii
    return (distance > inner * inner).astype(np.int8) + (distance > outer * outer).astype(np.int8)


def five(values) -> tuple[int, int, int, int, int]:
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]), int(values[4]))


def two_ints(values) -> tuple[int, int]:
    return (int(values[0]), int(values[1]))


def record_arrays(
    sample: dict, image: np.ndarray, mask: np.ndarray, radii: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize one record and compute native region statistics.

    image: float32 Kelvin [F,C,H,W]; mask: bool [F,1,H,W].
    Returns int16 cents, numeric [F,C,3,3], coverage [F,3], stats [F,C,3].
    """
    frames, bands, height, width = image.shape
    if (height, width) != (500, 500):
        raise ValueError(f"Unexpected window size {height}x{width}")
    quantized = np.rint(image * 100.0 - 15000.0)
    if quantized.min() < np.iinfo(np.int16).min or quantized.max() > np.iinfo(np.int16).max:
        raise ValueError("Brightness temperature outside int16 centi-Kelvin range")
    quantized = quantized.astype(np.int16)
    values = quantized.astype(np.float32) / 100.0 + 150.0
    numeric = np.zeros((frames, bands, 3, 3), dtype=np.float32)
    coverage = np.zeros((frames, 3), dtype=np.int32)
    stats = np.zeros((frames, bands, 3), dtype=np.float64)
    for frame in range(frames):
        valid = mask[frame, 0]
        if not valid.any():
            raise ValueError("Frame has no valid pixels")
        region = region_map(np.asarray(sample["centers"][frame], dtype=np.float32), height, radii)
        for index in range(3):
            selected = (region == index) & valid
            count = int(selected.sum())
            coverage[frame, index] = count
            if count == 0:
                continue
            for band in range(bands):
                subset = values[frame, band][selected]
                numeric[frame, band, index] = (subset.mean(), subset.min(), subset.max())
        for band in range(bands):
            subset = values[frame, band][valid].astype(np.float64)
            stats[frame, band] = (subset.sum(), (subset * subset).sum(), subset.size)
    return quantized, numeric, coverage, stats


def prepare(config: dict) -> dict:
    data = config["data"]
    cache = resolve(config, data["cache_dir"])
    work = resolve(config, data["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    csv_path = resolve(config, data["motis_csv"])
    records_path = resolve(config, data["records"])
    report = audit(csv_path, records_path, cache)
    pairs = report["splits"]["train"]["pairs"]
    folds = {item["record"]: item["fold"] for item in report["diagnostic_folds"]}
    if len(pairs) != len(folds):
        raise ValueError("Audit pairs and diagnostic folds disagree")
    brightness = np.load(cache / "brightness.npy", mmap_mode="r")
    validity = np.load(cache / "validity.npy", mmap_mode="r")
    train_samples = {sample["record"]: sample for sample in read_json(cache / "train.json")}
    validation_samples = read_json(cache / "validation.json")
    radii = two_ints(config["model"]["radii"])
    summary = {"records": {}, "splits": {}}
    for split, entries in (
        ("train", [{"sample": train_samples[pair["record"]], "pair": pair} for pair in pairs]),
        ("validation", [{"sample": sample, "pair": None} for sample in validation_samples]),
    ):
        count = len(entries)
        frames_total = int(config["model"]["frames"])
        images = np.lib.format.open_memmap(
            work / f"{split}_images.npy", mode="w+", dtype=np.int16,
            shape=(count, frames_total, 4, 500, 500),
        )
        masks = np.lib.format.open_memmap(
            work / f"{split}_masks.npy", mode="w+", dtype=np.bool_,
            shape=(count, frames_total, 1, 500, 500),
        )
        numeric = np.lib.format.open_memmap(
            work / f"{split}_numeric.npy", mode="w+", dtype=np.float32,
            shape=(count, frames_total, 4, 3, 3),
        )
        coverage = np.lib.format.open_memmap(
            work / f"{split}_coverage.npy", mode="w+", dtype=np.int32,
            shape=(count, frames_total, 3),
        )
        meta_records = []
        for position, entry in enumerate(entries):
            sample, pair = entry["sample"], entry["pair"]
            frames = np.asarray(sample["frames"], dtype=np.int64)
            if len(frames) != frames_total:
                raise ValueError(f"Record {sample['record']} has {len(frames)} frames")
            image = np.asarray(brightness[frames], dtype=np.float32)
            packed = np.asarray(validity[frames])
            unpacked = np.unpackbits(
                packed, axis=-1, count=250000, bitorder="little"
            ).reshape(frames_total, 4, 500, 500).astype(bool)
            mask = unpacked.all(axis=1, keepdims=True)
            quantized, region_numeric, region_coverage, stats = record_arrays(
                sample, image, mask, radii
            )
            images[position] = quantized
            masks[position] = mask
            numeric[position] = region_numeric
            coverage[position] = region_coverage
            meta_records.append({
                "record": int(sample["record"]),
                "name": sample["name"],
                "year": int(sample["year"]),
                "quality": sample["quality"],
                "dP": float(sample["dP"]),
                "params": None if pair is None else [float(pair["parameters"][column]) for column in MOTIS_COLUMNS],
                "fold": None if pair is None else int(folds[sample["record"]]),
                "frames": [int(value) for value in frames],
                "times": [float(value) for value in sample["time_offsets_minutes"]],
                "centers": [[float(value) for value in center] for center in sample["centers"]],
                "stats": stats.tolist(),
            })
        for array in (images, masks, numeric, coverage):
            array.flush()
        write_json(work / f"{split}_meta.json", {
            "schema": "m1-dataset/v1",
            "split": split,
            "bands": [8, 9, 10, 16],
            "centers_convention": "x, y (column, row)",
            "motis_columns": list(MOTIS_COLUMNS),
            "source": {
                "cache": str(cache),
                "csv": str(csv_path),
                "csv_sha256": report["inputs"]["csv"]["sha256"],
                "cache_fingerprint": read_json(cache / "index.json")["fingerprint"],
            },
            "records": meta_records,
        })
        summary["splits"][split] = {"count": count, "matched_parameters": sum(item["params"] is not None for item in meta_records)}
    summary["dev_fold_counts"] = {}
    for fold in folds.values():
        summary["dev_fold_counts"][str(fold)] = summary["dev_fold_counts"].get(str(fold), 0) + 1
    summary["report"] = {
        "paper_formula_vs_source_dP": report["paper_formula_vs_source_dP"],
        "oracle_oof": report.get("oracle_oof", {}),
    }
    write_json(work / "prepare_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Dataset and normalization


class M1Dataset(Dataset):
    def __init__(self, work: Path, split: str, indices):
        self.images = np.load(work / f"{split}_images.npy", mmap_mode="r")
        self.masks = np.load(work / f"{split}_masks.npy", mmap_mode="r")
        self.numeric = np.load(work / f"{split}_numeric.npy", mmap_mode="r")
        self.coverage = np.load(work / f"{split}_coverage.npy", mmap_mode="r")
        self.meta = read_json(work / f"{split}_meta.json")
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        index = self.indices[item]
        record = self.meta["records"][index]
        images = np.asarray(self.images[index], dtype=np.float32) / 100.0 + 150.0
        params = record["params"]
        target = np.array(params, dtype=np.float32) if params is not None else np.full(5, np.nan, np.float32)
        return {
            "images": torch.from_numpy(images),
            "mask": torch.from_numpy(np.array(self.masks[index])),
            "numeric": torch.from_numpy(np.array(self.numeric[index])),
            "coverage": torch.from_numpy(np.array(self.coverage[index])),
            "times": torch.tensor(record["times"], dtype=torch.float32),
            "centers": torch.tensor(record["centers"], dtype=torch.float32),
            "params": torch.from_numpy(target),
            "dp": torch.tensor(record["dP"], dtype=torch.float32),
            "record": index,
        }


def normalization(meta: dict, indices) -> tuple[np.ndarray, np.ndarray]:
    stats = np.asarray([meta["records"][index]["stats"] for index in indices], dtype=np.float64)
    count = stats[..., 2].sum(axis=(0, 1))
    if np.any(count <= 0):
        raise ValueError("Empty band in normalization statistics")
    mean = stats[..., 0].sum(axis=(0, 1)) / count
    variance = stats[..., 1].sum(axis=(0, 1)) / count - mean ** 2
    if np.any(variance <= 0):
        raise ValueError("Degenerate band variance")
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def target_statistics(meta: dict, indices, column: int) -> tuple[float, float]:
    values = np.asarray([meta["records"][index]["params"][column] for index in indices], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


# ---------------------------------------------------------------------------
# Training


def move(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device) if torch.is_tensor(value) else value for name, value in batch.items()}


def make_loader(dataset: Dataset, config: dict, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def model_arch(config: dict) -> dict:
    model = config["model"]
    return {
        "bands": int(model["bands"]),
        "frames": int(model["frames"]),
        "image_size": int(model["image_size"]),
        "radii": two_ints(model["radii"]),
    }


def build_expert(spec, config: dict, mean, std, target_mean, target_std, device) -> M1Expert:
    model = M1Expert(
        spec,
        **model_arch(config),
        channels=five(config["model"]["channels"]),
        frame_width=int(config["model"]["frame_width"]),
        temporal_width=int(config["model"]["temporal_width"]),
    )
    model.encoder.band_mean.copy_(torch.from_numpy(mean))
    model.encoder.band_std.copy_(torch.from_numpy(std))
    model.target_mean.fill_(float(target_mean))
    model.target_std.fill_(float(target_std))
    model.to(device)
    return model


def build_control(config: dict, mean, std, dp_mean, dp_std, device) -> M1Direct:
    model = M1Direct(
        **model_arch(config),
        channels=five(config["model"]["control_channels"]),
        frame_width=int(config["model"]["control_frame_width"]),
        temporal_width=int(config["model"]["control_temporal_width"]),
    )
    model.encoder.band_mean.copy_(torch.from_numpy(mean))
    model.encoder.band_std.copy_(torch.from_numpy(std))
    model.target_mean.fill_(float(dp_mean))
    model.target_std.fill_(float(dp_std))
    model.to(device)
    return model


def expert_loss(outputs, batch, column, mean, std):
    residual = (outputs[:, 1] - batch["params"][:, column]) / std
    return F.mse_loss(residual, torch.zeros_like(residual))


def control_loss(outputs, batch, mean, std):
    residual = (outputs[:, 0] - batch["dp"]) / std
    return F.mse_loss(residual, torch.zeros_like(residual))


def evaluate_loss(outputs, batch, column, mean, std) -> tuple[float, float, int]:
    if column is None:
        target = batch["dp"]
        prediction = outputs[:, 0]
    else:
        target = batch["params"][:, column]
        prediction = outputs[:, 1]
    error = prediction - target
    return float((error.double() ** 2).sum()), float(error.double().abs().sum()), int(error.numel())


def train_model(
    model,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    epochs: int,
    loss_fn,
    monitor,
    checkpoint_path: Path,
    log_path: Path,
    seed: int,
    target_mean: float,
    target_std: float,
) -> dict:
    seed_everything(seed)
    training = config["training"]
    compiled = compile_model(model, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["lr_factor"]),
        patience=int(training["lr_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )
    use_autocast = training["precision"] == "bf16"
    history = []
    best = float("inf")
    best_epoch = 0
    stale = 0
    first_step = None
    marker = 0.0
    peak = 0
    torch.cuda.reset_peak_memory_stats(device)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            start = time.perf_counter()
            total = 0.0
            seen = 0
            for step, batch in enumerate(train_loader):
                batch = move(batch, device)
                cold = first_step is None and epoch == 1 and step == 0
                if cold:
                    marker = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    outputs = compiled(
                        batch["images"], batch["mask"], batch["numeric"],
                        batch["coverage"], batch["times"], batch["centers"],
                    )
                    loss = loss_fn(outputs, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                optimizer.step()
                if cold:
                    torch.cuda.synchronize(device)
                    first_step = time.perf_counter() - marker
                total += float(loss.detach())
                seen += 1
            train_loss = total / max(seen, 1)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            if val_loader is not None:
                model.eval()
                squared, absolute, count = 0.0, 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = move(batch, device)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                            outputs = model(
                                batch["images"], batch["mask"], batch["numeric"],
                                batch["coverage"], batch["times"], batch["centers"],
                            )
                        s, a, c = monitor(outputs, batch)
                        squared += s
                        absolute += a
                        count += c
                val_rmse = float(np.sqrt(squared / max(count, 1)))
                val_mae = absolute / max(count, 1)
                val_loss = val_rmse / target_std
                scheduler.step(val_loss)
            else:
                val_loss, val_rmse, val_mae = train_loss, float("nan"), float("nan")
                scheduler.step(train_loss)
            peak = max(peak, torch.cuda.max_memory_allocated(device))
            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": seconds,
                "peak_memory_bytes": peak,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            history.append(entry)
            monitored = val_loss if val_loader is not None else train_loss
            improved = val_loader is None or monitored < best - 1e-12
            if improved:
                best = monitored
                best_epoch = epoch
                stale = 0
                optimizer_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                torch.save({
                    "state_dict": optimizer_state,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "epoch": epoch,
                    "best": best,
                    "first_step_seconds": first_step,
                    "phase": checkpoint_path.parent.name,
                }, checkpoint_path)
            else:
                stale += 1
            if val_loader is not None and stale >= int(training["early_stopping_patience"]):
                break
    return {
        "best_epoch": best_epoch,
        "best_monitor": best,
        "epochs_run": len(history),
        "first_step_seconds": first_step,
        "steady_epoch_seconds": float(np.median([item["seconds"] for item in history[1:]])) if len(history) > 1 else None,
        "peak_memory_bytes": peak,
        "history": history,
    }


def load_checkpoint(path: Path, device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint: {path}")
    return payload


def predict_expert(model, loader, device, use_autocast=True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    raw_outputs, projected = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                output = model(
                    batch["images"], batch["mask"], batch["numeric"],
                    batch["coverage"], batch["times"], batch["centers"],
                )
            raw_outputs.append(output[:, 0].float().cpu().numpy())
            projected.append(output[:, 1].float().cpu().numpy())
    return np.concatenate(raw_outputs), np.concatenate(projected)


def predict_control(model, loader, device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["images"], batch["mask"], batch["numeric"],
                    batch["coverage"], batch["times"], batch["centers"],
                )
            outputs.append(output[:, 0].float().cpu().numpy())
    return np.concatenate(outputs)


# ---------------------------------------------------------------------------
# Phases


def indices_with_parameters(meta: dict) -> list[int]:
    return [i for i, record in enumerate(meta["records"]) if record["params"] is not None]


def dev_indices(meta: dict) -> tuple[list[int], list[int]]:
    train = [i for i, record in enumerate(meta["records"]) if record["fold"] != 0]
    holdout = [i for i, record in enumerate(meta["records"]) if record["fold"] == 0]
    if len(train) + len(holdout) != len(meta["records"]):
        raise ValueError("Unassigned dev fold")
    return train, holdout


def quality_groups(meta: dict, indices) -> dict:
    """Positions refer to the prediction array, not to the global record index."""
    groups = {}
    for position, index in enumerate(indices):
        groups.setdefault(meta["records"][index]["quality"], []).append(position)
    return groups


def summarize_groups(prediction, target, groups) -> dict:
    return {name: metrics(np.asarray(prediction)[index], np.asarray(target)[index]) for name, index in groups.items()}


def run_dev(config: dict) -> dict:
    data = config["data"]
    work = resolve(config, data["work_dir"])
    output = resolve(config, config["output_dir"]) / "dev"
    output.mkdir(parents=True, exist_ok=True)
    device, environment = initialize_device(config)
    meta = read_json(work / "train_meta.json")
    train_idx, holdout_idx = dev_indices(meta)
    mean, std = normalization(meta, train_idx)
    train_dataset = M1Dataset(work, "train", train_idx)
    holdout_dataset = M1Dataset(work, "train", holdout_idx)
    val_loader = DataLoader(holdout_dataset, batch_size=int(config["training"]["eval_batch_size"]), shuffle=False)
    train_loader = make_loader(train_dataset, config, shuffle=True, generator=torch.Generator().manual_seed(int(config["training"]["seed"])))
    train_eval_loader = DataLoader(train_dataset, batch_size=int(config["training"]["eval_batch_size"]), shuffle=False)
    dp_train = np.asarray([meta["records"][i]["dP"] for i in train_idx])
    dp_holdout = np.asarray([meta["records"][i]["dP"] for i in holdout_idx])
    dp_mean, dp_std = float(dp_train.mean()), float(dp_train.std(ddof=0))
    result = {
        "status": "dev phase; 46-record holdout selected by fold 0 of the tabular audit",
        "environment": environment,
        "records": {"train": len(train_idx), "holdout": len(holdout_idx)},
        "normalization": {"mean_kelvin": mean.tolist(), "std_kelvin": std.tolist()},
        "targets": {},
        "experts": {},
        "control": {},
        "baseline": {},
    }
    predictions = {}
    histories = {}
    for position, spec in enumerate(EXPERTS):
        column = PARAMETER_NAMES.index(spec.parameter)
        target_mean, target_std = target_statistics(meta, train_idx, column)
        model = build_expert(spec, config, mean, std, target_mean, target_std, device)
        history = train_model(
            model,
            train_loader,
            val_loader,
            config,
            device,
            int(config["training"]["dev_epochs"]),
            lambda outputs, batch, column=column, mean=target_mean, std=target_std: expert_loss(outputs, batch, column, mean, std),
            lambda outputs, batch, column=column, mean=target_mean, std=target_std: evaluate_loss(outputs, batch, column, mean, std),
            output / f"{spec.name}_checkpoint.pt",
            output / f"{spec.name}.jsonl",
            int(config["training"]["seed"]) + position,
            target_mean,
            target_std,
        )
        payload = load_checkpoint(output / f"{spec.name}_checkpoint.pt", device)
        model.load_state_dict(payload["state_dict"])
        _, holdout = predict_expert(model, val_loader, device)
        train_predictions = predict_expert(model, train_eval_loader, device)
        targets = np.asarray([meta["records"][i]["params"][column] for i in holdout_idx])
        train_targets = np.asarray([meta["records"][i]["params"][column] for i in train_idx])
        baseline = float(np.mean(train_targets))
        trigger = None
        if spec.activation == "sigmoid":
            trigger = float(np.mean((holdout < 0.01 * EYE_SIZE_MAX) | (holdout > 0.99 * EYE_SIZE_MAX)))
        elif spec.activation == "softplus":
            trigger = float(np.mean(holdout < 0.01))
        result["experts"][spec.name] = {
            "parameter": spec.parameter,
            "parameters": model.parameter_count(),
            **history,
            "target_mean": target_mean,
            "target_std": target_std,
            "holdout": metrics(holdout, targets),
            "train": metrics(train_predictions[1], train_targets),
            "constant_baseline_rmse": float(np.sqrt(np.mean((baseline - targets) ** 2))),
            "projection_trigger_rate": trigger,
            "checkpoint": str(output / f"{spec.name}_checkpoint.pt"),
        }
        predictions[spec.name] = holdout
        histories[spec.name] = history
    direct = build_control(config, mean, std, dp_mean, dp_std, device)
    control_history = train_model(
        direct,
        train_loader,
        val_loader,
        config,
        device,
        int(config["training"]["dev_epochs"]),
        lambda outputs, batch: control_loss(outputs, batch, dp_mean, dp_std),
        lambda outputs, batch: evaluate_loss(outputs, batch, None, dp_mean, dp_std),
        output / "control_checkpoint.pt",
        output / "control.jsonl",
        int(config["training"]["seed"]) + 10,
        dp_mean,
        dp_std,
    )
    control_payload = load_checkpoint(output / "control_checkpoint.pt", device)
    direct.load_state_dict(control_payload["state_dict"])
    control_holdout = predict_control(direct, val_loader, device)
    result["control"] = {
        "parameters": direct.parameter_count(),
        **control_history,
        "holdout": metrics(control_holdout, dp_holdout),
        "checkpoint": str(output / "control_checkpoint.pt"),
    }
    result["baseline"]["holdout"] = metrics(np.full(len(dp_holdout), dp_mean), dp_holdout)
    real_params = np.asarray([meta["records"][i]["params"] for i in holdout_idx])
    train_params = np.asarray([meta["records"][i]["params"] for i in train_idx])
    m5 = M5.fit(train_params, dp_train)
    result["m5"] = {
        "coefficients": m5.coefficients.tolist(),
        "train_records": len(train_idx),
        "oracle_holdout": metrics(m5.predict(real_params), dp_holdout),
        "one_at_a_time": {},
        "all_predicted": {},
    }
    groups = quality_groups(meta, holdout_idx)
    predicted_matrix = np.column_stack([predictions[spec.name] for spec in EXPERTS])
    result["m5"]["all_predicted"] = {
        **metrics(m5.predict(predicted_matrix), dp_holdout),
        "by_quality": summarize_groups(m5.predict(predicted_matrix), dp_holdout, groups),
    }
    for position, spec in enumerate(EXPERTS):
        replaced = real_params.copy()
        replaced[:, position] = predictions[spec.name]
        result["m5"]["one_at_a_time"][spec.name] = metrics(m5.predict(replaced), dp_holdout)
    rows = []
    for row, index in enumerate(holdout_idx):
        record = meta["records"][index]
        row_data = {
            "record": record["record"], "name": record["name"], "quality": record["quality"],
            "dP": record["dP"], "dP_oracle": float(m5.predict(real_params[row:row + 1])[0]),
            "dP_all": float(m5.predict(predicted_matrix[row:row + 1])[0]),
            "dP_control": float(control_holdout[row]),
        }
        for column, column_name in enumerate(MOTIS_COLUMNS):
            row_data[f"real_{column_name}"] = record["params"][column]
        for spec in EXPERTS:
            row_data[f"pred_{spec.parameter}"] = float(predictions[spec.name][row])
        rows.append(row_data)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "metrics.json", result)
    return result


def run_final(config: dict) -> dict:
    data = config["data"]
    work = resolve(config, data["work_dir"])
    output = resolve(config, config["output_dir"]) / "final"
    output.mkdir(parents=True, exist_ok=True)
    dev_metrics = read_json(resolve(config, config["output_dir"]) / "dev" / "metrics.json")
    device, environment = initialize_device(config)
    train_meta = read_json(work / "train_meta.json")
    validation_meta = read_json(work / "validation_meta.json")
    train_idx = indices_with_parameters(train_meta)
    if len(train_idx) != len(train_meta["records"]):
        raise ValueError("Final training requires a parameter for every training record")
    mean, std = normalization(train_meta, train_idx)
    train_dataset = M1Dataset(work, "train", train_idx)
    validation_dataset = M1Dataset(work, "validation", range(len(validation_meta["records"])))
    train_loader = make_loader(train_dataset, config, shuffle=True, generator=torch.Generator().manual_seed(int(config["training"]["seed"])))
    validation_loader = DataLoader(validation_dataset, batch_size=int(config["training"]["eval_batch_size"]), shuffle=False)
    dp_train = np.asarray([train_meta["records"][i]["dP"] for i in train_idx])
    dp_validation = np.asarray([record["dP"] for record in validation_meta["records"]])
    dp_mean, dp_std = float(dp_train.mean()), float(dp_train.std(ddof=0))
    result = {
        "status": "final phase; fixed epochs from dev, validation is the 49-record 2026 split",
        "environment": environment,
        "records": {"train": len(train_idx), "validation": len(validation_meta["records"])},
        "normalization": {"mean_kelvin": mean.tolist(), "std_kelvin": std.tolist()},
        "dev_best_epochs": {},
        "experts": {},
        "control": {},
        "baseline": {},
    }
    predictions = {}
    for position, spec in enumerate(EXPERTS):
        column = PARAMETER_NAMES.index(spec.parameter)
        target_mean, target_std = target_statistics(train_meta, train_idx, column)
        epochs = max(1, int(dev_metrics["experts"][spec.name]["best_epoch"]))
        result["dev_best_epochs"][spec.name] = epochs
        model = build_expert(spec, config, mean, std, target_mean, target_std, device)
        history = train_model(
            model,
            train_loader,
            None,
            config,
            device,
            epochs,
            lambda outputs, batch, column=column, mean=target_mean, std=target_std: expert_loss(outputs, batch, column, mean, std),
            lambda outputs, batch, column=column, mean=target_mean, std=target_std: evaluate_loss(outputs, batch, column, mean, std),
            output / f"{spec.name}_checkpoint.pt",
            output / f"{spec.name}.jsonl",
            int(config["training"]["seed"]) + position,
            target_mean,
            target_std,
        )
        payload = load_checkpoint(output / f"{spec.name}_checkpoint.pt", device)
        model.load_state_dict(payload["state_dict"])
        _, projected = predict_expert(model, validation_loader, device)
        predictions[spec.name] = projected
        result["experts"][spec.name] = {
            "parameter": spec.parameter,
            "parameters": model.parameter_count(),
            **history,
            "target_mean": target_mean,
            "target_std": target_std,
            "checkpoint": str(output / f"{spec.name}_checkpoint.pt"),
        }
    direct = build_control(config, mean, std, dp_mean, dp_std, device)
    epochs = max(1, int(dev_metrics["control"]["best_epoch"]))
    result["dev_best_epochs"]["control"] = epochs
    control_history = train_model(
        direct,
        train_loader,
        None,
        config,
        device,
        epochs,
        lambda outputs, batch: control_loss(outputs, batch, dp_mean, dp_std),
        lambda outputs, batch: evaluate_loss(outputs, batch, None, dp_mean, dp_std),
        output / "control_checkpoint.pt",
        output / "control.jsonl",
        int(config["training"]["seed"]) + 10,
        dp_mean,
        dp_std,
    )
    control_payload = load_checkpoint(output / "control_checkpoint.pt", device)
    direct.load_state_dict(control_payload["state_dict"])
    control_validation = predict_control(direct, validation_loader, device)
    result["control"] = {
        "parameters": direct.parameter_count(),
        **control_history,
        "validation": metrics(control_validation, dp_validation),
        "checkpoint": str(output / "control_checkpoint.pt"),
    }
    result["baseline"]["validation"] = metrics(np.full(len(dp_validation), dp_mean), dp_validation)
    train_params = np.asarray([train_meta["records"][i]["params"] for i in train_idx])
    m5 = M5.fit(train_params, dp_train)
    predicted_matrix = np.column_stack([predictions[spec.name] for spec in EXPERTS])
    groups = quality_groups(validation_meta, range(len(validation_meta["records"])))
    result["m5"] = {
        "coefficients": m5.coefficients.tolist(),
        "train_records": len(train_idx),
        "validation": {
            **metrics(m5.predict(predicted_matrix), dp_validation),
            "by_quality": summarize_groups(m5.predict(predicted_matrix), dp_validation, groups),
        },
    }
    rows = []
    for row, record in enumerate(validation_meta["records"]):
        row_data = {
            "record": record["record"], "name": record["name"], "quality": record["quality"],
            "dP": record["dP"], "dP_m5": float(m5.predict(predicted_matrix[row:row + 1])[0]),
            "dP_control": float(control_validation[row]),
        }
        for spec in EXPERTS:
            row_data[f"pred_{spec.parameter}"] = float(predictions[spec.name][row])
        rows.append(row_data)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dev", "final"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1.yaml"))
    parser.add_argument("--epochs", type=int, help="diagnostic override of dev epochs")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None and args.command == "dev":
        config["training"]["dev_epochs"] = args.epochs
    if args.command == "prepare":
        summary = prepare(config)
        print(json.dumps({"splits": summary["splits"], "dev_fold_counts": summary["dev_fold_counts"]}, indent=2))
    elif args.command == "dev":
        result = run_dev(config)
        print(json.dumps({
            "experts": {name: {"best_epoch": value["best_epoch"], "holdout_rmse": value["holdout"]["rmse"]} for name, value in result["experts"].items()},
            "m5": result["m5"]["all_predicted"],
            "control": result["control"]["holdout"],
        }, indent=2))
    else:
        result = run_final(config)
        print(json.dumps({
            "m5": result["m5"]["validation"],
            "control": result["control"]["validation"],
            "baseline": result["baseline"]["validation"],
        }, indent=2))


if __name__ == "__main__":
    main()
