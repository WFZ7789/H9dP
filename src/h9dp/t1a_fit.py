"""Explicit real-record fitability diagnostic for T1A (test-1 a).

This is not the formal training engine. It repeatedly fits one fixed training
subset, with no augmentation/weight decay and no validation-set evaluation.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import operator
import os
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .common import check_space, fingerprint, write_json
from .dataset import SequenceDataset
from .engine import code_fingerprint, make_optimizer, seed_everything
from .model import build_model, model_architecture, regression_loss
from .prepare import load_cache
from .t1a import ARCHITECTURE


def select_records(index: dict, count: int = 24, seed: int = 1) -> list[dict]:
    """Uniform record sampling, independent of labels, names and quality."""
    count = operator.index(count)
    records = [operator.index(sample["record"]) for sample in index["samples"]]
    if len(records) != len(set(records)):
        raise ValueError("Duplicate record IDs, including across splits")
    training = sorted((s for s in index["samples"] if s["split"] == "train"),
                      key=lambda sample: sample["record"])
    if not 2 <= count <= len(training):
        raise ValueError("Need 2..number_of_training_records distinct records")
    selected = np.random.default_rng(seed).choice(len(training), size=count, replace=False)
    return sorted((copy.deepcopy(training[position]) for position in selected),
                  key=lambda sample: sample["record"])


def input_identity_audit(images: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor,
                         targets: torch.Tensor, records: list[int]) -> dict:
    """Hash the actual model inputs, not labels; quantify exact-input conflicts."""
    groups = defaultdict(list)
    hashes = []
    for position, record in enumerate(records):
        digest = hashlib.sha256()
        clean = images[position].masked_fill(~mask[position], 0)
        for tensor in (clean, offsets[position], mask[position]):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        value = digest.hexdigest()
        hashes.append({"record": record, "sha256": value})
        groups[value].append(position)
    values = targets.detach().cpu().double().numpy()
    irreducible_sum = 0.0
    duplicates = []
    for positions in groups.values():
        group = values[positions]
        irreducible_sum += np.square(group - group.mean()).sum().item()
        if len(positions) > 1:
            duplicates.append({
                "records": [records[i] for i in positions], "targets_hpa": group.tolist(),
                "conflicting_targets": bool(np.ptp(group) > 0),
            })
    return {
        "hashes": hashes, "unique_inputs": len(groups), "duplicate_groups": duplicates,
        "exact_input_irreducible_rmse_hpa": math.sqrt(irreducible_sum / len(records)),
    }


def fit_metrics(prediction, target) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 1 or prediction.shape != target.shape or len(target) < 2:
        raise ValueError("Need matching one-dimensional predictions/targets")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Non-finite fit metrics")
    errors = prediction - target
    target_std, predicted_std = target.std().item(), prediction.std().item()
    return {
        "count": len(target), "rmse_hpa": np.square(errors).mean().item() ** 0.5,
        "mae_hpa": np.abs(errors).mean().item(), "bias_hpa": errors.mean().item(),
        "max_abs_error_hpa": np.abs(errors).max().item(),
        "target_mean_hpa": target.mean().item(), "target_std_hpa": target_std,
        "target_min_hpa": target.min().item(), "target_max_hpa": target.max().item(),
        "prediction_mean_hpa": prediction.mean().item(), "prediction_std_hpa": predicted_std,
        "prediction_min_hpa": prediction.min().item(), "prediction_max_hpa": prediction.max().item(),
        "std_ratio": predicted_std / target_std if target_std > 0 else None,
        "correlation": np.corrcoef(prediction, target)[0, 1].item()
        if target_std > 1e-12 and predicted_std > 1e-12 else None,
        "negative_prediction_count": np.count_nonzero(prediction < 0).item(),
    }


def passes_gate(metrics: dict, rmse_target: float = 1.0) -> bool:
    ratio, correlation = metrics["std_ratio"], metrics["correlation"]
    return bool(metrics["rmse_hpa"] <= rmse_target and ratio is not None and 0.9 <= ratio <= 1.1
                and correlation is not None and correlation >= 0.99)


def _preload(config: dict, index: dict, selected: list[dict]) -> dict[str, torch.Tensor]:
    subset_index = dict(index, samples=selected)
    dataset = SequenceDataset(config["data"]["cache_dir"], subset_index, "train", augmentation=None)
    rows = [dataset[position] for position in range(len(dataset))]
    data = {key: torch.stack([row[key] for row in rows])
            for key in ("images", "mask", "time_offsets", "target")}
    if data["images"].shape != (len(selected), 6, 4, 500, 500):
        raise ValueError("Real fit check requires native six-frame four-band 500x500 inputs")
    data["images"].masked_fill_(~data["mask"], 0)
    return data


@torch.no_grad()
def _predict(model, data, batch_size, bf16=True):
    model.eval()
    predictions = []
    for start in range(0, len(data["target"]), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            value = model(data["images"][start:start + batch_size],
                          data["time_offsets"][start:start + batch_size],
                          data["mask"][start:start + batch_size])
        predictions.append(value.float().cpu())
    return torch.cat(predictions).numpy()


def _save_predictions(path: Path, selected: list[dict], prediction) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "record", "storm_id", "quality", "target_hpa", "prediction_hpa", "error_hpa"))
        writer.writeheader()
        for sample, value in zip(selected, prediction.tolist(), strict=True):
            writer.writerow({
                "record": sample["record"], "storm_id": sample["storm_id"], "quality": sample["quality"],
                "target_hpa": sample["dP"], "prediction_hpa": value, "error_hpa": value - sample["dP"],
            })


def _save_checkpoint(path: Path, model, optimizer, step, metadata) -> None:
    check_space(path, 64_000_000)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "kind": "t1a-fitability/v1", "architecture": ARCHITECTURE, "step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata,
    }, temporary)
    os.replace(temporary, path)


def _update(model, optimizer, data, permutation, batch_size, target_std, clip):
    model.train()
    count = len(data["target"])
    optimizer.zero_grad(set_to_none=True)
    online_squared = 0.0
    for start in range(0, count, batch_size):
        positions = permutation[start:start + batch_size]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(data["images"][positions], data["time_offsets"][positions],
                               data["mask"][positions])
            losses = regression_loss(prediction, data["target"][positions], target_std)
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("Non-finite training loss")
        (losses.sum() / count).backward()
        online_squared += (prediction.detach() - data["target"][positions]).square().sum().item()
    telemetry, before = {}, []
    for group in optimizer.param_groups:
        missing = sum(parameter.grad is None for parameter in group["params"])
        if missing:
            raise RuntimeError(f"Missing gradients in {group['name']}: {missing}")
        norm = torch.nn.utils.clip_grad_norm_(group["params"], clip, error_if_nonfinite=True).item()
        telemetry[group["name"]] = {
            "gradient_norm_before_clip": norm, "clip_multiplier": min(1.0, clip / (norm + 1e-12)),
            "missing_gradient_tensors": missing,
        }
        before.append([parameter.detach().clone() for parameter in group["params"]])
    optimizer.step()
    for group, previous in zip(optimizer.param_groups, before, strict=True):
        squared_norm = torch.stack([value.square().sum() for value in previous]).sum()
        squared_update = torch.stack([
            (parameter.detach() - value).square().sum()
            for parameter, value in zip(group["params"], previous, strict=True)
        ]).sum()
        telemetry[group["name"]]["relative_update_l2"] = (
            squared_update.sqrt() / squared_norm.sqrt().clamp_min(1e-12)).item()
    return {"online_rmse_hpa_before_update": math.sqrt(online_squared / count),
            "gradient_groups": telemetry}


def run_fit(config: dict, directory: Path, *, count=24, max_steps=500, batch_size=6,
            learning_rate=0.0003, seed=1, evaluate_every=10, rmse_target=1.0) -> dict:
    """Fit exactly one subset from fresh weights; never instantiate validation data."""
    if (count < 2 or max_steps < 1 or batch_size < 1 or evaluate_every < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(rmse_target) or rmse_target <= 0):
        raise ValueError("Invalid fit-check options")
    if model_architecture(config["model"]) != ARCHITECTURE:
        raise ValueError("This diagnostic only supports T1A")
    storage = check_space(directory, 64_000_000)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required for this real-data diagnostic")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {
        "status": "preparing", "model_name": "T1A", "name_meaning": "test-1 a",
        "architecture": ARCHITECTURE, "diagnostic_only": True,
        "formal_training_started": False, "validation_evaluated": False,
        "started_utc": datetime.now(UTC).isoformat(), "pid": os.getpid(), "storage": storage,
        "count": count, "seed": seed, "max_optimizer_steps": max_steps,
        "microbatch_size": batch_size, "effective_batch_size": count,
        "learning_rate": learning_rate, "weight_decay": 0.0, "augmentation": None,
        "gradient_clip_each_group": 5.0, "evaluate_every": evaluate_every,
        "precision": "bf16_autocast", "torch_compile": False,
        "gate": {"rmse_hpa_at_most": rmse_target, "std_ratio_range": [0.9, 1.1],
                 "correlation_at_least": 0.99, "consecutive_bf16_evaluations": 2,
                 "final_fp32_must_also_pass": True},
        "interpretation": "Training-subset memorization diagnostic, not generalization accuracy.",
    }
    write_json(directory / "run.json", metadata)
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = False
        seed_everything(seed)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        index = load_cache(config)
        selected = select_records(index, count, seed)
        records = [sample["record"] for sample in selected]
        write_json(directory / "selection.json", {
            "method": "uniform_without_replacement_from_sorted_training_records",
            "seed": seed, "records": records, "samples": selected,
            "data_fingerprint": index["fingerprint"],
        })
        data = _preload(config, index, selected)
        identity = input_identity_audit(data["images"], data["time_offsets"], data["mask"],
                                         data["target"], records)
        write_json(directory / "input_identity.json", identity)
        if identity["exact_input_irreducible_rmse_hpa"] > rmse_target:
            raise ValueError("Exact duplicate inputs with conflicting labels make the RMSE gate unreachable")
        target = index["target_normalization"]
        model = build_model(config["model"], target["mean_hpa"], target["std_hpa"]).to(device)
        effective = copy.deepcopy(config)
        effective["training"].update(learning_rate=learning_rate, weight_decay=0.0)
        optimizer = make_optimizer(model, effective)
        targets = data["target"].numpy().copy()
        metadata.update({
            "status": "running", "data_fingerprint": index["fingerprint"],
            "code_fingerprint": code_fingerprint(config), "configuration": config,
            "selected_records": records, "selection_fingerprint": fingerprint(selected),
            "target_normalization": target, "image_normalization": index["normalization"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "unique_inputs": identity["unique_inputs"],
            "shared_frame_occurrences": count * 6 - len({f for sample in selected for f in sample["frames"]}),
            "gpu_cached_input_bytes": sum(value.numel() * value.element_size() for value in data.values()),
            "runner_sha256": hashlib.sha256((Path(config["repository_root"]) / "scripts" / "fit_t1a.py").read_bytes()).hexdigest(),
        })
        write_json(directory / "run.json", metadata)
        data = {key: value.to(device) for key, value in data.items()}
        torch.cuda.reset_peak_memory_stats(device)
        initial_prediction = _predict(model, data, batch_size)
        initial = fit_metrics(initial_prediction, targets)
        _save_predictions(directory / "initial_predictions.csv", selected, initial_prediction)
        best, best_step, consecutive = initial, 0, 0
        _save_checkpoint(directory / "best.pt", model, optimizer, 0, metadata)
        history_path = directory / "history.jsonl"

        def record_evaluation(step, metrics, telemetry):
            entry = {"step": step, "elapsed_seconds": time.monotonic() - started,
                     **metrics, **telemetry}
            with history_path.open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
            metadata.update(step=step, latest_metrics=metrics, best_step=best_step, best_metrics=best,
                            elapsed_seconds=entry["elapsed_seconds"],
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            write_json(directory / "run.json", metadata)
            print(json.dumps(entry, allow_nan=False), flush=True)

        record_evaluation(0, initial, {})
        generator = torch.Generator().manual_seed(seed)
        telemetry = {}
        step = 0
        for step in range(1, max_steps + 1):
            check_space(directory)
            permutation = torch.randperm(count, generator=generator).to(device)
            telemetry = _update(model, optimizer, data, permutation, batch_size, model.target_std, 5.0)
            if step == 1 or step % evaluate_every == 0 or step == max_steps:
                prediction = _predict(model, data, batch_size)
                metrics = fit_metrics(prediction, targets)
                if metrics["rmse_hpa"] < best["rmse_hpa"]:
                    best, best_step = metrics, step
                    _save_checkpoint(directory / "best.pt", model, optimizer, step, metadata)
                consecutive = consecutive + 1 if passes_gate(metrics, rmse_target) else 0
                record_evaluation(step, metrics, telemetry)
                if consecutive >= 2:
                    break
        last_prediction = _predict(model, data, batch_size)
        last_metrics = fit_metrics(last_prediction, targets)
        _save_predictions(directory / "last_predictions.csv", selected, last_prediction)
        _save_checkpoint(directory / "last.pt", model, optimizer, step, metadata)
        state = torch.load(directory / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        best_prediction = _predict(model, data, batch_size)
        fp32_prediction = _predict(model, data, batch_size, bf16=False)
        fp32_metrics = fit_metrics(fp32_prediction, targets)
        _save_predictions(directory / "best_predictions.csv", selected, best_prediction)
        _save_predictions(directory / "best_predictions_fp32.csv", selected, fp32_prediction)
        report = {
            "passed": consecutive >= 2 and passes_gate(fp32_metrics, rmse_target),
            "stop_reason": "stable_bf16_fit_gate" if consecutive >= 2 else "optimizer_step_budget_exhausted",
            "optimizer_steps": step, "best_step": best_step, "initial": initial, "best": best,
            "last": last_metrics, "best_fp32": fp32_metrics,
            "initialization_baseline_rmse_hpa": initial["rmse_hpa"],
            "subset_optimal_constant_rmse_hpa": targets.astype(np.float64).std().item(),
            "final_gradient_groups": telemetry.get("gradient_groups"),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "elapsed_seconds": time.monotonic() - started,
            "real_records": count, "validation_evaluated": False, "formal_training_started": False,
            "interpretation": metadata["interpretation"],
        }
        write_json(directory / "report.json", report)
        metadata.update(status="completed", **report, finished_utc=datetime.now(UTC).isoformat())
        write_json(directory / "run.json", metadata)
        print(json.dumps({"run_directory": str(directory), **report}, allow_nan=False), flush=True)
        return report
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error), traceback=traceback.format_exc(),
                        elapsed_seconds=time.monotonic() - started)
        write_json(directory / "run.json", metadata)
        raise
