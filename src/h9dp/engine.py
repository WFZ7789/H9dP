"""Record-equal training, physical metrics and explicit optimization telemetry."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import check_space, fingerprint, write_json
from .dataset import make_loader
from .model import build_model, model_architecture, regression_loss
from .prepare import load_cache

CHECKPOINT_KIND = "h9dp-training/v2"


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} is not an integer") from error


def _parse_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} is not numeric") from error


def configuration_fingerprint(config):
    return fingerprint({key: config[key] for key in ("data", "model", "training", "augmentation")})


def code_fingerprint(config):
    root = Path(config["repository_root"])
    paths = sorted([*(root / "src" / "h9dp").glob("*.py"), root / "scripts" / "h9dp.py"])
    return fingerprint({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_device(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable in the current environment; check device visibility and sandbox access")
    if config["training"]["precision"] != "bf16" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires CUDA BF16 support")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "gpu": properties.name, "gpu_total_bytes": properties.total_memory,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "numpy": np.__version__, "mkl_threading_layer": os.environ.get("MKL_THREADING_LAYER"),
        "cpu_threads": torch.get_num_threads(),
    }


def compile_model(model, config):
    """Compile only the forward module; checkpoints keep the raw model state."""
    training = config["training"]
    if not training["torch_compile"]:
        return model
    mode = training.get("compile_mode", "default")
    supported_modes = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported torch.compile mode: {mode}")
    # Formal runs must fail visibly rather than suppressing a compiler failure.
    from torch._dynamo import config as dynamo_config
    dynamo_config.suppress_errors = False
    return torch.compile(model, mode=mode, dynamic=False)


def move_batch(batch, device):
    return {name: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def early_stopping_reached(epoch, best_epoch, patience):
    if patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    return patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience


def make_optimizer(model, config):
    groups = [
        {"name": "backbone", "params": [p for name, p in model.named_parameters() if not name.startswith("head.")]},
        {"name": "head", "params": list(model.head.parameters())},
    ]
    return torch.optim.AdamW(groups, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], fused=True)


def make_scheduler(optimizer, config):
    training = config["training"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=training["lr_factor"], patience=training["lr_patience"],
        min_lr=training["minimum_learning_rate"],
    )


def _forward(model, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(batch["images"], batch["time_offsets"], batch["mask"])
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite dP prediction")
    return result


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    predictions = []
    read_summaries = []
    seen = set()
    for host in loader:
        prediction = _forward(model, move_batch(host, device)).cpu().tolist()
        summary = getattr(model, "last_read_summary", None)
        if summary is not None:
            read_summaries.append((len(prediction), summary))
        for position, value in enumerate(prediction):
            record = _parse_int(host["record"][position], "evaluation record")
            if record in seen:
                raise ValueError(f"Duplicate evaluation record {record}")
            seen.add(record)
            target = _parse_float(host["target"][position], "evaluation target")
            predictions.append({
                "record": record, "storm_id": host["storm_id"][position],
                "quality": host["quality"][position], "MOTIS_dP": target,
                "predicted_dP": _parse_float(value, "predicted dP"),
                "error_hPa": _parse_float(value - target, "prediction error"),
            })
    if len(predictions) != len(loader.dataset) or not predictions:
        raise ValueError("Evaluation must cover every record exactly once")
    errors = np.asarray([row["error_hPa"] for row in predictions], dtype=np.float64)
    metrics = {
        "count": len(errors), "rmse": _parse_float(np.sqrt(np.mean(errors**2)), "validation RMSE"),
        "mae": _parse_float(np.mean(np.abs(errors)), "validation MAE"),
        "bias": _parse_float(errors.mean(), "validation bias"),
        "quality_rmse": {},
    }
    for quality in ("Good", "Fair", "Poor"):
        values = np.asarray([row["error_hPa"] for row in predictions if row["quality"] == quality])
        metrics["quality_rmse"][quality] = {
            "count": len(values),
            "rmse": _parse_float(np.sqrt(np.mean(values**2)), f"{quality} RMSE") if len(values) else None,
        }
    metrics.update(prediction_diagnostics(predictions, loader.dataset.samples))
    if read_summaries:
        metrics["reads"] = {
            name: (np.sum([count * np.asarray(summary[name]) for count, summary in read_summaries], axis=0) / len(predictions)).tolist()
            for name in read_summaries[0][1]
        }
    return metrics, predictions


def prediction_diagnostics(predictions, samples):
    """Diagnostic-only groups; never change loss weights or sample selection."""
    targets = np.asarray([row["MOTIS_dP"] for row in predictions], dtype=np.float64)
    predicted = np.asarray([row["predicted_dP"] for row in predictions], dtype=np.float64)
    errors = predicted - targets
    source = {sample["record"]: "FLDK-containing" if "FLDK" in sample["sources"] else "R3-only" for sample in samples}
    group_masks = {
        "dP<60": targets < 60,
        "60<=dP<90": (targets >= 60) & (targets < 90),
        "dP>=90": targets >= 90,
        **{name: np.asarray([source[row["record"]] == name for row in predictions]) for name in ("R3-only", "FLDK-containing")},
    }
    groups = {}
    for name, selected in group_masks.items():
        e = errors[selected]
        groups[name] = {
            "count": len(e), "rmse": np.sqrt(np.mean(e**2)).item() if len(e) else None,
            "mae": np.mean(np.abs(e)).item() if len(e) else None,
            "bias": e.mean().item() if len(e) else None,
        }
    return {
        "prediction_std": predicted.std().item(), "target_std": targets.std().item(),
        "prediction_range": [predicted.min().item(), predicted.max().item()], "groups": groups,
    }


def _atomic_checkpoint(path, state):
    check_space(path, 150_000_000)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_predictions(path, predictions):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    os.replace(temporary, path)


def _tensor_norm(tensors):
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value.detach().float()) for value in tensors]))


def _train_epoch(model, optimizer, loader, device, config, epoch, global_step, target_std=None):
    model.train()
    if target_std is None:
        target_std = model.target_std
    effective = config["training"]["effective_batch_size"]
    if effective < loader.batch_size or effective % loader.batch_size:
        raise ValueError("Effective batch size must be a positive multiple of microbatch size")
    accumulation = effective // loader.batch_size
    iterator = iter(loader)
    total_loss, squared_error, total_samples = 0.0, 0.0, 0
    seen = set()
    telemetry = {group["name"]: {"gradient_norm": [], "clip_multiplier": [], "relative_update": [], "missing_gradient_tensors": []} for group in optimizer.param_groups}
    for start_batch in range(0, len(loader), accumulation):
        check_space(config["output_dir"])
        group_samples = min(effective, len(loader.dataset) - start_batch * loader.batch_size)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(min(accumulation, len(loader) - start_batch)):
            host = next(iterator)
            for record in map(int, host["record"]):
                if record in seen:
                    raise ValueError(f"Training record {record} appeared twice in one epoch")
                seen.add(record)
            batch = move_batch(host, device)
            prediction = _forward(model, batch)
            losses = regression_loss(prediction, batch["target"], target_std)
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("Non-finite training loss")
            (losses.sum() / group_samples).backward()
            total_loss += _parse_float(losses.detach().sum(), "training loss")
            squared_error += _parse_float(
                (prediction.detach() - batch["target"]).float().square().sum(),
                "training squared error",
            )
            total_samples += len(losses)
        snapshots = []
        for group in optimizer.param_groups:
            name = group["name"]
            parameters = group["params"]
            threshold = config["training"][f"{name}_gradient_clip"]
            norm = _parse_float(
                torch.nn.utils.clip_grad_norm_(parameters, threshold, error_if_nonfinite=True),
                f"{name} gradient norm",
            )
            telemetry[name]["gradient_norm"].append(norm)
            telemetry[name]["clip_multiplier"].append(min(1.0, threshold / (norm + 1e-6)))
            telemetry[name]["missing_gradient_tensors"].append(sum(p.grad is None for p in parameters))
            snapshots.append([p.detach().clone() for p in parameters])
        optimizer.step()
        for group, old in zip(optimizer.param_groups, snapshots, strict=True):
            update = _tensor_norm([p.detach() - previous for p, previous in zip(group["params"], old, strict=True)])
            relative = _parse_float(
                update / _tensor_norm(old).clamp_min(1e-12),
                f"{group['name']} relative update",
            )
            telemetry[group["name"]]["relative_update"].append(relative)
        del snapshots
        global_step += 1
        if global_step % 5 == 0:
            print(json.dumps({
                "epoch": epoch + 1, "optimizer_step": global_step,
                "seen_records": total_samples, "unique_records": len(seen),
                "training_loss_normalized_mse": total_loss / total_samples,
                "gradient_norm": {name: values["gradient_norm"][-1] for name, values in telemetry.items()},
                "relative_update": {name: values["relative_update"][-1] for name, values in telemetry.items()},
            }), flush=True)
    expected = {sample["record"] for sample in loader.dataset.samples}
    if total_samples != len(loader.dataset) or seen != expected:
        raise ValueError("Training epoch did not cover every record exactly once")
    gradients = {}
    for name, values in telemetry.items():
        gradients[name] = {
            "norm_mean": _parse_float(np.mean(values["gradient_norm"]), f"{name} mean gradient norm"),
            "norm_p95": _parse_float(np.quantile(values["gradient_norm"], 0.95), f"{name} P95 gradient norm"),
            "clipped_step_fraction": _parse_float(
                np.mean(np.asarray(values["clip_multiplier"]) < 1),
                f"{name} clipped-step fraction",
            ),
            "clip_multiplier_mean": _parse_float(
                np.mean(values["clip_multiplier"]), f"{name} mean clip multiplier"
            ),
            "relative_update_mean": _parse_float(
                np.mean(values["relative_update"]), f"{name} mean relative update"
            ),
            "missing_gradient_tensors_max": max(values["missing_gradient_tensors"]),
        }
    return {
        "training_loss": total_loss / total_samples,
        "online_augmented_rmse": math.sqrt(squared_error / total_samples),
        "sampling": {"unit": "record", "replacement": False, "weight": 1.0, "count": total_samples, "unique": len(seen)},
        "gradients": gradients,
    }, global_step


def train(config, run_dir=None, resume=None):
    """Record-equal training; follow AGENTS.md for conditional sandbox access."""
    training = config["training"]
    if training.get("torch_compile") is not True:
        raise ValueError("Formal training requires training.torch_compile=true; eager is diagnostic-only")
    index = load_cache(config)
    if training["batch_size"] < 1 or training["effective_batch_size"] % training["batch_size"]:
        raise ValueError("Microbatch must be a positive divisor of effective batch size")
    config_id, code_id = configuration_fingerprint(config), code_fingerprint(config)
    architecture = model_architecture(config["model"])
    early_stop_patience = _parse_int(training.get("early_stopping_patience", 0), "early_stopping_patience")
    early_stopping_reached(0, 0, early_stop_patience)
    check_space(config["output_dir"], 400_000_000)
    state = None
    if resume:
        checkpoint_path = Path(resume).resolve()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("kind") != CHECKPOINT_KIND or state.get("architecture") != architecture:
            raise ValueError(f"Checkpoint requires matching architecture {architecture}; incompatible weights cannot be resumed")
        if any(state[key] != expected for key, expected in (("data_fingerprint", index["fingerprint"]), ("configuration_fingerprint", config_id), ("code_fingerprint", code_id))):
            raise ValueError("Checkpoint data/configuration/code fingerprint mismatch")
        directory = checkpoint_path.parent
        if run_dir and Path(run_dir).resolve() != directory:
            raise ValueError("Resume must preserve its original run directory")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = architecture.split("/")[0].replace("-", "_")
        directory = Path(run_dir).resolve() if run_dir else Path(config["output_dir"]) / f"{prefix}_{stamp}"
    device, environment = initialize_device(config)
    seed_everything(training["seed"])
    target = index["target_normalization"]
    model = build_model(config["model"], target["mean_hpa"], target["std_hpa"]).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    start_epoch, global_step, best_epoch, best_rmse = 0, 0, 0, math.inf
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = state["epoch_completed"], state["global_step"]
        best_epoch, best_rmse = state["best_epoch"], state["best_rmse"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    else:
        directory.mkdir(parents=True, exist_ok=False)
    forward_model = compile_model(model, config)
    environment["torch_compile"] = bool(training["torch_compile"])
    if training["torch_compile"]:
        environment["torch_compile_mode"] = training.get("compile_mode", "default")
    runtime = copy.deepcopy(config)
    runtime["output_dir"] = str(directory)
    run_state = {
        "status": "running", "formal_training_started": True, "architecture": architecture,
        "early_stopping_patience": early_stop_patience,
        "data_fingerprint": index["fingerprint"], "configuration_fingerprint": config_id,
        "code_fingerprint": code_id, "configuration": runtime, "environment": environment,
        "loss": "record_equal_normalized_mse", "record_weight": 1.0,
        "target_normalization": target, "parameter_count": sum(p.numel() for p in model.parameters()),
        "resume": str(resume) if resume else None, "epoch_completed": start_epoch,
    }
    write_json(directory / "run.json", run_state)
    validation = make_loader(config, index, "validation", training["batch_size"], augment=False)
    training_evaluation = make_loader(config, index, "train", training["batch_size"], augment=False)
    try:
        for epoch in range(start_epoch, training["epochs"]):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            current_lr = optimizer.param_groups[0]["lr"]
            loader = make_loader(runtime, index, "train", training["batch_size"], epoch)
            progress, global_step = _train_epoch(
                forward_model, optimizer, loader, device, runtime, epoch, global_step, model.target_std
            )
            del loader
            training_metrics, training_predictions = validate(forward_model, training_evaluation, device)
            metrics, predictions = validate(forward_model, validation, device)
            improved = metrics["rmse"] < best_rmse
            if improved:
                best_rmse, best_epoch = metrics["rmse"], epoch + 1
            scheduler.step(metrics["rmse"])
            state = {
                "kind": CHECKPOINT_KIND, "architecture": architecture,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch_completed": epoch + 1, "global_step": global_step, "best_rmse": best_rmse, "best_epoch": best_epoch,
                "model_config": config["model"], "normalization": index["normalization"], "target_normalization": target,
                "microbatch": training["batch_size"], "data_fingerprint": index["fingerprint"],
                "configuration_fingerprint": config_id, "code_fingerprint": code_id,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            if improved:
                _atomic_checkpoint(directory / "best.pt", state)
                _write_predictions(directory / "best_validation.csv", predictions)
                _write_predictions(directory / "best_train.csv", training_predictions)
                write_json(directory / "best_metrics.json", {"epoch": best_epoch, "training": training_metrics, "validation": metrics})
            _atomic_checkpoint(directory / "last.pt", state)
            history = {
                "epoch": epoch + 1, "global_step": global_step, "learning_rate": current_lr,
                **progress, "training_evaluation": training_metrics, "validation": metrics,
                "elapsed_seconds": time.monotonic() - started, "best_epoch": best_epoch,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            with (directory / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(history, ensure_ascii=False, allow_nan=False) + "\n")
            stop = early_stopping_reached(epoch + 1, best_epoch, early_stop_patience)
            run_state.update(epoch_completed=epoch + 1, best_rmse=best_rmse, best_epoch=best_epoch)
            if stop:
                run_state.update(status="early_stopped", stop_reason="validation_rmse_patience")
            write_json(directory / "run.json", run_state)
            print(json.dumps(history, ensure_ascii=False), flush=True)
            if stop:
                break
        if run_state["status"] == "running":
            run_state["status"] = "completed"
        write_json(directory / "run.json", run_state)
    except BaseException as error:
        run_state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        write_json(directory / "run.json", run_state)
        (directory / "failure.txt").write_text(traceback.format_exc())
        raise
