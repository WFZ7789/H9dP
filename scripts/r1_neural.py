#!/usr/bin/env python3
"""Train R1 with torch.compile/BF16 and export out-of-fold development predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from h9dp.r1 import QUALITY_WEIGHT, record_folds, scores
from h9dp.r1_net import R1Net
from scripts.r1_kernel import feature_groups, load_cache, nested_predict


def fit_normalizer(x: np.ndarray, available: np.ndarray):
    observed = x[available]
    valid = np.isfinite(observed)
    count = valid.sum(axis=0).clip(1)
    avg = np.where(valid, observed, 0.0).sum(axis=0, dtype=np.float64) / count
    std = np.sqrt((np.where(valid, observed - avg, 0.0) ** 2).sum(axis=0) / count).clip(0.10)
    return avg.astype(np.float32), std.astype(np.float32)


def standardized(x: np.ndarray, average: np.ndarray, scale: np.ndarray):
    return np.where(np.isfinite(x), ((x - average) / scale).clip(-6, 6), 0.0)


def learning_rate(epoch: int, epochs: int, peak: float):
    warmup = min(5, epochs)
    if epoch < warmup:
        return peak * (0.20 + 0.80 * (epoch + 1) / warmup)
    phase = (epoch - warmup + 1) / (epochs - warmup)
    return peak * (0.08 + 0.92 * (1 + math.cos(math.pi * phase)) / 2)


def batches(train: np.ndarray, size: int, generator: np.random.Generator):
    order = generator.permutation(train)
    for start in range(0, len(order), size):
        rows = order[start:start + size]
        count = len(rows)
        rows = np.pad(rows, (0, size - count), mode="edge")
        yield rows, np.arange(size) < count


def fit_fold(cache, views, times, valid, raw_profile, aux, train, held, args, seed, initial):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    location, scale = fit_normalizer(raw_profile[train], cache.data["valid"][train])
    profile = torch.as_tensor(standardized(raw_profile, location, scale), device="cuda")
    normal_target = np.asarray(cache.y - initial, dtype=np.float32)
    mean = float(normal_target[train].mean())
    std = max(float(normal_target[train].std()), 3.0)
    target = torch.as_tensor((normal_target - mean) / std, device="cuda")
    observed = np.isfinite(aux)
    loc_aux, scale_aux = fit_normalizer(aux[train], np.ones(len(train), dtype=bool))
    params = torch.as_tensor(standardized(aux, loc_aux, scale_aux), device="cuda")
    aux_valid = torch.as_tensor(observed, device="cuda")
    weights = torch.as_tensor([QUALITY_WEIGHT[q] for q in cache.quality], dtype=torch.float32, device="cuda")
    model = R1Net(raw_profile.shape[-1], args.input).cuda()
    torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
    shadow = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(31 / 32)).eval().requires_grad_(False)
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.025, fused=True)
    first_step = None
    start = time.monotonic()
    timings = []
    for epoch in range(args.epochs):
        epoch_started = time.monotonic()
        epoch_loss = torch.zeros((), device="cuda")
        optimizer.param_groups[0]["lr"] = learning_rate(epoch, args.epochs, args.lr)
        for rows, active in batches(train, args.batch_size, rng):
            step_start = time.monotonic()
            idx = torch.as_tensor(rows, device="cuda")
            active_t = torch.as_tensor(active, device="cuda")
            view = views[idx]
            if args.input != "profile":
                # Rotate both views by the same element of the D4 group; centers and label are unchanged.
                if rng.integers(2):
                    view = view.transpose(-1, -2)
                if rng.integers(2):
                    view = view.flip(-1)
                if rng.integers(2):
                    view = view.flip(-2)
                view = view.contiguous()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                estimated = compiled(view, profile[idx], times[idx], valid[idx]).float()
            w = weights[idx] * active_t
            primary = ((estimated[:, 0] - target[idx]).square() * w).sum() / w.sum()
            valid_param = aux_valid[idx] * w[:, None]
            secondary = ((estimated[:, 1:] - params[idx]).square() * valid_param).sum()
            loss = primary + args.aux * secondary / valid_param.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            shadow.update_parameters(model)
            epoch_loss = loss.detach()
            if first_step is None:
                torch.cuda.synchronize()
                first_step = time.monotonic() - step_start
                print(f"first optimizer step with compile/cache loading: {first_step:.3f}s", flush=True)
        torch.cuda.synchronize()
        timings.append(time.monotonic() - epoch_started)
        if (epoch + 1) % 16 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch+1}/{args.epochs} loss={float(epoch_loss):.5f} epoch_s={timings[-1]:.3f}", flush=True)
    prediction = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in np.array_split(held, math.ceil(len(held) / args.batch_size)):
            k = torch.as_tensor(block, device="cuda")
            prediction.append(shadow(views[k], profile[k], times[k], valid[k])[:, 0].float().cpu().numpy())
    forecast = np.concatenate(prediction) * std + mean + initial[held]
    if not np.isfinite(forecast).all():
        raise RuntimeError("Nonfinite prediction from compiled training")
    checkpoint = {"model": shadow.module.cpu().state_dict(), "profile_location": torch.from_numpy(location),
                  "profile_scale": torch.from_numpy(scale), "target_mean": mean, "target_scale": std,
                  "input": args.input, "target": args.target, "feature_dim": raw_profile.shape[-1],
                  "train_records": torch.as_tensor(cache.data["record"][train]),
                  "held_records": torch.as_tensor(cache.data["record"][held])}
    perf = {"cold_first_step_s": first_step, "epoch_s": timings,
            "median_steady_epoch_s": float(np.median(timings[1:] or timings)),
            "duration_s": time.monotonic() - start, "params": sum(p.numel() for p in shadow.parameters())}
    return forecast, checkpoint, perf


def train(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required; no eager fallback")
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = False
    # Early failure if the configured run/cache location is unavailable.
    cache = load_cache(args.features)
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton"))
    groups = feature_groups(cache.data) if args.target == "kernel" else None
    raw_profile = np.concatenate([cache.data[k] for k in ("regions", "profile", "structure")], axis=-1)
    views = torch.as_tensor(np.load(cache.root / "views.npy", allow_pickle=False), device="cuda")
    times = torch.as_tensor(cache.data["time"], dtype=torch.float32, device="cuda")
    valid = torch.as_tensor(cache.data["valid"], device="cuda")
    aux = np.asarray([r["params"] if r["params"] is not None else [np.nan] * 5
                      for r in cache.rows], dtype=np.float32)
    if aux.shape != (len(cache.rows), 5):
        raise ValueError("Expected five optional auxiliary parameters per record")
    metadata = {"schema": "r1-neural-development/v1", "features": str(cache.root.resolve()),
                "source_meta_sha256": cache.source_digest, "seeds": args.seeds, "folds": args.folds,
                "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr, "input": args.input,
                "target": args.target, "aux_weight": args.aux, "torch": torch.__version__,
                "device": torch.cuda.get_device_name(), "execution": "torch.compile(fullgraph=True) + CUDA BF16",
                "evaluation": "eager CUDA BF16, fixed terminal EMA"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    res = {}
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "neural"])
        for seed in args.seeds:
            assigned = record_folds(len(cache.rows), args.folds, seed)
            prediction = np.full(len(assigned), np.nan)
            fold_reports = []
            for fold in range(args.folds):
                held = np.flatnonzero(assigned == fold)
                train_rows = np.flatnonzero(assigned != fold)
                print(f"seed={seed} fold={fold} train={len(train_rows)} evaluate={len(held)}", flush=True)
                initial = np.zeros(len(assigned))
                if groups is not None:
                    baseline, oof, _ = nested_predict(groups, cache.y, cache.quality, train_rows, held,
                                                       seed, 3)
                    initial[train_rows], initial[held] = oof["structural"], baseline["structural"]
                fold_prediction, state, timing = fit_fold(cache, views, times, valid, raw_profile, aux,
                                                         train_rows, held, args, seed, initial)
                prediction[held] = fold_prediction
                torch.save(state, args.output / f"seed{seed}_fold{fold}.pt")
                metrics = scores(cache.y[held], prediction[held], cache.quality[held])
                print(json.dumps(metrics), flush=True)
                fold_reports.append({"metrics": metrics, "timing": timing})
                for i in held:
                    writer.writerow([seed, fold, cache.data["record"][i], cache.quality[i], cache.y[i], prediction[i]])
                stream.flush()
            res[str(seed)] = {"folds": fold_reports, "metrics": scores(cache.y, prediction, cache.quality)}
            print(json.dumps(res[str(seed)]["metrics"]), flush=True)
            (args.output / "metrics.json").write_text(json.dumps({**metadata, "results": res}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs=1, default=[1])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0016)
    parser.add_argument("--aux", type=float, default=0.20)
    parser.add_argument("--input", choices=("fused", "pixels", "profile"), default="fused")
    parser.add_argument("--target", choices=("direct", "kernel"), default="direct")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.aux < 0:
        parser.error("Training needs positive epochs, batch size and LR, and nonnegative auxiliary weight")
    train(args)


if __name__ == "__main__":
    main()
