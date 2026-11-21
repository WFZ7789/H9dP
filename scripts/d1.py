"""Record-level single-seed development evaluation of D1."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.d1_metric import coordinates, kernel_parts, solve_stage, update_metric
from h9dp.r1 import record_folds, scores

GROUPS = ("core", "cloud", "contrast", "scene")


def fit(train_x, held_x, tasks, quality):
    mean, std = tasks.mean(0), tasks.std(0).clip(1e-6)
    y = torch.as_tensor((tasks - mean) / std, device=train_x.device)
    var = torch.as_tensor(
        np.select((quality == "Good", quality == "Fair"), (0.055, 0.13), default=0.22), device=train_x.device
    )
    metric = torch.eye(train_x.size(-1), device=train_x.device, dtype=y.dtype).repeat(len(GROUPS), 1, 1)
    task_weight = torch.tensor([1.0, 0.16, 0.16, 0.10, 0.10], device=train_x.device, dtype=y.dtype)
    step = torch.compile(solve_stage, fullgraph=True, dynamic=False)
    update = torch.compile(update_metric, fullgraph=True, dynamic=False)
    timings = []
    for i in range(3):
        start = time.monotonic()
        dual, grad = step(train_x, y, var, metric)
        if i < 2:
            metric = update(train_x, grad, 1 / var, task_weight)
        torch.cuda.synchronize()
        timings.append(time.monotonic() - start)
    gram, _, _ = kernel_parts(held_x, train_x, metric)
    prediction = (gram @ dual[:, 0]).cpu().numpy() * std[0] + mean[0]
    return (
        prediction,
        timings,
        {
            "metric": metric.cpu(),
            "dual": dual.cpu(),
            "train_x": train_x.cpu(),
            "target_mean": mean.tolist(),
            "target_std": std.tolist(),
        },
    )


def evaluate(features, source, catalog, output):
    if not torch.cuda.is_available():
        raise RuntimeError("D1 formal fit requires CUDA")
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(output.parent / "triton"))
    raw = (source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    with np.load(features) as f:
        ids = f["record"]
        if (
            str(f["source_sha256"]) != hashlib.sha256(raw).hexdigest()
            or not np.array_equal(ids, [r["record"] for r in rows])
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("D1 descriptors differ from the source")
        groups = [f[k] for k in GROUPS]
        frames = int(f["valid"].sum())
    cat_bytes = catalog.read_bytes()
    entries = json.loads(cat_bytes)["records"]
    y = np.array([r["dP"] for r in rows])
    quality = np.array([r["quality"] for r in rows])
    for r in rows:
        ref = entries[r["record"]]
        if (ref["name"], ref["quality"], ref["dP"], ref["time"]["year"]) != (
            r["name"],
            r["quality"],
            r["dP"],
            r["year"],
        ):
            raise ValueError("Current MOTIS catalog and record cache differ")
    fold = record_folds(len(ids), 5, 1)
    pred = np.full(len(ids), np.nan)
    trace = []
    run = {
        "seed": 1,
        "folds": 5,
        "records": len(ids),
        "frames": frames,
        "stages": 3,
        "rank": 56,
        "features": str(features.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "catalog_sha256": hashlib.sha256(cat_bytes).hexdigest(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "execution": "CUDA FP64 torch.compile fullgraph; eager prediction",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2))
    with (output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction"])
        for k in range(5):
            train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            labels = []
            for i in train:
                v = entries[ids[i]]
                labels.append((y[i], v["b27"], v["b31"], np.log1p(v["pixels"]), v["t_cdo"]))
            tasks = np.array(labels)
            if not np.isfinite(tasks).all():
                raise ValueError("Incomplete D1 training labels")
            x = torch.as_tensor(np.stack([coordinates(g, train, held) for g in groups]), device="cuda")
            estimate, times, state = fit(
                x[:, : len(train)].contiguous(), x[:, len(train) :], tasks, quality[train]
            )
            if not np.isfinite(estimate).all():
                raise ArithmeticError("Nonfinite D1 predictions")
            torch.save(dict(**state, train_records=ids[train].tolist()), output / f"fold{k}.pt")
            pred[held] = estimate
            writer.writerows((1, k, ids[i], quality[i], y[i], pred[i]) for i in held)
            stream.flush()
            trace.append(times)
            print(f"fold={k} n={len(held)} stage_seconds={times}", flush=True)
    metrics = scores(y, pred, quality)
    (output / "metrics.json").write_text(json.dumps({"metrics": metrics, "stage_seconds": trace}, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--catalog", type=Path, default=Path("MOTIS_cyclone_records.json"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    evaluate(a.features, a.source, a.catalog, a.output)
