"""Single-seed record-level W1 fits using catalog dP and radiance features."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import _normalize, record_folds, scores, summarize_time
from h9dp.w1 import common_kernel, objective, quality_covariance

GROUPS = ("warm_radiance", "eye", "wall", "contrast", "environment")
QUALITY = {"Good": 0, "Fair": 1, "Poor": 2}
STEPS = 140


def fold_features(groups, times, train, held):
    valid = np.isfinite(groups["eye"]).any(-1)
    kernels, priors, names = [], [], []
    environment = []
    for family, values in groups.items():
        summaries = summarize_time(values, times, valid)
        for stat, values in summaries.items():
            a, b = _normalize(values[train], values[held])
            if not a.shape[1]:
                continue
            values = torch.as_tensor(np.concatenate((a, b)), device="cuda", dtype=torch.float64)
            linear = values @ values.T / values.size(1)
            norm = linear.diag()
            distance = (norm[:, None] + norm[None] - 2 * linear).clamp_min(0)
            radius = (3 * distance + 1e-8).sqrt()
            local = (1 + radius) * (-radius).exp()
            kernels.extend((linear, local))
            names.extend((f"{family}_{stat}_linear", f"{family}_{stat}_matern"))
            mass = {"recent": 0.11, "latest": 0.05, "trend": 0.025}[stat]
            priors.extend((mass, mass * 0.7))
            if family in ("eye", "warm_radiance", "contrast") and stat == "recent":
                knots = values.new_tensor([-3., -2., -1., 0., 1., 2., 3.])
                bump = (-0.5 * ((values[..., None] - knots) / 0.8).square()).exp()
                vectors = torch.nn.functional.normalize(bump, dim=-1).flatten(1)
                kernels.append(vectors @ vectors.T / values.size(1))
                names.append(f"{family}_{stat}_spline")
                priors.append(0.075)
            if family in ("wall", "environment", "eye") and stat == "recent":
                environment.append(local + 0.22 * linear)
    return torch.stack(kernels), torch.stack(environment).mean(0), np.log(priors), names


def fit(basis, nuisance, priors, y, quality):
    n = len(y)
    center, spread = float(y.mean()), max(float(y.std()), 1.)
    target = basis.new_tensor((y - center) / spread)
    q = torch.tensor([QUALITY[tag] for tag in quality], device=basis.device)
    weights = basis.new_tensor([1.0, 0.34, 0.18])[q]
    weights = weights / weights.mean()
    prior = basis.new_tensor(priors)
    log_mass = torch.nn.Parameter(prior.clone())
    settings = torch.nn.Parameter(basis.new_tensor([-2.7, -2.1, -1.6, -2.8, -3.]))
    optimizer = torch.optim.Adam([log_mass, settings], lr=0.045)
    compiled = torch.compile(objective, fullgraph=True, dynamic=False)
    inputs = basis[:, :n, :n].contiguous(), nuisance[:n, :n].contiguous(), target, q, weights, prior

    step_times = []
    for _ in range(STEPS):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(log_mass, settings, *inputs)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.monotonic() - start)
    if not torch.isfinite(loss):
        raise ArithmeticError("Nonfinite W1 training loss")
    with torch.no_grad():
        shared = common_kernel(log_mass, basis)
        covariance = quality_covariance(shared[:n, :n], inputs[1], settings, q)
        dual = torch.cholesky_solve(target[:, None], torch.linalg.cholesky(covariance))
        prediction = ((shared[n:, :n] @ dual).flatten() * spread + center).cpu().numpy()
    state = {"log_mass": log_mass.detach().cpu(), "settings": settings.detach().cpu(), "dual": dual.cpu(),
             "target_mean": center, "target_std": spread}
    return prediction, state, {"first_s": step_times[0], "steady_median_s": float(np.median(step_times[1:])),
                               "training_s": sum(step_times)}


def evaluate(source: Path, features: Path, output: Path):
    raw = (source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    source_digest = hashlib.sha256(raw).hexdigest()
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    with np.load(features) as data:
        if (str(data["mode"]) != "radiance" or str(data["source_sha256"]) != source_digest
                or not np.array_equal(data["record"], ids)
                or not np.array_equal(data["time"], times, equal_nan=True)):
            raise ValueError("W1 features and catalog-verified input metadata differ")
        groups = {key: data[key] for key in GROUPS}

    if not torch.cuda.is_available():
        raise RuntimeError("W1 requires CUDA for compiled training")
    output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(output.parent / "triton"))
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    assignment = record_folds(len(ids), 5, 1)
    target = np.array([catalog[i]["dP"] for i in ids])
    quality = np.array([catalog[i]["quality"] for i in ids])
    pred = np.full(len(ids), np.nan)
    run = {"seed": 1, "steps": STEPS, "folds": 5, "source_meta_sha256": source_digest,
           "catalog_sha256": catalog_digest, "features": str(features.resolve()),
           "records": len(ids), "frames": int(np.isfinite(groups["eye"]).any(-1).sum()),
           "torch": torch.__version__, "device": torch.cuda.get_device_name(),
           "training": "CUDA FP64 torch.compile(fullgraph=True)", "prediction": "eager CUDA FP64"}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    trace = []
    with (output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction"])
        for fold in range(5):
            train = np.flatnonzero(assignment != fold)
            held = np.flatnonzero(assignment == fold)
            start = time.monotonic()
            basis, nuisance, priors, names = fold_features(groups, times, train, held)
            torch.cuda.synchronize()
            preparation = time.monotonic() - start
            pred[held], state, timing = fit(basis, nuisance, priors, target[train], quality[train])
            trace.append({"fold": fold, **timing, "kernel_setup_s": preparation})
            torch.save({**state, "names": names, "train_records": ids[train].tolist()}, output / f"fold{fold}.pt")
            writer.writerows((1, fold, ids[i], quality[i], target[i], pred[i]) for i in held)
            stream.flush()
            print(trace[-1], flush=True)
    result = {**run, "trace": trace, "metrics": scores(target, pred, quality)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source, args.features, args.output)
