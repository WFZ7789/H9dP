"""Fit V1 using train-fold weighting of polar structure, cloud motion and scan similarity."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.r1 import _normalize, record_folds, scores, summarize_time


def embedding(frames, t, valid, train, order):
    # Only training scans determine the radiometric scale of a sector or radius.
    n = len(train)
    fit = frames[train].reshape(-1, frames.shape[-1])
    a, b = _normalize(fit, frames[order[n:]].reshape(-1, frames.shape[-1]))
    x = np.concatenate((a, b)).reshape(len(order) * 6, -1)
    x = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    dot = x @ x.T / x.shape[1]
    d = (dot.diag()[:, None] + dot.diag()[None, :] - 2 * dot).clamp_min(0)
    weights = np.where(valid, np.exp((t - np.max(np.where(valid, t, -np.inf), axis=1)[:, None]) / 180), 0)
    weights /= weights.sum(1, keepdims=True)
    w = torch.as_tensor(weights[order], dtype=x.dtype, device=x.device)
    out = []
    for width in (0.035, 0.14, 0.55):
        k = (-d / (2 * width)).exp().reshape(len(order), 6, len(order), 6)
        gram = torch.einsum("if,ifjh,jh->ij", w, k, w)
        scale = gram.diag().clamp_min(1e-6).rsqrt()
        out.append((gram * scale[:, None] * scale[None]).double())
    return out


def bank(groups, times, train, held):
    order = np.r_[train, held]
    valid = np.isfinite(groups["eye"]).any(-1)
    basis, names = [], []
    for name, frames in groups.items():
        stats = summarize_time(frames, times, valid)
        for stat, features in stats.items():
            a, b = _normalize(features[train], features[held])
            if not a.shape[1]:
                continue
            x = torch.as_tensor(np.concatenate((a, b)), dtype=torch.float64, device="cuda")
            dot = x @ x.T / x.shape[1]
            d = (dot.diag()[:, None] + dot.diag()[None, :] - 2 * dot).clamp_min(0)
            r = (3 * d + 1e-12).sqrt()
            basis.extend((dot, (1 + r) * (-r).exp()))
            names.extend((name + "_" + stat + "_linear", name + "_" + stat + "_matern"))
            if name in ("eye", "wall", "relief") and stat == "recent":
                for width in (0.7, 1.8):
                    knots = torch.linspace(-3.0, 3.0, 9, dtype=x.dtype, device=x.device)
                    phi = (-0.5 * ((x[..., None] - knots) / width).square()).exp()
                    phi = torch.nn.functional.normalize(phi, dim=-1).flatten(1)
                    basis.append(phi @ phi.T / x.shape[1])
                    names.append(f"{name}_spline_{width}")
        if name in ("sector", "waves"):
            basis.extend(embedding(frames, times, valid, train, order))
            names.extend(f"{name}_sequence_{i}" for i in range(3))
    return torch.stack(basis), names


def objective(log_mass, noise, basis, y, q, w, prior):
    gram = torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8
    cov = gram + torch.diag(noise.exp()[q] + 0.002)
    chol, _ = torch.linalg.cholesky_ex(cov)
    inv = torch.cholesky_inverse(chol)
    dual = inv @ y
    loss = ((dual / inv.diag()).square() * w).mean()
    evidence = (y @ dual + 2 * chol.diag().log().sum()) / len(y)
    return loss + 0.12 * evidence + 0.003 * (log_mass - prior).square().mean()


def fit(basis, names, y, quality):
    n = len(y)
    device = basis.device
    q = torch.tensor([{"Good": 0, "Fair": 1, "Poor": 2}[s] for s in quality], device=device)
    importance = torch.tensor([1.0, 0.38, 0.22], device=device, dtype=basis.dtype)[q]
    target = torch.tensor((y - y.mean()) / y.std(), device=device)
    prior = torch.tensor(
        [np.log(0.12 if "_sequence_" in k else 0.065 if "recent" in k else 0.035) for k in names],
        device=device,
    )
    mass = torch.nn.Parameter(prior.clone())
    noise = torch.nn.Parameter(torch.tensor([-2.8, -1.9, -1.4], device=device, dtype=basis.dtype))
    objective_step = torch.compile(objective, fullgraph=True, dynamic=False)
    opt = torch.optim.Adam([mass, noise], lr=0.045)
    block = basis[:, :n, :n].contiguous()
    step_times = []
    for i in range(110):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = objective_step(mass, noise, block, target, q, importance / importance.mean(), prior)
        loss.backward()
        opt.step()
        step_times.append(0.0)
        torch.cuda.synchronize()
        step_times[i] = time.monotonic() - start
    with torch.no_grad():
        kernel = torch.einsum("g,gij->ij", mass.exp(), basis) + 8
        chol = torch.linalg.cholesky(kernel[:n, :n] + torch.diag(noise.exp()[q] + 0.002))
        dual = torch.cholesky_solve(target[:, None], chol)
        pred = kernel[n:, :n] @ dual * y.std() + y.mean()
    state = {
        "names": names,
        "mass": mass.detach().cpu(),
        "noise": noise.detach().cpu(),
        "dual": dual.cpu(),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    }
    return pred.flatten().cpu().numpy(), state, [step_times[0], float(np.median(step_times[1:]))]


def evaluate(source, features, output):
    output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(output.parent / "triton"))
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(1)
    raw = (source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    with np.load(features) as f:
        ids, times = f["record"], f["time"]
        groups = {k: f[k] for k in ("eye", "wall", "relief", "waves", "motion", "sector")}
        if (
            str(f["source_sha256"]) != hashlib.sha256(raw).hexdigest()
            or not np.array_equal(ids, [r["record"] for r in rows])
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Feature provenance differs from source")
    y, quality = np.array([r["dP"] for r in rows]), np.array([r["quality"] for r in rows])
    fold = record_folds(len(rows), 5, 1)
    pred = np.full_like(y, np.nan)
    timing = []
    with (output / "predictions.csv").open("x", newline="") as stream:
        write = csv.writer(stream)
        write.writerow(["seed", "fold", "record", "quality", "target", "prediction"])
        for k in range(5):
            train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            kernels, names = bank(groups, times, train, held)
            pred[held], state, duration = fit(kernels, names, y[train], quality[train])
            torch.save(dict(train_ids=ids[train].tolist(), **state), output / f"fold{k}.pt")
            timing.append(duration)
            write.writerows((1, k, ids[i], quality[i], y[i], pred[i]) for i in held)
            stream.flush()
            print(f"fold {k} time {duration}", flush=True)
    result = {
        "seed": 1,
        "metrics": scores(y, pred, quality),
        "first_and_steady_s": timing,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "features": str(features.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "features", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    evaluate(a.source, a.features, a.output)
