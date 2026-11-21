"""Single-seed five-fold co-kriging of pressure and auxiliary image measurements."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.e1_joint import aux_loss, condition_aux, pressure_loss
from h9dp.r1 import record_folds, scores
from scripts.v1 import bank


def optimize(function, parameters, inputs, steps):
    compiled = torch.compile(function, fullgraph=True, dynamic=False)
    optimizer = torch.optim.Adam(parameters, lr=0.045)
    timings = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(*parameters, *inputs)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic() - start)
    if not torch.isfinite(loss):
        raise ArithmeticError("Non-finite terminal training objective")
    return [timings[0], float(np.median(timings[1:]))]


def fit(basis, names, y, labels, quality):
    n = len(y)
    device = basis.device

    def floating(values):
        return torch.as_tensor(values, device=device, dtype=basis.dtype)

    prior = floating([np.log(0.075 if "recent" in k or "_sequence_" in k else 0.025) for k in names])
    z = floating(((labels - labels.mean(0)) / labels.std(0).clip(1e-6)).T.copy())
    q = torch.tensor([{"Good": 0, "Fair": 1, "Poor": 2}[v] for v in quality], device=device)
    weight = floating([1.0, 0.42, 0.22])[q]
    weight = weight / weight.mean()
    target = floating((y - y.mean()) / y.std())
    block = basis[:, :n, :n].contiguous()
    mass = torch.nn.Parameter(prior.repeat(len(z), 1))
    noise = torch.nn.Parameter(floating([-2.7, -2.0, -1.6]).repeat(len(z), 1))
    aux_times = optimize(aux_loss, [mass, noise], (block, z, q, weight, prior), 90)
    with torch.no_grad():
        smoother, signal, residual, diag_inv, zdual = condition_aux(basis, mass, noise, z, q)
    pmass = torch.nn.Parameter(prior.clone())
    pnoise = torch.nn.Parameter(floating([-2.9, -1.9, -1.4]))
    beta = torch.nn.Parameter(
        torch.linalg.solve(z @ z.T + 0.05 * n * torch.eye(len(z), device=device), z @ target)
    )
    times = optimize(
        pressure_loss,
        [pmass, pnoise, beta],
        (
            block,
            smoother.contiguous(),
            signal[:, :n].contiguous(),
            residual[:, :n, :].contiguous(),
            diag_inv.contiguous(),
            zdual,
            target,
            q,
            weight,
            prior,
        ),
        110,
    )
    with torch.no_grad():
        cov = (
            torch.einsum("g,gij->ij", pmass.exp(), basis[:, :, :n])
            + 8
            + torch.einsum("a,aij->ij", beta.square(), residual)
        )
        mean = beta @ signal
        chol = torch.linalg.cholesky(cov[:n] + torch.diag(pnoise.exp()[q] + 0.002))
        dual = torch.cholesky_solve((target - mean[:n])[:, None], chol)
        pred = (mean[n:] + (cov[n:] @ dual).flatten()) * y.std() + y.mean()
    state = {
        key: value.detach().cpu()
        for key, value in {
            "aux_mass": mass,
            "aux_noise": noise,
            "pressure_mass": pmass,
            "pressure_noise": pnoise,
            "beta": beta,
            "dual": dual,
        }.items()
    }
    return pred.cpu().numpy(), state, {"aux": aux_times, "pressure": times}


def evaluate(args):
    raw = (args.source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    ids = np.array([r["record"] for r in rows])
    digest = hashlib.sha256(raw).hexdigest()
    groups = {}
    for file, keys in (
        (args.polar, ("eye", "wall", "relief", "waves", "sector", "motion")),
        (args.isotherms, ("shape", "spectrum", "peaks", "environment", "excess")),
    ):
        with np.load(file) as f:
            if str(f["source_sha256"]) != digest or not np.array_equal(ids, f["record"]):
                raise ValueError("Mismatched feature cache")
            if not np.array_equal(f["time"], [r["times"] for r in rows], equal_nan=True):
                raise ValueError("Mismatched frame slots")
            groups.update({k: f[k] for k in keys})
            times = f["time"]
    cat_raw = args.catalog.read_bytes()
    catalog = json.loads(cat_raw)["records"]
    if any(
        (catalog[i]["name"], catalog[i]["time"]["year"], catalog[i]["dP"]) != (r["name"], r["year"], r["dP"])
        for i, r in zip(ids, rows, strict=True)
    ):
        raise ValueError("Catalog records differ from the source")
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton"))
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    fold = record_folds(len(ids), 5, 1)
    y = np.array([r["dP"] for r in rows])
    quality = np.array([r["quality"] for r in rows])
    prediction = np.full(len(ids), np.nan)
    trace = []
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction"])
        for k in range(5):
            train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            aux = np.array(
                [
                    (r["b27"], r["b31"], np.log1p(r["pixels"]), r["t_cdo"])
                    for r in (catalog[i] for i in ids[train])
                ]
            )
            basis, names = bank(groups, times, train, held)
            print(f"fold {k} kernels {len(names)}", flush=True)
            prediction[held], state, timing = fit(basis, names, y[train], aux, quality[train])
            torch.save(
                {**state, "train_ids": ids[train].tolist(), "names": names}, args.output / f"fold{k}.pt"
            )
            trace.append(timing)
            writer.writerows((1, k, ids[i], quality[i], y[i], prediction[i]) for i in held)
            stream.flush()
            print(timing, flush=True)
    result = {
        "seed": 1,
        "metrics": scores(y, prediction, quality),
        "first_and_steady_s": trace,
        "torch": torch.__version__,
        "source_sha256": digest,
        "catalog_sha256": hashlib.sha256(cat_raw).hexdigest(),
        "device": torch.cuda.get_device_name(),
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(result["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "polar", "isotherms", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--catalog", default=Path("MOTIS_cyclone_records.json"), type=Path)
    evaluate(p.parse_args())
