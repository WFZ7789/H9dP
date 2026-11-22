"""Fold-local estimation of CDO temperature and CDO-conditioned eye pressure."""

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
from h9dp.r1 import record_folds, scores
from scripts.v1 import bank, objective

EYE = ("eye", "relief")
CLOUD = ("ring", "ring_sectors", "cloud_cover", "eye_cdo_relief")
QUALITY = {"Good": 0, "Fair": 1, "Poor": 2}
SEED = 1


def fit_posterior(basis, names, target, q, importance, steps, lr):
    if steps < 1:
        raise ValueError("At least one optimization step is required")
    n = len(target)
    center = float(target.mean())
    spread = max(float(target.std()), 1.)
    y = basis.new_tensor((target - center) / spread)
    importance = basis.new_tensor(importance)[q]
    importance /= importance.mean()
    prior = basis.new_tensor([np.log(0.1 if "recent" in k else 0.045) for k in names])
    log_mass = torch.nn.Parameter(prior.clone())
    noise = torch.nn.Parameter(basis.new_tensor([-2.9, -2., -1.5]))
    fn = torch.compile(objective, dynamic=False, fullgraph=True)
    fit = basis[:, :n, :n].contiguous()
    opt = torch.optim.Adam([log_mass, noise], lr=lr)
    elapsed = []
    loss = None
    for _ in range(steps):
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = fn(log_mass, noise, fit, y, q, importance, prior)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        elapsed.append(time.monotonic() - start)
    if loss is None or not torch.isfinite(loss):
        raise ArithmeticError("Nonfinite C1 training loss")
    with torch.no_grad():
        gram = torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8
        chol = torch.linalg.cholesky(gram[:n, :n] + torch.diag(noise.exp()[q] + 0.002))
        dual = torch.cholesky_solve(y[:, None], chol).flatten()
        loo = y - dual / torch.cholesky_inverse(chol).diag()
        held = gram[n:, :n] @ dual
        estimate = torch.cat((loo, held)) * spread + center
    state = {
        "names": names, "dual": dual.cpu(), "log_mass": log_mass.detach().cpu(),
        "noise": noise.detach().cpu(), "mean": center, "scale": spread,
    }
    timing = {"first_s": elapsed[0], "steady_median_s": float(np.median(elapsed[1:])), "training_s": sum(elapsed)}
    return estimate, state, timing


def pressure_basis(eye, cloud, temperature, n):
    # CDO modulates the contribution of warm-eye radiometry. Temperature
    # comes from an image-derived prediction for each row, including in fit.
    depth = (35 - temperature.clamp(-94., -50.)) / 100
    depth = depth / depth[:n].mean()
    x = (temperature - temperature[:n].mean()) / temperature[:n].std(unbiased=False).clamp_min(1.)
    contrast = x[:, None] - x
    conditioned = (-0.5 * (contrast / 1.3).square()).exp() * (depth[:, None] * depth)
    thermal_depth = (depth[:, None] * depth) * (0.72 + 0.28 * (-0.5 * contrast.square()).exp())
    cdo_signal = (x[:, None] * x + (-0.5 * contrast.square()).exp()) * 0.5
    return torch.cat((eye * thermal_depth, cloud, cdo_signal[None], (eye.mean(0) * conditioned)[None]))


def evaluate(args):
    raw = (args.source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    source_digest = hashlib.sha256(raw).hexdigest()
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    groups = {}
    for file, keys in ((args.polar, (*EYE, "wall")), (args.cdo, CLOUD)):
        with np.load(file) as data:
            if (str(data["source_sha256"]) != source_digest or not np.array_equal(data["record"], ids)
                    or not np.array_equal(data["time"], times, equal_nan=True)):
                raise ValueError(f"Feature provenance differs from {args.source}")
            groups.update({key: data[key] for key in keys})

    if not torch.cuda.is_available():
        raise RuntimeError("C1 requires CUDA for compiled training")
    torch.manual_seed(SEED)
    torch._dynamo.config.suppress_errors = False
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton"))
    y = np.array([catalog[i]["dP"] for i in ids])
    z = np.array([catalog[i]["t_cdo"] for i in ids])
    quality = np.array([catalog[i]["quality"] for i in ids])
    if not np.isfinite(z).all():
        raise ValueError("C1 auxiliary target requires finite catalog CDO temperatures")
    assignment = record_folds(len(ids), 5, SEED)
    estimates = np.full((2, len(ids)), np.nan)
    trace = []
    run = {
        "seed": SEED, "folds": 5, "records": len(ids),
        "frames": int(np.isfinite(groups["eye"]).any(-1).sum()),
        "catalog_sha256": catalog_digest, "source_meta_sha256": source_digest,
        "cdo_features": str(args.cdo.resolve()), "eye_features": str(args.polar.resolve()),
        "cuda": torch.cuda.get_device_name(), "torch": torch.__version__,
        "training": "CUDA FP64 torch.compile(fullgraph=True)", "posterior": "eager CUDA FP64",
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction",
                         "cdo_target_c", "cdo_prediction_c"])
        for k in range(5):
            train = np.flatnonzero(assignment != k)
            held = np.flatnonzero(assignment == k)
            n = len(train)
            tag = torch.as_tensor([QUALITY[q] for q in quality[train]], device="cuda")
            begin = time.monotonic()
            cloud, cloud_names = bank({name: groups[name] for name in (*CLOUD, "wall")}, times, train, held,
                                      valid=np.isfinite(groups["eye"]).any(-1))
            temperature, z_state, z_timing = fit_posterior(
                cloud, cloud_names, z[train], tag, (1., 0.65, 0.42), 80, 0.045,
            )
            eye, eye_names = bank({name: groups[name] for name in EYE}, times, train, held)
            pressure = pressure_basis(eye, cloud, temperature, n)
            prediction, y_state, p_timing = fit_posterior(
                pressure, [*eye_names, *cloud_names, "cdo_posterior", "eye_cdo_interaction"],
                y[train], tag, (1., 0.4, 0.22), 130, 0.045,
            )
            estimates[:, held] = torch.stack((prediction[n:], temperature[n:])).cpu().numpy()
            trace.append({"fold": k, "cdo": z_timing, "dp": p_timing, "wall_s": time.monotonic() - begin})
            torch.save({"train_records": ids[train].tolist(), "cdo": z_state, "dp": y_state},
                       args.output / f"fold{k}.pt")
            writer.writerows((SEED, k, ids[i], quality[i], y[i], estimates[0, i], z[i], estimates[1, i])
                             for i in held)
            stream.flush()
            print(trace[-1], flush=True)
    small_eye = np.array([catalog[i]["pixels"] < 34 for i in ids]) & (quality == "Good")
    cold_cloud = (z < -74.) & (quality == "Good")
    strata = {
        key: {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean((y[mask] - estimates[0, mask])**2)))}
        for key, mask in (("Good_pixels_lt_34", small_eye), ("Good_cdo_lt_minus_74_c", cold_cloud))
    }
    result = {**run, "trace": trace, "metrics": scores(y, estimates[0], quality),
              "cdo_metrics": scores(z, estimates[1], quality), "diagnostic_strata": strata}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"dp": result["metrics"], "cdo": result["cdo_metrics"], "strata": strata}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "polar", "cdo", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    evaluate(parser.parse_args())
