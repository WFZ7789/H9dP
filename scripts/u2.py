"""Compare direct dP and paired thermal supervision with matched image kernels."""

import csv
import json
import time
from datetime import datetime, timedelta

import numpy as np
import torch

from h9dp.r1 import record_folds, scores
from h9dp.u1 import calibrate_terms
from h9dp.u1_features import image_views
from h9dp.u2 import joint_loss, predict, view_bank
from h9dp.u2_geometry import column_views, geometry
from scripts.u1 import (
    configure,
    fixed_kernels,
    input_hashes,
    load_study,
    study_parser,
    timed,
    timed_solve,
)

STEPS = 105


def make_observations(y, terms=None):
    """Stack all dP rows and, only on exact training matches, two thermal sums."""
    n = len(y)
    rid = torch.arange(n, device=y.device)
    task = torch.zeros_like(rid)
    means, scales = [y.mean()], [y.std(unbiased=False).clamp_min(1e-6)]
    obs = (y - means[0]) / scales[0]
    state = {}
    blocks = rid[:, None]
    base = y.new_ones((1, 1))
    if terms is not None:
        paired = torch.isfinite(terms).all(1).nonzero().flatten()
        z = terms[paired]
        if len(z) <= z.shape[1]:
            raise ValueError("Insufficient paired training records")
        (offset, beta), timing = timed_solve(calibrate_terms, z, y[paired])
        # The first four predictors are B24, B36, upper and mid differences (including CDO scaling).
        parts = torch.column_stack((z[:, :4] @ beta[:4], z[:, 4:] @ beta[4:]))
        means.extend(parts.mean(0).unbind())
        scales.extend(parts.std(0, unbiased=False).clamp_min(1e-6).unbind())
        paired_values = torch.column_stack((y[paired], parts))
        corr = torch.corrcoef(paired_values.T)
        base = torch.linalg.cholesky(0.75 * corr + 0.25 * torch.eye(3, dtype=y.dtype, device=y.device))
        m = len(paired)
        blocks = torch.stack((paired, rid[:m] + n, rid[:m] + n + m), dim=1)
        rid = torch.cat((rid, paired, paired))
        task = torch.cat((task, torch.ones_like(paired), torch.full_like(paired, 2)))
        obs = torch.cat((obs, (parts[:, 0] - means[1]) / scales[1], (parts[:, 1] - means[2]) / scales[2]))
        state.update(beta=beta.cpu(), thermal_intercept=offset.cpu(), training_pairs=m, calibration_timing=timing)
    dim = len(means)
    design = torch.nn.functional.one_hot(task, dim).to(y.dtype)
    state.update(means=torch.stack(means).cpu(), scales=torch.stack(scales).cpu())
    return obs, rid, task, design, blocks, base, state


def fit(bank, y, quality, terms=None):
    obs, rid, task, design, blocks, base, state = make_observations(y, terms)
    dim = design.shape[1]
    mass0 = bank.new_full((len(bank), dim), np.log(1.8 / len(bank)))
    noise0 = bank.new_tensor(np.log([0.11, 0.18, 0.25]))[:, None].expand(3, dim).contiguous()
    log_mass, log_noise = (torch.nn.Parameter(value.clone()) for value in (mass0, noise0))
    corr, ncorr = (torch.nn.Parameter(bank.new_zeros((size, dim, dim))) for size in (len(bank), 3))
    parameters = (log_mass, corr, log_noise, ncorr)
    optimizer = torch.optim.Adam(parameters, lr=0.038)
    compiled = torch.compile(joint_loss, fullgraph=True, dynamic=False)
    block = bank[:, :len(y), :len(y)].contiguous()

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(*parameters, block, obs, rid, task, quality, design, blocks, base, mass0, noise0)
        loss.backward()
        optimizer.step()
        return loss.detach()

    loss, first = timed(update)
    times = []
    for _ in range(1, STEPS):
        loss, duration = timed(update)
        times.append(duration)
    if not torch.isfinite(loss):
        raise ArithmeticError("U2 nonfinite objective; do not use its posterior")
    with torch.no_grad():
        z, alpha, mean = predict(*parameters, bank, obs, rid, task, quality, design, base)
        pred = z * state["scales"][0].item() + state["means"][0].item()
    values = {"log_mass": log_mass, "corr": corr, "log_noise": log_noise, "ncorr": ncorr, "base": base,
              "record": rid, "task": task, "alpha": alpha, "intercept": mean}
    state.update({name: v.detach().cpu() for name, v in values.items()})
    return pred.cpu().numpy(), state, {"first_s": first, "steady_median_s": float(np.median(times)),
                                      "optimizer_s": first + sum(times)}


def features(study, path, mode):
    latitude = np.array([r["location"]["lat"] for r in study.refs])
    views = image_views(study.ground, study.field, study.times, latitude)
    del views["latitude"]
    if mode == "image":
        return views, None
    with np.load(path) as data:
        projection = data["projection"]
    valid = np.isfinite(study.field).any(axis=(2, 3, 4)) & np.isfinite(study.times)
    latest = np.where(valid, study.times, -np.inf).max(1)
    if not np.isfinite(latest).all():
        raise ValueError("Missing scan time for illumination")
    rays, context = [], []
    for row, lag in zip(study.refs, latest, strict=True):
        scan = datetime.fromisoformat(row["time"]["iso_utc"]) + timedelta(minutes=float(lag))
        ray, values = geometry(row["location"]["lat"], row["location"]["lon"], scan, projection)
        rays.append(ray)
        context.append(values)
    if mode in ("full", "navigation"):
        views.update(column_views(study.ground["eye_map"], rays, study.times))
    return views, np.stack(context)[:, :2 if mode == "navigation" else 4]


def evaluate(args):
    study = load_study(args)
    configure(args.output)
    start = time.monotonic()
    views, context = features(study, args.ground, args.mode)
    fixed, frames = fixed_kernels(study)
    torch.cuda.synchronize()
    setup_s = time.monotonic() - start
    ids = study.ids
    folds = record_folds(len(ids), 5, 1)
    target = np.array([r["dP"] for r in study.refs])
    tags = np.array([r["quality"] for r in study.refs])
    quality = torch.tensor([{"Good": 0, "Fair": 1, "Poor": 2}[q] for q in tags], device=fixed.device)
    predictions = {key: np.full(len(ids), np.nan) for key in ("direct", "joint", "hybrid")}
    trace = []
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction", "direct", "joint"])
        for fold in range(5):
            start = time.monotonic()
            train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            bank, names = view_bank(views, fixed, train, held, context)
            y = fixed.new_tensor(target[train])
            q = quality[train]
            direct, ds, dt = fit(bank, y, q)
            joint, js, jt = fit(bank, y, q, fixed.new_tensor(study.terms[train]))
            for key, value in (("direct", direct), ("joint", joint), ("hybrid", (direct + joint) * 0.5)):
                if not np.isfinite(value).all():
                    raise ArithmeticError(f"Nonfinite {key} prediction")
                predictions[key][held] = value
            torch.save({"train_ids": ids[train].tolist(), "held_ids": ids[held].tolist(), "kernels": names,
                        "direct": ds, "joint": js}, args.output / f"fold{fold}.pt")
            writer.writerows([1, fold, ids[i], tags[i], target[i], predictions["hybrid"][i],
                              predictions["direct"][i], predictions["joint"][i]] for i in held)
            stream.flush()
            trace.append({"fold": fold, "paired_train": js["training_pairs"], "direct": dt, "joint": jt,
                          "wall_s": time.monotonic() - start})
            print(trace[-1], flush=True)
    metrics = {k: scores(target, v, tags) for k, v in predictions.items()}
    result = {"seed": 1, "records": len(ids), "frames": frames, "mode": args.mode, "optimizer_steps": STEPS,
              "primary_metric": "all.rmse", "prediction_column": "hybrid", "metrics": metrics, "trace": trace,
              "paired": int(np.isfinite(study.terms).all(1).sum()), "setup_s": setup_s, "inputs": input_hashes(args),
              "catalog_sha256": study.catalog_digest, "source_meta_sha256": study.source_digest,
              "training": "torch.compile(fullgraph=True) CUDA FP64", "inference": "eager CUDA FP64",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({k: v["all"] for k, v in metrics.items()}, flush=True)


if __name__ == "__main__":
    p = study_parser(__doc__)
    p.add_argument("--mode", choices=("image", "context", "navigation", "full"), default="full")
    evaluate(p.parse_args())
