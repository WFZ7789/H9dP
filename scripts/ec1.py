"""Train a compact eye/CDO pixel model on the single seed-1 record split."""

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.eye_cdo import EyeCDO, posterior_loss
from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds, scores
from scripts.v1 import bank

EYE = ("eye", "relief")
CDO = ("ring", "ring_sectors", "cloud_cover", "eye_cdo_relief", "wall")
QUALITY = {"Good": 0, "Fair": 1, "Poor": 2}
STEPS = 80


def normalize_pixels(values):
    color = values[:, :, :4].astype(np.float32) / 2500.0
    valid = values[:, :, 4:5]
    return np.concatenate((np.where(valid, color, 0.), valid), axis=2)


def temporal_weights(times, valid):
    lag = np.where(valid, times, -np.inf)
    recent = np.exp((lag - lag.max(1, keepdims=True)) / 45.0)
    latest = np.eye(lag.shape[1])[lag.argmax(1)]
    return np.stack((recent / recent.sum(1, keepdims=True), latest), axis=1)


def fit(model, images, cloud, weights, pressure, cdo, y, t, quality):
    n = len(y)
    device = images.device
    y_center, y_scale = float(y.mean()), float(y.std())
    t_center, t_scale = float(t.mean()), float(t.std())
    y_value = pressure.new_tensor((y - y_center) / y_scale)
    t_value = pressure.new_tensor((t - t_center) / t_scale)
    q = torch.as_tensor([QUALITY[tag] for tag in quality], device=device)
    yw = pressure.new_tensor([1.0, 0.40, 0.22])[q]
    tw = pressure.new_tensor([1.0, 0.65, 0.42])[q]
    yw, tw = yw / yw.mean(), tw / tw.mean()

    def objective(eye_pixels, cdo_pixels, frame_weight, pressure_basis, cdo_basis):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            kp, kt, c_noise = model(eye_pixels, cdo_pixels, frame_weight, pressure_basis, cdo_basis)
        py, ey = posterior_loss(kp, model.pressure_noise.exp()[q] + 0.003, y_value, yw)
        pt, et = posterior_loss(kt, c_noise + 0.006, t_value, tw)
        return py + 0.12 * ey + 0.24 * (pt + 0.16 * et)

    compiled = torch.compile(objective, fullgraph=True, dynamic=False)
    optimizer = torch.optim.AdamW(
        [
            {"params": [*model.eye.parameters(), *model.cloud.parameters(), *model.noise.parameters()],
             "lr": 0.0018, "weight_decay": 0.015},
            {"params": [model.pressure_weight, model.cloud_weight, model.amplitude, model.pressure_noise],
             "lr": 0.045, "weight_decay": 0.0},
        ],
        fused=True,
    )
    inputs = (images[:n].contiguous(), cloud[:n].contiguous(), weights[:n].contiguous(),
              pressure[:, :n, :n].contiguous(), cdo[:, :n, :n].contiguous())
    steps = []
    for step in range(STEPS):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(*inputs)
        loss.backward()
        if step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize()
        steps.append(time.monotonic() - start)
    if not torch.isfinite(loss):
        raise ArithmeticError("Non-finite EC1 training objective")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        kp, kt, c_noise = model(images, cloud, weights, pressure, cdo)
        outputs = []
        for gram, variance, target, center, scale in (
            (kp, model.pressure_noise.exp()[q] + 0.003, y_value, y_center, y_scale),
            (kt, c_noise[:n] + 0.006, t_value, t_center, t_scale),
        ):
            factor = torch.linalg.cholesky(gram[:n, :n] + torch.diag(variance))
            dual = torch.cholesky_solve(target[:, None], factor)
            estimate = (gram[n:, :n] @ dual).flatten() * scale + center
            outputs.append(estimate.cpu().numpy())
        # Held-out observation standard deviation includes variation in the
        # posterior mean and the predicted temperature measurement noise.
        conditional = torch.linalg.solve_triangular(factor, kt[:n, n:], upper=False)
        posterior = kt.diag()[n:] - conditional.square().sum(0)
        deviation = (posterior.clamp_min(0) + c_noise[n:] + 0.006).sqrt() * t_scale
    timing = {"first_s": steps[0], "steady_median_s": float(np.median(steps[1:])), "training_s": sum(steps)}
    return [*outputs, deviation.cpu().numpy()], timing


def evaluate(args):
    raw = (args.source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    digest = hashlib.sha256(raw).hexdigest()
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    groups = {}
    sources = (
        (args.polar, (*EYE, "wall", "waves")),
        (args.cdo, CDO[:-1]),
        (args.sampling, ("integrated_eye", "eye_sampling")),
        (args.pixels, ("eye_map", "cdo_map")),
    )
    for path, keys in sources:
        with np.load(path) as data:
            if (str(data["source_sha256"]) != digest or not np.array_equal(ids, data["record"])
                    or not np.array_equal(times, data["time"], equal_nan=True)):
                raise ValueError(f"Mismatched EC1 source: {path}")
            groups.update({k: data[k] for k in keys})

    if not torch.cuda.is_available():
        raise RuntimeError("EC1 requires CUDA for compiled training")
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton"))
    torch._dynamo.config.suppress_errors = False
    images = torch.as_tensor(normalize_pixels(groups.pop("eye_map")), device="cuda", dtype=torch.bfloat16)
    cloud = torch.as_tensor(normalize_pixels(groups.pop("cdo_map")), device="cuda", dtype=torch.bfloat16)
    valid = np.isfinite(groups["eye"]).any(-1)
    weights = torch.as_tensor(temporal_weights(times, valid), device="cuda", dtype=torch.float32)
    y = np.array([catalog[i]["dP"] for i in ids])
    temperature = np.array([catalog[i]["t_cdo"] for i in ids])
    quality = np.array([catalog[i]["quality"] for i in ids])
    if not np.isfinite(temperature).all():
        raise ValueError("Missing EC1 CDO temperature label")
    assignment = record_folds(len(ids), 5, 1)
    prediction = np.full((3, len(ids)), np.nan)
    trace = []
    run = {
        "seed": 1, "folds": 5, "steps": STEPS, "records": len(rows), "frames": int(valid.sum()),
        "catalog_sha256": catalog_digest, "source_meta_sha256": digest,
        "features": [str(path.resolve()) for path, _ in sources],
        "training": "torch.compile(fullgraph=True) CUDA BF16 encoder, FP64 covariance",
        "prediction": "eager CUDA BF16 encoder, FP64 posterior", "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction",
                         "cdo_target_c", "cdo_prediction_c", "cdo_posterior_sd_c"])
        for fold in range(5):
            train = np.flatnonzero(assignment != fold)
            held = np.flatnonzero(assignment == fold)
            pressure, names = bank(groups, times, train, held, valid=valid)
            cdo, cdo_names = bank({name: groups[name] for name in CDO}, times, train, held, valid=valid)
            order = np.r_[train, held]
            torch.manual_seed(1)
            model = EyeCDO(len(names), len(cdo_names)).cuda()
            if fold == 0:
                run["parameters"] = sum(param.numel() for param in model.parameters())
                (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            print(f"fold={fold} train={len(train)} held={len(held)} params={run['parameters']}", flush=True)
            result, timing = fit(model, images[order], cloud[order], weights[order], pressure, cdo,
                                 y[train], temperature[train], quality[train])
            prediction[:, held] = result
            trace.append({"fold": fold, **timing})
            torch.save({"model": model.cpu().state_dict(), "train_records": ids[train].tolist(),
                        "pressure_components": names, "cdo_components": cdo_names}, args.output / f"fold{fold}.pt")
            writer.writerows((1, fold, ids[i], quality[i], y[i], prediction[0, i], temperature[i],
                              prediction[1, i], prediction[2, i]) for i in held)
            stream.flush()
            print(trace[-1], flush=True)

    z_error = temperature - prediction[1]
    good = quality == "Good"
    small = good & np.array([catalog[i]["pixels"] < 34 for i in ids])
    limits = np.argsort(prediction[2])
    quartiles = [
        {"records": len(part), "posterior_sd_c": float(np.mean(prediction[2, part])),
         "cdo_mae_c": float(np.mean(np.abs(z_error[part])))}
        for part in np.array_split(limits, 4)
    ]
    result = {
        **run, "trace": trace, "metrics": scores(y, prediction[0], quality),
        "cdo_metrics": scores(temperature, prediction[1], quality),
        "cdo_uncertainty_quartiles": quartiles,
        "small_eye_good_rmse": math.sqrt(float(np.mean((y[small] - prediction[0, small]) ** 2))),
        "small_eye_good_records": int(small.sum()),
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("metrics", "cdo_metrics", "cdo_uncertainty_quartiles",
                                             "small_eye_good_rmse")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "polar", "cdo", "sampling", "pixels", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    evaluate(parser.parse_args())
