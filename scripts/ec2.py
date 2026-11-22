"""Evaluate EC2 with rotation-marginalized eye/CDO pixels and regularized pressure fitting."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.a1 import sequence_matches
from h9dp.eye_cdo_kernel import auxiliary, objective, observation_noise, pixel_kernels
from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds, scores
from scripts.q1 import fit as fit_clarity_mixture
from scripts.v1 import bank

EYE = ("eye", "relief")
CDO = ("wall", "ring", "ring_sectors", "cloud_cover", "eye_cdo_relief")
CONTEXT = ("waves", "sector")
PIXELS = ("eye_recent_narrow", "eye_recent_wide", "eye_latest_narrow", "eye_latest_wide",
          "cdo_recent_narrow", "cdo_recent_wide")


def fold_basis(groups, times, pixels, train, held, clear, temperature):
    valid = np.isfinite(groups["eye"]).any(-1)
    basis, names = bank(groups, times, train, held, valid=valid)
    n = len(train)
    order = np.r_[train, held]
    pixel = pixels[:, order][:, :, order]
    eye_shape = torch.stack([item for item, name in zip(basis, names, strict=True)
                             if name in ("eye_recent_linear", "relief_recent_linear")]).mean(dim=0)
    cloud_shape = torch.stack([item for item, name in zip(basis, names, strict=True)
                               if name in ("wall_recent_matern", "ring_recent_matern",
                                           "ring_sectors_recent_matern", "cloud_cover_recent_matern")]).mean(dim=0)
    eye = 0.55 * pixel[1] + 0.20 * pixel[3] + 0.25 * eye_shape
    cloud = 0.62 * cloud_shape + 0.38 * pixel[5]
    clarity_source = 0.75 * eye + 0.25 * cloud + 1
    visibility, _ = auxiliary(clarity_source, clear, pixel.new_tensor(0.105))
    visibility = visibility.clamp(0.04, 0.96)

    scale = temperature.std(unbiased=False).clamp_min(1.)
    mean = temperature.mean()
    cloud_temperature, variance = auxiliary(cloud + 1, (temperature - mean) / scale, pixel.new_tensor(0.048))
    v = visibility - visibility[:n].mean()
    t = cloud_temperature.clamp(-3, 3)
    thermal = t[:, None] - t
    gate = (thermal.square() / -1.4).exp()
    extras = torch.stack((eye * (v[:, None] * v), cloud * (1 - visibility[:, None]) * (1 - visibility),
                          eye * cloud * gate, t[:, None] * t + gate))
    extra_names = ["visibility_eye", "obscured_cdo", "eye_cdo_product", "cdo_temperature"]
    if "eye_spectral_strata" in groups:
        columns = [item for item, name in zip(basis, names, strict=True)
                   if name.startswith("eye_spectral_strata_recent_")]
        interaction = torch.stack(columns).mean(0) * cloud * gate
        extras = torch.cat((extras, interaction[None]))
        extra_names.append("ground_spectrum_cloud")
    scale_variance = variance[:n].median().clamp_min(1e-8)
    conditions = torch.stack((torch.ones_like(v), 2 * v,
                              (variance / scale_variance).log().clamp(-1, 1.3)), dim=1)
    return (torch.cat((basis, pixel, extras)), conditions,
            [*names, *PIXELS, *extra_names],
            cloud_temperature[n:] * scale + mean, variance[n:].sqrt() * scale)


def fit(basis, conditions, names, y):
    n = len(y)
    mean, scale = float(y.mean()), max(float(y.std()), 1.)
    target = basis.new_tensor((y - mean) / scale)
    prior = basis.new_tensor([np.log(0.055 if "recent" in name else 0.03) for name in names])
    noise_prior = basis.new_tensor([-2.1, -0.4, 0.25])
    log_mass = torch.nn.Parameter(prior.clone())
    noise = torch.nn.Parameter(noise_prior.clone())
    compiled = torch.compile(objective, fullgraph=True, dynamic=False)
    optimizer = torch.optim.Adam([log_mass, noise], lr=0.043)
    block = basis[:, :n, :n].contiguous()
    trace = []
    for _ in range(140):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(log_mass, noise, block, conditions[:n].contiguous(), target, prior, noise_prior)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        trace.append(time.monotonic() - start)
    if not torch.isfinite(loss):
        raise ArithmeticError("EC2 pressure objective is nonfinite")
    with torch.no_grad():
        kernel = torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8
        covariance = kernel[:n, :n] + torch.diag(observation_noise(noise, conditions[:n]))
        chol = torch.linalg.cholesky(covariance)
        dual = torch.cholesky_solve(target[:, None], chol).flatten()
        forecast = (kernel[n:, :n] @ dual) * scale + mean
    state = {"names": names, "log_mass": log_mass.detach().cpu(), "noise": noise.detach().cpu(),
             "target_mean": mean, "target_scale": scale, "dual": dual.cpu()}
    timing = {"first_s": trace[0], "steady_median_s": float(np.median(trace[1:])), "training_s": sum(trace)}
    return forecast.cpu().numpy(), state, timing


def evaluate(args):
    source = (args.source / "all_meta.json").read_bytes()
    rows = json.loads(source)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    source_digest = hashlib.sha256(source).hexdigest()
    ids = np.array([row["record"] for row in rows])
    times = np.array([row["times"] for row in rows])
    groups = {}
    geometry = {}
    if args.ground is None:
        sources = ((args.polar, (*EYE, "wall", *CONTEXT)), (args.cdo, CDO[1:]),
                   (args.pixels, ("eye_map", "cdo_map")))
    else:
        sources = ((args.ground, (*EYE, *CDO, *CONTEXT, "eye_spectral_strata", "eye_map", "cdo_map")),)
    if args.analogs is not None:
        sources = (*sources, (args.analogs, ("map",)))
    for path, keys in sources:
        with np.load(path) as data:
            if (str(data["source_sha256"]) != source_digest or not np.array_equal(data["record"], ids)
                    or not np.array_equal(data["time"], times, equal_nan=True)):
                raise ValueError(f"Feature provenance mismatch: {path}")
            groups.update({key: data[key] for key in keys})
            if path == args.ground:
                if str(data["catalog_sha256"]) != catalog_digest or str(data["mode"]) != "ground_eye":
                    raise ValueError("Ground-plane features must match the catalog navigation")
                geometry = {
                    "reference": str(data["reference"]), "projection": data["projection"].tolist(),
                    "ground_step_km": 2.,
                }
    if not torch.cuda.is_available():
        raise RuntimeError("EC2 requires CUDA for compiled training")
    target = np.array([catalog[i]["dP"] for i in ids])
    temperature = np.array([catalog[i]["t_cdo"] for i in ids])
    quality = np.array([catalog[i]["quality"] for i in ids])
    if not np.isfinite(temperature).all():
        raise ValueError("Missing catalog CDO temperature")
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton"))
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(1)
    start = time.monotonic()
    eye, cloud = (torch.as_tensor(groups.pop(key), device="cuda") for key in ("eye_map", "cdo_map"))
    fixed = pixel_kernels(eye, cloud, torch.as_tensor(times, device="cuda", dtype=torch.float32))
    analog = None
    if args.analogs is not None:
        high, average, recent, lag = sequence_matches(groups.pop("map"), times)
        analog = 0.82 * (0.53 * high + 0.29 * average + 0.18 * recent)
        analog = analog + 0.12 * fixed[1] + 0.06 * fixed[5] - 0.03 * lag.abs().clamp_max(2)
    torch.cuda.synchronize()
    setup_s = time.monotonic() - start
    del eye, cloud
    y = np.full(len(ids), np.nan)
    z = np.full(len(ids), np.nan)
    sigma = np.full(len(ids), np.nan)
    visible = torch.tensor([{"Good": 1., "Fair": 0.5, "Poor": 0.}[tag] for tag in quality],
                           dtype=torch.float64, device="cuda")
    cdo_temp = torch.as_tensor(temperature, device="cuda")
    assignment = record_folds(len(ids), 5, 1)
    traces = []
    fields = ("seed", "fold", "record", "quality", "target", "prediction",
              "cdo_target_c", "cdo_prediction_c", "cdo_prediction_sd_c")
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for fold in range(5):
            train = np.flatnonzero(assignment != fold)
            held = np.flatnonzero(assignment == fold)
            start = time.monotonic()
            basis, conditions, names, temp, sd = fold_basis(
                groups, times, fixed, train, held, visible[train], cdo_temp[train],
            )
            if args.marginal_clarity:
                transfer = None if analog is None else analog[np.r_[train, held]][:, train]
                y[held], state, timing = fit_clarity_mixture(
                    basis, conditions, names, target[train], quality[train], transfer,
                )
            else:
                y[held], state, timing = fit(basis, conditions, names, target[train])
            z[held], sigma[held] = temp.cpu().numpy(), sd.cpu().numpy()
            traces.append({"fold": fold, **timing, "wall_s": time.monotonic() - start})
            torch.save({"train_records": ids[train].tolist(), **state}, args.output / f"fold{fold}.pt")
            writer.writerows((1, fold, ids[i], quality[i], target[i], y[i], temperature[i], z[i], sigma[i])
                             for i in held)
            stream.flush()
            print(f"fold={fold} {traces[-1]}", flush=True)
    result = {
        "seed": 1, "folds": 5, "records": len(rows),
        "frames": int(np.isfinite(groups["eye"]).any(-1).sum()),
        "pressure_source": "MOTIS_cyclone_records.json", "catalog_sha256": catalog_digest,
        "source_meta_sha256": source_digest, "features": [str(path) for path, _ in sources],
        "model": ("L1" if args.analogs is not None else
                  "Q1" if args.marginal_clarity else "EC2" if args.ground is None else "G1"),
        "navigation": geometry,
        "primary_metric": "metrics.all.rmse", "metrics": scores(target, y, quality),
        "cdo_metrics": scores(temperature, z, quality), "pixel_precompute_s": setup_s, "trace": traces,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "training": "torch.compile(fullgraph=True) CUDA FP64", "inference": "eager CUDA FP64",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    features = parser.add_mutually_exclusive_group(required=True)
    features.add_argument("--polar", type=Path, help="Native angular measurements for EC2")
    features.add_argument("--ground", type=Path, help="Ground-plane measurements for G1")
    parser.add_argument("--cdo", type=Path)
    parser.add_argument("--pixels", type=Path)
    parser.add_argument("--marginal-clarity", action="store_true", help="Marginalize eye clarity for Q1")
    parser.add_argument("--analogs", type=Path, help="Local residual transfer with a broad cloud map for L1")
    args = parser.parse_args()
    if args.marginal_clarity and args.ground is None:
        parser.error("--marginal-clarity requires --ground")
    if args.analogs is not None and not args.marginal_clarity:
        parser.error("--analogs requires --marginal-clarity")
    if args.polar is not None and (args.cdo is None or args.pixels is None):
        parser.error("--polar requires --cdo and --pixels")
    if args.ground is not None and (args.cdo is not None or args.pixels is not None):
        parser.error("--ground already includes CDO and eye maps")
    evaluate(args)
