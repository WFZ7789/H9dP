"""Evaluate both U1 routes in the same record folds; never reuse held predictions for fitting."""

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from h9dp.eye_cdo_kernel import pixel_kernels
from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds, scores
from h9dp.u1 import (
    calibrate_terms,
    expert_bank,
    press_pool,
    pressure_kernel,
    pressure_loss,
    pressure_predict,
)
from h9dp.u1_features import image_views, project_view, scene_kernels
from scripts.audit_motis_inputs import (
    COLUMNS,
    design_matrix,
    read_thermal_rows,
    record_key,
    unique_lookup,
)

GROUND = ("eye", "relief", "wall", "ring", "ring_sectors", "cloud_cover", "eye_cdo_relief",
          "waves", "sector", "eye_spectral_strata", "eye_map", "cdo_map")
ROUTES = ("termwise", "distilled", "closure", "direct", "hybrid")
RANK = 12
STEPS = 110


def load_features(path, mode, keys, ids, times, source_digest, catalog_digest=None):
    with np.load(path) as data:
        if (str(data["mode"]) != mode or str(data["source_sha256"]) != source_digest
                or not np.array_equal(data["record"], ids)
                or not np.array_equal(data["time"], times, equal_nan=True)
                or (catalog_digest is not None and str(data["catalog_sha256"]) != catalog_digest)):
            raise ValueError(f"Mismatched {mode} features: {path}")
        arrays = {k: data[k] for k in keys}
    if any(a.shape[:2] != times.shape for a in arrays.values()):
        raise ValueError(f"Mismatched feature slots: {path}")
    return arrays


def matched_terms(path, catalog, ids):
    table = read_thermal_rows(path)  # allowlist: pass keys and thermal terms only
    lookup = unique_lookup(table)
    result = np.full((len(ids), len(COLUMNS)), np.nan)
    matched = [(i, lookup[key]) for i, rid in enumerate(ids) if (key := record_key(catalog[rid])) in lookup]
    if not matched:
        raise ValueError("No exact thermal parameter matches")
    indices, positions = zip(*matched, strict=True)
    result[list(indices)] = design_matrix([table[j] for j in positions])
    return result


def timed(func, *args):
    torch.cuda.synchronize()
    start = time.monotonic()
    value = func(*args)
    torch.cuda.synchronize()
    return value, time.monotonic() - start


def timed_solve(function, *args):
    """Repeat the identical closed-form solve to time a warmed call, not a second fit."""
    compiled = torch.compile(function, fullgraph=True, dynamic=False)
    _, first = timed(compiled, *args)
    state, warm = timed(compiled, *args)
    return state, {"first_s": first, "warm_s": warm}


def fit_direct(x, fixed, y, q):
    n = len(y)
    groups = len(x)
    prior = x.new_tensor(np.log([0.45 / groups] * groups + [0.8 / groups] * groups
                                + [0.6] + [0.45 / len(fixed)] * len(fixed)))
    prior_noise = x.new_tensor(np.log([0.10, 0.18, 0.26]))
    mass, noise = (torch.nn.Parameter(p.clone()) for p in (prior, prior_noise))
    metric = torch.nn.Parameter(torch.zeros_like(x[:, 0, :]))
    loss_fn = torch.compile(pressure_loss, fullgraph=True, dynamic=False)
    optimizer = torch.optim.Adam([metric, mass, noise], lr=0.035)
    mean, sd = y.mean(), y.std(unbiased=False).clamp_min(1.)
    inputs = (x[:, :n].contiguous(), fixed[:, :n, :n].contiguous(), (y - mean) / sd, q, prior, prior_noise)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(mass, metric, noise, *inputs)
        loss.backward()
        optimizer.step()
        return loss.detach()

    loss, first = timed(update)
    durations = [first]
    for _ in range(1, STEPS):
        loss, duration = timed(update)
        durations.append(duration)
    if not torch.isfinite(loss):
        raise ArithmeticError("Nonfinite U1 pressure objective")
    with torch.no_grad():
        kernel = pressure_kernel(x, fixed, mass, metric)
        pred, intercept, dual = pressure_predict(kernel, y, q, noise)
    values = {"log_mass": mass, "metric": metric, "log_noise": noise, "intercept": intercept, "dual": dual}
    state = {k: v.detach().cpu() for k, v in values.items()}
    return pred, state, {"first_s": durations[0], "steady_median_s": float(np.median(durations[1:])),
                         "optimizer_s": sum(durations)}


def fit_motis(bank, train_y, train_terms):
    """The held rows of the bank are queries only; held table measurements are not arguments."""
    n = len(train_y)
    complete = torch.isfinite(train_terms).all(1)
    support = complete.nonzero().flatten()
    missing = (~complete).nonzero().flatten()
    other = torch.cat((missing, torch.arange(n, bank.shape[1], device=bank.device)))
    z = train_terms[support]
    if len(z) <= z.shape[1]:
        raise ValueError("Insufficient training pairs to calibrate M11")
    (offset, beta), calibration_time = timed_solve(calibrate_terms, z, train_y[support])
    measured = z @ beta + offset
    learn = torch.column_stack((z, measured))
    cols = bank.index_select(2, support)
    (retrieved, weights, mean, dual), retrieval_time = timed_solve(
        press_pool, cols.index_select(1, support), cols.index_select(1, other), learn,
    )
    termwise = retrieved[:, :-1] @ beta + offset
    distilled = retrieved[:, -1]
    thermal = 0.5 * (termwise + distilled)
    # On supported training rows subtract the true thermal component, not its in-sample fit.
    # On unsupported rows retrieval is already out of support. No missing target is invented.
    delta = train_y.clone()
    delta[support] -= measured
    delta[missing] -= thermal[:len(missing)]
    (correction, dw, dm, dd), residual_time = timed_solve(
        press_pool, bank[:, :n, :n].contiguous(), bank[:, n:, :n].contiguous(), delta[:, None],
    )
    values = {"support": support, "offset": offset, "beta": beta, "weight": weights, "mean": mean, "dual": dual,
              "residual_weight": dw, "residual_mean": dm, "residual_dual": dd}
    state = {k: v.detach().cpu() for k, v in values.items()}
    held = slice(len(missing), None)
    predictions = (termwise[held], distilled[held], thermal[held] + correction.flatten())
    return predictions, retrieved[held, :-1], state, {
        "calibration": calibration_time, "retrieval": retrieval_time, "residual": residual_time,
    }


@dataclass
class Study:
    ids: np.ndarray
    times: np.ndarray
    ground: dict
    field: np.ndarray
    terms: np.ndarray
    refs: list[dict]
    source_digest: str
    catalog_digest: str


def load_study(args):
    try:
        raw = (args.source / "all_meta.json").read_bytes()
        rows = json.loads(raw)["records"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"Cannot read record metadata from {args.source}") from error
    catalog, digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    source_digest = hashlib.sha256(raw).hexdigest()
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    ground = load_features(args.ground, "ground_eye", GROUND, ids, times, source_digest, digest)
    field = load_features(args.field, "field", ("map",), ids, times, source_digest)["map"]
    terms = matched_terms(args.thermal, catalog, ids)
    return Study(ids, times, ground, field, terms, [catalog[i] for i in ids], source_digest, digest)


def configure(output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for compiled training")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output.parent / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(output.parent / "triton"))
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(1)


def fixed_kernels(study):
    eye, cloud = [torch.as_tensor(study.ground[k], device="cuda") for k in ("eye_map", "cdo_map")]
    frames = int(eye[:, :, 4].flatten(2).any(2).sum())
    fixed = torch.cat((pixel_kernels(eye, cloud, torch.as_tensor(study.times, device="cuda", dtype=torch.float32)),
                       scene_kernels(study.field, study.times, eye.device)))
    return fixed, frames


def input_hashes(args):
    inputs = {}
    for path in (args.ground, args.field, args.thermal):
        with path.open("rb") as f:
            inputs[str(path.resolve())] = hashlib.file_digest(f, "sha256").hexdigest()
    return inputs


def evaluate(args):
    study = load_study(args)
    ids, terms = study.ids, study.terms
    y = np.array([r["dP"] for r in study.refs])
    quality = np.array([r["quality"] for r in study.refs])
    latitude = np.array([r["location"]["lat"] for r in study.refs])
    if not np.isfinite(latitude).all():
        raise ValueError("Missing crop latitude")
    configure(args.output)
    start = time.monotonic()
    views = image_views(study.ground, study.field, study.times, latitude)
    if args.no_latitude:
        del views["latitude"]
    fixed, frames = fixed_kernels(study)
    torch.cuda.synchronize()
    setup_s = time.monotonic() - start
    folds = record_folds(len(y), 5, 1)
    preds = {key: np.full(len(y), np.nan) for key in ROUTES}
    estimated_terms = np.full_like(terms, np.nan)
    oracle = np.full_like(y, np.nan)
    trace = []
    with (args.output / "predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "prediction", *ROUTES[:-1]])
        for k in range(5):
            start = time.monotonic()
            train, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            order = np.r_[train, held]
            x = fixed.new_tensor(np.stack([project_view(v, train, held, RANK) for v in views.values()]))
            gram = fixed[:, order][:, :, order]
            target = fixed.new_tensor(y[train])
            q = torch.tensor([{"Good": 0, "Fair": 1, "Poor": 2}[s] for s in quality[train]], device=x.device)
            direct, ds, dt = fit_direct(x, gram, target, q)
            (termwise, distilled, closure), estimates, ms, mt = fit_motis(
                expert_bank(x, gram), target, x.new_tensor(terms[train]),
            )
            outputs = (termwise, distilled, closure, direct, 0.5 * (direct + closure))
            for key, value in zip(ROUTES, outputs, strict=True):
                preds[key][held] = value.cpu().numpy()
            if not all(np.isfinite(p[held]).all() for p in preds.values()):
                raise ArithmeticError(f"Nonfinite prediction for fold {k}")
            estimated_terms[held] = estimates.cpu().numpy()
            # Evaluation only, after all held pressure predictions are materialized.
            oracle[held] = terms[held] @ ms["beta"].numpy() + ms["offset"].item()
            torch.save({"train_ids": ids[train].tolist(), "held_ids": ids[held].tolist(), "views": list(views),
                        "rank": RANK, "direct": ds, "motis": ms}, args.output / f"fold{k}.pt")
            writer.writerows([1, k, ids[i], quality[i], y[i], preds["hybrid"][i],
                              *(preds[key][i] for key in ROUTES[:-1])] for i in held)
            stream.flush()
            trace.append({"fold": k, "paired_train": len(ms["support"]), "direct": dt, "motis": mt,
                          "wall_s": time.monotonic() - start})
            print(f"fold {k}: {trace[-1]}", flush=True)
    paired = np.isfinite(oracle)
    np.savez_compressed(args.output / "thermal_oof.npz", record=ids, fold=folds, names=COLUMNS,
                        paired=paired, estimated_terms=estimated_terms, oracle=oracle)
    metrics = {key: scores(y, p, quality) for key, p in preds.items()}
    result = {"seed": 1, "catalog_sha256": study.catalog_digest, "source_meta_sha256": study.source_digest,
              "inputs": input_hashes(args),
              "records": len(y), "frames": frames, "paired": int(paired.sum()), "views": list(views),
              "primary_metric": "all.rmse", "prediction_column": "hybrid", "metrics": metrics,
              "oracle_paired_only": scores(y[paired], oracle[paired], quality[paired]),
              "scaled_term_rmse_paired": dict(zip(COLUMNS, np.mean((estimated_terms[paired] - terms[paired])**2,
                                                                   axis=0)**0.5, strict=True)),
              "setup_s": setup_s, "trace": trace, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "training": "torch.compile(fullgraph=True), CUDA FP64", "inference": "eager/closed-form"}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({k: v["all"] for k, v in metrics.items()}, flush=True)


def study_parser(description):
    p = argparse.ArgumentParser(description=description)
    for key in ("source", "ground", "field", "thermal", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    return p


if __name__ == "__main__":
    p = study_parser(__doc__)
    p.add_argument("--no-latitude", action="store_true", help="Omit the explicit center latitude view")
    evaluate(p.parse_args())
