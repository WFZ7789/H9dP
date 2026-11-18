#!/usr/bin/env python3
"""Compiled spectral Gaussian-process development evaluation for R1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.r1 import kernel_components, record_folds, scores
from h9dp.r1_gp import SpectralGP, group_weights, schedule
from scripts.r1_adaptive_kernel import assemble
from scripts.r1_kernel import Cache, load_cache

GEOMETRY = {
    "regions_recent": 1.4, "regions_latest": 1.0, "regions_trend": 0.20,
    "thermal_recent": 1.2, "thermal_latest": 0.5, "thermal_trend": 0.3,
    "core_recent": 0.75, "core_latest": 0.4, "core_trend": 0.15,
    "profile_recent": 0.5, "profile_latest": 0.20, "profile_trend": 0.18,
    "structure_recent": 0.35, "structure_latest": 0.22, "structure_trend": 0.18,
    "contour_recent": 0.9, "contour_latest": 0.48, "contour_trend": 0.22,
    "thermal_lift_recent": 0.9, "thermal_lift_latest": 0.3, "thermal_lift_trend": 0.15,
    "warm_brightness_recent": 0.8, "outer_cloud_recent": 0.75,
    "spectral_edge_recent": 0.4, "spectral_edge_latest": 0.2, "radial_slope_recent": 0.4,
}
SCENE = {"scene_recent": 0.42, "scene_core_recent": 0.32, "scene_latest": 0.12,
         "scene_trend": 0.10, "scene_core_latest": 0.08, "scene_core_trend": 0.07}
QUALITY_INDEX = {"Good": 0, "Fair": 1, "Poor": 2}


def configure(run: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("R1 GP requires CUDA compiled training; no eager fallback")
    torch._dynamo.config.suppress_errors = False
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(run.parent / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(run.parent / "triton"))


def matricize(groups, train, held):
    components = kernel_components(groups, train, held)
    geometry = [k for k in GEOMETRY if k in components]
    scene = [k for k in SCENE if k in components]
    if not geometry or not scene:
        raise ValueError("Spectral GP needs both geometry and scene groups")

    def stack(keys, index):
        values = np.stack([components[key][index] for key in keys])
        return torch.as_tensor(values, dtype=torch.float64, device="cuda")

    ga = group_weights(len(geometry), list(range(len(geometry))), [GEOMETRY[key] for key in geometry])
    sa = group_weights(len(scene), list(range(len(scene))), [SCENE[key] for key in scene])
    return stack(geometry, 0), stack(geometry, 1), stack(scene, 1), ga, sa, geometry, scene


def gp_predict(cache: Cache, feature_groups, train: np.ndarray, held: np.ndarray, steps: int, lr: float,
               tail_focus: float = 0.45):
    dot, distance, scene, gp, sp, geometry_names, scene_names = matricize(feature_groups, train, held)
    n = len(train)
    blocks = tuple(m[:, :n, :n].contiguous() for m in (dot, distance, scene))
    q = torch.tensor([QUALITY_INDEX[key] for key in cache.quality[train]], dtype=torch.long, device="cuda")
    target = cache.y[train]
    center, spread = float(target.mean()), max(float(target.std()), 1.0)
    y = torch.as_tensor((target - center) / spread, dtype=torch.float64, device="cuda")
    importance = torch.tensor([1.0, 0.42, 0.25], dtype=torch.float64, device="cuda")[q]
    importance = importance * (1 + tail_focus * torch.as_tensor(((target - 88) / 42).clip(0, 1), device="cuda"))
    importance = importance / importance.mean()
    estimates = []
    states = []
    timings = []
    for local, evidence in ((True, 0.08), (False, 0.24)):
        model = SpectralGP(gp, sp, local).cuda()
        objective = torch.compile(model, dynamic=False, fullgraph=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
        step_s = []
        loss = torch.full((), torch.nan, dtype=torch.float64, device="cuda")
        begin = time.monotonic()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = schedule(step, steps, lr)
            loss = objective(*blocks, y, q, importance, evidence)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_s.append(time.monotonic() - begin)
            begin = time.monotonic()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss from compiled spectral GP")
        with torch.no_grad():
            gram = model.kernel(dot, distance, scene)
            estimate = model.predict(gram[:n, :n], gram[n:, :n], y, q) * spread + center
        if not torch.isfinite(estimate).all():
            raise RuntimeError("Nonfinite spectral GP prediction")
        estimates.append(estimate.cpu().numpy())
        trace = {"local_init": local, "evidence": evidence, "terminal_objective": float(loss.detach()),
                 "first_step_s": step_s[0], "median_steady_step_s": float(np.median(step_s[1:] or step_s)),
                 "duration_s": sum(step_s), "scales": model.scales.detach().cpu().exp().tolist(),
                 "masses": model.masses.detach().cpu().exp().tolist(),
                 "geometry": dict(zip(geometry_names, (model.geometry_prior + model.geometry_delta).softmax(0)
                                      .detach().cpu().tolist(), strict=True)),
                 "scene": dict(zip(scene_names, (model.scene_prior + model.scene_delta).softmax(0)
                                   .detach().cpu().tolist(), strict=True))}
        timings.append(trace)
        states.append({"hyperparameters": model.cpu().state_dict(), "geometry": geometry_names, "scene": scene_names,
                       "target_mean": center, "target_scale": spread})
    return np.mean(estimates, axis=0), states, timings


def evaluate(args):
    cache = load_cache(args.features)
    args.output.mkdir(exist_ok=False)
    configure(args.output)
    groups = assemble(cache, contour=args.contour, spectral=args.spectral, snapshot=args.snapshot)
    metadata = {"schema": "r1-spectral-gp-development/v1", "features": str(cache.root.resolve()),
                "source_meta_sha256": cache.source_digest, "folds": args.folds, "seeds": args.seeds,
                "steps": args.steps, "lr": args.lr, "tail_focus": args.tail_focus,
                "contour": args.contour, "spectral": args.spectral, "snapshot": args.snapshot,
                "records": len(cache.rows), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(), "execution": "torch.compile(fullgraph=True) CUDA FP64 solve",
                "evaluation": "eager CUDA FP64 posterior mean; no outcome-conditioned update"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    res = {}
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "fold", "record", "quality", "target", "spectral_gp"])
        for seed in args.seeds:
            assignment = record_folds(len(cache.rows), args.folds, seed)
            prediction = np.full(len(assignment), np.nan)
            trace = []
            for fold in range(args.folds):
                train, held = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                print(f"seed={seed} fold={fold} train={len(train)} held={len(held)}", flush=True)
                forecast, states, timings = gp_predict(cache, groups, train, held, args.steps, args.lr, args.tail_focus)
                prediction[held] = forecast
                torch.save({"train_records": torch.as_tensor(cache.data["record"][train]), "states": states},
                           args.output / f"seed{seed}_fold{fold}.pt")
                metrics = scores(cache.y[held], forecast, cache.quality[held])
                trace.append({"timing_and_hyperparameters": timings, "metrics": metrics})
                print(f"seconds={[round(s['duration_s'], 3) for s in timings]} metrics={json.dumps(metrics)}", flush=True)
                for i in held:
                    writer.writerow([seed, fold, cache.data["record"][i], cache.quality[i], cache.y[i], prediction[i]])
                stream.flush()
            res[str(seed)] = {"trace": trace, "metrics": scores(cache.y, prediction, cache.quality)}
            print(json.dumps(res[str(seed)]["metrics"]), flush=True)
            (args.output / "metrics.json").write_text(json.dumps({**metadata, "results": res}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9463])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--tail-focus", type=float, default=0.45)
    parser.add_argument("--contour", action="store_true", help="Include native eye-closure descriptors")
    parser.add_argument("--spectral", action="store_true", help="Include separate spectral/thermodynamic groups")
    parser.add_argument("--snapshot", action="store_true", help="Use only one latest valid scan per record")
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0 or not np.isfinite(args.tail_focus) or args.tail_focus < 0:
        parser.error("GP training needs positive steps and LR and a finite nonnegative tail weight")
    evaluate(args)


if __name__ == "__main__":
    main()
