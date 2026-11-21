#!/usr/bin/env python3
"""P3.1 full-shape resource check and training-only reference-point geometry.

Synthetic optimizer steps measure resources, not fitting accuracy. One real
batch receives a read-only forward. No experiment checkpoint is produced.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def geometry_report(index, size, translation):
    import numpy as np

    # native_window snaps the geometric center with torch.round (ties to even).
    center = round(499 / 2)
    lower, upper = center - size // 2, center + size // 2
    groups = {name: {"unaugmented": [], "augmented_expected": [], "augmented_worst_case": []} for name in ("R3", "FLDK")}
    for sample in index["samples"]:
        if sample["split"] != "train":
            continue
        original = np.asarray(sample["centers"])
        for rotation in (False, True):
            xy = 499 - original if rotation else original
            lo = np.maximum(-translation, np.ceil(-xy.min(0))).astype(int)
            hi = np.minimum(translation, np.floor(499 - xy.max(0))).astype(int)
            dx = np.arange(lo[0], hi[0] + 1) if (lo <= hi).all() else np.array([0])
            dy = np.arange(lo[1], hi[1] + 1) if (lo <= hi).all() else np.array([0])
            for point, source in zip(xy, sample["sources"], strict=True):
                x, y = point
                px = np.mean((x + dx >= lower) & (x + dx <= upper))
                py = np.mean((y + dy >= lower) & (y + dy <= upper))
                inside = lower <= x <= upper and lower <= y <= upper
                group = groups[source]
                if not rotation:
                    group["unaugmented"].append(inside)
                # 50% translation, 50% rotation, exactly as the current config.
                group["augmented_expected"].append(.5 * inside + .5 * px * py)
                group["augmented_worst_case"].append((x + dx.min() >= lower) and (x + dx.max() <= upper) and (y + dy.min() >= lower) and (y + dy.max() <= upper))
    return {
        "training_only": True, "window_bounds_inclusive": [lower, upper],
        "reference_points_not_eye_labels": True,
        "augmentation_assumption": "rotation_probability=translation_probability=0.5; uniform feasible integer translations",
        "sources": {name: {"frame_occurrences": len(values["unaugmented"]),
                           **{key: np.mean(value).item() for key, value in values.items()}} for name, values in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/p3_coverage.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch

    from h9dp.common import check_space, load_config, write_json
    from h9dp.dataset import SequenceDataset
    from h9dp.engine import (
        code_fingerprint,
        configuration_fingerprint,
        initialize_device,
        seed_everything,
    )
    from h9dp.model import build_model, regression_loss
    from h9dp.prepare import load_cache

    config = load_config(args.config)
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    if batch_size < 1 or args.steps < 2:
        parser.error("Use positive batch size and at least two steps (first is warmup)")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output) if args.output else Path(config["output_dir"]) / "checks" / stamp
    check_space(output, 50_000_000)
    output.mkdir(parents=True, exist_ok=False)
    if any(config["augmentation"][key] != .5 for key in ("rotation_probability", "translation_probability")):
        raise ValueError("This geometry report assumes rotation/translation probabilities of 0.5")
    index = load_cache(config)
    device, environment = initialize_device(config)
    seed_everything(1)
    model = build_model(config["model"], index["target_normalization"]["mean_hpa"], index["target_normalization"]["std_hpa"]).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    report = {
        "status": "running", "synthetic_resource_check_not_fit": True,
        "architecture": config["model"]["architecture"], "parameter_count": parameters,
        "parameter_groups": {
            "query_and_role_parameters": sum(p.numel() for p in model.parameters(recurse=False)),
            **{name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
        },
        "batch_size": batch_size, "shape": [batch_size, 6, 4, 500, 500], "environment": environment,
        "code_fingerprint": code_fingerprint(config), "configuration_fingerprint": configuration_fingerprint(config),
        "data_fingerprint": index["fingerprint"],
        "geometry": geometry_report(index, config["model"]["center_window"], config["augmentation"]["maximum_translation"]),
    }
    write_json(output / "report.json", report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    images = torch.randn(batch_size, 6, 4, 500, 500, device=device)
    mask = torch.ones(batch_size, 6, 1, 500, 500, device=device, dtype=torch.bool)
    times = torch.tensor([-300., -240., -180., -120., -60., 0.], device=device).expand(batch_size, -1)
    target = torch.linspace(40., 100., batch_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    try:
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images, times, mask)
                loss = regression_loss(prediction, target, model.target_std).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or non-finite gradient in full-shape resource check")
            optimizer.step()
            torch.cuda.synchronize(device)
            timings.append(time.monotonic() - started)
            print({"synthetic_step": step + 1, "seconds": timings[-1], "gradient_norm": norm.item()}, flush=True)
        report.update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            step_seconds=timings, steady_records_per_second=batch_size * (len(timings)-1) / sum(timings[1:]),
        )
        model.eval()
        dataset = SequenceDataset(config["data"]["cache_dir"], index, "train")
        host = dataset[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model(host["images"][None].to(device), host["time_offsets"][None].to(device), host["mask"][None].to(device))
        if not torch.isfinite(actual).all():
            raise RuntimeError("Non-finite real-data forward")
        report.update(status="passed", real_batch_record=host["record"], real_batch_forward_finite=True,
                      root_storage=check_space(output))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json", report)
        print(str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
