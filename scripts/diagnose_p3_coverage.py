#!/usr/bin/env python3
"""Read-only P3.1 predictions, source masses and routing interventions.

Inference interventions are not retrained ablations. Reference cyclone centers
are not verified eye pixels; coverage metrics are not eye detection accuracy.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def route_statistics(centers, radii, references, native_size, center_size):
    import numpy as np

    # centers B,L,windows,T,2; references B,T,2. Actual 500x500 inputs.
    pixels = (centers + 1) * 249.5
    distances = np.linalg.norm(pixels - references[:, None, None], axis=-1)
    native_contains = (np.abs(np.round(pixels) - references[:, None, None]) <= native_size // 2).all(-1).any(2)
    fixed_contains = (np.abs(references - round(249.5)) <= center_size // 2).all(-1)
    result = []
    for layer in range(centers.shape[1]):
        entry = {
            "layer": layer + 1,
            "center_coordinate_std_across_records_pixels_mean": pixels[:, layer].std(0).mean().item(),
            "boundary_coordinate_fraction": np.mean(np.abs(centers[:, layer]) >= 1 - 1e-6).item(),
            "radius_pixels_q05_q50_q95": np.quantile(radii[:, layer], [.05, .5, .95]).tolist(),
            "nearest_reader_distance_latest_pixels_median": np.median(distances[:, layer, :, -1].min(1)).item(),
            "reference_in_any_native_window_all_frames_fraction": native_contains[:, layer].mean().item(),
            "reference_in_any_native_window_latest_fraction": native_contains[:, layer, -1].mean().item(),
            "reference_in_fixed_or_native_window_fraction": (native_contains[:, layer] | fixed_contains).mean().item(),
        }
        if layer:
            entry["max_axis_refinement_pixels"] = np.abs(pixels[:, layer] - pixels[:, layer - 1]).max().item()
        result.append(entry)
    return {"fixed_window_reference_fraction": fixed_contains.mean().item(), "layers": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs/p3_coverage.yaml"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output")
    args = parser.parse_args()
    import numpy as np
    import torch

    from h9dp.common import check_space, load_config, write_json
    from h9dp.dataset import SequenceDataset, make_loader
    from h9dp.engine import (
        code_fingerprint,
        initialize_device,
        move_batch,
        prediction_diagnostics,
    )
    from h9dp.model import build_model
    from h9dp.p3_coverage import ARCHITECTURE, CoveragePressureRegressor
    from h9dp.prepare import load_cache

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    index = load_cache(config)
    if state["architecture"] != ARCHITECTURE or state["data_fingerprint"] != index["fingerprint"]:
        raise ValueError("Checkpoint architecture/data mismatch")
    if state["model_config"] != config["model"]:
        raise ValueError("Model configuration mismatch")
    device, environment = initialize_device(config)
    target = state["target_normalization"]
    model = build_model(state["model_config"], target["mean_hpa"], target["std_hpa"]).to(device).eval()
    assert isinstance(model, CoveragePressureRegressor)
    model.load_state_dict(state["model"], strict=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output) if args.output else checkpoint.parent / "diagnostics" / f"{args.split}_{stamp}"
    check_space(output, 100_000_000)
    output.mkdir(parents=True, exist_ok=False)
    modes = ("normal", "no_dynamic", "fixed_route", "no_history")
    predictions = {mode: [] for mode in modes}
    details = {name: [] for name in ("centers", "radii", "brightness", "source_mass", "references", "record", "native_valid", "dynamic_valid")}
    by_record = {sample["record"]: sample for sample in index["samples"]}
    loader = make_loader(config, index, args.split, config["training"]["batch_size"], augment=False)
    assert isinstance(loader.dataset, SequenceDataset)
    with torch.no_grad():
        for host in loader:
            batch = move_batch(host, device)
            for mode in modes:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, reads = model.inspect_reads(batch["images"], batch["time_offsets"], batch["mask"], mode)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite prediction: {mode}")
                for position, value in enumerate(prediction.cpu().tolist()):
                    record = host["record"][position].item()
                    target_value = host["target"][position].item()
                    predictions[mode].append({
                        "record": record, "storm_id": host["storm_id"][position], "quality": host["quality"][position],
                        "MOTIS_dP": target_value, "predicted_dP": value, "error_hPa": value - target_value,
                    })
                if mode == "normal":
                    for name, key in (("centers", "centers"), ("radii", "radius_pixels"), ("native_valid", "native_valid_fraction"), ("dynamic_valid", "dynamic_valid_fraction")):
                        details[name].append(torch.stack([r[key] for r in reads["routes"]], dim=1).cpu().numpy())
                    for name in ("brightness", "source_mass"):
                        details[name].append(reads[name].cpu().numpy())
                    records = host["record"].tolist()
                    details["references"].append(np.asarray([by_record[r]["centers"] for r in records]))
                    details["record"].append(np.asarray(records))
    arrays = {name: np.concatenate(values) for name, values in details.items()}
    np.savez_compressed(
        output / "reads.npz", allow_pickle=False, centers=arrays["centers"], radii=arrays["radii"],
        brightness=arrays["brightness"], source_mass=arrays["source_mass"], references=arrays["references"],
        record=arrays["record"], native_valid=arrays["native_valid"], dynamic_valid=arrays["dynamic_valid"],
    )
    metrics = {}
    normal = np.asarray([row["predicted_dP"] for row in predictions["normal"]])
    for mode, rows in predictions.items():
        with (output / f"{mode}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        e = np.asarray([row["error_hPa"] for row in rows])
        changes = np.asarray([row["predicted_dP"] for row in rows]) - normal
        metrics[mode] = {
            "count": len(e), "rmse": np.sqrt(np.mean(e**2)).item(), "mae": np.abs(e).mean().item(), "bias": e.mean().item(),
            "mean_absolute_change_vs_normal": np.abs(changes).mean().item(),
            **prediction_diagnostics(rows, loader.dataset.samples),
        }
    saved = checkpoint.parent / f"best_{args.split}.csv"
    difference = None
    if checkpoint.name == "best.pt" and saved.exists():
        with saved.open() as stream:
            saved_values = {row["record"]: row["predicted_dP"] for row in csv.DictReader(stream)}
        # Explicit NumPy conversion validates numeric CSV values as a group.
        expected = np.asarray([saved_values[str(row["record"])] for row in predictions["normal"]], dtype=float)
        difference = np.abs(expected - normal).max().item()
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_epoch": state["epoch_completed"], "split": args.split,
        "training": False, "optimizer_steps": 0, "environment": environment,
        "warning": "Inference interventions, not retrained ablations. Reference points are not verified eyes.",
        "checkpoint_code_fingerprint": state["code_fingerprint"], "current_code_fingerprint": code_fingerprint(config),
        "max_prediction_difference_vs_saved_best_csv": difference, "predictions": metrics,
        "source_labels": ["global", "fixed_native", "dynamic_native", "dynamic_bilinear"],
        "source_mass_layer_role_source": arrays["source_mass"].mean((0, 3)).tolist(),
        "native_valid_fraction_layer_window": arrays["native_valid"].mean((0, 3)).tolist(),
        "dynamic_valid_fraction_layer_window": arrays["dynamic_valid"].mean((0, 3)).tolist(),
        "routing": route_statistics(arrays["centers"], arrays["radii"], arrays["references"], config["model"]["native_window"], config["model"]["center_window"]),
    }
    write_json(output / "summary.json", summary)
    print(str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
