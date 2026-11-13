#!/usr/bin/env python3
"""Plot B08 timelines for the largest best-checkpoint validation errors.

Example (read-only data verification and plotting; no training):
    conda run --no-capture-output -n fy3-aivis-v1 python scripts/plot_b08_errors.py \
        runs/training/RUN_NAME --output-dir runs/h9dp/diagnostics/b08_errors
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent
VMIN, VMAX = 185.0, 255.0


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON metadata: {path}") from error


def load_case(prediction: dict, sample: dict, label: dict, index: dict,
              brightness: np.ndarray, packed: np.ndarray) -> dict:
    from h9dp.hsd import brightness_temperature, earth_mask, read_counts

    record = int(prediction["record"])
    assert label["name"] == sample["name"]
    assert label["time"]["year"] == sample["year"]
    assert label["time"]["iso_utc"].replace("Z", "+00:00") == sample["target_time"]
    assert np.isclose(float(prediction["MOTIS_dP"]), label["dP"], atol=1e-4)
    assert prediction["storm_id"] == sample["storm_id"]
    images, frames = [], []
    for frame_id, offset, center in zip(
        sample["frames"], sample["time_offsets_minutes"], sample["centers"], strict=True
    ):
        frame = index["frames"][frame_id]
        path = Path(frame["files"][index["bands"].index(8)])
        header, counts, checksum = read_counts(path)
        assert header.band == 8
        assert header.observation_time.isoformat() == frame["observation_time"]
        x0, y0 = frame["window"]
        counts = counts[y0:y0 + 500, x0:x0 + 500]
        raw, raw_valid = brightness_temperature(
            counts, header, earth_mask(header, (x0, y0), 500)
        )
        cached = np.asarray(brightness[frame_id, index["bands"].index(8)])
        cached_valid = np.unpackbits(
            packed[frame_id, index["bands"].index(8)], count=500 * 500,
            bitorder="little",
        ).reshape(500, 500).astype(bool)
        assert raw.shape == (500, 500)
        assert np.array_equal(raw_valid, cached_valid), f"Mask mismatch: {path}"
        assert np.array_equal(raw, cached), f"Brightness mismatch: {path}"
        images.append(np.ma.array(raw, mask=~raw_valid))
        frames.append({
            "frame_id": frame_id, "observation_time_utc": header.observation_time.isoformat(),
            "offset_minutes": offset, "label_xy": center, "source": frame["source"],
            "b08_file": str(path), "sha256": checksum,
            "cache_matches_raw_hsd": True,
            "valid_fraction": float(raw_valid.mean()),
            "minimum_kelvin": float(raw[raw_valid].min()),
            "maximum_kelvin": float(raw[raw_valid].max()),
        })
    return {
        "record": record, "name": sample["name"], "year": sample["year"],
        "quality": sample["quality"], "target_time_utc": sample["target_time"],
        "target_dP": float(prediction["MOTIS_dP"]),
        "predicted_dP": float(prediction["predicted_dP"]),
        "error_hpa": float(prediction["error_hPa"]),
        "label_location": label["location"], "label_satellite": label["satellite"],
        "label_sat_zen_degrees": label["sat_zen"], "label_pixels": label["pixels"],
        "frames": frames, "images": images,
    }


def case_title(case: dict) -> str:
    return (
        f"{case['year']} {case['name']} | record {case['record']} | {case['quality']} | "
        f"target {case['target_time_utc'][:16].replace('T', ' ')} UTC\n"
        f"MOTIS dP {case['target_dP']:.2f} | predicted {case['predicted_dP']:.2f} | "
        f"error {case['error_hpa']:+.2f} hPa"
    )


def draw_frame(ax, image: np.ndarray, frame: dict):
    artist = ax.imshow(image, cmap="gray_r", vmin=VMIN, vmax=VMAX,
                       origin="upper", interpolation="nearest")
    x, y = frame["label_xy"]
    ax.plot(x, y, "+", color="#32ff5b", markersize=12, markeredgewidth=1.5)
    stamp = frame["observation_time_utc"]
    ax.set_title(
        f"{stamp[:10]} {stamp[11:19]} UTC\n"
        f"dt = {frame['offset_minutes']:+.2f} min | {frame['source']}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def save_plots(cases: list[dict], output: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    color_scale = ScalarMappable(norm=Normalize(VMIN, VMAX), cmap="gray_r")
    for case in cases:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.8))
        fig.subplots_adjust(left=0.02, right=0.90, bottom=0.055,
                            top=0.84, wspace=0.055, hspace=0.24)
        for ax, image, frame in zip(axes.flat, case["images"], case["frames"], strict=True):
            draw_frame(ax, image, frame)
        fig.suptitle(case_title(case), fontsize=13, y=0.975)
        color_ax = fig.add_axes((0.925, 0.15, 0.016, 0.61))
        fig.colorbar(color_scale, cax=color_ax, label="B08 brightness temperature (K)")
        fig.text(0.02, 0.015, "Chronological order: left to right, top to bottom. "
                 "White = colder; black = warmer. Green + = label coordinate (not an eye detection).",
                 fontsize=8)
        filename = f"record_{case['record']}_{case['year']}_{case['name']}_B08.png"
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        case["plot"] = filename

    fig, axes = plt.subplots(len(cases), 6, figsize=(20, 3.9 * len(cases)), squeeze=False)
    fig.subplots_adjust(left=0.015, right=0.925, bottom=0.025,
                        top=0.90, wspace=0.045, hspace=0.57)
    for row, case in zip(axes, cases, strict=True):
        for ax, image, frame in zip(row, case["images"], case["frames"], strict=True):
            draw_frame(ax, image, frame)
        position = row[0].get_position()
        fig.text(0.018, position.y1 + 0.047, case_title(case), fontsize=11, va="bottom")
    fig.suptitle("Top validation errors at best checkpoint: six-frame B08 timelines",
                 fontsize=17, y=0.994)
    color_ax = fig.add_axes((0.945, 0.15, 0.010, 0.68))
    fig.colorbar(color_scale, cax=color_ax, label="B08 brightness temperature (K)")
    fig.text(0.018, 0.008, "All panels: native 500 x 500, identical 185-255 K scale, no augmentation. "
             "Green + = record label coordinate projected into each frame; not the historical eye track.",
             fontsize=10)
    fig.savefig(output / "overview_B08.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    sys.path.insert(0, str(ROOT / "src"))
    run = read_json(args.run_dir / "run.json")
    metrics = read_json(args.run_dir / "best_metrics.json")
    cache = Path(run["configuration"]["data"]["cache_dir"])
    index = read_json(cache / "index.json")
    assert run["data_fingerprint"] == index["fingerprint"]
    labels = read_json(Path(run["configuration"]["data"]["labels"]))["records"]
    with (args.run_dir / "best_validation.csv").open(newline="") as stream:
        predictions = sorted(csv.DictReader(stream),
                             key=lambda row: abs(float(row["error_hPa"])), reverse=True)[:args.top]
    samples = {sample["record"]: sample for sample in index["samples"]}
    brightness = np.load(cache / "brightness.npy", mmap_mode="r")
    packed = np.load(cache / "validity.npy", mmap_mode="r")
    cases = []
    for prediction in predictions:
        record = int(prediction["record"])
        assert samples[record]["split"] == "validation"
        cases.append(load_case(prediction, samples[record], labels[record], index, brightness, packed))
    if shutil.disk_usage("/").used + 30_000_000 > 375_000_000_000:
        raise RuntimeError("Projected root filesystem usage exceeds 375 GB")
    output = args.output_dir or args.run_dir / "b08_errors"
    output.mkdir(parents=True, exist_ok=True)
    save_plots(cases, output)
    report = {
        "run_dir": str(args.run_dir.resolve()), "best_epoch": metrics["epoch"],
        "selection": "descending absolute error in best_validation.csv",
        "color_scale_kelvin": [VMIN, VMAX],
        "marker": "MOTIS target-time label coordinate projected into each frame; not historical eye position",
        "cases": [{k: v for k, v in case.items() if k != "images"} for case in cases],
    }
    (output / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(cases)} timelines, overview and metadata to {output}")
    print(f"Verified {sum(len(case['frames']) for case in cases)} B08 frames against original HSD.")
    for case in cases:
        print(case_title(case).replace("\n", " | "))


if __name__ == "__main__":
    main()
