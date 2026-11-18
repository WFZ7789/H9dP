#!/usr/bin/env python3
"""Build a label-free feature cache from the prepared 546-record Himawari cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from h9dp.r1 import FEATURE_SCHEMA, FRAME_COUNT, frame_features
from h9dp.r1_track import track_centers
from scripts.m1 import region_map

_STORE: dict | None = None
ROOT_LIMIT = 375_000_000_000


def initialize(root: Path):
    global _STORE
    _STORE = {key: np.load(root / f"all_{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("images", "masks", "numeric", "coverage")}


def extract(task: tuple[int, np.ndarray, np.ndarray, bool]):
    if _STORE is None:
        raise RuntimeError("Source cache was not initialized")
    index, centers, times, track = task
    images = _STORE["images"][index]
    masks = _STORE["masks"][index, :, 0]
    measured = masks.any(axis=(1, 2)) & np.isfinite(times)
    if not measured.any():
        raise ValueError(f"No usable observations for cache row {index}")
    offsets = np.full((FRAME_COUNT, 3), np.nan, dtype=np.float32)
    if track:
        centers, offsets = track_centers(images, masks, centers, times, measured)
    features = {i: frame_features(images[i], masks[i], centers[i]) for i in np.flatnonzero(measured)}
    example = next(iter(features.values()))
    profile = np.full((FRAME_COUNT, example.profile.size), np.nan, dtype=np.float32)
    structure = np.full((FRAME_COUNT, example.structure.size), np.nan, dtype=np.float32)
    views = np.zeros((FRAME_COUNT, *example.views.shape), dtype=np.int16)
    for i, item in features.items():
        profile[i], structure[i], views[i] = item.profile, item.structure, item.views
    regions = np.where(_STORE["coverage"][index, :, None, :, None] > 0,
                       _STORE["numeric"][index], np.nan).copy()
    for i in np.flatnonzero(offsets[:, 0] >= 0.75):
        areas = region_map(centers[i], masks.shape[-1], (32, 96))
        for region in range(3):
            pixels = (areas == region) & masks[i]
            if pixels.any():
                values = images[i][:, pixels].astype(np.float32) * 0.01 + 150
                regions[i, :, region] = np.stack((values.mean(axis=-1), values.min(axis=-1), values.max(axis=-1)), -1)
            else:
                regions[i, :, region] = np.nan
    return profile, structure, regions.reshape(FRAME_COUNT, -1), views, measured, offsets


def prepare(source: Path, output: Path, workers: int, track: bool):
    metadata_path = source / "all_meta.json"
    try:
        raw = metadata_path.read_bytes()
        metadata = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read source metadata from {metadata_path}") from exc
    if metadata.get("schema") != "m1-adapt-data/v1":
        raise ValueError("Unsupported source cache")
    records = metadata["records"]
    ids = [int(row["record"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated record IDs in source cache")
    centers = np.array([row["centers"] for row in records])
    times = np.array([row["times"] for row in records])
    estimated = len(records) * FRAME_COUNT * 10 * 64 * 64 * 2 + 64 * 1024 ** 2
    fs_root = shutil.disk_usage("/")
    fs_dest = shutil.disk_usage(output.parent)
    if fs_root.used >= ROOT_LIMIT or fs_dest.free < estimated or (
        output.parent.stat().st_dev == Path("/").stat().st_dev and fs_root.used + estimated >= ROOT_LIMIT
    ):
        raise OSError("Insufficient space under the storage policy")
    output.mkdir(exist_ok=False)
    views = np.lib.format.open_memmap(output / "views.npy", mode="w+", dtype=np.int16,
                                      shape=(len(records), FRAME_COUNT, 10, 64, 64))
    rows, structures, regions, valid, registrations = [], [], [], [], []
    tasks = [(i, center, time, track) for i, (center, time) in enumerate(zip(centers, times, strict=True))]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(source,)) as pool:
        for i, item in enumerate(pool.map(extract, tasks, chunksize=3)):
            profile, structure, region, views[i], available, offsets = item
            registrations.append(offsets)
            rows.append(profile)
            structures.append(structure)
            regions.append(region)
            valid.append(available)
            if (i + 1) % 50 == 0:
                print(f"features {i + 1}/{len(records)}", flush=True)
    views.flush()
    valid = np.stack(valid)
    registration = np.stack(registrations)
    np.savez_compressed(output / "features.npz", profile=np.stack(rows), structure=np.stack(structures),
                        regions=np.stack(regions), time=times, valid=valid, record=np.asarray(ids), registration=registration)
    manifest = {"schema": FEATURE_SCHEMA, "records": len(records), "usable_frames": int(valid.sum()),
                "centers": "column,line; edge origin (pixel center at i+0.5)",
                "tracking": track, "accepted_translations": int((registration[..., 2] == 1).sum()),
                "median_accepted_shift_px": (float(np.median(registration[registration[..., 2] == 1, 0]))
                                             if (registration[..., 2] == 1).any() else None),
                "source": str(source.resolve()), "source_meta_sha256": hashlib.sha256(raw).hexdigest(),
                "workers": workers, "views_shape": list(views.shape), "view_units": "(K - 240) * 100",
                "source_sizes": {p.name: p.stat().st_size for p in sorted(source.glob("all_*.npy"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--track", action="store_true", help="Register earlier scenes to later imagery within each record")
    args = parser.parse_args()
    prepare(args.source, args.output, args.workers, args.track)


if __name__ == "__main__":
    main()
