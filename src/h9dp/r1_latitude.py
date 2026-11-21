"""Native Himawari R3 navigation for R1; never geolocate a track-defined crop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from h9dp.hsd import pixel_latitude, read_header
from h9dp.prepare import SCHEMA as NATIVE_CACHE_SCHEMA
from h9dp.r1 import FRAME_COUNT, _normalize

LATITUDE_MODES = ("none", "center", "field")


def native_r3_paths(index: dict, records: np.ndarray, times: np.ndarray) -> list[list[Path | None]]:
    """Map unchanged native R3 images to record/frame slots, not from lat/lon or eye position."""
    if index.get("schema") != NATIVE_CACHE_SCHEMA or index.get("crop_size") != 500 or (
        times.shape != (len(records), FRAME_COUNT)
    ):
        raise ValueError("Expected a native500/v2 index and aligned six-frame times")
    samples = {row["record"]: row for row in index["samples"]}
    if len(samples) != len(index["samples"]) or len(set(records)) != len(records):
        raise ValueError("Repeated record in latitude source or feature cache")
    paths = []
    for record, time in zip(records, times, strict=True):
        sample = samples.get(record)
        if sample is None:
            paths.append([None] * FRAME_COUNT)
            continue
        if len(sample["frames"]) != FRAME_COUNT or not np.array_equal(time, sample["time_offsets_minutes"]):
            raise ValueError(f"Native image slots do not match feature times for record {record}")
        row = []
        for i in sample["frames"]:
            frame = index["frames"][i]
            if frame["source"] != "R3":
                row.append(None)
                continue
            if frame["window"] != [0, 0] or frame["source_shape"] != [500, 500]:
                raise ValueError(f"R3 frame {i} is not an unchanged native image")
            row.append(Path(frame["files"][0]))  # the index has verified co-navigation and B08 first
        paths.append(row)
    return paths


def latitude_fields(paths: list[list[Path | None]], valid: np.ndarray) -> np.ndarray:
    """Latitude at the centers of a fixed 3×3 grid in each latest usable native R3 image."""
    if valid.shape != (len(paths), FRAME_COUNT):
        raise ValueError("Latitude source and validity slots differ")
    fields = np.full((len(paths), 3, 3), np.nan)
    for i, (row, usable) in enumerate(zip(paths, valid, strict=True)):
        path = next((p for p, ok in zip(reversed(row), usable[::-1], strict=True) if p is not None and ok), None)
        if path is None:
            continue
        header = read_header(path)
        if not header.area.startswith("R3") or (header.band, header.height, header.width) != (8, 500, 500):
            raise ValueError(f"Not a native R3 B08 header: {path}")
        fraction = np.array([1, 3, 5]) / 6.0
        fields[i] = pixel_latitude(header, (header.width * fraction - 0.5)[None, :],
                                  (header.height * fraction - 0.5)[:, None])
    return fields


def latitude_kernel(fields: np.ndarray, train: np.ndarray, held: np.ndarray, *, with_field: bool) -> np.ndarray:
    """Fold-local PSD additive latitude term, exactly zero for rows without a center latitude."""
    if fields.ndim != 3 or fields.shape[1:] != (3, 3):
        raise ValueError("Expected one aligned 3×3 latitude grid per record")
    x = fields[np.r_[train, held]]
    center = x[:, 1, 1]
    available = np.isfinite(center)
    fit = available[:len(train)]
    if fit.sum() < 3:
        return np.zeros((len(x), len(x)), dtype=np.float64)
    values = center[:len(train)][fit]
    z = np.where(available, np.clip((center - values.mean()) / max(values.std(), 0.05), -5, 5), 0)
    knots = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    bumps = np.exp(-0.5 * ((z[:, None] - knots) / 0.85) ** 2)
    bumps /= np.linalg.norm(bumps, axis=1, keepdims=True)
    basis = np.column_stack((0.5 * z, bumps))
    basis = np.where(available[:, None], basis - basis[:len(train)][fit].mean(axis=0), 0)
    gram = basis @ basis.T
    if with_field:
        # Represent the shape of the geographic field, not nine copies of the center.
        anomaly = x.reshape(-1, 9)[:, [0, 1, 2, 3, 5, 6, 7, 8]] - center[:, None]
        a, b = _normalize(anomaly[:len(train)], anomaly[len(train):])
        if a.shape[1]:
            features = np.concatenate((a, b))
            gram += (0.25 / a.shape[1]) * (features @ features.T)
    return gram
