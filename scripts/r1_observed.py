#!/usr/bin/env python3
"""Bridge local Himawari frames to the native pressure levels of T-PARCII sondes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import numpy as np

from h9dp.observations import read_tracks, track_at
from h9dp.r1 import FRAME_COUNT, frame_features
from scripts.r1_adaptive_kernel import assemble
from scripts.r1_gp import configure, gp_predict
from scripts.r1_kernel import Cache, load_cache, read_json

LOCAL_PROFILES = {
    "2017-10": ("Lan", 2017, "2017288N09138"),
    "2022-09": ("Nanmadol", 2022, "2022254N24143"),
}
SOURCE_URL = "https://zenodo.org/records/10798896"


def terminal_pressure(file: Path) -> dict:
    """Use the clock and coordinates of the actual last valid pressure, not launch time."""
    launch = datetime.strptime(file.name[4:17], "%Y%m%dT%H%M").replace(tzinfo=UTC)
    with file.open(newline="", encoding="utf-8-sig") as stream:
        rows = [{key.strip(): part.strip() for key, part in row.items() if key and part is not None}
                for row in csv.DictReader(stream)]
    usable = []
    for row in rows:
        try:
            pressure = float(row["Press"])
        except (KeyError, ValueError):
            continue
        if np.isfinite(pressure) and pressure > 0:
            usable.append((row, pressure))
    if not usable:
        raise ValueError(f"No native pressure levels in {file}")
    row, pressure = usable[-1]
    seconds = int(row["sec"])
    timestamp = None
    time_basis = "no usable UTC clock"
    # Older native files use H:MM:SS. If the terminal clock is absent, use the
    # sonde-relative elapsed counter from a previous row stamped in UTC.
    for anchor, _ in reversed(usable):
        try:
            clock = time.fromisoformat(anchor["h:m:s"].zfill(8))
            anchor_seconds = int(anchor["sec"])
        except (KeyError, ValueError):
            continue
        stamp = datetime.combine(launch.date(), clock, tzinfo=UTC)
        nominal = launch + timedelta(seconds=anchor_seconds)
        stamp = min((stamp + timedelta(days=i) for i in (-1, 0, 1)), key=lambda x: abs(x - nominal))
        timestamp = stamp + timedelta(seconds=seconds - anchor_seconds)
        time_basis = "native UTC clock" if anchor_seconds == seconds else "native UTC clock plus sonde elapsed seconds"
        break
    try:
        lat, lon = float(row["lat"]), float(row["lon"])
    except (KeyError, ValueError):
        lat = lon = float("nan")
    return {"profile": str(file.resolve()), "source_url": SOURCE_URL, "launch_minute_utc": launch,
            "pressure_time_utc": timestamp, "original_clock_utc": row.get("h:m:s"),
            "elapsed_seconds": seconds, "terminal_time_basis": time_basis, "pressure_hpa": pressure, "altitude_m": None,
            "latitude": lat if abs(lat) <= 90 else None, "longitude": lon if abs(lon) <= 180 else None,
            "pressure_type": "near-surface last valid level of native eye dropsonde profile; not central pressure"}


def select_frames(candidates: list[tuple[datetime, int, int]], when: datetime, max_age: float):
    """Choose nearest available past scan for each slot, without using the same scan twice."""
    remaining = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))
    selected: list[tuple[datetime, int, int] | None] = [None] * FRAME_COUNT
    for j, offset in reversed(list(enumerate((-300, -180, -60, -30, -10, 0)))):
        want = when + timedelta(minutes=offset)
        eligible = [(i, choice) for i, choice in enumerate(remaining)
                    if choice[0] <= when and abs((choice[0] - want).total_seconds()) <= 60 * max_age]
        if eligible:
            i, value = min(eligible, key=lambda kv: (abs(kv[1][0]-want), -kv[1][0].timestamp()))
            selected[j] = value
            remaining = [x for k, x in enumerate(remaining) if k != i and x[0] != value[0]]
    return selected


def prepare_observations(cache: Cache, native_dir: Path, tracks_file: Path, catalog: Path, max_age: float,
                         *, snapshot: bool = False):
    catalog_rows, _ = read_json(catalog)
    times = [datetime.fromisoformat(catalog_rows["records"][r["record"]]["time"]["iso_utc"]) for r in cache.rows]
    with tracks_file.open(newline="", encoding="utf-8-sig") as stream:
        tracks = read_tracks(stream, 2017, 2022)
    manifest, _ = read_json(cache.root / "manifest.json")
    imagery = Path(manifest["source"])
    images, masks = (np.load(imagery / f"all_{kind}.npy", mmap_mode="r", allow_pickle=False)
                     for kind in ("images", "masks"))
    frame_shape = {"profile": cache.data["profile"].shape[-1:], "structure": cache.data["structure"].shape[-1:],
                   "regions": cache.data["regions"].shape[-1:]}
    provenance, data_rows, view_rows = [], [], []
    for file in sorted(native_dir.glob("DROP*.csv")):
        obs = terminal_pressure(file)
        launch = obs["launch_minute_utc"]
        if launch.strftime("%Y-%m") not in LOCAL_PROFILES:
            continue
        name, year, sid = LOCAL_PROFILES[launch.strftime("%Y-%m")]
        when = obs["pressure_time_utc"]
        obs.update(name=name, year=year, ibtracs_sid=sid, status="missing native terminal timestamp")
        provenance.append(obs)
        if when is None:
            continue
        env = track_at(tracks[sid], when.replace(tzinfo=None))
        if env is None or env[2].poci is None:
            obs["status"] = "no qualifying environmental POCI"
            continue
        poci = env[2]
        dp = poci.poci - obs["pressure_hpa"]
        obs.update(poci_hpa=poci.poci, poci_source="IBTrACS v04r01 USA_POCI, environment only",
                   poci_time_utc=poci.time.replace(tzinfo=UTC), dp_hpa=round(dp, 3))
        if dp <= 30:
            obs["status"] = "dP <= 30 hPa"
            continue
        scans = [(times[i] + timedelta(minutes=float(dt)), i, j)
                 for i, row in enumerate(cache.rows) if row["name"].casefold() == name.casefold() and row["year"] == year
                 for j, dt in enumerate(cache.data["time"][i]) if cache.data["valid"][i, j]]
        selected = select_frames(scans, when, max_age)
        if selected[-1] is None:
            obs["status"] = "no compatible recent past satellite scan in local cache"
            continue
        available = [x[0] for x in selected if x is not None]
        if len(available) < 2 and not snapshot:
            obs["status"] = "only one usable local satellite scan; not a sequence score"
            obs["available_scans"] = available
            continue
        if snapshot:
            selected[:-1] = [None] * (FRAME_COUNT - 1)
        data: dict[str, np.ndarray] = {k: np.full((FRAME_COUNT, *shape), np.nan, dtype=np.float32) for k, shape in frame_shape.items()}
        data.update(time=np.zeros(FRAME_COUNT), valid=np.zeros(FRAME_COUNT, dtype=bool))
        views = np.zeros((FRAME_COUNT, 10, 64, 64), dtype=np.int16)
        sources = []
        for k, scan in enumerate(selected):
            if scan is None:
                continue
            stamp, i, j = scan
            # Descriptor extraction is deliberately identical to R1 training; do not use sonde pressure to center.
            center = np.asarray(cache.rows[i]["centers"][j])
            item = frame_features(images[i, j], masks[i, j, 0], center)
            data["profile"][k], data["structure"][k], views[k] = item.profile, item.structure, item.views
            data["regions"][k] = cache.data["regions"][i, j]
            data["time"][k] = (stamp - when).total_seconds() / 60
            data["valid"][k] = True
            sources.append({"record": cache.rows[i]["record"], "slot": j, "scan_utc": stamp,
                            "scan_time_basis": cache.rows[i]["time_basis"], "selected_age_minutes": float(-data["time"][k])})
        obs.update(status="predicted native near-surface proxy", scans=sources)
        data_rows.append(data)
        view_rows.append(views)
    return provenance, data_rows, view_rows


def evaluate(args):
    cache = load_cache(args.features)
    rows, descriptors, views = prepare_observations(cache, args.native, args.ibtracs, args.catalog, args.max_age,
                                                       snapshot=args.snapshot)
    args.output.mkdir(exist_ok=False)
    result = {"schema": "r1-native-sonde-bridge/v1", "profiles": len(rows), "paired": len(descriptors),
              "max_slot_offset_min": args.max_age, "features": str(cache.root), "source_meta_sha256": cache.source_digest,
              "steps": args.steps, "spectral": args.spectral, "contour": args.contour, "mode": "snapshot" if args.snapshot else "sequence",
              "observations": rows, "scope": "native terminal pressure proxy, not verified cyclone-central pressure"}
    if descriptors:
        configure(args.output)
        external = {k: np.stack([d[k] for d in descriptors]) for k in descriptors[0]}
        valid_rows = [r for r in rows if r["status"] == "predicted native near-surface proxy"]
        groups = assemble(cache, external, np.stack(views), contour=args.contour, spectral=args.spectral,
                          snapshot=args.snapshot)
        train = np.arange(len(cache.rows))
        held = np.arange(len(cache.rows), len(cache.rows) + len(views))
        forecasts, _, timings = gp_predict(cache, groups, train, held, args.steps, 0.035)
        result["timing"] = timings
        prediction_error = []
        for r, prediction in zip(valid_rows, forecasts, strict=True):
            p = float(prediction)
            error = p - r["dp_hpa"]
            r.update(predicted_dp_hpa=p, predicted_pressure_hpa=r["poci_hpa"] - p, dp_error_hpa=error)
            prediction_error.append(error)
        e = np.asarray(prediction_error)
        result["metrics_descriptive_only"] = {"n": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))),
                                              "mae": float(np.mean(np.abs(e))), "bias": float(e.mean())}
    (args.output / "observations.json").write_text(json.dumps(result, default=str, indent=2) + "\n")
    print(json.dumps(result, default=str), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--native", required=True, type=Path)
    p.add_argument("--ibtracs", required=True, type=Path)
    p.add_argument("--catalog", type=Path, default=Path("MOTIS_cyclone_records.json"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-age", type=float, default=20)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--contour", action="store_true")
    p.add_argument("--spectral", action="store_true")
    p.add_argument("--snapshot", action="store_true", help="Match training and observation to exactly one scan")
    args = p.parse_args()
    if args.max_age <= 0 or args.steps <= 0:
        p.error("Maximum scan offset and steps must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
