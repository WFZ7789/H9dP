"""Collect real JMA station pressure observations; never use BST central pressure.

POCI is an independent environmental field from the previously approved source.
It does not create, replace, interpolate, or correct any observed pressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import threading
import time as clock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from h9dp.jma_observations import parse_jma_10min
from h9dp.observations import canonical_name, read_tracks
from scripts.collect_recon_poci import download
from scripts.collect_surface_poci import save_csv

# Observed events/stations to query, not pressure values from cyclone analyses.
CASES = (
    ("Goni", 2015, "47918", "石垣島", 91, "2015-08-23", "2015-08-24"),
    ("Chaba", 2016, "47929", "久米島", 91, "2016-10-03", "2016-10-04"),
    ("Trami", 2018, "47936", "那覇", 91, "2018-09-29", "2018-09-30"),
    ("Nanmadol", 2022, "47827", "鹿児島", 88, "2022-09-18", "2022-09-19"),
)


def observation_jobs(cases_path: Path | None = None) -> list[tuple]:
    """Expand explicit source queries, never cyclone-held-out or quota sampling."""
    cases = list(CASES)
    if cases_path is not None:
        try:
            config = json.loads(cases_path.read_text())
            cases = []
            for event in config["events"]:
                for station in event["stations"]:
                    info = config["stations"][station]
                    cases.append((event["name"], int(event["year"]), station, info["name"],
                                  int(info["prefecture"]), event["first_day_jst"], event["last_day_jst"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid JMA query configuration: {error}") from error
    jobs = {}
    for name, year, station, station_name, pref, first, last in cases:
        if not isinstance(station, str) or len(station) != 5 or not station.isascii() or not station.isdigit():
            raise ValueError("JMA station identifier must contain five ASCII digits")
        day, end = date.fromisoformat(first), date.fromisoformat(last)
        if day > end:
            raise ValueError("JMA query date range is reversed")
        while day <= end:
            key = station, day
            job = name, year, station, station_name, pref, day
            if key in jobs and jobs[key][:2] != job[:2]:
                raise ValueError(f"Ambiguous event tags for station/day {key}")
            jobs[key] = job
            day += timedelta(days=1)
    return sorted(jobs.values(), key=lambda job: (job[5], job[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poci-csv", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-request-interval", type=float, default=0.5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("count and workers must be positive")
    if not math.isfinite(args.min_request_interval) or args.min_request_interval < 0:
        parser.error("request interval must be finite and nonnegative")
    root = args.output.resolve()
    if not root.is_relative_to(Path("/mnt/data/H9dP")):
        parser.error("output must be below /mnt/data/H9dP")
    for name in list(os.environ):
        if "proxy" in name.casefold():
            del os.environ[name]
    stats = os.statvfs("/")
    root_used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    if root_used > 375_000_000_000:
        raise RuntimeError("Root filesystem exceeds 375 GB")
    root.mkdir(parents=True, exist_ok=True)
    jobs = observation_jobs(args.cases_json)
    projected_bytes = len(jobs) * 250_000 + 50_000_000
    space = os.statvfs(root)
    if space.f_bavail * space.f_frsize < projected_bytes:
        raise RuntimeError("Insufficient external storage for projected observation bank")
    plan = {"phase": "plan", "station_days": len(jobs),
            "stations": len({job[2] for job in jobs}),
            "cyclone_year_tags": len({job[:2] for job in jobs}),
            "projected_bytes": projected_bytes, "root_used_bytes": root_used,
            "minimum_request_interval_seconds": args.min_request_interval,
            "primary_delivery": "all_jma_dp_gt30.csv; sample export is not a bank-size limit"}
    print(json.dumps(plan), flush=True)
    if args.plan_only:
        return
    try:
        records_bytes = args.records.read_bytes()
        records = json.loads(records_bytes)["records"]
        existing = {(int(row["time"]["year"]), canonical_name(row["name"])) for row in records}
        existing_times = {(canonical_name(row["name"]), row["time"]["iso_utc"]) for row in records}
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot read record tags: {error}")
    # No USA_PRES, WMO_PRES or TOKYO_PRES is read or used as an observed pressure.
    with args.poci_csv.open() as source:
        tracks = read_tracks(source, 2015, 2025)
    poci_by_event = {}
    for points in tracks.values():
        for point in points:
            if point.poci is not None:
                poci_by_event.setdefault((point.season, canonical_name(point.name)), []).append(point)
    request_lock = threading.Lock()
    last_request = 0.0

    def collect(job):
        nonlocal last_request
        name, year, station, station_name, pref, day = job
        url = ("https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?"
               f"prec_no={pref}&block_no={station}&year={day.year}&month={day.month}&day={day.day}&view=")
        path = root / "raw" / f"{station}_{day:%Y%m%d}_10min.html"
        if not (path.exists() and path.with_suffix(".html.json").exists()):
            with request_lock:
                remaining = args.min_request_interval - (clock.monotonic() - last_request)
                if remaining > 0:
                    clock.sleep(remaining)
                last_request = clock.monotonic()
        download(url, path)
        observations = parse_jma_10min(path.read_text(), day, station, url)
        for row in observations:
            row.update({"station_name": station_name, "cyclone_tag": name, "event_year_tag": year})
        return observations

    observations, failures, empty_days = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                rows = future.result()
                observations.extend(rows)
                if not rows:
                    empty_days.append({"station": job[2], "day_jst": str(job[5])})
            except (OSError, ValueError) as error:
                failures.append({"station": job[2], "day_jst": str(job[5]), "error": str(error)})
            if done % 25 == 0 or done == len(jobs):
                print(json.dumps({"phase": "collect", "station_days_done": done,
                                  "observed_rows": len(observations), "failed_days": len(failures)}), flush=True)
    observations.sort(key=lambda row: (row["observation_time_utc"], row["station_id"]))
    if len({(row["station_id"], row["observation_time_utc"]) for row in observations}) != len(observations):
        raise ValueError("Duplicate physical observations in full observation bank")
    paired = []
    for row in observations:
        key = row["event_year_tag"], canonical_name(row["cyclone_tag"])
        points = poci_by_event.get(key, [])
        if not points:
            continue
        time = datetime.fromisoformat(row["observation_time_utc"]).replace(tzinfo=None)
        point = min(points, key=lambda item: abs((item.time - time).total_seconds()))
        if point.poci is None or abs((point.time - time).total_seconds()) > 10800:
            continue
        name = canonical_name(row["cyclone_tag"])
        paired.append({
            **row, "poci_environment_hpa": point.poci,
            "poci_environment_source": "IBTrACS v04r01 USA_POCI; environment only, not observed pressure",
            "poci_time_utc": point.time.isoformat() + "Z",
            "poci_offset_minutes": (point.time - time).total_seconds() / 60,
            "dp_observed_station_hpa": round(point.poci - row["slp_observed_hpa"], 3),
            "cyclone_in_existing_inventory_tag": key in existing,
            "exact_named_time_in_existing_inventory_tag": (name, row["observation_time_utc"]) in existing_times,
        })
    qualified = [row for row in paired if row["dp_observed_station_hpa"] > 30]
    selected = random.Random(args.seed).sample(qualified, min(args.count, len(qualified)))
    selected.sort(key=lambda row: (row["observation_time_utc"], row["station_id"]))
    if len({(row["station_id"], row["observation_time_utc"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate physical observations")
    save_csv(root / "all_jma_pressure_measurements.csv", observations)
    save_csv(root / "all_jma_poci_pairs.csv", paired)
    save_csv(root / "all_jma_dp_gt30.csv", qualified)
    save_csv(root / f"jma_observations_dp_gt30_sample_{len(selected)}.csv", selected)
    if len(selected) == 100:
        save_csv(root / "jma_observations_dp_gt30_100.csv", selected)
    (root / "source_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (root / "source_empty_days.json").write_text(json.dumps(empty_days, indent=2) + "\n")
    full_dp = [row["dp_observed_station_hpa"] for row in qualified]
    dp = [row["dp_observed_station_hpa"] for row in selected]
    slp = [row["slp_observed_hpa"] for row in selected]
    manifest = {
        "status": "collected_JMA_station_pressure_observations_and_separate_environment_POCI",
        "completed_utc": datetime.now(UTC).isoformat(), "requested": args.count,
        "seed": args.seed, "root_used_bytes_before": root_used, "proxy_environment_removed": True,
        "source_station_days": len(jobs), "observed_pressure_rows": len(observations),
        "cases_json": str(args.cases_json) if args.cases_json else None,
        "projected_bytes": projected_bytes, "source_failed_days": len(failures),
        "source_empty_days": len(empty_days), "successful_source_days": len(jobs) - len(failures),
        "primary_delivery_file": "all_jma_dp_gt30.csv", "selected_export_is_sample_only": True,
        "observed_stations": len({row["station_id"] for row in observations}),
        "qualified_stations": len({row["station_id"] for row in qualified}),
        "qualified_station_days": len({(row["station_id"], row["source_day_jst"]) for row in qualified}),
        "qualified_cyclone_year_tags": len({(row["event_year_tag"], row["cyclone_tag"]) for row in qualified}),
        "qualified_by_year": dict(Counter(row["event_year_tag"] for row in qualified)),
        "full_dp_min_hpa": min(full_dp) if full_dp else None,
        "full_dp_max_hpa": max(full_dp) if full_dp else None,
        "full_dp_bins": {"30_to_50": sum(30 < value <= 50 for value in full_dp),
                         "50_to_70": sum(50 < value <= 70 for value in full_dp),
                         "above_70": sum(value > 70 for value in full_dp)},
        "poci_paired_rows": len(paired), "qualified_dp_gt30": len(qualified), "selected": len(selected),
        "selected_by_station": dict(Counter(row["station_name"] for row in selected)),
        "selected_by_cyclone_tag": dict(Counter(row["cyclone_tag"] for row in selected)),
        "selected_dp_min_hpa": min(dp) if dp else None, "selected_dp_max_hpa": max(dp) if dp else None,
        "selected_slp_min_hpa": min(slp) if slp else None, "selected_slp_max_hpa": max(slp) if slp else None,
        "all_selected_quality_classes": sorted({row["slp_quality_class"] for row in selected}),
        "best_track_pressure_labels_used": 0, "pressure_values_interpolated": 0,
        "station_pressure_relabelled_as_cyclone_central_pressure": 0,
        "cyclone_exclusions": 0, "training_time_buffers": 0, "per_cyclone_quotas": False,
        "existing_cyclone_tagged_records": sum(row["cyclone_in_existing_inventory_tag"] for row in selected),
        "record_tags_source_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "poci_csv_sha256": hashlib.sha256(args.poci_csv.read_bytes()).hexdigest(),
        "caveats": [
            "Observed pressure is directly transcribed from JMA station observation tables, not BST.",
            "Sea-level pressure may include JMA station sea-level reduction; local station pressure is also retained.",
            "Observed station pressure is not automatically cyclone central pressure; eye/centre context remains to be verified.",
            "POCI is the previously approved environmental source only, never a replacement for measured pressure.",
            "JMA clocks are JST (UTC+09), including 24:00; all timestamps are converted to UTC.",
            "Satellite sequences and historical station-location metadata are not prepared in this collection step.",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
