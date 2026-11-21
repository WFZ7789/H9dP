"""Collect real WP surface SLP and analyzed POCI as review-only candidates.

This does not produce certified TC central pressures or train/evaluate a model.
All downloads and generated artifacts stay outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from h9dp.observations import (
    canonical_name,
    distance_km,
    pair_surface_rows,
    read_stations,
    read_tracks,
    select_pairs,
    track_at,
)

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.WP.list.v04r01.csv"
)
STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GLOBAL_HOURLY_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com"


def fetch(url: str, path: Path) -> Path:
    """Cache exact source bytes and provenance; reject HTML redirect destinations."""
    if urlsplit(url).scheme != "https":
        raise ValueError("Downloads require HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.with_suffix(path.suffix + ".json").exists():
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(url, timeout=90) as response, temporary.open("wb") as out:
                first = response.read(65536)
                if first.lstrip().startswith(b"<"):
                    raise ValueError(f"Expected CSV, got HTML/XML: {response.url}")
                out.write(first)
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                final_url = response.url
            temporary.replace(path)
            metadata = {
                "url": url, "final_url": final_url,
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "proxy_environment_removed": True,
            }
            path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
            return path
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-station-years", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-proxy-dp-hpa", type=float, default=30,
                        help="Surface-review threshold only; does not certify central dP")
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0 or args.max_station_years <= 0:
        parser.error("count, workers and max-station-years must be positive")
    if args.first_year > args.last_year:
        parser.error("first-year must not exceed last-year")
    if not math.isfinite(args.min_proxy_dp_hpa) or args.min_proxy_dp_hpa < 30:
        parser.error("surface-review proxy dP threshold must be finite and at least 30 hPa")
    output = args.output.resolve()
    if not output.is_relative_to(Path("/mnt/data/H9dP")):
        parser.error("output must be below /mnt/data/H9dP")
    for name in list(os.environ):
        if "proxy" in name.casefold():
            del os.environ[name]
    root_stats = os.statvfs("/")
    root_used = (root_stats.f_blocks - root_stats.f_bfree) * root_stats.f_frsize
    if root_used > 375_000_000_000:
        raise RuntimeError("Root filesystem exceeds repository's 375 GB cap")
    output.mkdir(parents=True, exist_ok=True)
    data_stats = os.statvfs(output)
    # Conservative 20 MB per annual station CSV, plus the ~115 MB WP catalogue.
    projected_bytes = args.max_station_years * 20_000_000 + 200_000_000
    if data_stats.f_bavail * data_stats.f_frsize < projected_bytes:
        raise RuntimeError("Insufficient /mnt/data space for the projected download")
    print(json.dumps({"phase": "start", "root_used_bytes": root_used,
                      "projected_bytes": projected_bytes, "proxy_environment_removed": True}), flush=True)
    raw = output / "raw"
    track_path = fetch(IBTRACS_URL, raw / "ibtracs.WP.list.v04r01.csv")
    stations_path = fetch(STATIONS_URL, raw / "isd-history.csv")
    with track_path.open() as source:
        all_tracks = read_tracks(source, args.first_year, args.last_year)
    try:
        records_bytes = args.records.read_bytes()
        records = json.loads(records_bytes)["records"]
        exposed = {(int(row["time"]["year"]), canonical_name(row["name"])) for row in records}
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot read existing cyclone inventory: {error}")
    # Cyclone identity and year are metadata, never collection eligibility gates.
    tracks = all_tracks
    inventory_matches = [
        {"sid": sid, "name": points[0].name, "season": points[0].season}
        for sid, points in tracks.items()
        if (points[0].season, canonical_name(points[0].name)) in exposed
    ]
    existing_times = {(canonical_name(row["name"]), row["time"]["iso_utc"]) for row in records}
    with stations_path.open() as source:
        stations = read_stations(source)
    # Densify positions only; POCI is never interpolated by this collector.
    hourly = defaultdict(list)
    for sid, points in tracks.items():
        time_utc = points[0].time.replace(minute=0, second=0)
        while time_utc <= points[-1].time:
            match = track_at(points, time_utc)
            if match:
                lat, lon, _ = match
                hourly[time_utc.year].append((lat, lon, sid))
            time_utc += timedelta(hours=1)
    jobs = []
    for station in stations:
        for year, positions in hourly.items():
            if not station.begin <= year <= station.end:
                continue
            near = [(distance_km(station.lat, station.lon, lat, lon), sid)
                    for lat, lon, sid in positions
                    if abs(station.lat - lat) <= 0.3 and abs(station.lon - lon) <= 0.5]
            near = [(distance, sid) for distance, sid in near if distance <= 25]
            if near:
                jobs.append((min(distance for distance, _ in near), station, year,
                             sorted({sid for _, sid in near})))
    jobs.sort(key=lambda job: (job[0], job[1].identifier, job[2]))
    planned_jobs = len(jobs)
    jobs = jobs[:args.max_station_years]
    print(json.dumps({"phase": "planned", "tracks": len(tracks),
                      "existing_cyclone_tags": len(inventory_matches), "station_years": len(jobs),
                      "available_station_years": planned_jobs}), flush=True)
    failures, pairs, completed = [], [], 0

    def collect(job):
        _, station, year, sids = job
        url = f"{GLOBAL_HOURLY_URL}/{year}/{station.identifier}.csv"
        path = fetch(url, raw / "global_hourly" / str(year) / f"{station.identifier}.csv")
        body = path.read_text()
        result = []
        for sid in sids:
            result.extend(pair_surface_rows(io.StringIO(body), station, tracks[sid], url))
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                pairs.extend(future.result())
            except (OSError, ValueError, csv.Error, KeyError) as error:
                failures.append({"station_id": job[1].identifier, "year": job[2], "error": str(error)})
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(json.dumps({"phase": "collect", "done": completed,
                                  "raw_pairs": len(pairs), "failures": len(failures)}), flush=True)
    pairs.sort(key=lambda row: (row["observation_time_utc"], row["sid"], row["station_id"],
                               row["observation_source_line"]))
    for row in pairs:
        name = canonical_name(row["name"])
        row["cyclone_in_existing_inventory"] = (row["season"], name) in exposed
        row["exact_named_time_in_existing_inventory"] = (name, row["observation_time_utc"]) in existing_times
    selected = select_pairs(pairs, args.count, args.seed, args.min_proxy_dp_hpa)
    save_csv(output / "all_surface_poci_pairs.csv", pairs)
    save_csv(output / "review_candidates.csv", selected)
    (output / "existing_cyclone_matches.json").write_text(json.dumps(inventory_matches, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    manifest = {
        "status": "strong_surface_review_only_not_central_pressure_ground_truth",
        "protocol_version": 2,
        "sampling_unit": "physical_observation",
        "cyclone_role": "tag_only", "year_role": "tag_only",
        "minimum_proxy_dp_hpa": args.min_proxy_dp_hpa,
        "group_quotas": False, "within_cyclone_time_separation_hours": 0,
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "root_used_bytes_before": root_used, "projected_bytes": projected_bytes,
        "proxy_environment_removed": True, "ibtracs_url": IBTRACS_URL,
        "existing_records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "existing_records_count": len(records), "excluded_exposed_cyclones": 0,
        "existing_cyclone_tags": len(inventory_matches),
        "station_years_planned": planned_jobs, "station_years_attempted": len(jobs),
        "failed_station_years": len(failures), "raw_pairs": len(pairs),
        "selected_pairs": len(selected), "selected_cyclones": len({row['sid'] for row in selected}),
        "selected_by_year": dict(Counter(row["season"] for row in selected)),
        "within_10_km": sum(row["centre_distance_km"] <= 10 for row in selected),
        "certified_central_pressure_records": 0, "model_evaluations": 0,
        "caveats": [
            "SLP is measured/reduced station sea-level pressure, not cyclone central pressure.",
            "USA_POCI is a best-track analysis and can be copied/interpolated upstream by IBTrACS.",
            "Track position is interpolated over <=6 h; nearest POCI retained with <=3 h offset.",
            "Cyclone/year and existing-record matches are tags only, never exclusion or group-sampling gates.",
            "Land/terrain/eye structure and satellite availability need review before final validation.",
            "No forecast or central-pressure accuracy can be claimed from these candidates alone.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
