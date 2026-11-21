"""Collect NHC Pacific aircraft vortex centre pressures and IBTrACS POCI.

Cyclones and years are tags only. Geometry, pressure and source quality govern
eligibility; there are no cyclone quotas or training-proximity exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from h9dp.observations import canonical_name, read_tracks
from h9dp.recon import filename_time, pair_vdm, parse_vdm
from scripts.collect_surface_poci import save_csv


def download(url: str, path: Path) -> Path:
    if urlsplit(url).scheme != "https":
        raise ValueError("Only HTTPS source downloads are allowed")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() and metadata_path.exists():
        return path
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(url, timeout=30) as response:
                data = response.read()
                final_url = response.url
            path.write_bytes(data)
            metadata_path.write_text(json.dumps({
                "url": url, "final_url": final_url, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "proxy_environment_removed": True,
            }, indent=2) + "\n")
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("count and workers must be positive")
    root = args.output.resolve()
    if not root.is_relative_to(Path("/mnt/data/H9dP")):
        parser.error("output must be below /mnt/data/H9dP")
    for name in list(os.environ):
        if "proxy" in name.casefold():
            del os.environ[name]
    stats = os.statvfs("/")
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    if used > 375_000_000_000:
        raise RuntimeError("Root filesystem exceeds 375 GB")
    root.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(root)
    if stats.f_bavail * stats.f_frsize < 100_000_000:
        raise RuntimeError("Insufficient external storage for archive messages")
    with args.track_csv.open() as source:
        tracks = read_tracks(source, 2015, 2025, basins=None)
    by_atcf = defaultdict(list)
    for points in tracks.values():
        for point in points:
            if point.atcf_id:
                by_atcf[point.atcf_id].append(point)
    try:
        records_bytes = args.records.read_bytes()
        records = json.loads(records_bytes)["records"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot read existing record tags: {error}")
    known = {(int(row["time"]["year"]), canonical_name(row["name"])) for row in records}
    known_times = {(canonical_name(row["name"]), row["time"]["iso_utc"]) for row in records}
    failures, jobs = [], []
    raw = root / "raw" / "nhc"

    def index(year):
        url = f"https://www.nhc.noaa.gov/archive/recon/{year}/REPPN2/"
        path = download(url, raw / str(year) / "index.html")
        names = sorted(set(re.findall(r'REPPN2-KNHC\.\d{12}\.txt', path.read_text())))
        return [(url + name, raw / str(year) / name) for name in names]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(index, year): year for year in range(2015, 2026)}
        for future in as_completed(futures):
            try:
                jobs.extend(future.result())
            except (OSError, ValueError) as error:
                failures.append({"index_year": futures[future], "error": str(error)})
    jobs.sort(key=lambda job: str(job[1]))
    print(json.dumps({"phase": "planned", "archive_messages": len(jobs),
                      "index_failures": len(failures), "cyclone_exclusions": 0}), flush=True)
    observations, parse_failures = [], []

    def collect(job):
        url, path = job
        download(url, path)
        return parse_vdm(path.read_text(), filename_time(path.name), url)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            url, _ = futures[future]
            try:
                observation = future.result()
                if observation is None:
                    parse_failures.append(url)
                else:
                    observations.append(observation)
            except (OSError, ValueError) as error:
                failures.append({"url": url, "error": str(error)})
            if done % 100 == 0 or done == len(jobs):
                print(json.dumps({"phase": "download", "done": done,
                                  "parsed": len(observations), "parse_failures": len(parse_failures),
                                  "download_failures": len(failures)}), flush=True)
    # Corrections/retransmissions of the same physical aircraft fix are not new observations.
    unique = {}
    for row in sorted(observations, key=lambda row: (row["message_sent_utc"], row["source_url"])):
        key = row["atcf_id"], row["observation_time_utc"], row["mission_id"]
        unique[key] = row
    pairs = []
    for observation in unique.values():
        pair = pair_vdm(observation, by_atcf.get(observation["atcf_id"], []))
        if pair is not None:
            name = canonical_name(pair["name"])
            pair["cyclone_in_existing_inventory_tag"] = (pair["year"], name) in known
            pair["exact_named_time_in_existing_inventory_tag"] = (name, pair["observation_time_utc"]) in known_times
            pairs.append(pair)
    pairs.sort(key=lambda row: (row["observation_time_utc"], row["source_url"]))
    eligible = [row for row in pairs if row["dp_hpa"] > 30 and row["ahi_visible"]
                and row["observation_time_utc"] >= "2015-07-07"]
    selected = random.Random(args.seed).sample(eligible, min(args.count, len(eligible)))
    selected.sort(key=lambda row: (row["observation_time_utc"], row["source_url"]))
    save_csv(root / "all_aircraft_poci_pairs.csv", pairs)
    save_csv(root / "all_qualified_aircraft_dp_gt30.csv", eligible)
    save_csv(root / "observations_dp_gt30.csv", selected)
    (root / "download_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (root / "parse_failures.json").write_text(json.dumps(parse_failures, indent=2) + "\n")
    dp = [row["dp_hpa"] for row in selected]
    manifest = {
        "status": "collected_aircraft_central_pressure_reports_not_station_proxies",
        "completed_utc": datetime.now(UTC).isoformat(), "root_used_bytes_before": used,
        "proxy_environment_removed": True, "cyclone_role": "tag_only", "year_role": "tag_only",
        "training_proximity_exclusion": False, "within_cyclone_time_spacing": False,
        "source_product": "NHC Pacific Vortex Data Message REPPN2 / URPN12",
        "track_source": str(args.track_csv), "record_tags_source_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "requested": args.count, "seed": args.seed, "archive_messages": len(jobs),
        "parsed_messages": len(observations), "unique_aircraft_fixes": len(unique),
        "poci_paired_fixes": len(pairs), "qualified_visible_dp_gt30": len(eligible),
        "selected_records": len(selected), "selected_cyclones_tag_count": len({row["sid"] for row in selected}),
        "selected_names": dict(Counter(row["name"] for row in selected)),
        "selected_by_year": dict(Counter(row["year"] for row in selected)),
        "selected_pressure_methods": dict(Counter(row["pressure_method"] for row in selected)),
        "dp_min_hpa": min(dp) if dp else None, "dp_max_hpa": max(dp) if dp else None,
        "dp_bins": {"30_to_50": sum(30 < value <= 50 for value in dp),
                    "50_to_70": sum(50 < value <= 70 for value in dp),
                    "70_to_90": sum(70 < value <= 90 for value in dp),
                    "above_90": sum(value > 90 for value in dp)},
        "download_failures": len(failures), "parse_failures": len(parse_failures),
        "model_evaluations": 0, "satellite_sequences_downloaded": 0,
        "caveats": [
            "These are aircraft centre-pressure reports, not raw dropsonde sensor profiles.",
            "Reports may include pressure extrapolation; the original messages and method flags are retained.",
            "POCI is nearest IBTrACS USA_POCI within three hours, possibly interpolated upstream.",
            "CP aircraft cases are retained only when geometrically visible to Himawari, not Atlantic substitutions.",
            "Satellite sequences and eye visibility at these fixes have not yet been prepared or checked.",
            "No future operational-generalization or model-accuracy claim is made.",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
