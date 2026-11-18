#!/usr/bin/env python
"""Collect measured buoy and JMA/MRI aircraft pressures for WNP typhoons.

The census enumerates every numbered WNP storm in 2016-2026 and attaches only
measured pressure observations.  Storm identity/timing is metadata; no
best-track pressure, wind, or position enters the result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from h9dp.typhoon_census import (
    DIGITAL_TYPHOON_URL,
    JMA_BUOY_URL,
    ZENODO_FILE_URL,
    ZENODO_RECORD,
    Storm,
    associate_buoy_observations,
    deduplicate_buoy_records,
    parse_buoy_database,
    parse_digital_typhoon_year,
    parse_jma_buoy_txt,
    parse_zenodo_dropsonde,
    sha256_file,
)

BUOY_PAGE = (
    "https://www.data.jma.go.jp/gmd/kaiyou/db/vessel_obs/"
    "data-report/html/buoy/buoy.html"
)
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
RAW_AIRCRAFT_GROUPS = [
    {
        "season": 2017,
        "storm_name": "LAN",
        "files": [
            "DROP20171021T0530S7099015-LV2_pub.csv",
            "DROP20171021T0627S7099025-LV2_pub.csv",
            "DROP20171021T0639S7099023-LV2_pub.csv",
            "DROP20171021T0641S7099012-LV2_pub.csv",
        ],
    },
    {
        "season": 2022,
        "storm_name": "NANMADOL",
        "files": [
            "DROP20220917T0046S1099042-LV2.4_pub.csv",
            "DROP20220917T0046S1099043-LV2.4_pub.csv",
            "DROP20220917T0119S2099042-LV2.4_pub.csv",
            "DROP20220917T0120S2099043-LV2.4_pub.csv",
        ],
    },
]

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
USER_AGENT = "h9dp-typhoon-census/1.0 (normal-user research collection)"
LOG_LOCK = threading.Lock()


def unset_proxy_environment() -> None:
    for name in list(os.environ):
        if "proxy" in name.lower():
            os.environ.pop(name, None)


def fetch_bytes(url: str, path: Path, *, refresh: bool, log) -> bytes:  # noqa: ANN001
    if path.exists() and not refresh:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
            with OPENER.open(request, timeout=30) as response:
                data = response.read()
            if not data:
                raise ValueError("empty response")
            path.write_bytes(data)
            log(f"fetched {url} -> {path} ({len(data)} bytes)")
            return data
        except urllib.error.HTTPError as error:
            last_error = error
            log(f"HTTP {error.code} for {url}; not retrying")
            break
        except Exception as error:  # noqa: BLE001 - retry transport errors
            last_error = error
            log(f"fetch attempt {attempt}/3 failed for {url}: {error}")
            time.sleep(attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/data/H9dP/datasets/typhoon_census_2016_2026_20261115"),
    )
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--aircraft-config",
        type=Path,
        default=Path("configs/aircraft_pressure_records_2016_2026.json"),
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-aircraft", action="store_true")
    args = parser.parse_args()

    unset_proxy_environment()
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log_path = output / "collection.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}"
        with LOG_LOCK:
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    log(f"start output={output} years={args.start_year}-{args.end_year}")
    log("proxy environment variables unset; urllib opener has an empty ProxyHandler")

    failures: list[dict] = []
    source_files: list[dict] = []

    def fetch_many(tasks: list[tuple[str, Path]], workers: int = 6) -> None:
        """Fetch independent files concurrently; cache writes are per-path."""

        def run(url: str, path: Path) -> tuple[str, Path]:
            fetch_bytes(url, path, refresh=args.refresh, log=log)
            return url, path

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, url, path): (url, path) for url, path in tasks}
            for future in concurrent.futures.as_completed(futures):
                url, path = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001
                    failures.append({"stage": "fetch", "url": url, "error": str(error)})
                    log(f"fetch failed {url}: {error}")
                    continue
                source_files.append(
                    {
                        "url": url,
                        "path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    def fetch(url: str, path: Path) -> bytes:
        data = fetch_bytes(url, path, refresh=args.refresh, log=log)
        if not path.exists():  # defensive: cache write should have happened
            raise RuntimeError(f"cache missing for {url}")
        source_files.append(
            {
                "url": url,
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        return data

    # 1. Storm identity and active windows (no pressure/wind columns read).
    storms: list[Storm] = []
    for year in range(args.start_year, args.end_year + 1):
        url = DIGITAL_TYPHOON_URL.format(year=year)
        try:
            body = fetch(url, raw / "digital_typhoon" / f"{year}.html").decode(
                "utf-8", errors="replace"
            )
            year_storms = parse_digital_typhoon_year(body, year, url)
            log(f"digital-typhoon {year}: {len(year_storms)} storms")
            storms.extend(year_storms)
        except Exception as error:  # noqa: BLE001
            failures.append({"stage": "storm_index", "year": year, "error": str(error)})
            log(f"storm index failed for {year}: {error}")
    if not storms:
        log("no storms parsed; aborting")
        return 1
    # Strict identity de-duplication.
    unique: dict[str, Storm] = {}
    for storm in storms:
        unique[storm.storm_id] = storm
    storms = sorted(unique.values(), key=lambda item: item.birth_utc)

    # 2. JMA drifting-buoy hourly observations.
    buoy_database: dict[int, list[str]] = {}
    try:
        page = fetch(BUOY_PAGE, raw / "jma_buoy" / "buoy.html").decode(
            "utf-8", errors="replace"
        )
        buoy_database = parse_buoy_database(page)
        log(f"JMA buoy database years: {sorted(buoy_database)}")
    except Exception as error:  # noqa: BLE001
        failures.append({"stage": "buoy_database", "error": str(error)})
        log(f"JMA buoy database failed: {error}")

    buoy_records: list[dict] = []
    buoy_tasks: list[tuple[str, Path, int, str]] = []
    for year in range(args.start_year, args.end_year + 1):
        deployments = buoy_database.get(year, [])
        if not deployments:
            log(f"JMA buoy {year}: no deployments listed")
            continue
        for deployment in deployments:
            url = JMA_BUOY_URL.format(year=year, buoy=deployment)
            target = raw / "jma_buoy" / str(year) / f"{deployment}.TXT"
            buoy_tasks.append((url, target, year, deployment))
    # Six workers keep TLS handshakes from serialising; files are independent.
    fetch_many([(url, target) for url, target, _, _ in buoy_tasks], workers=6)
    for url, target, year, deployment in buoy_tasks:
        if not target.exists():
            failures.append(
                {
                    "stage": "jma_buoy",
                    "year": year,
                    "deployment": deployment,
                    "url": url,
                    "error": "cache file missing after fetch",
                }
            )
            log(f"JMA buoy {year}/{deployment} unavailable")
            continue
        try:
            body = target.read_text(encoding="utf-8", errors="replace")
            parsed = parse_jma_buoy_txt(
                body,
                deployment_id=f"{year}_{deployment}",
                source_url=url,
                calendar_year=year,
            )
            buoy_records.extend(parsed)
            log(f"JMA buoy {year}/{deployment}: {len(parsed)} pressure records")
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "stage": "jma_buoy_parse",
                    "year": year,
                    "deployment": deployment,
                    "url": url,
                    "error": str(error),
                }
            )
            log(f"JMA buoy {year}/{deployment} parse failed: {error}")

    buoy_records = deduplicate_buoy_records(buoy_records)
    log(f"JMA buoy total unique pressure records: {len(buoy_records)}")
    buoy_by_storm = associate_buoy_observations(storms, buoy_records)

    # 3. Aircraft measurements: native Zenodo profiles and curated published values.
    aircraft_by_storm: dict[str, list[dict]] = {storm.storm_id: [] for storm in storms}
    mission_by_storm: dict[str, list[dict]] = {storm.storm_id: [] for storm in storms}
    unresolved_aircraft: list[dict] = []
    by_name = {(storm.season, storm.name.upper()): storm for storm in storms}
    config: dict = {"records": [], "missions_without_public_numeric_pressure": []}

    if not args.skip_aircraft:
        try:
            api_bytes = fetch(ZENODO_API, raw / "aircraft" / "zenodo_record.json")
            api = json.loads(api_bytes)
            file_urls = {
                item["key"]: item["links"]["self"] for item in api.get("files", [])
            }
            for group in RAW_AIRCRAFT_GROUPS:
                storm = by_name.get((group["season"], group["storm_name"]))
                if storm is None:
                    unresolved_aircraft.append(
                        {"stage": "raw_profile", "reason": "storm not in index", **group}
                    )
                    continue
                for filename in group["files"]:
                    url = file_urls.get(filename, ZENODO_FILE_URL.format(name=filename))
                    target = raw / "aircraft" / f"zenodo_{ZENODO_RECORD}" / filename
                    body = fetch(url, target).decode("utf-8", errors="replace")
                    profile = parse_zenodo_dropsonde(
                        body, filename=filename, source_url=url
                    )
                    aircraft_by_storm[storm.storm_id].append(
                        {
                            "source_kind": "native eye dropsonde profile",
                            "platform": "T-PARCII Gulfstream II eye dropsonde (JMA/MRI collaboration)",
                            "observation_time_utc": profile["launch_time_utc"],
                            "raw_profile_acquired": True,
                            **profile,
                        }
                    )
                    log(
                        f"aircraft raw {filename}: near-surface {profile['near_surface_pressure_hpa']} hPa"
                    )
        except Exception as error:  # noqa: BLE001
            failures.append({"stage": "aircraft_zenodo", "error": str(error)})
            log(f"aircraft Zenodo collection failed: {error}")

        if args.aircraft_config.exists():
            try:
                config = json.loads(args.aircraft_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append({"stage": "aircraft_config", "error": str(error)})
                log(f"aircraft config failed: {error}")
            for record in config.get("records", []):
                storm = by_name.get((record["season"], record["storm_name"]))
                if storm is None:
                    unresolved_aircraft.append(
                        {
                            "stage": "published",
                            "reason": "storm not in index",
                            "record": record,
                        }
                    )
                    continue
                enriched = {
                    "source_kind": record["source_kind"],
                    "platform": "T-PARCII Gulfstream II (JMA/MRI collaboration)",
                    "pressure_is_best_track": False,
                    **record,
                }
                aircraft_by_storm[storm.storm_id].append(enriched)
        for mission in config.get("missions_without_public_numeric_pressure", []):
            storm = by_name.get((mission["season"], mission["storm_name"]))
            if storm is None:
                unresolved_aircraft.append(
                    {"stage": "mission", "reason": "storm not in index", **mission}
                )
                continue
            mission_by_storm[storm.storm_id].append(
                {"measured_pressure_hpa": None, "pressure_is_best_track": False, **mission}
            )

    for entries in aircraft_by_storm.values():
        entries.sort(key=lambda item: (item.get("observation_time_utc") or "", item["pressure_hpa"]))
    for entries in mission_by_storm.values():
        entries.sort(key=lambda item: item.get("storm_name", ""))

    # 4. Assemble and write.
    generated = datetime.now(UTC)
    storms_payload = []
    for storm in storms:
        buoy_entries = buoy_by_storm.get(storm.storm_id, [])
        aircraft_entries = aircraft_by_storm.get(storm.storm_id, [])
        missions = mission_by_storm.get(storm.storm_id, [])
        if buoy_entries:
            buoy_status = "collected"
        else:
            buoy_status = "no_platform_above_4hpa_depression"
        if aircraft_entries:
            aircraft_status = "collected"
        elif missions:
            aircraft_status = "mission_documented_no_public_numeric_pressure"
        else:
            aircraft_status = "no_jma_mri_aircraft_measurement_found"
        storms_payload.append(
            {
                "storm_id": storm.storm_id,
                "name": storm.name,
                "season": storm.season,
                "active_window_utc": {
                    "birth": storm.birth_utc.isoformat().replace("+00:00", "Z"),
                    "death": storm.death_utc.isoformat().replace("+00:00", "Z"),
                },
                "storm_identity_source": storm.source_url,
                "status": {
                    "buoy_measured_pressure": buoy_status,
                    "jma_mri_aircraft_measured_pressure": aircraft_status,
                },
                "buoy_pressure_minima": buoy_entries,
                "jma_aircraft_measurements": aircraft_entries,
                "jma_aircraft_missions_without_numeric_pressure": missions,
            }
        )

    buoy_pairs = [entry for storm in storms_payload for entry in storm["buoy_pressure_minima"]]
    series_count = sum(len(entry["series"]) for entry in buoy_pairs)
    confidence_counts = {
        label: sum(1 for entry in buoy_pairs if entry["association_confidence"] == label)
        for label in ("high", "medium", "low")
    }
    unique_buoy_observations = len(
        {
            (entry["wmo_id"], entry["minimum_time_utc"], entry["minimum_pressure_hpa"])
            for entry in buoy_pairs
        }
    )
    aircraft_records = [
        entry for storm in storms_payload for entry in storm["jma_aircraft_measurements"]
    ]
    native_profiles = [
        entry for entry in aircraft_records if entry["source_kind"] == "native eye dropsonde profile"
    ]
    with_buoy = [storm for storm in storms_payload if storm["buoy_pressure_minima"]]
    with_aircraft = [storm for storm in storms_payload if storm["jma_aircraft_measurements"]]
    summary = {
        "storm_count": len(storms_payload),
        "years": [args.start_year, args.end_year],
        "storms_with_buoy_measurement": len(with_buoy),
        "storms_with_jma_aircraft_measurement": len(with_aircraft),
        "buoy_platform_storm_pairs": len(buoy_pairs),
        "unique_buoy_minimum_observations": unique_buoy_observations,
        "buoy_association_confidence": confidence_counts,
        "buoy_minimum_records": len(buoy_pairs),
        "buoy_series_records": series_count,
        "buoy_platforms_represented": len({entry["platform_id"] for entry in buoy_pairs}),
        "aircraft_measurement_records": len(aircraft_records),
        "native_aircraft_dropsonde_profiles": len(native_profiles),
        "published_aircraft_pressure_records": len(aircraft_records) - len(native_profiles),
        "aircraft_mission_only_records": sum(
            len(storm["jma_aircraft_missions_without_numeric_pressure"])
            for storm in storms_payload
        ),
        "forum_derived_aircraft_records": sum(
            1 for entry in aircraft_records if "forum" in entry["source_kind"]
        ),
        "best_track_pressure_fields_in_result": 0,
    }
    payload = {
        "schema_version": 1,
        "generated_utc": generated.isoformat().replace("+00:00", "Z"),
        "scope": {
            "basin": "Western North Pacific (Digital Typhoon WNP numbered storms)",
            "start_year": args.start_year,
            "end_year": args.end_year,
            "storm_count": len(storms_payload),
        },
        "constraints": {
            "best_track_pressure_used": False,
            "best_track_wind_used": False,
            "best_track_position_used": False,
            "association_basis": (
                "storm active window plus own-buoy pressure depression; "
                "no track geometry and no spatial distance"
            ),
            "buoy_threshold": "pressure at least 4 hPa below the same buoy's pre-storm baseline",
            "aircraft_rule": (
                "only dropsonde-measured or dropsonde-derived pressures; "
                "mission-only records keep pressure_hpa=null"
            ),
            "storm_index_fields_read": ["number", "name", "birth_utc", "death_utc"],
        },
        "sources": {
            "storm_index": "Digital Typhoon WNP yearly lists (metadata only; pressure column ignored)",
            "buoy_primary": "JMA drifting ocean meteorological buoy hourly TXT, 2016-2026",
            "aircraft_raw": f"Zenodo record {ZENODO_RECORD} (T-PARCII eye dropsonde archives)",
            "aircraft_published": str(args.aircraft_config),
        },
        "summary": summary,
        "collection": {
            "outcome": "completed" if not failures else "completed_with_failures",
            "failures": failures,
            "unresolved_aircraft": unresolved_aircraft,
            "source_file_count": len(source_files),
        },
        "coverage_notes": {
            "forum_sweep": (
                "targeted, not exhaustive: only the HALONG 2025 post was admitted, "
                "because it reproduces official JMA TEMP_DROP values; no forum-wide "
                "crawl of all 283 storms was completed"
            ),
            "ndbc_buoys": (
                "attempted as a secondary network; several historical/stdmet requests "
                "were reset or timed out, so no unauditable NDBC record was included"
            ),
            "jma_aircraft_missing_numeric": [
                "HAISHEN 2020",
                "MINDULLE 2021",
                "BARIJAT 2024",
            ],
            "season_2026": "as of 2026-11-15; the season is still ongoing",
            "buoy_association": (
                "time-window plus own-buoy depression only; no spatial distance or track position"
            ),
        },
        "storms": storms_payload,
    }
    census_path = output / "census.json"
    census_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {census_path} ({census_path.stat().st_size} bytes)")

    with (output / "storms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "storm_id",
                "name",
                "season",
                "birth_utc",
                "death_utc",
                "buoy_platforms",
                "buoy_series_records",
                "lowest_buoy_pressure_hpa",
                "aircraft_measurements",
                "aircraft_status",
            ]
        )
        for storm in storms_payload:
            minima = storm["buoy_pressure_minima"]
            writer.writerow(
                [
                    storm["storm_id"],
                    storm["name"],
                    storm["season"],
                    storm["active_window_utc"]["birth"],
                    storm["active_window_utc"]["death"],
                    len(minima),
                    sum(len(item["series"]) for item in minima),
                    min((item["minimum_pressure_hpa"] for item in minima), default=""),
                    len(storm["jma_aircraft_measurements"]),
                    storm["status"]["jma_mri_aircraft_measured_pressure"],
                ]
            )

    with (output / "buoy_minima.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "storm_id",
                "name",
                "season",
                "platform_id",
                "wmo_id",
                "minimum_time_utc",
                "minimum_pressure_hpa",
                "baseline_pressure_hpa",
                "depression_hpa",
                "latitude_deg",
                "longitude_deg",
                "n_records_in_storm_window",
                "association_confidence",
                "concurrent_storm_ids",
                "source_url",
            ]
        )
        for storm in storms_payload:
            for item in storm["buoy_pressure_minima"]:
                writer.writerow(
                    [
                        storm["storm_id"],
                        storm["name"],
                        storm["season"],
                        item["platform_id"],
                        item["wmo_id"],
                        item["minimum_time_utc"],
                        item["minimum_pressure_hpa"],
                        item["baseline_pressure_hpa"],
                        item["depression_hpa"],
                        item["latitude_deg"],
                        item["longitude_deg"],
                        item["n_records_in_storm_window"],
                        item["association_confidence"],
                        " ".join(item["concurrent_storm_ids"]),
                        item["source_url"],
                    ]
                )

    with (output / "aircraft_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "storm_id",
                "name",
                "season",
                "source_kind",
                "observation_time_utc",
                "date_utc",
                "pressure_hpa",
                "pressure_definition",
                "latitude_deg",
                "longitude_deg",
                "raw_profile_acquired",
                "source_url",
                "note",
            ]
        )
        for storm in storms_payload:
            for item in storm["jma_aircraft_measurements"]:
                writer.writerow(
                    [
                        storm["storm_id"],
                        storm["name"],
                        storm["season"],
                        item.get("source_kind", ""),
                        item.get("observation_time_utc") or "",
                        item.get("date_utc", ""),
                        item.get("pressure_hpa", ""),
                        item.get("pressure_definition", ""),
                        item.get("latitude_deg", ""),
                        item.get("longitude_deg", ""),
                        item.get("raw_profile_acquired", ""),
                        item.get("source_url", ""),
                        item.get("note", ""),
                    ]
                )

    manifest = {
        "generated_utc": generated.isoformat().replace("+00:00", "Z"),
        "command": " ".join([sys.executable, "-m", "scripts.collect_typhoon_census", *sys.argv[1:]]),
        "python": sys.version,
        "output": str(output),
        "summary": summary,
        "outcome": "completed" if not failures else "completed_with_failures",
        "failures": failures,
        "source_files": source_files,
        "network_proxy_policy": (
            "all env names containing PROXY unset; urllib opener uses an empty ProxyHandler"
        ),
        "constraints": {
            "best_track_pressure_used": False,
            "best_track_pressure_fields_in_result": [],
        },
        "artifact_bytes": {
            "census.json": census_path.stat().st_size,
            "storms.csv": (output / "storms.csv").stat().st_size,
            "buoy_minima.csv": (output / "buoy_minima.csv").stat().st_size,
            "aircraft_measurements.csv": (output / "aircraft_measurements.csv").stat().st_size,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        "summary "
        + json.dumps(
            {
                key: summary[key]
                for key in (
                    "storm_count",
                    "storms_with_buoy_measurement",
                    "storms_with_jma_aircraft_measurement",
                    "buoy_platform_storm_pairs",
                    "buoy_series_records",
                    "aircraft_measurement_records",
                    "native_aircraft_dropsonde_profiles",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
