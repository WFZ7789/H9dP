"""Offline source-integrity audit and separate MOTIS adopted-pressure inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from h9dp.observations import canonical_name, distance_km, read_tracks
from scripts.collect_surface_poci import save_csv


def inventory_motis_adopted(source: Path, tracks: dict, exposed: set) -> list[dict]:
    """Retain published labels as adopted/interpolated, not raw pressure readings."""
    points_by_name = {}
    for points in tracks.values():
        for point in points:
            if point.poci is not None:
                points_by_name.setdefault((point.season, canonical_name(point.name)), []).append(point)
    result = []
    with source.open() as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if row["Region"] != "WP" or not row["dP Obs"].strip():
                continue
            try:
                year = int(row["Year"])
                date, clock = row["Date"].zfill(4), row["Time"].zfill(4)
                time = datetime.strptime(f"{year}{date}{clock}", "%Y%m%d%H%M")  # noqa: DTZ007 — source is explicitly UTC
                poci, dp = float(row["POCI"]), float(row["dP Obs"])
                lat, lon = float(row["Lat"]), (float(row["Lon"]) + 180) % 360 - 180
            except ValueError:
                continue
            if not 2015 <= year <= 2025:
                continue
            key = (year, canonical_name(row["Name"]))
            points = points_by_name.get(key, [])
            nearest = min(points, key=lambda point: abs((point.time - time).total_seconds())) if points else None
            if nearest is not None and (
                abs((nearest.time - time).total_seconds()) > 10800
                or distance_km(lat, lon, nearest.lat, nearest.lon) > 100
            ):
                nearest = None
            result.append({
                "name": row["Name"], "year": year, "time_utc": time.isoformat() + "Z",
                "lat": lat, "lon": lon, "quality": row["Eye clarity"],
                "poci_motis_era5_hpa": poci, "dp_motis_adopted_hpa": dp,
                "pc_motis_adopted_hpa": round(poci - dp, 3),
                "poci_ibtracs_hpa": nearest.poci if nearest else "",
                "poci_ibtracs_time_utc": nearest.time.isoformat() + "Z" if nearest else "",
                "ibtracs_sid": nearest.sid if nearest else "",
                "already_in_existing_cyclone_inventory": key in exposed,
                "observation_class": "published_adopted_pressure_may_include_interpolation_or_radar",
                "is_raw_observation": False, "source_csv": str(source), "source_line": line_number,
            })
    return result


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON source {path}: {error}") from error


def audit(collection: Path, motis: Path, records: Path) -> dict:
    manifest = read_json(collection / "manifest.json")
    if manifest.get("protocol_version") != 2:
        raise ValueError("Rejected legacy cyclone-grouped/weak-dP collection; use record-level protocol v2")
    if manifest.get("cyclone_role") != "tag_only" or manifest.get("year_role") != "tag_only":
        raise ValueError("Cyclone and year must be tags only")
    with (collection / "review_candidates.csv").open() as handle:
        selected = list(csv.DictReader(handle))
    records_bytes = records.read_bytes()
    if hashlib.sha256(records_bytes).hexdigest() != manifest["existing_records_sha256"]:
        raise ValueError("Existing cyclone inventory changed since collection")
    exposed = {(int(row["time"]["year"]), canonical_name(row["name"]))
               for row in read_json(records)["records"]}
    observation_keys = set()
    minimum_proxy_dp = max(30, manifest["minimum_proxy_dp_hpa"])
    for row in selected:
        key = row["station_id"], row["observation_time_utc"]
        if key in observation_keys:
            raise ValueError("Duplicate physical station observation")
        observation_keys.add(key)
        if float(row["dp_station_proxy_hpa"]) <= minimum_proxy_dp:
            raise ValueError("Surface-review proxy dP must exceed the threshold")
        if row["is_central_pressure_truth"] != "False":
            raise ValueError("Surface screening must not certify central pressure")
        if row["slp_qc"] not in {"1", "5"} or float(row["centre_distance_km"]) > 25:
            raise ValueError("Candidate fails SLP QC or distance rule")
        if abs(float(row["poci_offset_minutes"])) > 180:
            raise ValueError("POCI time offset exceeds three hours")
    verified = 0
    for path in (collection / "raw").rglob("*.csv"):
        metadata = read_json(path.with_suffix(".csv.json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Raw source integrity mismatch: {path}")
        verified += 1
    with (collection / "raw" / "ibtracs.WP.list.v04r01.csv").open() as handle:
        tracks = read_tracks(handle, 2015, 2025)
    adopted = inventory_motis_adopted(motis, tracks, exposed)
    save_csv(collection / "motis_adopted_wp_2015_2025.csv", adopted)
    summary = {
        "audit_utc": datetime.now(UTC).isoformat(), "audit_passed": True,
        "verified_raw_csv_hashes": verified, "selected_pairs": len(selected),
        "selected_cyclones": len({row["sid"] for row in selected}),
        "cyclone_role": "tag_only", "year_role": "tag_only",
        "minimum_proxy_dp_hpa": minimum_proxy_dp,
        "source_slp_qc": dict(Counter(row["slp_qc"] for row in selected)),
        "source_report_type": dict(Counter(row["report_type"] for row in selected)),
        "numeric_summary": {key: {"min": min(values), "median": median(values), "max": max(values)}
                            for key in ("slp_observed_hpa", "poci_hpa", "centre_distance_km", "poci_offset_minutes")
                            if (values := [float(row[key]) for row in selected])},
        "motis_adopted_wp_2015_2025": len(adopted),
        "motis_existing_cyclone_tags": sum(row["already_in_existing_cyclone_inventory"] for row in adopted),
        "motis_with_ibtracs_poci": sum(row["poci_ibtracs_hpa"] != "" for row in adopted),
        "motis_source_sha256": hashlib.sha256(motis.read_bytes()).hexdigest(),
        "certified_central_pressure_records": 0,
        "note": "Surface candidates and published adopted-pressure inventory are separate, not pooled truth.",
    }
    (collection / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--motis-csv", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    args = parser.parse_args()
    if not args.collection.resolve().is_relative_to(Path("/mnt/data/H9dP")):
        parser.error("collection must be below /mnt/data/H9dP")
    print(json.dumps(audit(args.collection, args.motis_csv, args.records), indent=2))


if __name__ == "__main__":
    main()
