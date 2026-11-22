"""Audit the provenance and quality of station observations paired with environmental pressure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from h9dp.pressure_labels import read_catalog


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON source {path}: {error}") from error


def audit(collection: Path, records: Path) -> dict:
    manifest = read_json(collection / "manifest.json")
    if manifest.get("protocol_version") != 2:
        raise ValueError("Rejected legacy cyclone-grouped/weak-dP collection; use record-level protocol v2")
    if manifest.get("cyclone_role") != "tag_only" or manifest.get("year_role") != "tag_only":
        raise ValueError("Cyclone and year must be tags only")
    with (collection / "review_candidates.csv").open() as handle:
        selected = list(csv.DictReader(handle))
    _, catalog_digest = read_catalog(records)
    if catalog_digest != manifest["existing_records_sha256"]:
        raise ValueError("Existing cyclone inventory changed since collection")
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
        "certified_central_pressure_records": 0,
    }
    (collection / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("MOTIS_cyclone_records.json"))
    args = parser.parse_args()
    if not args.collection.resolve().is_relative_to(Path("/mnt/data/H9dP")):
        parser.error("collection must be below /mnt/data/H9dP")
    print(json.dumps(audit(args.collection, args.records), indent=2))


if __name__ == "__main__":
    main()
