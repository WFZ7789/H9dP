"""Validate cached MOTIS targets against the catalog in this checkout."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / "MOTIS_cyclone_records.json"


def read_catalog(path: str | Path | None = None) -> tuple[list[dict], str]:
    catalog = CATALOG_PATH.resolve()
    if path is not None and Path(path).resolve() != catalog:
        raise ValueError(f"MOTIS pressure labels must come from {catalog}")
    raw = catalog.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "motis-cyclone-scan-records/v1":
        raise ValueError(f"Invalid MOTIS record catalog: {catalog}")
    return data["records"], hashlib.sha256(raw).hexdigest()


def verify_catalog_labels(samples: list[dict], catalog: list[dict]) -> None:
    """Confirm the identity and dP of each cached record before supervised use."""
    seen = set()
    for sample in samples:
        record = sample["record"]
        if type(record) is not int or not 0 <= record < len(catalog) or record in seen:
            raise ValueError(f"Invalid or repeated MOTIS record ID {record}")
        seen.add(record)
        ref = catalog[record]
        target = ref["dP"]
        if (type(target) not in (int, float) or not math.isfinite(target)
                or sample["dP"] != target or sample["quality"] != ref["quality"]
                or sample["name"] != ref["name"] or sample["year"] != ref["time"]["year"]):
            raise ValueError(f"Cached record {record} differs from the MOTIS catalog")
        if "target_time" in sample and (
            datetime.fromisoformat(sample["target_time"]) != datetime.fromisoformat(ref["time"]["iso_utc"])
        ):
            raise ValueError(f"Cached record {record} has a different MOTIS observation time")
