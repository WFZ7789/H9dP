"""Conservative surface-pressure/POCI pairing; not central-pressure ground truth."""

from __future__ import annotations

import bisect
import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrackPoint:
    sid: str
    name: str
    season: int
    time: datetime
    lat: float
    lon: float
    poci: float | None
    atcf_id: str


@dataclass(frozen=True)
class Station:
    identifier: str
    name: str
    lat: float
    lon: float
    elevation: float
    begin: int
    end: int


def canonical_name(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, including the dateline."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2 - phi1) / 2) ** 2
    a += math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, a)))


def optional_number(value: str) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def read_tracks(
    lines: Iterable[str], first_year: int, last_year: int,
    basins: tuple[str, ...] | None = ("WP",),
) -> dict[str, list[TrackPoint]]:
    """Read MAIN tracks, WP by default. USA_POCI is never an observation."""
    tracks = defaultdict(list)
    for row in csv.DictReader(lines):
        if row["TRACK_TYPE"].strip() != "main":
            continue
        if basins is not None and row["BASIN"].strip() not in basins:
            continue
        try:
            season = int(row["SEASON"])
            observation_time = datetime.fromisoformat(row["ISO_TIME"].strip())
        except ValueError:
            continue
        if not first_year <= season <= last_year or row["NATURE"].strip() != "TS":
            continue
        lat, lon = optional_number(row["LAT"]), optional_number(row["LON"])
        if lat is None or lon is None:
            continue
        poci = optional_number(row["USA_POCI"])
        if poci is not None and not 900 <= poci <= 1100:
            poci = None
        point = TrackPoint(
            row["SID"].strip(), row["NAME"].strip(), season,
            observation_time, lat, lon, poci,
            row["USA_ATCF_ID"].strip(),
        )
        tracks[point.sid].append(point)
    return {sid: sorted(points, key=lambda point: point.time) for sid, points in tracks.items()}


def read_stations(lines: Iterable[str]) -> list[Station]:
    """Low-elevation stations in the Himawari/WP region; no pressure-based filter."""
    stations = []
    for row in csv.DictReader(lines):
        lat, lon = optional_number(row["LAT"]), optional_number(row["LON"])
        elevation = optional_number(row["ELEV(M)"])
        if lat is None or lon is None or elevation is None:
            continue
        if not (5 <= lat <= 45 and 110 <= lon <= 160 and -10 <= elevation <= 100):
            continue
        try:
            begin, end = int(row["BEGIN"][:4]), int(row["END"][:4])
        except ValueError:
            continue
        stations.append(Station(
            row["USAF"] + row["WBAN"], row["STATION NAME"], lat, lon, elevation,
            begin, end,
        ))
    return stations


def track_at(points: list[TrackPoint], time: datetime) -> tuple[float, float, TrackPoint] | None:
    """Interpolate position over <=6 h, but retain the nearest unmodified POCI."""
    times = [point.time for point in points]
    index = bisect.bisect_left(times, time)
    if index < len(points) and points[index].time == time:
        point = points[index]
        return (point.lat, point.lon, point) if point.poci is not None else None
    if index == 0 or index == len(points):
        return None
    before, after = points[index - 1:index + 1]
    gap = (after.time - before.time).total_seconds()
    if not 0 < gap <= 21600:
        return None
    nearest = min((before, after), key=lambda point: abs((point.time - time).total_seconds()))
    if nearest.poci is None or abs((nearest.time - time).total_seconds()) > 10800:
        return None
    weight = (time - before.time).total_seconds() / gap
    lon_delta = (after.lon - before.lon + 180) % 360 - 180
    lon = (before.lon + weight * lon_delta + 180) % 360 - 180
    return before.lat + weight * (after.lat - before.lat), lon, nearest


def parse_slp(value: str) -> tuple[float, str] | None:
    """ISD sea-level pressure is in tenths of hPa; accept only passed QC 1/5."""
    parts = value.split(",")
    if len(parts) != 2 or parts[1] not in {"1", "5"}:
        return None
    raw = optional_number(parts[0])
    if raw is None or raw == 99999 or not 8000 <= raw <= 11000:
        return None
    return raw / 10, parts[1]


def pair_surface_rows(
    lines: Iterable[str], station: Station, points: list[TrackPoint],
    source_url: str, max_distance_km: float = 25,
) -> list[dict]:
    """Keep measured SLP separate from inferred centre position and analyzed POCI."""
    pairs = []
    for line_number, row in enumerate(csv.DictReader(lines), start=2):
        if row["STATION"] != station.identifier:
            continue
        slp = parse_slp(row.get("SLP", ""))
        lat, lon = optional_number(row["LATITUDE"]), optional_number(row["LONGITUDE"])
        elevation = optional_number(row["ELEVATION"])
        if slp is None or lat is None or lon is None or elevation is None:
            continue
        if not -10 <= elevation <= 100 or distance_km(lat, lon, station.lat, station.lon) > 5:
            continue
        time = datetime.fromisoformat(row["DATE"])
        match = track_at(points, time)
        if match is None:
            continue
        centre_lat, centre_lon, point = match
        poci = point.poci
        if poci is None:
            continue
        distance = distance_km(lat, lon, centre_lat, centre_lon)
        if distance > max_distance_km:
            continue
        pressure, qc = slp
        pairs.append({
            "sid": point.sid, "name": point.name, "season": point.season,
            "atcf_id": point.atcf_id, "observation_time_utc": time.isoformat() + "Z",
            "station_id": station.identifier, "station_name": station.name,
            "station_lat": lat, "station_lon": lon, "station_elevation_m": elevation,
            "slp_observed_hpa": pressure, "slp_qc": qc,
            "report_type": row.get("REPORT_TYPE", ""),
            "centre_lat_interpolated": centre_lat, "centre_lon_interpolated": centre_lon,
            "centre_distance_km": round(distance, 3),
            "poci_hpa": poci, "poci_time_utc": point.time.isoformat() + "Z",
            "poci_offset_minutes": (point.time - time).total_seconds() / 60,
            "poci_source": "IBTrACS v04r01 USA_POCI (JTWC analysis; not measured)",
            "dp_station_proxy_hpa": round(poci - pressure, 3),
            "is_central_pressure_truth": False,
            "validation_status": "candidate_only_needs_eye_and_station_review",
            "observation_source_url": source_url, "observation_source_line": line_number,
        })
    return pairs


def select_pairs(
    rows: list[dict], count: int = 100, seed: int = 20261114,
    min_proxy_dp_hpa: float = 30,
) -> list[dict]:
    """Surface-review screening only: strong proxy dP, observation-level sampling.

    Cyclone and year are tags only. This never certifies central pressure.
    """
    import random

    if not math.isfinite(min_proxy_dp_hpa) or min_proxy_dp_hpa < 30:
        raise ValueError("Surface-review proxy dP threshold must be finite and at least 30 hPa")
    unique = {}
    for row in rows:
        proxy_dp = optional_number(str(row.get("dp_station_proxy_hpa", "")))
        if proxy_dp is None or proxy_dp <= min_proxy_dp_hpa:
            continue
        key = row["station_id"], row["observation_time_utc"]
        rank = (row["centre_distance_km"], row.get("observation_source_url", ""),
                row["observation_source_line"])
        if key not in unique or rank < unique[key][0]:
            unique[key] = rank, row
    ordered = sorted((row for _, row in unique.values()),
                     key=lambda row: (row["observation_time_utc"], row["station_id"]))
    if len(ordered) > count:
        ordered = random.Random(seed).sample(ordered, count)
    return sorted(ordered, key=lambda row: (row["observation_time_utc"], row["station_id"]))
