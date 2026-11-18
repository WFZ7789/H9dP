"""Parse aircraft vortex centre-pressure reports, not best-track pressure labels."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta

from h9dp.observations import TrackPoint, distance_km


def filename_time(filename: str) -> datetime:
    match = re.search(r"\.(\d{12})\.txt$", filename)
    if match is None:
        raise ValueError(f"Missing archive timestamp: {filename}")
    value = match[1]
    return datetime.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}T{value[8:10]}:{value[10:12]}:00")


def ahi_zenith(lat: float, lon: float, satellite_lon: float = 140.7) -> float | None:
    """Geodetic view geometry for AHI's orbital slot; None means behind Earth."""
    major, minor, orbit = 6378.137, 6356.7523, 42164.0
    phi, lam, sat = map(math.radians, (lat, lon, satellite_lon))
    radius = major / math.sqrt(1 - (1 - minor ** 2 / major ** 2) * math.sin(phi) ** 2)
    x = radius * math.cos(phi) * math.cos(lam)
    y = radius * math.cos(phi) * math.sin(lam)
    z = radius * minor ** 2 / major ** 2 * math.sin(phi)
    normal = (x / major ** 2, y / major ** 2, z / minor ** 2)
    ray = (orbit * math.cos(sat) - x, orbit * math.sin(sat) - y, -z)
    dot = sum(a * b for a, b in zip(normal, ray, strict=True))
    if dot <= 0:
        return None
    cosine = dot / math.sqrt(sum(a * a for a in normal) * sum(b * b for b in ray))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def parse_vdm(body: str, sent: datetime, source_url: str) -> dict | None:
    """Support old H-pressure/degree-minute and new D-pressure/decimal formats."""
    storm = re.search(r"VORTEX DATA MESSAGE\s+((?:AL|EP|CP|WP)\d{6})", body)
    clock = re.search(r"^A\.\s*(\d{1,2})/(\d{2}):(\d{2})(?::(\d{2}))?Z", body, re.MULTILINE)
    position = re.search(
        r"^B\.\s*(\d+(?:\.\d+)?)\s*deg(?:\s+(\d+(?:\.\d+)?)\s*min)?\s*([NS])"
        r"\s+(\d+(?:\.\d+)?)\s*deg(?:\s+(\d+(?:\.\d+)?)\s*min)?\s*([EW])",
        body, re.MULTILINE,
    )
    pressures = re.findall(r"^([DH])\.\s*(\d+(?:\.\d+)?)\s*(?:mb|hPa)\b", body, re.MULTILINE)
    if storm is None or clock is None or position is None or len(pressures) != 1:
        return None
    try:
        day, hour, minute, second = (int(value or 0) for value in clock.groups())
        first = sent.replace(day=1)
        months = (first - timedelta(days=1), first, first + timedelta(days=32))
        candidates = []
        for month in months:
            try:
                candidates.append(month.replace(day=day, hour=hour, minute=minute, second=second))
            except ValueError:
                continue
        if not candidates:
            return None
        time = min(candidates, key=lambda value: abs((value - sent).total_seconds()))
        lat = (float(position[1]) + float(position[2] or 0) / 60) * (-1 if position[3] == "S" else 1)
        lon = (float(position[4]) + float(position[5] or 0) / 60) * (-1 if position[6] == "W" else 1)
        pressure = float(pressures[0][1])
    except ValueError:
        return None
    if abs((time - sent).total_seconds()) > 172800 or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if not 800 <= pressure <= 1100:
        return None
    mission = re.search(r"^[PU]\.\s*((?:AF|NOAA|TEAL|NASA)[^\r\n]+)", body, re.MULTILINE)
    mission_text = mission[1].strip() if mission else "unspecified"
    if "EXTRAP" in body.upper():
        basis = "aircraft_report_pressure_extrapolation_mentioned"
    elif re.search(r"(?:CNTR|CENTER|CENTRE).*DROPSONDE", body):
        basis = "aircraft_report_center_dropsonde_mentioned"
    else:
        basis = "aircraft_report_pressure_basis_not_explicit"
    return {
        "atcf_id": storm[1], "observation_time_utc": time.isoformat() + "Z",
        "lat": lat, "lon": lon, "pc_observed_hpa": pressure,
        "pressure_message_field": pressures[0][0], "pressure_method": basis,
        "mission": mission_text, "mission_id": " ".join(mission_text.split()[:2]),
        "source_url": source_url, "message_sent_utc": sent.isoformat() + "Z",
        "observation_class": "aircraft_vortex_central_pressure_report",
        "is_surface_station_proxy": False,
    }


def pair_vdm(observation: dict, points: list[TrackPoint]) -> dict | None:
    time = datetime.fromisoformat(observation["observation_time_utc"].removesuffix("Z"))
    available = [point for point in points if point.poci is not None]
    if not available:
        return None
    nearest = min(available, key=lambda point: abs((point.time - time).total_seconds()))
    if abs((nearest.time - time).total_seconds()) > 10800:
        return None
    poci = nearest.poci
    if poci is None:
        return None
    zenith = ahi_zenith(observation["lat"], observation["lon"])
    return {
        **observation, "sid": nearest.sid, "name": nearest.name, "year": time.year,
        "basin_tag": "CP" if -180 <= observation["lon"] <= -140 else "EP",
        "poci_hpa": poci, "dp_hpa": round(poci - observation["pc_observed_hpa"], 3),
        "poci_source": "IBTrACS v04r01 USA_POCI",
        "poci_time_utc": nearest.time.isoformat() + "Z",
        "poci_offset_minutes": (nearest.time - time).total_seconds() / 60,
        "poci_position_distance_km": round(distance_km(
            observation["lat"], observation["lon"], nearest.lat, nearest.lon), 3),
        "ahi_visible": zenith is not None,
        "ahi_zenith_degrees": round(zenith, 3) if zenith is not None else None,
    }
