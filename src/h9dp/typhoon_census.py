"""Measured-pressure census helpers for WNP typhoons, 2016-2026.

This module only parses measured observations.  Storm identity and active
windows are metadata; no best-track pressure, wind, or position is used in
the resulting observation records.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

JMA_BUOY_SOURCE = "JMA drifting ocean meteorological buoy (漂流型海洋気象ブイロボット)"
JMA_BUOY_URL = (
    "https://www.data.jma.go.jp/gmd/kaiyou/data/db/vessel_obs/"
    "data-report/data/{year}/buoy/txt/{buoy}.TXT"
)
DIGITAL_TYPHOON_URL = "https://agora.ex.nii.ac.jp/digital-typhoon/year/wnp/{year}.html.en"
ZENODO_RECORD = 10798896
ZENODO_FILE_URL = "https://zenodo.org/api/records/10798896/files/{name}/content"


@dataclass(frozen=True)
class Storm:
    storm_id: str
    name: str
    season: int
    birth_utc: datetime
    death_utc: datetime
    source_url: str


class _TableRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []


def parse_digital_typhoon_year(body: str, season: int, source_url: str) -> list[Storm]:
    """Parse storm number/name/birth/death only.

    The page also publishes a best-track minimum pressure column.  It is
    deliberately ignored here and never reaches the census.
    """
    parser = _TableRows()
    parser.feed(body)
    storms: list[Storm] = []
    for row in parser.rows:
        if len(row) < 7 or not re.fullmatch(r"\d{6}", row[1]):
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", row[4]):
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", row[5]):
            continue
        if str(season) not in row[1]:
            continue
        storms.append(
            Storm(
                storm_id=row[1],
                name=row[2],
                season=season,
                birth_utc=datetime.fromisoformat(row[4]).replace(tzinfo=UTC),
                death_utc=datetime.fromisoformat(row[5]).replace(tzinfo=UTC),
                source_url=source_url,
            )
        )
    return storms


def parse_buoy_database(body: str) -> dict[int, list[str]]:
    """Read the year -> deployment-id JavaScript table on the JMA buoy page."""
    match = re.search(r"const buoyDatabase\s*=\s*\{(.*?)\n\s*\};", body, re.S)
    if match is None:
        raise ValueError("JMA buoyDatabase table not found")
    database: dict[int, list[str]] = {}
    for year_text, values in re.findall(r'"(\d{4})"\s*:\s*\[([^\]]*)\]', match.group(1)):
        database[int(year_text)] = re.findall(r'"([0-9]{2}-[0-9]{2})"', values)
    if not database:
        raise ValueError("JMA buoyDatabase table is empty")
    return database


def _optional_tenths(text: str, scale: int = 10) -> float | None:
    text = text.strip()
    if not text or not text.isdigit():
        return None
    value = int(text)
    if value in {99999, 9999, 999}:
        return None
    return value / scale


def parse_jma_buoy_txt(
    body: str,
    *,
    deployment_id: str,
    source_url: str,
    calendar_year: int,
) -> list[dict]:
    """Parse JMA drifting-buoy hourly records.

    Fixed columns follow the published 52-character format:
    WMO id, UTC year/month/day/hour, N, latitude (1/1000 deg), E, longitude
    (1/1000 deg), pressure (1/10 hPa), SST (1/10 C), wave height (1/10 m),
    wave period (s).  Blank pressure is missing, never zero-filled.
    """
    records: list[dict] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if len(line) < 42:
            continue
        wmo = line[0:5].strip()
        if not re.fullmatch(r"\d{5}", wmo):
            continue
        try:
            year = int(line[6:10])
            month = int(line[11:13])
            day = int(line[14:16])
            hour = int(line[17:19])
            stamp = datetime(year, month, day, hour, tzinfo=UTC)
        except ValueError:
            continue
        latitude = _optional_tenths(line[22:27], 1000)
        longitude = _optional_tenths(line[29:35], 1000)
        if line[20] == "S" and latitude is not None:
            latitude = -latitude
        if line[28] == "W" and longitude is not None:
            longitude = -longitude
        pressure = _optional_tenths(line[36:42], 10)
        if pressure is None or not 800 <= pressure <= 1100:
            continue
        records.append(
            {
                "platform_id": deployment_id,
                "wmo_id": wmo,
                "observation_time_utc": stamp.isoformat().replace("+00:00", "Z"),
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "pressure_hpa": pressure,
                "pressure_type": "measured sea-level pressure at drifting buoy",
                "sst_c": _optional_tenths(line[42:46], 10),
                "wave_height_m": _optional_tenths(line[46:50], 10),
                "wave_period_s": _optional_tenths(line[50:53], 1),
                "source": JMA_BUOY_SOURCE,
                "source_url": source_url,
                "source_line": line_number,
                "calendar_year_file": calendar_year,
                "pressure_is_best_track": False,
            }
        )
    return records


def parse_zenodo_dropsonde(body: str, *, filename: str, source_url: str) -> dict:
    """Parse one T-PARCII eye dropsonde profile (Level 2.x CSV).

    Pressure is the aircraft/sonde-measured value in hPa.  The lowest value
    in an eye profile is reported; it is near the surface but is not silently
    relabelled as a verified cyclone central pressure.
    """
    match = re.search(r"DROP(\d{8})T(\d{4})", filename)
    if match is None:
        raise ValueError(f"No launch time in {filename}")
    launch = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M").replace(
        tzinfo=UTC
    )
    rows = list(csv.DictReader(io.StringIO(body)))
    levels: list[tuple[float, float | None, float | None]] = []
    first_lat = first_lon = None
    launch_clock = None
    for row in rows:
        clean = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}
        pressure_text = clean.get("Press", "")
        try:
            pressure = float(pressure_text)
        except ValueError:
            continue
        if pressure <= 0 or pressure == -999:
            continue
        try:
            lat = float(clean.get("lat", "-999"))
            lon = float(clean.get("lon", "-999"))
        except ValueError:
            lat = lon = -999
        if first_lat is None and lat != -999 and lon != -999:
            first_lat, first_lon = lat, lon
            launch_clock = clean.get("h:m:s")
        levels.append((pressure, None if lat == -999 else lat, None if lon == -999 else lon))
    if not levels:
        raise ValueError(f"No pressure levels in {filename}")
    pressures = [level[0] for level in levels]
    if launch_clock:
        clock = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", launch_clock)
        if clock:
            launch = launch.replace(
                hour=int(clock.group(1)),
                minute=int(clock.group(2)),
                second=int(clock.group(3)),
            )
    return {
        "profile_id": filename.removesuffix("-LV2_pub.csv").removesuffix("-LV2.4_pub.csv"),
        "launch_time_utc": launch.isoformat().replace("+00:00", "Z"),
        "launch_clock_utc": launch_clock,
        "time_precision": "second, from the first valid sonde row; archive filename gives the minute",
        "latitude_deg": first_lat,
        "longitude_deg": first_lon,
        "n_pressure_levels": len(levels),
        "top_pressure_hpa": min(pressures),
        "near_surface_pressure_hpa": levels[-1][0],
        "pressure_hpa": levels[-1][0],
        "pressure_definition": (
            "near-surface measured pressure (last valid level) in an eye dropsonde profile; "
            "not relabelled as a verified cyclone central pressure"
        ),
        "pressure_is_aircraft_measured": True,
        "pressure_is_best_track": False,
        "central_pressure_verified": False,
        "source": "T-PARCII eye dropsonde (JMA/MRI collaboration), Zenodo Level 2",
        "source_url": source_url,
        "source_file": filename,
    }


def associate_buoy_observations(
    storms: list[Storm],
    records: list[dict],
    *,
    minimum_depression_hpa: float = 4.0,
    baseline_start_hours: int = 72,
    baseline_end_hours: int = 24,
    edge_hours: int = 6,
) -> dict[str, list[dict]]:
    """Attach buoy pressure minima to storm active windows by time only.

    No storm position or best-track field is used.  A platform is attached
    when its pressure falls at least ``minimum_depression_hpa`` below its own
    pre-storm baseline.  Association is therefore deliberately conservative
    and explicitly marked as time-window based, not spatial.
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        record["_stamp"] = datetime.fromisoformat(
            record["observation_time_utc"].replace("Z", "+00:00")
        )
        grouped.setdefault(record["platform_id"], []).append(record)
    for series in grouped.values():
        series.sort(key=lambda item: item["_stamp"])

    windows = {
        storm.storm_id: (
            storm.birth_utc - timedelta(hours=edge_hours),
            storm.death_utc + timedelta(hours=edge_hours),
        )
        for storm in storms
    }
    result: dict[str, list[dict]] = {storm.storm_id: [] for storm in storms}
    for storm in storms:
        window_start, window_end = windows[storm.storm_id]
        for platform_id, series in grouped.items():
            times = [item["_stamp"] for item in series]
            left = bisect_left(times, window_start)
            right = bisect_right(times, window_end)
            in_window = series[left:right]
            if not in_window:
                continue
            baseline_left = bisect_left(
                times, storm.birth_utc - timedelta(hours=baseline_start_hours)
            )
            baseline_right = bisect_left(
                times, storm.birth_utc - timedelta(hours=baseline_end_hours)
            )
            baseline_rows = series[baseline_left:baseline_right]
            if len(baseline_rows) < 4:
                baseline_left = bisect_left(
                    times, storm.birth_utc - timedelta(hours=48)
                )
                baseline_right = bisect_left(
                    times, storm.birth_utc - timedelta(hours=6)
                )
                baseline_rows = series[baseline_left:baseline_right]
            if len(baseline_rows) < 4:
                continue
            baseline = statistics.fmean(float(item["pressure_hpa"]) for item in baseline_rows)
            minimum = min(in_window, key=lambda item: float(item["pressure_hpa"]))
            depression = baseline - float(minimum["pressure_hpa"])
            if depression < minimum_depression_hpa:
                continue
            min_stamp = minimum["_stamp"]
            concurrent = [
                other.storm_id
                for other in storms
                if other.storm_id != storm.storm_id
                and windows[other.storm_id][0] <= min_stamp <= windows[other.storm_id][1]
            ]
            base_confidence = (
                "high" if depression >= 12 else "medium" if depression >= 7 else "low"
            )
            confidence = (
                "medium" if base_confidence == "high" else "low"
            ) if concurrent else base_confidence
            series_rows = in_window
            result[storm.storm_id].append(
                {
                    "platform_id": platform_id,
                    "wmo_id": minimum["wmo_id"],
                    "network": "JMA drifting ocean meteorological buoy",
                    "minimum_pressure_hpa": float(minimum["pressure_hpa"]),
                    "minimum_time_utc": minimum["observation_time_utc"],
                    "latitude_deg": minimum["latitude_deg"],
                    "longitude_deg": minimum["longitude_deg"],
                    "baseline_pressure_hpa": round(baseline, 2),
                    "depression_hpa": round(depression, 2),
                    "baseline_window": "pre-storm 72-24 h, fallback 48-6 h",
                    "n_records_in_storm_window": len(in_window),
                    "n_baseline_records": len(baseline_rows),
                    "association_basis": "active storm window plus own-buoy pressure depression; no track geometry",
                    "association_confidence": confidence,
                    "attribution_ambiguity": bool(concurrent),
                    "concurrent_storm_ids": concurrent,
                    "source": JMA_BUOY_SOURCE,
                    "source_url": minimum["source_url"],
                    "pressure_is_best_track": False,
                    "series_scope": "all usable hourly records in the active window (edge +-6 h)",
                    "series": [
                        {
                            "observation_time_utc": row["observation_time_utc"],
                            "pressure_hpa": float(row["pressure_hpa"]),
                            "latitude_deg": row["latitude_deg"],
                            "longitude_deg": row["longitude_deg"],
                        }
                        for row in series_rows
                    ],
                }
            )
    for entries in result.values():
        entries.sort(key=lambda item: item["minimum_time_utc"])
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_buoy_records(records: list[dict]) -> list[dict]:
    """Drop repeated copies of one physical observation across year files.

    The same buoy can appear in two adjacent calendar-year files with two
    deployment labels (for example 2019_19-15 and 2020_19-15).  Identity is
    therefore the WMO number plus time and measured values, not the file label.
    """
    merged: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["wmo_id"],
            record["observation_time_utc"],
            record["pressure_hpa"],
            record["latitude_deg"],
            record["longitude_deg"],
        )
        if key in merged:
            target = merged[key]
            deployments = target.setdefault("deployment_ids", [target["platform_id"]])
            if record["platform_id"] not in deployments:
                deployments.append(record["platform_id"])
            years = target.setdefault(
                "calendar_year_files",
                [target.get("calendar_year_file")],
            )
            if record.get("calendar_year_file") not in years:
                years.append(record.get("calendar_year_file"))
            continue
        record["deployment_ids"] = [record["platform_id"]]
        record["calendar_year_files"] = [record.get("calendar_year_file")]
        merged[key] = record
    unique = list(merged.values())
    unique.sort(key=lambda item: (item["platform_id"], item["observation_time_utc"]))
    return unique
