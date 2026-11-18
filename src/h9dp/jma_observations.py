"""JMA meteorological-station observations, never JMA best-track pressures."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser


class PressureTable(HTMLParser):
    """Read the labelled observation table, including original quality classes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active = False
        self.rows: list[list[tuple[str, str]]] = []
        self.row: list[tuple[str, str]] = []
        self.cell_parts: list[str] | None = None
        self.cell_class = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "tablefix1":
            self.active = True
        if not self.active:
            return
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"}:
            self.cell_parts = []
            self.cell_class = attributes.get("class") or ""
        if tag == "br" and self.cell_parts is not None:
            self.cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self.active and self.cell_parts is not None:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.active:
            return
        if tag in {"td", "th"} and self.cell_parts is not None:
            self.row.append(("".join(self.cell_parts).strip(), self.cell_class))
            self.cell_parts = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []
        if tag == "table":
            self.active = False


def normal_pressure(cell: tuple[str, str]) -> float | None:
    text, classes = cell
    if "data_0_0" not in classes.split() or re.fullmatch(r"\d{3,4}(?:\.\d+)?", text) is None:
        return None
    try:
        pressure = float(text)
    except ValueError:
        return None
    return pressure if 800 <= pressure <= 1100 else None


def parse_jma_10min(body: str, day_jst: date, station_id: str, source_url: str) -> list[dict]:
    """Keep measured station/SLP values and JST clocks; convert timestamps to UTC.

    Published sea-level pressure is station-observation based and may include
    sea-level reduction. It is not automatically cyclone central pressure.
    """
    table = PressureTable()
    table.feed(body)
    header = " ".join(text for row in table.rows[:2] for text, _ in row)
    if not all(word in header for word in ("気圧", "現地", "海面")):
        raise ValueError("Missing or changed JMA pressure-table headers")
    midnight = datetime.fromisoformat(day_jst.isoformat() + "T00:00:00+09:00")
    observations = []
    for row_index, row in enumerate(table.rows, start=1):
        if len(row) < 10 or re.fullmatch(r"\d{2}:\d{2}", row[0][0]) is None:
            continue
        try:
            hour, minute = map(int, row[0][0].split(":"))
        except ValueError:
            continue
        if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute != 0):
            continue
        slp = normal_pressure(row[2])
        if slp is None:
            continue
        local = midnight + timedelta(hours=hour, minutes=minute)
        utc = local.astimezone(UTC)
        observations.append({
            "station_id": station_id, "source_day_jst": day_jst.isoformat(),
            "source_clock_jst": row[0][0], "observation_time_jst": local.isoformat(),
            "observation_time_utc": utc.isoformat().replace("+00:00", "Z"),
            "station_pressure_observed_hpa": normal_pressure(row[1]),
            "slp_observed_hpa": slp, "slp_quality_class": row[2][1],
            "wind_mean_ms_raw": row[6][0], "wind_direction_raw": row[7][0],
            "wind_gust_ms_raw": row[8][0], "wind_gust_direction_raw": row[9][0],
            "pressure_source": "JMA 10-minute meteorological-station observations",
            "pressure_is_best_track": False, "is_verified_cyclone_central_pressure": False,
            "pressure_representativeness": "measured_at_station; centre/eye context not yet verified",
            "source_url": source_url, "source_table_row": row_index,
        })
    return observations
