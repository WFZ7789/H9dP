from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from h9dp.typhoon_census import (
    Storm,
    associate_buoy_observations,
    deduplicate_buoy_records,
    parse_buoy_database,
    parse_digital_typhoon_year,
    parse_jma_buoy_txt,
    parse_zenodo_dropsonde,
)


def test_parse_buoy_database() -> None:
    body = 'const buoyDatabase = {\n    "2022": ["21-12", "22-01"],\n    "2026": ["25-13"]\n};\n'
    assert parse_buoy_database(body) == {2022: ["21-12", "22-01"], 2026: ["25-13"]}


def test_parse_jma_buoy_fixed_columns_and_missing_pressure() -> None:
    body = (
        "21595 2022  6 29 15 N 26994 E135002 10138 286  13  7\n"
        "21597 2016  5 21 21 N 20854 E131227            10  9\n"
        "garbage\n"
    )
    rows = parse_jma_buoy_txt(
        body, deployment_id="2022_22-09", source_url="https://example.test", calendar_year=2022
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["wmo_id"] == "21595"
    assert row["observation_time_utc"] == "2022-06-29T15:00:00Z"
    assert row["latitude_deg"] == 26.994
    assert row["longitude_deg"] == 135.002
    assert row["pressure_hpa"] == 1013.8
    assert row["sst_c"] == 28.6
    assert row["pressure_is_best_track"] is False


def test_parse_digital_typhoon_ignores_best_track_pressure_column() -> None:
    body = """
    <table>
      <tr><td>1</td><td><a>201601</a></td><td>NEPARTAK</td>
          <td>W. N. Pacific</td><td>2016-07-03 00:00</td>
          <td>2016-07-09 06:00</td><td>6 Days</td><td>900</td></tr>
    </table>
    """
    storms = parse_digital_typhoon_year(body, 2016, "https://example.test/2016")
    assert len(storms) == 1
    assert storms[0].storm_id == "201601"
    assert storms[0].name == "NEPARTAK"
    assert storms[0].birth_utc == datetime(2016, 7, 3, tzinfo=UTC)
    assert storms[0].death_utc == datetime(2016, 7, 9, 6, tzinfo=UTC)


def test_parse_zenodo_dropsonde_uses_near_surface_level() -> None:
    body = (
        "sec, h:m:s, lat, lon, Press, U wind, V wind, Humi\n"
        "0000,5:30:49,22.11,131.81,180.50,-999,-999,6.78\n"
        "0001,--:--:--,22.12,131.82,925.70,-999,-999,90.0\n"
    )
    profile = parse_zenodo_dropsonde(
        body, filename="DROP20171021T0530S7099015-LV2_pub.csv", source_url="https://example.test"
    )
    assert profile["pressure_hpa"] == 925.7
    assert profile["near_surface_pressure_hpa"] == 925.7
    assert profile["top_pressure_hpa"] == 180.5
    assert profile["launch_time_utc"] == "2017-10-21T05:30:49Z"
    assert profile["pressure_is_best_track"] is False


def test_association_uses_own_buoy_baseline() -> None:
    start = datetime(2022, 9, 13, 18, tzinfo=UTC)
    storm = Storm(
        storm_id="202214",
        name="NANMADOL",
        season=2022,
        birth_utc=start,
        death_utc=datetime(2022, 9, 20, tzinfo=UTC),
        source_url="https://example.test",
    )
    records = []
    for hour in range(200):
        stamp = start - timedelta(hours=72) + timedelta(hours=hour)
        pressure = 1010.0 if stamp < start else 1000.0
        records.append(
            {
                "platform_id": "2022_22-09",
                "wmo_id": "21595",
                "observation_time_utc": stamp.isoformat().replace("+00:00", "Z"),
                "latitude_deg": 30.0,
                "longitude_deg": 135.0,
                "pressure_hpa": pressure,
                "source_url": "https://example.test",
            }
        )
    unique = deduplicate_buoy_records(records)
    result = associate_buoy_observations([storm], unique)
    entry = result["202214"][0]
    assert entry["minimum_pressure_hpa"] == 1000.0
    assert entry["baseline_pressure_hpa"] == 1010.0
    assert entry["depression_hpa"] == 10.0
    assert entry["association_confidence"] == "medium"
    assert entry["pressure_is_best_track"] is False


def test_aircraft_config_never_claims_best_track_pressure() -> None:
    with open("configs/aircraft_pressure_records_2016_2026.json") as handle:
        config = json.load(handle)
    for record in config["records"]:
        assert record["pressure_is_best_track"] is False
        assert record["pressure_hpa"] is None or 800 <= record["pressure_hpa"] <= 1100
    for mission in config["missions_without_public_numeric_pressure"]:
        assert "pressure_hpa" not in mission
