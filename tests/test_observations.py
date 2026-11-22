"""Small synthetic observation/POCI fixtures; no downloads or training."""

import csv
import io
from datetime import datetime, timedelta

import pytest

from h9dp.observations import (
    Station,
    TrackPoint,
    distance_km,
    pair_surface_rows,
    parse_slp,
    read_stations,
    read_tracks,
    select_pairs,
    track_at,
)
from scripts.audit_surface_poci import audit, read_json


def csv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return io.StringIO(output.getvalue())


BASE_TIME = datetime.fromisoformat("2020-08-01T00:00:00")


def track(hour: int = 0, lon: float = 130, poci: float | None = 1008):
    return TrackPoint("storm1", "Example", 2020, BASE_TIME + timedelta(hours=hour), 25, lon, poci, "WP012020")


@pytest.mark.parametrize("value", ["99999,1", "10080,2", "10080,9", "nan,1", "bad,1", "10080"])
def test_slp_rejects_missing_failed_qc_and_malformed(value):
    assert parse_slp(value) is None


def test_slp_units_and_passed_qc():
    assert parse_slp("09876,1") == (987.6, "1")
    assert parse_slp("10080,5") == (1008.0, "5")


def test_track_position_interpolates_but_poci_does_not():
    points = [track(), track(6, 132, 1010)]
    match = track_at(points, BASE_TIME + timedelta(hours=2))
    assert match is not None
    lat, lon, nearest = match
    assert lat == 25
    assert lon == pytest.approx(130 + 2 / 3)
    assert nearest.poci == 1008
    assert track_at(points, BASE_TIME - timedelta(hours=1)) is None
    assert track_at([track(), track(12)], BASE_TIME + timedelta(hours=2)) is None
    assert track_at([track(poci=None)], BASE_TIME) is None


def test_dateline_distance_and_position():
    assert distance_km(0, 179.9, 0, -179.9) == pytest.approx(22.239, abs=0.001)
    match = track_at([track(0, 179), track(6, -179)], BASE_TIME + timedelta(hours=3))
    assert match is not None
    _, lon, _ = match
    assert lon == -180


def test_surface_pair_retains_provenance_and_never_claims_central_truth():
    station = Station("12345699999", "TEST", 25, 130, 10, 2010, 2025)
    row = {
        "STATION": station.identifier, "DATE": "2020-08-01T00:00:00",
        "LATITUDE": "25", "LONGITUDE": "130", "ELEVATION": "10",
        "SLP": "09876,1", "REPORT_TYPE": "FM-12",
    }
    pairs = pair_surface_rows(csv_text([row]), station, [track()], "https://example.test/raw.csv")
    assert len(pairs) == 1
    assert pairs[0]["slp_observed_hpa"] == 987.6
    assert pairs[0]["dp_station_proxy_hpa"] == pytest.approx(20.4)
    assert pairs[0]["centre_distance_km"] == 0
    assert pairs[0]["observation_source_line"] == 2
    assert pairs[0]["is_central_pressure_truth"] is False
    assert pairs[0]["poci_time_utc"] == "2020-08-01T00:00:00Z"
    for change in ({"SLP": "09876,2"}, {"LONGITUDE": "135"}, {"ELEVATION": "500"}):
        assert not pair_surface_rows(csv_text([{**row, **change}]), station, [track()], "https://example.test")


def test_ibtracs_units_row_and_other_track_types_are_not_data():
    row = {
        "BASIN": "WP", "TRACK_TYPE": "main", "SEASON": "2020", "NATURE": "TS",
        "LAT": "25", "LON": "130", "USA_POCI": "1008", "SID": "storm1",
        "NAME": "EXAMPLE", "ISO_TIME": "2020-08-01 00:00:00", "USA_ATCF_ID": "WP012020",
    }
    rows = [dict.fromkeys(row, " "), row, {**row, "TRACK_TYPE": "spur"},
            {**row, "SEASON": "bad"}, {**row, "ISO_TIME": "bad"}]
    tracks = read_tracks(csv_text(rows), 2016, 2025)
    assert len(tracks["storm1"]) == 1
    assert tracks["storm1"][0].poci == 1008


def test_station_filter_skips_bad_dates_and_missing_coordinates():
    row = {"USAF": "123456", "WBAN": "99999", "STATION NAME": "TEST",
           "LAT": "25", "LON": "130", "ELEV(M)": "10", "BEGIN": "20100101", "END": "20251231"}
    stations = read_stations(csv_text([row, {**row, "BEGIN": "bad"}, {**row, "LAT": ""}]))
    assert len(stations) == 1
    assert stations[0].identifier == "12345699999"


def selection_row(minute: int = 0, station: str = "one", proxy_dp: float = 40):
    return {"sid": "same_cyclone", "name": "Existing", "season": 2020,
            "observation_time_utc": (BASE_TIME + timedelta(minutes=minute)).isoformat() + "Z",
            "station_id": station, "centre_distance_km": 2, "observation_source_line": minute + 2,
            "dp_station_proxy_hpa": proxy_dp, "cyclone_in_existing_inventory": True}


def test_selection_has_no_cyclone_quota_hour_collapse_or_time_separation():
    rows = [selection_row(minute) for minute in range(24)]
    rows.append(selection_row(0, station="two"))
    assert len(select_pairs(rows, count=100)) == 25
    assert all(row["cyclone_in_existing_inventory"] for row in select_pairs(rows))


def test_selection_is_reproducible_and_independent_of_cyclone_and_year_tags():
    rows = [selection_row(minute) for minute in range(24)]
    selected = select_pairs(rows, count=10, seed=1)
    assert selected == select_pairs(list(reversed(rows)), count=10, seed=1)
    changed_tags = [{**row, "sid": f"changed_{i}", "season": 1900 + i,
                     "name": f"Another{i}", "cyclone_in_existing_inventory": False}
                    for i, row in enumerate(rows)]
    assert [row["observation_time_utc"] for row in selected] == [
        row["observation_time_utc"] for row in select_pairs(changed_tags, 10, 1)]


def test_training_validation_test_tags_and_six_hour_proximity_do_not_affect_selection():
    cases = [(split, offset) for split in ("train", "validation", "test")
             for offset in (0, 1, 359, 360, 361)]
    rows = [{**selection_row(minute=i), "existing_split": split,
             "minutes_from_training_record": offset,
             "exact_named_time_in_existing_inventory": offset == 0}
            for i, (split, offset) in enumerate(cases)]
    assert len(select_pairs(rows, count=100)) == len(cases)
    selected = select_pairs(rows, count=7, seed=1)
    retagged = [{**row, "existing_split": "none", "minutes_from_training_record": 10000,
                 "exact_named_time_in_existing_inventory": False} for row in rows]
    assert [row["observation_time_utc"] for row in selected] == [
        row["observation_time_utc"] for row in select_pairs(retagged, count=7, seed=1)]


def test_surface_screening_rejects_weak_dp_and_deduplicates_physical_observations():
    rows = [selection_row(proxy_dp=value) for value in (0, 15, 30)]
    assert select_pairs(rows) == []
    strong = selection_row(proxy_dp=30.1)
    assert select_pairs([strong, strong]) == [strong]


def test_offline_audit_rejects_the_failed_legacy_collection(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Rejected legacy"):
        audit(tmp_path, tmp_path / "unused.json")


def test_json_audit_errors_have_source_context(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("not JSON")
    with pytest.raises(ValueError, match="Cannot read JSON source"):
        read_json(source)
    with pytest.raises(ValueError, match="Cannot read JSON source"):
        read_json(tmp_path / "absent.json")
