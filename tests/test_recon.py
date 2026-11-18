"""Synthetic aircraft VDM fixtures; no network, training, or external archives."""

from datetime import datetime

import pytest

from h9dp.observations import TrackPoint
from h9dp.recon import ahi_zenith, filename_time, pair_vdm, parse_vdm

OLD = """URPN12 KNHC 230612
VORTEX DATA MESSAGE   CP032015
A. 23/05:45:10Z
B. 14 deg 57 min N
  161 deg 39 min W
C. 850 mb 1485 m
D. 28 kt
H. 960 mb
P. AF304 0303C KILO OB 04
CNTR DROPSONDE SFC WIND 205 / 8 KT
"""
NEW = """URPN12 KNHC 220546
VORTEX DATA MESSAGE   EP142018
A. 22/05:20:20Z
B. 14.51 deg N 154.11 deg W
C. 700 mb 2515 m
D. 930 mb
H. 146 kt
U. AF307 1214E LANE OB 05
"""


def test_old_vdm_uses_h_pressure_and_degree_minutes():
    row = parse_vdm(OLD, filename_time("REPPN2-KNHC.201508230612.txt"), "https://example.test/old")
    assert row is not None
    assert row["pc_observed_hpa"] == 960
    assert row["pressure_message_field"] == "H"
    assert row["lat"] == pytest.approx(14.95)
    assert row["lon"] == pytest.approx(-161.65)
    assert row["pressure_method"] == "aircraft_report_center_dropsonde_mentioned"
    assert row["mission_id"] == "AF304 0303C"
    assert row["observation_time_utc"] == "2015-08-23T05:45:10Z"


def test_new_vdm_uses_d_pressure_not_h_wind():
    row = parse_vdm(NEW, filename_time("REPPN2-KNHC.201808220546.txt"), "https://example.test/new")
    assert row is not None
    assert row["pc_observed_hpa"] == 930
    assert row["pressure_message_field"] == "D"
    assert row["lat"] == 14.51
    assert row["lon"] == -154.11
    assert row["mission_id"] == "AF307 1214E"


def test_month_boundary_fix_is_not_assigned_to_wrong_month():
    sent = datetime.fromisoformat("2018-09-01T00:05:00")
    row = parse_vdm(NEW.replace("22/05:20:20Z", "31/23:55:00Z"), sent, "https://example.test")
    assert row is not None
    assert row["observation_time_utc"] == "2018-08-31T23:55:00Z"


def test_malformed_pressure_or_position_is_not_a_central_observation():
    sent = filename_time("REPPN2-KNHC.201808220546.txt")
    assert parse_vdm(NEW.replace("D. 930 mb", "D. NA"), sent, "https://example.test") is None
    assert parse_vdm(NEW.replace("D. 930 mb", "D. 130 mb"), sent, "https://example.test") is None
    assert parse_vdm(NEW.replace("14.51 deg N", "unknown N"), sent, "https://example.test") is None


def test_visibility_rejects_atlantic_and_retains_hawaii():
    assert ahi_zenith(0, 140.7) == pytest.approx(0, abs=0.000001)
    assert ahi_zenith(20, -160) is not None
    assert ahi_zenith(20, -60) is None


def test_pair_uses_observed_center_pressure_and_unmodified_poci():
    observation = parse_vdm(NEW, filename_time("REPPN2-KNHC.201808220546.txt"), "https://example.test")
    assert observation is not None
    point = TrackPoint("2018227N10112", "LANE", 2018, datetime.fromisoformat("2018-08-22T06:00:00"),
                       14.5, -154.1, 1008, "EP142018")
    row = pair_vdm(observation, [point])
    assert row is not None
    assert row["dp_hpa"] == 78
    assert row["ahi_visible"] is True
    assert row["poci_offset_minutes"] == pytest.approx(39 + 40 / 60)
    assert row["is_surface_station_proxy"] is False
    assert pair_vdm(observation, []) is None
    far = TrackPoint("sid", "LANE", 2018, datetime.fromisoformat("2018-08-23T06:00:00"),
                     14.5, -154.1, 1008, "EP142018")
    assert pair_vdm(observation, [far]) is None
