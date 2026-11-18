"""Synthetic query expansion, not real-data collection or an aircraft-data result."""

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.collect_jma_observations import observation_jobs


def write_config(tmp_path, events):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({
        "stations": {"47918": {"name": "石垣島", "prefecture": 91},
                     "47917": {"name": "西表島", "prefecture": 91}},
        "events": events,
    }))
    return path


def event(**changes):
    return {"name": "Example", "year": 2020, "stations": ["47918", "47917"],
            "first_day_jst": "2020-09-01", "last_day_jst": "2020-09-02", **changes}


def test_original_queries_remain_reproducible():
    jobs = observation_jobs()
    assert len(jobs) == 8
    assert len({job[2] for job in jobs}) == 4


def test_expand_source_days_and_deduplicate_only_repeated_station_days(tmp_path):
    path = write_config(tmp_path, [event(), event()])
    jobs = observation_jobs(path)
    assert len(jobs) == 4
    assert {job[5] for job in jobs} == {date(2020, 9, 1), date(2020, 9, 2)}
    # Distinct stations at the same time are distinct observations, not duplicates.
    assert {job[2] for job in jobs} == {"47918", "47917"}


def test_conflicting_event_labels_are_reported_not_double_counted(tmp_path):
    with pytest.raises(ValueError, match="Ambiguous event tags"):
        observation_jobs(write_config(tmp_path, [event(), event(name="Other")]))


def test_reversed_query_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reversed"):
        observation_jobs(write_config(tmp_path, [event(last_day_jst="2020-08-31")]))


def test_malformed_configuration_is_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Invalid JMA query configuration"):
        observation_jobs(path)


def test_expanded_plan_covers_more_sources_but_is_not_a_collection_result():
    path = Path(__file__).resolve().parents[1] / "configs/jma_observation_cases_2015_2025.json"
    jobs = observation_jobs(path)
    assert len(jobs) > 400
    assert len({job[2] for job in jobs}) > 30
    assert len({job[:2] for job in jobs}) > 30
    assert {job[1] for job in jobs} == set(range(2015, 2026))
