"""Synthetic JMA observation tables; never use best-track pressures."""

from datetime import date

import pytest

from h9dp.jma_observations import normal_pressure, parse_jma_10min


def table(clock="00:10", local="945.1", slp="948.1", quality="data_0_0"):
    values = [clock, local, slp, "--", "25.1", "95", "12.3", "南", "24.6", "南南西", "0"]
    cells = "".join(f'<td class="{quality if i == 2 else "data_0_0"}">{value}</td>'
                    for i, value in enumerate(values))
    return ('<table id="tablefix1"><tr><th>時分</th><th>気圧(hPa)</th></tr>'
            '<tr><th>現地</th><th>海面</th></tr><tr>' + cells + "</tr></table>")


def test_jma_values_are_observed_station_columns_not_bst_or_wind():
    rows = parse_jma_10min(table(), date(2015, 8, 23), "47918", "https://example.test/jma")
    assert len(rows) == 1
    row = rows[0]
    assert row["station_pressure_observed_hpa"] == 945.1
    assert row["slp_observed_hpa"] == 948.1
    assert row["wind_mean_ms_raw"] == "12.3"
    assert row["source_table_row"] == 3
    assert row["pressure_is_best_track"] is False
    assert row["is_verified_cyclone_central_pressure"] is False


def test_jma_jst_is_converted_to_previous_utc_day():
    row = parse_jma_10min(table(), date(2015, 8, 23), "47918", "https://example.test")[0]
    assert row["observation_time_jst"] == "2015-08-23T00:10:00+09:00"
    assert row["observation_time_utc"] == "2015-08-22T15:10:00Z"


def test_jma_2400_rolls_forward_locally_and_keeps_source_clock():
    row = parse_jma_10min(table(clock="24:00"), date(2015, 8, 23), "47918", "https://example.test")[0]
    assert row["observation_time_jst"] == "2015-08-24T00:00:00+09:00"
    assert row["observation_time_utc"] == "2015-08-23T15:00:00Z"
    assert row["source_clock_jst"] == "24:00"


@pytest.mark.parametrize("value", ["--", "///", "×", "948.1)", "948.1 ]", "nan", "100"])
def test_bad_or_qualified_pressures_are_not_filled_from_analysis(value):
    assert normal_pressure((value, "data_0_0")) is None
    assert parse_jma_10min(table(slp=value), date(2015, 8, 23), "47918", "https://example.test") == []


def test_non_normal_quality_and_changed_header_are_not_silently_accepted():
    assert parse_jma_10min(table(quality="data_0_1"), date(2015, 8, 23), "47918", "https://example.test") == []
    with pytest.raises(ValueError, match="pressure-table headers"):
        parse_jma_10min(table().replace("海面", "不明"), date(2015, 8, 23), "47918", "https://example.test")
