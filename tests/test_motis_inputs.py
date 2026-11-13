"""Small synthetic checks for the MOTIS tabular audit; no external data needed."""

import json
from pathlib import Path

import numpy as np

from scripts.audit_motis_inputs import (
    COLUMNS,
    audit,
    design_matrix,
    observation_key,
    oof_predictions,
    unique_lookup,
)


def source_row(**changes):
    row = {
        "Name": "Example", "Year": "2020", "Date": "704", "Time": "425",
        "Satellite": "AQUA", "T CDO": "-65", "dP MOTIS": "20", "npix": "16",
        **dict.fromkeys(COLUMNS, "0"),
    }
    row.update(changes)
    return row


def test_key_normalizes_formatting_but_not_observation_time():
    key = observation_key(" Example ", "2020", "0704", "0425", "aqua")
    assert key == observation_key("example", 2020, 704, 425, "AQUA")
    assert key != observation_key("example", 2020, 704, 426, "AQUA")
    with np.testing.assert_raises(ValueError):
        observation_key("example", 2020, 230, 425, "AQUA")


def test_duplicate_pass_is_not_silently_overwritten():
    with np.testing.assert_raises_regex(ValueError, "Ambiguous"):
        unique_lookup([source_row(), source_row(Name=" EXAMPLE ")])


def test_cdo_scale_caps_warm_values_and_preserves_column_order():
    rows = [source_row(**{"B24 dT": "2", "T CDO": temperature}) for temperature in ("-65", "-40")]
    features = design_matrix(rows)
    assert features.shape == (2, 10)
    np.testing.assert_allclose(features[:, 0], [2.0, 1.7])
    assert np.count_nonzero(features[:, 1:]) == 0
    with np.testing.assert_raises_regex(ValueError, "finite"):
        design_matrix([source_row(**{"B24 dT": "nan"})])


def test_oof_is_reproducible_and_excludes_held_out_targets():
    features = np.arange(60, dtype=float).reshape(30, 2)
    target = np.sin(np.arange(30))
    predicted, folds = oof_predictions(features, target)
    repeated, repeated_folds = oof_predictions(features, target)
    np.testing.assert_array_equal(folds, repeated_folds)
    np.testing.assert_allclose(predicted, repeated)
    assert set(folds) == set(range(5))
    changed = target.copy()
    changed[folds == 0] += 1000
    changed_predictions, _ = oof_predictions(features, changed)
    np.testing.assert_allclose(predicted[folds == 0], changed_predictions[folds == 0])
    with np.testing.assert_raises_regex(ValueError, "OOF"):
        oof_predictions(features[:2], target[:2])


def test_audit_uses_zero_based_ids_and_exact_matching(tmp_path: Path):
    import csv

    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_row()))
        writer.writeheader()
        writer.writerow(source_row())
    records = tmp_path / "records.json"
    record = {
        "name": "Example", "satellite": "AQUA", "t_cdo": -64, "pixels": 25,
        "time": {"year": 2020, "date_mmdd": 704, "time_utc_hhmm": 425,
                 "iso_utc": "2020-07-04T04:25:00Z"},
    }
    records.write_text(json.dumps({"records": [record]}))
    sample = {
        "record": 0, "name": "Example", "year": 2020,
        "target_time": "2020-07-04T04:25:00+00:00", "quality": "Fair", "dP": 21,
    }
    (tmp_path / "train.json").write_text(json.dumps([sample]))
    (tmp_path / "validation.json").write_text("[]")
    report = audit(source, records, tmp_path)
    assert report["splits"]["train"]["matched"] == 1
    assert report["splits"]["train"]["pairs"][0]["record"] == 0
    assert report["splits"]["train"]["source_dP_vs_current_dP"]["rmse"] == 1
    assert report["splits"]["train"]["non_null_current_field_counts"]["t_cdo"] == 1
    assert report["splits"]["train"]["source_T CDO_vs_current_t_cdo"]["rmse"] == 1
    assert report["splits"]["train"]["source_npix_vs_current_pixels"]["rmse"] == 9
    assert "oracle_oof" not in report
    # A one-minute gap remains missing; no nearest-pass matching is performed.
    record["time"]["time_utc_hhmm"] = 426
    record["time"]["iso_utc"] = "2020-07-04T04:26:00Z"
    records.write_text(json.dumps({"records": [record]}))
    sample["target_time"] = "2020-07-04T04:26:00+00:00"
    (tmp_path / "train.json").write_text(json.dumps([sample]))
    assert audit(source, records, tmp_path)["splits"]["train"]["matched"] == 0
