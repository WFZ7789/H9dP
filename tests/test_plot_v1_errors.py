import csv

import numpy as np
import pytest

from h9dp.r1 import record_folds
from scripts.plot_v1_errors import PRESSURE_BINS, load_predictions, summarize_bins


@pytest.fixture
def prediction_table(tmp_path):
    targets = (29.0, 30.0, 50.0, 80.0, 100.0)
    catalog = [
        {"name": "Synthetic", "time": {"year": 2026}, "quality": "Good", "dP": target, "poci": poci}
        for target, poci in zip(targets, (1000, 1001, 1002, 1000, 1004), strict=True)
    ]
    samples = [
        {"record": i, "name": label["name"], "year": 2026, "quality": "Good", "dP": label["dP"]}
        for i, label in enumerate(catalog)
    ]
    rows = [
        (1, fold, i, "Good", target, target + i - 2)
        for i, (target, fold) in enumerate(zip(targets, record_folds(5, 5, 1), strict=True))
    ]
    path = tmp_path / "predictions.csv"

    def write(rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("seed", "fold", "record", "quality", "target", "prediction"))
            writer.writerows(rows)

    return path, samples, catalog, rows, write


def test_residual_sign_and_bin_boundaries(prediction_table):
    path, samples, catalog, rows, write = prediction_table
    write(rows)
    predictions = load_predictions(path, samples, catalog)
    np.testing.assert_equal(predictions.dp_error, [-2, -1, 0, 1, 2])
    np.testing.assert_equal(predictions.quality, ["Good"] * 5)
    assert list(summarize_bins(predictions.target_dp, predictions.dp_error)) == [
        (-np.inf, 30, 1, -2, 2),
        (30, 40, 1, -1, 1),
        (50, 60, 1, 0, 0),
        (80, 100, 1, 1, 1),
        (100, np.inf, 1, 2, 2),
    ]


def test_bin_by_predicted_pressure_not_target_pressure(prediction_table):
    path, samples, catalog, rows, write = prediction_table
    write(rows)
    predictions = load_predictions(path, samples, catalog)
    np.testing.assert_equal(predictions.target_pressure, [971, 971, 952, 920, 904])
    np.testing.assert_equal(predictions.predicted_pressure, [973, 972, 952, 919, 902])
    np.testing.assert_equal(predictions.predicted_pressure - predictions.target_pressure, -predictions.dp_error)
    error = -predictions.dp_error
    # Record 3 crosses the 920 hPa boundary when binned by the prediction.
    by_target = list(summarize_bins(predictions.target_pressure, error, PRESSURE_BINS))
    by_prediction = list(summarize_bins(predictions.predicted_pressure, error, PRESSURE_BINS))
    assert by_target[:2] == [(900, 920, 1, -2, 2), (920, 940, 1, -1, 1)]
    assert by_prediction == [
        (900, 920, 2, -1.5, np.sqrt(2.5)),
        (940, 960, 1, 0, 0),
        (960, np.inf, 2, 1.5, np.sqrt(2.5)),
    ]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing", "Missing out-of-fold"),
        ("duplicate", "Unexpected seed, record, or fold"),
        ("target", "Invalid prediction or MOTIS label"),
        ("nonfinite", "Invalid prediction or MOTIS label"),
        ("poci", "Missing or invalid MOTIS POCI"),
    ],
)
def test_reject_bad_predictions(prediction_table, change, error):
    path, samples, catalog, rows, write = prediction_table
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "target":
        rows[0] = (*rows[0][:4], 30.0, rows[0][5])
    elif change == "poci":
        del catalog[0]["poci"]
    else:
        rows[0] = (*rows[0][:5], float("nan"))
    write(rows)
    with pytest.raises(ValueError, match=error):
        load_predictions(path, samples, catalog)
