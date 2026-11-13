"""Fitability helpers only: no archive extraction, real data or model fitting."""

import copy

import numpy as np
import pytest
import torch

from h9dp.t1a_fit import fit_metrics, input_identity_audit, passes_gate, select_records


def sample_index():
    return {"samples": [
        {"record": i, "split": "train" if i < 40 else "validation",
         "dP": 40.0 + i, "name": f"name-{i}", "quality": "Good", "frames": [i] * 6}
        for i in range(50)
    ]}


def test_selection_is_one_reproducible_training_only_set_of_24():
    index = sample_index()
    first = select_records(index)
    second = select_records(index)
    assert first == second and len(first) == 24
    assert len({sample["record"] for sample in first}) == 24
    assert all(sample["split"] == "train" for sample in first)
    assert select_records(index, seed=43) != first
    first[0]["frames"][0] = -100
    assert index["samples"][first[0]["record"]]["frames"][0] != -100


def test_selection_does_not_depend_on_targets_quality_names_or_input_order():
    index = sample_index()
    modified = copy.deepcopy(index)
    for sample in modified["samples"]:
        sample.update(dP=9999, name="other", quality="Poor")
    modified["samples"].reverse()
    assert [s["record"] for s in select_records(index)] == [
        s["record"] for s in select_records(modified)
    ]


@pytest.mark.parametrize("count", [0, 1, 41])
def test_invalid_subset_size(count):
    with pytest.raises(ValueError):
        select_records(sample_index(), count=count)


def test_duplicate_record_across_splits_is_rejected():
    index = sample_index()
    index["samples"][-1]["record"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        select_records(index)


def test_identity_audit_detects_conflicting_labels_and_ignores_masked_fill():
    images = torch.ones(2, 6, 4, 4, 4)
    offsets = torch.tensor([[-300, -180, -60, -30, -10, 0]], dtype=torch.float32).expand(2, -1).clone()
    mask = torch.ones(2, 6, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, 0] = False
    images[0, ..., 0, 0] = float("nan")
    images[1, ..., 0, 0] = 9999
    targets = torch.tensor([40.0, 42.0])
    report = input_identity_audit(images, offsets, mask, targets, [1, 2])
    assert report["unique_inputs"] == 1
    assert report["duplicate_groups"][0]["conflicting_targets"]
    assert report["exact_input_irreducible_rmse_hpa"] == 1.0
    offsets[1, -1] = 0.5
    assert input_identity_audit(images, offsets, mask, targets, [1, 2])["unique_inputs"] == 2


def test_metrics_report_real_units_spread_and_correlation():
    targets = np.array([40.0, 50.0, 70.0])
    metrics = fit_metrics(targets + 0.5, targets)
    assert metrics["rmse_hpa"] == 0.5 and metrics["mae_hpa"] == 0.5
    assert metrics["bias_hpa"] == 0.5
    assert metrics["std_ratio"] == pytest.approx(1.0)
    assert metrics["correlation"] == pytest.approx(1.0)
    assert passes_gate(metrics)


def test_constant_prediction_does_not_pass_gate():
    metrics = fit_metrics([50.0, 50.0, 50.0], [40.0, 50.0, 70.0])
    assert metrics["std_ratio"] == 0 and metrics["correlation"] is None
    assert not passes_gate(metrics)


def test_exact_fit_passes_but_bad_spread_or_large_error_does_not():
    metrics = fit_metrics([30, 50, 80], [30, 50, 80])
    assert passes_gate(metrics)
    assert not passes_gate(dict(metrics, std_ratio=0.5))
    assert not passes_gate(dict(metrics, rmse_hpa=1.1))
    assert not passes_gate(dict(metrics, correlation=0.5))


@pytest.mark.parametrize("predictions,targets", [
    ([1, float("nan")], [1, 2]), ([1, 2], [1, float("inf")]),
])
def test_nonfinite_metrics_fail(predictions, targets):
    with pytest.raises(FloatingPointError):
        fit_metrics(predictions, targets)


def test_metric_shape_mismatch():
    with pytest.raises(ValueError):
        fit_metrics([1, 2, 3], [1, 2])
