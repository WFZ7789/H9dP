"""Synthetic checks for the M1 experts, readout, M5 mapping, and data preparation."""

import numpy as np
import torch

from h9dp.m1 import (
    EXPERTS,
    EYE_SIZE_MAX,
    M5,
    PARAMETER_NAMES,
    M1Direct,
    M1Expert,
    metrics,
)
from scripts.m1 import record_arrays, region_map


def small_expert(spec):
    return M1Expert(
        spec, bands=2, frames=2, image_size=64, radii=(2, 6),
        channels=(4, 6, 8, 8, 8), frame_width=8, temporal_width=8,
    )


def small_control():
    return M1Direct(
        bands=2, frames=2, image_size=64, radii=(2, 6),
        channels=(4, 6, 8, 8, 8), frame_width=8, temporal_width=8,
    )


def small_batch(batch=2):
    return {
        "images": torch.full((batch, 2, 2, 64, 64), 250.0),
        "mask": torch.ones(batch, 2, 1, 64, 64, dtype=torch.bool),
        "numeric": torch.full((batch, 2, 2, 3, 3), 250.0),
        "coverage": torch.full((batch, 2, 3), 64 * 64, dtype=torch.int32),
        "times": torch.zeros(batch, 2),
        "centers": torch.tensor([[[32.0, 32.0]] * 2] * batch),
    }


def test_expert_forward_is_finite_and_bounded():
    for spec in EXPERTS:
        model = small_expert(spec)
        arguments = list(small_batch().values())
        output = model(*arguments).detach()
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()
        if spec.activation == "sigmoid":
            assert float(output[:, 1].max()) <= EYE_SIZE_MAX + 1e-6
            assert float(output[:, 1].min()) >= 0.0
        if spec.activation == "softplus":
            assert float(output[:, 1].min()) >= 0.0


def test_linear_outputs_are_in_physical_units():
    model = small_expert(EXPERTS[0])
    with torch.no_grad():
        for parameter in model.temporal.parameters():
            parameter.zero_()
    model.target_mean.fill_(10.0)
    model.target_std.fill_(2.0)
    output = model(*small_batch().values())
    assert torch.allclose(output[:, 1], torch.full((2,), 10.0))

    control = small_control()
    with torch.no_grad():
        for parameter in control.temporal.parameters():
            parameter.zero_()
    control.target_mean.fill_(60.0)
    control.target_std.fill_(15.0)
    prediction = control(*small_batch().values())
    assert torch.allclose(prediction[:, 0], torch.full((2,), 60.0))


def test_control_forward_is_finite():
    model = small_control()
    output = model(*small_batch().values())
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_full_parameter_counts():
    assert M1Expert(EXPERTS[0]).parameter_count() == 17285
    assert M1Direct().parameter_count() == 88353


def test_experts_do_not_share_weights():
    first, second = small_expert(EXPERTS[0]), small_expert(EXPERTS[0])
    before = [parameter.detach().clone() for parameter in second.parameters()]
    with torch.no_grad():
        for parameter in first.parameters():
            parameter.add_(1.0)
    after = [parameter.detach().clone() for parameter in second.parameters()]
    for left, right in zip(before, after, strict=True):
        assert torch.equal(left, right)


def test_region_map_boundaries_use_distance_from_x_y_center():
    size = 200
    center = np.array([100.5, 100.5], dtype=np.float32)
    region = region_map(center, size, (10, 30))
    assert region[100, 100] == 0
    assert region[100, 110] == 0  # exactly 10 px: inner boundary is inclusive
    assert region[100, 111] == 1
    assert region[100, 130] == 1
    assert region[100, 131] == 2
    # Swapping the center moves the region horizontally, not vertically.
    shifted = region_map(np.array([140.5, 100.5], dtype=np.float32), size, (10, 30))
    assert shifted[100, 140] == 0
    assert shifted[100, 100] == 2


def test_record_arrays_quantize_and_exclude_invalid_pixels():
    image = np.full((1, 4, 500, 500), 250.0, dtype=np.float32)
    image[0, :, 100:120, 100:120] = 300.0
    mask = np.ones((1, 1, 500, 500), dtype=bool)
    mask[0, 0, 100:120, 100:120] = False
    sample = {"centers": [[250.0, 250.0]]}
    quantized, numeric, coverage, stats = record_arrays(sample, image, mask, (32, 96))
    assert quantized.dtype == np.int16
    assert np.allclose(quantized.astype(np.float32) / 100.0 + 150.0, image, atol=0.006)
    assert coverage[0].tolist() == list(coverage[0])
    assert coverage[0].sum() == int(mask.sum())
    assert numeric[0, :, 0, 1].min() > 249.99  # invalid hot pixels are excluded
    assert float(numeric.max()) < 250.01
    assert stats[0, 0, 2] == float(mask.sum())


def test_m5_recovers_known_linear_mapping():
    rng = np.random.default_rng(1)
    parameters = rng.normal(size=(200, 5)) * np.array([4.0, 4.0, 0.1, 5.0, 10.0]) + np.array([3.0, 3.0, 0.2, 5.0, -65.0])
    coefficients = np.array([12.8, 3.4313, 5.6742, 61.6813, 2.5781])
    scale = (35.0 - np.minimum(parameters[:, 4], -50.0)) / 100.0
    design = np.column_stack([np.ones(len(parameters)), scale[:, None] * parameters[:, :4]])
    target = design @ coefficients
    fitted = M5.fit(parameters, target)
    np.testing.assert_allclose(fitted.predict(parameters), target, atol=1e-9)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-9)


def test_m5_caps_cdo_temperature_at_minus_50():
    parameters = np.array([[0.0, 0.0, 0.0, 0.0, -80.0], [0.0, 0.0, 0.0, 0.0, -40.0]])
    fitted = M5(np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
    prediction = fitted.predict(parameters)
    np.testing.assert_allclose(prediction[0], prediction[1])


def test_m5_rejects_rank_deficiency():
    parameters = np.zeros((4, 5))
    with np.testing.assert_raises(ValueError):
        M5.fit(parameters, np.arange(4.0))


def test_metrics_report_signed_bias():
    result = metrics([1.0, 2.0], [2.0, 2.0])
    assert result["rmse"] == float(np.sqrt(0.5))
    assert result["bias"] == -0.5
    assert result["count"] == 2
    with np.testing.assert_raises(ValueError):
        metrics([1.0], [1.0, 2.0])


def test_parameter_names_match_expert_order():
    assert tuple(spec.parameter for spec in EXPERTS) == PARAMETER_NAMES
