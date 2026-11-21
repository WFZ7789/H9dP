"""T1A contract/gradient checks with small synthetic inputs, no dataset or fit."""

import io
import json
import os
import subprocess
import sys

import pytest
import torch

from h9dp.common import load_config
from h9dp.engine import make_optimizer
from h9dp.model import build_model, model_architecture, regression_loss
from h9dp.t1a import (
    ARCHITECTURE,
    T1APressureRegressor,
    haar_split,
    masked_local_mean,
    statistical_readout,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_model(checkpointing=False):
    torch.manual_seed(1)
    return T1APressureRegressor({
        "widths": [4, 6, 8, 12], "blocks_per_stage": 1, "head_width": 8,
        "activation_checkpointing": checkpointing,
    }, target_mean=40.0, target_std=15.0)


def inputs(height=32, width=32, batch=1):
    images = torch.randn(batch, 6, 4, height, width, generator=torch.Generator().manual_seed(1))
    mask = torch.ones(batch, 6, 1, height, width, dtype=torch.bool)
    times = torch.tensor([[-300.2, -180.1, -60.2, -30.2, -10.2, 4.8]]).expand(batch, -1).clone()
    return images, times, mask


def test_haar_is_signed_invertible_and_energy_preserving():
    source = torch.randn(2, 4, 18, 20)
    mask = torch.ones(2, 1, 18, 20, dtype=torch.bool)
    coefficients, valid = haar_split(source, mask)
    ll, lh, hl, hh = coefficients.chunk(4, dim=1)
    reconstructed = torch.empty_like(source)
    reconstructed[..., 0::2, 0::2] = (ll - lh - hl + hh) / 2
    reconstructed[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2
    reconstructed[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2
    reconstructed[..., 1::2, 1::2] = (ll + lh + hl + hh) / 2
    assert valid.all()
    torch.testing.assert_close(source, reconstructed)
    torch.testing.assert_close(source.square().sum(), coefficients.square().sum())
    reversed_coefficients, _ = haar_split(-source, mask)
    torch.testing.assert_close(reversed_coefficients, -coefficients)


def test_haar_rejects_partial_cells_and_pads_odd_edges_as_invalid():
    source = torch.ones(1, 4, 5, 7)
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    mask[..., 0, 0] = False
    source[..., 0, 0] = float("nan")
    coefficients, valid = haar_split(source, mask)
    assert coefficients.shape == (1, 16, 3, 4)
    assert not valid[..., 0, 0].any()
    assert not valid[..., -1, :].any() and not valid[..., :, -1].any()
    assert torch.isfinite(coefficients).all()
    assert not coefficients.masked_select(~valid).any()


def test_local_mean_ignores_invalid_values_and_padding():
    values = torch.full((1, 2, 5, 5), 7.0)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 0, :] = False
    values[..., 0, :] = float("nan")
    mean = masked_local_mean(values, mask, 3)
    torch.testing.assert_close(mean, torch.full_like(mean, 7.0))


def test_statistical_readout_uses_valid_count_not_padded_area():
    values = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0, float("nan"), 999.0]]]])
    mask = torch.tensor([[[[True, True, True, True, False, False]]]])
    summary = statistical_readout(values, mask, 0.5)
    expected = torch.tensor([[0.0, (2.5 + 1e-8) ** 0.5 - 1e-4, 1.5, -1.5]])
    torch.testing.assert_close(summary, expected)
    short = statistical_readout(values[..., :4], mask[..., :4], 0.5)
    torch.testing.assert_close(summary, short)


def test_empty_readout_is_finite_zero_with_finite_gradient():
    values = torch.randn(2, 3, 4, 5, requires_grad=True)
    mask = torch.zeros(2, 1, 4, 5, dtype=torch.bool)
    summary = statistical_readout(values, mask, 0.1)
    assert torch.isfinite(summary).all() and not summary.any()
    summary.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_factory_normalization_budget_optimizer_and_shared_data_contract():
    config = load_config("configs/t1a.yaml")
    baseline = load_config("configs/train.yaml")
    model = build_model(config["model"], target_mean=42.0, target_std=17.0)
    assert isinstance(model, T1APressureRegressor)
    assert model_architecture(config["model"]) == ARCHITECTURE
    assert model.target_mean.item() == 42.0 and model.target_std.item() == 17.0
    assert 400_000 < sum(p.numel() for p in model.parameters()) < 800_000
    assert config["data"] == baseline["data"]
    assert config["augmentation"] == baseline["augmentation"]
    assert config["output_dir"].endswith("/runs/t1a")
    assert config["training"]["torch_compile"] is True
    assert config["training"]["compile_mode"] == "default"
    optimizer = make_optimizer(model, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in model.parameters()})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    assert {group["name"] for group in optimizer.param_groups} == {"head", "backbone"}
    # The existing normalized MSE remains record-equal, independent of architecture.
    prediction = torch.tensor([42.0, 59.0])
    target = torch.tensor([59.0, 42.0])
    torch.testing.assert_close(regression_loss(prediction, target, 17.0).mean(), torch.tensor(1.0))


@pytest.mark.parametrize("height,width", [(32, 32), (35, 37)])
def test_forward_shape_mask_invariance_and_odd_dimensions(height, width):
    model = tiny_model().eval()
    images, times, mask = inputs(height, width, batch=2)
    mask[..., :4, :] = False
    with torch.no_grad():
        expected = model(images, times, mask)
        actual = model(images.masked_fill(~mask, float("nan")), times, mask)
    assert expected.shape == (2,) and torch.isfinite(expected).all()
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_backward_reaches_every_stage_time_encoder_and_all_six_frames(checkpointing):
    model = tiny_model(checkpointing).train()
    images, times, mask = inputs()
    mask[..., :2, :] = False
    images.requires_grad_()
    times.requires_grad_()
    loss = regression_loss(model(images, times, mask), torch.tensor([55.0]), 15.0).mean()
    loss.backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert not images.grad.masked_select(~mask).any()
    assert (images.grad.abs().sum(dim=(2, 3, 4)) > 0).all()
    assert times.grad is not None and torch.isfinite(times.grad).all() and times.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_checkpointing_matches_uncheckpointed_outputs_and_gradients():
    plain, checked = tiny_model(False).train(), tiny_model(True).train()
    checked.load_state_dict(plain.state_dict())
    images, times, mask = inputs()
    first, second = plain(images, times, mask), checked(images, times, mask)
    torch.testing.assert_close(first, second)
    first.sum().backward()
    second.sum().backward()
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters(), strict=True):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad)


def test_temporal_fusion_is_not_exchangeable_and_uses_actual_gaps():
    model = tiny_model().eval()
    images, times, mask = inputs()
    # Keep the nearest frame identical; change only the ordering of the history.
    reordered = images[:, [4, 3, 2, 1, 0, 5]]
    different_gaps = times.clone()
    different_gaps[:, 1] = -120.0
    with torch.no_grad():
        original = model.extract_features(images, times, mask)
        swapped = model.extract_features(reordered, times, mask)
        retimed = model.extract_features(images, different_gaps, mask)
    assert (original - swapped).abs().max() > 1e-5
    assert (original - retimed).abs().max() > 1e-5
    # This checks wiring only; supervised mechanism fitting is a separate CLI check.


def test_linear_head_can_predict_below_mean_and_state_roundtrips():
    model = tiny_model().eval()
    output = model.head[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(-4)
    images, times, mask = inputs()
    expected = model(images, times, mask)
    torch.testing.assert_close(expected, torch.tensor([-20.0]))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = tiny_model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(images, times, mask), expected)


@pytest.mark.parametrize("mutation,message", [
    ("shape", "Expected"),
    ("mask_dtype", "boolean"),
    ("empty", "invalid frame"),
    ("nan_image", "Non-finite"),
    ("nan_time", "Non-finite"),
    ("order", "increasing"),
])
def test_invalid_inputs_fail_explicitly(mutation, message):
    images, times, mask = inputs()
    if mutation == "shape":
        images = images[:, :5]
    elif mutation == "mask_dtype":
        mask = mask.float()
    elif mutation == "empty":
        mask[:, 2] = False
    elif mutation == "nan_image":
        images[..., 3, 3] = float("nan")
    elif mutation == "nan_time":
        times[:, 1] = float("nan")
    elif mutation == "order":
        times[:, 2] = times[:, 1]
    with pytest.raises(ValueError, match=message):
        tiny_model()(images, times, mask)


@pytest.mark.parametrize("config", [
    {"widths": [4, 8]}, {"widths": [4, 0, 8, 12]}, {"blocks_per_stage": 0},
    {"tail_fraction": 0}, {"tail_fraction": 0.6}, {"tail_fraction": "bad"},
])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        T1APressureRegressor(config)


def test_check_cli_is_synthetic_and_does_not_fit_by_default():
    environment = dict(os.environ, MKL_THREADING_LAYER="GNU", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "scripts/check_t1a.py", "--size", "32"],
                            capture_output=True, text=True, check=True, env=environment)
    report = json.loads(result.stdout)
    assert report["architecture"] == ARCHITECTURE and report["synthetic_only"]
    assert report["finite_output"] and report["output_shape"] == [1]
    assert report["mechanism_fits"] == {}
    assert report["parameter_updates"] == 0
