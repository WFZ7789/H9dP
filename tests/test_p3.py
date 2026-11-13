"""Small synthetic checks only: no archive, optimizer step or training launch."""

import io

import pytest
import torch
from torch import nn

from h9dp.common import load_config
from h9dp.engine import early_stopping_reached, initialize_device, make_optimizer
from h9dp.model import DensePressureRegressor, build_model, model_architecture
from h9dp.p3 import (
    ARCHITECTURE,
    PARAMETER_LIMIT,
    DynamicReadBlock,
    P3PressureRegressor,
    global_observations,
    patch_offsets,
    sample_observations,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(height=24, width=32, batch=1):
    generator = torch.Generator().manual_seed(12)
    images = torch.randn(batch, 6, 4, height, width, generator=generator)
    mask = torch.ones(batch, 6, 1, height, width, dtype=torch.bool)
    times = torch.tensor([[-300.2, -180.1, -60.2, -30.2, -10.2, -0.2]]).expand(batch, -1).clone()
    return images, times, mask


def test_complete_parameter_budget_and_factory():
    config = load_config("configs/p3_minimal.yaml")
    model = build_model(config["model"])
    assert isinstance(model, P3PressureRegressor)
    assert model_architecture(config["model"]) == ARCHITECTURE
    assert sum(p.numel() for p in model.parameters()) == 108_551 < PARAMETER_LIMIT
    assert all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(module, (nn.Conv2d, nn.Conv3d)) for module in model.modules())
    optimizer = make_optimizer(model, config)
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in model.parameters()})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    assert {group["name"] for group in optimizer.param_groups} == {"backbone", "head"}
    baseline = load_config("configs/train.yaml")
    assert config["data"] == baseline["data"]
    assert config["augmentation"] == baseline["augmentation"]
    legacy_config = baseline["model"]
    legacy_config.update(channels=8, head_width=8)
    legacy = build_model(legacy_config)
    assert isinstance(legacy, DensePressureRegressor)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"architecture": "missing"})
    with pytest.raises(ValueError, match="parameters"):
        P3PressureRegressor({"width": 128})


def test_native500_shape_mask_invariance_and_sparse_embeddings():
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs(500, 500)
    mask[..., :12, :] = False
    altered = images.masked_fill(~mask, float("nan"))
    embedded_sizes = []
    hook = model.input_projection.register_forward_pre_hook(
        lambda _module, args: embedded_sizes.append(args[0].numel() // 4)
    )
    with torch.no_grad():
        reference = model(images, times, mask)
        prediction, details = model.inspect_reads(altered, times, mask)
    hook.remove()
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all() and (prediction > 0).all()
    torch.testing.assert_close(reference, prediction)
    assert max(embedded_sizes) <= 16 * 6 * (3 * 3 + 5 * 5)
    assert len(details["routes"]) == 2
    assert details["brightness"].shape == (1, 16, 4)
    for route in details["routes"]:
        assert route["centers"].shape == (1, 16, 6, 2)
        assert route["radii"].shape == (1, 16, 6, 2)
        assert (route["centers"].abs() <= 1).all()
        assert (route["radii"] > 0).all() and (route["radii"] <= 0.75).all()


def test_native_reads_retain_single_pixel_temperature():
    images = torch.full((1, 6, 4, 9, 9), 250.0)
    images[:, :, 0, 4, 4] = 290.0
    mask = torch.ones(1, 6, 1, 9, 9, dtype=torch.bool)
    grid = (patch_offsets(3, images.device) * (2 / 8)).reshape(1, 1, 1, 9, 2).expand(1, 1, 6, 9, 2)
    raw, valid = sample_observations(images, mask, grid, "nearest")
    assert valid.all()
    torch.testing.assert_close(raw[0, 0, :, 4, 0], torch.full((6,), 290.0))
    torch.testing.assert_close(raw[0, 0, :, 4, 1:], torch.full((6, 3), 250.0))
    coarse, _, _ = global_observations(images, mask, 1)
    assert (coarse[..., 0] < 251).all()  # The fine path does not use this average.


def test_masked_interpolation_and_outside_reads():
    images = torch.full((1, 6, 4, 2, 2), 280.0)
    mask = torch.zeros(1, 6, 1, 2, 2, dtype=torch.bool)
    mask[..., 0, 0] = True
    images = images.masked_fill(~mask, 0)
    grid = torch.tensor([[-0.5, -0.5], [2.0, 2.0]]).reshape(1, 1, 1, 2, 2).expand(1, 1, 6, 2, 2)
    raw, valid = sample_observations(images, mask, grid, "bilinear")
    assert valid[..., 0].all() and not valid[..., 1].any()
    torch.testing.assert_close(raw[..., 0, :], torch.full((1, 1, 6, 4), 280.0))
    assert not raw[..., 1, :].any()


@pytest.mark.parametrize("bf16", [False, True])
def test_raw_brightness_path_retains_absolute_values(bf16):
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        for band in range(4):
            images[:, :, band] = 0.25 * band + 0.5
        _, details = model.inspect_reads(images, times, mask)
    expected = torch.tensor([0.5, 0.75, 1.0, 1.25]).expand(1, 16, -1)
    torch.testing.assert_close(details["brightness"], expected)
    # Bypass the normalized latent features: the head must still read raw values.
    with torch.no_grad():
        first = model.head.mlp[0]
        last = model.head.mlp[-1]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 64] = 1
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1
        original = model(images, times, mask)
        changed = images.clone()
        changed[:, :, 0] += 0.5
        shifted = model(changed, times, mask)
    assert (shifted > original).all()


def test_global_coverage_handles_fully_invalid_local_reads():
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs()
    mask.zero_()
    mask[..., -1, -1] = True
    images.fill_(float("nan"))
    images[..., -1, -1] = 0.75
    prediction, details = model.inspect_reads(images, times, mask)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(details["brightness"], torch.full((1, 16, 4), 0.75))


def test_synthetic_backward_reaches_input_and_dynamic_router():
    torch.manual_seed(13)
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs(batch=2)
    images.requires_grad_()
    times.requires_grad_()
    prediction = model(images, times, mask)
    prediction.mean().backward()  # Gradient check only; no optimizer step.
    assert images.grad is not None and torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
    assert times.grad is not None and torch.isfinite(times.grad).all() and times.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for block in model.blocks:
        assert isinstance(block, DynamicReadBlock)
        output = block.router[-1]
        assert isinstance(output, nn.Linear)
        assert output.weight.grad is not None
        assert (output.weight.grad.abs().sum(1) > 0).all()  # x, y and scale all trainable.


def test_dynamic_reads_are_content_conditioned_and_roundtrip():
    torch.manual_seed(14)
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs()
    first, details = model.inspect_reads(images, times, mask)
    _, changed = model.inspect_reads(images + 1, times, mask)
    assert not torch.equal(details["routes"][0]["centers"], changed["routes"][0]["centers"])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = P3PressureRegressor({}).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(first, restored(images, times, mask))


@pytest.mark.parametrize("failure", ["all_masked", "one_masked_frame", "time", "bands", "mask_dtype", "nonfinite_time"])
def test_invalid_inputs(failure):
    model = P3PressureRegressor({}).eval()
    images, times, mask = inputs()
    if failure == "all_masked":
        mask.zero_()
    elif failure == "one_masked_frame":
        mask[:, 2] = False
    elif failure == "time":
        times = times[:, :-1]
    elif failure == "bands":
        images = images[:, :, :3]
    elif failure == "mask_dtype":
        mask = mask.float()
    else:
        times[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(images, times, mask)


@pytest.mark.parametrize("config", [{"queries": 15}, {"layers": 0}, {"heads": 3}, {"native_patch": 2}, {"dropout": 1}, {"max_radius": -1}])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        P3PressureRegressor(config)


def test_cuda_unavailable_reports_resource_not_approval(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU unavailable") as error:
        initialize_device(load_config("configs/p3_minimal.yaml"))
    assert "elevated" not in str(error.value).lower()
    assert "approval" not in str(error.value).lower()


def test_early_stopping_is_optional_and_uses_validation_best_epoch():
    assert not early_stopping_reached(100, 55, 0)
    assert not early_stopping_reached(74, 55, 20)
    assert early_stopping_reached(75, 55, 20)
    assert not early_stopping_reached(75, 75, 20)
    with pytest.raises(ValueError):
        early_stopping_reached(1, 1, -1)
