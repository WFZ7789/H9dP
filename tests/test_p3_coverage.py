"""Small synthetic checks; no archive extraction or fitting."""

import io
import math

import pytest
import torch

from h9dp.model import build_model, model_architecture
from h9dp.p3_coverage import (
    ARCHITECTURE,
    CoverageBlock,
    CoveragePressureRegressor,
    native_window,
    online_attention,
)


def tiny_config(**updates):
    config = {"architecture": ARCHITECTURE, "width": 32, "heads": 4, "layers": 2, "queries": 16,
              "global_grid": 3, "center_window": 9, "native_window": 5, "dynamic_patch": 5,
              "attention_chunk": 31, "checkpoint_chunks": True, "dropout": 0.}
    config.update(updates)
    return config


def inputs(batch=1, side=24):
    torch.manual_seed(123)
    images = torch.randn(batch, 6, 4, side, side)
    mask = torch.ones(batch, 6, 1, side, side, dtype=torch.bool)
    times = torch.tensor([[-300., -240., -180., -120., -60., 0.]]).expand(batch, -1)
    return images, times, mask


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_factory_parameter_budget_and_default_unchanged():
    model = build_model({"architecture": ARCHITECTURE})
    assert isinstance(model, CoveragePressureRegressor)
    assert 110_000 < sum(p.numel() for p in model.parameters()) < 155_000
    assert model_architecture({}) == "dense250-residual16/v2"
    with pytest.raises(ValueError, match="155,000"):
        CoveragePressureRegressor(tiny_config(width=128))


@pytest.mark.parametrize("chunk,checkpointing", [(1, False), (7, True), (29, False)])
def test_online_attention_output_and_gradients_match_dense(chunk, checkpointing):
    torch.manual_seed(2)
    shapes = [(2, 2, 3, 4), (2, 2, 17, 4), (2, 2, 17, 4), (2, 17, 8)]
    values = [torch.randn(*shape, requires_grad=True) for shape in shapes]
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[0, :8] = False  # Includes an entirely empty first chunk.
    q, k, v, raw = values
    logits = (q @ k.transpose(-1, -2) / 2).masked_fill(~valid[:, None, None], -torch.inf)
    weights = logits.softmax(-1)
    expected = (weights @ v, (weights @ raw[:, None]).mean(1))
    expected_grads = torch.autograd.grad(expected[0].square().sum() + expected[1].square().sum(), values, retain_graph=True)
    result = online_attention(q, k, v, raw, valid, chunk, checkpointing)
    actual_grads = torch.autograd.grad(result[0].square().sum() + result[1].square().sum(), values)
    for x, y in zip(result, expected, strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
    for x, y in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(x, y, atol=4e-6, rtol=4e-5)


def test_empty_attention_is_finite_zero_and_bf16_raw_path_is_fp32():
    q = torch.zeros(1, 2, 2, 4)
    k = torch.zeros(1, 2, 5, 4)
    v = torch.zeros_like(k)
    raw = torch.tensor([[[1.001, 2.002, 3.003, 4.004]]] * 1).expand(1, 5, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, observed = online_attention(q, k, v, raw, torch.ones(1, 5, dtype=torch.bool), 2)
    assert observed.dtype == torch.float32
    torch.testing.assert_close(observed[0, 0], raw[0, 0], rtol=0, atol=1e-6)
    result = online_attention(q, k, v, raw, torch.zeros(1, 5, dtype=torch.bool), 2)
    assert all(torch.isfinite(x).all() and (x == 0).all() for x in result)


def test_empty_chunk_does_not_erase_extremely_negative_logits():
    q = torch.ones(1, 1, 1, 1)
    k = torch.full((1, 1, 4, 1), -1000.)
    v = torch.ones_like(k)
    raw = torch.full((1, 4, 4), 2.)
    valid = torch.tensor([[False, False, True, True]])
    feature, observed = online_attention(q, k, v, raw, valid, 2)
    torch.testing.assert_close(feature, torch.ones_like(feature))
    torch.testing.assert_close(observed, torch.full_like(observed, 2.))


def test_diagnostic_groups_keep_counts_and_extreme_bias():
    from h9dp.engine import prediction_diagnostics

    predictions = [
        {"record": 0, "MOTIS_dP": 40., "predicted_dP": 60.},
        {"record": 1, "MOTIS_dP": 80., "predicted_dP": 80.},
        {"record": 2, "MOTIS_dP": 100., "predicted_dP": 80.},
    ]
    samples = [{"record": i, "sources": ["R3" if i < 2 else "FLDK"]} for i in range(3)]
    report = prediction_diagnostics(predictions, samples)
    assert report["groups"]["dP<60"] == {"count": 1, "rmse": 20., "mae": 20., "bias": 20.}
    assert report["groups"]["dP>=90"]["bias"] == -20.
    assert report["groups"]["R3-only"]["count"] == 2
    assert report["groups"]["FLDK-containing"]["count"] == 1
    assert report["prediction_std"] < report["target_std"]


def test_complete_native_window_preserves_pixels_and_offcenter_target():
    images, _, mask = inputs(side=31)
    images.zero_()
    images[:, :, :, 4, 25] = torch.tensor([1., 2., 3., 4.])[None, None]
    centers = torch.tensor([25 * 2 / 30 - 1, 4 * 2 / 30 - 1])[None, None, None].expand(1, 1, 6, 2)
    raw, valid, xy = native_window(images, mask, centers, 5)
    assert valid.all()
    torch.testing.assert_close(raw[0, 0, 0, 12], torch.tensor([1., 2., 3., 4.]), rtol=0, atol=0)
    assert raw.shape == (1, 1, 6, 25, 4)
    assert len(torch.unique(xy[0, 0, 0], dim=0)) == 25
    _, boundary_valid, _ = native_window(images, mask, torch.ones_like(centers), 5)
    assert not boundary_valid.all()


def test_forward_history_roles_masks_and_router_gradients():
    model = CoveragePressureRegressor(tiny_config())
    images, times, mask = inputs()
    prediction, detail = model._forward(images, times, mask)
    assert prediction.shape == (1,) and torch.isfinite(prediction).all()
    assert detail["brightness"].shape == (1, 2, 3, 6, 4)
    assert detail["source_mass"].shape == (1, 2, 3, 6, 4)
    torch.testing.assert_close(detail["source_mass"].sum(-1), torch.ones(1, 2, 3, 6), atol=2e-6, rtol=2e-6)
    prediction.sum().backward()
    for block in model.blocks:
        assert isinstance(block, CoverageBlock)
        output = block.router[-1]
        assert isinstance(output, torch.nn.Linear)
        grad = output.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad[:2].abs().sum() > 0 and grad[2].abs().sum() > 0
    shift = (detail["routes"][1]["centers"] - detail["routes"][0]["centers"]).abs()
    assert (shift <= 16 * 2 / 23 + 1e-6).all()
    history = [torch.randn(1, 3, 6, 4, requires_grad=True) for _ in range(2)]
    masses = [torch.ones(1, 3, 6, 4) / 4 for _ in range(2)]
    model.head(torch.randn(1, 16, 32), history, masses).sum().backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in history)


def test_masked_pixels_do_not_affect_predictions_and_local_fallback():
    model = CoveragePressureRegressor(tiny_config()).eval()
    images, times, mask = inputs()
    mask[..., 6:18, 6:18] = False
    alternative = images.clone().masked_fill(~mask, math.nan)
    with torch.no_grad():
        a = model(images, times, mask)
        b = model(alternative, times, mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for mode in ("no_dynamic", "fixed_route", "no_history"):
            p, _ = model.inspect_reads(alternative, times, mask, mode)
            assert torch.isfinite(p).all()
    mask[:, 0] = False
    with pytest.raises(ValueError, match="invalid frame"):
        model(images, times, mask)


def test_router_can_reach_edge_and_residual_updates_are_bounded():
    model = CoveragePressureRegressor(tiny_config()).eval()
    with torch.no_grad():
        for block, bias in zip(model.blocks, (4., -4.), strict=True):
            assert isinstance(block, CoverageBlock)
            output = block.router[-1]
            assert isinstance(output, torch.nn.Linear) and output.bias is not None
            output.bias[:2].fill_(bias)
        _, details = model.inspect_reads(*inputs(side=80))
    assert details["routes"][0]["centers"].min() > .98
    delta = details["routes"][1]["centers"] - details["routes"][0]["centers"]
    assert delta.abs().max() <= 16 * 2 / 79 + 1e-6


def test_checkpoint_roundtrip_cpu_bf16_and_deterministic_inspection():
    model = CoveragePressureRegressor(tiny_config()).eval()
    data = inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(*data)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = CoveragePressureRegressor(tiny_config()).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual, details = restored.inspect_reads(*data)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert torch.isfinite(details["brightness"]).all()
    assert details["brightness"].dtype == torch.float32


@pytest.mark.parametrize("update", [{"center_window": 8}, {"queries": 12}, {"dynamic_windows": 2}, {"attention_chunk": 0}, {"refine_pixels": math.inf}])
def test_invalid_config(update):
    with pytest.raises(ValueError):
        CoveragePressureRegressor(tiny_config(**update))
