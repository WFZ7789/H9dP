"""Audit instruments must be correct before interpreting model behavior."""

import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_p3_coverage.py"))


@pytest.mark.parametrize("chunk", [1, 4, 30])
def test_attention_entropy_matches_dense_with_empty_chunks(chunk):
    torch.manual_seed(1)
    q, k = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 17, 4)
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[0, :6] = False
    weights = (q @ k.transpose(-1, -2) / 2).masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
    n = valid.sum(-1)[:, None, None]
    stats = AUDIT["attention_statistics"](q, k, valid, chunk)
    torch.testing.assert_close(stats["kl_from_uniform_nats"], n.float().log() - entropy, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(stats["entropy_effective_fraction"], entropy.exp() / n)
    torch.testing.assert_close(stats["inverse_square_effective_fraction"], 1 / (weights.square().sum(-1) * n))
    torch.testing.assert_close(stats["maximum_weight"], weights.amax(-1))


def test_entropy_uniform_and_masked_very_negative_scores():
    q = torch.ones(1, 1, 1, 1)
    k = torch.full((1, 1, 4, 1), -1000.)
    valid = torch.tensor([[False, False, True, True]])
    stats = AUDIT["attention_statistics"](q, k, valid, 2)
    assert abs(stats["kl_from_uniform_nats"].item()) < 1e-4
    assert abs(stats["entropy_over_log_count"].item() - 1) < 1e-4
    assert abs(stats["maximum_weight"].item() - .5) < 1e-6


@pytest.mark.parametrize("side", [2, 4, 8])
def test_shuffle_preserves_band_time_identity_and_tile_histograms(side):
    base = torch.arange(64.).reshape(1, 1, 1, 8, 8)
    images = torch.cat((base, base + 1000), dim=2).expand(1, 6, 2, 8, 8)
    shuffled = AUDIT["shuffle_pixels_in_tiles"](images, side)
    assert not torch.equal(images, shuffled)
    torch.testing.assert_close(shuffled[:, :, 1] - shuffled[:, :, 0], torch.full((1, 6, 8, 8), 1000.))
    torch.testing.assert_close(shuffled[:, 0], shuffled[:, -1])
    for y in range(0, 8, side):
        for x in range(0, 8, side):
            expected = images[..., y:y + side, x:x + side].flatten(-2).sort().values
            actual = shuffled[..., y:y + side, x:x + side].flatten(-2).sort().values
            torch.testing.assert_close(expected, actual)
    mask = images[:, :, :1] % 2 == 0
    shuffled_mask = AUDIT["shuffle_pixels_in_tiles"](mask, side)
    assert torch.equal(shuffled[:, :, :1] % 2 == 0, shuffled_mask)


def test_uniform_attention_matches_zero_logit_attention():
    q, k, v, raw = torch.randn(2, 3, 4, 2), torch.randn(2, 3, 5, 2), torch.randn(2, 3, 5, 2), torch.randn(2, 5, 8)
    valid = torch.tensor([[False, True, True, False, True], [True] * 5])
    feature, observed = AUDIT["uniform_attention"](q, k, v, raw, valid, 2)
    weights = valid.float() / valid.sum(-1, keepdim=True)
    expected_feature = (weights[:, None, :, None] * v).sum(2)[:, :, None].expand_as(feature)
    expected_raw = (weights[..., None] * raw).sum(1)[:, None].expand_as(observed)
    torch.testing.assert_close(feature, expected_feature)
    torch.testing.assert_close(observed, expected_raw)


def test_head_slices_match_real_456_dimensional_contract():
    parts = AUDIT["head_slices"](SimpleNamespace(width=64, blocks=[0, 1]))
    assert parts == {"hidden": slice(0, 192), "brightness": slice(192, 336),
                     "contrasts": slice(336, 432), "source_mass": slice(432, 456)}


def test_full_trace_handles_bf16_without_changing_model_outputs():
    from h9dp.p3_coverage import CoveragePressureRegressor

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = CoveragePressureRegressor({"width": 32, "heads": 4, "layers": 2, "center_window": 9,
                                          "native_window": 5, "dynamic_patch": 5, "global_grid": 3,
                                          "attention_chunk": 31, "dropout": 0.}).eval().requires_grad_(False)
        images = torch.randn(1, 6, 4, 24, 24)
        mask = torch.ones(1, 6, 1, 24, 24, dtype=torch.bool)
        times = torch.tensor([[-300., -240., -180., -120., -60., 0.]])
        storage = {}
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected, _ = model.inspect_reads(images, times, mask)
            with AUDIT["make_trace"](model, storage):
                actual, _ = model.inspect_reads(images, times, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert storage["layer1_query_cosine"][0].shape == (1, 240)
        assert len(storage["layer2_kl_from_uniform_nats"]) == 1
    finally:
        torch.set_num_threads(previous)


def test_data_protocol_reports_overlap_future_frames_and_order():
    index = {"bands": [8, 9, 10, 16], "normalization": {}, "target_normalization": {},
             "frames": [{"files": ["a", "b", "c", "d"]}],
             "samples": [{"split": split, "frames": [0] * 6, "storm_id": "2026:synthetic",
                          "time_offsets_minutes": [-25, -20, -15, -10, -5, 1]} for split in ("train", "validation")]}
    result = AUDIT["data_protocol"](index)
    json.dumps(result, allow_nan=False)
    assert result["shared_storms"] == result["shared_original_frame_groups"] == 1
    assert result["train"]["future_frames"] == 1
    assert result["validation"]["all_sequences_strictly_increasing"]
    assert math.isfinite(result["train"]["time_offsets_minutes_q0_q50_q100"][1])
