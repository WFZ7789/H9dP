from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from h9dp.r1 import frame_features, record_folds, ridge_path, scores, summarize_time
from h9dp.r1_gp import SpectralGP
from h9dp.r1_metric import image_fingerprint, relevance_components
from h9dp.r1_net import R1Net
from h9dp.r1_track import register_earlier
from scripts.r1_adaptive_kernel import latest_snapshot
from scripts.r1_neural import batches, fit_normalizer, standardized
from scripts.r1_observed import select_frames, terminal_pressure


def test_asymmetric_xy_center_and_mask_are_respected():
    image = np.full((4, 256, 300), 7000, dtype=np.int16)  # 220 K
    image[:, 70:81, 192:203] = 11000                   # 260 K at (x,y) = (197,75)
    valid = np.ones(image.shape[1:], dtype=bool)
    valid[75, 198] = False
    image[:, 75, 198] = 20000                         # must not count as a warm measurement
    features = frame_features(image, valid, np.array([197.5, 75.5]))
    assert features.profile[0] == pytest.approx(260)
    assert features.structure[1] == pytest.approx(260)
    assert features.views.shape == (10, 64, 64)
    assert features.views[0, 32, 32] == 2000
    assert features.views[0, 32, 33] == 0
    assert features.views[4, 32, 33] == 0


def test_fingerprint_ignores_invalid_block_and_differences_use_different_bands():
    views = np.zeros((1, 2, 10, 64, 64), dtype=np.int16)
    views[:, 0, 9] = 100
    views[:, 0, 7] = 2000
    views[:, 0, 9, :4, :4] = 0
    scene, core = image_fingerprint(views)
    assert scene.shape == (1, 2, 1024)
    assert core.shape == (1, 2, 256)
    np.testing.assert_allclose(core[0, 0, 64:128], 20)
    np.testing.assert_allclose(core[0, 0, 192:], 20)
    assert np.isnan(scene[0, 0, 0])
    assert np.isnan(scene[0, 1]).all()


def test_temporal_summary_uses_most_recent_available_not_last_slot():
    values = np.array([[[10.0, 0], [20, np.nan], [9999, 9999], [99, 10]]])
    times = np.array([[-100.0, -10, 0, np.nan]])
    valid = np.array([[True, True, False, True]])
    result = summarize_time(values, times, valid)
    expected = (10 * np.exp(-2) + 20) / (1 + np.exp(-2))
    np.testing.assert_array_equal(result["latest"], [[20, 0]])
    np.testing.assert_allclose(result["recent"], [[expected, 0]])
    np.testing.assert_allclose(result["trend"], [[expected - 10, 0]])
    with pytest.raises(ValueError, match="at least one"):
        summarize_time(values, times, np.zeros_like(valid))


def test_record_folds_repeat_and_cover_without_group_metadata():
    first = record_folds(21, 5, 12)
    np.testing.assert_array_equal(first, record_folds(21, 5, 12))
    assert set(np.bincount(first)) == {4, 5}
    assert not np.array_equal(first, record_folds(21, 5, 13))
    with pytest.raises(ValueError):
        record_folds(3, 4, 12)


def test_weighted_kernel_ridge_matches_primal_unpenalized_intercept():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(12, 5))
    y = 13 + rng.normal(size=9)
    w = np.linspace(0.4, 2.0, len(y))
    penalty = np.array([0.003, 1.4])
    identity = np.diag([0, 1, 1, 1, 1, 1])
    design = np.column_stack((np.ones(len(x)), x))
    a, b = design[:len(y)], design[len(y):]
    primal = np.column_stack([b @ np.linalg.solve(a.T @ (a * w[:, None]) + alpha * identity, a.T @ (w * y))
                              for alpha in penalty])
    actual = ridge_path(x @ x.T, y, w, penalty)
    np.testing.assert_allclose(actual, primal, atol=1e-9)


def test_relevance_only_reads_training_labels():
    rng = np.random.default_rng(17)
    x = {"scene": rng.normal(size=(13, 10))}
    y = rng.normal(size=13)
    quality = np.full(len(y), "Fair")
    train, held = np.arange(10), np.arange(10, 13)
    first = relevance_components(x, train, held, y, quality)
    y[held] += 250_000
    quality[held] = "Good"
    second = relevance_components(x, train, held, y, quality)
    np.testing.assert_array_equal(first[1]["scene"][1], second[1]["scene"][1])
    gram = first[1]["scene"][0]
    assert np.linalg.eigvalsh(gram).min() > -1e-9


def test_normalizer_and_fixed_batch_padding():
    data = np.array([[[3.0, np.nan], [100, 100]], [[5.0, np.nan], [7.0, np.nan]]])
    loc, scale = fit_normalizer(data, np.array([[True, False], [True, True]]))
    np.testing.assert_allclose(loc, [5, 0])
    assert np.all(standardized(data, loc, scale)[[0, 1], [0, 1], 1] == 0)
    result = list(batches(np.arange(6), 4, np.random.default_rng(8)))
    assert all(len(rows) == 4 for rows, _ in result)
    np.testing.assert_array_equal(np.sort(np.concatenate([rows[valid] for rows, valid in result])), np.arange(6))


def test_model_missing_frames_have_no_effect_and_metrics_use_prediction_minus_target():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    net = R1Net(8, "fused").eval()
    x = torch.randn(2, 3, 10, 64, 64)
    x[:, :, [4, 9]] = 100
    profile = torch.randn(2, 3, 8)
    times = torch.tensor([[-180.0, -30, 0], [-180, -30, 0]])
    valid = torch.tensor([[True, False, True], [True, True, False]])
    with torch.no_grad():
        before = net(x, profile, times, valid)
        x[~valid] = 500
        profile[~valid] = -20
        times[~valid] = 30
        after = net(x, profile, times, valid)
    assert before.shape == (2, 6)
    torch.testing.assert_close(before, after)
    result = scores(np.array([40.0, 70.0]), np.array([44.0, 68.0]), np.array(["Good", "Fair"]))
    assert result["all"]["bias"] == pytest.approx(1.0)
    assert result["Good"]["rmse"] == pytest.approx(4.0)


def test_tracking_recovers_xy_translation_but_preserves_prior_without_coverage():
    y, x = np.mgrid[:256, :256]
    shape = (1800 * np.exp(-((x - 130) ** 2 + (y - 125) ** 2) / 150)
             + 950 * np.exp(-((x - 113) ** 2 + (y - 138) ** 2) / 220))
    late = np.repeat(np.asarray(7000 + shape, dtype=np.int16)[None], 4, axis=0)
    early = np.roll(late, shift=(10, -18), axis=(1, 2))
    ok = np.ones(shape.shape, dtype=bool)
    center = np.array([128.5, 128.5])
    result = register_earlier(early, ok, center, late, ok, center, 60)
    assert result.accepted
    np.testing.assert_array_equal(result.center, center + [-18, 10])
    blocked = register_earlier(early, ~ok, center, late, ok, center, 60)
    assert not blocked.accepted
    np.testing.assert_array_equal(blocked.center, center)


def test_gp_loss_is_finite_and_connected_to_trainable_scales():
    x = torch.tensor([[1.0, 0.0], [0.7, 1.0], [-1.0, 0.4]], dtype=torch.float64)
    dot = (x @ x.T).unsqueeze(0)
    norm = dot[0].diag()
    distance = norm[:, None] + norm - 2 * dot
    prior = torch.ones(1, dtype=torch.float64)
    model = SpectralGP(prior, prior, local=True)
    y = torch.tensor([0.4, -1.0, 0.9], dtype=torch.float64)
    quality = torch.tensor([0, 1, 2])
    loss = model(dot, distance, distance, y, quality, torch.ones_like(y), 0.08)
    loss.backward()
    assert torch.isfinite(loss)
    gradient = model.scales.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_snapshot_uses_one_frame_and_sonde_pairing_uses_only_past_scans(tmp_path):
    select = latest_snapshot({"valid": np.array([[True, False, True]]), "time": np.array([[-8.0, 2.0, -2.0]])})
    np.testing.assert_array_equal(select["valid"], [[False, False, True]])
    native = tmp_path / "DROP20220916T2359S1-LV2.4_pub.csv"
    native.write_text("sec, h:m:s, lat, lon, Press\n0,23:59:50,24,130,500\n"
                      "20,--:--:--,25,131,911.5\n21,00:00:11,25,131,-999\n")
    pressure = terminal_pressure(native)
    when = datetime(2022, 9, 17, 0, 0, 10, tzinfo=UTC)
    assert pressure["pressure_time_utc"] == when
    assert pressure["pressure_hpa"] == 911.5
    assert pressure["altitude_m"] is None
    before = when - timedelta(minutes=7)
    scans = [(before, 4, 0), (before, 5, 1), (when + timedelta(minutes=1), 5, 2)]
    assert [x for x in select_frames(scans, when, 20) if x is not None] == [(before, 4, 0)]
