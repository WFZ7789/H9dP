import numpy as np
import pytest
import torch

from h9dp.u1 import ordinary_weights, press_pool
from h9dp.u1_features import (
    mixing_profile,
    project_view,
    relative_radiance,
    scene_kernels,
)
from scripts.u1 import load_features


def test_mixing_uses_colocated_warm_pixels_and_excludes_masked_counts():
    y, x = np.mgrid[-24:25, -24:25]
    radius = np.hypot(y, x)
    patch = np.zeros((5, 49, 49))
    patch[:4] = np.array([3, 5, 7, 9])[:, None, None] * x + 800
    patch[4] = 1
    patch[4, 24, 28] = 0
    profile = mixing_profile(patch, radius).reshape(3, 17)
    assert (profile[:, 4:8] > profile[:, :4]).all()
    patch[:4, 24, 28] = 32000
    np.testing.assert_array_equal(mixing_profile(patch, radius), profile.ravel())
    # Reverse only the WV gradient: independent marginal quantiles wouldn't notice.
    patch[0] = 800 - 3 * x
    flipped = mixing_profile(patch, radius).reshape(3, 17)
    assert (flipped[:, 8] < 0).all() and (profile[:, 8] > 0).all()
    np.testing.assert_allclose(relative_radiance(np.full((4, 1), 240)), 1)
    patch[4] = 0
    assert np.isnan(mixing_profile(patch, radius)).all()


def test_training_projection_does_not_change_with_other_held_rows():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(12, 5))
    raw[:8, 2] = np.nan
    train, held = np.arange(8), np.arange(8, 12)
    base = project_view(raw, train, held, 6)
    raw[11] = 1e6
    modified = project_view(raw, train, held, 6)
    np.testing.assert_array_equal(base[:-1], modified[:-1])
    assert np.isfinite(modified).all() and not modified[:, 4:].any()


def test_press_has_an_unpenalized_intercept_and_matches_explicit_deletion():
    x = torch.arange(6, dtype=torch.float64) / 3
    k = (-(x[:, None] - x).square()).exp()
    cov = k + torch.diag(0.1 + 0.04 * x)
    target = torch.stack((3 + x.sin(), 9 - x.square()), dim=1)
    _, dual, diag = ordinary_weights(torch.linalg.cholesky(cov), target)
    loo = target - dual / diag[:, None]
    for i in range(len(x)):
        keep = torch.arange(len(x)) != i
        mean, reduced, _ = ordinary_weights(torch.linalg.cholesky(cov[keep][:, keep]), target[keep])
        torch.testing.assert_close(loo[i], (k[i, keep] @ reduced + mean).flatten())
    pred, weights, *_ = press_pool(k[None, :4, :4], k[None, 4:, :4], target[:4])
    pred2, weights2, *_ = press_pool(k[None, :4, :4], k[None, 4:5, :4], target[:4] + 15)
    torch.testing.assert_close(pred2, pred[:1] + 15)
    torch.testing.assert_close(weights2, weights)
    torch.testing.assert_close(weights.sum(0), torch.ones_like(weights[0]))


def test_scene_similarity_is_psd_with_missing_scans():
    rng = np.random.default_rng(1)
    maps = rng.normal(size=(5, 3, 4, 20, 20)).astype(np.float32)
    maps[1, 1:] = np.nan
    times = np.tile([-90., -20., 0.], (5, 1))
    grams = scene_kernels(maps, times, "cpu")
    torch.testing.assert_close(grams, grams.transpose(-1, -2), atol=1e-6, rtol=1e-6)
    assert torch.linalg.eigvalsh(grams).min() > -1e-6
    torch.testing.assert_close(grams.diagonal(dim1=1, dim2=2), torch.ones_like(grams[:, 0]), atol=1e-6, rtol=1e-6)


def test_feature_provenance_rejects_different_time_order_or_catalog(tmp_path):
    ids = np.array([4, 7])
    times = np.array([[-5., 0.], [np.nan, -2.]])
    path = tmp_path / "ground.npz"
    np.savez(path, source_sha256="source", catalog_sha256="catalog", mode="test", record=ids, time=times, g=times)
    values = load_features(path, "test", ("g",), ids, times, "source", "catalog")
    np.testing.assert_array_equal(values["g"], times)
    for rid, stamp, digest in ((ids[::-1], times, "catalog"), (ids, times[:, ::-1], "catalog"), (ids, times, "bad")):
        with pytest.raises(ValueError, match="Mismatched"):
            load_features(path, "test", ("g",), rid, stamp, "source", digest)
