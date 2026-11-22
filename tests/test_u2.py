from datetime import UTC, datetime

import numpy as np
import torch

from h9dp.u2 import block_errors, restricted_solve, system, task_covariance
from h9dp.u2_geometry import column_profile, geometry

PROJECTION = (140.7, 1., 1., 0., 0., 42164., 6378.137, 6356.7523)


def test_line_of_sight_and_solar_covariates_have_physical_directions():
    day = datetime(2025, 3, 22, 3, tzinfo=UTC)
    ray, values = geometry(0., 140.7, day, PROJECTION)
    np.testing.assert_allclose(ray, 0, atol=1e-12)
    np.testing.assert_allclose(values[1], 1)
    assert values[-1] > 0.95
    _, night = geometry(0., 140.7, day.replace(hour=15), PROJECTION)
    assert night[-1] == 0 and night[-2] == values[-2]
    ray, values = geometry(26., 153., day, PROJECTION)
    assert ray[0] < 0 < ray[1] and values[1] > 1


def test_parallax_sampling_follows_emitter_but_b16_anchor_does_not_shift():
    y, x = np.mgrid[-24:25, -24:25]
    patch = np.ones((5, 49, 49))
    for b, dx in enumerate((5.5, 4., 2.3, 0.)):
        patch[b] = 2400 * np.exp(-((x - dx)**2 + y*y) / 9)
    z = column_profile(patch, np.array([0, 1.]))[:32].reshape(4, 2, 4)
    assert np.all(z[2, :, :3] > z[0, :, :3])
    np.testing.assert_allclose(z[:, :, 3], np.tile(z[0, :, 3], (4, 1)))
    patch[4] = 0
    assert np.isnan(column_profile(patch, np.array([0, 1.]))).all()


def test_block_press_removes_both_auxiliaries_and_refits_each_intercept():
    x = torch.arange(7, dtype=torch.float64) / 3
    g = torch.stack([(-(x[:, None] - x).square()).exp(), torch.outer(x, x)])
    pair = torch.tensor([0, 2, 3, 6])
    record = torch.cat((torch.arange(7), pair, pair))
    task = torch.cat((torch.zeros(7), torch.ones(4), torch.full((4,), 2))).long()
    h = torch.nn.functional.one_hot(task, 3).to(x.dtype)
    q = torch.arange(7) % 3
    base = torch.tensor([[1., 0., 0.], [0.6, 0.8, 0.], [0.45, -0.4, 0.7]], dtype=x.dtype)
    signal = task_covariance(x.new_zeros(2, 3), x.new_zeros(2, 3, 3), base)
    noise = task_covariance(x.new_full((3, 3), -2.), x.new_zeros(3, 3, 3), base)
    cov = system(signal, noise, g, record, task, q)
    assert torch.linalg.eigvalsh(cov).min() > 0
    y = x[record].sin() + (0.2 + x[record]) * task
    p, alpha, _, _ = restricted_solve(cov, y, h)
    block = torch.column_stack((pair, 7 + torch.arange(4), 11 + torch.arange(4)))
    errors = block_errors(p, alpha, block)
    for j, dropped in enumerate(block):
        keep = record != pair[j]
        _, dual, intercept, _ = restricted_solve(cov[keep][:, keep], y[keep], h[keep])
        reference = y[dropped] - h[dropped] @ intercept - cov[dropped][:, keep] @ dual
        torch.testing.assert_close(errors[j], reference)
    # The same-record measurement cross-noise must not be present for different rows.
    uncorrelated = torch.diag_embed(noise.diagonal(dim1=1, dim2=2))
    diff = cov - system(signal, uncorrelated, g, record, task, q)
    assert diff[0, 7] > 0 and diff[0, 8] == 0
