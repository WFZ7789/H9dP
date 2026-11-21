import numpy as np
import torch

from h9dp.d1_metric import coordinates, kernel_parts, solve_stage
from h9dp.v1 import ANGLES, RADII, angular_motion, polar_scan


def test_angular_motion_recovers_signed_shift_and_excludes_missing_rings():
    rng = np.random.default_rng(1)
    early = rng.normal(240, 4, (4, len(RADII), ANGLES))
    late = np.roll(early, -2, axis=-1)
    early[:, :4] = np.nan
    motion = angular_motion(early, late, 20).reshape(-1, 4)
    assert np.isnan(motion[0]).all()
    np.testing.assert_allclose(motion[4:, 0], -2 * 2 * np.pi / ANGLES * RADII[6:-1] / 20)
    np.testing.assert_allclose(motion[4:, 1], 1)


def test_invalid_polar_scan_stays_missing():
    mask = np.zeros((50, 50), dtype=bool)
    image = np.ones((4, 50, 50), dtype=np.int16)
    assert np.isnan(polar_scan(image, mask, (25, 25))).all()


def test_d1_pca_coordinates_independent_of_other_held_records():
    rng = np.random.default_rng(1)
    group = rng.normal(size=(12, 3, 6))
    train, held = np.arange(9), np.arange(9, 12)
    before = coordinates(group, train, held)
    group[-1] = -5000
    np.testing.assert_allclose(coordinates(group, train, held)[:-1], before[:-1])


def test_recursive_metric_gradients_match_posterior_derivatives():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 7, 4, generator=gen, dtype=torch.float64)
    metric = torch.eye(4, dtype=x.dtype).repeat(2, 1, 1)
    y = torch.randn(7, 2, generator=gen, dtype=x.dtype)
    dual, grad = solve_stage(x, y, x.new_full((7,), 0.1), metric)
    z = x.detach().clone().requires_grad_()
    prediction = kernel_parts(z, x, metric)[0] @ dual
    (automatic,) = torch.autograd.grad(prediction[1, 0], z)
    torch.testing.assert_close(grad[:, 1, 0] / 2, automatic[:, 1], atol=1e-8, rtol=1e-7)
