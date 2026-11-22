import numpy as np
import torch

from h9dp.analog_residual import double_deletion_residual
from h9dp.clarity_mixture import class_posterior
from h9dp.eye_cdo_kernel import pixel_mean, rotation_kernel
from h9dp.ground_eye import pixel_jacobian
from h9dp.hsd import read_header
from test_hsd import synthetic_hsd


def test_eye_rotation_kernel_is_psd_and_individually_invariant():
    grid = torch.arange(150, dtype=torch.float64).reshape(3, 2, 5, 5).sin()
    kernel = rotation_kernel(grid, (0.08, 0.22))
    transformed = grid.clone()
    transformed[0] = grid[0].rot90(1, dims=(-2, -1)).flip(-1)
    torch.testing.assert_close(rotation_kernel(transformed, (0.08, 0.22)), kernel)
    assert torch.linalg.eigvalsh(kernel).min() >= -1e-12


def test_pixel_mean_retains_mask_and_respects_latest_time():
    pixels = torch.zeros(2, 3, 5, 2, 2, dtype=torch.int16)
    pixels[0, 0, :4] = 2500
    pixels[0, 0, 4] = 1
    pixels[0, 1, :4, 0, 0] = 5000
    pixels[0, 1, 4, 0, 0] = 1
    times = torch.tensor([[-45., 0., 15.], [-10., -5., 0.]])
    result = pixel_mean(pixels, times, latest=True)
    assert result[0, 0, 0, 0].item() == 2
    assert result[0, 0, 0, 1].item() == 0
    assert torch.count_nonzero(result[1]) == 0
    assert torch.isfinite(pixel_mean(pixels, times)).all()


def test_ground_scale_increases_away_from_nadir(tmp_path):
    header = read_header(synthetic_hsd(tmp_path / "reference.DAT", "<"))
    center = pixel_jacobian(header, 0., 140.7)
    np.testing.assert_allclose(center, np.eye(2) / 2, atol=2e-6)
    edge = pixel_jacobian(header, 34., 122.)
    assert np.linalg.det(edge) > 0
    assert np.linalg.det(edge) < np.linalg.det(center)


def test_calibrated_clarity_omits_its_own_training_label():
    x = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float64)
    kernel = (-(x[:, None] - x).square()).exp()
    labels = torch.eye(3, dtype=kernel.dtype)[[0, 1, 1, 2]]
    estimate = class_posterior(kernel, labels)
    changed = labels.clone()
    changed[1] = labels[0]
    torch.testing.assert_close(class_posterior(kernel, changed)[1], estimate[1])
    torch.testing.assert_close(estimate.sum(dim=1), torch.ones(5, dtype=kernel.dtype))


def test_analog_residual_matches_two_explicit_deletions():
    x = torch.arange(5, dtype=torch.float64) / 2
    base = (-(x[:, None] - x).square()).exp()
    covariance = base + torch.diag(0.5 + 0.09 * x)
    cross = base * (0.9 + 0.03 * x[:, None])
    target = x.sin()
    inverse = torch.linalg.inv(covariance)
    dual = inverse @ target
    result = double_deletion_residual(cross @ inverse, inverse, dual, cross @ dual, target)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                assert result[i, j].item() == 0
            else:
                kept = [k for k in range(len(x)) if k not in (i, j)]
                reduced = torch.linalg.solve(covariance[kept][:, kept], target[kept])
                torch.testing.assert_close(result[i, j], target[j] - cross[j, kept] @ reduced)
