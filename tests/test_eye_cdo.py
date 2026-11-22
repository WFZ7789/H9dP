import numpy as np
import torch

from h9dp.eye_cdo import EyeCDO, crop, grids
from h9dp.eye_sampling import HALF, WIDTHS, fit_sequence


def test_eye_crop_retains_boundary_pixels_and_their_validity():
    image = np.full((4, 61, 61), 9000, dtype=np.int16)
    image[:, 6, 54] = 10200
    mask = np.ones((61, 61), dtype=bool)
    mask[54, 6] = False
    eye, _ = grids()
    result = crop(image, mask, (30, 30), eye)
    assert result.shape == (5, 49, 49)
    np.testing.assert_array_equal(result[:4, 0, -1], 1200)
    assert result[4, 0, -1] == 1
    np.testing.assert_array_equal(result[:, -1, 0], 0)


def test_shared_eye_width_and_radiometry_for_pixel_integrated_gaussians():
    from scipy.special import erf

    px = np.arange(-HALF, HALF + 1)
    width = WIDTHS[3]
    patches = []
    for phase in (-0.75, 0., 0.75):
        y = width * np.sqrt(np.pi / 2) * (
            erf((px + 0.5 - phase) / (np.sqrt(2) * width))
            - erf((px - 0.5 - phase) / (np.sqrt(2) * width))
        )
        image = np.outer(y, y).reshape(-1)
        patches.append(212 + np.arange(12, 16)[:, None] * image)
    radiometry, fits = fit_sequence(np.array(patches, dtype=np.float32), np.array([-20, -10, 0]))
    np.testing.assert_allclose(fits[:, 0], width)
    np.testing.assert_allclose(radiometry[:, :4], [[224,225,226,227]] * 3, atol=1e-3)


def test_eye_and_cloud_branch_have_gradients_from_raw_pixels():
    gen = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    eye = torch.randn(2, 6, 5, 49, 49, generator=gen, requires_grad=True)
    cdo = torch.randn(2, 6, 5, 48, 48, generator=gen, requires_grad=True)
    model = EyeCDO(1, 1)
    features = torch.eye(2, dtype=torch.float64)[None]
    weights = torch.full((2, 2, 6), 1 / 6)
    pressure, temperature, uncertainty = model(eye, cdo, weights, features, features)
    assert pressure.shape == temperature.shape == (2, 2)
    assert (uncertainty > 0).all()
    eye_grad, cdo_grad = torch.autograd.grad(pressure[0, 1] + temperature[0, 1] + uncertainty[0], (eye, cdo))
    assert eye_grad[0].abs().sum() > 0
    assert cdo_grad[0].abs().sum() > 0
