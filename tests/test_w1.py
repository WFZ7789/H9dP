import numpy as np
import torch

from h9dp.w1 import latent_loo, quality_covariance, warm_core


def test_warm_core_recovers_compact_eye_offset():
    y, x = np.mgrid[:101, :101]
    warm = 216 + 62 * np.exp(-((y - 47) ** 2 + (x - 55) ** 2) / 8)
    image = np.broadcast_to(np.rint((warm - 150) * 100).astype(np.int16), (4, 101, 101))
    center, feature = warm_core(image, np.ones((101, 101), dtype=bool), (50.5, 50.5))
    assert center == (47, 55)
    assert feature.shape == (56,)
    assert np.isfinite(feature).all()
    _, missing = warm_core(image, np.zeros((101, 101), dtype=bool), (50.5, 50.5))
    assert np.isnan(missing).all()


def test_latent_loo_excludes_query_label_and_uses_shared_covariance():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(7, generator=generator, dtype=x.dtype)
    common = x @ x.T + 1
    nuisance = x[:, :2] @ x[:, :2].T
    q = torch.arange(7) % 3
    settings = x.new_tensor([-2.5, -1.7, -1., -0.65, -1.2])
    fit = quality_covariance(common, nuisance, settings, q)
    inverse = torch.linalg.inv(fit)
    estimates = latent_loo(common, inverse, inverse @ y)
    for i in range(len(x)):
        train = torch.arange(len(x)) != i
        dual = torch.linalg.solve(fit[train][:, train], y[train])
        torch.testing.assert_close(estimates[i], common[i, train] @ dual)
