import numpy as np
import torch

from h9dp.e1 import isotherms
from h9dp.e1_joint import condition_aux, pressure_loo


def test_joint_leave_one_out_removes_all_measurements_of_query():
    generator = torch.Generator().manual_seed(1)
    n, tasks = 7, 4
    x = torch.randn(3, n, 5, generator=generator, dtype=torch.float64)
    kernel = x @ x.transpose(1, 2) / 5
    y = torch.randn(n, generator=generator, dtype=x.dtype)
    z = torch.randn(tasks, n, generator=generator, dtype=x.dtype)
    q = torch.arange(n) % 3
    mass = torch.randn(tasks, 3, generator=generator, dtype=x.dtype)
    noise = torch.randn(tasks, 3, generator=generator, dtype=x.dtype)
    pmass, pnoise = mass[0], noise[0]
    beta = x.new_tensor([-0.8, 0.3, 0.55, 0.42])
    smooth, signal, residual, diag, dual = condition_aux(kernel, mass, noise, z, q)
    loo, _ = pressure_loo(pmass, pnoise, beta, kernel, smooth, signal, residual, diag, dual, y, q)
    for i in range(n):
        train = torch.arange(n)[torch.arange(n) != i]
        order = torch.cat((train, torch.tensor([i])))
        b = kernel[:, order][:, :, order]
        _, s, r, _, _ = condition_aux(b, mass, noise, z[:, train], q[train])
        cov = (
            torch.einsum("g,gij->ij", pmass.exp(), b[:, :, :-1])
            + 8
            + torch.einsum("a,aij->ij", beta.square(), r)
        )
        mu = beta @ s
        coef = torch.linalg.solve(cov[:-1] + torch.diag(pnoise.exp()[q[train]] + 0.002), y[train] - mu[:-1])
        torch.testing.assert_close(loo[i], y[i] - mu[-1] - cov[-1] @ coef)


def test_warm_isotherms_shrink_for_higher_levels():
    dy, dx = np.mgrid[-64:65, -64:65]
    warm = 200 + 20 * np.exp(-(dy**2 + dx**2) / 80)
    image = np.repeat(np.rint((warm[None] - 150) * 100).astype(np.int16), 4, axis=0)
    result = isotherms(image, np.ones_like(warm, dtype=bool), [64.5, 64.5])
    size = result["shape"].reshape(4, 4, 7)[:, :, 0]
    assert np.all(np.diff(size, axis=1) < 0)
