"""Condition a pressure process on training-only MODIS radiometry and eye geometry."""

import torch


def aux_loss(log_mass, log_noise, basis, z, q, weight, prior):
    k = torch.einsum("ag,gij->aij", log_mass.exp(), basis) + 8
    chol, _ = torch.linalg.cholesky_ex(k + torch.diag_embed(log_noise.exp()[:, q] + 0.004))
    inv = torch.cholesky_inverse(chol)
    dual = (inv @ z[..., None]).squeeze(-1)
    loo = dual / inv.diagonal(dim1=-1, dim2=-2)
    evidence = ((z * dual).sum(-1) + 2 * chol.diagonal(dim1=-1, dim2=-2).log().sum(-1)) / z.size(1)
    return (
        (loo.square() * weight).mean() + 0.13 * evidence.mean() + 0.003 * (log_mass - prior).square().mean()
    )


def condition_aux(basis, mass, noise, z, q):
    n = z.shape[1]
    k = torch.einsum("ag,gij->aij", mass.exp(), basis) + 8
    d = noise.exp()[:, q] + 0.004
    chol = torch.linalg.cholesky(k[:, :n, :n] + torch.diag_embed(d))
    inv = torch.cholesky_inverse(chol)
    # Posterior mean and posterior cross-covariance against training locations.
    smoother = k[:, :, :n] @ inv
    signal = (smoother @ z[..., None]).squeeze(-1)
    residual = smoother * d[:, None, :]
    dual = (inv @ z[..., None]).squeeze(-1)
    return smoother[:, :n], signal, residual, inv.diagonal(dim1=-1, dim2=-2), dual


def pressure_loo(log_mass, log_noise, beta, basis, smooth, signal, residual, diag_inv, zdual, y, q):
    cov = (
        torch.einsum("g,gij->ij", log_mass.exp(), basis)
        + 8
        + torch.einsum("a,aij->ij", beta.square(), residual)
    )
    chol, _ = torch.linalg.cholesky_ex(cov + torch.diag(log_noise.exp()[q] + 0.002))
    inv = torch.cholesky_inverse(chol)
    centered = y - beta @ signal
    dual = inv @ centered
    # Each block removes pressure AND all four auxiliary measurements at this record.
    h = beta[:, None, None] * smooth
    left = inv @ h
    cross = -left.diagonal(dim1=-1, dim2=-2).T
    zz = torch.einsum("aji,bji->iab", h, left) + torch.diag_embed(diag_inv.T)
    small = torch.cat(
        (torch.cat((inv.diag()[:, None], cross), 1)[:, None, :], torch.cat((cross[:, :, None], zz), 2)), 1
    )
    dz = zdual - (h.transpose(1, 2) @ dual)
    joint = torch.cat((dual[:, None], dz.T), 1)
    errors = torch.linalg.solve(small, joint.unsqueeze(-1)).squeeze(-1)
    evidence = (centered @ dual + 2 * chol.diag().log().sum()) / len(y)
    return errors[:, 0], evidence


def pressure_loss(mass, noise, beta, basis, smooth, signal, residual, diag_inv, zdual, y, q, weight, prior):
    residuals, evidence = pressure_loo(
        mass, noise, beta, basis, smooth, signal, residual, diag_inv, zdual, y, q
    )
    return (
        (residuals.square() * weight).mean()
        + 0.10 * evidence
        + 0.003 * (mass - prior).square().mean()
        + 0.003 * beta.square().mean()
    )
