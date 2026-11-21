"""Multi-output recursive feature machine for native D1 cloud distributions."""

import numpy as np
import torch

from h9dp.r1 import _normalize


def coordinates(group, train, held, rank=56):
    """Fold-local PCA of recent, latest and change descriptors on one physical region."""
    parts = []
    for i, weight in enumerate((0.55, 0.32, 0.13)):
        a, b = _normalize(group[train, i].astype(float), group[held, i].astype(float))
        if a.shape[1]:
            parts.append(np.concatenate((a, b)) * np.sqrt(weight / a.shape[1]))
    if not parts:
        return np.zeros((len(train) + len(held), rank))
    x = np.column_stack(parts)
    x -= x[: len(train)].mean(axis=0)
    _, _, axes = np.linalg.svd(x[: len(train)], full_matrices=False)
    reduced = x @ axes[:rank].T
    return np.pad(reduced, ((0, 0), (0, rank - reduced.shape[1])))


def kernel_parts(x, z, metric):
    xm = x @ metric
    dot = xm @ z.transpose(1, 2)
    dist = ((xm * x).sum(-1)[:, :, None] + ((z @ metric) * z).sum(-1)[:, None, :] - 2 * dot).clamp_min(0)
    # Smooth regional and joint Matérn 3/2 responses plus a linear extrapolation term.
    radial = (3 * dist + 1e-8).sqrt()
    joint = (3 * dist.mean(0) + 1e-8).sqrt()
    kernel = ((1 + radial) * (-radial).exp() + 0.24 * dot).mean(0)
    kernel = kernel + 0.8 * (1 + joint) * (-joint).exp() + 8
    return kernel, radial, joint


def solve_stage(x, y, variance, metric):
    kernel, radial, joint = kernel_parts(x, x, metric)
    dual = torch.cholesky_solve(y, torch.linalg.cholesky(kernel + torch.diag(variance)))
    # Derivatives of every task with respect to each region's original PCA coordinates.
    # The common factor 1 / region_count cancels in trace normalization.
    slope = -3 * ((-radial).exp() + 0.8 * (-joint).exp()[None])
    xm = x @ metric
    slopes_dual = slope[:, :, :, None] * dual[None, None]
    gradients = (
        slopes_dual.sum(2)[..., None] * xm[:, :, None]
        - torch.einsum("gijt,gjd->gitd", slopes_dual, xm)
        + 0.24 * torch.einsum("jt,gjd->gtd", dual, xm)[:, None]
    )
    return dual, gradients


def update_metric(x, gradients, importance, task_weight):
    weighted = gradients * importance[None, :, None, None].sqrt() * task_weight[None, None, :, None].sqrt()
    g = weighted.flatten(1, 2)
    covariance = g.transpose(1, 2) @ g
    eig, vec = torch.linalg.eigh(covariance)
    learned = (vec * eig.clamp_min(1e-12).sqrt()[:, None, :]) @ vec.transpose(1, 2)
    # Match the mean pairwise squared distance of the original training inputs.
    scale = x.square().sum((1, 2)) / ((x @ learned) * x).sum((1, 2)).clamp_min(1e-14)
    return 0.42 * torch.eye(x.size(2), dtype=x.dtype, device=x.device) + 0.58 * learned * scale[:, None, None]
