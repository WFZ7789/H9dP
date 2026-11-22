"""Multi-output kriging with partial paired measurements and block PRESS by record."""

import numpy as np
import torch

from h9dp.r1 import _normalize
from h9dp.u1 import distances, matern


def view_bank(views, fixed, train, held, context=None):
    """Kernel scales and brightness knots use only this fold's training statistics."""
    parts = []
    names = []
    for name, values in views.items():
        a, b = _normalize(values[train], values[held])
        if not a.shape[1]:
            continue
        x = fixed.new_tensor(np.concatenate((a, b)))
        dot = x @ x.T / x.shape[1]
        parts.extend((dot, matern(distances(dot))))
        names.extend((name + ":linear", name + ":smooth"))
        if name in ("eye", "relief", "mixing", "column_recent", "overcast", "background"):
            knots = torch.linspace(-3, 3, 9, device=x.device, dtype=x.dtype)
            phi = torch.nn.functional.normalize((-0.5 * (x[:, :, None] - knots).square()).exp(), dim=-1)
            parts.append(phi.flatten(1) @ phi.flatten(1).T / x.shape[1])
            names.append(name + ":spline")
    if context is not None:
        if context.shape[1] not in (2, 4) or not np.isfinite(context).all():
            raise ValueError("Expected finite navigation, optionally with illumination")
        a, b = _normalize(context[train], context[held])
        g = fixed.new_tensor(np.concatenate((a, b)))
        response = (parts[names.index("mixing:smooth")] + parts[names.index("overcast:smooth")]) * 0.5
        # Only affine dependence on physical context; no location/date neighborhood matching.
        keys = ("latitude", "airmass", "daily_sun", "daylight")[:g.shape[1]]
        for i, key in enumerate(keys):
            linear = g[:, i, None] * g[:, i]
            parts.extend((linear, linear * response))
            names.extend((key, key + ":thermal_gain"))
    order = np.r_[train, held]
    bank = torch.cat((torch.stack(parts), fixed[:, order][:, :, order]))
    names += ["eye_recent_narrow", "eye_recent_wide", "eye_latest_narrow", "eye_latest_wide",
              "cdo_narrow", "cdo_wide", "scene_2", "scene_8", "scene_32"]
    scale = bank[:, :len(train), :len(train)].diagonal(dim1=1, dim2=2).mean(1).clamp_min(0.05)
    return bank / scale[:, None, None], names


def task_covariance(log_var, adjustments, base):
    """A PSD task covariance per view (or per quality for measurement noise)."""
    factor = torch.nn.functional.normalize(base + 0.6 * adjustments.tril(-1).tanh(), dim=-1)
    factor = factor * (0.5 * log_var).exp()[:, :, None]
    return factor @ factor.transpose(1, 2)


def system(prior, noise, bank, record, task, quality):
    gram = torch.einsum("gab,gij->abij", prior, bank)
    signal = gram[task[:, None], task, record[:, None], record]
    shared_record = record[:, None] == record
    error = noise[quality[record, None], task[:, None], task] * shared_record
    jitter = 0.002 * torch.eye(len(record), dtype=bank.dtype, device=bank.device)
    return signal + error + jitter


def restricted_solve(cov, y, design):
    """Separate unpenalized intercepts; P and P y yield exact deletion at fixed covariance."""
    chol, _ = torch.linalg.cholesky_ex(cov)
    inverse = torch.cholesky_inverse(chol)
    weighted = inverse @ design
    normal = design.T @ weighted
    mean_operator = torch.linalg.solve(normal, weighted.T)
    mean = mean_operator @ y
    p = inverse - weighted @ mean_operator
    alpha = inverse @ (y - design @ mean)
    return p, alpha, mean, chol


def block_errors(precision, alpha, blocks):
    """Each row lists all jointly observed target indices for one paired record."""
    pivots = precision[blocks[:, :, None], blocks[:, None, :]]
    return torch.linalg.solve(pivots, alpha[blocks].unsqueeze(-1)).squeeze(-1)


def joint_loss(log_mass, corr, log_noise, ncorr, bank, y, record, task, q, design, blocks,
               base, mass0, noise0):
    cov = system(task_covariance(log_mass, corr, base), task_covariance(log_noise, ncorr, base),
                 bank, record, task, q)
    precision, alpha, _, chol = restricted_solve(cov, y, design)
    errors = alpha[:len(q)] / precision.diag()[:len(q)]
    if design.shape[1] > 1:
        removed = block_errors(precision, alpha, blocks)
        errors = errors.scatter(0, blocks[:, 0], removed[:, 0])
        aux_loss = 0.06 * removed[:, 1:].square().mean()
    else:
        aux_loss = 0.
    evidence = (y @ alpha + 2 * chol.diag().log().sum()) / len(y)
    shrink = (0.012 * (log_mass - mass0).square().mean() + 0.014 * (log_noise - noise0).square().mean()
              + 0.02 * corr.square().mean() + 0.02 * ncorr.square().mean())
    return errors.square().mean() + aux_loss + 0.14 * evidence + shrink


def predict(log_mass, corr, log_noise, ncorr, bank, y, record, task, q, design, base):
    n = len(q)
    prior = task_covariance(log_mass, corr, base)
    cov = system(prior, task_covariance(log_noise, ncorr, base), bank[:, :n, :n], record, task, q)
    _, alpha, mean, _ = restricted_solve(cov, y, design)
    cross = torch.einsum("gb,gij->bij", prior[:, 0, :], bank[:, n:, :n])
    return mean[0] + cross[task, :, record].T @ alpha, alpha, mean
