"""Marginalize latent eye clarity when predicting pressure from imagery."""

import torch

from h9dp.analog_residual import double_deletion_residual, transfer_weights


def class_posterior(kernel, labels, regularizer=0.095):
    n = len(labels)
    chol = torch.linalg.cholesky(kernel[:n, :n] + regularizer * torch.eye(n, dtype=kernel.dtype, device=kernel.device))
    inverse = torch.cholesky_inverse(chol)
    center = labels.mean(dim=0)
    residual = labels - center
    dual = inverse @ residual
    diag = inverse.diag()[:, None]
    # Account for removing the sample from both the solve and the class mean.
    loo = labels - (dual + inverse.sum(1)[:, None] * residual / (n - 1)) / diag
    test = center + kernel[n:, :n] @ dual
    probability = torch.cat((loo, test)).clamp_min(0.025)
    return probability / probability.sum(dim=1, keepdim=True)


def class_covariance(clarity_settings):
    gradient = clarity_settings.new_tensor([1., 0., -1.])
    return torch.diag(clarity_settings[:3].exp()) + clarity_settings[3].exp() * torch.outer(gradient, gradient)


def relaxed_loo(cross_covariance, inverse, dual):
    """Exact deletion from the fitted covariance with a distinct query covariance."""
    leverage = (cross_covariance * inverse.T).sum(dim=1)
    return cross_covariance @ dual - leverage * dual / inverse.diag()


def pressure_model(log_mass, clarity_settings, log_noise, basis, regime, labels, probabilities, target,
                   variance, prior, prior_clarity, prior_noise, analog=None, transfer=None):
    shared = torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8
    conditional = class_covariance(clarity_settings) @ labels.T
    covariance = shared + regime * (labels @ conditional)
    noise = (labels @ log_noise[:3] + log_noise[3] * variance).clamp(-7, 1).exp() + 0.003
    chol, _ = torch.linalg.cholesky_ex(covariance + torch.diag(noise))
    inverse = torch.cholesky_inverse(chol)
    dual = inverse @ target
    relaxed = shared + regime * (probabilities @ conditional)
    if analog is None:
        predicted = relaxed_loo(relaxed, inverse, dual)
    else:
        base = relaxed @ dual
        influence = relaxed @ inverse
        predicted = base - influence.diag() * dual / inverse.diag()
        residuals = double_deletion_residual(influence, inverse, dual, base, target)
        predicted = predicted + (transfer_weights(analog, transfer, leave_out=True) * residuals).sum(dim=1)
    error = predicted - target
    evidence = (target @ dual + 2 * chol.diag().log().sum()) / len(target)
    penalty = ((log_mass - prior).square().mean() + (clarity_settings - prior_clarity).square().mean()
               + (log_noise - prior_noise).square().mean())
    if transfer is not None:
        penalty = penalty + 0.5 * transfer.square().mean()
    return error.square().mean() + 0.13 * evidence + 0.012 * penalty
