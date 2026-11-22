"""Rotation-marginalized pixel kernels and fold-local cloud diagnostics."""

from itertools import product

import torch


def pixel_mean(pixels, times, *, latest=False):
    """Average valid measurements in native coordinates and retain their support."""
    valid = pixels[:, :, 4:5] > 0
    available = valid.flatten(2).any(2) & torch.isfinite(times)
    offsets = torch.where(available, times, -torch.inf)
    last = offsets.amax(dim=1, keepdim=True)
    weight = (offsets == last).to(torch.float32) if latest else ((offsets - last) / 45).exp()
    weight = torch.where(available, weight, 0)
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1)
    spec = torch.where(valid, pixels[:, :, :4].to(torch.float32) / 2500, 0)
    support = (valid * weight[:, :, None, None, None]).sum(dim=1)
    value = (spec * weight[:, :, None, None, None]).sum(dim=1) / support.clamp_min(1e-8)
    return torch.cat((value * support.sqrt(), support * 0.4), dim=1)


def rotation_kernel(maps, widths, *, polar=False):
    """Average a radial basis kernel over a finite rotation/reflection group.

    All transforms preserve the coordinate metric. Its group average is the
    inner product of group-averaged feature maps, so the result is PSD.
    """
    rows = len(maps)
    flat = maps.reshape(rows, -1)
    norm = flat.square().mean(dim=1)
    kernels = maps.new_zeros((len(widths), rows, rows))
    turns = range(8 if polar else 4)
    scales = maps.new_tensor(widths)[:, None, None]
    for turn, reflect in product(turns, (False, True)):
        if polar:
            reflected = maps.flip(-2) if reflect else maps
            copy = reflected.roll(turn * maps.shape[-2] // 8, dims=-2)
        else:
            reflected = maps.flip(-1) if reflect else maps
            copy = reflected.rot90(turn, dims=(-2, -1))
        dot = flat @ copy.reshape(rows, -1).T / flat.shape[1]
        distance = (norm[:, None] + norm[None, :] - 2 * dot).clamp_min(0)
        kernels += (-distance / (2 * scales)).exp()
    kernels /= 2 * len(turns)
    return (kernels + kernels.transpose(-1, -2)).double() / 2


def pixel_kernels(eye, cloud, times):
    """Use every eye pixel at two time scales and circularly align the CDO."""
    axis = torch.arange(eye.shape[-1], device=eye.device) - eye.shape[-1] // 2
    distance = axis.square()[:, None] + axis.square()
    eye_weight = (0.20 + (-distance / (2 * 10**2)).exp()).sqrt()
    eye_recent = rotation_kernel(pixel_mean(eye, times) * eye_weight, (0.035, 0.16))
    eye_latest = rotation_kernel(pixel_mean(eye, times, latest=True) * eye_weight, (0.035, 0.16))
    cloud_recent = rotation_kernel(pixel_mean(cloud, times), (0.025, 0.12), polar=True)
    return torch.cat((eye_recent, eye_latest, cloud_recent))


def auxiliary(kernel, target, noise):
    """Fit a centered kernel ridge posterior and predict train rows by exact LOO."""
    n = len(target)
    center = target.mean()
    train = kernel[:n, :n] + torch.diag(noise.expand(n))
    chol = torch.linalg.cholesky(train)
    inverse = torch.cholesky_inverse(chol)
    dual = inverse @ (target - center)
    loo = target - dual / inverse.diag()
    cross = kernel[n:, :n]
    held = center + cross @ dual
    lower = torch.linalg.solve_triangular(chol, cross.T, upper=False)
    pred_variance = kernel.diag()[n:] + noise - lower.square().sum(dim=0)
    return torch.cat((loo, held)), torch.cat((inverse.diag().reciprocal(), pred_variance.clamp_min(1e-8)))


def observation_noise(noise_coefficients, conditions):
    return (conditions @ noise_coefficients).clamp(-7, 1).exp() + 0.003


def objective(log_mass, noise_coefficients, basis, conditions, target, prior, prior_noise):
    """Optimize equally weighted leave-one-out pressure errors and a variance prior."""
    kernel = torch.einsum("g,gij->ij", log_mass.exp(), basis) + 8
    covariance = kernel + torch.diag(observation_noise(noise_coefficients, conditions))
    chol, _ = torch.linalg.cholesky_ex(covariance)
    inverse = torch.cholesky_inverse(chol)
    dual = inverse @ target
    rmse2 = (dual / inverse.diag()).square().mean()
    evidence = (target @ dual + 2 * chol.diag().log().sum()) / len(target)
    penalty = (log_mass - prior).square().mean() + (noise_coefficients - prior_noise).square().mean()
    return rmse2 + 0.12 * evidence + 0.008 * penalty
