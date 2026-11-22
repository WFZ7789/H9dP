"""U1: term-wise thermal retrieval and direct multiple-view pressure regression."""

import torch


def matern(distance2):
    r = (3 * distance2.clamp_min(0) + 1e-12).sqrt()
    return (1 + r) * (-r).exp()


def distances(dot):
    diag = dot.diagonal(dim1=-2, dim2=-1)
    return diag[..., :, None] + diag[..., None, :] - 2 * dot


def expert_bank(x, fixed):
    """A common background plus one specialist view; all matrices are PSD."""
    dot = x @ x.transpose(1, 2)
    views = torch.cat((0.45 * dot + matern(distances(dot)), fixed))
    common = views.mean(0, keepdim=True)
    return torch.cat((common, 0.35 * common + 0.65 * views))


def pressure_kernel(x, fixed, log_mass, metric):
    scaled = x * metric.tanh().exp()[:, None, :]
    dot = scaled @ scaled.transpose(1, 2)
    d = distances(dot)
    basis = torch.cat((dot, matern(d), matern(d.mean(0))[None], fixed))
    return torch.einsum("g,gij->ij", log_mass.exp(), basis)


def ordinary_weights(chol, targets):
    """Ordinary kriging with an unpenalized constant mean (also for batched kernels)."""
    inv = torch.cholesky_inverse(chol)
    one = inv.sum(dim=-1, keepdim=True)
    total = one.sum(dim=-2, keepdim=True)
    mean = (one.transpose(-1, -2) @ targets) / total
    dual = inv @ (targets - mean)
    # Removing the fitted intercept is essential for exact LOO, even for a flat field.
    precision_diag = inv.diagonal(dim1=-2, dim2=-1) - one.squeeze(-1).square() / total.squeeze(-1)
    return mean, dual, precision_diag


def pressure_loss(log_mass, metric, log_noise, x, fixed, target, quality, prior_mass, prior_noise):
    kernel = pressure_kernel(x, fixed, log_mass, metric)
    noise = log_noise[quality].clamp(-7, 2).exp() + 0.003
    chol, _ = torch.linalg.cholesky_ex(kernel + torch.diag(noise))
    _, dual, precision = ordinary_weights(chol, target[:, None])
    press = (dual.flatten() / precision).square().mean()
    evidence = (target @ dual.flatten() + 2 * chol.diag().log().sum()) / len(target)
    penalty = (0.012 * (log_mass - prior_mass).square().mean() + 0.025 * metric.square().mean()
               + 0.01 * (log_noise - prior_noise).square().mean())
    return press + 0.14 * evidence + penalty


def pressure_predict(kernel, y, quality, log_noise):
    n = len(y)
    noise = log_noise[quality].clamp(-7, 2).exp() + 0.003
    chol = torch.linalg.cholesky(kernel[:n, :n] + torch.diag(noise))
    mean, dual, _ = ordinary_weights(chol, y[:, None])
    return (mean + kernel[n:, :n] @ dual).flatten(), mean, dual


def calibrate_terms(terms, pressure):
    """Regularized M11 readout; terms already contain the sample's CDO multiplier."""
    center = terms.mean(0)
    scale = terms.std(0, unbiased=False).clamp_min(1e-6)
    z = (terms - center) / scale
    penalty = torch.eye(z.shape[1], dtype=z.dtype, device=z.device) * (len(z) * 0.002)
    beta = torch.linalg.solve(z.T @ z + penalty, z.T @ (pressure - pressure.mean())) / scale
    return pressure.mean() - center @ beta, beta


def press_pool(train_kernels, cross_kernels, targets):
    """Multi-target ordinary kriging, with PRESS mixing per task on training rows only.

    `cross_kernels` addresses other rows against this same support. It is not
    used in the solves, the LOO errors or the specialist weights.
    """
    n = len(targets)
    scale = train_kernels.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-6)
    train = train_kernels / scale[:, None, None]
    cross = cross_kernels / scale[:, None, None]
    eye = torch.eye(n, dtype=train.dtype, device=train.device)
    cov = torch.cat((train + 0.07 * eye, train + 0.24 * eye))
    chol = torch.linalg.cholesky(cov)
    mean, dual, precision = ordinary_weights(chol, targets)
    loo_error = dual / precision[:, :, None]
    mse = loo_error.square().mean(1).clamp_min(1e-12)
    weight = (-16 * mse.log()).softmax(dim=0)
    prediction = (torch.cat((cross, cross)) @ dual + mean) * weight[:, None, :]
    return prediction.sum(0), weight, mean, dual
