"""Fit a pressure process that averages over eye clarity inferred from the image."""

import time

import numpy as np
import torch

from h9dp.analog_residual import transfer_weights
from h9dp.clarity_mixture import class_covariance, class_posterior, pressure_model, relaxed_loo


def fit(basis, conditions, names, y, quality, analog=None):
    n = len(y)
    quality_number = {"Good": 0, "Fair": 1, "Poor": 2}
    q = torch.as_tensor([quality_number[tag] for tag in quality], device=basis.device)
    labels = torch.nn.functional.one_hot(q, 3).to(basis.dtype)
    center = float(y.mean())
    scale = max(float(y.std()), 1.)
    target = basis.new_tensor((y - center) / scale)
    eye_terms = ("eye_recent_linear", "relief_recent_linear", "eye_recent_narrow", "eye_latest_wide")
    cloud_terms = ("wall_recent_matern", "ring_recent_matern", "cloud_cover_recent_matern", "cdo_recent_wide")
    eye = basis[[names.index(term) for term in eye_terms]].mean(0)
    cloud = basis[[names.index(term) for term in cloud_terms]].mean(0)
    probabilities = class_posterior(0.72 * eye + 0.28 * cloud + 1, labels)
    regime = 0.48 * eye + 0.52 * cloud + 1

    prior = basis.new_tensor([np.log(0.06 if "recent" in name else 0.03) for name in names])
    prior_clarity = basis.new_tensor([-3.25, -3.25, -3.25, -3.25])
    prior_noise = basis.new_tensor([-2.5, -1.95, -1.65, 0.25])
    mass = torch.nn.Parameter(prior.clone())
    clarity = torch.nn.Parameter(prior_clarity.clone())
    noise = torch.nn.Parameter(prior_noise.clone())
    terms = (basis[:, :n, :n].contiguous(), regime[:n, :n].contiguous(),
             labels, probabilities[:n].contiguous(), target, conditions[:n, 2].contiguous(),
             prior, prior_clarity, prior_noise)
    transfer = torch.nn.Parameter(basis.new_zeros(2)) if analog is not None else None
    similarity = analog[:n, :n].contiguous() if analog is not None else None
    compiled = torch.compile(pressure_model, fullgraph=True, dynamic=False)
    parameters = [mass, clarity, noise] if transfer is None else [mass, clarity, noise, transfer]
    optimizer = torch.optim.Adam(parameters, lr=0.043)
    timings = []
    for _ in range(145):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = compiled(mass, clarity, noise, *terms, similarity, transfer)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic() - start)
    if not torch.isfinite(loss):
        raise ArithmeticError("Q1 training objective was nonfinite")

    with torch.no_grad():
        shared = torch.einsum("g,gij->ij", mass.exp(), basis[:, :, :n]) + 8
        conditional = class_covariance(clarity) @ labels.T
        train = shared[:n] + regime[:n, :n] * (labels @ conditional)
        observation = (labels @ noise[:3] + noise[3] * conditions[:n, 2]).clamp(-7, 1).exp() + 0.003
        chol = torch.linalg.cholesky(train + torch.diag(observation))
        dual = torch.cholesky_solve(target[:, None], chol).flatten()
        cross = shared[n:] + regime[n:, :n] * (probabilities[n:] @ conditional)
        forecast = cross @ dual
        if analog is not None:
            relaxed = shared[:n] + regime[:n, :n] * (probabilities[:n] @ conditional)
            residual = target - relaxed_loo(relaxed, torch.cholesky_inverse(chol), dual)
            forecast = forecast + transfer_weights(analog[n:, :n], transfer) @ residual
        forecast = forecast * scale + center
    state = {
        "names": names, "mass": mass.detach().cpu(), "noise": noise.detach().cpu(),
        "clarity": clarity.detach().cpu(), "fit_class": q.cpu(), "posterior_class": probabilities[n:].cpu(),
        "target_mean": center, "target_scale": scale, "dual": dual.cpu(),
    }
    if transfer is not None:
        state["transfer"] = transfer.detach().cpu()
    trace = {"first_s": timings[0], "steady_median_s": float(np.median(timings[1:])), "training_s": sum(timings)}
    return forecast.cpu().numpy(), state, trace
