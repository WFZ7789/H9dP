"""Differentiable multiple-kernel regression for the R1 small-data regime."""

from __future__ import annotations

import math

import torch
from torch import nn

LATITUDE_LOG_MASS = math.log(0.10)


def group_weights(groups: int, indices: list[int], weights: list[float]) -> torch.Tensor:
    vector = torch.full((groups,), 0.08, dtype=torch.float64)
    vector[indices] = torch.tensor(weights, dtype=torch.float64)
    return vector / vector.sum()


class SpectralGP(nn.Module):
    """A global morphology prior plus separate smooth and local scene kernels."""

    geometry_prior: torch.Tensor
    scene_prior: torch.Tensor
    scale_prior: torch.Tensor
    mass_prior: torch.Tensor

    def __init__(self, geometry_prior: torch.Tensor, scene_prior: torch.Tensor, local: bool, *, latitude: bool = False):
        super().__init__()
        self.register_buffer("geometry_prior", geometry_prior.log())
        self.register_buffer("scene_prior", scene_prior.log())
        self.geometry_delta = nn.Parameter(torch.zeros_like(geometry_prior))
        self.scene_delta = nn.Parameter(torch.zeros_like(scene_prior))
        self.scales = nn.Parameter(torch.tensor([0.8, -1.8 if local else 0.0, -0.9, -2.25, -2.0, -1.35, -0.8],
                                                dtype=torch.float64))
        self.register_buffer("scale_prior", self.scales.detach().clone())
        self.masses = nn.Parameter(torch.tensor([0.0, -0.9, -0.8, -2.4], dtype=torch.float64))
        self.register_buffer("mass_prior", self.masses.detach().clone())
        self.log_latitude_mass = (nn.Parameter(torch.tensor(LATITUDE_LOG_MASS, dtype=torch.float64))
                                  if latitude else None)

    def kernel(self, dot: torch.Tensor, distance: torch.Tensor, scene: torch.Tensor,
               latitude: torch.Tensor | None = None) -> torch.Tensor:
        # Each component is PSD; positive mixtures remain PSD at every step of hyperparameter fitting.
        geometry = (self.geometry_prior + self.geometry_delta).softmax(0)
        visual = (self.scene_prior + self.scene_delta).softmax(0)
        p = self.scales.clamp(-7, 5).exp()
        weight = self.masses.clamp(-7, 5).exp()
        gd = torch.einsum("g,gij->ij", geometry, distance)
        sd = torch.einsum("g,gij->ij", visual, scene)
        line = torch.einsum("g,gij->ij", geometry, dot)
        smooth = torch.sqrt(3 * gd.clamp_min(1e-12) / p[0])
        kernel = (weight[0] * (1 + smooth) * (-smooth).exp() + weight[1] * line
                  + weight[2] * (-sd / (2 * p[1])).exp() + weight[3] * (-gd / (2 * p[2])).exp())
        if self.log_latitude_mass is not None and latitude is not None:
            return kernel + self.log_latitude_mass.clamp(-7, 5).exp() * latitude
        if self.log_latitude_mass is not None or latitude is not None:
            raise ValueError("Latitude kernel and model configuration must agree")
        return kernel

    def system(self, kernel: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        variance = self.scales[3:6].clamp(-8, 2).exp() + 0.003
        shared_mean = self.scales[6].clamp(-3, 4).exp() * 20.0
        return kernel + shared_mean + torch.diag(variance[quality])

    def forward(self, dot: torch.Tensor, distance: torch.Tensor, scene: torch.Tensor, y: torch.Tensor,
                quality: torch.Tensor, importance: torch.Tensor, evidence: float,
                latitude: torch.Tensor | None = None) -> torch.Tensor:
        kernel = self.kernel(dot, distance, scene, latitude)
        chol, _ = torch.linalg.cholesky_ex(self.system(kernel, quality), check_errors=False)
        inverse = torch.cholesky_inverse(chol)
        dual = inverse @ y
        residual = dual / inverse.diag()
        loo = (residual.square() * importance).mean()
        nll = ((y @ dual) * 0.5 + chol.diag().log().sum()) / len(y)
        regularizer = (self.geometry_delta.square().mean() + self.scene_delta.square().mean()
                       + 0.2 * (self.scales - self.scale_prior).square().mean()
                       + 0.15 * (self.masses - self.mass_prior).square().mean())
        if self.log_latitude_mass is not None:
            regularizer = regularizer + 0.15 * (self.log_latitude_mass - LATITUDE_LOG_MASS).square()
        return (1 - evidence) * loo + evidence * nll + 0.012 * regularizer

    @torch.no_grad()
    def predict(self, fit: torch.Tensor, held: torch.Tensor, y: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        chol = torch.linalg.cholesky(self.system(fit, quality))
        dual = torch.cholesky_solve(y[:, None], chol).squeeze(1)
        return (held + self.scales[6].clamp(-3, 4).exp() * 20.0) @ dual


def schedule(step: int, steps: int, lr: float) -> float:
    return lr * (0.12 + 0.88 * (1 + math.cos(math.pi * step / steps)) / 2)
