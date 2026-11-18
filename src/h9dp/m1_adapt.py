"""Training-only helpers for rotating, nested M1 cross-fitting.

No record is permanently withheld. Outer folds isolate the complete stack;
inner folds provide out-of-sample expert predictions to its calibrator.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from h9dp.m1 import M5


def grouped_folds(records: list[dict], count: int, seed: int) -> np.ndarray:
    """Balance record counts while keeping a whole year/storm in one fold."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["storm_id"], []).append(index)
    if count < 2 or len(groups) < count:
        raise ValueError("Cross-fitting needs at least two nonempty storm groups")
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: -len(groups[key]))
    loads = np.zeros(count, dtype=np.int64)
    result = np.full(len(records), -1, dtype=np.int64)
    for key in keys:
        candidates = np.flatnonzero(loads == loads.min())
        fold = int(rng.choice(candidates))
        result[groups[key]] = fold
        loads[fold] += len(groups[key])
    return result


def normalization(records: list[dict], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Use only the current training subset's observed pixels, never zeros."""
    if not indices:
        raise ValueError("Empty training subset")
    statistics = np.asarray([records[i]["stats"] for i in indices], dtype=np.float64)
    totals = statistics.sum(axis=(0, 1))
    if np.any(totals[:, 2] <= 0):
        raise ValueError("A band has no observed training pixels")
    mean = totals[:, 0] / totals[:, 2]
    variance = totals[:, 1] / totals[:, 2] - mean ** 2
    return mean.astype(np.float32), np.sqrt(variance.clip(min=1e-6)).astype(np.float32)


def quality_weights(records: list[dict], mapping: dict[str, float]) -> np.ndarray:
    weights = np.asarray([mapping[r["quality"]] for r in records], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Quality weights must be finite and positive; no records are excluded")
    return weights


def design_matrix(parameters: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 5 or not np.isfinite(parameters).all():
        raise ValueError("Expected finite [N,5] physical parameters")
    scale = (35.0 - np.minimum(parameters[:, 4], -50.0)) / 100.0
    return np.column_stack([np.ones(len(parameters)), scale[:, None] * parameters[:, :4]])


@dataclass
class CalibratedM5:
    coefficients: np.ndarray

    @classmethod
    def fit(
        cls, predictions: np.ndarray, target: np.ndarray, weights: np.ndarray,
        prior: M5, alpha: float,
    ) -> "CalibratedM5":
        """Ridge towards an oracle M5, fitted to INNER-OOF predictions only.

        Standardize features before penalizing slopes. The intercept is free;
        alpha is in weighted-observation units, not physical-parameter units.
        """
        x = design_matrix(predictions)
        y = np.asarray(target, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if len(y) != len(x) or weights.shape != y.shape or len(y) < 2:
            raise ValueError("Calibration arrays must have matching nonempty rows")
        if alpha <= 0 or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Calibration requires finite targets and positive weights/alpha")
        mean = np.average(x[:, 1:], axis=0, weights=weights)
        std = np.sqrt(np.average((x[:, 1:] - mean) ** 2, axis=0, weights=weights)).clip(min=1e-6)
        z = np.column_stack([np.ones(len(x)), (x[:, 1:] - mean) / std])
        prior_z = np.r_[prior.coefficients[0] + mean @ prior.coefficients[1:], prior.coefficients[1:] * std]
        penalty = np.diag([0.0, alpha, alpha, alpha, alpha])
        beta = np.linalg.solve(z.T @ (weights[:, None] * z) + penalty,
                               z.T @ (weights * y) + penalty @ prior_z)
        slopes = beta[1:] / std
        return cls(np.r_[beta[0] - mean @ slopes, slopes])

    def predict(self, parameters: np.ndarray) -> np.ndarray:
        return np.maximum(design_matrix(parameters) @ self.coefficients, 0.0)


def learning_rate(epoch: int, base: float, milestones: list[int]) -> float:
    """Fixed, replayable schedule shared by inner/outer/full-data training."""
    return base * 0.5 ** sum(epoch >= milestone for milestone in milestones)


class JointM1(nn.Module):
    """Five existing experts with a fixed, differentiable physical M5 head."""

    def __init__(self, experts: list[nn.Module], coefficients: np.ndarray):
        super().__init__()
        if len(experts) != 5 or np.asarray(coefficients).shape != (5,):
            raise ValueError("Joint M1 requires five experts and five M5 coefficients")
        self.experts = nn.ModuleList(experts)
        self.coefficients: torch.Tensor
        self.register_buffer("coefficients", torch.tensor(coefficients, dtype=torch.float32))

    def forward(self, images, mask, numeric, coverage, times, centers) -> torch.Tensor:
        parameters = torch.stack([
            expert(images, mask, numeric, coverage, times, centers)[:, 1].float()
            for expert in self.experts
        ], dim=1)
        scale = (35.0 - parameters[:, 4].clamp(max=-50.0)) / 100.0
        dp = self.coefficients[0] + scale * (parameters[:, :4] * self.coefficients[1:]).sum(dim=1)
        return torch.cat([parameters, dp[:, None]], dim=1)


def joint_loss(
    output: torch.Tensor, targets: torch.Tensor, dp: torch.Tensor,
    weights: torch.Tensor, parameter_std: torch.Tensor, dp_std: float,
    parameter_weight: float,
) -> torch.Tensor:
    """All records supervise dP; only genuinely labeled records supervise E."""
    present = torch.isfinite(targets).all(dim=1)
    safe_targets = torch.nan_to_num(targets)
    parameter_error = ((output[:, :5] - safe_targets) / parameter_std).square().mean(dim=1)
    auxiliary = (parameter_error * present * weights).sum() / present.sum().clamp(min=1)
    main = (((output[:, 5] - dp) / dp_std).square() * weights).mean()
    return main + parameter_weight * auxiliary
