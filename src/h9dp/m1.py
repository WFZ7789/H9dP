"""M1: independent small experts for MOTIS intermediate parameters.

Each expert sees the same six-frame sequence and predicts one scalar parameter.
The M5 mapping combines four of them with a CDO-temperature scale.  The module
contains no training loop and no data access; see scripts/m1.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PARAMETER_NAMES = ("b24_dt", "b36_dt", "eye_size", "b33_dt_trunc", "t_cdo")
M5_PARAMETER_INDICES = (0, 1, 2, 3)
CDO_PARAMETER_INDEX = 4
EYE_SIZE_MAX = 1.0 / math.sqrt(8.0)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    parameter: str
    activation: str

    @property
    def bounded(self) -> bool:
        return self.activation in {"sigmoid", "softplus"}


EXPERTS = (
    ExpertSpec("e1", "b24_dt", "linear"),
    ExpertSpec("e2", "b36_dt", "linear"),
    ExpertSpec("e3", "eye_size", "sigmoid"),
    ExpertSpec("e4", "b33_dt_trunc", "softplus"),
    ExpertSpec("e5", "t_cdo", "linear"),
)


def _separable(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
    )


class M1Encoder(nn.Module):
    """Shared structural template; every expert owns a separate instance."""

    def __init__(
        self,
        bands: int = 4,
        frames: int = 6,
        image_size: int = 500,
        radii: tuple[int, int] = (32, 96),
        channels: tuple[int, int, int, int, int] = (8, 12, 24, 32, 32),
        frame_width: int = 32,
    ):
        super().__init__()
        stem, projection, first, second, third = channels
        feature = third
        if image_size % 2:
            raise ValueError("image_size must be even")
        self.bands, self.frames = bands, frames
        self.image_size = image_size
        self.radii = tuple(int(value) for value in radii)
        if len(self.radii) != 2 or not 0 < self.radii[0] < self.radii[1]:
            raise ValueError("radii must be two increasing positive values")
        self.output_size, self.scale = self._geometry(image_size)
        self.stem = nn.Sequential(
            nn.Conv2d(bands + 1, stem, 3, padding=1, bias=False),
            nn.GroupNorm(2, stem),
            nn.SiLU(),
        )
        self.pool_projection = nn.Sequential(
            nn.Conv2d(2 * stem, projection, 1, bias=False),
            nn.GroupNorm(3, projection),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            _separable(projection, first, 2),
            _separable(first, second, 2),
            _separable(second, third, 1),
        )
        readout = 3 * 2 * feature + bands * 3 * 3 + 3
        self.frame_projection = nn.Linear(readout, frame_width)
        self.feature_width = feature
        self.frame_width = frame_width
        self.band_mean: torch.Tensor
        self.band_std: torch.Tensor
        self.register_buffer("band_mean", torch.zeros(bands))
        self.register_buffer("band_std", torch.ones(bands))

    @staticmethod
    def _geometry(image_size: int) -> tuple[int, int]:
        size = image_size // 2
        stride = 2
        for step in (2, 2, 1):
            size = (size + 2 - 3) // step + 1
            stride *= step
        if stride * size < image_size or stride * (size - 1) + 1 > image_size:
            raise ValueError("geometry does not cover the native window")
        return size, stride

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        numeric: torch.Tensor,
        coverage: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """images: Kelvin [B,F,C,H,W]; mask [B,F,1,H,W]; numeric [B,F,C,3,3]."""
        batch, frames = images.shape[:2]
        valid = mask.to(images.dtype)
        mean = self.band_mean[None, None, :, None, None]
        std = self.band_std[None, None, :, None, None]
        values = (images - mean) / std * valid
        stacked = torch.cat([values, valid], dim=2)
        features = self.stem(stacked.reshape(batch * frames, -1, *images.shape[-2:]))
        pooled = torch.cat([F.avg_pool2d(features, 2), F.max_pool2d(features, 2)], dim=1)
        features = self.pool_projection(pooled)
        features = self.blocks(features)
        features = features.reshape(batch, frames, -1, self.output_size, self.output_size)
        region = self._region_features(features, mask, centers)
        # Zero placeholders for missing regions are not measurements in Kelvin.
        present = coverage[:, :, None, :, None] > 0
        normalized = torch.where(present, (numeric - mean) / std, 0.0)
        fraction = coverage.to(images.dtype) / float(self.image_size ** 2)
        combined = torch.cat(
            [region.reshape(batch, frames, -1), normalized.reshape(batch, frames, -1), fraction],
            dim=-1,
        )
        return F.silu(self.frame_projection(combined))

    def _region_features(
        self, features: torch.Tensor, mask: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, _channels = features.shape[:3]
        size = self.output_size
        coords = torch.arange(size, device=features.device, dtype=features.dtype) * self.scale + 0.5
        rows = coords.floor().long().clamp(max=self.image_size - 1)
        dx = coords.view(1, 1, 1, size) - centers[..., 0].view(batch, frames, 1, 1)
        dy = coords.view(1, 1, size, 1) - centers[..., 1].view(batch, frames, 1, 1)
        distance = dx * dx + dy * dy
        inner, outer = self.radii
        region = (distance > inner * inner).to(torch.int8) + (distance > outer * outer).to(torch.int8)
        sampled = mask[:, :, 0][:, :, rows][:, :, :, rows]
        outputs = []
        for index in range(3):
            weights = (region == index) & sampled
            weight = weights.to(features.dtype)
            count = weight.sum(dim=(-1, -2))
            denominator = count.clamp(min=1).unsqueeze(-1)
            total = (features * weight.unsqueeze(2)).sum(dim=(-1, -2)) / denominator
            masked = features.masked_fill(~weights.unsqueeze(2), float("-inf")).amax(dim=(-1, -2))
            maximum = torch.where(count.unsqueeze(-1) > 0, masked, torch.zeros_like(masked))
            outputs.extend([total, maximum])
        return torch.cat(outputs, dim=-1)


class M1Expert(nn.Module):
    def __init__(
        self,
        spec: ExpertSpec,
        bands: int = 4,
        frames: int = 6,
        image_size: int = 500,
        radii: tuple[int, int] = (32, 96),
        channels: tuple[int, int, int, int, int] = (8, 12, 24, 32, 32),
        frame_width: int = 32,
        temporal_width: int = 32,
    ):
        super().__init__()
        self.spec = spec
        self.encoder = M1Encoder(bands, frames, image_size, radii, channels, frame_width)
        temporal_input = frames * frame_width + frames
        self.temporal = nn.Sequential(
            nn.Linear(temporal_input, temporal_width), nn.SiLU(), nn.Linear(temporal_width, 1)
        )
        self.target_mean: torch.Tensor
        self.target_std: torch.Tensor
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        numeric: torch.Tensor,
        coverage: torch.Tensor,
        times: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        frame = self.encoder(images, mask, numeric, coverage, centers)
        batch = frame.shape[0]
        inputs = torch.cat([frame.reshape(batch, -1), times / 300.0], dim=-1)
        head = self.temporal(inputs)
        if self.spec.activation == "sigmoid":
            raw, projected = head, torch.sigmoid(head) * EYE_SIZE_MAX
        elif self.spec.activation == "softplus":
            raw, projected = head, F.softplus(head)
        elif self.spec.activation == "linear":
            projected = head * self.target_std + self.target_mean
            raw = projected
        else:
            raise ValueError(f"Unknown activation: {self.spec.activation}")
        return torch.cat([raw, projected], dim=-1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class M1Direct(nn.Module):
    """Budget-matched control: same readout, trained directly on dP."""

    def __init__(
        self,
        bands: int = 4,
        frames: int = 6,
        image_size: int = 500,
        radii: tuple[int, int] = (32, 96),
        channels: tuple[int, int, int, int, int] = (16, 24, 40, 48, 48),
        frame_width: int = 88,
        temporal_width: int = 96,
    ):
        super().__init__()
        self.encoder = M1Encoder(bands, frames, image_size, radii, channels, frame_width)
        temporal_input = frames * frame_width + frames
        self.temporal = nn.Sequential(
            nn.Linear(temporal_input, temporal_width), nn.SiLU(), nn.Linear(temporal_width, 1)
        )
        self.target_mean: torch.Tensor
        self.target_std: torch.Tensor
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def forward(self, images, mask, numeric, coverage, times, centers):
        """Returns [B,1] dP in hPa; the control has no intermediate parameters."""
        frame = self.encoder(images, mask, numeric, coverage, centers)
        batch = frame.shape[0]
        inputs = torch.cat([frame.reshape(batch, -1), times / 300.0], dim=-1)
        return self.temporal(inputs) * self.target_std + self.target_mean

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class M5:
    coefficients: np.ndarray

    @staticmethod
    def fit(parameters: np.ndarray, target: np.ndarray) -> M5:
        parameters = np.asarray(parameters, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if parameters.ndim != 2 or parameters.shape[1] != len(PARAMETER_NAMES):
            raise ValueError("M5 expects five parameters per record")
        if len(parameters) != len(target) or len(target) < 2:
            raise ValueError("M5 needs matching, nonempty inputs")
        scale = (35.0 - np.minimum(parameters[:, CDO_PARAMETER_INDEX], -50.0)) / 100.0
        design = np.column_stack(
            [np.ones(len(parameters)), scale[:, None] * parameters[:, M5_PARAMETER_INDICES]]
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            raise ValueError("M5 design matrix is rank deficient")
        return M5(coefficients)

    def predict(self, parameters: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64)
        scale = (35.0 - np.minimum(parameters[:, CDO_PARAMETER_INDEX], -50.0)) / 100.0
        design = np.column_stack(
            [np.ones(len(parameters)), scale[:, None] * parameters[:, M5_PARAMETER_INDICES]]
        )
        return design @ self.coefficients


def metrics(prediction, target) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("metrics require matching nonempty arrays")
    errors = prediction - target
    return {
        "count": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }
