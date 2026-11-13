"""Six-frame regression with sixteen residual blocks on a dense 250x250 map."""

from __future__ import annotations

import math
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .p3 import ARCHITECTURE as P3_ARCHITECTURE
from .p3 import P3PressureRegressor
from .p3_coverage import ARCHITECTURE as COVERAGE_ARCHITECTURE
from .p3_coverage import CoveragePressureRegressor
from .t1a import ARCHITECTURE as T1A_ARCHITECTURE
from .t1a import T1APressureRegressor

ARCHITECTURE = "dense250-residual16/v2"
IMAGE_CHANNELS = 6 * 4 * 4
MASK_CHANNELS = 6 * 4
TIME_CHANNELS = 6


def pack_observations(images, time_offsets, mask):
    """Rearrange 2x2 pixels into channels; no averaging or pixel selection."""
    batch, times, channels, height, width = images.shape
    image_planes = F.pixel_unshuffle(
        images.masked_fill(~mask, 0).reshape(batch, times * channels, height, width), 2
    )
    mask_planes = F.pixel_unshuffle(mask[:, :, 0].to(images.dtype), 2)
    time_planes = (time_offsets / 300).to(images.dtype)[:, :, None, None]
    time_planes = time_planes.expand(-1, -1, height // 2, width // 2)
    valid = F.pixel_unshuffle(mask.any(dim=1).float(), 2).bool().any(dim=1, keepdim=True)
    return torch.cat((image_planes, mask_planes, time_planes), dim=1), valid


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups = math.gcd(channels, 8)
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, features):
        # Unnormalized identity carries amplitude and full spatial layout.
        return F.silu(features + self.residual_scale * self.branch(features))


class ScalarReadout(nn.Module):
    """Spatial reduction occurs only here, after all sixteen dense blocks."""

    def __init__(self, channels, hidden, dropout):
        super().__init__()
        self.spatial_score = nn.Conv2d(channels, 1, 1)
        output_layer = nn.Linear(hidden, 1)
        self.regression = nn.Sequential(
            nn.Linear(channels * 3, hidden), nn.SiLU(),
            nn.Dropout(dropout), output_layer,
        )
        nn.init.xavier_uniform_(output_layer.weight, gain=0.2)
        nn.init.zeros_(output_layer.bias)

    def forward(self, features, valid):
        values = features.float().flatten(2)
        weights = self.spatial_score(features).float().flatten(2)
        weights = weights.masked_fill(~valid.flatten(2), -torch.inf).softmax(dim=-1)
        focused = (values * weights).sum(-1)
        mean = (values * valid.flatten(2)).sum(-1) / valid.flatten(2).sum(-1).clamp_min(1)
        maximum = values.masked_fill(~valid.flatten(2), -torch.inf).amax(-1)
        return self.regression(torch.cat((focused, mean, maximum), dim=-1)).squeeze(-1)


class DensePressureRegressor(nn.Module):
    def __init__(self, config, target_mean=70.0, target_std=20.0):
        super().__init__()
        channels = operator.index(config["channels"])
        count = operator.index(config["residual_blocks"])
        dilations = list(config["dilations"])
        if count != 16 or len(dilations) != count or any(d < 1 for d in dilations):
            raise ValueError("Exactly sixteen convolutional residual blocks are required")
        if channels < 1 or not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid channels or training-target normalization")
        self.checkpointing = config["activation_checkpointing"]
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))
        self.stem = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS + MASK_CHANNELS + TIME_CHANNELS, channels, 1),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels, dilation) for dilation in dilations])
        self.head = ScalarReadout(channels, config["head_width"], config["head_dropout"])

    def forward(self, images, time_offsets, mask):
        if images.ndim != 5 or images.shape[1:3] != (6, 4):
            raise ValueError("Expected [batch,6,4,height,width] images")
        batch, times, _, height, width = images.shape
        if mask.shape != (batch, times, 1, height, width) or time_offsets.shape != (batch, times):
            raise ValueError("Mask/time shape does not match images")
        if height % 2 or width % 2:
            raise ValueError("Pixel rearrangement requires even image dimensions")
        valid_frames = mask.flatten(2).any(-1).all()
        if not torch.compiler.is_compiling() and not bool(valid_frames):
            raise ValueError("Cannot encode an entirely invalid image")
        packed, valid = pack_observations(images, time_offsets, mask)
        value = self.stem(packed) * valid
        for block in self.blocks:
            if self.training and self.checkpointing and torch.is_grad_enabled():
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
            value = value * valid
        score = self.head(value, valid)
        return F.softplus(self.target_mean + self.target_std * score.float())


def model_architecture(config):
    architecture = config.get("architecture", ARCHITECTURE)
    if architecture not in (ARCHITECTURE, P3_ARCHITECTURE, COVERAGE_ARCHITECTURE, T1A_ARCHITECTURE):
        raise ValueError(f"Unknown model architecture: {architecture}")
    return architecture


def build_model(config, target_mean=70.0, target_std=20.0):
    architecture = model_architecture(config)
    model_type = {ARCHITECTURE: DensePressureRegressor, P3_ARCHITECTURE: P3PressureRegressor,
                  COVERAGE_ARCHITECTURE: CoveragePressureRegressor,
                  T1A_ARCHITECTURE: T1APressureRegressor}[architecture]
    return model_type(config, target_mean, target_std)


def regression_loss(prediction, target, target_std):
    """Every record has the same coefficient in normalized physical MSE."""
    return ((prediction.float() - target.float()) / target_std).square()
