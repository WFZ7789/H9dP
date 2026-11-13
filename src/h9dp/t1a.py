"""T1A (test-1 a): signed wavelets, local relations, then temporal map fusion.

All inputs follow the existing native500 training contract. No queries, sampled
windows, per-image normalization, or auxiliary supervision are used.
"""

from __future__ import annotations

import math
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ARCHITECTURE = "t1a/v1"


def haar_split(features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal signed Haar coefficients; only complete valid 2x2 cells survive.

    Channel order is LL, LH, HL, HH. Odd bottom/right edges are padded as invalid.
    The transform is invertible on complete cells, not after masks/projections.
    """
    height, width = features.shape[-2:]
    features = features.masked_fill(~valid, 0)
    features = F.pad(features, (0, width % 2, 0, height % 2))
    valid = F.pad(valid, (0, width % 2, 0, height % 2), value=False)
    a, b = features[..., 0::2, 0::2], features[..., 0::2, 1::2]
    c, d = features[..., 1::2, 0::2], features[..., 1::2, 1::2]
    complete = (valid[..., 0::2, 0::2] & valid[..., 0::2, 1::2]
                & valid[..., 1::2, 0::2] & valid[..., 1::2, 1::2])
    coefficients = torch.cat(((a + b + c + d) / 2, (-a - b + c + d) / 2,
                              (-a + b - c + d) / 2, (a - b - c + d) / 2), dim=1)
    return coefficients.masked_fill(~complete, 0), complete


def masked_local_mean(features: torch.Tensor, valid: torch.Tensor, kernel: int) -> torch.Tensor:
    weights = valid.to(features.dtype)
    numerator = F.avg_pool2d(features.masked_fill(~valid, 0), kernel, 1, kernel // 2)
    denominator = F.avg_pool2d(weights, kernel, 1, kernel // 2)
    return numerator / denominator.clamp_min(1 / (kernel * kernel))


class RelationBlock(nn.Module):
    """Absolute features and two signed neighbourhood contrasts before reduction."""

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Conv2d(3 * channels, 2 * channels, 1)
        self.spatial = nn.Conv2d(2 * channels, 2 * channels, 5, padding=2,
                                 groups=2 * channels, bias=False)
        self.project = nn.Conv2d(2 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        features = features.masked_fill(~valid, 0)
        contrast3 = features - masked_local_mean(features, valid, 3)
        contrast7 = features - masked_local_mean(features, valid, 7)
        mixed = F.gelu(self.mix(torch.cat((features, contrast3, contrast7), dim=1)))
        support = F.avg_pool2d(valid.to(mixed.dtype), 5, 1, 2).clamp_min(1 / 25)
        local = self.spatial(mixed.masked_fill(~valid, 0)) / support
        update = self.project(F.gelu(local))
        return (features + self.scale * update).masked_fill(~valid, 0)


class SpatialStage(nn.Module):
    def __init__(self, incoming: int, channels: int, blocks: int, checkpointing: bool):
        super().__init__()
        self.project = nn.Conv2d(4 * incoming, channels, 1)
        self.blocks = nn.ModuleList([RelationBlock(channels) for _ in range(blocks)])
        self.checkpointing = checkpointing

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, valid = haar_split(features, valid)
        features = F.gelu(self.project(features)).masked_fill(~valid, 0)
        for block in self.blocks:
            if self.training and self.checkpointing and torch.is_grad_enabled():
                features = checkpoint(block, features, valid, use_reentrant=False)
            else:
                features = block(features, valid)
        return features, valid


class TemporalMapFusion(nn.Module):
    """Non-exchangeable six-frame fusion before global spatial pooling."""

    def __init__(self, channels: int, time_width: int):
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(3, time_width), nn.GELU())
        incoming = 6 * (channels + time_width + 1)
        self.mix = nn.Sequential(nn.Conv2d(incoming, 2 * channels, 1), nn.GELU(),
                                 nn.Conv2d(2 * channels, channels, 1))
        self.refine = RelationBlock(channels)

    def forward(self, features: torch.Tensor, valid: torch.Tensor,
                time_offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _, height, width = features.shape
        tau = time_offsets.float() / 300
        gaps = torch.cat((torch.zeros_like(tau[:, :1]), tau[:, 1:] - tau[:, :-1]), dim=1)
        encoded = self.time_embedding(torch.stack((tau, gaps, tau.square()), dim=-1))
        encoded = encoded.to(features.dtype)[..., None, None].expand(-1, -1, -1, height, width)
        inputs = torch.cat((features, encoded.masked_fill(~valid, 0), valid.to(features.dtype)), dim=2)
        fused = self.mix(inputs.reshape(batch, -1, height, width))
        closest = time_offsets.abs().argmin(dim=1)
        anchor = features[torch.arange(batch, device=features.device), closest]
        support = valid.any(dim=1)
        fused = (anchor + fused).masked_fill(~support, 0)
        return self.refine(fused, support), support


def statistical_readout(features: torch.Tensor, valid: torch.Tensor, fraction: float) -> torch.Tensor:
    """Masked mean/std and upper/lower tail means, with a per-image valid count."""
    values = features.float().flatten(2)
    support = valid.flatten(2)
    counts = support.sum(-1)
    denominator = counts.clamp_min(1)
    mean = values.masked_fill(~support, 0).sum(-1) / denominator
    variance = ((values - mean[..., None]).square().masked_fill(~support, 0)).sum(-1) / denominator
    # Finite gradient at zero variance; exactly zero for an empty/constant map.
    deviation = (variance + 1e-8).sqrt() - 1e-4
    capacity = max(1, math.ceil(values.shape[-1] * fraction))
    requested = (counts.float() * fraction).ceil().long()
    ranks = torch.arange(capacity, device=features.device)[None, None, :]
    tails = []
    for upper in (True, False):
        masked = values.masked_fill(~support, -torch.inf if upper else torch.inf)
        selected = masked.topk(capacity, dim=-1, largest=upper).values
        included = (ranks < requested[..., None]) & torch.isfinite(selected)
        tails.append(torch.where(included, selected, 0).sum(-1) / requested.clamp_min(1))
    summaries = torch.cat((mean, deviation, *tails), dim=1)
    return summaries.masked_fill(~(counts > 0), 0)


class T1APressureRegressor(nn.Module):
    """Six four-band images -> one unconstrained dP estimate in hPa."""

    target_mean: torch.Tensor
    target_std: torch.Tensor

    def __init__(self, config: dict, target_mean: float = 70.0, target_std: float = 20.0):
        super().__init__()
        widths = tuple(operator.index(value) for value in config.get("widths", (32, 48, 64, 96)))
        blocks = operator.index(config.get("blocks_per_stage", 2))
        time_width = operator.index(config.get("time_width", 4))
        head_width = operator.index(config.get("head_width", 64))
        try:
            self.tail_fraction = float(config.get("tail_fraction", 0.1))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("tail_fraction must be numeric") from error
        if len(widths) != 4 or min(widths) < 1 or min(blocks, time_width, head_width) < 1:
            raise ValueError("T1A needs four positive widths and positive block/time/head sizes")
        if not 0 < self.tail_fraction <= 0.5:
            raise ValueError("tail_fraction must be in (0, 0.5]")
        if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid training-target normalization")
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))
        incoming = (4, *widths[:-1])
        checkpointing = bool(config.get("activation_checkpointing", True))
        self.stages = nn.ModuleList([
            SpatialStage(previous, width, blocks, checkpointing)
            for previous, width in zip(incoming, widths, strict=True)
        ])
        self.temporal = nn.ModuleList([TemporalMapFusion(width, time_width) for width in widths[1:]])
        self.head = nn.Sequential(nn.Linear(4 * sum(widths[1:]), head_width), nn.GELU(),
                                  nn.Linear(head_width, 1))
        output = self.head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.normal_(output.weight, std=0.02)
        nn.init.zeros_(output.bias)

    def extract_features(self, images: torch.Tensor, time_offsets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (6, 4):
            raise ValueError("Expected [batch,6,4,height,width] images")
        batch, times, _, height, width = images.shape
        if min(height, width) < 16:
            raise ValueError("T1A requires image dimensions >=16")
        if mask.shape != (batch, times, 1, height, width) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean [batch,6,1,height,width] mask")
        if time_offsets.shape != (batch, times):
            raise ValueError("Expected [batch,6] time offsets in minutes")
        clean = images.masked_fill(~mask, 0)
        if not torch.compiler.is_compiling():
            if not bool(mask.flatten(2).any(-1).all()):
                raise ValueError("Cannot encode an entirely invalid frame")
            if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(time_offsets).all()):
                raise ValueError("Non-finite valid observation or time offset")
            if not bool((time_offsets[:, 1:] > time_offsets[:, :-1]).all()):
                raise ValueError("Time offsets must be strictly increasing")
        features = clean.reshape(batch * times, 4, height, width)
        valid = mask.reshape(batch * times, 1, height, width)
        summaries = []
        for level, stage in enumerate(self.stages):
            features, valid = stage(features, valid)
            if level:
                h, w = features.shape[-2:]
                fused, support = self.temporal[level - 1](
                    features.reshape(batch, times, -1, h, w),
                    valid.reshape(batch, times, 1, h, w), time_offsets,
                )
                summaries.append(statistical_readout(fused, support, self.tail_fraction))
        return torch.cat(summaries, dim=1)

    def forward(self, images: torch.Tensor, time_offsets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        score = self.head(self.extract_features(images, time_offsets, mask)).squeeze(-1)
        # No softplus/clamping: train/evaluation use the same linear hPa output.
        return self.target_mean + self.target_std * score.float()
