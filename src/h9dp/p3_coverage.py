"""P3.1: shared dense raw windows, residual routing and role/time-aware readout.

There is no learned spatial encoder. K/V are shared across queries; exact online
softmax bounds attention memory without reducing observations to patch tokens.
"""

from __future__ import annotations

import math
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .p3 import (
    PARAMETER_LIMIT,
    global_observations,
    patch_offsets,
    position_encoding,
    sample_observations,
)

ARCHITECTURE = "p3-coverage/v1"
ROLE_SLICES = (slice(0, 4), slice(4, 12), slice(12, 16))


def _attention_chunk(query, key, value, raw, valid):
    # Both softmax and the unprojected brightness path accumulate in FP32,
    # including under outer CUDA/CPU BF16 autocast.
    with torch.autocast(query.device.type, enabled=False):
        scores = query.float() @ key.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        maximum = scores.amax(-1, keepdim=True).detach()
        safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = (scores - safe_maximum).exp()
        return maximum, weights.sum(-1, keepdim=True), weights @ value.float(), weights @ raw.float()[:, None]


def online_attention(query, key, value, raw, valid, chunk_size, checkpoint_chunks=False):
    """Exact masked attention for [batch,heads,queries/points,head_width].

    Raw observations [batch,points,channels] bypass value projection. Empty
    chunks and wholly masked memories yield zeros, not NaNs. Chunk checkpointing
    trades recomputation for training memory; it does not detach routing.
    """
    if chunk_size < 1 or key.shape[-2] < 1:
        raise ValueError("Attention requires positive chunk size and memory length")
    def compute(start):
        stop = start + chunk_size
        args = (query, key[..., start:stop, :], value[..., start:stop, :], raw[:, start:stop], valid[:, start:stop])
        if checkpoint_chunks and torch.is_grad_enabled():
            return checkpoint(_attention_chunk, *args, use_reentrant=False)
        return _attention_chunk(*args)

    maximum, denominator, features, brightness = compute(0)
    for start in range(chunk_size, key.shape[-2], chunk_size):
        local_max, local_sum, local_features, local_raw = compute(start)
        merged = torch.maximum(maximum, local_max)
        old_scale = torch.where(torch.isfinite(maximum), (maximum - merged).exp(), torch.zeros_like(maximum))
        new_scale = torch.where(torch.isfinite(local_max), (local_max - merged).exp(), torch.zeros_like(local_max))
        denominator = denominator * old_scale + local_sum * new_scale
        features = features * old_scale + local_features * new_scale
        brightness = brightness * old_scale + local_raw * new_scale
        maximum = merged
    return features / denominator.clamp_min(1e-12), (brightness / denominator.clamp_min(1e-12)).mean(1)


def native_window(clean, mask, centers, size):
    """Complete integer native windows at [B,windows,T,2] normalized centers."""
    height, width = clean.shape[-2:]
    step = clean.new_tensor([2 / (width - 1), 2 / (height - 1)])
    half_extent = clean.new_tensor([(width - 1) / 2, (height - 1) / 2])
    snapped = ((centers + 1) * half_extent).round() * step - 1
    xy = snapped.unsqueeze(-2) + patch_offsets(size, clean.device) * ((size - 1) / 2) * step
    raw, valid = sample_observations(clean, mask, xy, "nearest")
    return raw, valid, xy


def role_mean(value):
    return torch.stack([value[:, section].mean(1) for section in ROLE_SLICES], dim=1)


class CoverageBlock(nn.Module):
    def __init__(self, width, heads, ratio, dropout, chunk_size, checkpoint_chunks):
        super().__init__()
        self.heads, self.chunk_size = heads, chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.background_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        # Reuse background_attention's Q/K/V/output weights for dense reads.
        self.ffn = nn.Sequential(nn.Linear(width, width * ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * ratio, width))
        self.dropout = nn.Dropout(dropout)
        route_output = nn.Linear(32, 3)
        nn.init.normal_(route_output.weight, std=0.01)
        nn.init.zeros_(route_output.bias)
        self.router = nn.Sequential(nn.Linear(width, 32), nn.GELU(), route_output)

    def contextualize(self, queries, memory, valid):
        q = self.self_norm(queries)
        queries = queries + self.dropout(self.self_attention(q, q, q, need_weights=False)[0])
        context = self.background_attention(self.cross_norm(queries), memory, memory, key_padding_mask=~valid, need_weights=False)[0]
        return queries + self.dropout(context)

    def read(self, queries, temporal, memory, raw, valid):
        batch, times, points, width = memory.shape
        count = queries.shape[1]
        attention = self.background_attention
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        q = F.linear(self.cross_norm(queries)[:, None] + temporal[:, :, None], weight[:width], bias[:width])
        k = F.linear(memory, weight[width:2 * width], bias[width:2 * width])
        v = F.linear(memory, weight[2 * width:], bias[2 * width:])
        def split(value, length):
            return value.reshape(batch * times, length, self.heads, width // self.heads).transpose(1, 2)
        feature, observed = online_attention(
            split(q, count), split(k, points), split(v, points), raw.reshape(batch * times, points, -1),
            valid.reshape(batch * times, points), self.chunk_size, self.training and self.checkpoint_chunks,
        )
        feature = feature.transpose(1, 2).reshape(batch, times, count, width).mean(1)
        queries = queries + self.dropout(attention.out_proj(feature.to(queries.dtype)))
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        observed = observed.reshape(batch, times, count, -1).transpose(1, 2)
        return queries, observed[..., :4], observed[..., 4:]


class CoverageReadout(nn.Module):
    def __init__(self, width, layers, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        # Role features; every layer/role/frame/band; two role contrasts;
        # four source masses per layer and role, averaged only over time.
        inputs = 3 * width + layers * (3 * 6 * 4 + 2 * 6 * 4 + 3 * 4)
        output = nn.Linear(32, 1)
        nn.init.xavier_uniform_(output.weight, gain=0.2)
        nn.init.zeros_(output.bias)
        self.mlp = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Dropout(dropout), output)

    def forward(self, queries, history, source_history):
        features = role_mean(self.norm(queries)).flatten(1)
        values = torch.stack(history, dim=1)  # B,L,role,T,band
        contrasts = torch.stack((values[:, :, 0] - values[:, :, 1], values[:, :, 0] - values[:, :, 2]), dim=2)
        coverage = torch.stack(source_history, dim=1).mean(-2)
        return self.mlp(torch.cat((features, values.flatten(1), contrasts.flatten(1), coverage.flatten(1)), dim=-1)).squeeze(-1)


class CoveragePressureRegressor(nn.Module):
    target_mean: torch.Tensor
    target_std: torch.Tensor
    anchors: torch.Tensor

    def __init__(self, config, target_mean=70.0, target_std=20.0):
        super().__init__()
        width = operator.index(config.get("width", 64))
        heads = operator.index(config.get("heads", 4))
        layers = operator.index(config.get("layers", 2))
        ratio = operator.index(config.get("ffn_ratio", 2))
        self.center_size = operator.index(config.get("center_window", 129))
        self.native_size = operator.index(config.get("native_window", 25))
        self.dynamic_size = operator.index(config.get("dynamic_patch", 13))
        self.global_grid = operator.index(config.get("global_grid", 16))
        chunk = operator.index(config.get("attention_chunk", 2048))
        try:
            dropout = float(config.get("dropout", 0.1))
            self.refine_pixels = float(config.get("refine_pixels", 16))
            self.minimum_radius = float(config.get("minimum_radius_pixels", 4))
            self.maximum_radius = float(config.get("maximum_radius_pixels", 48))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Routing bounds and dropout must be numeric") from error
        if width < 8 or heads < 1 or width % heads or layers < 1 or ratio < 1 or chunk < 1 or self.global_grid < 1:
            raise ValueError("Invalid coverage width/heads/layers/attention settings")
        if config.get("queries", 16) != 16 or config.get("dynamic_windows", 4) != 4:
            raise ValueError("P3.1 requires 16 role queries and 4 dynamic windows")
        if any(size < 3 or size % 2 == 0 for size in (self.center_size, self.native_size, self.dynamic_size)):
            raise ValueError("Native and dynamic windows must have odd sizes >=3")
        if not (0 <= dropout < 1 and self.refine_pixels > 0 and 0 < self.minimum_radius < self.maximum_radius):
            raise ValueError("Invalid dropout or routing bounds")
        if not all(math.isfinite(x) for x in (self.refine_pixels, self.minimum_radius, self.maximum_radius, target_mean, target_std)) or target_std <= 0:
            raise ValueError("Routing/target normalization must be finite, target std positive")
        self.width = width
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))
        self.register_buffer("anchors", torch.tensor([[0., 0.], [-.65, -.65], [.65, -.65], [0., .65]]))
        self.input_projection = nn.Linear(4, width)
        self.queries = nn.Parameter(torch.randn(16, width) * 0.02)
        self.role_embedding = nn.Parameter(torch.randn(3, width) * 0.02)
        self.blocks = nn.ModuleList([CoverageBlock(width, heads, ratio, dropout, chunk, bool(config.get("checkpoint_chunks", True))) for _ in range(layers)])
        self.head = CoverageReadout(width, layers, dropout)
        self.last_read_summary = None
        if sum(p.numel() for p in self.parameters()) >= PARAMETER_LIMIT:
            raise ValueError(f"Complete P3.1 model must have <{PARAMETER_LIMIT:,} parameters")

    def _embed(self, raw, xy, time_offsets):
        # xy/raw [B,T,N,2/4]. Coordinate encoding does not mix spatial pixels.
        time = (time_offsets.float() / 300)[:, :, None, None].expand(*xy.shape[:-1], 1)
        return self.input_projection(raw) + position_encoding(torch.cat((xy, time), dim=-1), self.width)

    def _forward(self, images, time_offsets, mask, intervention="normal"):
        if images.ndim != 5 or images.shape[1:3] != (6, 4) or not images.is_floating_point() or min(images.shape[-2:]) < 2:
            raise ValueError("Expected floating [B,6,4,H,W] images")
        batch, times, _, height, width = images.shape
        if mask.shape != (batch, times, 1, height, width) or mask.dtype != torch.bool or time_offsets.shape != (batch, times):
            raise ValueError("Mask/time shape or dtype mismatch")
        if not bool(mask.flatten(2).any(-1).all()) or not bool(torch.isfinite(time_offsets).all()):
            raise ValueError("Entirely invalid frame or non-finite time")
        if intervention not in ("normal", "no_dynamic", "fixed_route", "no_history"):
            raise ValueError(f"Unknown read intervention: {intervention}")
        clean = images.float().masked_fill(~mask, 0)
        global_raw, global_valid, global_xy = global_observations(clean, mask, self.global_grid)
        zero = clean.new_zeros(batch, 1, times, 2)
        fixed_raw, fixed_valid, fixed_xy = native_window(clean, mask, zero, self.center_size)
        fixed_raw, fixed_valid, fixed_xy = (v[:, 0] for v in (fixed_raw, fixed_valid, fixed_xy))
        base_raw = torch.cat((global_raw, fixed_raw), dim=2)
        base_valid = torch.cat((global_valid, fixed_valid), dim=2)
        base_xy = torch.cat((global_xy, fixed_xy), dim=2)
        base_memory = self._embed(base_raw, base_xy, time_offsets)
        background = base_memory[:, :, :global_raw.shape[2]].flatten(1, 2)
        background_valid = global_valid.flatten(1, 2)
        role = torch.cat([self.role_embedding[i:i+1].expand(section.stop - section.start, -1) for i, section in enumerate(ROLE_SLICES)])
        queries = (self.queries + role)[None].expand(batch, -1, -1)
        temporal_coordinates = clean.new_zeros(batch, times, 3)
        temporal_coordinates[..., 2] = time_offsets / 300
        temporal = position_encoding(temporal_coordinates, self.width)
        step = clean.new_tensor([2 / (width - 1), 2 / (height - 1)])
        centers = self.anchors[None, :, None].expand(batch, -1, times, -1)
        log_radius = clean.new_full((batch, 4, times, 1), math.log(math.sqrt(self.minimum_radius * self.maximum_radius)))
        history, mass_history, routes = [], [], []
        for layer, block in enumerate(self.blocks):
            assert isinstance(block, CoverageBlock)
            queries = block.contextualize(queries, background, background_valid)
            route = block.router(queries[:, :4, None] + temporal[:, None]).float()
            if intervention == "fixed_route":
                route = route * 0
            if layer == 0:
                centers = (torch.atanh(self.anchors)[None, :, None] + route[..., :2]).tanh()
            else:
                centers = (centers + self.refine_pixels * step * route[..., :2].tanh()).clamp(-1, 1)
            log_radius = (log_radius + .5 * route[..., 2:].tanh()).clamp(math.log(self.minimum_radius), math.log(self.maximum_radius))
            radius = log_radius.exp()
            native_raw, native_valid, native_xy = native_window(clean, mask, centers, self.native_size)
            dynamic_xy = centers.unsqueeze(-2) + radius.unsqueeze(-2) * step * patch_offsets(self.dynamic_size, images.device)
            dynamic_raw, dynamic_valid = sample_observations(clean, mask, dynamic_xy, "bilinear")
            local_raw = torch.cat((native_raw, dynamic_raw), dim=-2)
            local_valid = torch.cat((native_valid, dynamic_valid), dim=-1)
            local_xy = torch.cat((native_xy, dynamic_xy), dim=-2)
            # Window identity remains encoded by position; no replication per query.
            local_raw = local_raw.permute(0, 2, 1, 3, 4).flatten(2, 3)
            local_xy = local_xy.permute(0, 2, 1, 3, 4).flatten(2, 3)
            local_valid = local_valid.permute(0, 2, 1, 3).flatten(2, 3)
            if intervention == "no_dynamic":
                local_valid = torch.zeros_like(local_valid)
            memory = torch.cat((base_memory, self._embed(local_raw, local_xy, time_offsets)), dim=2)
            raw = torch.cat((base_raw, local_raw), dim=2)
            valid = torch.cat((base_valid, local_valid), dim=2)
            # Source masses use the exact same normalized weights as brightness.
            sources = clean.new_zeros(raw.shape[-2], 4)
            global_end, fixed_end = global_raw.shape[-2], base_raw.shape[-2]
            sources[:global_end, 0] = 1
            sources[global_end:fixed_end, 1] = 1
            labels = torch.tensor([2] * self.native_size**2 + [3] * self.dynamic_size**2, device=images.device).repeat(4)
            sources[fixed_end:] = F.one_hot(labels, 4).float()
            augmented_raw = torch.cat((raw, sources[None, None].expand(batch, times, -1, -1)), dim=-1)
            queries, observed, masses = block.read(queries, temporal, memory, augmented_raw, valid)
            history.append(role_mean(observed))
            mass_history.append(role_mean(masses))
            routes.append({"centers": centers, "radius_pixels": radius, "native_valid_fraction": native_valid.float().mean(-1), "dynamic_valid_fraction": dynamic_valid.float().mean(-1)})
        read_history = history if intervention != "no_history" else [torch.zeros_like(x) if i < len(history)-1 else x for i, x in enumerate(history)]
        score = self.head(queries, read_history, mass_history)
        prediction = F.softplus(self.target_mean + self.target_std * score.float())
        details = {"routes": routes, "brightness": torch.stack(history, 1), "source_mass": torch.stack(mass_history, 1)}
        return prediction, details

    def forward(self, images, time_offsets, mask):
        prediction, details = self._forward(images, time_offsets, mask)
        if not self.training:
            # Small detached summaries only; never retain dense attention maps.
            centers = torch.stack([r["centers"].detach() for r in details["routes"]], dim=1)
            scale = centers.new_tensor([(images.shape[-1] - 1) / 2, (images.shape[-2] - 1) / 2])
            refinement = (centers[:, 1:] - centers[:, :-1]).abs() * scale
            self.last_read_summary = {
                "source_mass": details["source_mass"].detach().mean((0, 2, 3)).cpu().tolist(),
                "radius_pixels_mean": torch.stack([r["radius_pixels"].detach().mean() for r in details["routes"]]).mean().item(),
                "native_valid_fraction": torch.stack([r["native_valid_fraction"].detach().mean() for r in details["routes"]]).mean().item(),
                "dynamic_valid_fraction": torch.stack([r["dynamic_valid_fraction"].detach().mean() for r in details["routes"]]).mean().item(),
                "boundary_coordinate_fraction": (centers.abs() >= 1 - 1e-6).float().mean().item(),
                "mean_axis_refinement_pixels": refinement.mean().item() if refinement.numel() else 0.,
            }
        return prediction

    @torch.no_grad()
    def inspect_reads(self, images, time_offsets, mask, intervention="normal"):
        if self.training:
            raise ValueError("Call eval() before read-only inspection")
        return self._forward(images, time_offsets, mask, intervention)
