"""P3 minimal: sparse dynamic-resolution attention over original brightness fields.

No spatial feature encoder precedes attention. Non-learned masked global summaries
provide coverage; native-pixel and differentiable variable-scale reads revisit
original observations at every block. Inputs use the dataset's fixed per-band
normalization, not per-image normalization.
"""

from __future__ import annotations

import math
import operator

import torch
import torch.nn.functional as F
from torch import nn

ARCHITECTURE = "p3-minimal/v1"
PARAMETER_LIMIT = 155_000


def position_encoding(coordinates, width):
    """Fixed Fourier encoding of normalized x, y and actual minutes / 300."""
    count = math.ceil(width / 6)
    frequencies = torch.exp(torch.linspace(0, math.log(32), count, device=coordinates.device)) * math.pi
    phase = coordinates.float().unsqueeze(-1) * frequencies
    return torch.stack((phase.sin(), phase.cos()), dim=-1).transpose(-3, -2).flatten(-3)[..., :width]


def patch_offsets(size, device):
    axis = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def global_observations(clean, mask, size):
    """Masked raw-band averages and their locations, not learned features."""
    batch, times, bands, height, width = clean.shape
    coverage = F.adaptive_avg_pool2d(mask.float().flatten(0, 1), size)
    values = F.adaptive_avg_pool2d(clean.flatten(0, 1), size) / coverage.clamp_min(1e-8)
    valid = coverage[:, 0] > 0
    values = values.permute(0, 2, 3, 1).reshape(batch, times, size * size, bands)
    # Adaptive pooling cells may overlap when dimensions are not divisible.
    index = torch.arange(size, device=clean.device, dtype=torch.float32)
    x = ((index * width / size).floor() + ((index + 1) * width / size).ceil() - 1) / (width - 1) - 1
    y = ((index * height / size).floor() + ((index + 1) * height / size).ceil() - 1) / (height - 1) - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).reshape(1, 1, size * size, 2).expand(batch, times, -1, -1)
    return values, valid.reshape(batch, times, -1), xy


def sample_observations(clean, mask, grid, mode):
    """Read original bands at [B,Q,T,N,2] locations, correcting masked interpolation.

    ``clean`` must already have invalid values replaced by zero. Bilinear reads
    renormalize by valid interpolation weight, so missing pixels do not cool a
    warm observation. Native reads use nearest sampling at pixel-aligned grids.
    """
    batch, times, bands, _, _ = clean.shape
    queries, points = grid.shape[1], grid.shape[3]
    flat_grid = grid.permute(0, 2, 1, 3, 4).reshape(batch * times, queries * points, 1, 2).float()
    values = F.grid_sample(clean.flatten(0, 1), flat_grid, mode=mode, padding_mode="zeros", align_corners=True)
    coverage = F.grid_sample(mask.float().flatten(0, 1), flat_grid, mode=mode, padding_mode="zeros", align_corners=True)
    values = (values / coverage.clamp_min(1e-6)).squeeze(-1).transpose(1, 2)
    values = values.reshape(batch, times, queries, points, bands).permute(0, 2, 1, 3, 4)
    valid = coverage.reshape(batch, times, queries, points).permute(0, 2, 1, 3) > 1e-6
    valid = valid & (grid.abs() <= 1).all(-1)
    return values.masked_fill(~valid.unsqueeze(-1), 0), valid


class DynamicReadBlock(nn.Module):
    def __init__(self, width, heads, ffn_ratio, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        # Keep read weights normalized, including during training, for the raw-value path.
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, width * ffn_ratio), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width * ffn_ratio, width))
        self.dropout = nn.Dropout(dropout)
        route_output = nn.Linear(32, 3)
        nn.init.normal_(route_output.weight, std=0.01)
        nn.init.zeros_(route_output.bias)
        self.router = nn.Sequential(nn.Linear(width, 32), nn.GELU(), route_output)

    def contextualize(self, queries, memory, valid):
        value = self.self_norm(queries)
        queries = queries + self.dropout(self.self_attention(value, value, value, need_weights=False)[0])
        # The same cross-attention parameters first read the background to condition routing.
        context = self.cross_attention(self.cross_norm(queries), memory, memory,
                                       key_padding_mask=~valid, need_weights=False)[0]
        return queries + self.dropout(context)

    def read(self, queries, memory, valid, raw):
        batch, count, width = queries.shape
        value, weights = self.cross_attention(
            self.cross_norm(queries).reshape(batch * count, 1, width), memory, memory,
            key_padding_mask=~valid, need_weights=True, average_attn_weights=True,
        )
        queries = queries + self.dropout(value.reshape(batch, count, width))
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        # No LayerNorm or learned value projection intervenes on this numerical path.
        read_weights = weights[:, 0].float()
        read_weights = read_weights / read_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        brightness = (read_weights.unsqueeze(-1) * raw.float()).sum(-2)
        return queries, brightness.reshape(batch, count, 4)


class P3Readout(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        output = nn.Linear(32, 1)
        nn.init.xavier_uniform_(output.weight, gain=0.2)
        nn.init.zeros_(output.bias)
        self.mlp = nn.Sequential(nn.Linear(width + 4, 32), nn.GELU(),
                                 nn.Dropout(dropout), output)

    def forward(self, queries, brightness):
        value = torch.cat((self.norm(queries).mean(1), brightness.mean(1)), dim=-1)
        return self.mlp(value).squeeze(-1)


class P3PressureRegressor(nn.Module):
    target_mean: torch.Tensor
    target_std: torch.Tensor
    anchors: torch.Tensor

    def __init__(self, config, target_mean=70.0, target_std=20.0):
        super().__init__()
        width = operator.index(config.get("width", 64))
        heads = operator.index(config.get("heads", 4))
        layers = operator.index(config.get("layers", 2))
        count = operator.index(config.get("queries", 16))
        ratio = operator.index(config.get("ffn_ratio", 2))
        self.global_grid = operator.index(config.get("global_grid", 8))
        self.native_patch = operator.index(config.get("native_patch", 3))
        self.dynamic_patch = operator.index(config.get("dynamic_patch", 5))
        try:
            self.max_radius = float(config.get("max_radius", 0.75))
            self.max_center_shift = float(config.get("max_center_shift", 0.5))
            dropout = float(config.get("dropout", 0.1))
        except (TypeError, ValueError) as error:
            raise ValueError("P3 read extent and dropout must be numeric") from error
        if width < 8 or heads < 1 or width % heads or layers < 1 or ratio < 1:
            raise ValueError("Invalid P3 width, heads, layers or FFN ratio")
        if count < 4 or math.isqrt(count) ** 2 != count or self.global_grid < 1:
            raise ValueError("P3 queries must be a square >= 4; global_grid must be positive")
        if any(size < 3 or size % 2 == 0 for size in (self.native_patch, self.dynamic_patch)):
            raise ValueError("P3 read patches must have odd sizes >= 3")
        if not 0 < self.max_radius <= 1 or not 0 < self.max_center_shift <= 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid P3 read extent or dropout")
        if not math.isfinite(target_mean) or not math.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid training-target normalization")
        self.width = width
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32))
        self.input_projection = nn.Linear(4, width)
        self.queries = nn.Parameter(torch.randn(count, width) * 0.02)
        anchors = patch_offsets(math.isqrt(count), torch.device("cpu")) * 0.75
        anchors[0] = 0  # Always initialize one reader at the window center; no label location is used.
        self.register_buffer("anchors", anchors)
        self.blocks = nn.ModuleList([DynamicReadBlock(width, heads, ratio, dropout) for _ in range(layers)])
        self.head = P3Readout(width, dropout)
        total = sum(parameter.numel() for parameter in self.parameters())
        if total >= PARAMETER_LIMIT:
            raise ValueError(f"P3 total parameters {total:,} must be < {PARAMETER_LIMIT:,}")

    def _embed(self, raw, xy, time_offsets):
        time = (time_offsets.float() / 300).unsqueeze(-1).unsqueeze(-1)
        time = time.expand(*xy.shape[:-1], 1)
        return self.input_projection(raw) + position_encoding(torch.cat((xy, time), dim=-1), self.width)

    def _forward(self, images, time_offsets, mask):
        if images.ndim != 5 or images.shape[1:3] != (6, 4):
            raise ValueError("Expected [batch,6,4,height,width] images")
        batch, times, _, height, width = images.shape
        if min(height, width) < 2 or not images.is_floating_point():
            raise ValueError("Expected floating images with spatial dimensions >= 2")
        if mask.shape != (batch, times, 1, height, width) or mask.dtype != torch.bool:
            raise ValueError("Expected matching boolean image mask")
        if time_offsets.shape != (batch, times):
            raise ValueError("Time shape does not match images")
        if not torch.compiler.is_compiling():
            if not bool(mask.flatten(2).any(-1).all()):
                raise ValueError("Cannot read an entirely invalid image")
            if not bool(torch.isfinite(time_offsets).all()):
                raise ValueError("Actual time offsets must be finite")
        clean = images.float().masked_fill(~mask, 0)
        raw_global, valid_global, xy_global = global_observations(clean, mask, self.global_grid)
        memory_global = self._embed(raw_global, xy_global, time_offsets).flatten(1, 2)
        raw_global = raw_global.flatten(1, 2)
        valid_global = valid_global.flatten(1, 2)
        count = self.queries.shape[0]
        anchor_coordinates = torch.cat((self.anchors, torch.zeros_like(self.anchors[:, :1])), dim=-1)
        queries = (self.queries + position_encoding(anchor_coordinates, self.width)).unsqueeze(0).expand(batch, -1, -1)
        temporal_coordinates = images.new_zeros(batch, times, 3, dtype=torch.float32)
        temporal_coordinates[..., 2] = time_offsets / 300
        temporal = position_encoding(temporal_coordinates, self.width)
        pixel_step = images.new_tensor([2 / (width - 1), 2 / (height - 1)], dtype=torch.float32)
        native_offsets = patch_offsets(self.native_patch, images.device) * ((self.native_patch - 1) / 2)
        scaled_offsets = patch_offsets(self.dynamic_patch, images.device)
        minimum_radius = (pixel_step * ((self.dynamic_patch - 1) / 2)).clamp_max(self.max_radius)
        maximum_radius = torch.full_like(minimum_radius, self.max_radius)
        routes = []
        brightness = images.new_zeros((batch, count, 4), dtype=torch.float32)
        for block in self.blocks:
            assert isinstance(block, DynamicReadBlock)
            queries = block.contextualize(queries, memory_global, valid_global)
            route = block.router(queries[:, :, None, :] + temporal[:, None, :, :]).float()
            centers = (self.anchors[None, :, None, :] + self.max_center_shift * route[..., :2].tanh()).clamp(-1, 1)
            radii = minimum_radius * (maximum_radius / minimum_radius).pow(route[..., 2:].sigmoid())
            # Snap only the fine branch to exact original pixels. Its coordinates
            # are discrete; the variable-scale bilinear branch trains the router.
            snapped = ((centers + 1) / pixel_step).round() * pixel_step - 1
            native_xy = snapped.unsqueeze(-2) + native_offsets * pixel_step
            dynamic_xy = centers.unsqueeze(-2) + radii.unsqueeze(-2) * scaled_offsets
            native_raw, native_valid = sample_observations(clean, mask, native_xy, "nearest")
            dynamic_raw, dynamic_valid = sample_observations(clean, mask, dynamic_xy, "bilinear")
            local_raw = torch.cat((native_raw, dynamic_raw), dim=-2)
            local_valid = torch.cat((native_valid, dynamic_valid), dim=-1).flatten(2, 3)
            local_xy = torch.cat((native_xy, dynamic_xy), dim=-2)
            local_memory = self._embed(local_raw, local_xy, time_offsets[:, None, :]).flatten(2, 3)
            # Global coverage also prevents all-masked local reads from producing NaNs.
            memory = torch.cat((memory_global[:, None].expand(-1, count, -1, -1), local_memory), dim=2)
            raw = torch.cat((raw_global[:, None].expand(-1, count, -1, -1), local_raw.flatten(2, 3)), dim=2)
            valid = torch.cat((valid_global[:, None].expand(-1, count, -1), local_valid), dim=2)
            queries, brightness = block.read(queries, memory.flatten(0, 1), valid.flatten(0, 1), raw.flatten(0, 1))
            routes.append({"centers": centers, "radii": radii})
        score = self.head(queries, brightness)
        prediction = F.softplus(self.target_mean + self.target_std * score.float())
        return prediction, {"routes": routes, "brightness": brightness}

    def forward(self, images, time_offsets, mask):
        return self._forward(images, time_offsets, mask)[0]

    @torch.no_grad()
    def inspect_reads(self, images, time_offsets, mask):
        """Return prediction and small routing/value diagnostics; call eval() first."""
        return self._forward(images, time_offsets, mask)
