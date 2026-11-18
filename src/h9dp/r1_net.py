"""Small direct multispectral/temporal regressor for the R1 development route."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(8, 16, 5, 2, 2, bias=False), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1, bias=False), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, 2, 1, bias=False), nn.GroupNorm(4, 32), nn.SiLU(),
        )
        self.project = nn.Sequential(nn.Linear(32 * 6, 64), nn.LayerNorm(64), nn.SiLU())

    def forward(self, view: torch.Tensor) -> torch.Tensor:
        valid = view[:, 4:5].float() * 0.01
        band = view[:, :4].float() * (1.0 / 1600)
        spectral = (band[:, 1:] - band[:, :-1]) * (2.0 * valid)
        image = torch.cat((band * valid, spectral, valid), dim=1).contiguous(memory_format=torch.channels_last)
        x = self.body(image)
        pooled = torch.cat((x.mean(dim=(-1, -2)), x.amax(dim=(-1, -2)),
                            F.avg_pool2d(x, 4).flatten(1)), dim=-1)
        return self.project(pooled)


class R1Net(nn.Module):
    """Joint end-to-end dP prediction with optional auxiliary, non-bottleneck parameter heads."""

    def __init__(self, feature_dim: int, mode: str = "fused"):
        super().__init__()
        if mode not in {"fused", "pixels", "profile"}:
            raise ValueError(f"Unsupported R1 mode {mode}")
        self.eye = ScaleEncoder() if mode != "profile" else None
        self.environment = ScaleEncoder() if mode != "profile" else None
        self.profile = (nn.Sequential(nn.Dropout(0.13), nn.Linear(feature_dim, 96), nn.LayerNorm(96),
                                      nn.SiLU(), nn.Dropout(0.12), nn.Linear(96, 64), nn.SiLU())
                        if mode != "pixels" else None)
        width = 2 + (128 if self.eye is not None else 0) + (64 if self.profile is not None else 0)
        self.frame = nn.Sequential(nn.Linear(width, 80), nn.LayerNorm(80), nn.SiLU())
        self.attention = nn.Linear(80, 1)
        self.head = nn.Sequential(nn.Dropout(0.13), nn.Linear(80 * 3, 80), nn.LayerNorm(80),
                                  nn.SiLU(), nn.Dropout(0.15), nn.Linear(80, 6))

    def forward(self, views: torch.Tensor, profile: torch.Tensor, times: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        batch, frames = times.shape
        times = torch.where(valid, times, -1e4)
        age = times - times.amax(dim=1, keepdim=True)
        time_features = torch.stack((age.clamp(-360, 0) / 300, (age / 60).exp()), dim=-1)
        parts = [time_features]
        if self.eye is not None and self.environment is not None:
            image = views.flatten(0, 1)
            parts.extend((self.eye(image[:, :5]).reshape(batch, frames, -1),
                          self.environment(image[:, 5:]).reshape(batch, frames, -1)))
        if self.profile is not None:
            parts.append(self.profile(profile))
        x = self.frame(torch.cat(parts, dim=-1))
        latest = x.gather(1, times.argmax(dim=1)[:, None, None].expand(-1, 1, x.shape[-1])).squeeze(1)
        attention = (self.attention(x).squeeze(-1).float() + age / 70).masked_fill(~valid, -1e4)
        near = (x * attention.softmax(dim=1)[..., None].to(x.dtype)).sum(dim=1)
        history_weights = (age / 150).exp() * valid
        history_weights = history_weights / history_weights.sum(1, keepdim=True).clamp_min(1e-6)
        history = (x * history_weights[..., None].to(x.dtype)).sum(dim=1)
        return self.head(torch.cat((latest, near, near - history), dim=-1))
