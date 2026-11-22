"""Native eye patches and polar CDO maps for a compact two-branch model."""

from functools import lru_cache

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch import nn
from torch.nn import functional as F

from h9dp.d1 import locate

EYE_SIZE = 49
CDO_ANGLES = 48
CDO_RADII = 48


@lru_cache(maxsize=1)
def grids():
    axis = np.arange(EYE_SIZE, dtype=float) - EYE_SIZE // 2
    dy, dx = np.meshgrid(axis, axis, indexing="ij")
    angle = np.arange(CDO_ANGLES) * (2 * np.pi / CDO_ANGLES)
    radius = 5 + 2 * np.arange(CDO_RADII)
    return np.stack((dy, dx)), np.stack((np.sin(angle)[:, None] * radius, np.cos(angle)[:, None] * radius))


def crop(image, mask, center, offsets):
    coordinates = np.asarray(center)[:, None, None] + offsets
    ok = map_coordinates(mask.astype(np.float32), coordinates, order=1) > 0.999
    bands = np.stack([map_coordinates(v, coordinates, order=1, output=np.float32) for v in image])
    # Image cache units are centikelvin above 150 K. Save offsets from 240 K.
    bt = np.where(ok, np.rint(bands) - 9000, 0).astype(np.int16)
    return np.concatenate((bt, ok[None].astype(np.int16)))


def sequence(images, masks, centers, times):
    valid = masks.any((-1, -2)) & np.isfinite(times)
    eye_grid, cdo_grid = grids()
    eye = np.zeros((len(times), 5, EYE_SIZE, EYE_SIZE), dtype=np.int16)
    cloud = np.zeros((len(times), 5, CDO_ANGLES, CDO_RADII), dtype=np.int16)
    for i in np.flatnonzero(valid):
        x, y = np.rint(np.asarray(centers[i]) - 0.5).astype(int)
        center = locate(images[i], masks[i], (y, x))[0]
        eye[i] = crop(images[i], masks[i], center, eye_grid)
        cloud[i] = crop(images[i], masks[i], center, cdo_grid)
    return {"eye_map": eye, "cdo_map": cloud}


class Encoder(nn.Module):
    def __init__(self, polar=False):
        super().__init__()
        self.polar = polar
        self.filter = nn.Conv2d(5, 8, kernel_size=3, padding=0 if polar else 1)
        self.context = nn.Conv2d(8, 10, kernel_size=3, stride=2, padding=1)
        self.projection = nn.Linear(60 if polar else 90, 20)

    def forward(self, frames):
        n, count, features, height, width = frames.shape
        image = frames.reshape(n * count, features, height, width)
        if self.polar:
            image = F.pad(torch.cat((image[:, :, -1:], image, image[:, :, :1]), dim=2), (1, 1))
        x = F.silu(self.filter(image))
        x = F.silu(self.context(x))
        shape = (1, 3) if self.polar else (2, 3)
        averages = F.adaptive_avg_pool2d(x, shape).flatten(1)
        if self.polar:
            maxes = F.adaptive_max_pool2d(x, shape).flatten(1)
            codes = torch.cat((averages, maxes), dim=1)
        else:
            maxes = F.adaptive_max_pool2d(x, (1, 3)).flatten(1)
            codes = torch.cat((averages, maxes), dim=1)
        return torch.tanh(self.projection(codes)).reshape(n, count, -1)


class EyeCDO(nn.Module):
    """Learn native-pixel and cloud-cover descriptors plus heteroscedastic CDO uncertainty."""

    def __init__(self, pressure_components, cdo_components):
        super().__init__()
        self.eye = Encoder()
        self.cloud = Encoder(polar=True)
        self.noise = nn.Linear(40, 1)
        self.pressure_weight = nn.Parameter(torch.full((pressure_components,), -2.9, dtype=torch.float64))
        self.cloud_weight = nn.Parameter(torch.full((cdo_components,), -2.5, dtype=torch.float64))
        self.amplitude = nn.Parameter(torch.tensor([-1.0, -1.9, -2.7, -0.9], dtype=torch.float64))
        self.pressure_noise = nn.Parameter(torch.tensor([-2.8, -2.1, -1.5], dtype=torch.float64))

    @staticmethod
    def summarize(code, weight):
        return torch.einsum("ntc,nkt->nkc", code, weight.to(code.dtype)).flatten(1)

    @staticmethod
    def affinity(code):
        dot = code @ code.T / code.size(-1)
        norm = dot.diag()
        distance = (norm[:, None] + norm[None, :] - 2 * dot).clamp_min(0)
        # Spread in learned normalized codes is much smaller than in raw brightness.
        return dot + (-distance / 0.14).exp()

    def forward(self, eye, cloud, weights, pressure, cdo):
        e = self.summarize(self.eye(eye), weights)
        c = self.summarize(self.cloud(cloud), weights)
        c_noise = self.noise(c).flatten().double().clamp(-5.0, 2.5)
        e_gram, c_gram = self.affinity(e.double()), self.affinity(c.double())
        scales = self.amplitude.exp()
        k = torch.einsum("g,gij->ij", self.pressure_weight.exp(), pressure) + 8
        k = k + scales[0] * e_gram + scales[1] * c_gram + scales[2] * e_gram * c_gram
        z = torch.einsum("g,gij->ij", self.cloud_weight.exp(), cdo) + 8 + scales[3] * c_gram
        return k, z, 0.032 * c_noise.exp()


def posterior_loss(gram, obs_noise, target, importance):
    chol, _ = torch.linalg.cholesky_ex(gram + torch.diag(obs_noise))
    inverse = torch.cholesky_inverse(chol)
    dual = inverse @ target
    loo = dual / inverse.diag()
    nll = (target @ dual + 2 * chol.diag().log().sum()) / len(target)
    return (loo.square() * importance).mean(), nll
