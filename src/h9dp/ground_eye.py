"""Map native eye and overcast samples to local ground distances."""

import math

import numpy as np

from h9dp import cdo_ring, v1
from h9dp.d1 import locate
from h9dp.eye_cdo import crop, grids
from h9dp.hsd import Header, lonlat_to_pixel

GROUND_STEP_KM = 2.0


def pixel_jacobian(header: Header, lat: float, lon: float) -> np.ndarray:
    """Local d(row,column)/d(south,east) in pixels per ground kilometer."""
    _, _, _, _, _, _, major, minor = header.projection
    eccentricity2 = 1 - (minor / major) ** 2
    w = math.sqrt(1 - eccentricity2 * math.sin(math.radians(lat)) ** 2)
    north_radius = major * (1 - eccentricity2) / w**3
    east_radius = major * math.cos(math.radians(lat)) / w
    dlat, dlon = math.degrees(0.5 / north_radius), math.degrees(0.5 / east_radius)
    south = np.subtract(lonlat_to_pixel(header, lat - dlat, lon), lonlat_to_pixel(header, lat + dlat, lon))
    east = np.subtract(lonlat_to_pixel(header, lat, lon + dlon), lonlat_to_pixel(header, lat, lon - dlon))
    return np.column_stack((south[::-1], east[::-1]))


def spectral_strata(patch, radius):
    """Measure colocated WV spectra across B16 brightness ranks at each distance."""
    bands = patch[:4].astype(np.float32) * 0.01 + 240
    available = patch[4] > 0
    spectra = np.concatenate((bands, np.diff(bands, axis=0)))
    result = []
    for low, high in ((0, 6), (6, 14), (14, 24)):
        ok = available & (radius >= low) & (radius < high)
        values = spectra[:, ok]
        if values.shape[1] < 12:
            result.append(np.full((4, 7), np.nan))
            continue
        order = np.argsort(values[3], kind="stable")
        buckets = np.split(order, (len(order) // 6, len(order) // 2, 5 * len(order) // 6))
        result.append(np.stack([values[:, part].mean(axis=1) for part in buckets]))
    return np.stack(result).ravel()


def sequence(images, masks, centers, times, jacobian):
    eye_grid, cloud_grid = grids()
    radius = np.linalg.norm(eye_grid, axis=0)
    def ground_grid(grid):
        return np.einsum("ij,jkl->ikl", jacobian * GROUND_STEP_KM, grid)

    eye_offsets, cloud_offsets = ground_grid(eye_grid), ground_grid(cloud_grid)
    profile_offsets = ground_grid(v1.offsets())
    ring_offsets = ground_grid(cdo_ring.offsets())
    valid = masks.any(axis=(-1, -2)) & np.isfinite(times)
    usable = np.flatnonzero(valid)
    profile = np.full((len(times), 4, len(v1.RADII), v1.ANGLES), np.nan)
    eye_maps = np.zeros((len(times), 5, *eye_offsets.shape[1:]), dtype=np.int16)
    cloud_maps = np.zeros((len(times), 5, *cloud_offsets.shape[1:]), dtype=np.int16)
    features = {}
    for i in usable:
        x, y = np.rint(np.asarray(centers[i]) - 0.5).astype(int)
        center = locate(images[i], masks[i], (y, x))[0]
        eye_maps[i] = crop(images[i], masks[i], center, eye_offsets)
        cloud_maps[i] = crop(images[i], masks[i], center, cloud_offsets)
        ring_map = crop(images[i], masks[i], center, ring_offsets)
        support = ring_map[4] > 0
        cloud = np.where(support, ring_map[:4].astype(np.float32) * 0.01 + 240, np.nan)
        polar_map = crop(images[i], masks[i], center, profile_offsets)
        profile[i] = np.where(polar_map[4], polar_map[:4].astype(np.float32) * 0.01 + 240, np.nan)
        desc = cdo_ring.describe_cloud(cloud, support)
        desc["eye_spectral_strata"] = spectral_strata(eye_maps[i], radius)
        if not features:
            features = {name: np.full((len(times), values.size), np.nan, dtype=np.float32)
                        for name, values in desc.items()}
        for name, values in desc.items():
            features[name][i] = values
    if not features:
        raise ValueError("Missing ground-plane observations for every available time")
    return {**v1.describe_sequence(profile, usable, times, include_motion=False), **features,
            "eye_map": eye_maps, "cdo_map": cloud_maps}
