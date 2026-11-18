"""Direct HSD v1.3 decoding, native pixel geometry and infrared calibration."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import math
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

BANDS = (8, 9, 10, 16)
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)


@dataclass(frozen=True)
class Header:
    endian: str
    satellite: str
    area: str
    band: int
    width: int
    height: int
    header_bytes: int
    data_bytes: int
    compression: int
    projection: tuple
    segment_first_line: int
    observation_start_mjd: float
    observation_end_mjd: float
    wavelength_um: float
    error_count: int
    outside_count: int
    gain: float
    intercept: float
    correction: tuple
    constants: tuple
    quality_flags: tuple

    @property
    def observation_time(self) -> datetime:
        return MJD_EPOCH + timedelta(
            days=(self.observation_start_mjd + self.observation_end_mjd) / 2
        )


def _parse_header(raw: bytes) -> Header:
    if len(raw) < 282 or raw[0] != 1 or raw[5] not in (0, 1):
        raise ValueError("Invalid HSD basic header")
    endian = "<" if raw[5] == 0 else ">"
    header_bytes, data_bytes = struct.unpack_from(endian + "II", raw, 70)
    if len(raw) < header_bytes:
        raise ValueError("Truncated HSD header")
    blocks = {}
    offset = 0
    for expected in range(1, 12):
        if offset + 3 > header_bytes or raw[offset] != expected:
            raise ValueError(f"Missing HSD header block {expected}")
        size = struct.unpack_from(endian + ("I" if expected == 10 else "H"), raw, offset + 1)[0]
        if size < 3 or offset + size > header_bytes:
            raise ValueError(f"Invalid HSD header block length: {expected}")
        blocks[expected] = raw[offset : offset + size]
        offset += size
    if offset != header_bytes:
        raise ValueError("HSD header lengths do not sum to total header length")
    bit_depth, width, height = struct.unpack_from(endian + "HHH", blocks[2], 3)
    if bit_depth != 16 or width == 0 or height == 0:
        raise ValueError("Unsupported HSD data shape/bit depth")
    cal = blocks[5]
    band = struct.unpack_from(endian + "H", cal, 3)[0]
    if band not in BANDS:
        raise ValueError(f"Expected infrared band in {BANDS}, got {band}")
    compression = blocks[2][9]
    if compression == 0 and data_bytes != width * height * 2:
        raise ValueError("HSD data length does not match image dimensions")
    return Header(
        endian=endian,
        satellite=raw[6:22].rstrip(b"\x00 ").decode("ascii"),
        area=raw[38:42].decode("ascii"),
        band=band,
        width=width,
        height=height,
        header_bytes=header_bytes,
        data_bytes=data_bytes,
        compression=compression,
        projection=struct.unpack_from(endian + "dIIffddd", blocks[3], 3),
        segment_first_line=struct.unpack_from(endian + "H", blocks[7], 5)[0],
        observation_start_mjd=struct.unpack_from(endian + "d", raw, 46)[0],
        observation_end_mjd=struct.unpack_from(endian + "d", raw, 54)[0],
        wavelength_um=struct.unpack_from(endian + "d", cal, 5)[0],
        error_count=struct.unpack_from(endian + "H", cal, 15)[0],
        outside_count=struct.unpack_from(endian + "H", cal, 17)[0],
        gain=struct.unpack_from(endian + "d", cal, 19)[0],
        intercept=struct.unpack_from(endian + "d", cal, 27)[0],
        correction=struct.unpack_from(endian + "ddd", cal, 35),
        constants=struct.unpack_from(endian + "ddd", cal, 83),
        quality_flags=tuple(raw[78:82]),
    )


def read_header(path: str | Path) -> Header:
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        prefix = stream.read(282)
        if len(prefix) != 282 or prefix[5] not in (0, 1):
            raise ValueError(f"Invalid/truncated HSD: {path}")
        total = struct.unpack_from(("<" if prefix[5] == 0 else ">") + "I", prefix, 70)[0]
        if total < 282 or total > 10_000_000:
            raise ValueError(f"Invalid HSD header length: {path}")
        header = _parse_header(prefix + stream.read(total - 282))
    return header


def read_counts(path: str | Path) -> tuple[Header, np.ndarray, str]:
    path = Path(path)
    original = path.read_bytes()
    checksum = hashlib.sha256(original).hexdigest()
    raw = bz2.decompress(original) if path.suffix == ".bz2" else gzip.decompress(original) if path.suffix == ".gz" else original
    header = _parse_header(raw)
    if len(raw) != header.header_bytes + header.data_bytes:
        raise ValueError(f"HSD file size does not match header: {path}")
    payload = raw[header.header_bytes :]
    if header.compression == 1:
        payload = gzip.decompress(payload)
    elif header.compression == 2:
        payload = bz2.decompress(payload)
    elif header.compression != 0:
        raise ValueError(f"Unknown HSD data compression: {header.compression}")
    if len(payload) != header.width * header.height * 2:
        raise ValueError(f"Truncated image payload: {path}")
    counts = np.frombuffer(payload, dtype=header.endian + "u2").reshape(header.height, header.width)
    return header, counts, checksum


def lonlat_to_pixel(header: Header, latitude: float, longitude: float) -> tuple[float, float]:
    """Return zero-based local (column, row), respecting segment first line."""
    lon0, cfac, lfac, coff, loff, distance, req, rpol = header.projection
    lat = math.radians(latitude)
    lon = math.radians(longitude - lon0)
    phi = math.atan((rpol / req) ** 2 * math.tan(lat))
    radius = rpol / math.sqrt(1 - (1 - (rpol / req) ** 2) * math.cos(phi) ** 2)
    r1 = distance - radius * math.cos(phi) * math.cos(lon)
    r2 = -radius * math.cos(phi) * math.sin(lon)
    r3 = radius * math.sin(phi)
    x = math.degrees(math.atan2(-r2, r1))
    y = math.degrees(math.asin(-r3 / math.sqrt(r1 * r1 + r2 * r2 + r3 * r3)))
    return coff + x * cfac / 65536 - 1, loff + y * lfac / 65536 - header.segment_first_line


def crop_window(header: Header, latitude: float, longitude: float, size: int = 500) -> tuple[int, int]:
    col, row = lonlat_to_pixel(header, latitude, longitude)
    if header.width < size or header.height < size:
        raise ValueError("Downloaded segment is smaller than requested crop")
    if (header.height, header.width) == (size, size):
        x0, y0 = 0, 0
    else:
        x0 = max(0, min(math.floor(col + 0.5) - size // 2, header.width - size))
        y0 = max(0, min(math.floor(row + 0.5) - size // 2, header.height - size))
    if not (0 <= col - x0 < size and 0 <= row - y0 < size):
        raise ValueError("Label coordinate does not lie in the downloaded/cropped block")
    return x0, y0


def earth_mask(header: Header, window: tuple[int, int], size: int) -> np.ndarray:
    x0, y0 = window
    _, cfac, lfac, coff, loff, distance, req, rpol = header.projection
    x = np.deg2rad((np.arange(x0, x0 + size, dtype=np.float64) + 1 - coff) * 65536 / cfac)[None, :]
    y = np.deg2rad((np.arange(y0, y0 + size, dtype=np.float64) + header.segment_first_line - loff) * 65536 / lfac)[:, None]
    coefficient = np.cos(y) ** 2 + (req / rpol) ** 2 * np.sin(y) ** 2
    discriminant = (distance * np.cos(x) * np.cos(y)) ** 2 - coefficient * (distance**2 - req**2)
    return discriminant >= 0


def brightness_temperature(counts: np.ndarray, header: Header, on_earth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """JMA Planck conversion in float64; result in Kelvin float32."""
    radiance = counts.astype(np.float64) * header.gain + header.intercept
    valid = on_earth & (counts != header.error_count) & (counts != header.outside_count) & (radiance > 0)
    result = np.zeros(counts.shape, dtype=np.float32)
    if not valid.any():
        return result, valid
    speed, planck, boltzmann = header.constants
    wavelength = header.wavelength_um * 1e-6
    # The header radiance unit is per micrometre; Planck below uses per metre.
    radiance_si = radiance[valid] * 1e6
    effective = (planck * speed / (boltzmann * wavelength)) / np.log1p(
        (2 * planck * speed**2 / wavelength**5) / radiance_si
    )
    c0, c1, c2 = header.correction
    temperature = c0 + c1 * effective + c2 * effective**2
    result[valid] = temperature.astype(np.float32)
    valid &= np.isfinite(result) & (result > 0)
    result[~valid] = 0
    return result, valid
