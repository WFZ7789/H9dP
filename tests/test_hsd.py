import struct
from dataclasses import replace

import numpy as np
import pytest

from h9dp.hsd import (
    brightness_temperature,
    crop_window,
    earth_mask,
    lonlat_to_pixel,
    pixel_latitude,
    read_counts,
    read_header,
)


def synthetic_hsd(path, endian, counts=None, *, band=8, start=60000.0):
    if counts is None:
        counts = np.full((8, 8), 1000)
        counts[0, :3] = [65535, 65534, 0]
    counts = counts.astype(endian + "u2")
    height, width = counts.shape
    sizes = [282, 50, 127, 139, 147, 259, 47, 81, 75, 47, 259]
    blocks = [bytearray(length) for length in sizes]
    for number, block in enumerate(blocks, 1):
        block[0] = number
        struct.pack_into(endian + ("I" if number == 10 else "H"), block, 1, len(block))
    basic, data, projection, _, cal, _, segment, *_ = blocks
    struct.pack_into(endian + "H", basic, 3, 11)
    basic[5] = 0 if endian == "<" else 1
    basic[6:16] = b"Himawari-9"
    basic[38:42] = b"R304"
    struct.pack_into(endian + "ddd", basic, 46, start, start + 0.001, start + 0.002)
    struct.pack_into(endian + "II", basic, 70, sum(sizes), counts.nbytes)
    struct.pack_into(endian + "HHH", data, 3, 16, width, height)
    struct.pack_into(endian + "dIIffddd", projection, 3, 140.7, 20466275, 20466275,
                     (width + 1) / 2, (height + 1) / 2, 42164.0, 6378.137, 6356.7523)
    struct.pack_into(endian + "H", cal, 3, band)
    struct.pack_into(endian + "d", cal, 5, 10.0)
    struct.pack_into(endian + "HHH", cal, 13, 12, 65535, 65534)
    struct.pack_into(endian + "dd", cal, 19, 0.01, 0.0)
    struct.pack_into(endian + "ddd", cal, 35, 0.0, 1.0, 0.0)
    struct.pack_into(endian + "ddd", cal, 83, 299792458.0, 6.62607015e-34, 1.380649e-23)
    segment[3:5] = bytes([1, 1])
    struct.pack_into(endian + "H", segment, 5, 1)
    path.write_bytes(b"".join(blocks) + counts.tobytes())
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
def test_hsd_decode_and_known_planck_temperature(tmp_path, endian):
    path = synthetic_hsd(tmp_path / "sample.DAT", endian)
    header, counts, checksum = read_counts(path)
    assert header == read_header(path)
    assert header.band == 8 and counts.shape == (8, 8) and len(checksum) == 64
    mask = earth_mask(header, (0, 0), 8)
    assert mask.all()
    temperature, valid = brightness_temperature(counts, header, mask)
    # Fixed scalar Planck reference: 10 W/(m² sr µm), lambda=10 µm.
    assert temperature[1, 1] == pytest.approx(300.473799917899, abs=0.0001)
    assert not valid[0, :3].any()
    assert np.all(temperature[0, :3] == 0)
    outside = np.ones((8, 8), dtype=bool)
    outside[1, 1] = False
    temperature, valid = brightness_temperature(counts, header, outside)
    assert not valid[1, 1] and temperature[1, 1] == 0
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="size"):
        read_counts(path)


def test_segment_geometry_and_edge_crop(tmp_path):
    header = read_header(synthetic_hsd(tmp_path / "sample.DAT", "<"))
    col, row = lonlat_to_pixel(header, 0, 140.7)
    assert col == pytest.approx(3.5)
    assert row == pytest.approx(3.5)
    header = replace(header, width=5500, height=550, segment_first_line=2201, projection=(140.7, 20466275, 20466275, 2750.5, 2750.5, 42164., 6378.137, 6356.7523))
    assert crop_window(header, 0, 140.7) == (2500, 50)
    assert not earth_mask(header, (0, 0), 8).any()


def test_pixel_latitude_inverts_segmented_projection_and_masks_space(tmp_path):
    h = read_header(synthetic_hsd(tmp_path / "sample.DAT", "<"))
    h = replace(h, projection=(*h.projection[:3], 2750.5, 2750.5, *h.projection[5:]), segment_first_line=1001)
    expected = np.array([0.0, 28.5, -22.5, 41.5])
    points = [lonlat_to_pixel(h, lat, lon) for lat, lon in zip(expected, [140.7, 160.0, 125.0, 132.0], strict=True)]
    columns, rows = np.array(points).T
    np.testing.assert_allclose(pixel_latitude(h, columns, rows), expected, atol=1e-10)
    col = np.array([0.0, 2749.5])[None, :]
    row = np.array([-1000.0, 1749.5])[:, None]
    lat = pixel_latitude(h, col, row)
    assert np.isnan(lat[0, 0]) and lat[1, 1] == pytest.approx(0.0)

