"""Synthetic tests of the C++20 DAT and posterior paths; no local archive required."""

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_hsd import synthetic_hsd

from h9dp.hsd import BANDS
from scripts.check_v1_inference import _descriptors
from scripts.export_v1_inference import GROUPS, pack, write_model
from scripts.v1 import bank


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    if not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("C++20 compiler and CMake needed")
    src = Path(__file__).resolve().parents[1] / "inference"
    build = tmp_path_factory.mktemp("v1_cpp")
    config = subprocess.run(
        ["cmake", "-S", src, "-B", build, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={sys.prefix}"],
        check=False, capture_output=True, text=True,
    )
    if config.returncode and "Could NOT find ZLIB" in config.stderr:
        pytest.skip("zlib development files needed")
    assert config.returncode == 0, config.stdout + config.stderr
    subprocess.run(["cmake", "--build", build, "-j2"], check=True)
    return build / "inference_best_cli"


def _model(path, groups, times):
    train, held = np.arange(6), np.arange(6, len(times))
    kernels, names = bank(groups, times, train, held, device="cpu")
    masses = np.linspace(0.002, 0.01, len(names))
    dual = np.linspace(-0.14, 0.16, len(train))
    mean, std = 62., 20.
    write_model(path, {g: x[train] for g, x in groups.items()}, times[train], names, masses, dual, mean, std)
    gram = np.einsum("g,gij->ij", masses, kernels.numpy()) + 8
    return gram[len(train):, :len(train)] @ dual * std + mean


def test_cpp_posterior_parity_and_truncated_model(cli, tmp_path):
    rng = np.random.default_rng(1)
    groups = {g: rng.normal(size=(8, 6, d)).astype(np.float32) for g, d in GROUPS.items()}
    times = np.tile([-300., -180., -60., -30., -10., 0.], (8, 1))
    for values in groups.values():
        values[:, :, 0] = np.nan  # A column discarded by training normalization.
        values[-1, 2] = np.nan
    groups["motion"][:] = np.nan  # An entire group may have no supported statistics.
    times[-1, 2] = np.nan
    model, request = tmp_path / "fold.v1m", tmp_path / "held.v1q"
    expected = _model(model, groups, times)
    pack(request, {g: x[6:] for g, x in groups.items()}, times[6:])
    result = subprocess.check_output([cli, "--features", request, model], text=True)
    np.testing.assert_allclose(np.fromstring(result, sep=" "), expected, atol=1e-5, rtol=0)
    model.write_bytes(model.read_bytes()[:-1])
    failed = subprocess.run([cli, "--features", request, model], capture_output=True, text=True)
    assert failed.returncode != 0 and "Truncated V1 file" in failed.stderr


def test_cpp_dat_end_to_end(cli, tmp_path):
    y, x = np.indices((500, 500), dtype=float) - 254
    radius, azimuth = np.hypot(x, y), np.arctan2(y, x)
    scans = []
    for i, minutes in enumerate([0, 75, 148, 220, 240, 250]):
        files = []
        for j, band in enumerate(BANDS):
            t = azimuth + 0.23 * i + 0.13 * j
            counts = (480 + 180 * np.exp(-radius / 30) + 35 * np.sin(radius / 5)
                      + 33 * np.cos(t) + 22 * np.sin(3 * t) + 10 * j + i).astype(np.int32)
            counts[:80, -155:] = 65535
            path = synthetic_hsd(tmp_path / f"{i}_B{band:02}.DAT", "<" if j % 2 == 0 else ">", counts,
                                 band=band, start=60000 + minutes / 1440)
            if (i, j) == (0, 0):
                compressed = path.with_suffix(".DAT.gz")
                compressed.write_bytes(gzip.compress(path.read_bytes()))
                path = compressed
            files.append(path)
        scans.append(files)
    query = _descriptors(scans, 0., 140.7)
    rng = np.random.default_rng(1)
    groups = {g: np.concatenate((x + rng.normal(0, 0.7, (6, *x.shape)).astype(np.float32), x[None]))
              for g, x in query.items()}
    times = np.tile([0., 75., 148., 220., 240., 250.], (7, 1))
    model = tmp_path / "model.v1m"
    expected, = _model(model, groups, times)
    # The public library path, as exercised by the CLI, sorts scans and bands.
    data = [p for scan in reversed(scans) for p in reversed(scan)]
    result = subprocess.check_output([cli, "0", "140.7", model, "--", *data], text=True)
    assert float(result) == pytest.approx(expected, abs=1e-4)
    data[0] = data[1]
    failed = subprocess.run([cli, "0", "140.7", model, "--", *data], capture_output=True, text=True)
    assert failed.returncode != 0 and "Duplicate HSD band" in failed.stderr
