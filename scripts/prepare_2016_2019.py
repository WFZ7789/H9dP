"""Prepare 2016-2019 Himawari-8 full-disk 1000x1000 crops.

Every byte is fetched with the AWS CLI, and every child process runs without
proxy environment variables.  Each task downloads one band file to a scratch
path, decompresses it in memory, crops the 1000x1000 window centred on the
record position, converts the window to brightness temperature, writes the
crop, and deletes the scratch file.  No full-disk intermediate is kept.

Commands run from the repository root with the fy3-aivis-v1 environment:

    PYTHONPATH=src:. python -m scripts.prepare_2016_2019 --records 28
    PYTHONPATH=src:. python -m scripts.prepare_2016_2019 --workers 64
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from h9dp.hsd import (
    Header,
    brightness_temperature,
    crop_window,
    earth_mask,
    lonlat_to_pixel,
    read_header,
)
from h9dp.pressure_labels import read_catalog

BANDS = (8, 9, 10, 16)
FRAME_OFFSETS_MINUTES = (-300, -180, -60, -30, -10, 0)
SCHEMA = "himawari-1000-2016-2019/v1"
BUCKET = "noaa-himawari8"
REGION = "us-east-1"
FIRST_YEAR, LAST_YEAR = 2016, 2019


def child_environment() -> dict[str, str]:
    """Environment for AWS CLI children with every proxy variable removed."""
    return {name: value for name, value in os.environ.items() if "proxy" not in name.lower()}


def key_for(nominal: datetime, band: int) -> str:
    stamp = nominal.strftime("%Y%m%d_%H%M")
    return (
        f"AHI-L1b-FLDK/{nominal:%Y/%m/%d/%H%M}/"
        f"HS_H08_{stamp}_B{band:02d}_FLDK_R20_S0101.DAT.bz2"
    )


def download(key: str, destination: Path, attempts: int = 4) -> Path:
    command = [
        "aws", "s3", "cp", "--no-sign-request", "--region", REGION,
        f"s3://{BUCKET}/{key}", str(destination), "--only-show-errors",
    ]
    for attempt in range(attempts):
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=1800,
                           env=child_environment())
            return destination
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if attempt + 1 == attempts:
                detail = getattr(error, "stderr", b"") or b""
                raise RuntimeError(f"aws s3 cp failed for {key}: {detail.decode(errors='replace').strip()}") from error
            time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


def probe_header(key: str, destination: Path) -> Header:
    command = [
        "aws", "s3api", "get-object", "--no-sign-request", "--region", REGION,
        "--bucket", BUCKET, "--key", key, "--range", "bytes=0-262143", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=600, env=child_environment())
    try:
        return read_header(destination)
    except ValueError as error:
        raise ValueError(f"Cannot decode HSD header: {key}") from error


def nominal_times(target: datetime) -> list[datetime]:
    times = []
    for offset in FRAME_OFFSETS_MINUTES:
        moment = target + timedelta(minutes=offset)
        times.append(moment.replace(minute=moment.minute - moment.minute % 10, second=0, microsecond=0))
    return times


def array_paths(directory: Path, position: int) -> tuple[Path, Path]:
    return directory / "brightness" / f"{position:04d}.npy", directory / "validity" / f"{position:04d}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/H9dP/datasets/himawari_1000_2016_2019"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--records", type=str, help="Optional comma-separated MOTIS record indices")
    args = parser.parse_args()

    removed = sorted(name for name in os.environ if "proxy" in name.lower())
    print(f"Proxy variables removed from AWS CLI children: {removed or 'none'}", flush=True)

    labels, _ = read_catalog(args.labels)
    selected = [
        (index, record)
        for index, record in enumerate(labels)
        if FIRST_YEAR <= record["time"]["year"] <= LAST_YEAR
    ]
    if args.records:
        wanted = {int(value) for value in args.records.split(",")}
        selected = [(index, record) for index, record in selected if index in wanted]
    if not selected:
        raise SystemExit("No records selected")
    frames_per_record = len(FRAME_OFFSETS_MINUTES)
    tasks_per_record = frames_per_record * len(BANDS)
    print(f"{len(selected)} records, {len(selected) * tasks_per_record} band tasks", flush=True)

    output = args.output
    for name in ("brightness", "validity", "done", "scratch"):
        (output / name).mkdir(parents=True, exist_ok=True)

    probe_key = key_for(datetime(2016, 10, 3, 4, 40, tzinfo=UTC), 8)
    reference = probe_header(probe_key, output / "scratch" / "probe.bz2")
    (output / "scratch" / "probe.bz2").unlink(missing_ok=True)
    full_disk = replace(reference, height=5500, segment_first_line=1)
    if (full_disk.height, full_disk.width) != (5500, 5500):
        raise SystemExit(f"Unexpected full-disk geometry: {full_disk.height}x{full_disk.width}")
    print(f"Full-disk reference: {full_disk.height}x{full_disk.width} projection={full_disk.projection}", flush=True)

    entries = []
    for position, (index, record) in enumerate(selected):
        target = datetime.fromisoformat(record["time"]["iso_utc"])
        window = crop_window(full_disk, record["location"]["lat"], record["location"]["lon"], args.size)
        column, row = lonlat_to_pixel(full_disk, record["location"]["lat"], record["location"]["lon"])
        frames = [
            {
                "offset_minutes": offset,
                "nominal": nominal.isoformat(),
                "keys": {band: key_for(nominal, band) for band in BANDS},
            }
            for offset, nominal in zip(FRAME_OFFSETS_MINUTES, nominal_times(target), strict=True)
        ]
        entries.append({
            "position": position, "record": index, "name": record["name"],
            "target_time": record["time"]["iso_utc"], "lat": record["location"]["lat"],
            "lon": record["location"]["lon"], "dP": record["dP"], "quality": record["quality"],
            "window": list(window), "center": [column - window[0], row - window[1]], "frames": frames,
        })

    index_path = output / "index.json"
    index = {
        "schema": SCHEMA, "crop_size": args.size, "bands": list(BANDS),
        "frame_offsets_minutes": list(FRAME_OFFSETS_MINUTES),
        "source": "NOAA AHI-L1b-FLDK R20 S0101 single full-disk segment, fetched with AWS CLI without proxy",
        "satellite": "H08", "records": entries,
        "arrays": {"brightness": "brightness/{position:04d}.npy", "validity": "validity/{position:04d}.npy"},
        "status": "preparing",
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")

    pending = [entry for entry in entries if not (output / "done" / f"{entry['position']:04d}").exists()]
    print(f"{len(entries) - len(pending)} records already complete, {len(pending)} pending", flush=True)

    tasks = [
        (entry["position"], frame, band)
        for entry in pending
        for frame in range(frames_per_record)
        for band in BANDS
    ]
    for entry in pending:
        brightness, validity = array_paths(output, entry["position"])
        for path, dtype, shape in (
            (brightness, np.float32, (frames_per_record, len(BANDS), args.size, args.size)),
            (validity, np.uint8, (frames_per_record, len(BANDS), args.size * args.size // 8)),
        ):
            if path.exists():
                continue
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            array.flush()
            del array

    lock = threading.Lock()
    counters = Counter()
    failures: list[dict] = []
    remaining = Counter({entry["position"]: tasks_per_record for entry in pending})
    started = time.monotonic()

    def work(task: tuple[int, int, int]) -> None:
        position, frame, band = task
        entry = entries[position]
        key = entry["frames"][frame]["keys"][band]
        scratch = output / "scratch" / f"{position:04d}_{frame:02d}_{band:02d}.bz2"
        try:
            download(key, scratch)
            counters["bytes"] += scratch.stat().st_size
            header = read_header(scratch)
            payload = bz2.decompress(scratch.read_bytes())
            scratch.unlink(missing_ok=True)
            if (header.height, header.width) != (5500, 5500):
                raise ValueError(f"Unexpected geometry {header.height}x{header.width}")
            pixels = header.height * header.width
            counts = np.frombuffer(payload, dtype=header.endian + "u2", count=pixels,
                                   offset=header.header_bytes).reshape(header.height, header.width)
            x0, y0 = entry["window"]
            window = counts[y0:y0 + args.size, x0:x0 + args.size]
            mask = earth_mask(header, (x0, y0), args.size)
            temperature, valid = brightness_temperature(window, header, mask)
            brightness, validity = array_paths(output, position)
            band_index = BANDS.index(band)
            image = np.lib.format.open_memmap(brightness, mode="r+")
            image[frame, band_index] = temperature
            image.flush()
            del image
            packed = np.lib.format.open_memmap(validity, mode="r+")
            packed[frame, band_index] = np.packbits(valid.reshape(-1), bitorder="little")
            packed.flush()
            del packed
            del payload, counts, window, temperature, valid
            with lock:
                remaining[position] -= 1
                counters["files"] += 1
                if remaining[position] == 0:
                    (output / "done" / f"{position:04d}").write_text("")
                    counters["records"] += 1
                done = counters["files"]
                if done % 25 == 0 or done == len(tasks):
                    elapsed = time.monotonic() - started
                    rate = counters["bytes"] / 1e6 / elapsed if elapsed else 0
                    eta = (len(tasks) - done) * elapsed / done / 60 if done else 0
                    print(f"{done}/{len(tasks)} files, {counters['records']}/{len(pending)} records, "
                          f"{counters['bytes'] / 1e9:.1f} GB, {rate:.2f} MB/s, ETA {eta:.0f} min", flush=True)
        except Exception as error:  # noqa: BLE001 - recorded and reported, not swallowed silently
            scratch.unlink(missing_ok=True)
            with lock:
                failures.append({"position": position, "record": entry["record"], "frame": frame,
                                 "band": band, "key": key, "error": f"{type(error).__name__}: {error}"})
                counters["failed"] += 1
                print(f"FAILED {entry['record']} frame {frame} B{band:02d}: {type(error).__name__}: {error}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, tasks))

    index["status"] = "complete" if not failures else "complete-with-failures"
    index["summary"] = {
        "records": len(entries), "files": len(tasks), "failed": len(failures),
        "bytes": counters["bytes"], "elapsed_seconds": time.monotonic() - started,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n")
    print(f"Done: {counters['records']}/{len(pending)} pending records, {counters['failed']} failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
