"""Build pressure-independent D1 descriptors from the native-resolution record cache."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from h9dp.d1 import frame_signature
from h9dp.r1 import summarize_time


def initialize(source):
    global images, masks
    images, masks = [np.load(source / f"all_{name}.npy", mmap_mode="r") for name in ("images", "masks")]


def extract(task):
    i, xy, times = task
    valid = np.isfinite(times) & masks[i, :, 0].any(axis=(1, 2))
    parts = [frame_signature(images[i, j], masks[i, j, 0], xy[j]) for j in np.flatnonzero(valid)]
    result = []
    for values in zip(*parts, strict=True):
        frames = np.full((len(times), values[0].size), np.nan)
        frames[valid] = values
        stats = summarize_time(frames[None], np.array(times)[None], valid[None])
        result.append(np.stack([stats[s][0] for s in ("recent", "latest", "trend")]).astype(np.float32))
    return result, valid


def prepare(source, output, workers):
    if output.exists():
        raise FileExistsError(output)
    raw = (source / "all_meta.json").read_bytes()
    records = json.loads(raw)["records"]
    tasks = [(i, row["centers"], row["times"]) for i, row in enumerate(records)]
    with ProcessPoolExecutor(workers, initializer=initialize, initargs=(source,)) as pool:
        result = list(pool.map(extract, tasks, chunksize=2))
    rows, valid = zip(*result, strict=True)
    groups = dict(
        zip(
            ("core", "cloud", "contrast", "scene"),
            (np.stack(v) for v in zip(*rows, strict=True)),
            strict=True,
        )
    )
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            record=[r["record"] for r in records],
            valid=valid,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            **groups,
        )
    print(f"records={len(rows)} frames={np.sum(valid)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    prepare(a.source, a.output, a.workers)
