"""Extract six-scan Himawari descriptors and save them outside Git."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from h9dp.a1 import sequence as field_sequence
from h9dp.cdo_ring import sequence as cdo_sequence
from h9dp.e1 import sequence as isotherm_sequence
from h9dp.eye_cdo import sequence as pixel_sequence
from h9dp.eye_sampling import sequence as sampling_sequence
from h9dp.h1 import sequence as column_sequence
from h9dp.v1 import sequence as rotation_sequence
from h9dp.w1 import sequence as radiance_sequence

SEQUENCES = {
    "rotation": rotation_sequence,
    "isotherms": isotherm_sequence,
    "field": field_sequence,
    "column": column_sequence,
    "radiance": radiance_sequence,
    "cdo": cdo_sequence,
    "sampling": sampling_sequence,
    "eye_cdo": pixel_sequence,
}


def initialize(source, mode):
    global images, masks, sequence
    sequence = SEQUENCES[mode]
    images, masks = [np.load(source / f"all_{name}.npy", mmap_mode="r") for name in ("images", "masks")]


def extract(task):
    i, centers, times = task
    return sequence(images[i], masks[i, :, 0], centers, np.array(times))


def prepare(source, output, mode="rotation"):
    raw = (source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    with ProcessPoolExecutor(8, initializer=initialize, initargs=(source, mode)) as pool:
        groups = list(
            pool.map(extract, [(i, r["centers"], r["times"]) for i, r in enumerate(rows)], chunksize=2)
        )
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            mode=mode,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            record=[r["record"] for r in rows],
            time=[r["times"] for r in rows],
            **{k: np.stack([g[k] for g in groups]) for k in groups[0]},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=SEQUENCES, default="rotation")
    a = p.parse_args()
    prepare(a.source, a.output, a.mode)
