"""Prepare earth-distance eye and CDO descriptors from the native six-scan cache."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from h9dp.ground_eye import pixel_jacobian, sequence
from h9dp.hsd import read_header
from h9dp.pressure_labels import read_catalog, verify_catalog_labels


def open_cache(source):
    global images, masks
    images = np.load(source / "all_images.npy", mmap_mode="r")
    masks = np.load(source / "all_masks.npy", mmap_mode="r")


def extract(task):
    index, centers, times, jacobian = task
    return sequence(images[index], masks[index, :, 0], centers, np.asarray(times), jacobian)


def prepare(source, reference_path, output):
    start = time.monotonic()
    raw = (source / "all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    catalog, digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    header = read_header(reference_path)
    jacobians = np.stack([pixel_jacobian(header, **{k: catalog[row["record"]]["location"][k]
                                                  for k in ("lat", "lon")}) for row in rows])
    with ProcessPoolExecutor(8, initializer=open_cache, initargs=(source,)) as pool:
        tasks = [(i, row["centers"], row["times"], jacobians[i]) for i, row in enumerate(rows)]
        groups = list(pool.map(extract, tasks, chunksize=4))
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, mode="ground_eye", source_sha256=hashlib.sha256(raw).hexdigest(), catalog_sha256=digest,
            pixel_jacobian=jacobians, reference=str(reference_path.resolve()), projection=header.projection,
            record=[r["record"] for r in rows], time=[r["times"] for r in rows],
            **{name: np.stack([g[name] for g in groups]) for name in groups[0]},
        )
    print(json.dumps({"records": len(rows), "elapsed_s": time.monotonic() - start,
                      "projection": header.projection}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "reference", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.reference, args.output)
