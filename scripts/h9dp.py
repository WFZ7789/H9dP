#!/usr/bin/env python3
"""Repository CLI for v2/P3/T1A; no editable install is needed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Spawned data workers must inherit this before NumPy/PyTorch import MKL/OpenMP.
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="HIMAWARI native500 pressure-deficit models (v2 / P3 / T1A)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
        if name == "train":
            command.add_argument("--run-dir")
            command.add_argument("--resume")
    args = parser.parse_args()
    from h9dp.common import load_config

    config = load_config(args.config)
    if args.command == "prepare":
        from h9dp.prepare import prepare

        prepare(config)
    else:
        from h9dp.engine import train

        train(config, args.run_dir, args.resume)


if __name__ == "__main__":
    main()
