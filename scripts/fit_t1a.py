#!/usr/bin/env python3
"""Fit T1A (test-1 a) to exactly one fixed set of 24 real training records.

No augmentation, weight decay, validation, or formal full-data training.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "t1a.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--rmse-target", type=float, default=1.0)
    args = parser.parse_args()

    from h9dp.common import load_config
    from h9dp.t1a_fit import run_fit

    config = load_config(args.config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (args.run_dir or Path(config["output_dir"]) / f"fit24_{stamp}").resolve()
    result = run_fit(config, directory, count=24, max_steps=args.max_steps,
                     batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
                     evaluate_every=args.evaluate_every, rmse_target=args.rmse_target)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
