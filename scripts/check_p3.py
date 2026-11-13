#!/usr/bin/env python3
"""Synthetic forward-only P3 check: no dataset access, fitting or output files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "p3_minimal.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    import torch

    from h9dp.common import load_config
    from h9dp.model import build_model, model_architecture
    from h9dp.p3 import ARCHITECTURE, PARAMETER_LIMIT, P3PressureRegressor

    config = load_config(args.config)
    if model_architecture(config["model"]) != ARCHITECTURE:
        raise ValueError("This synthetic check requires a P3 configuration")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    model = build_model(config["model"]).to(device).eval()
    assert isinstance(model, P3PressureRegressor)
    size = config["data"]["crop_size"]
    images = torch.randn(1, 6, 4, size, size, device=device)
    mask = torch.ones(1, 6, 1, size, size, dtype=torch.bool, device=device)
    times = torch.tensor([[-300.2, -180.2, -60.2, -30.2, -10.2, -0.2]], device=device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
        prediction, details = model.inspect_reads(images, times, mask)
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Non-finite synthetic prediction")
    result = {
        "synthetic_only": True,
        "architecture": ARCHITECTURE,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_limit_exclusive": PARAMETER_LIMIT,
        "device": str(device),
        "bf16_autocast": args.bf16,
        "input_shape": list(images.shape),
        "output_shape": list(prediction.shape),
        "finite_output": True,
        "routes": [{"center_shape": list(route["centers"].shape),
                    "radius_min": route["radii"].min().item(),
                    "radius_max": route["radii"].max().item()} for route in details["routes"]],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
