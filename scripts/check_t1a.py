#!/usr/bin/env python3
"""Synthetic T1A forward/backward and optional supervised mechanism checks.

Does not read real data, launch full training, or write files. Mechanism fits
use fresh reduced-width models, never the model used for the resource check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, str(ROOT / "src"))


def synthetic_task(kind, device):
    import torch

    size = 32
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    radius = ((x - 15.5).square() + (y - 15.5).square()).sqrt()
    ring = (radius >= 6) & (radius < 9)
    broken = ring.clone()
    removed = ring & (y < 11)
    broken[removed] = False
    destinations = torch.nonzero((radius > 11) & (y > 20))[:torch.nonzero(removed).shape[0]]
    broken[destinations[:, 0], destinations[:, 1]] = True
    assert ring.sum() == broken.sum()
    examples, targets = [], []
    for variant in range(4):
        for category in range(2):
            if kind == "spatial":
                pattern = (ring if category else broken).float()
                pattern = torch.rot90(pattern, variant, dims=(-2, -1))
                pattern = torch.roll(pattern, (variant % 2, -(variant // 2)), dims=(-2, -1))
                pattern = (pattern * 2 - 0.5) * (0.8 + 0.1 * variant)
                frames = pattern[None].expand(6, -1, -1)
            else:
                # Identical frame multiset and final frame, opposite history order.
                sequence = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 0.0])
                if not category:
                    sequence = sequence[[4, 3, 2, 1, 0, 5]]
                frames = sequence[:, None, None] * (ring.float()[None] + 0.5)
                frames = frames * (0.8 + 0.1 * variant)
            channel_scale = torch.tensor([1.0, 0.7, -0.5, 0.3])
            examples.append(frames[:, None] * channel_scale[None, :, None, None])
            targets.append(1.0 if category else -1.0)
    images = torch.stack(examples).to(device)
    labels = torch.tensor(targets, device=device)
    mask = torch.ones(8, 6, 1, size, size, dtype=torch.bool, device=device)
    offsets = torch.tensor([[-300.2, -180.1, -60.2, -30.2, -10.2, 4.8]], device=device).expand(8, -1)
    if kind == "spatial":
        assert torch.equal(images[0].flatten(-2).sort().values, images[1].flatten(-2).sort().values)
    else:
        assert torch.equal(images[0, -1], images[1, -1])
        assert torch.equal(images[0].sort(dim=0).values, images[1].sort(dim=0).values)
    return images, offsets, mask, labels


def fit_mechanism(kind, steps, device, tolerance):
    import torch

    from h9dp.t1a import T1APressureRegressor

    torch.manual_seed(101 if kind == "spatial" else 102)
    model = T1APressureRegressor({
        "widths": [8, 12, 16, 24], "blocks_per_stage": 1, "head_width": 24,
        "activation_checkpointing": False,
    }, target_mean=0.0, target_std=1.0).to(device)
    images, offsets, mask, labels = synthetic_task(kind, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(images, offsets, mask)
        loss = (prediction - labels).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite {kind} mechanism fit")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.detach().item())
    model.eval()
    with torch.no_grad():
        prediction = model(images, offsets, mask)
        rmse = (prediction - labels).square().mean().sqrt().item()
    return {
        "samples": len(labels), "optimizer_steps": steps, "precision": "float32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "constant_baseline_rmse": 1.0, "initial_mse": losses[0], "final_rmse": rmse,
        "tolerance_rmse": tolerance, "passed": rmse < tolerance,
        "predictions": prediction.cpu().tolist(),
        "interpretation": "Synthetic capacity check only; not real-data accuracy or held-out generalization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "t1a.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--native", action="store_true", help="Use the native 500x500 input")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true", help="One gradient check, no optimizer update")
    parser.add_argument("--fit-steps", type=int, default=0, help="Opt in to two tiny synthetic supervised fits")
    parser.add_argument("--fit-tolerance", type=float, default=0.1)
    args = parser.parse_args()
    if args.size < 16 or args.batch_size < 1 or args.fit_steps < 0 or args.fit_tolerance <= 0:
        parser.error("Invalid size, batch size, fit steps or tolerance")
    import torch

    from h9dp.common import load_config
    from h9dp.model import build_model, model_architecture, regression_loss
    from h9dp.t1a import ARCHITECTURE

    config = load_config(args.config)
    if model_architecture(config["model"]) != ARCHITECTURE:
        raise ValueError("This check requires a t1a/v1 configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.device == "cuda" and args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    model.train(args.backward)
    size = 500 if args.native else args.size
    images = torch.randn(args.batch_size, 6, 4, size, size, device=device, requires_grad=args.backward)
    mask = torch.ones(args.batch_size, 6, 1, size, size, dtype=torch.bool, device=device)
    offsets = torch.tensor([[-300.2, -180.1, -60.2, -30.2, -10.2, 4.8]], device=device).expand(args.batch_size, -1)
    shapes = []
    handles = [stage.register_forward_hook(lambda _module, _inputs, result: shapes.append(list(result[0].shape)))
               for stage in model.stages]
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = None
    with torch.set_grad_enabled(args.backward), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
        prediction = model(images, offsets, mask)
        if args.backward:
            loss = regression_loss(prediction, torch.full_like(prediction, 75.0), 20.0).mean()
    if args.backward:
        assert loss is not None
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Non-finite prediction")
    gradients = {}
    if args.backward:
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Missing/non-finite gradient: {name}")
        assert images.grad is not None
        if not bool(torch.isfinite(images.grad).all()):
            raise FloatingPointError("Non-finite input gradient")
        gradients = {"all_parameter_gradients_finite": True, "input_gradient_finite": True,
                     "input_gradient_l1_by_frame": images.grad.abs().sum(dim=(0, 2, 3, 4)).cpu().tolist()}
    report = {
        "synthetic_only": True, "architecture": ARCHITECTURE,
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "seed": 42,
        "bf16_autocast": args.bf16, "input_shape": list(images.shape),
        "output_shape": list(prediction.shape), "finite_output": True,
        "prediction_hpa_untrained": prediction.detach().cpu().tolist(), "spatial_shapes": shapes,
        "forward_backward_seconds": elapsed, "gradient_check": gradients,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "resource_check_parameter_updates": 0, "parameter_updates": 0, "mechanism_fits": {},
    }
    # Release native tensors before fitting fresh, tiny synthetic models.
    del model, images, mask, offsets, prediction
    del loss
    if args.fit_steps:
        for kind in ("spatial", "temporal"):
            report["mechanism_fits"][kind] = fit_mechanism(kind, args.fit_steps, device, args.fit_tolerance)
        report["parameter_updates"] = 2 * args.fit_steps
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(not result["passed"] for result in report["mechanism_fits"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
