#!/usr/bin/env python3
"""Read-only mechanistic P3.1 audit. No optimizer, fitting or weight mutation.

Interventions on a trained model are dependency probes, not retrained ablations
or proof of unique causality. No dense attention matrices are saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "src"))


def attention_statistics(query, key, valid, chunk_size):
    """Streaming exact entropy, uniform KL and effective fractions per head/query."""
    import torch

    maximum = denominator = weighted_logit = squared = None
    with torch.autocast(query.device.type, enabled=False):
        for start in range(0, key.shape[-2], chunk_size):
            mask = valid[:, None, None, start:start + chunk_size]
            score = (query.float() @ key[..., start:start + chunk_size, :].float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
            score = score.masked_fill(~mask, -torch.inf)
            m = score.amax(-1)
            safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            weight = (score - safe[..., None]).exp()
            z = weight.sum(-1)
            moment = (weight * score.masked_fill(~mask, 0)).sum(-1)
            square = weight.square().sum(-1)
            if maximum is None:
                maximum, denominator, weighted_logit, squared = m, z, moment, square
            else:
                assert denominator is not None and weighted_logit is not None and squared is not None
                merged = torch.maximum(maximum, m)
                a = torch.where(torch.isfinite(maximum), (maximum - merged).exp(), torch.zeros_like(m))
                b = torch.where(torch.isfinite(m), (m - merged).exp(), torch.zeros_like(m))
                denominator = denominator * a + z * b
                weighted_logit = weighted_logit * a + moment * b
                squared = squared * a.square() + square * b.square()
                maximum = merged
        assert maximum is not None and denominator is not None and weighted_logit is not None and squared is not None
        count = valid.sum(-1)[:, None, None].float()
        if not bool((count > 0).all()):
            raise ValueError("Entropy audit requires a nonempty memory per frame")
        entropy = maximum + denominator.log() - weighted_logit / denominator
        return {
            "kl_from_uniform_nats": count.log() - entropy,
            "entropy_over_log_count": entropy / count.log().clamp_min(1e-12),
            "entropy_effective_fraction": entropy.exp() / count,
            "inverse_square_effective_fraction": denominator.square() / squared.clamp_min(1e-30) / count,
            "maximum_weight": 1 / denominator,
        }


def shuffle_pixels_in_tiles(value, side, seed=20261101):
    """Same permutation across time/bands/mask: preserve joint pixel trajectories.

    Within each tile the multichannel empirical distribution is unchanged,
    while adjacency and pixel-position pairing are disrupted.
    """
    import torch

    batch, times, channels, height, width = value.shape
    if height % side or width % side:
        raise ValueError("Tile side must divide both image dimensions")
    generator = torch.Generator(device=value.device).manual_seed(seed)
    order = torch.randperm(side * side, generator=generator, device=value.device)
    tiles = value.reshape(batch, times, channels, height // side, side, width // side, side)
    tiles = tiles.permute(0, 1, 2, 3, 5, 4, 6).flatten(-2).index_select(-1, order)
    return tiles.reshape(batch, times, channels, height // side, width // side, side, side).permute(0, 1, 2, 3, 5, 4, 6).reshape_as(value)


def uniform_attention(query, key, value, raw, valid, chunk_size, checkpoint_chunks=False):
    """Replace only dense-read weights with exact valid-count uniform weights."""
    del key, chunk_size, checkpoint_chunks
    count = valid.sum(-1).clamp_min(1)
    features = (value.float() * valid[:, None, :, None]).sum(-2) / count[:, None, None]
    brightness = (raw.float() * valid[..., None]).sum(1) / count[:, None]
    return features[:, :, None].expand(-1, -1, query.shape[-2], -1), brightness[:, None].expand(-1, query.shape[-2], -1)


def head_slices(model):
    width, layers = model.width, len(model.blocks)
    start = 3 * width
    raw_end = start + layers * 3 * 6 * 4
    contrast_end = raw_end + layers * 2 * 6 * 4
    return {"hidden": slice(0, start), "brightness": slice(start, raw_end),
            "contrasts": slice(raw_end, contrast_end), "source_mass": slice(contrast_end, contrast_end + layers * 3 * 4)}


def query_statistics(value):
    import torch

    with torch.autocast(value.device.type, enabled=False):
        value = value.float()
        normalized = torch.nn.functional.normalize(value, dim=-1)
        cosine = (normalized[:, :, None] * normalized[:, None, :]).sum(-1)
        off_diagonal = ~torch.eye(value.shape[1], device=value.device, dtype=torch.bool)
        return torch.stack((value.square().mean((1, 2)).sqrt(),
                            value.std(1, correction=0).square().mean(1).sqrt(),
                            cosine[:, off_diagonal].mean(1)), dim=1)


@contextmanager
def make_trace(model, storage):
    import torch

    from h9dp.p3_coverage import CoverageBlock

    original_read = CoverageBlock.read
    original_context = CoverageBlock.contextualize
    layer_map = {id(block): i for i, block in enumerate(model.blocks)}

    def traced_context(block, queries, memory, valid):
        if block.training:
            raise ValueError("Context-stage audit requires eval mode")
        layer = layer_map[id(block)]
        self_outputs = []
        def capture_self(module, inputs, output):
            del module, inputs
            self_outputs.append(output[0])
        handle = block.self_attention.register_forward_hook(capture_self)
        try:
            result = original_context(block, queries, memory, valid)
        finally:
            handle.remove()
        self_delta = self_outputs[0]
        stages = {"before": queries, "self_update": self_delta, "after_self": queries + self_delta,
                  "background_update": result - queries - self_delta, "after_background": result}
        for name, value in stages.items():
            storage.setdefault(f"layer{layer+1}_context_stage_{name}", []).append(query_statistics(value).cpu().numpy())
        return result

    def traced_read(block, queries, temporal, memory, raw, valid):
        layer = layer_map[id(block)]
        batch, times, points, width = memory.shape
        attention = block.background_attention
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        q = torch.nn.functional.linear(block.cross_norm(queries)[:, None] + temporal[:, :, None], weight[:width], bias[:width])
        k = torch.nn.functional.linear(memory, weight[width:2 * width], bias[width:2 * width])
        q = q.reshape(batch * times, 16, block.heads, width // block.heads).transpose(1, 2)
        k = k.reshape(batch * times, points, block.heads, width // block.heads).transpose(1, 2)
        flat_valid = valid.reshape(batch * times, points)
        stats = attention_statistics(q, k, flat_valid, block.chunk_size)
        lo, hi = model.global_grid**2, model.global_grid**2 + model.center_size**2
        conditional = attention_statistics(q, k[..., lo:hi, :], flat_valid[:, lo:hi], block.chunk_size)
        for name, tensor in stats.items():
            storage.setdefault(f"layer{layer+1}_{name}", []).append(tensor.cpu().numpy())
        storage.setdefault(f"layer{layer+1}_fixed_kl_nats", []).append(conditional["kl_from_uniform_nats"].cpu().numpy())
        counts = (raw[..., 4:] * valid[..., None]).sum(2) / valid.sum(2, keepdim=True)
        storage.setdefault(f"layer{layer+1}_uniform_source_mass", []).append(counts.float().cpu().numpy())
        result = original_read(block, queries, temporal, memory, raw, valid)
        storage.setdefault(f"layer{layer+1}_query_brightness", []).append(result[1].float().cpu().numpy())
        with torch.autocast(queries.device.type, enabled=False):
            normalized = torch.nn.functional.normalize(block.cross_norm(queries).float(), dim=-1)
            cosine = (normalized[:, :, None] * normalized[:, None, :]).sum(-1)
            off_diagonal = ~torch.eye(16, device=cosine.device, dtype=torch.bool)
        storage.setdefault(f"layer{layer+1}_query_cosine", []).append(cosine[:, off_diagonal].float().cpu().numpy())
        return result

    with patch.object(CoverageBlock, "read", traced_read), patch.object(CoverageBlock, "contextualize", traced_context):
        yield


def forward_case(model, batch, case):
    import torch

    import h9dp.p3_coverage as coverage

    images, times, mask = batch["images"], batch["time_offsets"], batch["mask"]
    fp32 = case == "fp32" or case.endswith("_fp32")
    case = case.removesuffix("_fp32")
    with ExitStack() as stack:
        if case.startswith("shuffle_"):
            try:
                side = int(case.split("_")[-1])
            except ValueError as error:
                raise ValueError("Malformed shuffle case") from error
            images = shuffle_pixels_in_tiles(images, side)
            mask = shuffle_pixels_in_tiles(mask, side)
        elif case == "reverse_images":
            images, mask = images.flip(1), mask.flip(1)
        elif case == "zero_time":
            times = torch.zeros_like(times)
        elif case == "repeat_latest":
            images, mask = images[:, -1:].expand_as(images), mask[:, -1:].expand_as(mask)
        elif case == "uniform_read":
            stack.enter_context(patch.object(coverage, "online_attention", uniform_attention))
        elif case == "no_spatial_pe":
            def embed_without_xy(raw, xy, offsets):
                time = (offsets.float() / 300)[:, :, None, None].expand(*xy.shape[:-1], 1)
                coordinates = torch.cat((torch.zeros_like(xy), time), dim=-1)
                return model.input_projection(raw) + coverage.position_encoding(coordinates, model.width)
            stack.enter_context(patch.object(model, "_embed", embed_without_xy))
        elif case in ("no_raw_history", "no_hidden", "no_explicit_contrasts"):
            parts = head_slices(model)
            selected = {"no_raw_history": slice(parts["brightness"].start, parts["contrasts"].stop),
                        "no_hidden": parts["hidden"], "no_explicit_contrasts": parts["contrasts"]}[case]
            def zero_head_part(module, inputs):
                del module
                value = inputs[0].clone()
                value[:, selected] = 0
                return (value,)
            handle = model.head.mlp[0].register_forward_pre_hook(zero_head_part)
            stack.callback(handle.remove)
        elif case not in ("normal", "fp32"):
            raise ValueError(f"Unknown case {case}")
        with nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16):
            return model.inspect_reads(images, times, mask)


def metric_report(rows, normal=None):
    import numpy as np

    target = np.asarray([r["target"] for r in rows])
    predicted = np.asarray([r["prediction"] for r in rows])
    errors = predicted - target
    result = {"count": len(rows), "rmse": np.sqrt(np.mean(errors**2)).item(), "mae": np.abs(errors).mean().item(),
              "bias": errors.mean().item(), "prediction_std": predicted.std().item(),
              "target_std": target.std().item(), "range": [predicted.min().item(), predicted.max().item()]}
    if normal is not None:
        changes = predicted - np.asarray([r["prediction"] for r in normal])
        result.update(mean_absolute_change=np.abs(changes).mean().item(), max_absolute_change=np.abs(changes).max().item())
    return result


def summarize_trace(storage, model):
    import numpy as np

    result = {}
    for name, batches in storage.items():
        values = np.concatenate(batches)
        if "query_brightness" in name or name == "head_input":
            continue
        if "context_stage" in name:
            result[name] = {"rms_mean": values[:, 0].mean().item(), "query_std_rms_mean": values[:, 1].mean().item(),
                            "query_cosine_mean": values[:, 2].mean().item()}
        elif "uniform_source_mass" in name:
            result[name] = values.mean((0, 1)).tolist()
        else:
            result[name] = {"mean": values.mean().item(), "q05_q50_q95": np.quantile(values, [.05, .5, .95]).tolist()}
    for layer in range(len(model.blocks)):
        values = np.concatenate(storage[f"layer{layer+1}_query_brightness"])
        roles = np.stack([values[:, section].mean(1) for section in (slice(0, 4), slice(4, 12), slice(12, 16))], axis=1)
        differences = np.stack((roles[:, 0] - roles[:, 1], roles[:, 0] - roles[:, 2]), axis=1)
        result[f"layer{layer+1}_role_contrasts"] = {
            "normalized_brightness_rms": np.sqrt(np.mean(roles**2)).item(),
            "between_record_std_rms": np.sqrt(np.mean(roles.std(0)**2)).item(),
            "contrast_rms": np.sqrt(np.mean(differences**2)).item(),
            "within_role_query_std_rms": np.sqrt(np.mean(values.std(1)**2)).item(),
        }
    inputs = np.concatenate(storage["head_input"])
    first = model.head.mlp[0]
    weight = first.weight.detach().float().cpu().numpy()
    result["head_parts"] = {}
    for name, section in head_slices(model).items():
        x = inputs[:, section]
        contribution = x @ weight[:, section].T
        result["head_parts"][name] = {
            "input_rms": np.sqrt(np.mean(x**2)).item(), "input_between_record_std_rms": np.sqrt(np.mean(x.std(0)**2)).item(),
            "first_linear_contribution_between_record_std_rms": np.sqrt(np.mean(contribution.std(0)**2)).item(),
        }
    singular = np.linalg.svd(inputs - inputs.mean(0), compute_uv=False)
    energy = singular**2 / max((singular**2).sum(), 1e-20)
    result["head_input_pca"] = {"first5_variance_fractions": energy[:5].tolist(),
                                "components_for_95_percent": np.searchsorted(np.cumsum(energy), .95).item() + 1}
    return result, inputs


def gradient_probe(model, batch):
    import numpy as np
    import torch

    import h9dp.p3_coverage as coverage

    original_sample, original_embed = coverage.sample_observations, model._embed
    local_points = 4 * (model.native_size**2 + model.dynamic_size**2)
    predictions, result = {}, {}
    for case in ("full", "detach_dynamic_raw", "detach_local_position"):
        with ExitStack() as stack:
            outputs = []
            def capture(module, inputs, output, captured=outputs):
                del module, inputs
                captured.append(output)
            for block in model.blocks:
                handle = block.router.register_forward_hook(capture)
                stack.callback(handle.remove)
            if case == "detach_dynamic_raw":
                def detach_raw(clean, mask, grid, mode):
                    raw, valid = original_sample(clean, mask, grid, mode)
                    return (raw.detach() if mode == "bilinear" else raw), valid
                stack.enter_context(patch.object(coverage, "sample_observations", detach_raw))
            elif case == "detach_local_position":
                def detach_xy(raw, xy, offsets):
                    return original_embed(raw, xy.detach() if raw.shape[2] == local_points else xy, offsets)
                stack.enter_context(patch.object(model, "_embed", detach_xy))
            images = batch["images"].detach().clone().requires_grad_(True)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _ = model._forward(images, batch["time_offsets"], batch["mask"])
                loss = ((prediction - batch["target"]) / model.target_std).square().mean()
            gradients = torch.autograd.grad(loss, [*outputs, images], allow_unused=True)
            predictions[case] = prediction.detach().cpu().numpy()
            result[case] = {"loss": loss.item(), "layers": []}
            for gradient in gradients[:-1]:
                if gradient is None:
                    raise RuntimeError("Disconnected router output")
                g = gradient.detach().float()
                result[case]["layers"].append({"xy_gradient_rms": g[..., :2].square().mean().sqrt().item(),
                                                 "scale_gradient_rms": g[..., 2].square().mean().sqrt().item()})
            image_grad = gradients[-1]
            if image_grad is None:
                raise RuntimeError("Disconnected image input")
            result[case]["image_gradient_rms"] = image_grad.float().square().mean().sqrt().item()
        del gradients, images, loss, prediction, outputs
    result["max_forward_difference_across_detach_probes"] = max(np.abs(v - predictions["full"]).max().item() for v in predictions.values())
    if result["max_forward_difference_across_detach_probes"] > 1e-6:
        raise RuntimeError("Gradient isolation changed forward values")
    return result


def initialization_probe(model, model_config, target, batch, device):
    """Same three records; compare trained queries with untrained seed-42 queries."""
    import numpy as np
    import torch

    from h9dp.model import build_model

    result = {}
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(42)
        initial = build_model(model_config, target["mean_hpa"], target["std_hpa"]).to(device).eval().requires_grad_(False)
        for name, current in (("trained", model), ("initialized_untrained", initial)):
            storage = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), make_trace(current, storage):
                current.inspect_reads(batch["images"], batch["time_offsets"], batch["mask"])
            result[name] = {}
            for key, batches in storage.items():
                if "context_stage" in key:
                    values = np.concatenate(batches).mean(0)
                    result[name][key] = {"rms_mean": values[0].item(), "query_std_rms_mean": values[1].item(),
                                         "query_cosine_mean": values[2].item()}
        del initial
    return result


def data_protocol(index):
    import numpy as np

    train = [s for s in index["samples"] if s["split"] == "train"]
    validation = [s for s in index["samples"] if s["split"] == "validation"]
    def keys(rows):
        return {tuple(index["frames"][f]["files"]) for s in rows for f in s["frames"]}
    result = {"bands": index["bands"], "normalization": index["normalization"], "target_normalization": index["target_normalization"],
              "shared_original_frame_groups": len(keys(train) & keys(validation)),
              "shared_storms": len({s["storm_id"] for s in train} & {s["storm_id"] for s in validation})}
    for name, rows in (("train", train), ("validation", validation)):
        offsets = np.asarray([s["time_offsets_minutes"] for s in rows])
        result[name] = {"count": len(rows), "future_frames": np.asarray(np.count_nonzero(offsets > 0)).item(),
                        "time_offsets_minutes_q0_q50_q100": np.quantile(offsets, [0, .5, 1]).tolist(),
                        "all_sequences_strictly_increasing": bool((np.diff(offsets, axis=1) > 0).all())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--config", default=str(ROOT / "configs/p3_coverage.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args()
    import numpy as np
    import torch

    from h9dp.common import check_space, load_config, write_json
    from h9dp.dataset import SequenceDataset, make_loader
    from h9dp.engine import code_fingerprint, initialize_device, move_batch
    from h9dp.model import build_model
    from h9dp.p3_coverage import CoveragePressureRegressor
    from h9dp.prepare import load_cache

    config = load_config(args.config)
    run = Path(args.run_dir).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output) if args.output else run / f"audit_{stamp}"
    check_space(output, 100_000_000)
    output.mkdir(parents=True, exist_ok=False)
    index = load_cache(config)
    device, environment = initialize_device(config)
    paths = [run / name for name in ("best.pt", "last.pt")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {"status": "running", "training": False, "optimizer_steps": 0, "checkpoint_hashes_before": hashes,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "warning": "Inference/gradient dependency probes, not retrained ablations; reference points are not verified eyes.",
              "environment": environment, "data_protocol": data_protocol(index), "checkpoints": {}}
    write_json(output / "report.json", report)
    cases = ("normal", "uniform_read", "no_spatial_pe", "shuffle_4", "shuffle_10", "shuffle_50", "shuffle_500",
             "reverse_images", "zero_time", "repeat_latest", "no_raw_history", "no_hidden", "no_explicit_contrasts", "fp32",
             "shuffle_4_fp32", "shuffle_10_fp32", "reverse_images_fp32")
    for checkpoint in paths:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["data_fingerprint"] != index["fingerprint"] or state["code_fingerprint"] != code_fingerprint(config):
            raise ValueError("Audit requires matching training data and source code")
        target = state["target_normalization"]
        model = build_model(state["model_config"], target["mean_hpa"], target["std_hpa"]).to(device).eval().requires_grad_(False)
        assert isinstance(model, CoveragePressureRegressor)
        model.load_state_dict(state["model"], strict=True)
        checkpoint_report = {"epoch": state["epoch_completed"], "splits": {}}
        splits = ("train", "validation") if checkpoint.name == "best.pt" else ("validation",)
        for split in splits:
            storage, predictions, means, masks = {}, {case: [] for case in cases}, [], []
            def capture_head(module, inputs, captured=storage):
                del module
                captured.setdefault("head_input", []).append(inputs[0].float().cpu().numpy())
            loader = make_loader(config, index, split, config["training"]["batch_size"], augment=False)
            with torch.no_grad():
                for host in loader:
                    batch = move_batch(host, device)
                    current_cases = cases if split == "validation" and checkpoint.name == "best.pt" else ("normal", "uniform_read")
                    for case in current_cases:
                        with ExitStack() as stack:
                            if case == "normal":
                                stack.enter_context(make_trace(model, storage))
                                handle = model.head.mlp[0].register_forward_pre_hook(capture_head)
                                stack.callback(handle.remove)
                            prediction, _ = forward_case(model, batch, case)
                        if not torch.isfinite(prediction).all():
                            raise FloatingPointError(f"Nonfinite output {checkpoint.name}/{case}")
                        for i, value in enumerate(prediction.cpu().tolist()):
                            predictions[case].append({"record": host["record"][i].item(), "target": host["target"][i].item(), "prediction": value})
                    images, mask = batch["images"], batch["mask"]
                    means.append((images.sum((-1, -2)) / mask.sum((-1, -2))).cpu().numpy())
                    masks.append(mask.float().mean((-1, -2, -3)).cpu().numpy())
            split_report, head_inputs = summarize_trace(storage, model)
            split_report["predictions"] = {
                case: {**metric_report(rows, predictions["fp32"] if case.endswith("_fp32") else predictions["normal"]),
                       "change_reference": "fp32" if case.endswith("_fp32") else "normal"}
                for case, rows in predictions.items() if rows
            }
            if checkpoint.name == "best.pt":
                with (run / f"best_{split}.csv").open() as stream:
                    saved = {row["record"]: row["predicted_dP"] for row in csv.DictReader(stream)}
                expected = np.asarray([saved[str(r["record"])] for r in predictions["normal"]], dtype=float)
                actual = np.asarray([r["prediction"] for r in predictions["normal"]])
                split_report["max_difference_vs_training_csv"] = np.abs(actual - expected).max().item()
            split_report["valid_fraction_min_mean"] = [np.concatenate(masks).min().item(), np.concatenate(masks).mean().item()]
            simple_means = np.concatenate(means).mean(1)
            normal_pred = np.asarray([r["prediction"] for r in predictions["normal"]])
            truth = np.asarray([r["target"] for r in predictions["normal"]])
            split_report["correlations_with_global_band_means"] = {
                "prediction": [np.corrcoef(normal_pred, simple_means[:, c])[0, 1].item() for c in range(4)],
                "target": [np.corrcoef(truth, simple_means[:, c])[0, 1].item() for c in range(4)],
            }
            prefix = f"{checkpoint.stem}_{split}"
            np.savez_compressed(output / f"{prefix}_features.npz", allow_pickle=False, head_input=head_inputs,
                                normalized_global_band_means=simple_means, target=truth,
                                brightness_layer1=np.concatenate(storage["layer1_query_brightness"]),
                                brightness_layer2=np.concatenate(storage["layer2_query_brightness"]))
            with (output / f"{prefix}_predictions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["case", "record", "target", "prediction"])
                writer.writeheader()
                for case, rows in predictions.items():
                    writer.writerows({"case": case, **row} for row in rows)
            checkpoint_report["splits"][split] = split_report
            print({"checkpoint": checkpoint.name, "split": split, "normal": split_report["predictions"]["normal"],
                   "uniform": split_report["predictions"]["uniform_read"]}, flush=True)
        dataset = SequenceDataset(config["data"]["cache_dir"], index, "train")
        if checkpoint.name == "best.pt":
            positions = [next(i for i, s in enumerate(dataset.samples) if predicate(s["dP"]))
                         for predicate in (lambda x: x < 60, lambda x: 60 <= x < 90, lambda x: x >= 90)]
            from torch.utils.data import default_collate
            batch = move_batch(default_collate([dataset[i] for i in positions]), device)
            checkpoint_report["gradient_probe_training_records"] = [dataset.samples[i]["record"] for i in positions]
            checkpoint_report["gradient_probe"] = gradient_probe(model, batch)
            checkpoint_report["initialization_probe_same_training_records"] = initialization_probe(model, state["model_config"], target, batch, device)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Audit accumulated parameter gradients")
        report["checkpoints"][checkpoint.name] = checkpoint_report
        write_json(output / "report.json", report)
        del model
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if after != hashes:
        raise RuntimeError("Checkpoint files changed during audit")
    report.update(status="completed", checkpoint_hashes_after=after, checkpoints_unchanged=True)
    write_json(output / "report.json", report)
    print(str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
