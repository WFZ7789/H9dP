"""Small compiled joint-dP updates of existing M1 experts; no larger model.

The source outer folds and their train-only M5 coefficients are retained.
There is no learned post-hoc calibrator in this experiment.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from h9dp.engine import compile_model, initialize_device, seed_everything
from h9dp.m1 import EXPERTS, metrics
from h9dp.m1_adapt import JointM1, joint_loss, normalization, quality_weights
from scripts.m1 import (
    M1Dataset,
    build_expert,
    read_json,
    resolve,
    target_statistics,
    write_json,
)
from scripts.m1_adapt import loader, padded_batch, summarize


def train_fold(source, destination, outer, epochs, source_epoch, lr, parameter_weight, device):
    manifest = read_json(source / "run.json")
    config = manifest["config"]
    work = resolve(config, config["data"]["work_dir"])
    meta = read_json(work / "all_meta.json")
    records = meta["records"]
    folds = np.asarray(manifest["folds"])
    training = np.flatnonzero(folds != outer).tolist()
    evaluation = np.flatnonzero(folds == outer).tolist()
    paired = [i for i in training if records[i]["params"] is not None]
    mean, std = normalization(records, training)
    seed = int(config["training"]["seed"]) + 5000 + 1000 * outer
    seed_everything(seed)
    experts: list[nn.Module] = []
    for j, spec in enumerate(EXPERTS):
        tm, ts = target_statistics(meta, paired, j)
        expert = build_expert(spec, config, mean, std, tm, ts, device)
        payload = torch.load(source / f"outer{outer}/base/{spec.name}.pt", map_location="cpu", weights_only=False)
        if payload["epoch"] != source_epoch or payload["indices"] != training:
            raise ValueError("Source checkpoint epoch/fold mismatch")
        expert.load_state_dict(payload["model"])
        experts.append(expert)
    teacher = read_json(source / f"outer{outer}/calibration_epoch{source_epoch:03d}.json")
    model = JointM1(experts, np.asarray(teacher["teacher_coefficients"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator().manual_seed(seed)
    directory = destination / f"outer{outer}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "joint.pt"
    start = 1
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        generator.set_state(payload["generator"])
        start = payload["epoch"] + 1
        if start > epochs + 1:
            raise ValueError("Cannot rewind a joint checkpoint")
    train_loader = loader(M1Dataset(work, "all", training), config, generator, True)
    eval_loader = loader(M1Dataset(work, "all", evaluation), config)
    weights = torch.tensor(quality_weights(records, config["training"]["quality_weights"]), dtype=torch.float32, device=device)
    parameters = np.asarray([records[i]["params"] for i in paired])
    parameter_std = torch.tensor(parameters.std(axis=0).clip(min=1e-6), dtype=torch.float32, device=device)
    dp_std = max(float(np.std([records[i]["dP"] for i in training])), 1e-6)
    compiled = compile_model(model, config)
    batch_size = int(config["training"]["batch_size"])
    first_step = None
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, epochs + 1):
        model.train()
        total, count_total = 0.0, 0
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        for batch in train_loader:
            batch, count = padded_batch(batch, device, batch_size)
            cold = time.perf_counter() if first_step is None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = compiled(batch["images"], batch["mask"], batch["numeric"], batch["coverage"], batch["times"], batch["centers"])
            loss = joint_loss(output[:count], batch["params"][:count], batch["dp"][:count],
                              weights[batch["record"][:count]], parameter_std, dp_std, parameter_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            if cold is not None:
                torch.cuda.synchronize(device)
                first_step = time.perf_counter() - cold
            total += float(loss.detach()) * count
            count_total += count
        torch.cuda.synchronize(device)
        entry = {"outer": outer, "joint_epoch": epoch, "loss": total / count_total,
                 "seconds": time.perf_counter() - begin, "first_step_seconds": first_step,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                 "dp_training_count": len(training), "parameter_training_count": len(paired)}
        with (directory / "joint.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "generator": generator.get_state()}, temporary)
        temporary.replace(checkpoint)
        print(json.dumps(entry), flush=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in eval_loader:
            batch, count = padded_batch(batch, device, batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["images"], batch["mask"], batch["numeric"], batch["coverage"], batch["times"], batch["centers"])
            predictions.extend(output[:count].float().cpu().tolist())
    return evaluation, np.asarray(predictions)


def run(source: Path, destination: Path, epochs: int, source_epoch: int, lr: float, parameter_weight: float):
    parent = read_json(source / "run.json")
    config = parent["config"]
    if not config["training"]["torch_compile"]:
        raise ValueError("Formal joint training requires torch.compile")
    if epochs < 1 or lr <= 0 or parameter_weight < 0:
        raise ValueError("Invalid joint training settings")
    work = resolve(config, config["data"]["work_dir"])
    meta = read_json(work / "all_meta.json")
    records = meta["records"]
    identity = {"source": str(source), "source_identity": parent["identity"],
                "source_epoch": source_epoch, "learning_rate": lr, "parameter_weight": parameter_weight}
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "run.json"
    if manifest.exists() and read_json(manifest)["identity"] != identity:
        raise ValueError("Joint resume identity mismatch")
    actual_identity = hashlib.sha256(json.dumps({"config": config, "meta": meta}, sort_keys=True).encode()).hexdigest()
    if actual_identity != parent["identity"]:
        raise ValueError("Source data/config changed")
    compiler_cache = resolve(config, config["compiler_cache"])
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(compiler_cache / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(compiler_cache / "triton"))
    device, environment = initialize_device(config)
    write_json(manifest, {"identity": identity, "status": "running", "target_epoch": epochs,
                          "environment": environment, "no_permanent_holdout": True,
                          "training": "compiled BF16", "evaluation": "eager BF16"})
    prediction = np.full((len(records), 6), np.nan)
    begin = time.perf_counter()
    for outer in range(int(config["crossfit"]["outer_folds"])):
        indices, values = train_fold(source, destination, outer, epochs, source_epoch, lr, parameter_weight, device)
        prediction[indices] = values
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite or incomplete joint predictions")
    target = np.asarray([r["dP"] for r in records])
    report = {"status": "completed adaptive joint development stage; NOT a blind test",
              "joint_epochs": epochs, "source_epoch": source_epoch, "identity": identity,
              "elapsed_seconds": time.perf_counter() - begin,
              "dp": summarize(prediction[:, 5], target, records), "parameters_Good": {}}
    paired_good = [i for i, r in enumerate(records) if r["params"] is not None and r["quality"] == "Good"]
    for j, spec in enumerate(EXPERTS):
        report["parameters_Good"][spec.name] = metrics(prediction[paired_good, j],
                                                      np.asarray([records[i]["params"][j] for i in paired_good]))
    for origin in ("old", "new"):
        chosen = np.asarray([r["origin"] == origin and r["quality"] == "Good" for r in records], dtype=np.bool_)
        report[f"{origin}_Good"] = metrics(prediction[chosen, 5], target[chosen])
    with (destination / f"predictions_joint_epoch{epochs:03d}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record", "storm_id", "quality", "dP", "joint_dP"] + [s.parameter for s in EXPERTS])
        for i, record in enumerate(records):
            writer.writerow([record["record"], record["storm_id"], record["quality"], record["dP"], prediction[i, 5]] + prediction[i, :5].tolist())
    write_json(destination / f"summary_joint_epoch{epochs:03d}.json", report)
    state = read_json(manifest)
    state.update(status="paused_for_diagnosis", completed_epoch=epochs)
    write_json(manifest, state)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-epoch", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--parameter-weight", type=float, default=0.25)
    args = parser.parse_args()
    result = run(args.source_run.resolve(), args.run_dir.resolve(), args.epochs,
                 args.source_epoch, args.learning_rate, args.parameter_weight)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
