"""Diagnose why V1 works: signal blocks, baselines, ablations and label compression.

This is a diagnostic re-use of the fitted V1 machinery, not a new model run:
the fold assignment, features, label weights and fit routine are those of the
formal V1 experiment (`scripts/v1.py`). Ablations zero kernel rows instead of
rebuilding the feature bank, so representation, normalization and optimizer
stay fixed across variants.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import _normalize, record_folds, scores, summarize_time
from scripts.v1 import bank, fit

GROUPS = ("eye", "wall", "relief", "waves", "motion", "sector")
CHANNELS = ("B08", "B09", "B10", "B16", "B08-B09", "B09-B10", "B10-B16", "B16-B08")
EYE_RADII = (1, 2, 3, 5, 7, 10, 14, 19)
WALL_RADII = (25, 33, 44, 58, 76, 98, 126, 160, 198, 228)
QUANTILES = (0.03, 0.2, 0.5, 0.8, 0.97)


def load(source, features):
    try:
        raw = (source / "all_meta.json").read_bytes()
        rows = json.loads(raw)["records"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"Cannot read {source / 'all_meta.json'}") from error
    catalog, digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    with np.load(features) as f:
        ids, times = f["record"], f["time"]
        groups = {k: f[k] for k in GROUPS}
        if (
            str(f["source_sha256"]) != hashlib.sha256(raw).hexdigest()
            or not np.array_equal(ids, [r["record"] for r in rows])
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Feature provenance differs from source")
    y = np.array([r["dP"] for r in rows])
    quality = np.array([r["quality"] for r in rows])
    return ids, times, groups, y, quality, digest


def ridge_oof(x, y, fold, lam=1.0):
    """Record-level ridge in dual form, with training-only normalization."""
    pred = np.full_like(y, np.nan, dtype=float)
    for k in range(5):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        a, b = (torch.as_tensor(v, dtype=torch.float64) for v in _normalize(x[train], x[held]))
        target = torch.as_tensor(y[train], dtype=torch.float64)
        gram = a @ a.T / a.shape[1]
        alpha = torch.linalg.solve(gram + lam * torch.eye(len(train), dtype=gram.dtype), target - target.mean())
        pred[held] = (b @ a.T / a.shape[1] @ alpha + target.mean()).numpy()
    return pred


def single_feature_oof(x, y, fold):
    """Best univariate feature, selected on the training fold only."""
    pred = np.full_like(y, np.nan, dtype=float)
    chosen = []
    for k in range(5):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        correlations = []
        for j in range(x.shape[1]):
            ok = np.isfinite(x[train, j])
            correlations.append(
                np.corrcoef(x[train, j][ok], y[train][ok])[0, 1]
                if ok.sum() > 10 and np.std(x[train, j][ok]) > 0
                else 0.0
            )
        j = int(np.argmax(np.abs(correlations)))
        chosen.append(j)
        ok, column = np.isfinite(x[train, j]), x[:, j]
        slope = np.cov(column[train][ok], y[train][ok], bias=True)[0, 1] / column[train][ok].var()
        filled = np.where(np.isfinite(column[held]), column[held], column[train][ok].mean())
        pred[held] = y[train].mean() + slope * (filled - column[train][ok].mean())
    return pred, chosen


def knn_oof(x, y, fold, k=25):
    """Distance-weighted analog forecast in the standardized eye descriptor."""
    pred = np.full_like(y, np.nan, dtype=float)
    for f in range(5):
        train, held = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        a, b = _normalize(x[train], x[held])
        d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d, axis=1)[:, :k]
        w = 1 / (np.take_along_axis(d, idx, 1) + 1e-6)
        pred[held] = (w * y[train][idx]).sum(1) / w.sum(1)
    return pred


def paired_bootstrap(pred, reference, y, draws=5000, seed=1):
    """Paired record bootstrap of the MSE difference to the full model."""
    a, b = (pred - y) ** 2, (reference - y) ** 2
    idx = np.random.default_rng(seed).integers(0, len(y), (draws, len(y)))
    diff = (a[idx] - b[idx]).mean(1)
    lo, hi = np.quantile(diff, (0.025, 0.975))
    return {
        "delta_mse": float(a.mean() - b.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "separable": bool(lo > 0 or hi < 0),
    }


def summarize(pred, y, quality, fold) -> dict[str, Any]:
    slope, _ = np.polyfit(y, pred, 1)
    return {
        "metrics": scores(y, pred, quality),
        "correlation": float(np.corrcoef(y, pred)[0, 1]),
        "r2": float(1 - np.mean((pred - y) ** 2) / np.var(y)),
        "compression_slope": float(slope),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(y)),
        "fold_rmse": {str(k): float(np.sqrt(np.mean((pred[fold == k] - y[fold == k]) ** 2))) for k in range(5)},
    }


def feature_label(group, j, stat):
    radii, width = (WALL_RADII, 50) if group == "wall" else (EYE_RADII, 40)
    return {
        "group": group,
        "stat": stat,
        "channel": CHANNELS[j // width],
        "radius_px": radii[(j % width) // 5],
        "quantile": QUANTILES[(j % width) % 5],
    }


def top_features(groups, times, y, per_group=3):
    valid = np.isfinite(groups["eye"]).any(-1)
    out = []
    for name in ("eye", "wall", "relief"):
        stats = summarize_time(groups[name], times, valid)
        for stat in ("recent", "latest", "trend"):
            x = stats[stat]
            for j in range(x.shape[1]):
                col = x[:, j]
                ok = np.isfinite(col)
                if ok.sum() < 100 or np.nanstd(col) == 0:
                    continue
                out.append(
                    feature_label(name, j, stat)
                    | {"pearson": float(np.corrcoef(col[ok], y[ok])[0, 1])}
                )
    out.sort(key=lambda r: -abs(r["pearson"]))
    kept = []
    for group in ("eye", "wall", "relief"):
        kept.extend([r for r in out if r["group"] == group][:per_group])
    return kept


def label_structure(ids, y, quality, fold):
    """How much of dP the MOTIS catalog's own summary columns already encode."""
    catalog, _ = read_catalog()
    rows = [catalog[i] for i in ids]
    pearson = {}
    for key in ("t_cdo", "b27", "b31", "roci", "poci"):
        x = np.array([r[key] if r.get(key) is not None else np.nan for r in rows], dtype=float)
        ok = np.isfinite(x)
        pearson[key] = {"n": int(ok.sum()), "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1])}
    result = {"pearson": pearson}
    for name, columns in {
        "t_cdo": ("t_cdo",),
        "t_cdo_roci": ("t_cdo", "roci"),
        "motis_summary": ("t_cdo", "b27", "b31", "roci"),
    }.items():
        x = np.stack([[np.nan if r.get(c) is None else r[c] for c in columns] for r in rows]).astype(float)
        result[name] = summarize(ridge_oof(x, y, fold, lam=0.1), y, quality, fold)
    return result


def mass_table(formal):
    rows = []
    for k in range(5):
        state = torch.load(formal / f"fold{k}.pt", map_location="cpu", weights_only=False)
        mass = state["mass"].exp().numpy()
        names = state["names"]
        total = float(mass.sum())
        row = {"fold": k, "total": total, "noise": state["noise"].exp().tolist()}
        for group in GROUPS:
            row[group] = float(mass[[i for i, n in enumerate(names) if n.startswith(group + "_")]].sum() / total)
        # Spline kernels are built on the "recent" descriptors, so count them there.
        for stat in ("recent", "latest", "trend", "sequence", "spline"):
            row[stat] = float(
                mass[
                    [i for i, n in enumerate(names) if stat in n or (stat == "recent" and "spline" in n)]
                ].sum()
                / total
            )
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--formal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    ids, times, groups, y, quality, digest = load(a.source, a.features)
    fold = record_folds(len(y), 5, 1)
    valid = np.isfinite(groups["eye"]).any(-1)
    if not np.all(valid.any(1) & np.isfinite(times).all(1)):
        raise ValueError("Every record needs one finite frame and time")
    result: dict[str, Any] = {"seed": 1, "n": len(y), "catalog_sha256": digest}

    # Baselines: what a label prior, one cloud number, or a plain linear map gets.
    train_mean = np.full_like(y, np.nan, dtype=float)
    for k in range(5):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        train_mean[held] = y[train].mean()
    result["train_mean"] = summarize(train_mean, y, quality, fold)

    stats = {g: summarize_time(groups[g], times, valid) for g in GROUPS}
    blocks = {g: np.concatenate([stats[g][s] for s in ("recent", "latest", "trend")], axis=1) for g in GROUPS}
    inner = np.concatenate([blocks[g] for g in ("eye", "wall", "relief")], axis=1)
    single, selected = single_feature_oof(inner, y, fold)
    result["single_feature"] = summarize(single, y, quality, fold)
    widths = {g: blocks[g].shape[1] for g in ("eye", "wall", "relief")}
    labels = []
    for index in selected:
        offset = 0
        for group, width in widths.items():
            if index < offset + width:
                local, stride = index - offset, width // 3
                labels.append(
                    feature_label(group, local % stride, ("recent", "latest", "trend")[local // stride])
                )
                break
            offset += width
    result["single_feature_labels"] = labels
    for g in GROUPS:
        result["ridge_" + g] = summarize(ridge_oof(blocks[g], y, fold), y, quality, fold)
    result["ridge_all"] = summarize(
        ridge_oof(np.concatenate([blocks[g] for g in GROUPS], axis=1), y, fold), y, quality, fold
    )
    result["knn_eye"] = summarize(knn_oof(blocks["eye"], y, fold), y, quality, fold)
    result["top_features"] = top_features(groups, times, y)
    result["label_structure"] = label_structure(ids, y, quality, fold)

    # What the formal fit actually put its kernel mass on.
    result["mass"] = mass_table(a.formal)

    # Ablations: same folds, features and optimizer; only selected kernel rows survive.
    torch.manual_seed(1)
    variants = {
        "full": lambda n: True,
        "no_eye": lambda n: not n.startswith("eye_"),
        "no_wall": lambda n: not n.startswith("wall_"),
        "no_relief": lambda n: not n.startswith("relief_"),
        "no_waves": lambda n: not n.startswith("waves_"),
        "no_sector": lambda n: not n.startswith("sector_"),
        "no_motion": lambda n: not n.startswith("motion_"),
        "eye_only": lambda n: n.startswith("eye_"),
        "inner_only": lambda n: n.startswith(("eye_", "wall_", "relief_")),
        "no_inner": lambda n: not n.startswith(("eye_", "wall_", "relief_")),
        "no_spline": lambda n: "spline" not in n,
        "no_sequence": lambda n: "_sequence_" not in n,
        "no_trend": lambda n: "_trend_" not in n,
        "static": lambda n: not (n.startswith("motion_") or "_sequence_" in n),
    }
    ablations = {name: np.full_like(y, np.nan, dtype=float) for name in variants}
    equal = np.full_like(y, np.nan, dtype=float)
    for k in range(5):
        train, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        basis, names = bank(groups, times, train, held)
        for name, keep in variants.items():
            mask = torch.tensor([keep(n) for n in names], dtype=torch.bool, device=basis.device)
            pred, _, _ = fit(basis * mask[:, None, None], names, y[train], quality[train])
            ablations[name][held] = pred
        kernel = basis.mean(0) + 8
        cov = kernel[: len(train), : len(train)] + torch.eye(len(train), dtype=kernel.dtype, device=kernel.device) * 0.01
        dual = torch.cholesky_solve(
            torch.tensor(y[train] - y[train].mean(), dtype=kernel.dtype, device=kernel.device)[:, None],
            torch.linalg.cholesky(cov),
        )
        equal[held] = (kernel[len(train) :, : len(train)] @ dual).flatten().cpu().numpy() + y[train].mean()
    result["ablation"] = {name: summarize(pred, y, quality, fold) for name, pred in ablations.items()}
    result["ablation_bootstrap"] = {
        name: paired_bootstrap(pred, ablations["full"], y) for name, pred in ablations.items() if name != "full"
    }
    result["equal_mass_gp"] = summarize(equal, y, quality, fold)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    for key in ("train_mean", "single_feature", "knn_eye", "ridge_eye", "ridge_all", "equal_mass_gp"):
        block: dict[str, Any] = result[key]
        print(key, block["metrics"]["all"]["rmse"], block["r2"])
    ablation: dict[str, Any] = result["ablation"]
    print(json.dumps({name: round(v["metrics"]["all"]["rmse"], 3) for name, v in ablation.items()}, indent=1))


if __name__ == "__main__":
    main()
