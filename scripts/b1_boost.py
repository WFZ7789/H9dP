"""Compare three non-GP estimators on the same record-level R1 outer folds."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.b1 import grow_tree, ridge_trend
from h9dp.r1 import record_folds, scores
from scripts.r1_adaptive_kernel import assemble
from scripts.r1_gp import configure
from scripts.r1_kernel import load_cache

GROUPS = ('regions_recent', 'regions_latest', 'regions_trend', 'thermal_recent', 'thermal_latest',
          'thermal_trend', 'core_recent', 'profile_recent', 'scene_core_recent')
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
TREES, DEPTH, SHRINK, LEAF_PENALTY = 128, 3, 0.08, 4.0


def fold_design(groups, train, held):
    """Keep and scale columns on train only. Store the mapping for replay."""
    xs, feature_group, raw_column, means, scales = [], [], [], [], []
    for i, name in enumerate(GROUPS):
        values = groups[name]
        a = values[train]
        keep = np.flatnonzero(np.isfinite(a).sum(axis=0) >= max(3, np.ceil(0.15 * len(a))))
        if not len(keep):
            continue
        mean = np.nanmean(a[:, keep], axis=0)
        scale = np.nanstd(a[:, keep], axis=0).clip(0.05)
        x = (values[np.ix_(np.r_[train, held], keep)] - mean) / scale
        xs.append(np.where(np.isfinite(x), x.clip(-5, 5), 0))
        feature_group.extend([i] * len(keep))
        raw_column.extend(keep)
        means.extend(mean)
        scales.extend(scale)
    if not xs:
        raise ValueError('No supported columns in training records')
    state: dict[str, np.ndarray] = {'group': np.asarray(feature_group, dtype=np.int64),
                                    'column': np.asarray(raw_column, dtype=np.int64),
                                    'center': np.asarray(means), 'scale': np.asarray(scales)}
    return np.concatenate(xs, axis=1).astype(np.float32), state


def tree_prediction(x, feature, threshold, leaves):
    """Vectorized replay; -1 denotes an intentionally inactive split."""
    bits = (feature >= 0) & (x[:, feature.clip(0)] > threshold)
    node = (bits * (2 ** np.arange(bits.shape[-1] - 1, -1, -1))).sum(axis=-1)
    return leaves[np.arange(len(leaves)), node].sum(axis=1, dtype=np.float64)


def fit_fold(groups, cache, train, held, seed, compiled_ridge, compiled_tree):
    x, state = fold_design(groups, train, held)
    n = len(train)
    dims = state['group']
    # Equal total linear-kernel scale per descriptor family (not per coordinate).
    scale = 1 / np.sqrt(np.bincount(dims)[dims] * len(np.unique(dims)))
    y = torch.as_tensor(cache.y[train], device='cuda')
    w = torch.tensor([{'Good': 1.0, 'Fair': 0.42, 'Poor': 0.25}[q] for q in cache.quality[train]],
                     dtype=torch.float64, device='cuda')
    w /= w.mean()
    a = torch.as_tensor(x[:n], device='cuda')
    torch.cuda.synchronize()
    start = time.monotonic()
    slope, bias = compiled_ridge(a.double(), y, w, torch.as_tensor(scale, device='cuda'), 0.25)
    torch.cuda.synchronize()
    ridge_s = time.monotonic() - start
    state.update(ridge_slope=slope.cpu().numpy(), ridge_bias=bias.cpu().numpy())
    ptrend = x.astype(float) @ state['ridge_slope'] + state['ridge_bias']
    predictions = {'ridge': ptrend[n:]}
    cuts = np.quantile(x[:n], QUANTILES, axis=0).T.astype(np.float32)
    gates = (a.T[:, None, :] > torch.as_tensor(cuts, device='cuda')[:, :, None]).reshape(-1, n).float()
    gates = torch.cat((gates, gates.new_zeros((1, n))))
    # Reuse the same label-free candidate draws in the two boosted arms.
    rng = np.random.default_rng(seed)
    parts = [np.flatnonzero(dims == g) for g in np.unique(dims)]
    draws = []
    for _ in range(TREES):
        columns = np.concatenate([rng.choice(p, min(16, len(p)), replace=False) for p in parts])
        draws.append(np.r_[(columns[:, None] * len(QUANTILES) + np.arange(len(QUANTILES))).ravel(), len(gates)-1])
    candidates = torch.as_tensor(np.stack(draws), device='cuda')
    timing: dict[str, float | dict[str, float]] = {'ridge_s': ridge_s}
    weight = w.float()
    for mode in ('trees', 'hybrid'):
        base = np.full(len(x), float(w @ y / w.sum())) if mode == 'trees' else ptrend
        error = (y - torch.as_tensor(base[:n], device='cuda')).float()
        splits, values, times = [], [], []
        for ids in candidates:
            start = time.monotonic()
            decision, leaves, correction = compiled_tree(gates.index_select(0, ids), error, weight,
                                                          DEPTH, SHRINK, LEAF_PENALTY)
            error = error - correction
            splits.append(ids[decision])
            values.append(leaves)
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        timing[mode] = {'first_tree_s': times[0], 'median_steady_tree_s': float(np.median(times[1:])),
                        'all_trees_s': sum(times)}
        split = torch.stack(splits).cpu().numpy()
        feature = np.where(split == len(gates)-1, -1, split // len(QUANTILES))
        threshold = cuts[feature.clip(0), split % len(QUANTILES)]
        values_np = torch.stack(values).cpu().numpy()
        predictions[mode] = base[n:] + tree_prediction(x[n:], feature, threshold, values_np)
        state.update({f'{mode}_feature': feature, f'{mode}_cut': threshold, f'{mode}_leaves': values_np})
        if mode == 'trees':
            state['trees_intercept'] = np.asarray(base[0])
    return predictions, state, timing


def run(features: Path, output: Path, seeds: list[int]):
    cache = load_cache(features)
    output.mkdir(exist_ok=False)
    configure(output)  # CUDA required, suppress_errors=False
    groups = assemble(cache)
    ridge = torch.compile(ridge_trend, fullgraph=True, dynamic=False)
    tree = torch.compile(grow_tree, fullgraph=True, dynamic=False)
    modes = ('ridge', 'trees', 'hybrid')
    meta = {'schema': 'b1-thermal-boost/v1', 'features': str(features.resolve()),
            'source_meta_sha256': cache.source_digest, 'groups': GROUPS, 'quantiles': QUANTILES,
            'records': len(cache.rows), 'folds': 5, 'seeds': seeds, 'trees': TREES, 'depth': DEPTH,
            'shrinkage': SHRINK, 'leaf_penalty': LEAF_PENALTY, 'ridge_penalty': 0.25,
            'execution': 'torch.compile(fullgraph=True) CUDA ridge FP64 and tree growth FP32; NumPy held replay',
            'torch': torch.__version__, 'device': torch.cuda.get_device_name()}
    (output / 'run.json').write_text(json.dumps(meta, indent=2) + '\n')
    results = {}
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'fold', 'record', 'quality', 'target', *modes])
        for seed in seeds:
            assignment = record_folds(len(cache.rows), 5, seed)
            pred = {m: np.full(len(assignment), np.nan) for m in modes}
            trace = []
            for fold in range(5):
                train, held = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                estimate, state, timing = fit_fold(groups, cache, train, held, seed, ridge, tree)
                for m in modes:
                    pred[m][held] = estimate[m]
                np.savez_compressed(output / f'seed{seed}_fold{fold}.npz', allow_pickle=False, **state,
                                    train_records=cache.data['record'][train])
                for i in held:
                    writer.writerow([seed, fold, cache.data['record'][i], cache.quality[i], cache.y[i],
                                     *(pred[m][i] for m in modes)])
                stream.flush()
                trace.append(timing)
                print(f'seed={seed} fold={fold} timings={timing}', flush=True)
            results[str(seed)] = {'metrics': {m: scores(cache.y, pred[m], cache.quality) for m in modes},
                                  'trace': trace}
            print(results[str(seed)]['metrics'], flush=True)
            (output / 'metrics.json').write_text(json.dumps({**meta, 'results': results}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs=1, default=[1])
    args = parser.parse_args()
    run(args.features, args.output, args.seeds)
