"""Pair record-level OOF forecasts and test a fixed, image-only registration gate."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from h9dp.r1 import record_folds, scores
from scripts.r1_kernel import Cache, load_cache, read_json


def registration_summary(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median supported match NCC (including rejected matches); no-match fallback to xy."""
    if trace.ndim != 3 or trace.shape[2] != 3:
        raise ValueError("Expected the tracked [record,frame,(shift,NCC,accepted)] trace")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        ncc = np.nanmedian(trace[..., 1], axis=1)
    any_move = (trace[..., 2] == 1).any(axis=1)
    weight = np.where(any_move & np.isfinite(ncc), np.clip((ncc - 0.80) / 0.18, 0, 1), 0)
    return ncc, weight


def read_oof(path: Path, cache: Cache, seeds: list[int], folds: int) -> dict[int, np.ndarray]:
    """Require a bijection to the cached records and the same outer split, independent of CSV order."""
    lookup = {int(record): i for i, record in enumerate(cache.data['record'])}
    if len(lookup) != len(cache.rows) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected unique records and distinct seeds in {path}")
    split = {s: record_folds(len(lookup), folds, s) for s in seeds}
    preds = {s: np.full(len(lookup), np.nan) for s in seeds}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            seed, record = int(row['seed']), int(row['record'])
            if seed not in preds or record not in lookup:
                raise ValueError(f"Unexpected seed/record in {path}: {seed}/{record}")
            i = lookup[record]
            p = float(row['spectral_gp'])
            if (np.isfinite(preds[seed][i]) or not np.isfinite(p) or int(row['fold']) != split[seed][i]
                    or float(row['target']) != cache.y[i] or row['quality'] != cache.quality[i]):
                raise ValueError(f"Duplicate, nonfinite or misaligned forecast: {seed}/{record} in {path}")
            preds[seed][i] = p
    if not all(np.isfinite(p).all() for p in preds.values()):
        raise ValueError(f"Incomplete OOF coverage in {path}")
    return preds


def rotation_error(predictions: np.ndarray, target: np.ndarray) -> dict:
    """Descriptive MSE decomposition on the *same* records, not a noise floor."""
    if predictions.ndim != 2 or predictions.shape[1:] != target.shape or not predictions.size:
        raise ValueError("Expected nonempty [seed,record] forecasts and aligned targets")
    errors = predictions - target
    persistent = errors.mean(axis=0) ** 2
    k = max(1, int(np.ceil(0.1 * target.size)))
    return {'records': target.size, 'seeds': len(errors), 'mean_mse': float(np.mean(errors ** 2)),
            'squared_mean_error': float(persistent.mean()), 'split_variance': float(errors.var(axis=0).mean()),
            'top_decile_n': k, 'top_decile_share_of_persistent_sse': (
                float(np.sort(persistent)[-k:].sum() / persistent.sum()) if persistent.sum() > 0 else None)}


def _protocol(meta):
    keys = ('schema', 'execution', 'folds', 'steps', 'lr', 'tail_focus', 'snapshot', 'spectral', 'contour')
    return tuple(meta[k] for k in keys) + (meta.get('warm_column', False), meta.get('response', 'none'),
                                           meta.get('latitude_mode', 'none'), meta.get('latitude_index_sha256'),
                                           meta.get('aperture', False))


def audit(xy: Path, tracked: list[Path], output: Path) -> None:
    if not tracked:
        raise ValueError("Expected at least one tracked run")
    runs = [xy, *tracked]
    info = [read_json(run / 'run.json')[0] for run in runs]
    cache = load_cache(Path(info[1]['features']))
    tracked_manifest, _ = read_json(cache.root / 'manifest.json')
    forecasts = []
    for index, (run, meta) in enumerate(zip(runs, info, strict=True)):
        features = Path(meta['features'])
        manifest, _ = read_json(features / 'manifest.json')
        # Early v2 fixed-center manifests predate the tracking flag.
        if (manifest.get('tracking', False) is not (index != 0) or manifest['source_meta_sha256'] != cache.source_digest
                or manifest.get('native_r3_index_sha256') != tracked_manifest.get('native_r3_index_sha256')
                or meta['source_meta_sha256'] != cache.source_digest or meta['records'] != len(cache.rows)
                or meta['schema'] != 'r1-spectral-gp-development/v1'
                or _protocol(meta) != _protocol(info[0])
                or (index > 0 and features.resolve() != cache.root.resolve())):
            raise ValueError(f"Incompatible run or feature provenance: {run}")
        forecasts.append(read_oof(run / 'predictions.csv', cache, meta['seeds'], meta['folds']))
    moving = {}
    for part in forecasts[1:]:
        if moving.keys() & part.keys():
            raise ValueError("Repeated seed across tracked runs")
        moving.update(part)
    if moving.keys() != forecasts[0].keys():
        raise ValueError("Need matching seeds on both sides")
    ncc, weight = registration_summary(cache.data['registration'])
    report = {'schema': 'r1-tracking-oof-audit/v1', 'runs': [str(run.resolve()) for run in runs],
              'source_meta_sha256': cache.source_digest, 'folds': info[0]['folds'], 'results': {}}
    bands = {'low': ncc < 0.85, 'middle': (ncc >= 0.85) & (ncc < 0.95), 'high': ncc >= 0.95,
             'unsupported': ~np.isfinite(ncc)}
    output.mkdir(exist_ok=False)
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'fold', 'record', 'quality', 'target', 'median_ncc', 'weight', 'xy', 'tracked',
                         'half', 'gated'])
        for seed, fixed in forecasts[0].items():
            moved = moving[seed]
            methods = {'xy': fixed, 'tracked': moved, 'half': (moved + fixed) / 2,
                       'gated': fixed + weight * (moved - fixed)}
            splits = record_folds(len(fixed), info[0]['folds'], seed)
            for i, record in enumerate(cache.data['record']):
                writer.writerow([seed, splits[i], record, cache.quality[i], cache.y[i], ncc[i], weight[i],
                                 *(p[i] for p in methods.values())])
            metrics = {name: scores(cache.y, p, cache.quality) for name, p in methods.items()}
            by_ncc = {name: {k: scores(cache.y[mask], p[mask], cache.quality[mask]) for k, p in methods.items()}
                      for name, mask in bands.items() if mask.any()}
            report['results'][str(seed)] = {'metrics': metrics, 'by_median_ncc': by_ncc}
    halves = np.stack([(p + moving[seed]) / 2 for seed, p in forecasts[0].items()])
    decomposition = {'all': rotation_error(halves, cache.y)}
    for name in np.unique(cache.quality):
        mask = cache.quality == name
        decomposition[name] = rotation_error(halves[:, mask], cache.y[mask])
    report['half_rotation_error'] = decomposition
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xy', type=Path, required=True)
    parser.add_argument('--tracked', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.xy, args.tracked, args.output)


if __name__ == '__main__':
    main()
