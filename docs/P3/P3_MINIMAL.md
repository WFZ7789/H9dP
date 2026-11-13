# P3 initial minimal validation version

> 历史文档：保留当时方案与结果，不作为当前模型路线。命令默认从仓库根目录执行；
> 以后正式训练遵循 [AGENTS.md](../../AGENTS.md)，必须开启 torch.compile，旧 eager 设置仅为历史记录。

## Scope

P3 estimates one MOTIS pressure deficit dP (hPa) directly from six original-resolution
B08/B09/B10/B16 brightness-temperature fields. It does **not** retrieve fixed-level
temperatures, warm-core height or a temperature profile. Water-vapor channels have
scene-dependent weighting functions; learned features must not be presented as
verified layer-wise warm-core anomalies.

Implementation: `src/h9dp/p3.py`; configuration: `configs/p3_minimal.yaml`.
Architecture/checkpoint identifier: `p3-minimal/v1`.

Status (2026-11-01): the first full-data run stopped at epoch 40; epoch 20 was
best, with validation RMSE 15.069 hPa versus v2's 10.963 hPa. This is a failed
performance baseline, not a validated improvement. Results and read-only
ablations are documented in [P3_WORK_REPORT.md](P3_WORK_REPORT.md).

The default `configs/train.yaml` still selects the original v2 convolutional model.
No dependencies were added.

## Small model, original observations

The full P3 model has **108,551 trainable parameters**, including input mapping,
queries, attention, dynamic routers and the regression head. Construction rejects
any configuration with **155,000 or more** parameters. Fixed positional encodings,
anchors and target-normalization buffers are not trainable.

| Component | Parameters |
| --- | ---: |
| Per-pixel four-band linear input mapping, 4 → 64 | 320 |
| 16 learned queries | 1,024 |
| Two self-attention / cross-attention / FFN blocks | 100,480 |
| Two routers, each 64 → 32 → 3 | 4,358 |
| Readout normalization and 68 → 32 → 1 head | 2,369 |
| **Total** | **108,551** |

Each block has width 64, four heads and FFN width 128. All six frames share weights.
There is no CNN, learned spatial pre-encoder, large flattened-patch projection or
full pixel-to-pixel self-attention. Full-resolution images remain accessible;
64-dimensional embeddings are created only for observations actually read.

Each block performs:

1. Self-attention among 16 queries, then a background cross-attention read. The
   latter conditions routing on the input, not just a learned fixed location.
2. A shared per-frame router predicts x/y displacement and continuous scale using
   each query and the actual observation time offset. Anchors cover the window,
   with one reader initialized at its center. No label location or storm identity
   is supplied to the model.
3. Read **3×3 exact native pixels** around a pixel-snapped center, plus **5×5
   bilinear samples** over the predicted extent. Native reads preserve original
   pixel values; the bilinear branch supplies gradients for position and scale.
4. Jointly attend to these local reads and **8×8 masked raw-band background
   summaries per frame**. These non-learned averages are internal coarse reads,
   not a replacement for original images. Global coverage remains even when all
   chosen local samples are invalid. Both cross-attention calls share parameters.

At 500×500, the global memory has 384 observations and each query has 204 local
observations per block. The model does not construct a `[B,6,500,500,64]` feature
volume. A small parameter count alone would not ensure affordable attention;
this bounded read budget also limits computation.

## Numerical brightness path

- Keep the dataset's fixed training-only per-band means and standard deviations.
  No per-image centering or contrast normalization is added.
- Masks remove invalid values before pooling/sampling. Bilinear reads divide by
  interpolated valid coverage, avoiding artificial cooling near missing pixels.
- Last-block attention weights also directly aggregate the original four-band
  readings, bypassing learned value projections and LayerNorm. This four-value
  summary joins normalized latent features in the regression head. Read weights
  are renormalized in FP32, including under BF16 autocast.
- Head output uses the existing training-target normalization and positive scalar
  mapping. Main supervision remains record-equal normalized dP MSE; no brightness
  reconstruction task, quality weighting or pseudo-profile supervision is added.

This protects access to observations, **not** a guarantee that the network will
identify every eye or infer the correct physics. Sixteen sparse readers can still
miss a small feature. Read centers/scales and real-case error analysis must be
checked after authorized training. `model.eval(); model.inspect_reads(...)`
returns predictions, each block's per-frame centers/radii and final per-query
brightness summaries without storing large attention maps. Coordinates use x/y
in [-1, 1]; these diagnostics are not physical layer heights.

## Overfitting controls and comparison

- Small capacity; residual/FFN dropout 0.1; AdamW weight decay 0.001.
- Cross-attention probability dropout is zero to retain normalized numerical
  reads; residual dropout still applies to attention features.
- P3 uses at most 100 epochs and stops after 20 consecutive epochs without an
  improved validation RMSE. `run.json` records `early_stopped`; best and last
  checkpoints remain available. Early stopping is optional and defaults to
  disabled for the unchanged v2 configuration.
- Keep the prepared cache, record split, training-only normalization, synchronized
  augmentations and record-equal objective identical to v2 for the first
  comparison. No data is extracted, copied or relabeled for P3.
- The current 280/49 record split includes storms appearing in both sets, with
  some shared frames. This initial implementation deliberately does not change
  that baseline; it cannot establish unseen-storm generalization. A separate
  storm-grouped, frame-disjoint evaluation remains necessary.
- P3 output defaults to `/mnt/data/H9dP/runs/p3/`. It does not overwrite existing
  runs. Resume checks architecture as well as data/configuration/code fingerprints;
  a v2 checkpoint cannot be resumed as P3.

## Synthetic validation, not training

Use the Conda environment `fy3-aivis-v1`. Invoking `$CONDA_PREFIX/bin/python`
explicitly avoids a tool-managed Python elsewhere on PATH:

```bash
conda activate fy3-aivis-v1
"$CONDA_PREFIX/bin/python" -m pytest -q
"$CONDA_PREFIX/bin/python" -m pytest tests/test_p3.py -q
"$CONDA_PREFIX/bin/python" scripts/check_p3.py
"$CONDA_PREFIX/bin/python" scripts/check_p3.py --device cuda --bf16
```

`check_p3.py` is forward-only: synthetic `[1,6,4,500,500]` input, parameter count,
finite scalar output and routing shapes. It does not read the dataset, create an
optimizer, write outputs or fit a sample. Unit tests also check raw-pixel retention,
mask-normalized interpolation, BF16 value preservation, sparse embedding sizes,
content-dependent routing, synthetic gradients, strict weight roundtrips, the
parameter ceiling and optional early stopping. Gradient tests do not perform an
optimizer step. None of these checks measures predictive dP accuracy.

Static checks can use `ruff check src/h9dp/p3.py src/h9dp/model.py
src/h9dp/engine.py tests/test_p3.py scripts/check_p3.py`; LSP import resolution uses
`extraPaths: ["src"]` in `pyrightconfig.json`. Historical scripts under `runs/`
remain preserved and excluded from project source analysis.

## Training and sandbox access

**Run training and tests as a normal user by default.** Under AGENTS.md, request
normal-user execution outside the sandbox only after confirming that sandbox
restrictions prevent GPU use or access to required external data. Do not request
root/sudo privileges. If no approval entry point exists, no approval is required;
use the available normal-user tools and report any actual resource limitation
instead of blocking on approval. Commands below are documentation, not executed
by the synthetic check. Check actual/projected root usage
(<375,000,000,000 bytes) and bulk-storage capacity before launching:

```bash
df -B1 / /mnt/data
"$CONDA_PREFIX/bin/python" -u scripts/h9dp.py train --config configs/p3_minimal.yaml
"$CONDA_PREFIX/bin/python" -u scripts/h9dp.py train --config configs/p3_minimal.yaml \
    --resume /mnt/data/H9dP/runs/p3/ACTUAL_RUN/last.pt
```

The existing native500_v2 cache is reused; no `prepare` invocation is required
when it already matches the baseline data configuration. `torch_compile` is off
for the minimal version. The completed full-data run and its failure diagnosis are recorded in
[P3_WORK_REPORT.md](P3_WORK_REPORT.md); synthetic checks alone are not evidence
of predictive accuracy.
