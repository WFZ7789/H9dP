# Repository Guidelines

## 首要永久规则：全项目只用种子 1

根目录 [SEED](SEED) 记录项目随机种子，永久固定为 **1**。今后数据抽样、划分、
训练、评估和随机诊断都使用这一个实验种子。一个种子跑出什么结果，就如实报告
什么结果。全项目禁止多种子对照、挑选种子和跨种子平均。只有用户对具体任务
作出明确的特殊说明时，才按其指定的范围例外执行。历史运行记录保留当时的实际种子。

## Project Structure & Module Organization

Python modules live under `src/h9dp/`, tests under `tests/`, runnable utilities
under `scripts/`, and configuration under `configs/`. Reserve `cpp/` for future
profile-justified C++ components. Preserve the original archive and datasets.
Store large datasets and outputs outside the repository, preferably under
`/mnt/data/H9dP/`.

## Documentation & Experiment Reports

- Put model plans, designs, training protocols, audits, and textual results under
  `docs/<model_name>/`; use `docs/` for genuinely shared documentation. Do NOT add
  these documents to the repository root. Root Markdown entry points are only
  `README.md` and `AGENTS.md`.
- Maintain `docs/README.md` as the documentation index and link it from the root
  README. Each model directory should have its own README linking plans/results.
  Separate substantial result reports from plans, e.g. `docs/T1A/PLAN.md` and
  `docs/T1A/FIT24_RESULTS.md`. T1A means **test-1 a**.
- Reports must distinguish planned/running/completed work, synthetic vs real
  data, training-subset fitting vs validation/test/generalization, and eager vs
  compiled execution. Record the actual protocol, commands, seeds, record count,
  metrics, and external artifact location. Never present an unexecuted plan as
  an experimental result.
- Markdown links are relative to the document containing them. Shell commands
  use the repository root as working directory unless explicitly stated. Fix
  incoming and outgoing links when moving documents; check with
  `tests/test_docs.py`.
- Keep checkpoints, data, raw logs, and machine-generated run artifacts outside
  Git under `/mnt/data/H9dP/`; link to their locations from textual reports.
  Do not relocate or edit immutable historical run/source snapshots when
  reorganizing the live documentation.

## Build, Test, and Development Commands

Use the Conda environment `fy3-aivis-v1`:

```bash
conda activate fy3-aivis-v1
python --version                 # Check the active interpreter.
git status --short               # Review pending changes.
df -B1 / /mnt/data               # Check filesystem usage in bytes.
```

Run the suite with `conda run -n fy3-aivis-v1 python -m pytest -q`. If tools
shadow `python`, use the environment interpreter explicitly, e.g.
`MKL_THREADING_LAYER=GNU "$CONDA_PREFIX/bin/python" -m pytest -q` after activation.
Use Ruff on changed Python paths (`ruff check <changed-paths>`); no formatter is
configured. Model-specific commands and actual results belong in `docs/`.
For C++ components, document the selected compiler and build system before
requiring their commands.

## Network Operations

Before any operation that can reach the network (data downloads, dependency
installation, dataset or checkpoint fetches), unset every environment variable
whose name contains `PROXY` in any case. Never run such an operation with a
proxy configured.

```bash
for name in $(env | cut -d= -f1 | grep -i proxy); do unset "$name"; done
```

Apply the same rule to scripts and subprocesses: do not pass proxy variables
through to child processes, for example by launching them with explicit
`env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY`.
Record the unset step in the run record so the network path stays reproducible.

## Coding Style & Naming Conventions

Use four-space indentation in Python, `snake_case` for modules and functions,
`PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Prefer C++ for
performance-critical components when justified by profiling. Keep interfaces
between Python and C++ explicit. No formatter or linter is configured; record
new tooling and its invocation when introduced. Satpy is FORBIDDEN.

## Permanent Observation/Test-Set Policy

**This is a permanent, project-wide rule for current and future collection,
validation, testing, and reporting, not an exception for one experiment.**

- **真实的泛化能力，最终交给未来的独立观测和长期业务运行来检验。**
  Future independent observations and long-term operational use are the ultimate
  test of real generalization; do not impose cyclone independence on the current
  observed/test set as a substitute for that future evidence.
- **任何气旋（包括训练集、验证集、测试集中的气旋）均可被纳入实测集（测试集）。**
  Cyclone identity and year are metadata tags of equal standing, not grouping,
  exclusion, quota, or split-binding units. Their presence in another set does
  not make an observation ineligible or invalidate its observed-set result.
- **气旋名称、与训练记录不超过 6 h 的时间间隔，不构成任何影响。**
  Do not reject, downweight, cap, bind splits, or discount records/results because
  they share a cyclone name or are within six hours of training records. This
  includes exact named-time matches, which are recorded as metadata, not an
  automatic exclusion. Do not introduce a different time buffer instead.
- Select and evaluate at the observation/record level. Do not introduce
  cyclone-held-out collection, per-cyclone quotas, cyclone-balanced sampling, or
  mandatory within-cyclone spacing unless the user explicitly changes this policy.
  Only deduplicate actual repeated copies of an observation within the collected
  set; distinct observations are not duplicates merely because they share a
  cyclone, hour, or proximity to a training/validation/test record.
- Report the actual observation source, protocol, and metrics. Do not call an
  observed-set result invalid merely because of cyclone overlap or <=6 h training
  proximity; do not claim it already proves future operational generalization.
- Preserve immutable historical run protocols, results, and snapshots. Older
  cyclone-grouped/time-isolated protocols describe historical work and do not
  override this permanent rule for future work.

### Current Useful-Pressure Collection Target

The current collection target is **dP > 30 hPa**. Weak-pressure rows do not count
as the requested useful set. Surface-station proxy pressure deficits must not be
presented as verified cyclone-central-pressure observations. This target is
separate from the permanent eligibility policy above.

**JMA observations include actual aircraft/dropsonde, station/surface, marine
measurements and observation reports, NOT JMA BST.** Include observations JMA
publishes, receives, or obtains through collaborations; do not restrict searches
to ground-station tables or confuse agency ownership with observation provenance.
For aircraft sources, preserve pressure type, measurement altitude, provider,
original time precision and quality. Flight-level pressure is not sea-level or
cyclone-central pressure. Distinguish native records, published observational
estimates, and ancillary files without pressure; do not inflate their counts. Never use JMA BST or other best-track central-pressure
analyses as measured-pressure labels, truth, or substitutes to fill an observed
sample quota. Preserve original measured station pressure, published sea-level
pressure, quality flags, and observation times (JST must be converted to UTC).
The previously approved IBTrACS USA_POCI and MOTIS/ERA5 POCI remain separately
labelled environmental fields only; they must never generate or replace an
observed pressure. Published adopted/interpolated labels are not raw measurements
unless their original observation provenance is actually retrieved.

## Testing Guidelines

The pytest suite uses `tests/test_<module>.py` naming. Add focused tests for new
behavior, especially dataset parsing, shapes, numerical results, and Python/C++
interfaces. No coverage threshold is configured. Use small synthetic fixtures
so routine checks do not require archive extraction or supervised model fitting.
Real-data fitability/training checks are explicit standalone experiments, not
ordinary pytest cases. Document suite commands and results under `docs/`.

## Commit & Pull Request Guidelines

There are no commits from which to infer message conventions. Use concise,
imperative subjects such as `Add dataset metadata reader`. Keep commits focused.
Pull requests should explain the change, validation performed, dependencies,
and any data or configuration effects. Link relevant issues when available.
Exclude extracted data, checkpoints, generated outputs, and caches from Git.

## Storage & Training Requirements

All future formal training runs MUST enable `torch.compile`. Keep BF16 and
activation checkpointing where configured. Do not silently fall back to eager
execution if compilation fails; report and resolve the actual failure first.
Report compilation/cold-start overhead separately from steady-state timing.
Explicit diagnostic/fitability experiments may retain their declared eager
protocol; do not change an already-running experiment mid-run.

Root filesystem `/` usage must never exceed **375 GB**. Check usage and projected
space before extraction, installation, or large writes. Prefer `/mnt/data` for
bulk storage; it is a mechanical disk, so account for slower random access.
Run training and tests as a normal user in the current environment by default.
Only request execution outside the sandbox, still as a normal user, after
confirming that sandbox restrictions prevent GPU use or access to required
external data. This is not a request for root or sudo privileges. Do not request
approval merely because training is launched, resumed, scripted, or backgrounded.
If no permission/approval entry point is available, no approval is required; use
the available normal-user execution tools instead of blocking on approval. If
GPU or data access remains unavailable, report the actual resource limitation
rather than treating missing approval as the cause.
