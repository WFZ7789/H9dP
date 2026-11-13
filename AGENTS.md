# Repository Guidelines

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

## Coding Style & Naming Conventions

Use four-space indentation in Python, `snake_case` for modules and functions,
`PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Prefer C++ for
performance-critical components when justified by profiling. Keep interfaces
between Python and C++ explicit. No formatter or linter is configured; record
new tooling and its invocation when introduced. Satpy is FORBIDDEN.

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
