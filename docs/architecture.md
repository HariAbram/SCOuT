# SCOuT Architecture

## Overview

SCOuT explores compiler and runtime configurations for a target program. Each run follows the same basic loop:

1. Load a JSON config.
2. Generate a candidate configuration.
3. Build the target.
4. Measure the result.
5. Record metrics and continue searching.

## Main components

- `main.py`: CLI entry point and search dispatch
- `src/config.py`: config loading and validation
- `src/explore.py`: Optuna driver and wrappers for custom searches
- `src/build.py`: single-source, CMake, and Make builds
- `src/evaluator.py`: shared build/evaluation cache and measurement dispatch
- `src/metrics.py`: `perf`, `likwid`, and `parser` backends
- `src/misc.py`: shared sampling and utility helpers
- `src/searchMethods/`: heuristic search implementations
- `src/polyMorph/`: optional Tadashi-based SYCL transformation workflow

## Search dispatch

`main.py` dispatches these study names directly:

- `wavefront`
- `tabu`
- `beam_tabu`
- `anneal`

Any other `search.study` value falls back to Optuna.

`main.py` also exposes a separate `polymorph` mode for Tadashi-based transformation search. That path is configured through a top-level `polyMorph` block rather than the regular SCOuT build/search fields. The package is imported lazily only after that mode is selected, so missing Tadashi dependencies cannot break the normal parameter-tuning CLI.

Core and optional dependency boundaries:

- `requirements.txt` contains only dependencies needed by parameter tuning.
- `requirements-polymorph.txt` includes the core requirements and Tadashi.
- Parameter-tuning code must not import `src.polyMorph`.
- `src.polyMorph.__init__` is a lazy facade; importing it does not load the runner.

## Config shape

Each config must define:

- one build target: `source` or `project`
- one backend: `perf`, `likwid`, or `parser`
- at least one objective

Common fields:

- `compiler`
- `compiler_flags_base`
- `compiler_flags`
- `compiler_flag_pool`
- `compiler_params`
- `compiler_params_select`
- `program_args`
- `env`
- `objectives`
- `search`
- `runs`

## Build and measurement

Supported build modes:

- direct compilation of a single source file
- CMake projects
- Make projects

Supported measurement backends:

- `perf`
- `likwid`
- `parser`

All search methods use the shared evaluator. Builds are cached by compiler inputs, independently from runtime-environment evaluations. AdaptiveCpp runtime-cache reuse is the default and cold-cache experiments are opt-in.

## Outputs

SCOuT can produce:

- CSV logs
- Pareto CSVs for multi-objective Optuna runs
- failed-build logs
- temporary work directories under `/tmp`
