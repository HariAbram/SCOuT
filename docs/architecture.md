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
- `src/metrics.py`: `perf`, `likwid`, and `parser` backends
- `src/misc.py`: shared sampling and utility helpers
- `src/searchMethods/`: heuristic search implementations

## Search dispatch

`main.py` dispatches these study names directly:

- `wavefront`
- `tabu`
- `beam_tabu`
- `anneal`

Any other `search.study` value falls back to Optuna.

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

All search methods use the same build-then-measure pattern, even if they generate candidates differently.

## Outputs

SCOuT can produce:

- CSV logs
- Pareto CSVs for multi-objective Optuna runs
- failed-build logs
- temporary work directories under `/tmp`

