# SCOuT

SCOuT is a Python-based design-space exploration tool for compiler flags and runtime environment variables. It builds either a single source file or a full project, runs the produced binary under a measurement backend, and searches for better configurations using Optuna or several custom heuristic search strategies.

The repository has two major parts:

- The Python orchestration layer in `main.py` and `src/`
- A large collection of benchmark projects and datasets in `test-benchmarks/`

## What SCOuT does

At a high level, each experiment follows the same loop:

1. Load a JSON configuration file.
2. Sample a compiler configuration and, optionally, runtime environment variables.
3. Build the target binary.
4. Measure the binary with one of the supported backends:
   - `perf`
   - `likwid`
   - `parser` for programs that emit `[SYCL][avg]` / `[SYCL][sum]` timing lines
5. Extract one or more objective metrics.
6. Feed the result back into the selected search strategy.
7. Write CSV logs and optional Pareto-front output.

## Repository map

- `main.py`: CLI entry point and mode dispatch.
- `src/config.py`: Typed config loading and validation.
- `src/build.py`: Build helpers for single-source, CMake, and Make projects.
- `src/metrics.py`: Measurement and parsing backends.
- `src/explore.py`: Optuna study driver plus wrappers for custom search methods.
- `src/misc.py`: Sampling helpers, cache cleanup, and utility functions.
- `src/searchMethods/`: Custom search strategies:
  - `wavefront_flags.py`
  - `tabu_flags.py`
  - `beam_tabu.py`
  - `anneal_flags.py`
- `configs/dse_config.json`: Example configuration.
- `test-benchmarks/`: Benchmark programs and datasets used as tuning targets.

## Entry point

Run SCOuT with:

```bash
python3 main.py --mode parameter_tuning configs/dse_config.json --trials 100
```

If you omit `--mode` in an interactive terminal, SCOuT prompts for it and currently defaults to `parameter_tuning`.

Useful CLI arguments:

- `config`: Path to the JSON config file.
- `--trials`: Number of Optuna trials. Defaults to `50` when omitted.
- `--interactive`: Prompt for missing values.
- `--pareto-log`: Reserved flag; current Pareto logging is controlled by config values.

## Search methods

The active search method is chosen from `search.study` in the config:

- `optuna`: Multi-objective or single-objective Optuna search.
- `wavefront`: Layered flag-combination search over `wavefront.flag_atoms`.
- `tabu`: Neighborhood search over variants, parameters, pool flags, and environment settings.
- `beam_tabu`: Beam expansion plus tabu filtering.
- `anneal`: Simulated annealing over the flag/environment state.

Important implementation note:

- `main.py` only dispatches explicitly for `wavefront`, `tabu`, `beam_tabu`, and `anneal`.
- Any other `search.study` value currently falls back to `explore_optuna()`.
- The sample config uses `"study": "synergy"`, but there is no `src/searchMethods/synergy_flags.py` in this repository, so that value behaves like Optuna today.

## Configuration model

Each config must define exactly one build target:

- `source`: Path to a single source file to compile directly
- `project`: A buildable project description for `cmake` or `make`

Common top-level config sections:

- `backend`
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
- `csv_log`
- backend-specific block: `perf`, `likwid`, or `parser`

Detailed field-by-field documentation lives in [docs/architecture.md](/home/hari/git/SCOuT/docs/architecture.md).

## Measurement backends

### `perf`

Runs the binary under `perf stat`, parses raw event counters, and derives `CPI` if both `cycles` and `instructions` are present.

### `likwid`

Runs the binary under `likwid-perfctr` and extracts configured metrics from LIKWID tables. Aggregation modes include `avg`, `max`, `min`, and `median`.

### `parser`

Runs the binary normally and parses lines of the form:

```text
[SYCL][avg] kernel 2: 0.000664 s over 1000 iters
```

This backend is meant for applications that already print structured SYCL timing information.

## Build behavior

SCOuT supports:

- Direct compilation of a single file with the configured compiler and flags
- CMake projects via `cmake -S ... -B ...` and `cmake --build`
- Make projects via `make`

The build helpers write stdout/stderr logs for failing build steps into per-run `logs/` directories under the generated working directory.

## Outputs

Depending on the selected strategy and config, SCOuT can write:

- Trial or iteration CSV logs
- Pareto front CSVs for multi-objective Optuna studies
- Failed-build logs
- Temporary working directories containing build artifacts and command logs

Custom search methods usually create a temporary work root under `/tmp` with names such as:

- `SCOuT_*`
- `SCOuT_tabu_*`
- `SCOuT_beamtabu_*`
- `SCOuT_anneal_*`

## Current caveats

- `requirments.txt` is misspelled and currently contains shell text as well as a package pin; it is not a standard pip requirements file.
- The example config includes fields for multiple search methods at once; only the selected study is used during a run.
- `beam_tabu` is implemented in `src/searchMethods/beam_tabu.py`, but `Config` does not currently expose a typed `beam_tabu` field, so that module falls back to defaults unless the config object is extended.
- `explore_synergy()` exists in `src/explore.py`, but the referenced module is absent.

## Detailed docs

For a deeper walkthrough of the code, configuration schema, execution flow, and implementation caveats, see [docs/architecture.md](/home/hari/git/SCOuT/docs/architecture.md).

For a dedicated explanation of the implemented search methods, see [docs/search-algorithms.md](/home/hari/git/SCOuT/docs/search-algorithms.md).
