# SCOuT Architecture And Usage Guide

## Overview

SCOuT is a design-space exploration framework for compile-time and runtime tuning. Its job is to take a workload, generate candidate compiler and environment configurations, build the workload, measure the result, and keep searching for better configurations according to one or more objectives.

The codebase is small and centered around a single execution pipeline:

1. Parse CLI arguments in `main.py`.
2. Load and validate JSON into a `Config` object in `src/config.py`.
3. Dispatch to a search driver in `src/explore.py`.
4. Build a binary through `src/build.py`.
5. Measure it through `src/metrics.py`.
6. Record the resulting metrics and continue the search.

## High-Level Execution Flow

### 1. CLI layer: `main.py`

`main.py` is the only entry point. Right now it exposes one mode, `parameter_tuning`, but the structure is intentionally prepared for future modes.

Relevant helpers:

- `_prompt_path()`: interactive path prompt
- `_prompt_int()`: interactive integer prompt
- `_fmt_dur()`: wall-time formatting
- `_run_from_config()`: maps `search.study` to a search driver

Dispatch behavior:

- `wavefront` -> `explore_wavefront(cfg)`
- `tabu` -> `explore_tabu(cfg)`
- `beam_tabu` -> `explore_beam_tabu(cfg)`
- `anneal` -> `explore_anneal(cfg)`
- anything else -> `explore_optuna(cfg, trials)`

That last rule matters: unknown study names silently fall back to Optuna.

### 2. Config loading: `src/config.py`

`Config.load()` turns JSON into a mostly typed dataclass graph.

Key design decisions:

- Exactly one of `source` or `project` must be provided.
- `backend` must be one of `perf`, `likwid`, or `parser`.
- `program_args` are normalized into a flat list.
- `objectives` is a list, so multi-objective optimization is supported natively.
- Environment variables can be unconditional lists or conditional objects with `when` predicates.

The main dataclasses are:

- `Objective`
- `PerfConfig`
- `MetricSpec`
- `ParserConfig`
- `LikwidConfig`
- `BuildProject`
- `SearchSpec`
- `WavefrontSpec`
- `Config`

### 3. Search driver: `src/explore.py`

`src/explore.py` contains:

- the Optuna implementation in `explore_optuna()`
- thin wrappers that import and run custom search strategies

This file is effectively the bridge between typed configuration and actual experiment execution.

### 4. Build stage: `src/build.py`

Builds happen in one of two ways:

- `compile_single_source()`: direct compiler invocation for a source file
- `compile_project()`: delegated build for `cmake` or `make` projects

Supporting helpers:

- `_run()`: subprocess wrapper that prints commands and captures stdout/stderr
- `_save_log()`: stores command output for failing steps
- `_last_executable()`: best-effort search for the latest executable in a build tree

### 5. Measurement stage: `src/metrics.py`

There are three backends:

- `measure_perf()`
- `measure_likwid()`
- `measure_parser_sycl()`

All three follow the same model:

1. Run optional warm-up iterations.
2. Run measured iterations.
3. Parse data into a `MetricDict`.
4. Average per-run values.
5. Clear the AdaptiveCpp runtime cache via `clear_acpp_runtime_cache()`.

## Search Strategies

## Optuna: `explore_optuna()`

This is the most complete and most central implementation.

What it does:

- Creates a temporary work directory.
- Chooses an Optuna sampler from `search.sampler`.
- Supports multi-objective studies through `directions`.
- Samples compiler flags and environment settings.
- Caches duplicate evaluations to avoid rebuilding identical configurations.
- Builds and measures the target.
- Stores metrics, flags, environment, and binary path in `trial.user_attrs`.
- Exports CSV logs and Pareto fronts.

Sampler support:

- `tpe`
- `nsga3`
- `rs`
- `cmaes`

Compiler sampling is delegated to `src/misc.py`:

- `suggest_compiler_flags()`
- `suggest_env()`
- `_complexity_limits()`
- `_select_param_subset()`
- `_select_pool_flags()`

Important Optuna behavior:

- During early trials, the code intentionally limits how many parametric flags and pool flags can be activated.
- Duplicate configurations are detected by `(flag_key, sorted_env_items)` and reused from an in-memory cache.
- For multi-objective studies, Pareto CSVs are updated after each completed trial.

## Wavefront: `src/searchMethods/wavefront_flags.py`

Wavefront search explores combinations of flag atoms in layers by combination size.

Conceptually:

- Wave `k=0`: evaluate the baseline
- Wave `k=1`: evaluate one additional atom at a time
- Wave `k=2`: evaluate pairs
- continue until `max_k`

Modes:

- `full`: evaluate all combinations at a given depth
- `beam`: only expand the best candidates from the previous wave

Wavefront-specific features:

- environment-combo generation with predicate-aware enumeration
- optional per-wave cap
- significance-based early stopping
- CSV export of evaluated candidates

The implementation uses the first objective only for ranking.

## Tabu: `src/searchMethods/tabu_flags.py`

Tabu search treats a configuration as a mutable state:

- variant flag
- selected parametric flags and their values
- selected pool flags
- selected environment assignment

The search repeatedly samples a neighborhood, evaluates admissible neighbors, and uses a bounded tabu list to avoid cycling.

Move families can be enabled or disabled independently:

- variant moves
- param moves
- pool moves
- environment moves

The implementation also supports significance thresholds to distinguish meaningful improvement from noise.

## Beam + Tabu: `src/searchMethods/beam_tabu.py`

This strategy combines:

- beam-style frontier expansion
- tabu-style move blocking

It focuses primarily on combinations of flag atoms layered on top of `base_flags`.

Important caveat:

- The implementation reads `cfg.beam_tabu`, but `Config` does not currently declare or populate a `beam_tabu` field. In practice that means this module tends to run with its own defaults unless the config object is extended elsewhere.

## Simulated annealing: `src/searchMethods/anneal_flags.py`

The annealing search mutates:

- variant
- param selections
- pool flags
- environment choice

Acceptance behavior follows the standard simulated annealing pattern:

- always accept better states
- sometimes accept worse states with temperature-dependent probability

Key annealing parameters:

- `T0`
- `alpha`
- `max_iters`
- `max_no_improve`
- `neighbor_mode`
- `env_mode`
- `env_cap`

## Configuration Reference

## Required top-level fields

### `backend`

Selects the measurement backend:

- `perf`
- `likwid`
- `parser`

### `source` or `project`

Exactly one must be set.

`source`:

- path to a single file compiled directly

`project`:

- `dir`: project root
- `build_system`: `cmake` or `make`
- `target`: optional build target or executable name
- `make_vars`: extra `make` variables
- `make_flags_var`: variable name that receives compiler flags
- `cmake_defs`: extra `-D...` definitions
- `cmake_flag_vars`: CMake cache variables to append flags into

### `compiler`

Compiler executable, for example `acpp`, `clang++`, or `dpcpp`.

### `compiler_flags_base`

Always-on baseline flags. This is the foundation under every sampled configuration.

### `compiler_flags`

A list of mutually selectable variant strings. Optuna samples one of them; the custom search methods may treat them as a variant dimension or as extra atoms depending on the module.

### `compiler_flag_pool`

A list of optional on/off flags that may be added on top of the baseline and variant.

### `compiler_params`

Parametric options such as:

```json
"-march": ["native", "znver4", "skylake-avx512"]
```

Supported forms:

- plain list of values
- object with `values` and optional `sep`

Example:

```json
"-mprefer-vector-width": {
  "values": [128, 256, 512],
  "sep": "="
}
```

### `compiler_params_select`

Controls how many parametric options are active at once.

Supported keys:

- `k`
- `min`
- `max`
- `always`

Important rule:

- do not mix `k` with `min`/`max`

### `program_args`

Program command-line arguments. This can be:

- a shell-style string
- a list of strings

`Config.load()` normalizes it into `List[str]`.

### `env`

Defines the runtime environment search space.

Supported forms:

- unconditional list
- conditional object with `when` and `values`

Example:

```json
"OMP_PLACES": {
  "when": { "ACPP_VISIBILITY_MASK": "omp" },
  "values": ["cores", "threads"]
}
```

Conditional variables are only assigned when their predicate matches the already chosen environment.

### `objectives`

List of metrics to optimize.

Example:

```json
"objectives": [
  { "metric": "CPI", "goal": "min" },
  { "metric": "Runtime (RDTSC) [s]", "goal": "min" }
]
```

Goals:

- `min`
- `max`

If `objectives` is missing, the loader falls back to the backend-local `objective` block for backward compatibility.

### `search`

General search configuration:

- `study`
- `sampler`
- `n_startup_trials`
- `population_size`
- `random_seed`

Typical values:

- `study`: `optuna`, `wavefront`, `tabu`, `beam_tabu`, `anneal`
- `sampler`: `tpe`, `nsga3`, `rs`, `cmaes`

### `runs`

Number of measured executions per candidate after any warm-up runs.

### Logging fields

- `csv_log`
- `pareto_log`
- `failed_builds`
- `sqlite_log`

Note that `Config` maps `failed_builds` into `cfg.fail_log`.

## Backend-specific config

## `perf`

Fields:

- `events`
- `core_list`
- `warmup_runs`

Behavior:

- uses `perf stat`
- parses counts from stderr
- derives `CPI` when possible

## `likwid`

Fields:

- `group`
- `events`
- `metrics`
- `core_list`
- `warmup_runs`

Each `metrics` entry may be:

- a plain string
- an object with:
  - `name`
  - `agg`
  - `var`

Aggregation modes:

- `avg`
- `max`
- `min`
- `median`

## `parser`

Fields:

- `label`
- `kernels`
- `aggregate`
- `warmup_runs`
- `core_list`
- `prefix`
- `run_cwd`

`run_cwd` may be:

- `binary_dir`
- `project_dir`
- `workdir`

Expected runtime output:

```text
[SYCL][avg] kernel 0: 0.001234 s over 1000 iters
```

Produced metric names:

- `sycl_kernel_<id>_<label>_s`
- `sycl_<label>_<aggregate>_s`
- `sycl_iters`

## Output Files And Working Directories

## Temporary work roots

Most runs create a temporary root under `/tmp`, for example:

- `SCOuT_xxxxxxxx`
- `SCOuT_tabu_xxxxxxxx`
- `SCOuT_beamtabu_xxxxxxxx`
- `SCOuT_anneal_xxxxxxxx`

Inside those directories you will find:

- build directories
- compiled binaries
- per-step command logs
- parser failure logs
- generated CSVs for custom strategies

## CSV logs

Optuna writes rows containing:

- objective values
- compiler flags
- serialized environment
- binary path
- extra metrics discovered during measurement

Custom search methods write similar strategy-specific CSVs.

## Pareto CSV

For multi-objective Optuna studies, SCOuT writes the current Pareto-optimal set to CSV if `pareto_log` is configured.

## Important Implementation Notes

## 1. Unknown `search.study` values fall back to Optuna

This is current behavior in `main.py`. It is convenient, but it also means typos or stale config values do not fail fast.

## 2. The sample config uses `study: "synergy"`

There is an `explore_synergy()` wrapper in `src/explore.py`, but the repository does not contain `src/searchMethods/synergy_flags.py`. Because `main.py` does not dispatch `"synergy"` explicitly, the sample config currently behaves as an Optuna run.

## 3. `beam_tabu` config is only partially wired

`src/searchMethods/beam_tabu.py` expects `cfg.beam_tabu`, but `Config` does not currently expose that field. The JSON example includes a `beam_tabu` block, but the loader does not store it on the `Config` dataclass.

## 4. Requirements setup needs cleanup

The repository contains `requirments.txt`, not `requirements.txt`, and its contents are not a normal pip-only requirements file.

Current contents:

- `optuna==4.4.0`
- `pip3 install cmaes`

That second line is an installation command, not a requirements entry.

## 5. Search-method behavior is not fully uniform

The modules agree on the build-measure-evaluate pattern, but they do not all consume config in exactly the same way:

- Optuna uses the typed `Config` fields most directly.
- Wavefront uses a typed `WavefrontSpec`.
- Tabu and anneal consume raw dict-like blocks more loosely.
- Beam-tabu currently depends on a field the loader does not provide.

## Suggested Mental Model For The Code

When reading or extending SCOuT, the simplest way to think about it is:

- `main.py` decides what kind of run to start.
- `Config.load()` translates JSON into structured Python objects.
- `src/misc.py` defines how candidate configurations are sampled or compared.
- `src/build.py` turns a candidate into a binary.
- `src/metrics.py` turns a binary execution into metrics.
- `src/explore.py` and `src/searchMethods/*` decide which candidates to try next.

If you want to add a new optimization strategy, the most natural integration point is:

1. add a new search module in `src/searchMethods/`
2. add a wrapper in `src/explore.py`
3. add dispatch logic in `main.py`
4. extend `Config` if the new strategy needs its own config block

## Practical Starting Points

If you want to work on this codebase, these are the best files to start with:

- `main.py`: understand the CLI and dispatch rules
- `src/config.py`: understand what the JSON means
- `src/explore.py`: understand the default Optuna workflow
- `src/metrics.py`: understand what metrics are actually available
- `src/searchMethods/tabu_flags.py` or `wavefront_flags.py`: understand the custom search implementations
