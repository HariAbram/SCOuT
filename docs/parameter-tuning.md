# Parameter tuning guide

This guide covers SCOuT's compiler-flag and runtime-environment tuning mode. It does not cover PolyMorph.

## Install

Use Python 3.10 or newer and install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the measurement tool used by your configuration (`perf` or LIKWID). The `parser` backend needs no external counter tool, but the benchmark must print the documented `[SYCL]` timing lines.

## Run

Run repository-provided configurations from the repository root so their relative benchmark paths resolve correctly.

```bash
python3 main.py \
  --mode parameter_tuning \
  --config configs/flag-env/optuna_random_perf.json \
  --trials 50
```

`--trials` controls Optuna. Wavefront, tabu, beam-tabu, and annealing use the limits in their own configuration blocks.

## Project builds

Keep the build target and runnable file separate:

```json
"project": {
  "dir": "test-benchmarks/parameterSearch/microSYCL",
  "build_system": "make",
  "target": "all",
  "executable": "bin/main-acpp-generic",
  "make_flags_var": "EXTRA_CFLAGS",
  "make_vars": {
    "BACKEND": "generic"
  }
}
```

- `target` is passed to Make or CMake.
- `executable` is the file SCOuT runs after the build. For Make it is relative to `project.dir`; for CMake it is relative to the generated build directory.
- Always set `executable` when the target is phony, such as `all`.

For a CMake project, use `build_system: "cmake"`:

```json
"project": {
  "dir": "path/to/project",
  "build_system": "cmake",
  "target": "my_program",
  "executable": "bin/my_program",
  "cmake_flag_vars": ["CMAKE_CXX_FLAGS"],
  "cmake_defs": [
    "BUILD_TESTING=OFF",
    "MY_BACKEND=CPU"
  ]
}
```

For every distinct compiler configuration, SCOuT performs the equivalent of:

```bash
cmake -S path/to/project -B <workdir> \
  -DCMAKE_CXX_COMPILER=<compiler> \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS:STRING=<sampled-flags> \
  -DBUILD_TESTING=OFF \
  -DMY_BACKEND=CPU
cmake --build <workdir> --parallel --target my_program
```

- `compiler` becomes `CMAKE_CXX_COMPILER`.
- Each `cmake_defs` entry becomes a `-D` CMake definition.
- Every variable in `cmake_flag_vars` receives the complete generated flag string.
- `executable` is relative to SCOuT's generated CMake build directory, not `project.dir`. Use the path CMake generates, such as `my_program`, `bin/my_program`, or `Release/my_program.exe`.
- If a project exposes its own flag cache variable, use that instead of `CMAKE_CXX_FLAGS`. The miniBUDE example uses `CXX_EXTRA_FLAGS` for this reason.

A complete repository-backed example is available at
[`configs/example_configs/cmake_project.json`](../configs/example_configs/cmake_project.json).
Run it from the repository root:

```bash
python3 main.py --mode parameter_tuning \
  --config configs/example_configs/cmake_project.json \
  --trials 10
```

The example expects the AdaptiveCpp CMake package under `/opt/AdaptiveCpp`, as
required by the included miniBUDE `CMakeLists.txt`. If your installation is
elsewhere, update that benchmark's `HIPSYCL_INSTALL_DIR` setting or use your own
CMake project and provide its package-location definition in `cmake_defs`.

For a single source, use `"source": "path/to/file.cpp"` instead of `project`.

## Compiler and runtime spaces

```json
"compiler_flags_base": "--acpp-platform=cpu --acpp-targets=generic",
"compiler_flags": ["-O2", "-O3"],
"compiler_flag_pool": ["-fvectorize", "-ffast-math"],
"compiler_params": {
  "-march": ["native", "znver4"],
  "--tile={}": [8, 16, 32]
},
"compiler_params_select": {
  "min": 1,
  "max": 2,
  "always": ["-march"]
},
"env": {
  "ACPP_VISIBILITY_MASK": ["omp", "generic"],
  "OMP_PLACES": {
    "when": {"ACPP_VISIBILITY_MASK": "omp"},
    "values": ["cores", "threads"]
  }
}
```

Conditional environment variables must appear after every variable referenced by `when`. SCOuT removes managed-but-inactive variables from the inherited process environment, preventing shell settings from leaking into a candidate.

## Build and AdaptiveCpp cache reuse

SCOuT caches these separately:

- Build cache: compiler, compiler flags, source/project, and build settings.
- Evaluation cache: build key plus runtime environment.

Changing only an environment variable therefore reuses the existing executable. If the compiler space contains one configuration, SCOuT builds exactly once during that process.

AdaptiveCpp runtime/JIT cache behavior is explicit:

```json
"runtime_cache_policy": "reuse"
```

- `reuse` is the default and avoids repeated runtime compilation.
- `cold` clears `~/.acpp/apps` after each measured configuration. Use it only when intentionally studying cold-start behavior; it affects the user's complete AdaptiveCpp application cache.

For large runtime spaces, set an environment cap in a custom search, for example `"env_mode": "sample", "env_cap": 16`. A fixed environment is written directly in that search block using `"env": {...}`.

## Search methods

- `optuna`: supports single- and multi-objective studies. Use `tpe`, `rs`, or `nsga3` for SCOuT's categorical space.
- `wavefront`: explores flag-atom combinations by size.
- `tabu`: moves among variants, compiler parameters, pool flags, and environments.
- `beam_tabu`: retains multiple candidates and uses tabu move memory.
- `anneal`: stochastic local search with seeded acceptance decisions.

Custom searches currently require exactly one objective. SCOuT rejects incompatible configurations rather than creating malformed result files.

Example search selection:

```json
"search": {
  "study": "tabu",
  "random_seed": 13
},
"tabu": {
  "max_iters": 40,
  "max_no_improve": 10,
  "tabu_tenure": 12,
  "neighborhood": 16,
  "env_mode": "sample",
  "env_cap": 8,
  "results_csv": "/tmp/scout/tabu.csv"
}
```

Unknown study and sampler names are configuration errors. CMA-ES is not offered because SCOuT's flag and environment dimensions are categorical; use TPE, random sampling, or NSGA-III.

## Measurement backends

For `perf`, commonly use:

```json
"perf": {
  "events": ["cycles", "instructions"],
  "warmup_runs": 1
},
"objectives": [{"metric": "CPI", "goal": "min"}]
```

For the parser backend, a program line looks like:

```text
[SYCL][avg] kernel 2: 1.25e-06 s over 1000 iters
```

The aggregate objective is named `sycl_<label>_<aggregate>_s`, such as `sycl_avg_sum_s`.

LIKWID metric names in `objectives` must also be present in `likwid.metrics`.

## Outputs and failures

- `csv_log`: completed evaluation rows.
- `pareto_log`: current Pareto front for multi-objective Optuna studies.
- `failed_builds`: failed and pruned Optuna trials with their reason.
- `sqlite_log`: persistent Optuna study database, written during optimization.

Parent directories are created automatically. If a CSV already exists, SCOuT creates a timestamped filename instead of overwriting it.

Temporary build artifacts and logs are printed at startup, normally under `/tmp/SCOuT_*`.
