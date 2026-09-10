# SCOuT

SCOuT is a Python tool for design-space exploration of compiler flags and runtime environment settings. It builds a target program, measures the result, and searches for better configurations with Optuna or custom search strategies.

## Quick start

Run an experiment with:

```bash
python3 -m pip install -r requirements.txt
python3 main.py --mode parameter_tuning --config configs/example_configs/dse_config.json --trials 100
```

SCOuT reads a JSON config, builds the target, runs the selected measurement backend, and logs the results.

For a CMake-project example, use
[`configs/example_configs/cmake_project.json`](configs/example_configs/cmake_project.json).

PolyMorph is an optional Tadashi-based SYCL transformation subsystem. It is
not imported or required by parameter-tuning runs. Install it separately only
when needed:

```bash
python3 -m pip install -r requirements-polymorph.txt
python3 main.py --mode polymorph --config configs/polyMorph/O1/matrixT-sycl/config.json --trials 50
```

The Tadashi/Polly LLVM toolchain must also be available on `PATH` for
transformation and code-generation operations.

## Project layout

- `main.py`: CLI entry point
- `src/`: core logic for config loading, building, measuring, and search
- `src/polyMorph/`: optional Tadashi-based transformation discovery and search
- `configs/dse_config.json`: example configuration
- `test-benchmarks/`: benchmark projects and datasets

## Supported features

- Single-source, CMake, and Make-based builds
- Measurement backends: `perf`, `likwid`, and output parsing via `parser`
- Search methods: Optuna, `wavefront`, `tabu`, `beam_tabu`, and `anneal`

## More details

- Architecture and configuration: [docs/architecture.md](/home/hari/git/SCOuT/docs/architecture.md)
- Parameter-tuning usage: [docs/parameter-tuning.md](/home/hari/git/SCOuT/docs/parameter-tuning.md)
- Search methods: [docs/search-algorithms.md](/home/hari/git/SCOuT/docs/search-algorithms.md)
- PolyMorph configuration: [docs/polymorph-config-knobs.md](/home/hari/git/SCOuT/docs/polymorph-config-knobs.md)
- PolyMorph adaptive tree search: [docs/polymorph-adaptive-tree-search.md](/home/hari/git/SCOuT/docs/polymorph-adaptive-tree-search.md)
