# SCOuT

SCOuT is a Python tool for design-space exploration of compiler flags and runtime environment settings. It builds a target program, measures the result, and searches for better configurations with Optuna or custom search strategies.

## Quick start

Run an experiment with:

```bash
python3 main.py --mode parameter_tuning configs/dse_config.json --trials 100
```

SCOuT reads a JSON config, builds the target, runs the selected measurement backend, and logs the results.

## Project layout

- `main.py`: CLI entry point
- `src/`: core logic for config loading, building, measuring, and search
- `configs/dse_config.json`: example configuration
- `test-benchmarks/`: benchmark projects and datasets

## Supported features

- Single-source, CMake, and Make-based builds
- Measurement backends: `perf`, `likwid`, and output parsing via `parser`
- Search methods: Optuna, `wavefront`, `tabu`, `beam_tabu`, and `anneal`

## More details

- Architecture and configuration: [docs/architecture.md](/home/hari/git/SCOuT/docs/architecture.md)
- Search methods: [docs/search-algorithms.md](/home/hari/git/SCOuT/docs/search-algorithms.md)
