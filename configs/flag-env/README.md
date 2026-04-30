# Flag And Env Search Config Suite

These configs exercise the original SCOuT compiler-flag and environment search flow using the same `microSYCL` benchmark style as the existing [configs/dse_config.json](/home/hari/git/SCOuT/configs/dse_config.json).

Files:

- `optuna_tpe_likwid.json`: main multi-objective Optuna example with conditional env settings
- `optuna_nsga3_likwid.json`: same style with the `nsga3` sampler and `k`-based param selection
- `optuna_random_perf.json`: single-objective `perf` backend with random search
- `optuna_cmaes_parser.json`: parser backend example with CMA-ES and parser-specific settings
- `wavefront_product_env.json`: wavefront search with product env expansion
- `tabu_sample_env.json`: tabu search with sampled env combinations
- `anneal_fixed_env.json`: annealing with fixed env behavior and anneal-specific knobs
- `beam_tabu_experimental.json`: documents intended beam-tabu knobs, but this is not fully wired in code today

Run examples:

```bash
python3 main.py --mode parameter_tuning configs/flag-env/optuna_tpe_likwid.json --trials 20
python3 main.py --mode parameter_tuning configs/flag-env/wavefront_product_env.json
python3 main.py --mode parameter_tuning configs/flag-env/tabu_sample_env.json
python3 main.py --mode parameter_tuning configs/flag-env/anneal_fixed_env.json
```

Notes:

- These configs use project builds, because `microSYCL` is not a single-file benchmark.
- `beam_tabu_experimental.json` loads as JSON, but its `beam_tabu` block is currently ignored by [src/config.py](/home/hari/git/SCOuT/src/config.py). That matches the current codebase caveat.
