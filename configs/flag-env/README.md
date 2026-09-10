# Flag And Env Search Config Suite

These configs exercise the original SCOuT compiler-flag and environment search flow using the same `microSYCL` benchmark style as the existing [configs/dse_config.json](/home/hari/git/SCOuT/configs/dse_config.json).

Files:

- `optuna_tpe_likwid.json`: main multi-objective Optuna example with conditional env settings
- `optuna_nsga3_likwid.json`: same style with the `nsga3` sampler and `k`-based param selection
- `optuna_random_perf.json`: single-objective `perf` backend with random search
- `optuna_random_parser.json`: parser backend example using random sampling because the space is categorical
- `wavefront_product_env.json`: wavefront search with product env expansion
- `tabu_sample_env.json`: tabu search with sampled env combinations
- `anneal_fixed_env.json`: annealing with fixed env behavior and anneal-specific knobs
- `beam_tabu_experimental.json`: beam-tabu example (the filename is retained for compatibility)

Run examples:

```bash
python3 main.py --mode parameter_tuning --config configs/flag-env/optuna_tpe_likwid.json --trials 20
python3 main.py --mode parameter_tuning --config configs/flag-env/wavefront_product_env.json
python3 main.py --mode parameter_tuning --config configs/flag-env/tabu_sample_env.json
python3 main.py --mode parameter_tuning --config configs/flag-env/anneal_fixed_env.json
```

Notes:

- These configs use project builds, because `microSYCL` is not a single-file benchmark.
- Project examples distinguish the phony Make target (`all`) from the executable (`bin/main-acpp-generic`).
- All examples reuse host and AdaptiveCpp runtime compilation where the build inputs are unchanged.
