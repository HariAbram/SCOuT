# polyMorph Configuration Knobs

This document lists the configuration fields used by `--mode polymorph`. The options are split into the outer `polyMorph` block and the nested `polyMorph.search` block. Free-form search policy parameters are placed under `polyMorph.search.constraints`.

## Minimal Shape

```json
{
  "backend": "parser",
  "parser": {
    "label": "avg",
    "aggregate": "sum",
    "run_cwd": "workdir"
  },
  "objectives": [
    {
      "metric": "sycl_avg_sum_s",
      "goal": "min"
    }
  ],
  "polyMorph": {
    "project_root": "test-benchmarks/polyMorph/O1/gmm-sycl",
    "source": "test-benchmarks/polyMorph/O1/gmm-sycl/main.cpp",
    "compiler": "acpp",
    "flags": ["-O1"],
    "mcts_search": true,
    "build_system": "make",
    "exec_name": "build/gmm-search",
    "runtime_args": ["500", "data", "result"],
    "search": {
      "n_trials": 20,
      "repeat": 2,
      "target_backend": "omp",
      "max_transforms_per_trial": 5,
      "constraints": {}
    }
  }
}
```

## Top-Level `polyMorph` Fields

| Field | Default | Meaning |
|---|---:|---|
| `project_root` | required | Benchmark directory used for building and running. |
| `source` | `null` | Source file passed to Tadashi and to the build system. Use the file that contains the relevant SYCL kernels when possible. |
| `compiler` | `"acpp"` | Compiler executable used by the benchmark Makefile/CMake invocation. |
| `flags` | `[]` | Extra compiler flags, for example `["-O2"]`. |
| `mcts_search` | `false` | Enables the Monte Carlo tree search path. Current search work should use this mode. |
| `build_system` | `"make"` | Either `"make"` or `"cmake"`. |
| `build_target` | `null` | Optional build target. |
| `build_dir` | `null` | Optional build directory for CMake-style builds. |
| `exec_name` | `null` | Output executable path, usually under `build/`. Required for search mode. |
| `runtime_args` | `[]` | Arguments passed to the benchmark executable. |
| `allow_illegal` | `false` | If `true`, do not reject illegal Tadashi schedules. Normally leave this `false`. |
| `print_available_transformations` | `false` | Print Tadashi transformations available at schedule nodes. Useful for debugging candidate availability. |
| `list_only` | `false` | List transformations without applying or measuring. |
| `discover` | `false` | Discovery/debug mode for SCoPs and transformations. |
| `measure` | `false` | Measure a configured transformed build path outside search. |
| `save_jscops` | `null` | Directory where extracted JScop files are copied. |
| `transforms` | `[]` | Manual transformation sequence for non-search use. |
| `generated_infix` | `"pass1"` | Infix used for generated source names in manual transformation mode. Search uses `search.generated_infix`. |

## Tadashi/Polly Codegen Pipeline

polyMorph uses Tadashi's Polly translator in two phases: SCoP/JScop discovery and transformed object generation. The `polyMorph.flags` list is passed to Tadashi as user compiler options, but Tadashi also adds its own optimization choices internally.

For SCoP/JScop discovery, Tadashi first compiles the configured source to LLVM bitcode:

```text
<compiler> -O0 -Xclang -disable-O0-optnone <polyMorph.flags> -c -emit-llvm <source> -o <tmp>/<source>.O0.bc
```

For `compiler: "acpp"` and `flags: ["-O1"]`, the command shape is therefore roughly:

```text
acpp -O0 -Xclang -disable-O0-optnone -O1 -c -emit-llvm main.cpp -o main.O0.bc
```

Tadashi names this file `O0.bc` and logs the step as compiling with O0, but clang-like drivers usually let the later optimization flag win. Since `polyMorph.flags` appear after Tadashi's `-O0`, `-O1`, `-O2`, or `-O3` may affect the initial bitcode and therefore which SCoPs/JScops are discovered.

SCOuT intentionally preserves LLVM/Polly names during SCoP extraction. Keeping descriptive JScop names makes it possible to filter SCoPs that are related to SYCL kernel wrappers instead of host/helper affine loops.

Tadashi then canonicalizes the bitcode and exports JScops:

```text
opt [-load=LLVMPolly.so] -polly-canonicalize <source>.O0.bc -o <source>.pre_polly.bc
opt [-load=LLVMPolly.so] -polly-import-jscop-dir=<tmp> -aa-pipeline=basic-aa -polly-codegen <source>.pre_polly.bc -polly-export-jscop -o=/dev/null
```

The configured flags are not passed directly to these `opt` commands. They only matter through the bitcode created in the first compile step.

When code is generated after applying transformations, Tadashi rewrites the exported JScop schedules, imports them back through Polly, then hardcodes LLVM optimization for the generated object:

```text
opt [-load=LLVMPolly.so] -polly-import-jscop-dir=<tmp> -aa-pipeline=basic-aa -polly-codegen <source>.pre_polly.bc -polly-import-jscop -disable-polly-legality -polly-parallel-force -o=<source>.post_polly.bc
opt -O3 <source>.post_polly.bc -o=<source>.bc
llc -O3 -relocation-model=pic --filetype=obj <source>.bc -o=<generated>.o
```

Thus `polyMorph.flags` mainly influence initial bitcode/SCoP discovery and the benchmark Makefile invocation. The final Tadashi-generated object currently goes through `opt -O3` and `llc -O3` regardless of `flags`.

In MCTS mode, SCOuT generates the baseline through the same no-op Tadashi path before measuring it. This makes baseline and transformed trials comparable:

```text
baseline:  original schedule -> JScop import -> opt -O3 -> llc -O3 -> object
trial:     transformed schedule -> JScop import -> opt -O3 -> llc -O3 -> object
```

## `polyMorph.search` Fields

| Field | Default | Meaning |
|---|---:|---|
| `n_trials` | `20` | Number of MCTS trials. Can be overridden with `main.py --trials`. |
| `repeat` | `3` | Number of full measurement repetitions per evaluated candidate. |
| `seed` | `0` | Random seed used by MCTS tie-breaking. |
| `timeout` | `null` | Search time limit in seconds. |
| `generated_infix` | `"mcts"` | Infix used for generated trial sources. |
| `baseline_exec_name` | `null` | Baseline executable path. If omitted, derives from `exec_name`. |
| `enumerate_only` | `false` | Enumerate candidate transformations and exit before search. |
| `max_transforms_per_trial` | `1` | Maximum transformation sequence length in a trial. |
| `tile_sizes` | `[8, 16, 32, 64, 128]` | Candidate tile sizes used to infer `TILE_1D`, `TILE_2D`, and `TILE_3D` arguments. |
| `allow_transforms` | `null` | Optional allow-list of Tadashi transform names. If present, only these transforms are considered. |
| `block_transforms` | `["SET_PARALLEL"]` | Transform names to exclude. |
| `legality_aware_args` | `true` | Infer arguments from Tadashi's current-node argument descriptors and validate them with Tadashi before search. |
| `static_pruning` | `false` | Enable rule-based candidate and sequence pruning. |
| `analytical_model` | `false` | Enable static heuristic scoring and risk estimation. |
| `constraint_aware` | `false` | Apply score/risk constraints and performance-bias pruning. |
| `top_k` | `null` | Keep only a diverse top-k candidate set after scoring. |
| `case_retrieval` | `false` | Seed MCTS with prior history records. |
| `structural_retrieval` | `true` | When retrieving history, prefer structurally similar kernels. |
| `retrieval_top_k` | `3` | Number of retrieved history records used for seeding. |
| `learned_model` | `true` | Apply learned statistics from prior history/cache records to candidate scoring. |
| `learned_model_min_observations` | `1` | Minimum observations before learned statistics affect a candidate. |
| `cache_evaluations` | `true` | Reuse prior evaluations with matching source, sequence, objective, repeat, backend, and search settings. |
| `cache_jsonl` | `null` | Path to evaluation cache JSONL. If omitted, a path is derived from result/history paths. |
| `history_jsonl` | `null` | Path where trial history records are appended. |
| `result_json` | `null` | Path for final result summary. |
| `trial_csv` | `null` | Path for per-trial CSV output. |
| `pareto_csv` | `null` | Path for Pareto output in multi-objective mode. |
| `multi_fidelity` | `false` | Run a low-fidelity measurement before full repeat and prune clearly poor candidates. When disabled, `trial_warmup_runs` still warms transformed binaries before measurement. |
| `early_stop_worse_than` | `1.15` | Low-fidelity candidate is pruned if it is worse than the baseline by this factor. |
| `target_backend` | `null` | If set, runs baseline and trials with `ACPP_VISIBILITY_MASK=<target_backend>`. |
| `backend_sensitivity_masks` | `[]` | Backend masks to compare for sensitivity analysis, for example `["omp", "ocl"]`. |
| `backend_sensitivity_repeat` | `1` | Repeat count for backend-sensitivity measurements. |
| `backend_sensitivity_per_trial` | `false` | If `true`, run backend-sensitivity analysis for every successful transformed trial. Otherwise it is baseline-only. |
| `correctness_outputs` | `[]` | Output files compared against the baseline after each candidate. |
| `correctness_tolerance` | `1.0e-6` | Numeric tolerance for correctness output comparison. |
| `correctness_required` | `true` | If `true`, missing correctness output files fail the trial. |
| `replay_top_k` | `0` | Replay the top-k completed trials after search. |
| `final_validation_enabled` | `true` | After search, rebuild the best/top candidates and remeasure them against repeated baseline samples. |
| `final_validation_repeats` | `20` | Number of interleaved baseline/candidate measurement rounds per validated candidate. |
| `final_validation_warmup_runs` | `1` | Warmup runs for baseline and candidate before final validation samples are collected. |
| `trial_warmup_runs` | `final_validation_warmup_runs` | Warmup executions for each transformed trial before the full measured repeat. This is independent of `multi_fidelity`. |
| `final_validation_top_k` | `3` | Number of unique top search candidates to validate. |
| `final_validation_min_speedup` | `1.0` | Minimum validated speedup required before a result can be marked significant. |
| `final_validation_noise_factor` | `2.0` | Require speedup to exceed `1 + noise_factor * max(baseline_rsd, candidate_rsd)`. |
| `baseline_resample_interval` | `0` | If greater than zero, remeasure the baseline after this many completed trials and store drift samples. |

## `search.constraints` Fields

The `constraints` object is intentionally open-ended. Missing keys use implementation defaults.

### Candidate Legality and Static Pruning

| Key | Default | Meaning |
|---|---:|---|
| `min_tile_size` | `1` | Reject tile sizes smaller than this. |
| `max_tile_size` | `256` | Reject tile sizes larger than this. |
| `max_tile_volume` | `65536` | Reject tile products larger than this. |
| `max_abs_shift` | `128` | Reject shift magnitudes larger than this. |
| `max_yaml_bytes` | `0` | If nonzero, reject candidates whose node YAML exceeds this size. |
| `sycl_kernel_scop_filter` | `true` | Only enumerate transformations for JScops whose exported filename/function identifier looks SYCL-kernel related. Matching markers include `sycl`, `hipsycl`, `acpp`, `__sscp`, `sscp`, `nd_item`, `handler`, `parallel_for`, `kernel`, `queue`, and `runtest`. This avoids tuning affine host/helper loops discovered in the same source file. |
| `sycl_kernel_scop_markers` | built-in marker list | Optional replacement list of marker strings used by `sycl_kernel_scop_filter`. Use this if a compiler/runtime names outlined kernels differently. |
| `sycl_kernel_scop_filter_fallback_all` | `true` | If no SCoP matches the marker list, keep all SCoPs and print their identifiers instead of producing an empty search space. Set to `false` for strict filtering. |
| `prune_tiling_small_scops` | `false` | Reject tiling on SCoPs classified as small. |
| `max_same_transform_per_sequence` | `2` | Reject repeated identical transform type on the same `(scop, node)`. |
| `max_transforms_per_scop` | `0` | If nonzero, cap transforms targeting the same SCoP in one sequence. |
| `tree_max_transforms_per_scop` | `1` | Fallback per-SCoP cap used by MCTS sequence construction. |

### Analytical Scoring

| Key | Default | Meaning |
|---|---:|---|
| `preferred_tile_size` | `32` | Legacy tile-size center used only by the fallback heuristic. |
| `preferred_tile_size_fallback_weight` | `0.20` when Pluto is enabled, otherwise `1.0` | Multiplier for the legacy preferred-size fallback. Set to `0.0` to avoid biasing tile search toward 32 when Pluto evidence is absent. |
| `pluto_cost_model` | `true` | Use the simplified Pluto-style affine reuse/dependence-distance prior when scoring candidates. |
| `pluto_cost_weight` | `1.0` | Multiplier for the simplified Pluto-style prior. |
| `pluto_stream_pair_weight` | `0.15` | Weight assigned to stream-shaped access pairs when no exact repeated-array reuse is found. Lower values prevent stream-only evidence from over-rewarding tiling. |
| `large_tile_volume` | `4096` | Tile volume above which register/cache pressure risk increases. |
| `max_predicted_risk` | unset | If set, prune candidates whose predicted risk is higher. |
| `min_predicted_score` | unset | If set, prune candidates whose predicted score is lower. |

### MCTS Search Policy

| Key | Default | Meaning |
|---|---:|---|
| `mcts_include_stop_action` | `true` | Include a no-transformation `STOP` action as a valid first trial. |
| `mcts_max_selection_retries` | `100` | Minimum retry budget for illegal/no-op selections. Actual budget is `max(n_trials * 10, this value)`. |
| `single_transform_screening` | `true` | Start with one-transform trials to estimate individual candidate quality. |
| `single_transform_screening_limit` | `0` | Number of screening trials. If `0`, uses `min(16, candidate_count)`. |
| `screening_per_family` | `2` | During screening, try candidates across transform families instead of only globally highest-ranked candidates. |
| `tree_family_novelty_bonus` | `0.20` | Prior bonus for a transform family that has not yet been tried. |
| `tree_tile_size_novelty_bonus` | `0.25` | Prior bonus for tile sizes that have been tried fewer times, encouraging exploration across the configured `tile_sizes`. |
| `tree_unvisited_bonus` | `1.0` | UCT-style bonus for unvisited children. |
| `tree_exploration` | `0.8` | Exploration coefficient for visited children. |
| `tree_failure_penalty` | `0.25` | Penalty multiplier for candidates/prefixes with failed, pruned, or early-stopped history. |
| `tree_min_visits_before_expand` | `1` | Prefix visits required before adding more transforms after it. |
| `tree_expand_min_best_speedup` | `0.90` | Minimum best speedup required to expand a prefix. |
| `tree_prune_min_visits` | `2` | Prefix visits required before pruning based on poor best reward. |
| `tree_prune_best_speedup_below` | `0.95` | Prefix is bad if its best speedup remains below this after enough visits. |
| `tree_prune_after_early_stops` | `3` | Prune a prefix after this many early stops if it has no speedup above `1.0`. |
| `candidate_blacklist_failures` | `2` | Blacklist a candidate after this many failed/pruned/early-stopped observations. |
| `candidate_blacklist_min_best_speedup` | `0.90` | Candidate is blacklisted only if its best speedup is below this threshold. |
| `disable_family_after_failures` | `4` | Disable a transform family during screening after this many failed candidates, if the failure fraction threshold is also met. |
| `disable_family_failure_fraction` | `0.50` | Fraction of visible screening candidates in a family that must fail before the family is disabled. |
| `cache_hit_reward_scale` | `0.95` | Reward multiplier used when a cached terminal sequence is selected. The tree is updated, but the trial is retried instead of counted as a completed trial. |
| `baseline_warmup_runs` | `final_validation_warmup_runs`, at least `1` | Warmup executions before measuring the baseline objective. Prevents AdaptiveCpp JIT latency from entering the baseline metric. |

### Multi-Fidelity Early Stopping

| Key | Default | Meaning |
|---|---:|---|
| `early_stop_warmup_runs` | `1` | Warmup runs before low-fidelity measurement. Useful for AdaptiveCpp JIT stabilization. |
| `early_stop_measure_runs` | `1` | Number of measured low-fidelity runs. |

### Backend Sensitivity

| Key | Default | Meaning |
|---|---:|---|
| `backend_sensitivity_rel_threshold` | `0.20` | Mark backend-sensitive if objective values across masks vary by more than this relative threshold. |
| `backend_sensitivity_warmup_runs` | `final_validation_warmup_runs`, at least `1` | Warmup executions before measuring each backend mask. |

### Performance-Bias Feedback

These keys control the lightweight per-candidate bias derived from previous measured outcomes. They are not the old compiler/runtime debug feedback path.

| Key | Default | Meaning |
|---|---:|---|
| `performance_feedback_bias_weight` | `0.35` | Weight applied when prior measured outcomes adjust score and risk. |
| `performance_feedback_min_observations_for_prune` | `1` | Minimum observations required before performance-bias pruning can happen. |
| `max_performance_feedback_penalty` | unset | If set, prune candidates whose accumulated performance penalty exceeds this value. |

### Logging

| Key | Default | Meaning |
|---|---:|---|
| `print_candidate_family_counts` | `true` | Print transform-family counts before and after pruning/top-k selection. |

## Parser Measurement Knobs

Most polyMorph benchmark configs use the parser backend to read `[SYCL][avg]` or `[SYCL][sum]` lines from program output.

| Field | Default | Meaning |
|---|---:|---|
| `parser.label` | `"avg"` | Parse average or sum lines. |
| `parser.kernels` | `null` | Optional list of kernel IDs to include. If absent, all parsed kernels are used. |
| `parser.aggregate` | `"sum"` | Aggregate kernel metrics with `"sum"`, `"mean"`, `"max"`, or `"min"`. |
| `parser.warmup_runs` | `0` | Extra parser warmup runs. |
| `parser.core_list` | `null` | Optional CPU affinity list. |
| `parser.prefix` | `null` | Optional command prefix, for example `["numactl", "-N", "0", "-m", "0"]`. |
| `parser.run_cwd` | `"binary_dir"` | Run directory: `"binary_dir"`, `"project_dir"`, or `"workdir"`. |

## Practical Profiles

### Faster Feasibility Run

```json
"search": {
  "n_trials": 5,
  "repeat": 1,
  "max_transforms_per_trial": 2,
  "multi_fidelity": true,
  "early_stop_worse_than": 1.10,
  "backend_sensitivity_per_trial": false,
  "replay_top_k": 0,
  "final_validation_enabled": false,
  "constraints": {
    "single_transform_screening_limit": 8,
    "mcts_max_selection_retries": 50
  }
}
```

### More Thorough Search

```json
"search": {
  "n_trials": 50,
  "repeat": 2,
  "max_transforms_per_trial": 5,
  "static_pruning": true,
  "analytical_model": true,
  "constraint_aware": true,
  "learned_model": true,
  "cache_evaluations": true,
  "case_retrieval": true,
  "final_validation_enabled": true,
  "final_validation_repeats": 20,
  "final_validation_top_k": 3,
  "constraints": {
    "single_transform_screening_limit": 16,
    "screening_per_family": 2,
    "tree_exploration": 0.8,
    "tree_family_novelty_bonus": 0.20,
    "tree_expand_min_best_speedup": 0.90
  }
}
```

### Backend-Pinned Search

```json
"search": {
  "target_backend": "omp",
  "backend_sensitivity_masks": ["omp", "ocl"],
  "backend_sensitivity_repeat": 2,
  "backend_sensitivity_per_trial": false
}
```

This measures the objective on `omp`, checks backend sensitivity once on the baseline, and avoids running every trial under both masks.
