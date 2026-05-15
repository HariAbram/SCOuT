# polyMorph Post Processing

Utilities in this directory clean or inspect files produced by polyMorph search runs.

## Summarize Search Results

Print a compact terminal table:

```bash
python3 post_processing/summarize_polymorph_results.py
```

Include transform/regression/kernel detail columns:

```bash
python3 post_processing/summarize_polymorph_results.py --details
```

Show only benchmarks with speedup at least 1.05, sorted by speedup:

```bash
python3 post_processing/summarize_polymorph_results.py --min-speedup 1.05 --sort speedup
```

## Generate Figures

Create PNG figures under `post_processing/figures`:

```bash
python3 post_processing/plot_polymorph_results.py
```

The plotting script generates:

- `best_speedup_by_benchmark.png`
- `speedup_by_optimization_level.png`
- `trial_outcomes.png`
- `transform_type_counts.png`
- per-benchmark trial curves under `trial_curves/`

It requires `matplotlib`.

## Plot Search Trees

Draw one MCTS prefix tree per benchmark. Nodes are transform prefixes; color shows
observed speedup and labels include the prefix rank.

```bash
python3 post_processing/plot_polymorph_tree.py --root configs/polyMorph/O1/gmm-sycl
```

Useful options:

```bash
python3 post_processing/plot_polymorph_tree.py --root configs/polyMorph --benchmark matrixT --max-prefixes 60
python3 post_processing/plot_polymorph_tree.py --root configs/polyMorph/O1 --metric mean_speedup --format svg
python3 post_processing/plot_polymorph_tree.py --root configs/polyMorph/O1/ace-sycl --label-mode table --label-top 12
python3 post_processing/plot_polymorph_tree.py --root configs/polyMorph/O1/ace-sycl --label-mode compact
```

Figures are written under `post_processing/figures/search_trees` by default.

## Clean Search Outputs

Dry run:

```bash
python3 post_processing/clean_polymorph_search.py
```

Delete matched generated files:

```bash
python3 post_processing/clean_polymorph_search.py --apply
```

Also remove benchmark `build/` directories:

```bash
python3 post_processing/clean_polymorph_search.py --apply --build-dirs
```

The cleaner targets generated result files, trial CSVs, cache/history JSONL files,
temporary transformed sources, object/dependency files, baseline/search binaries,
and parser logs under `configs/polyMorph` and `test-benchmarks/polyMorph`.
