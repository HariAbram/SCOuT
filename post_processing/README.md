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
