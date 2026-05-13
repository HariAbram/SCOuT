# polyMorph Post Processing

Utilities in this directory clean or inspect files produced by polyMorph search runs.

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
