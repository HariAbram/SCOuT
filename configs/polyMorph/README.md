# polyMorph Config Suite

These configs use the existing `microSYCL` benchmark under `test-benchmarks/microSYCL` as the reference project.

Files:

- `discover.json`: exercises `discover` and implicit source detection
- `list_available.json`: exercises `print_available_transformations`
- `codegen_snapshot.json`: exercises project rebuild, `generated_infix`, `save_jscops`, `build_target`, `build_dir`, and explicit `source`
- `manual_transforms.json`: exercises explicit `transforms`, `allow_illegal`, `measure`, and `runtime_args`
- `search_enumerate.json`: exercises Optuna candidate enumeration and most search knobs without running trials
- `search_run.json`: exercises an actual Optuna run with result export
- `search_enhanced.json`: enables the optional analytical scoring, static pruning, constraint-aware search, compiler feedback, case retrieval, and multi-objective Pareto export
- `search_legacy.json`: sets `polyMorph.search.legacy=true` to run the old candidate sampling and runtime-style measurement path

Run examples:

```bash
python3 main.py --mode polymorph configs/polyMorph/discover.json
python3 main.py --mode polymorph configs/polyMorph/list_available.json
python3 main.py --mode polymorph configs/polyMorph/codegen_snapshot.json
python3 main.py --mode polymorph configs/polyMorph/manual_transforms.json
python3 main.py --mode polymorph configs/polyMorph/search_enumerate.json
python3 main.py --mode polymorph configs/polyMorph/search_run.json
python3 main.py --mode polymorph configs/polyMorph/search_enhanced.json
python3 main.py --mode polymorph configs/polyMorph/search_legacy.json
```

Notes:

- `manual_transforms.json` contains example transform specs. They are meant as editable placeholders and may need adjustment after inspecting available SCoPs and node transforms on your machine.
- The benchmark is `make`-based, so this suite covers the `make` path rather than the `cmake` path.
- All enhanced search features are opt-in. Existing configs keep the previous behavior unless new `polyMorph.search` switches are enabled.
- Use `polyMorph.search.legacy=true` when you want the previous polyMorph search behavior even if other enhanced options are present in a copied config.
