# polyMorph Config Suite

These configs use the existing `microSYCL` benchmark under `test-benchmarks/microSYCL` as the reference project.

Files:

- `discover.json`: exercises `discover` and implicit source detection
- `list_available.json`: exercises `print_available_transformations`
- `codegen_snapshot.json`: exercises project rebuild, `generated_infix`, `save_jscops`, `build_target`, `build_dir`, and explicit `source`
- `manual_transforms.json`: exercises explicit `transforms`, `allow_illegal`, `measure`, and `runtime_args`
- `search_enumerate.json`: exercises Optuna candidate enumeration and most search knobs without running trials
- `search_run.json`: exercises an actual Optuna run with result export

Run examples:

```bash
python3 main.py --mode polymorph configs/polyMorph/discover.json
python3 main.py --mode polymorph configs/polyMorph/list_available.json
python3 main.py --mode polymorph configs/polyMorph/codegen_snapshot.json
python3 main.py --mode polymorph configs/polyMorph/manual_transforms.json
python3 main.py --mode polymorph configs/polyMorph/search_enumerate.json
python3 main.py --mode polymorph configs/polyMorph/search_run.json
```

Notes:

- `manual_transforms.json` contains example transform specs. They are meant as editable placeholders and may need adjustment after inspecting available SCoPs and node transforms on your machine.
- The benchmark is `make`-based, so this suite covers the `make` path rather than the `cmake` path.
