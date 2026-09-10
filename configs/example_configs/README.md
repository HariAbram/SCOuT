# Example Configurations

## Parameter tuning

- `dse_config.json`: detailed Make-project example containing all supported search blocks
- `cmake_project.json`: complete CMake-project example using the included miniBUDE SYCL benchmark

Run the CMake example from the repository root:

```bash
python3 main.py --mode parameter_tuning \
  --config configs/example_configs/cmake_project.json \
  --trials 10
```

The CMake example uses `target: "bude"` for the build target and
`executable: "bude"` for the runnable path inside SCOuT's generated build
directory. It passes sampled flags through miniBUDE's `CXX_EXTRA_FLAGS` CMake
cache variable.

## PolyMorph

Files:

- `discover.json`: exercises `discover` and implicit source detection
- `list_available.json`: exercises `print_available_transformations`
- `codegen_snapshot.json`: exercises project rebuild, `generated_infix`, `save_jscops`, `build_target`, `build_dir`, and explicit `source`
- `manual_transforms.json`: exercises explicit `transforms`, `allow_illegal`, `measure`, and `runtime_args`
- `search_enumerate.json`: exercises MCTS candidate enumeration and most search knobs without running trials
- `search_run.json`: exercises an actual MCTS run with result export
- `search_enhanced.json`: enables analytical scoring, static pruning, constraint-aware search, case retrieval, and multi-objective Pareto export

Run examples:

```bash
python3 -m pip install -r requirements-polymorph.txt
python3 main.py --mode polymorph --config configs/example_configs/discover.json
python3 main.py --mode polymorph --config configs/example_configs/list_available.json
python3 main.py --mode polymorph --config configs/example_configs/codegen_snapshot.json
python3 main.py --mode polymorph --config configs/example_configs/manual_transforms.json
python3 main.py --mode polymorph --config configs/example_configs/search_enumerate.json
python3 main.py --mode polymorph --config configs/example_configs/search_run.json
python3 main.py --mode polymorph --config configs/example_configs/search_enhanced.json
```

Notes:

- `manual_transforms.json` contains example transform specs. They are meant as editable placeholders and may need adjustment after inspecting available SCoPs and node transforms on your machine.
- The benchmark is `make`-based, so this suite covers the `make` path rather than the `cmake` path.
- Search uses the current polyMorph pipeline; legacy search mode has been removed.
