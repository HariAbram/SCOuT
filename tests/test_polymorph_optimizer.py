from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

from src.config import PolyMorphSearchSpec
from src.polyMorph.features import (
    enrich_candidate,
    sequence_feature_summary,
    sequence_signature,
    structural_signature,
)
from src.polyMorph.history import append_history, load_history, retrieve_sequences
from src.polyMorph.pruning import analytical_score, static_prune_candidate, static_prune_sequence
from src.polyMorph.runner import (
    AdaptiveTreeState,
    analyze_kernel_timing_deltas,
    append_evaluation_cache,
    apply_learned_model_to_candidates,
    apply_hot_kernel_filter_to_candidates,
    build_hot_kernel_filter,
    build_learned_candidate_model,
    candidate_args_invalid_for_node,
    candidate_args_for_node,
    capture_correctness_outputs,
    cleanup_trial_artifacts,
    load_evaluation_cache,
    print_candidates,
    SearchSpaceExhausted,
    select_diverse_top_k,
    suppress_external_viewers,
    tadashi_compiler_options,
    verify_correctness_outputs,
    _short_jscop_backup_path,
    _sort_tree_candidates,
)


class DummyNode:
    node_type = "band"
    yaml_str = "schedule: loop\nread: A[i]\nwrite: B[i]\n"
    available_transformations = []


class PlutoNode:
    node_type = "band"
    yaml_str = "schedule: [i0, i1]\nread: A[i0, i1]\nread: A[i0 + 4, i1]\nwrite: B[i0, i1]\n"
    available_transformations = []


class PolyMorphOptimizerTests(unittest.TestCase):
    def test_config_defaults_keep_new_features_disabled(self) -> None:
        spec = PolyMorphSearchSpec.from_dict({})
        self.assertFalse(spec.static_pruning)
        self.assertFalse(spec.analytical_model)
        self.assertFalse(spec.constraint_aware)
        self.assertFalse(spec.case_retrieval)
        self.assertIn("SET_PARALLEL", spec.block_transforms)

    def test_tadashi_options_disable_llvm_names_only_for_scop_population(self) -> None:
        self.assertEqual(tadashi_compiler_options(["-O2"], False), ["-O2"])
        self.assertEqual(
            tadashi_compiler_options(["-O2"], True),
            ["-O2", "-mllvm", "-polly-use-llvm-names=false"],
        )
        self.assertEqual(
            tadashi_compiler_options(["-O2", "-mllvm", "-polly-use-llvm-names=false"], True),
            ["-O2", "-mllvm", "-polly-use-llvm-names=false"],
        )

    def test_config_parses_new_options(self) -> None:
        spec = PolyMorphSearchSpec.from_dict(
            {
                "static_pruning": True,
                "analytical_model": True,
                "constraint_aware": True,
                "case_retrieval": True,
                "structural_retrieval": False,
                "cache_jsonl": "/tmp/polymorph-cache.jsonl",
                "cache_evaluations": False,
                "multi_fidelity": False,
                "early_stop_worse_than": 1.4,
                "top_k": 7,
                "retrieval_top_k": 2,
                "history_jsonl": "/tmp/history.jsonl",
                "pareto_csv": "/tmp/pareto.csv",
                "constraints": {"max_tile_size": 64},
                "block_transforms": [],
                "backend_sensitivity_masks": ["omp", "cuda"],
                "backend_sensitivity_repeat": 2,
                "backend_sensitivity_per_trial": True,
                "legality_aware_args": False,
                "correctness_outputs": ["result.txt"],
                "correctness_tolerance": 1.0e-4,
                "correctness_required": False,
                "learned_model": False,
                "learned_model_min_observations": 2,
                "target_backend": "cuda",
                "ablation_enabled": False,
                "replay_top_k": 3,
            }
        )
        self.assertTrue(spec.static_pruning)
        self.assertTrue(spec.analytical_model)
        self.assertTrue(spec.constraint_aware)
        self.assertTrue(spec.case_retrieval)
        self.assertFalse(spec.structural_retrieval)
        self.assertEqual(spec.cache_jsonl, "/tmp/polymorph-cache.jsonl")
        self.assertFalse(spec.cache_evaluations)
        self.assertFalse(spec.multi_fidelity)
        self.assertEqual(spec.early_stop_worse_than, 1.4)
        self.assertEqual(spec.top_k, 7)
        self.assertEqual(spec.retrieval_top_k, 2)
        self.assertEqual(spec.constraints["max_tile_size"], 64)
        self.assertEqual(spec.block_transforms, [])
        self.assertEqual(spec.backend_sensitivity_masks, ["omp", "cuda"])
        self.assertEqual(spec.backend_sensitivity_repeat, 2)
        self.assertTrue(spec.backend_sensitivity_per_trial)
        self.assertFalse(spec.legality_aware_args)
        self.assertEqual(spec.correctness_outputs, ["result.txt"])
        self.assertEqual(spec.correctness_tolerance, 1.0e-4)
        self.assertFalse(spec.correctness_required)
        self.assertFalse(spec.learned_model)
        self.assertEqual(spec.learned_model_min_observations, 2)
        self.assertEqual(spec.target_backend, "cuda")
        self.assertFalse(spec.ablation_enabled)
        self.assertEqual(spec.replay_top_k, 3)

    def test_enrich_score_and_prune_candidate(self) -> None:
        candidate = {"scop": 0, "node": 1, "tr": "TILE_2D", "args": [16, 32]}
        enriched = enrich_candidate(candidate, DummyNode())
        prediction = analytical_score(enriched, {"preferred_tile_size": 32})
        reasons = static_prune_candidate(enriched, {"max_tile_size": 64})
        self.assertIn("features", enriched)
        self.assertGreater(prediction["score"], 0.5)
        self.assertEqual(reasons, [])

    def test_static_pruning_rejects_large_tile(self) -> None:
        candidate = enrich_candidate(
            {"scop": 0, "node": 1, "tr": "TILE_2D", "args": [16, 1024]},
            DummyNode(),
        )
        reasons = static_prune_candidate(candidate, {"max_tile_size": 128})
        self.assertTrue(any("max_tile_size" in reason for reason in reasons))

    def test_analytical_score_uses_scop_classification(self) -> None:
        memory_heavy = enrich_candidate(
            {
                "scop": 0,
                "node": 1,
                "tr": "TILE_2D",
                "args": [32, 32],
                "scop_classification": {
                    "labels": ["band", "2d_loop", "memory_heavy_hint"],
                    "loop_rank": 2,
                    "sequence_child_count": 0,
                },
            },
            DummyNode(),
        )
        shallow = enrich_candidate(
            {
                "scop": 0,
                "node": 1,
                "tr": "TILE_2D",
                "args": [32, 32],
                "scop_classification": {
                    "labels": ["band", "1d_loop", "small_scop"],
                    "loop_rank": 1,
                    "sequence_child_count": 0,
                },
            },
            DummyNode(),
        )
        good = analytical_score(memory_heavy, {"preferred_tile_size": 32})
        bad = analytical_score(shallow, {"preferred_tile_size": 32})
        self.assertGreater(good["score"], bad["score"])
        self.assertGreater(bad["risk"], good["risk"])

    def test_analytical_score_uses_simplified_pluto_distance(self) -> None:
        candidate = enrich_candidate(
            {
                "scop": 0,
                "node": 1,
                "tr": "TILE_2D",
                "args": [8, 8],
                "scop_classification": {
                    "labels": ["band", "2d_loop", "memory_heavy_hint"],
                    "loop_rank": 2,
                    "sequence_child_count": 0,
                },
            },
            PlutoNode(),
        )
        prediction = analytical_score(candidate, {"preferred_tile_size": 32})
        features = candidate["features"]
        self.assertEqual(features["pluto_max_distance"], 4.0)
        self.assertGreater(features["pluto_reuse_pair_count"], 0)
        self.assertTrue(
            any("Pluto-style tiling prior" in reason for reason in prediction["reasons"])
        )

    def test_static_pruning_uses_scop_classification(self) -> None:
        candidate = enrich_candidate(
            {
                "scop": 0,
                "node": 1,
                "tr": "INTERCHANGE",
                "args": [],
                "scop_classification": {
                    "labels": ["band", "1d_loop"],
                    "loop_rank": 1,
                },
            },
            DummyNode(),
        )
        reasons = static_prune_candidate(candidate, {})
        self.assertTrue(any("loop_rank" in reason for reason in reasons))

    def test_sequence_pruning_allows_same_transform_on_different_scopes(self) -> None:
        reasons = static_prune_sequence(
            [
                {"scop": 0, "node": 2, "tr": "TILE_1D", "args": [32]},
                {"scop": 4, "node": 3, "tr": "TILE_1D", "args": [16]},
            ],
            {"max_same_transform_per_sequence": 1},
        )
        self.assertEqual(reasons, [])

    def test_sequence_pruning_rejects_repeated_transform_on_same_target(self) -> None:
        reasons = static_prune_sequence(
            [
                {"scop": 0, "node": 2, "tr": "TILE_1D", "args": [32]},
                {"scop": 0, "node": 2, "tr": "TILE_1D", "args": [16]},
            ],
            {"max_same_transform_per_sequence": 1},
        )
        self.assertTrue(any("scop=0, node=2" in reason for reason in reasons))

    def test_kernel_timing_delta_classifies_single_kernel_regression(self) -> None:
        analysis = analyze_kernel_timing_deltas(
            {
                "sycl_kernel_1_avg_s": 1.0,
                "sycl_kernel_2_avg_s": 1.0,
            },
            {
                "sycl_kernel_1_avg_s": 1.0,
                "sycl_kernel_2_avg_s": 1.6,
            },
            [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}],
        )
        self.assertEqual(analysis["classification"], "single_kernel_regression")
        self.assertEqual(analysis["worst_kernel"], 2)
        self.assertEqual(analysis["attribution"][0]["suspected_kernel"], 2)

    def test_history_retrieves_available_successful_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.jsonl")
            transforms = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [16]}]
            append_history(
                path,
                {
                    "status": "complete",
                    "source_hash": "abc",
                    "speedup": 1.3,
                    "transforms": transforms,
                },
            )
            records = load_history(path)
            retrieved = retrieve_sequences(
                history=records,
                source_hash="abc",
                candidates=list(transforms),
                limit=1,
            )
            self.assertEqual(retrieved, [transforms])

    def test_history_retrieval_uses_structural_similarity(self) -> None:
        transforms = [{"scop": 0, "node": 1, "tr": "FUSE", "args": [0, 1]}]
        candidates = list(transforms)
        sig = {
            "scop_count": 1,
            "node_count": 1,
            "transform_counts": {"FUSE": 1},
            "node_type_counts": {"NodeType.BAND": 1},
            "available_transform_counts": {"FUSE": 1},
        }
        retrieved = retrieve_sequences(
            history=[
                {
                    "status": "complete",
                    "source_hash": "different",
                    "structural_signature": sig,
                    "speedup": 1.2,
                    "transforms": transforms,
                }
            ],
            source_hash="current",
            structural_sig=sig,
            candidates=candidates,
            limit=1,
        )
        self.assertEqual(retrieved, [transforms])

    def test_cleanup_trial_artifacts_removes_only_trial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            original = root / "main.cpp"
            generated = root / "main-INFIX-enhanced-3.o"
            output = build / "main-trial-3"
            trial_obj = build / "main-trial-3-source.o"
            unrelated_obj = root / "keep-me.o"
            for path in [original, generated, output, trial_obj, unrelated_obj]:
                path.write_text("x", encoding="utf-8")

            app = SimpleNamespace(
                project_root=root,
                build_dir=build,
                source=generated,
                original_source=original,
                output_binary=output,
            )

            removed = cleanup_trial_artifacts(app, generated_infix="enhanced-3")
            self.assertEqual({path.name for path in removed}, {
                "main-INFIX-enhanced-3.o",
                "main-trial-3",
                "main-trial-3-source.o",
            })
            self.assertTrue(original.exists())
            self.assertTrue(unrelated_obj.exists())

    def test_print_candidates_suppresses_details_by_default(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            print_candidates(
                [{"scop": 0, "node": 1, "tr": "TILE_2D", "args": [8, 8]}],
                verbose=False,
            )
        text = buf.getvalue()
        self.assertIn("Enumerated 1 candidate", text)
        self.assertIn("suppressed", text)
        self.assertNotIn("TILE_2D", text)

    def test_select_diverse_top_k_keeps_multiple_transform_types(self) -> None:
        candidates = [
            {"tr": "TILE_1D", "predictions": {"score": 0.9, "risk": 0.1}},
            {"tr": "TILE_1D", "predictions": {"score": 0.89, "risk": 0.1}},
            {"tr": "TILE_1D", "predictions": {"score": 0.88, "risk": 0.1}},
            {"tr": "FUSE", "predictions": {"score": 0.7, "risk": 0.2}},
            {"tr": "FULL_SHIFT_VAL", "predictions": {"score": 0.65, "risk": 0.2}},
        ]
        selected = select_diverse_top_k(candidates, 3)
        self.assertEqual(
            {candidate["tr"] for candidate in selected},
            {"TILE_1D", "FUSE", "FULL_SHIFT_VAL"},
        )

    def test_adaptive_tree_prunes_bad_prefixes(self) -> None:
        tree = AdaptiveTreeState(
            constraints={"tree_prune_min_visits": 2, "tree_prune_best_speedup_below": 0.95},
            rng=__import__("random").Random(0),
        )
        prefix = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}]
        tree.update(prefix, 0.8, "early_stop")
        self.assertFalse(tree.is_bad_prefix(prefix))
        tree.update(prefix, 0.7, "early_stop")
        self.assertTrue(tree.is_bad_prefix(prefix))

    def test_adaptive_tree_tracks_tried_sequences(self) -> None:
        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        prefix = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}]
        tree.mark_tried(prefix)
        self.assertIn(sequence_signature(prefix), tree.tried_sequences)

    def test_adaptive_tree_tracks_terminal_sequences_separately(self) -> None:
        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        prefix = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}]
        child = [*prefix, {"scop": 1, "node": 1, "tr": "INTERCHANGE", "args": []}]
        tree.mark_existing(prefix, 1.1, "complete")
        self.assertTrue(tree.is_terminal_evaluated(prefix))
        self.assertFalse(tree.is_terminal_evaluated(child))
        self.assertIn(sequence_signature(prefix), tree.tried_sequences)

    def test_adaptive_tree_saturates_terminal_repeats(self) -> None:
        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        prefix = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}]
        tree.mark_existing(prefix, 1.1, "complete")
        tree.mark_terminal_repeat(prefix)
        self.assertTrue(tree.is_saturated(prefix))
        self.assertEqual(tree.terminal_repeat_counts[sequence_signature(prefix)], 1)

    def test_saturated_prefixes_are_not_ranked_for_selection(self) -> None:
        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        saturated = {"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}
        other = {"scop": 1, "node": 1, "tr": "INTERCHANGE", "args": []}
        tree.mark_saturated([saturated], "test")
        ranked = _sort_tree_candidates([saturated, other], tree, [])
        self.assertEqual(ranked, [other])

    def test_adaptive_tree_disables_scops_after_filename_too_long(self) -> None:
        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        specs = [{"scop": 6, "node": 22, "tr": "TILE_1D", "args": [16]}]
        tree.disable_scops_from_specs(specs, "JScop filename too long")
        self.assertIn(6, tree.disabled_scops)
        self.assertTrue(tree.is_blacklisted_candidate(specs[0]))

    def test_hot_kernel_filter_keeps_scops_mapped_to_hot_kernels(self) -> None:
        candidates = [
            {"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]},
            {"scop": 1, "node": 1, "tr": "TILE_1D", "args": [32]},
            {"scop": 2, "node": 1, "tr": "TILE_1D", "args": [32]},
        ]
        metrics = {
            "sycl_kernel_1_avg_s": 1.0,
            "sycl_kernel_2_avg_s": 5.0,
            "sycl_kernel_3_avg_s": 0.5,
        }
        hot_filter = build_hot_kernel_filter(candidates, metrics, {"hot_kernel_top_k": 1})
        self.assertTrue(hot_filter.enabled)
        self.assertEqual(hot_filter.hot_kernels, [2])
        self.assertEqual(hot_filter.allowed_scops, {1})
        filtered = apply_hot_kernel_filter_to_candidates(candidates, hot_filter)
        self.assertEqual([candidate["scop"] for candidate in filtered], [1])
        self.assertEqual(filtered[0]["source_info"]["suspected_kernel"], 2)

    def test_hot_kernel_filter_disables_when_only_one_kernel_is_measured(self) -> None:
        candidates = [{"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]}]
        hot_filter = build_hot_kernel_filter(candidates, {"sycl_kernel_1_avg_s": 1.0}, {})
        self.assertFalse(hot_filter.enabled)

    def test_search_space_exhausted_is_a_prune_signal(self) -> None:
        self.assertTrue(issubclass(SearchSpaceExhausted, Exception))

    def test_long_jscop_backup_path_is_shortened(self) -> None:
        path = Path("/tmp") / ("x" * 240 + ".jscop.bak")
        shortened = _short_jscop_backup_path(path)
        self.assertLess(len(shortened.name.encode("utf-8")), 255)
        self.assertTrue(shortened.name.startswith("jscop-backup-"))

    def test_seed_tree_disables_historic_filename_too_long_scops(self) -> None:
        from src.polyMorph.runner import seed_tree_from_records

        tree = AdaptiveTreeState(constraints={}, rng=__import__("random").Random(0))
        specs = [{"scop": 5, "node": 22, "tr": "TILE_1D", "args": [32]}]
        seed_tree_from_records(
            tree,
            [
                {
                    "status": "failed",
                    "transforms": specs,
                    "failure": "[Errno 36] File name too long: bad.jscop.bak",
                }
            ],
        )
        self.assertIn(5, tree.disabled_scops)
        self.assertTrue(tree.is_blacklisted_candidate(specs[0]))

    def test_adaptive_tree_disables_fuse_after_enough_screening_failures(self) -> None:
        tree = AdaptiveTreeState(
            constraints={"disable_family_after_failures": 2, "disable_family_failure_fraction": 0.5},
            rng=__import__("random").Random(0),
        )
        candidates = [
            {"scop": 0, "node": 2, "tr": "FUSE", "args": [0, 1]},
            {"scop": 0, "node": 2, "tr": "FUSE", "args": [1, 2]},
            {"scop": 1, "node": 2, "tr": "FUSE", "args": [0, 1]},
            {"scop": 1, "node": 2, "tr": "FUSE", "args": [1, 2]},
        ]
        tree.mark_existing([candidates[0]], 0.0, "failed")
        self.assertFalse(tree.disable_family_if_screening_unpromising("FUSE", candidates))
        tree.mark_existing([candidates[1]], 0.0, "failed")
        self.assertTrue(tree.disable_family_if_screening_unpromising("FUSE", candidates))
        self.assertIn("FUSE", tree.disabled_families)
        self.assertTrue(tree.is_blacklisted_candidate(candidates[0]))

    def test_fuse_candidate_args_are_checked_against_sequence_children(self) -> None:
        node = SimpleNamespace(node_type="NodeType.SEQUENCE", yaml_str="sequence:\n- filter: A\n- filter: B\n")
        self.assertIsNone(candidate_args_invalid_for_node("FUSE", [0, 1], node))
        self.assertIn("exceed", candidate_args_invalid_for_node("FUSE", [0, 2], node) or "")
        self.assertIsNone(candidate_args_invalid_for_node("TILE_1D", [32], node))
        band_node = SimpleNamespace(node_type="NodeType.BAND", yaml_str="sequence:\n- filter: A\n- filter: B\n")
        self.assertIn("sequence", candidate_args_invalid_for_node("FUSE", [0, 1], band_node) or "")

    def test_legality_aware_fuse_args_derive_adjacent_children(self) -> None:
        node = SimpleNamespace(node_type="NodeType.SEQUENCE", yaml_str="sequence:\n- filter: A\n- filter: B\n- filter: C\n")
        band_node = SimpleNamespace(node_type="NodeType.BAND", yaml_str="sequence:\n- filter: A\n- filter: B\n- filter: C\n")
        poly = SimpleNamespace(
            search=SimpleNamespace(
                legality_aware_args=True,
                tile_sizes=[8],
            )
        )
        self.assertEqual(candidate_args_for_node("FUSE", poly, node), [[0, 1], [1, 2]])
        self.assertEqual(candidate_args_for_node("FUSE", poly, band_node), [])

    def test_transform_args_are_inferred_without_explicit_args(self) -> None:
        node = SimpleNamespace(yaml_str="[p_0, p_1] -> { Stmt[i0, i1] }")
        band_node = SimpleNamespace(yaml_str="[p_0, p_1] -> { Stmt[i0, i1] }", band_member_count=2)
        poly = SimpleNamespace(
            search=SimpleNamespace(
                legality_aware_args=True,
                tile_sizes=[8, 16],
                constraints={"max_abs_shift": 1},
            )
        )
        self.assertEqual(candidate_args_for_node("TILE_2D", poly, node), [[8, 8], [16, 16]])
        self.assertIn([1, 1], candidate_args_for_node("FULL_SHIFT_VAR", poly, node))
        self.assertIn([0, 1], candidate_args_for_node("FULL_SHIFT_PARAM", poly, node))
        self.assertEqual(candidate_args_for_node("SET_LOOP_OPT", poly, node), [])
        self.assertEqual(
            candidate_args_for_node("SET_LOOP_OPT", poly, band_node),
            [[0, 1], [0, 3], [1, 1], [1, 3]],
        )

    def test_legality_aware_args_use_tadashi_arg_descriptors(self) -> None:
        class DescriptorNode:
            yaml_str = "[p_0] -> { Stmt[i0, i1] }"

            def __init__(self, descriptors, valid):
                self._descriptors = descriptors
                self._valid = valid

            def available_args(self, _tr):
                return self._descriptors

            def valid_args(self, _tr, *args):
                return tuple(args) in self._valid

        class Bound:
            def __init__(self, lower=None, upper=None):
                self.lower = lower
                self.upper = upper

        poly = SimpleNamespace(
            search=SimpleNamespace(
                legality_aware_args=True,
                tile_sizes=[8, 16, 32],
                constraints={"max_abs_shift": 1},
            )
        )
        split_node = DescriptorNode([Bound(1, 3)], {(1,)})
        shift_node = DescriptorNode([[0], Bound()], {(0, -1), (0, 1)})

        self.assertEqual(candidate_args_for_node("SPLIT", poly, split_node), [[1]])
        self.assertEqual(
            candidate_args_for_node("FULL_SHIFT_VAR", poly, shift_node),
            [[0, -1], [0, 1]],
        )

    def test_correctness_output_capture_and_numeric_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "result.txt"
            output.write_text("value 1.000000\n", encoding="utf-8")
            app = SimpleNamespace(project_root=root, output_binary=root / "bin", runtime_args=[])
            cfg = SimpleNamespace(backend="parser", parser=SimpleNamespace(run_cwd="workdir"))
            poly = SimpleNamespace(
                search=SimpleNamespace(
                    correctness_outputs=["result.txt"],
                    correctness_required=True,
                    correctness_tolerance=1.0e-4,
                )
            )
            baseline = capture_correctness_outputs(app, cfg, poly)
            output.write_text("value 1.000050\n", encoding="utf-8")
            result = verify_correctness_outputs(baseline, app, cfg, poly)
            self.assertTrue(result["ok"])

    def test_learned_candidate_model_updates_scores(self) -> None:
        candidate = {
            "tr": "TILE_1D",
            "args": [32],
            "predictions": {"score": 0.5, "risk": 0.1, "reasons": []},
        }
        model = build_learned_candidate_model(
            [
                {
                    "status": "complete",
                    "speedup": 1.5,
                    "transforms": [{"tr": "TILE_1D", "args": [32]}],
                    "backend_sensitivity": {
                        "per_backend": {"cuda": {"ok": True, "objective": 1.0}},
                    },
                }
            ],
            target_backend="cuda",
        )
        apply_learned_model_to_candidates([candidate], model, min_observations=1)
        self.assertGreater(candidate["predictions"]["score"], 0.5)
        self.assertEqual(candidate["predictions"]["learned_observations"], 1)

    def test_sequence_feature_summary_records_transform_shape(self) -> None:
        summary = sequence_feature_summary([
            {"scop": 0, "node": 1, "tr": "TILE_1D", "args": [32]},
            {"scop": 1, "node": 2, "tr": "FULL_SHIFT_VAL", "args": [1]},
        ])
        self.assertEqual(summary["length"], 2)
        self.assertEqual(summary["transform_counts"]["TILE_1D"], 1)
        self.assertEqual(summary["max_tile_size"], 32.0)

    def test_evaluation_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            append_evaluation_cache(path, {"key": "abc", "status": "complete", "metrics": {"runtime": 1.0}})
            loaded = load_evaluation_cache(path)
            self.assertEqual(loaded["abc"]["metrics"]["runtime"], 1.0)

    def test_structural_signature_counts_candidate_shapes(self) -> None:
        sig = structural_signature(
            [
                {
                    "scop": 0,
                    "node": 1,
                    "tr": "FUSE",
                    "features": {
                        "node_type": "NodeType.BAND",
                        "available_transformations": ["FUSE", "TILE_1D"],
                    },
                }
            ]
        )
        self.assertEqual(sig["transform_counts"]["FUSE"], 1)
        self.assertEqual(sig["available_transform_counts"]["TILE_1D"], 1)

    def test_suppress_external_viewers_prepends_noop_dotty(self) -> None:
        old_path = os.environ.get("PATH", "")
        with suppress_external_viewers():
            first = Path(os.environ["PATH"].split(os.pathsep)[0])
            self.assertTrue((first / "dotty").exists())
        self.assertEqual(os.environ.get("PATH", ""), old_path)


if __name__ == "__main__":
    unittest.main()
