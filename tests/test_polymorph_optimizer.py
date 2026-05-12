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
from src.polyMorph.features import enrich_candidate, sequence_feature_summary, structural_signature
from src.polyMorph.feedback import (
    analyze_runtime_feedback,
    parse_adaptivecpp_runtime_feedback,
    parse_compiler_feedback,
)
from src.polyMorph.history import append_history, load_history, retrieve_sequences
from src.polyMorph.pruning import analytical_score, static_prune_candidate, static_prune_sequence
from src.polyMorph.runner import (
    AdaptiveTreeState,
    append_evaluation_cache,
    apply_learned_model_to_candidates,
    build_learned_candidate_model,
    candidate_args_invalid_for_node,
    candidate_args_for_node,
    capture_correctness_outputs,
    cleanup_trial_artifacts,
    load_evaluation_cache,
    print_candidates,
    select_diverse_top_k,
    suppress_external_viewers,
    verify_correctness_outputs,
)


class DummyNode:
    node_type = "band"
    yaml_str = "schedule: loop\nread: A[i]\nwrite: B[i]\n"
    available_transformations = []


class PolyMorphOptimizerTests(unittest.TestCase):
    def test_config_defaults_keep_new_features_disabled(self) -> None:
        spec = PolyMorphSearchSpec.from_dict({})
        self.assertFalse(spec.static_pruning)
        self.assertFalse(spec.analytical_model)
        self.assertFalse(spec.constraint_aware)
        self.assertFalse(spec.compiler_feedback)
        self.assertFalse(spec.case_retrieval)
        self.assertIn("SET_PARALLEL", spec.block_transforms)

    def test_config_parses_new_options(self) -> None:
        spec = PolyMorphSearchSpec.from_dict(
            {
                "static_pruning": True,
                "analytical_model": True,
                "constraint_aware": True,
                "compiler_feedback": True,
                "runtime_feedback": True,
                "case_retrieval": True,
                "structural_retrieval": False,
                "search_strategy": "beam_optuna",
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
                "compiler_feedback_flags": ["-Rpass=loop-vectorize"],
                "runtime_feedback_masks": ["omp", "cuda"],
                "runtime_feedback_debug_level": 4,
                "runtime_feedback_repeat": 1,
                "runtime_feedback_env": {"XDG_CACHE_HOME": "/tmp/acpp-cache"},
                "legality_aware_args": False,
                "correctness_outputs": ["result.txt"],
                "correctness_tolerance": 1.0e-4,
                "correctness_required": False,
                "learned_model": False,
                "learned_model_min_observations": 2,
                "target_backend": "cuda",
            }
        )
        self.assertTrue(spec.static_pruning)
        self.assertTrue(spec.analytical_model)
        self.assertTrue(spec.constraint_aware)
        self.assertTrue(spec.compiler_feedback)
        self.assertTrue(spec.runtime_feedback)
        self.assertTrue(spec.case_retrieval)
        self.assertFalse(spec.structural_retrieval)
        self.assertEqual(spec.search_strategy, "adaptive_tree")
        self.assertEqual(spec.cache_jsonl, "/tmp/polymorph-cache.jsonl")
        self.assertFalse(spec.cache_evaluations)
        self.assertFalse(spec.multi_fidelity)
        self.assertEqual(spec.early_stop_worse_than, 1.4)
        self.assertEqual(spec.top_k, 7)
        self.assertEqual(spec.retrieval_top_k, 2)
        self.assertEqual(spec.constraints["max_tile_size"], 64)
        self.assertEqual(spec.block_transforms, [])
        self.assertEqual(spec.runtime_feedback_masks, ["omp", "cuda"])
        self.assertEqual(spec.runtime_feedback_env["XDG_CACHE_HOME"], "/tmp/acpp-cache")
        self.assertFalse(spec.legality_aware_args)
        self.assertEqual(spec.correctness_outputs, ["result.txt"])
        self.assertEqual(spec.correctness_tolerance, 1.0e-4)
        self.assertFalse(spec.correctness_required)
        self.assertFalse(spec.learned_model)
        self.assertEqual(spec.learned_model_min_observations, 2)
        self.assertEqual(spec.target_backend, "cuda")

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

    def test_feedback_parser_finds_vectorization_and_pressure_hints(self) -> None:
        feedback = parse_compiler_feedback(
            stderr=(
                "remark: loop vectorized\n"
                "warning: loop not vectorized: unsafe dependency\n"
                "register pressure is high\n"
            )
        )
        self.assertEqual(feedback["vectorized_count"], 1)
        self.assertEqual(feedback["missed_vectorization_count"], 1)
        self.assertTrue(feedback["register_pressure_hint"])

    def test_adaptivecpp_runtime_feedback_parser_counts_jit_signals(self) -> None:
        feedback = parse_adaptivecpp_runtime_feedback(
            stdout=(
                "[AdaptiveCpp Info] backend_loader: Successfully opened plugin: librt-backend-omp.so\n"
                "[AdaptiveCpp Info] Registering backend: 'omp'...\n"
                "[AdaptiveCpp Info] kernel_cache: Registering kernel foo\n"
                "[AdaptiveCpp Info] hcf_cache: Registering HCF object 123...\n"
                "[AdaptiveCpp Info] hcf_cache: Registering kernel info for kernel foo\n"
                "[AdaptiveCpp Info] adaptivity_engine: Inferred pointer alignment of 32 for kernel argument 0\n"
                "[AdaptiveCpp Info] adaptivity_engine: Inferred noalias pointer semantics for kernel argument 0\n"
                "[AdaptiveCpp Info] kernel_cache: Persistent cache hit for id 1.2\n"
                "[AdaptiveCpp Info] Load module: /tmp/a.jit.so\n"
                "[AdaptiveCpp Info] omp_queue: Successfully compiled SSCP kernels to module 0x1\n"
                "[AdaptiveCpp Info] omp_queue: Submitting kernel...\n"
                "[AdaptiveCpp Info] runtime: ******* rt shutdown ********\n"
            ),
            mask="omp",
        )
        self.assertEqual(feedback["backend_plugin_count"], 1)
        self.assertEqual(feedback["registered_backend_count"], 1)
        self.assertEqual(feedback["registered_kernel_count"], 1)
        self.assertEqual(feedback["hcf_object_count"], 1)
        self.assertEqual(feedback["pointer_alignment_count"], 1)
        self.assertEqual(feedback["noalias_count"], 1)
        self.assertEqual(feedback["cache_hit_count"], 1)
        self.assertEqual(feedback["jit_compile_hint_count"], 2)
        self.assertTrue(feedback["rt_shutdown"])

    def test_runtime_feedback_analysis_penalizes_and_marks_backend_sensitivity(self) -> None:
        runtime_feedback = {
            "omp": {
                "error_count": 0,
                "warning_count": 1,
                "cache_hit_count": 100,
                "cache_miss_count": 0,
                "jit_compile_hint_count": 2,
                "kernel_submit_count": 20,
                "memcpy_submit_count": 30,
                "rt_shutdown": True,
            },
            "cuda": {
                "error_count": 2,
                "warning_count": 12,
                "cache_hit_count": 10,
                "cache_miss_count": 4,
                "jit_compile_hint_count": 50,
                "kernel_submit_count": 20,
                "memcpy_submit_count": 200,
                "rt_shutdown": False,
            },
        }
        baseline = {
            "omp": {
                "jit_compile_hint_count": 2,
                "kernel_submit_count": 20,
                "memcpy_submit_count": 30,
            },
            "cuda": {
                "jit_compile_hint_count": 2,
                "kernel_submit_count": 20,
                "memcpy_submit_count": 30,
            },
        }
        analysis = analyze_runtime_feedback(
            runtime_feedback,
            baseline=baseline,
            constraints={"runtime_jit_threshold": 5},
        )
        self.assertGreater(analysis["penalty"], 0.0)
        self.assertTrue(analysis["backend_sensitive"])
        self.assertLess(analysis["score"], 1.0)

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

    def test_fuse_candidate_args_are_checked_against_sequence_children(self) -> None:
        node = SimpleNamespace(yaml_str="sequence:\n- filter: A\n- filter: B\n")
        self.assertIsNone(candidate_args_invalid_for_node("FUSE", [0, 1], node))
        self.assertIn("exceed", candidate_args_invalid_for_node("FUSE", [0, 2], node) or "")
        self.assertIsNone(candidate_args_invalid_for_node("TILE_1D", [32], node))

    def test_legality_aware_fuse_args_derive_adjacent_children(self) -> None:
        node = SimpleNamespace(yaml_str="sequence:\n- filter: A\n- filter: B\n- filter: C\n")
        poly = SimpleNamespace(
            search=SimpleNamespace(
                legality_aware_args=True,
                tile_sizes=[8],
                scale_factors=[2],
                shift_values=[1],
            )
        )
        self.assertEqual(candidate_args_for_node("FUSE", poly, node), [[0, 1], [1, 2]])

    def test_transform_args_are_inferred_without_explicit_args(self) -> None:
        node = SimpleNamespace(yaml_str="[p_0, p_1] -> { Stmt[i0, i1] }")
        poly = SimpleNamespace(
            search=SimpleNamespace(
                legality_aware_args=True,
                tile_sizes=[8, 16],
                scale_factors=[2],
                shift_values=[-1, 1],
            )
        )
        self.assertEqual(candidate_args_for_node("TILE_2D", poly, node), [[8, 8], [16, 16]])
        self.assertIn([1, 1], candidate_args_for_node("FULL_SHIFT_VAR", poly, node))
        self.assertIn([0, 1], candidate_args_for_node("FULL_SHIFT_PARAM", poly, node))
        self.assertEqual(candidate_args_for_node("SET_LOOP_OPT", poly, node), [[0, 1], [0, 3], [1, 1], [1, 3]])

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
                    "runtime_feedback_analysis": {
                        "per_backend": {"cuda": {"score": 0.8}},
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
