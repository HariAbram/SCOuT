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
from src.polyMorph.features import enrich_candidate
from src.polyMorph.feedback import (
    analyze_runtime_feedback,
    parse_adaptivecpp_runtime_feedback,
    parse_compiler_feedback,
)
from src.polyMorph.history import append_history, load_history, retrieve_sequences
from src.polyMorph.pruning import analytical_score, static_prune_candidate, static_prune_sequence
from src.polyMorph.runner import cleanup_trial_artifacts, print_candidates, suppress_external_viewers


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
            }
        )
        self.assertTrue(spec.static_pruning)
        self.assertTrue(spec.analytical_model)
        self.assertTrue(spec.constraint_aware)
        self.assertTrue(spec.compiler_feedback)
        self.assertTrue(spec.runtime_feedback)
        self.assertTrue(spec.case_retrieval)
        self.assertEqual(spec.top_k, 7)
        self.assertEqual(spec.retrieval_top_k, 2)
        self.assertEqual(spec.constraints["max_tile_size"], 64)
        self.assertEqual(spec.block_transforms, [])
        self.assertEqual(spec.runtime_feedback_masks, ["omp", "cuda"])
        self.assertEqual(spec.runtime_feedback_env["XDG_CACHE_HOME"], "/tmp/acpp-cache")

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

    def test_suppress_external_viewers_prepends_noop_dotty(self) -> None:
        old_path = os.environ.get("PATH", "")
        with suppress_external_viewers():
            first = Path(os.environ["PATH"].split(os.pathsep)[0])
            self.assertTrue((first / "dotty").exists())
        self.assertEqual(os.environ.get("PATH", ""), old_path)


if __name__ == "__main__":
    unittest.main()
