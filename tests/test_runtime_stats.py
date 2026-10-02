from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.stats import RuntimeStats, resolve_stats_path


def _read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.reader(fp))


class RuntimeStatsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.stats = RuntimeStats(flush_interval=0.0)
        self.path = self.root / "stats" / "scout_stats.csv"
        self.stats.start(self.path)

    def tearDown(self):
        self.stats.finish()
        self.stats.path = None  # keep the atexit hook from recreating the file
        self.tempdir.cleanup()

    def test_records_build_and_run_totals(self):
        with self.stats.time_build():
            time.sleep(0.01)
        with self.stats.time_run():
            time.sleep(0.02)

        snap = self.stats.snapshot()
        self.assertEqual(snap["build_count"], 1)
        self.assertEqual(snap["run_count"], 1)
        self.assertGreater(snap["build_time_s"], 0.0)
        self.assertGreater(snap["run_time_s"], 0.0)
        remainder = snap["elapsed_s"] - snap["build_time_s"] - snap["run_time_s"]
        self.assertAlmostEqual(snap["scout_time_s"], max(0.0, remainder), places=6)

    def test_scout_time_never_negative(self):
        snap = self.stats.snapshot()
        self.assertGreaterEqual(snap["scout_time_s"], 0.0)

    def test_flush_writes_header_then_appends(self):
        self.stats.flush()
        self.stats.flush()
        rows = _read_rows(self.path)
        self.assertEqual(rows[0][0], "timestamp")
        self.assertEqual(len(rows), 3)  # header + two snapshots
        self.assertEqual(len(rows[1]), len(rows[0]))

    def test_throttle_limits_snapshots(self):
        throttled = RuntimeStats(flush_interval=3600.0)
        path = self.root / "throttled.csv"
        throttled.start(path)
        throttled.maybe_flush()  # first call always writes
        throttled.maybe_flush()  # within interval -> skipped
        throttled.maybe_flush()  # within interval -> skipped
        rows = _read_rows(path)
        throttled.finish()  # final flush; not part of the assertion
        throttled.path = None
        self.assertEqual(len(rows), 2)  # header + one snapshot

    def test_resolve_stats_path_precedence(self):
        cli = self.root / "cli.csv"
        config = self.root / "config.csv"
        env = {"SCOUT_STATS_FILE": str(self.root / "env.csv")}
        self.assertEqual(resolve_stats_path(cli, config, env), cli)
        self.assertEqual(resolve_stats_path(None, config, env), config)
        self.assertEqual(resolve_stats_path(None, None, env), self.root / "env.csv")
        self.assertEqual(resolve_stats_path(None, None, {}), Path.cwd() / "scout_stats.csv")


class EvaluatorStatsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_config(self) -> Path:
        source = self.root / "program.cpp"
        source.write_text("int main() { return 0; }\n")
        raw = {
            "backend": "perf",
            "source": str(source),
            "compiler": "c++",
            "compiler_flags": ["-O2"],
            "env": {"MODE": ["a", "b"]},
            "objectives": [{"metric": "CPI", "goal": "min"}],
            "perf": {"events": ["cycles", "instructions"]},
            "search": {"study": "optuna", "sampler": "rs"},
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(raw))
        return path

    def test_evaluator_records_build_and_run_timing(self):
        # Imported lazily: the evaluator pulls in Optuna, which the pure-stats
        # tests above deliberately avoid needing.
        from src import evaluator as evaluator_mod
        from src.config import Config
        from src.evaluator import Evaluator

        cfg = Config.load(self.write_config())
        stats = RuntimeStats(flush_interval=3600.0)
        stats.start(self.root / "stats.csv")

        def fake_compile(_compiler, _source, _flags, output, trial=None):
            time.sleep(0.005)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("binary")
            output.chmod(0o755)
            return output

        def fake_measure(_cfg, _binary, _args, env, _runs, **_kwargs):
            time.sleep(0.005)
            return {"CPI": 1.0 if env["MODE"] == "a" else 2.0}

        try:
            with patch.object(evaluator_mod, "STATS", stats), patch(
                "src.evaluator.compile_single_source", fake_compile
            ), patch("src.evaluator.measure_perf", fake_measure):
                evaluator = Evaluator(cfg, self.root / "work")
                evaluator.evaluate("-O2", {"MODE": "a"}, self.root / "run-a")
                evaluator.evaluate("-O2", {"MODE": "b"}, self.root / "run-b")
                evaluator.evaluate("-O2", {"MODE": "a"}, self.root / "run-a2")
        finally:
            stats.finish()
            stats.path = None

        snap = stats.snapshot()
        # One real build; two measurements (the third call is a cached evaluation).
        self.assertEqual(snap["build_count"], 1)
        self.assertEqual(snap["run_count"], 2)
        self.assertGreater(snap["build_time_s"], 0.0)
        self.assertGreater(snap["run_time_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
