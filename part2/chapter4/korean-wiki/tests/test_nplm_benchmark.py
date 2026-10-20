"""Small correctness/protection checks; no production throughput measurements."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_nplm as benchmark


class ModelTests(unittest.TestCase):
    def test_parameter_formula_equals_actual_model_including_biases(self):
        config = benchmark.BenchmarkConfig(vocab_size=17, context_size=3, embedding_dim=4,
                                            hidden_dim=6, batch_size=4, cpu_threads=1)
        model = benchmark.build_model(config)
        self.assertEqual(benchmark.parameter_count(17, 3, 4, 6), 265)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 265)
        memory = benchmark.theoretical_memory(config)
        self.assertEqual(memory["parameter_gradient_adam_bytes_lower_bound"], 265 * 16)
        self.assertEqual(memory["single_logits_tensor_bytes"], 4 * 17 * 4)
        self.assertIn("lower bound", memory["scope"])

    def test_finite_gradients_and_actual_parameter_update(self):
        import torch
        config = benchmark.BenchmarkConfig(vocab_size=11, context_size=3, embedding_dim=4,
                                            hidden_dim=8, batch_size=4, cpu_threads=1)
        benchmark._configure_torch(config)
        model = benchmark.build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
        contexts = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        targets = torch.tensor([3, 4, 5, 6])
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = benchmark.train_step(model, optimizer, contexts, targets)
        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertTrue(all(parameter.grad is not None and bool(torch.all(torch.isfinite(parameter.grad)))
                            for parameter in model.parameters()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        self.assertTrue(optimizer.state)

    def test_loss_decreases_on_fixed_tiny_batch_correctness_only(self):
        import torch
        config = benchmark.BenchmarkConfig(vocab_size=7, context_size=2, embedding_dim=4,
                                            hidden_dim=8, batch_size=4, cpu_threads=1)
        benchmark._configure_torch(config)
        model = benchmark.build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.04, foreach=False, fused=False)
        contexts = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
        targets = torch.tensor([2, 3, 4, 5])
        with torch.no_grad():
            initial = float(torch.nn.functional.cross_entropy(model(contexts), targets))
        for _ in range(12):
            benchmark.train_step(model, optimizer, contexts, targets)
        with torch.no_grad():
            final = float(torch.nn.functional.cross_entropy(model(contexts), targets))
        self.assertTrue(math.isfinite(final))
        self.assertLess(final, initial)


class GuardTests(unittest.TestCase):
    def test_all_partial_or_running_manifests_block_timing(self):
        for filename in ["step5-progress.json", "step5-state.json", "step6-numeric-state.json",
                         "step6-numeric-progress.json", "token-frequency-state.json",
                         "vocabulary-coverage-progress.json"]:
            for status in ["partial", "running", "waiting_for_source", "waiting_for_raw_frequencies"]:
                with self.subTest(filename=filename, status=status), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    reports = root / "reports"
                    reports.mkdir()
                    (reports / filename).write_text(json.dumps({"status": status}))
                    snapshot = benchmark.pipeline_idle_snapshot(root, check_processes=False)
                    self.assertFalse(snapshot["idle"])
                    self.assertIn(filename, snapshot["blockers"][0])

    def test_complete_manifest_and_inaccessible_manifest_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            path = reports / "step5-state.json"
            path.write_text(json.dumps({"status": "complete"}))
            self.assertTrue(benchmark.pipeline_idle_snapshot(root, check_processes=False)["idle"])
            path.write_text("{not complete json")
            self.assertFalse(benchmark.pipeline_idle_snapshot(root, check_processes=False)["idle"])

    def test_live_worker_process_blocks_timing_even_before_its_progress_file(self):
        listing = "1234 /workspace/.venv/bin/python /workspace/analyze_vocabulary_coverage.py\n"
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, listing, "")
            snapshot = benchmark.pipeline_idle_snapshot(directory)
            self.assertFalse(snapshot["idle"])
            self.assertEqual(snapshot["active_processes"][0]["pid"], 1234)
            with self.assertRaises(RuntimeError):
                benchmark.assert_pipeline_idle(directory)

    def test_busy_guard_refuses_suite_before_starting_a_model_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            (root / "reports/step6-numeric-progress.json").write_text('{"status":"partial"}')
            with patch.object(benchmark.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "", "")
                with self.assertRaises(RuntimeError):
                    benchmark.run_suite([benchmark.BenchmarkConfig()], root / "result.json", root)
                self.assertEqual(run.call_count, 1)  # Only ps, no training process.
            self.assertFalse((root / "result.json").exists())


class MetadataTests(unittest.TestCase):
    def test_smoke_has_no_timing_or_training_quality_claim(self):
        result = benchmark.smoke_test(["cpu"])
        self.assertEqual(result["status"], "smoke-test-only")
        self.assertFalse(result["timing_comparison_valid"])
        self.assertFalse(result["corpus_training_performed"])
        self.assertIsNone(result["throughput"])
        self.assertIsNone(result["time_budgets"])
        self.assertTrue(result["results"][0]["parameter_updated"])

    def test_budget_units_are_prediction_events_and_explicit_extrapolations(self):
        result = benchmark.extrapolated_budgets(100, 10, 0.75)
        self.assertEqual(result["kind"], "EXTRAPOLATION_ONLY")
        self.assertEqual(result["budgets"]["3h"]["target_prediction_events"], 810000)
        self.assertEqual(result["budgets"]["8h"]["target_prediction_events"], 2160000)
        self.assertIn("not unique tokens", result["unit"])
        self.assertIsNone(result["quality_results"])
        for rate in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                benchmark.extrapolated_budgets(rate, 10, 0.75)

    def test_bounded_configuration_and_invalid_settings(self):
        config = benchmark.BenchmarkConfig()
        config.validate()
        for change in [{"vocab_size": 1}, {"device": "cuda"}, {"repeats": 21},
                       {"steps_per_repeat": 101}, {"warmup_steps": 0}, {"batch_size": 0},
                       {"max_seconds": 301}, {"utilization_margin": 0},
                       {"batch_size": 100000}, {"hidden_dim": 1000000}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config, **change).validate()

    def test_suite_isolated_child_metadata_and_existing_report_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "benchmark.json"
            config = benchmark.BenchmarkConfig(vocab_size=17, context_size=3, embedding_dim=4,
                                                hidden_dim=8, batch_size=4, cpu_threads=1)
            commands = []

            def fake_child(command, **kwargs):
                commands.append((command, kwargs))
                target = Path(command[command.index("--child-output") + 1])
                target.write_text(json.dumps({"status": "complete", "kind": "SYNTHETIC_MICROBENCHMARK"}))
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.object(benchmark, "assert_pipeline_idle", return_value={"idle": True}), \
                    patch.object(benchmark.subprocess, "run", side_effect=fake_child):
                result = benchmark.run_suite([config], output, root)
                self.assertEqual(result["status"], "complete")
                self.assertIsNone(result["actual_3h_or_8h_results"])
                self.assertIsNone(result["quality_results"])
                self.assertFalse(result["corpus_training_performed"])
                self.assertEqual(len(commands), 1)
                self.assertEqual(commands[0][1]["env"]["OMP_NUM_THREADS"], "1")
                self.assertEqual(commands[0][1]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
                original = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    benchmark.run_suite([config], output, root)
                self.assertEqual(output.read_bytes(), original)

    def test_experiment_plan_separates_time_vocabulary_data_and_heldout(self):
        plan = benchmark.EXPERIMENT_DESIGN
        for text in ["같은 8시간 학습 경로", "3시간과 8시간 체크포인트", "문서 ID", "train split에서만",
                     "사전 크기 비교", "데이터 양 비교", "예측한 target 수", "고유 학습 데이터 수"]:
            self.assertIn(text, plan)


if __name__ == "__main__":
    unittest.main()
