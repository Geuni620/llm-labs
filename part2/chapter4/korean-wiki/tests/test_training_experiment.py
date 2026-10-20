"""Corpus correctness and tiny real weight updates; no throughput claims."""

from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_training_experiment as prepare
from training_corpus import (ArrayCorpus, PARAGRAPH_DTYPE, RESERVED, WindowSampler,
                             choose_vocabulary, document_split, encode_paragraph,
                             identity, stable_int, subset_key)


def corpus_fixture(path, context=2):
    tokens = RESERVED + ["서울/NNP", "은/JX", "<sp>/WS", "수도/NNG", "./SF", "1/SN"]
    paragraphs = [[2] * context + [4, 5, 6, 7, 8, 3], [2] * context + [9, 8, 3]]
    outputs = {}
    for split in ("train", "validation", "test"):
        ids = np.asarray(sum(paragraphs, []), dtype="<u2")
        ids.tofile(path / f"{split}.ids.u16")
        rows = [(0, len(paragraphs[0]), 0, 1, 1),
                (len(paragraphs[0]), len(paragraphs[1]), 2 ** 31, 1, 2)]
        np.save(path / f"{split}.paragraphs.npy", np.asarray(rows, dtype=PARAGRAPH_DTYPE), allow_pickle=False)
        for suffix in ("ids.u16", "paragraphs.npy"):
            outputs[f"{split}.{suffix}"] = identity(path / f"{split}.{suffix}")
    (path / "vocabulary.json").write_text(json.dumps({"tokens": tokens}))
    outputs["vocabulary.json"] = identity(path / "vocabulary.json")
    (path / "manifest.json").write_text(json.dumps({"status": "complete", "context_size": context,
                                                    "outputs": outputs}))
    return tokens


class CorpusTests(unittest.TestCase):
    def test_document_partition_and_nested_subsets(self):
        ids = [str(i) for i in range(10000)]
        partitions = {split: {i for i in ids if document_split(i) == split}
                      for split in ("train", "validation", "test")}
        self.assertTrue(all(partitions.values()))
        self.assertFalse(partitions["train"] & partitions["validation"])
        self.assertEqual(sum(map(len, partitions.values())), len(ids))
        small = {i for i in ids if subset_key(i) < int(.1 * 2 ** 32)}
        medium = {i for i in ids if subset_key(i) < int(.25 * 2 ** 32)}
        self.assertLess(small, medium)
        self.assertLess(medium, set(ids))
        self.assertEqual(document_split("page"), document_split("page"))

    def test_train_vocabulary_reserved_ids_and_exact_ties(self):
        counts = Counter({"가/NNG": 4, "나/NNG": 4, "<sp>/WS": 10, "1/SN": 2})
        self.assertEqual(choose_vocabulary(counts, 7), RESERVED + ["<sp>/WS", "가/NNG", "나/NNG"])
        with self.assertRaises(ValueError):
            choose_vocabulary(counts, 65537)
        with self.assertRaises(ValueError):
            choose_vocabulary(counts, 9)

    def test_oov_drops_whole_sentence_but_keeps_paragraph(self):
        lookup = dict(zip(RESERVED + ["서울/NNP", "./SF"], range(6)))
        ids, kept = encode_paragraph(["서울/NNP ./SF", "미등록/NNG ./SF"], lookup, 3)
        self.assertEqual(ids.tolist(), [2, 2, 2, 4, 5, 3])
        self.assertEqual(kept, 1)
        self.assertNotIn(1, ids.tolist())
        self.assertEqual(encode_paragraph(["미등록/NNG"], lookup, 3), (None, 0))

    def test_every_context_stays_inside_its_original_paragraph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            corpus_fixture(path)
            corpus = ArrayCorpus(path, "train", 2)
            x, y, p = corpus.batch(np.arange(corpus.windows))
            self.assertEqual(y.tolist(), [4, 5, 6, 7, 8, 3, 9, 8, 3])
            self.assertEqual(x[6].tolist(), [2, 2])
            self.assertEqual(x[5].tolist(), [7, 8])
            self.assertEqual(p.tolist(), [0] * 6 + [1] * 3)
            small = ArrayCorpus(path, "train", 2, fraction=.1)
            self.assertEqual(small.windows, 6)
            with self.assertRaises(IndexError):
                corpus.batch([corpus.windows])

    def test_tampered_file_or_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            corpus_fixture(path)
            with self.assertRaises(ValueError):
                ArrayCorpus(path, "train", 3)
            with (path / "train.ids.u16").open("ab") as stream:
                stream.write(b"xx")
            with self.assertRaises(ValueError):
                ArrayCorpus(path, "train", 2)

    def test_window_permutation_visits_every_target_before_repetition(self):
        for size in (1, 2, 13, 100):
            sampler = WindowSampler(size)
            result = sampler.take(size * 2)
            self.assertEqual(sorted(result[:size].tolist()), list(range(size)))
            self.assertEqual(sorted(result[size:].tolist()), list(range(size)))

    def test_sampling_resume_and_batch_chunking_preserve_order(self):
        original = WindowSampler(101, seed=9).take(300)
        sampler = WindowSampler(101, seed=9)
        first = sampler.take(113)
        resumed = WindowSampler(101, seed=9, consumed=sampler.consumed)
        self.assertEqual(np.concatenate([first, resumed.take(187)]).tolist(), original.tolist())

    def test_holdout_overlap_exclusion_and_train_only_counting(self):
        rows = [({"split": "train"}, ["공유/NNG", "학습/NNG", "학습/NNG"]),
                ({"split": "validation"}, ["공유/NNG", "검증/NNG"]),
                ({"split": "test"}, ["공유/NNG", "시험/NNG"])]
        with patch.object(prepare, "iter_records", side_effect=lambda *_: iter(rows)):
            heldout, overlap = prepare.holdout_keys(None, None)
            self.assertEqual(overlap, 1)
            stats = Counter()
            result = list(prepare.eligible_records(None, None, heldout, stats))
        self.assertEqual([lines for _, lines in result], [["학습/NNG"], ["공유/NNG", "검증/NNG"], ["시험/NNG"]])
        self.assertEqual(stats["train_heldout_overlap_removed_sentences"], 1)
        self.assertEqual(stats["train_tokenized_duplicates_removed_sentences"], 1)
        self.assertEqual(stats["test_validation_overlap_removed_sentences"], 1)

    def test_sentence_boundary_spaces_do_not_hide_a_heldout_overlap(self):
        rows = [({"split": "train"}, ["서울/NNP ./SF <sp>/WS", "부산/NNP ./SF"]),
                ({"split": "validation"}, ["서울/NNP ./SF"])]
        with patch.object(prepare, "iter_records", side_effect=lambda *_: iter(rows)):
            heldout, _ = prepare.holdout_keys(None, None)
            stats = Counter()
            result = list(prepare.eligible_records(None, None, heldout, stats))
        self.assertEqual([lines for _, lines in result], [["부산/NNP ./SF"], ["서울/NNP ./SF"]])
        self.assertEqual(stats["train_heldout_overlap_removed_sentences"], 1)


class ProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        (root / "parquet/data/kowiki").mkdir(parents=True)
        (root / "sentences_dedup").mkdir()
        (root / "provenance").mkdir()
        first = [{"id": "A", "title": "서울", "text": "# 서울\n서울(Seoul)은 수도이다. 다른 도시도 있다.\n서울은 수도이다."},
                 {"id": "B", "title": "수식", "text": "수식은 {\\displaystyle x} 이다."}]
        second = [{"id": "C", "title": "부산", "text": "서울은 수도이다.\n부산은 도시이다."}]
        for i, records in enumerate((first, second)):
            pq.write_table(pa.Table.from_pylist(records), root / f"parquet/data/kowiki/000_0000{i}.parquet")
        for i in range(32):
            text = "서울은 수도이다.\n다른 도시도 있다.\n\n" if i == 0 else "부산은 도시이다.\n\n" if i == 1 else ""
            (root / f"sentences_dedup/part_{i:02d}.txt").write_text(text)

    def test_reconstructs_article_after_filters_and_global_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            counts = prepare.rebuild_provenance(root, root, 42)
            self.assertEqual(counts, {"paragraphs": 2, "sentences": 3})
            with gzip.open(root / "provenance/part_00.jsonl.gz", "rt") as stream:
                row = json.loads(stream.readline())
            self.assertEqual((row["doc_id"], row["sentences"], row["source"]), ("A", 2, [0, 0, 0, 2]))
            with gzip.open(root / "provenance/part_01.jsonl.gz", "rt") as stream:
                self.assertEqual(json.loads(stream.readline())["doc_id"], "C")

    def test_altered_paragraph_is_not_silently_assigned_to_an_article(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / "sentences_dedup/part_00.txt").write_text("다른 도시도 있다.\n서울은 수도이다.\n\n")
            with self.assertRaisesRegex(ValueError, "alignment failed"):
                prepare.rebuild_provenance(root, root, 42)


class TrainingTests(unittest.TestCase):
    def test_budget_and_fraction_validation(self):
        from train_nplm import TrainingConfig
        TrainingConfig().validate()
        for settings in ({"fraction": .5}, {"budget_seconds": 8 * 3600 + 1},
                         {"checkpoints_seconds": (3, 2)}, {"temperature": 0},
                         {"checkpoints_seconds": (30000,)}, {"max_steps": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(TrainingConfig(), **settings).validate()

    def test_actual_tiny_training_saves_changed_weights_and_common_evaluation(self):
        import torch
        import train_nplm
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / "corpus"
            corpus.mkdir()
            corpus_fixture(corpus)
            output = root / "run"
            output.mkdir()
            config = train_nplm.TrainingConfig(context_size=2, embedding_dim=4, hidden_dim=8,
                                               batch_size=4, evaluation_windows=5, generation_tokens=2,
                                               budget_seconds=30, checkpoints_seconds=(.001,), max_steps=2,
                                               cpu_threads=1)
            with patch.object(train_nplm.torch, "set_num_interop_threads"):
                train_nplm._run_locked(corpus, output, config, False)
            initial, *rest = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            final = rest[-1]
            self.assertEqual(final["steps"], 2)
            self.assertEqual(final["target_prediction_events"], 8)
            self.assertEqual(final["evaluation_window_sha256"], initial["evaluation_window_sha256"])
            self.assertTrue(np.isfinite(final["validation"]["loss"]))
            self.assertEqual(final["validation"]["targets"], 5)
            saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["sampler_consumed"], 8)
            self.assertEqual(saved["state"]["status"], "step-limited-pilot-complete")
            self.assertTrue((output / "model-0.001s.pt").exists())
            self.assertEqual(json.loads((output / "run.json").read_text())["kind"], "ACTUAL_WIKI_TRAINING")
            self.assertNotEqual(initial["validation"]["loss"], final["validation"]["loss"])

    def test_stopped_training_resumes_optimizer_and_identical_next_windows(self):
        import torch
        import train_nplm
        original_take = WindowSampler.take
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / "corpus"
            corpus.mkdir()
            corpus_fixture(corpus)
            config = train_nplm.TrainingConfig(context_size=2, embedding_dim=4, hidden_dim=8,
                                               batch_size=4, evaluation_windows=5, generation_tokens=1,
                                               budget_seconds=30, checkpoints_seconds=(20,), cpu_threads=1)
            def stop_after(output, targets):
                def take(sampler, count):
                    result = original_take(sampler, count)
                    if sampler.consumed >= targets:
                        (output / "STOP").touch()
                    return result
                return take
            uninterrupted, interrupted = root / "control", root / "resumed"
            uninterrupted.mkdir()
            interrupted.mkdir()
            with patch.object(train_nplm.torch, "set_num_interop_threads"):
                with patch.object(WindowSampler, "take", stop_after(uninterrupted, 12)):
                    train_nplm._run_locked(corpus, uninterrupted, config, False)
                with patch.object(WindowSampler, "take", stop_after(interrupted, 8)):
                    train_nplm._run_locked(corpus, interrupted, config, False)
                (interrupted / "STOP").unlink()
                with patch.object(WindowSampler, "take", stop_after(interrupted, 12)):
                    train_nplm._run_locked(corpus, interrupted, config, True)
            left = torch.load(uninterrupted / "latest.pt", weights_only=True)
            right = torch.load(interrupted / "latest.pt", weights_only=True)
            self.assertEqual(left["sampler_consumed"], right["sampler_consumed"])
            self.assertEqual(left["state"]["steps"], 3)
            self.assertEqual(right["state"]["steps"], 3)
            self.assertTrue(all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"]))
            self.assertTrue(torch.equal(left["visited_paragraphs"], right["visited_paragraphs"]))


class ComparisonTests(unittest.TestCase):
    def test_missing_durations_are_not_filled_by_pilot_or_short_final(self):
        from build_training_comparison import collect
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = collect(root / "runs", root / "corpus", root / "comparison")
            self.assertEqual(result["status"], "in_progress")
            self.assertTrue(all(row["result"] is None for row in result["rows"]))
            self.assertIn("대기 중", (root / "comparison.txt").read_text())

    def metadata(self):
        from dataclasses import asdict
        from train_nplm import TrainingConfig
        return {"config": asdict(TrainingConfig()), "corpus_identity": "corpus", "vocabulary_identity": "vocab",
                "initialization_sha256": "initial", "evaluation_window_sha256": "evaluation", "scripts": {},
                "runtime": {}, "training_paragraphs": 10, "training_target_windows": 100}

    def test_incomparable_model_or_evaluation_conditions_are_rejected(self):
        from build_training_comparison import collect
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for fraction in (10, 25):
                directory = root / f"data-{fraction}pct"
                directory.mkdir()
                metadata = self.metadata()
                if fraction == 25:
                    metadata["evaluation_window_sha256"] = "different-targets"
                (directory / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "do not share"):
                collect(root, root / "corpus", root / "comparison")

    def test_checkpoint_cannot_be_claimed_before_nominal_duration(self):
        from build_training_comparison import collect
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "data-10pct"
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps(self.metadata()))
            metric = {"nominal_seconds": 10800, "actual_elapsed_seconds": 100, "steps": 1,
                      "epochs_equivalent": 0.1, "validation": {}}
            (directory / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
            with self.assertRaisesRegex(ValueError, "before its stated duration"):
                collect(root, root / "corpus", root / "comparison")

    def test_incomplete_verification_refuses_corpus_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reports = root / "reports"
            reports.mkdir()
            for name in ("step5-summary.json", "token-frequency-summary.json", "step6-numeric-summary.json",
                         "numeric-token-frequency-summary.json", "vocabulary-coverage.json"):
                (reports / name).write_text('{"status":"complete"}')
            (reports / "step5-verification.json").write_text('{"scope":"full","passed":false}')
            with patch("benchmark_nplm.assert_pipeline_idle", return_value={"idle": True}):
                with self.assertRaisesRegex(ValueError, "has not passed"):
                    prepare.validate_source(root)


class ContinuationTests(unittest.TestCase):
    def test_waits_for_every_completed_writer_report(self):
        from continue_experiment import READY_REPORTS, writers_complete
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "reports").mkdir()
            for name in READY_REPORTS:
                (root / "reports" / name).write_text('{"status":"complete"}')
            self.assertTrue(writers_complete(root))
            (root / "reports" / READY_REPORTS[-1]).write_text('{"status":"partial"}')
            self.assertFalse(writers_complete(root))

    def test_selects_matching_full_knee_and_successful_fastest_device(self):
        from continue_experiment import choose_configuration
        coverage = {"status": "complete", "scope": {"shard_count": 32},
                    "variants": [{"id": "numeric", "candidate_knee": {"k": 36000}}]}
        def row(device, rate, k=36000, status="complete"):
            return {"status": status, "config": {"vocab_size": k, "context_size": 8, "embedding_dim": 64,
                    "hidden_dim": 128, "batch_size": 128, "cpu_threads": 4, "device": device},
                    "timing": {"median_target_predictions_per_second": rate}}
        benchmark = {"status": "complete", "results": [row("cpu", 100), row("mps", 200), row("cpu", 1000, 16384)]}
        decision = choose_configuration(coverage, benchmark)
        self.assertEqual((decision["vocab_size"], decision["device"]), (36000, "mps"))
        self.assertFalse(decision["quality_optimum_proven"])
        coverage["status"] = "partial"
        with self.assertRaises(ValueError):
            choose_configuration(coverage, benchmark)

    def test_missing_successful_profile_is_not_guessed(self):
        from continue_experiment import choose_configuration
        coverage = {"status": "complete", "scope": {"shard_count": 32},
                    "variants": [{"id": "numeric", "candidate_knee": {"k": 36000}}]}
        with self.assertRaisesRegex(ValueError, "No successful"):
            choose_configuration(coverage, {"status": "complete", "results": []})


if __name__ == "__main__":
    unittest.main()
