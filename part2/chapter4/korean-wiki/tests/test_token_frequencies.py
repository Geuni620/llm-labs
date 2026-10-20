"""Frequency counting preserves every exact form/POS token and completed scope."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step6_count_tokens as step6
import verify_token_frequencies as verify


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{zlib.crc32(data):08x}"}


def fixture_manifest(root, completed=2):
    token_dir, reports = root / "tokens", root / "reports"
    token_dir.mkdir()
    reports.mkdir()
    config = {"model_type": "cong-global", "kiwipiepy_version": "0.24.0",
              "kiwipiepy_model_version": "0.24.0", "tokenizer_sha256": "test-tokenizer"}
    texts = ["차/NNG 차/VV <sp>/WS 1987/SN ./SF\r\n\r\n",
             "차/NNG 서울/NNP 의/JKG ./SF\r\n\r\n"]
    records, combined = {}, Counter()
    for index, text in enumerate(texts):
        name = f"part_{index:02d}.txt"
        path = token_dir / name
        path.write_bytes(text.encode())
        counts = Counter(text.splitlines()[0].split(" "))
        if index < completed:
            records[name] = {"sentences": 1, "paragraphs": 1, "tokens": sum(counts.values()),
                             "space_tokens": counts["<sp>/WS"], "input": {}, "output": identity(path)}
            combined.update(counts)
    state = {"config": config, "completed": records, "status": "complete" if completed == 2 else "running"}
    (reports / "step5-state.json").write_text(json.dumps(state), encoding="utf-8")
    if completed == 2:
        summary = {"status": "complete", "config": config, "shard_count": 2,
                   "sentences": 2, "paragraphs": 2, "tokens": sum(combined.values()),
                   "space_tokens": combined["<sp>/WS"], "shards": [{"file": name, **row} for name, row in records.items()]}
        (reports / "step5-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return records, combined


class FrequencyTests(unittest.TestCase):
    def test_count_keeps_whole_pos_keys_numbers_spaces_and_punctuation(self):
        text = ("차/NNG 차/VV <sp>/WS 1987/SN ./SF\r\n\r\n"
                "차/NNG A/SL a/SL New_York/NNP //SW ./SF\r\n\r\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part_00.txt"
            path.write_bytes(text.encode())
            counts, stats = step6.count_shard(path, identity(path))
            self.assertEqual(counts, Counter({"차/NNG": 2, "./SF": 2, "차/VV": 1,
                                             "<sp>/WS": 1, "1987/SN": 1, "A/SL": 1,
                                             "a/SL": 1, "New_York/NNP": 1, "//SW": 1}))
            self.assertEqual(stats["tokens"], 11)
            self.assertEqual(stats["sentences"], 2)
            self.assertEqual(stats["paragraphs"], 2)
            self.assertEqual(stats["space_tokens"], 1)

    def test_malformed_empty_tokens_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part_00.txt"
            for text in ["차/NNG  ./SF\n\n", "차/NNG\t./SF\n\n", "차/NNG /NNG\n\n"]:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        step6.count_shard(path)

    def test_changed_source_rejected_and_bad_cache_identity_recounted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, caches = root / "part_00.txt", root / "cache"
            path.write_text("차/NNG 차/VV ./SF\n\n", encoding="utf-8")
            expected = identity(path)
            first, stats, hit = step6.load_or_count_shard(path, expected, "model-v1", caches)
            self.assertFalse(hit)
            self.assertEqual(first, Counter({"차/NNG": 1, "차/VV": 1, "./SF": 1}))
            again, _, hit = step6.load_or_count_shard(path, expected, "model-v1", caches)
            self.assertTrue(hit)
            self.assertEqual(again, first)
            cache_path = next(caches.glob("*.json.gz"))
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                cache = json.load(handle)
            cache["source"]["sha256"] = "0" * 64
            cache["counts"] = {"손상/NNG": 3}
            with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                json.dump(cache, handle)
            recounted, _, hit = step6.load_or_count_shard(path, expected, "model-v1", caches)
            self.assertFalse(hit)
            self.assertEqual(recounted, first)
            path.write_text("해/NNG 해/VV ./SF\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                step6.load_or_count_shard(path, expected, "model-v1", caches)

    def test_fingerprint_change_invalidates_cached_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "part_00.txt"
            path.write_text("서울/NNP ./SF\n\n", encoding="utf-8")
            expected = identity(path)
            step6.load_or_count_shard(path, expected, "first-model", root / "cache")
            _, _, hit = step6.load_or_count_shard(path, expected, "second-model", root / "cache")
            self.assertFalse(hit)
        self.assertEqual(step6.tokenizer_fingerprint({"a": 1, "b": 2}), step6.tokenizer_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(step6.tokenizer_fingerprint({"model_type": "cong-global"}),
                            step6.tokenizer_fingerprint({"model_type": "knlm"}))

    def test_full_table_sort_ties_percentages_and_sum(self):
        counts = Counter({"차/NNG": 5, "a/SL": 2, "A/SL": 2, "의/JKG": 2, "<sp>/WS": 4})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frequencies.txt"
            result = step6.write_frequency_table(output, counts, {"status": "complete"})
            checked = verify.verify_table(output, counts, total_tokens=15)
            self.assertEqual(checked["errors"], [])
            self.assertEqual(checked["rows"], 5)
            self.assertEqual(checked["sum_counts"], 15)
            self.assertEqual(result["tokens"], 15)
            self.assertEqual(result["written_tokens"], 15)
            self.assertEqual(result["rows"], 5)
            data_rows = [line for line in output.read_text().splitlines() if line and not line.startswith("#")][1:]
            self.assertEqual([line.split("\t")[1] for line in data_rows],
                             [key for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))])

    def test_preview_is_explicitly_partial_and_final_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = Counter({"차/NNG": 3, "./SF": 2, "<sp>/WS": 1})
            preview = root / "partial.txt"
            result = step6.write_frequency_table(preview, counts, {"status": "partial"}, limit=2)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["tokens"], 6)
            self.assertEqual(result["written_tokens"], 5)
            self.assertIn("partial", preview.read_text())
            final = root / "final.txt"
            final.write_text("사용자가 가진 기존 파일", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                step6.write_frequency_table(final, counts, {"status": "complete"})
            self.assertEqual(final.read_text(), "사용자가 가진 기존 파일")

    def test_only_completed_manifest_shards_can_be_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_manifest(root, completed=1)
            with self.assertRaises(ValueError):
                step6.run(root=root, watch=False, expected_shards=["part_00.txt", "part_01.txt"])
            self.assertFalse((root / "token_frequencies.txt").exists())

    def test_small_complete_run_uses_all_manifest_tokens_without_vocabulary_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, counts = fixture_manifest(root)
            summary = step6.run(root=root, watch=False, expected_shards=["part_00.txt", "part_01.txt"])
            self.assertEqual(summary["shard_count"], 2)
            self.assertEqual(summary["tokens"], 9)
            self.assertEqual(summary["unique_types"], len(counts))
            self.assertTrue(summary["frequency_sum_matches_step5"])
            self.assertTrue(summary["space_token_count_matches_step5"])
            checked = verify.verify_table(root / "token_frequencies.txt", counts, total_tokens=9)
            self.assertEqual(checked["errors"], [])
            self.assertEqual(checked["rows"], len(counts))
            passed_step5 = {"scope": "full", "passed": True,
                            "shards": [{"file": name, "output": record["output"]} for name, record in records.items()]}
            (root / "reports/step5-verification.json").write_text(json.dumps(passed_step5), encoding="utf-8")
            verification = verify.verify_outputs(root, expected_shards=["part_00.txt", "part_01.txt"])
            self.assertTrue(verification["passed"], verification["errors"])
            self.assertEqual(verification["unique_types"], 7)
            cache_path = Path(summary["source_shards"][0]["cache"]["path"])
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                cache = json.load(handle)
            cache["counts"]["차/NNG"] -= 1
            cache["counts"]["조작/NNG"] = 1
            with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                json.dump(cache, handle)
            corrupted = verify.verify_outputs(root, expected_shards=["part_00.txt", "part_01.txt"])
            self.assertFalse(corrupted["passed"])
            self.assertIn("cache_identity", {value["kind"] for value in corrupted["errors"]})


class TableVerifierTests(unittest.TestCase):
    def test_real_corpus_binary_rounding_boundaries(self):
        # Exact cumulative counts from the four failed full-corpus rows. Use
        # two-row tables so the count/sum/rank checks remain independently tested.
        total = 292363011
        cases = [(290103189, "99.227049280"), (290867265, "99.488394242"),
                 (292119121, "99.916579735"), (292277734, "99.970831809")]
        for cumulative, observed in cases:
            with self.subTest(cumulative=cumulative), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(f"{cumulative / total * 100:.9f}", observed)
                remaining = total - cumulative
                text = ("rank\ttoken\tcount\tpercent\tcumulative_percent\n"
                        f"1\tA/SL\t{cumulative}\t{observed}\t{observed}\n"
                        f"2\tB/SL\t{remaining}\t{remaining / total * 100:.9f}\t100.000000000\n")
                path = Path(directory) / "rounding.txt"
                path.write_text(text)
                result = verify.verify_table(path, Counter({"A/SL": cumulative, "B/SL": remaining}), total)
                self.assertEqual(result["errors"], [])

    def test_float_allowance_does_not_hide_a_wrong_display_digit(self):
        self.assertFalse(verify.valid_rounded_percent("50.000000001", 1, 2))
        self.assertFalse(verify.valid_rounded_percent("99.227049282", 290103189, 292363011))
        self.assertFalse(verify.valid_rounded_percent("100.000000001", 1, 1))
        self.assertTrue(verify.valid_rounded_percent("0.000000000", 0, 2))

    def test_percentage_format_bounds_and_nonfinite_values(self):
        for value in ("NaN", "Infinity", "1.000000e2", "50.00000000", "-0.000000000", "100.0000000000"):
            with self.subTest(value=value):
                self.assertFalse(verify.valid_rounded_percent(value, 1, 2))
        self.assertFalse(verify.valid_rounded_percent("100.000000000", 2, 1))
        self.assertFalse(verify.valid_rounded_percent("0.000000000", 0, 0))

    def test_bad_rank_sort_ties_and_ratios_are_detected(self):
        text = ("rank\ttoken\tcount\tpercent\tcumulative_percent\n"
                "1\t차/NNG\t2\t50.000000000\t50.000000000\n"
                "3\tA/SL\t2\t49.000000000\t99.000000000\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(text, encoding="utf-8")
            result = verify.verify_table(path, Counter({"차/NNG": 2, "A/SL": 2}), total_tokens=4)
            kinds = {value["kind"] for value in result["errors"]}
            self.assertTrue({"rank", "sort", "percent", "cumulative_percent"}.issubset(kinds))

    def test_duplicate_or_missing_types_cannot_pass_with_equal_sum(self):
        text = ("rank\ttoken\tcount\tpercent\tcumulative_percent\n"
                "1\tA/SL\t2\t50.000000000\t50.000000000\n"
                "2\tA/SL\t2\t50.000000000\t100.000000000\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(text, encoding="utf-8")
            result = verify.verify_table(path, Counter({"A/SL": 2, "차/NNG": 2}), total_tokens=4)
            kinds = {value["kind"] for value in result["errors"]}
            self.assertIn("duplicate_token", kinds)
            self.assertIn("missing_types", kinds)


if __name__ == "__main__":
    unittest.main()
