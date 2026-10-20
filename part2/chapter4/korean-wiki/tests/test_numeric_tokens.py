"""Exact numeric rules, immutable source files and resumable derived frequency data."""

import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step6_count_tokens as frequencies
import step6_split_numbers as numeric


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{zlib.crc32(data):08x}"}


def fixture(root, completed=2):
    tokens, reports = root / "tokens", root / "reports"
    tokens.mkdir()
    reports.mkdir()
    config = {"model_type": "cong-global", "kiwipiepy_version": "0.24.0",
              "kiwipiepy_model_version": "0.24.0", "tokenizer_sha256": "fixture-only"}
    texts = ["서울/NNP <sp>/WS 2010/SN ./SF <sp>/WS\r\n\r\n",
             "차/NNG <sp>/WS 2-1/W_SERIAL ./SF\r\n996,000/SN ./SF\r\n\r\n"]
    rows, counts = {}, Counter()
    for index, text in enumerate(texts):
        name = f"part_{index:02d}.txt"
        path = tokens / name
        path.write_bytes(text.encode())
        table, stats = frequencies.count_shard(path)
        if index < completed:
            rows[name] = {key: stats[key] for key in frequencies.COUNT_KEYS}
            rows[name].update({"input": {}, "output": identity(path)})
            counts.update(table)
    status = "complete" if completed == 2 else "running"
    (reports / "step5-state.json").write_text(json.dumps({"config": config, "status": status, "completed": rows}))
    if completed == 2:
        summary = {"status": "complete", "config": config, "shard_count": 2,
                   **{key: sum(row[key] for row in rows.values()) for key in frequencies.COUNT_KEYS},
                   "shards": [{"file": name, **row} for name, row in rows.items()]}
        (reports / "step5-summary.json").write_text(json.dumps(summary))
    return rows, counts


class NumericRulesTests(unittest.TestCase):
    def test_reference_numeric_examples_and_numeric_edge_cases(self):
        examples = {
            "2010/SN": ("2/SN", "0/SN", "1/SN", "0/SN"),
            "2-1/W_SERIAL": ("2/SN", "-/SW", "1/SN"),
            "996,000/SN": ("9/SN", "9/SN", "6/SN", ",/SW", "0/SN", "0/SN", "0/SN"),
            "-12.50/SN": ("-/SW", "1/SN", "2/SN", "./SW", "5/SN", "0/SN"),
            "０１２/SN": ("0/SN", "1/SN", "2/SN"),
            "0/SN": ("0/SN",), "０/W_SERIAL": ("0/SN",),
            "-/W_SERIAL": ("-/SW",), "12/3/W_SERIAL": ("1/SN", "2/SN", "//SW", "3/SN"),
            "A_2/W_SERIAL": ("A/SW", "_/SW", "2/SN"),
            "²/SN": ("²/SN",), "١/SN": ("١/SN",),
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(numeric.split_numeric_token(source), expected)

    def test_ordinary_keys_remain_exact_and_are_not_cached(self):
        numeric._cached_numeric_expansion.cache_clear()
        for token in ["서울/NNP", "차/NNG", "차/VV", "New_York/NNP", "D./SL", "<sp>/WS",
                      "2010/SL", "//SW", "./SF", "/SN", "SN", "2010/SN-extra"]:
            self.assertEqual(numeric.split_numeric_token(token), (token,))
        self.assertEqual(numeric._cached_numeric_expansion.cache_info().currsize, 0)
        numeric.split_numeric_token("2010/SN")
        numeric.split_numeric_token("2010/SN")
        info = numeric._cached_numeric_expansion.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))

    def test_matches_supplied_reference_helper_for_every_sample_key(self):
        tree = ast.parse((ROOT / "processed-reference/step6a_build_vocab.py").read_text())
        names = {"_SPLIT_POS_TAGS", "_FW_TO_HW"}
        body = [node for node in tree.body
                if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                                        for target in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "_split_number_token")]
        namespace = {}
        exec(compile(ast.Module(body=body, type_ignores=[]), "reference-helper", "exec"), namespace)
        reference = namespace["_split_number_token"]
        for surface in ["0", "2010", "996,000", "2-1", "-0.25", "１２３", "12/3", "a_2", "²", "١", ""]:
            for tag in ["SN", "W_SERIAL", "NNG", "SL", "WS"]:
                token = f"{surface}/{tag}"
                self.assertEqual(numeric.split_numeric_token(token), tuple(reference(token)))

    def test_line_endings_blank_lines_and_trailing_space_tokens_are_preserved(self):
        for ending in ["\r\n", "\n", ""]:
            self.assertEqual(numeric.transform_token_line("2010/SN <sp>/WS" + ending),
                             "2/SN 0/SN 1/SN 0/SN <sp>/WS" + ending)
            self.assertEqual(numeric.transform_token_line(ending), ending)
        for malformed in ["차/NNG  ./SF", "차/NNG\t./SF", " 차/NNG"]:
            with self.assertRaises(ValueError):
                numeric.transform_token_line(malformed)

    def test_weighted_frequency_counts_repeated_digits_and_existing_separators(self):
        counts = Counter({"111/SN": 3, "1/SN": 2, "2-1/W_SERIAL": 4, "-/SW": 1, "<sp>/WS": 10})
        result = numeric.transform_frequency_counts(counts)
        self.assertEqual(result, Counter({"1/SN": 15, "<sp>/WS": 10, "-/SW": 5, "2/SN": 4}))
        self.assertEqual(sum(result.values()), 34)
        with self.assertRaises(ValueError):
            numeric.transform_frequency_counts(Counter({"0/SN": 0}))


class NumericShardTests(unittest.TestCase):
    def test_actual_file_boundaries_counts_hashes_and_raw_bytes_are_preserved(self):
        raw = ("2010/SN <sp>/WS 서울/NNP ./SF <sp>/WS\r\n"
               "2-1/W_SERIAL ./SF\r\n\r\n996,000/SN ./SF\r\n\r\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.txt", root / "output.txt"
            source.write_bytes(raw)
            before = identity(source)
            _, source_stats = frequencies.count_shard(source)
            result = numeric.transform_shard(source, output, before, source_stats)
            self.assertEqual(source.read_bytes(), raw)
            expected = b"".join(numeric.transform_token_line(line.decode()).encode() for line in raw.splitlines(keepends=True))
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(result["input"], before)
            self.assertEqual(result["output"], identity(output))
            self.assertEqual((result["sentences"], result["paragraphs"], result["blank_lines"], result["space_tokens"]), (3, 2, 2, 2))
            expanded, stats = frequencies.count_shard(output)
            self.assertEqual(result["tokens"], sum(expanded.values()))
            self.assertEqual(result["numeric_source_types"], 3)
            self.assertEqual(result["numeric_source_occurrences"], 3)
            self.assertEqual(result["added_tokens"], 11)
            self.assertEqual(stats["space_tokens"], source_stats["space_tokens"])

    def test_wrong_source_identity_or_counts_never_publishes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.txt", root / "output.txt"
            source.write_bytes(b"2010/SN\r\n\r\n")
            wrong = {**identity(source), "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                numeric.transform_shard(source, output, wrong)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".numeric-*.partial")), [])
            counts = {"tokens": 1, "sentences": 2, "paragraphs": 1, "space_tokens": 0}
            with self.assertRaises(ValueError):
                numeric.transform_shard(source, output, identity(source), counts)
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.txt", root / "output.txt"
            source.write_bytes(b"2010/SN\r\n")
            output.write_text("user data")
            with self.assertRaises(FileExistsError):
                numeric.transform_shard(source, output, identity(source))
            self.assertEqual(output.read_text(), "user data")


class NumericPipelineTests(unittest.TestCase):
    NAMES = ["part_00.txt", "part_01.txt"]

    def test_complete_pipeline_matches_independent_materialized_counts_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, raw_counts = fixture(root)
            originals = {name: (root / "tokens" / name).read_bytes() for name in self.NAMES}
            frequencies.run(root, expected_shards=self.NAMES, preview=False)
            result = numeric.run(root, expected_shards=self.NAMES)
            expected, sentence_count, paragraph_count = Counter(), 0, 0
            for name in self.NAMES:
                counts, stats = frequencies.count_shard(root / "tokens_numeric" / name)
                expected.update(counts)
                sentence_count += stats["sentences"]
                paragraph_count += stats["paragraphs"]
                self.assertEqual((root / "tokens" / name).read_bytes(), originals[name])
            self.assertEqual(expected, numeric.transform_frequency_counts(raw_counts))
            self.assertEqual(result["tokens"], sum(expected.values()))
            self.assertEqual(result["unique_types"], len(expected))
            self.assertEqual((result["sentences"], result["paragraphs"]), (sentence_count, paragraph_count))
            self.assertEqual(result["numeric_source_types"], 3)
            self.assertTrue(result["frequency_sum_matches_materialized_outputs"])
            self.assertEqual(numeric.run(root, resume=True, expected_shards=self.NAMES), result)
            with self.assertRaises(FileExistsError):
                numeric.run(root, expected_shards=self.NAMES)
            text = (root / "token_frequencies.numeric.txt").read_text()
            parsed = Counter({columns[1]: int(columns[2]) for line in text.splitlines()
                              if line and not line.startswith("#") and not line.startswith("rank\t")
                              for columns in [line.split("\t")]})
            self.assertEqual(parsed, expected)

    def test_resume_rejects_modified_completed_output_and_configuration(self):
        for changed in ["output", "config"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)
                frequencies.run(root, expected_shards=self.NAMES, preview=False)
                numeric.run(root, expected_shards=self.NAMES)
                if changed == "output":
                    with (root / "tokens_numeric/part_00.txt").open("ab") as stream:
                        stream.write(b"\r\n")
                else:
                    path = root / "reports/step6-numeric-state.json"
                    state = json.loads(path.read_text())
                    state["config"]["unicode_version"] = "different"
                    path.write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    numeric.run(root, resume=True, expected_shards=self.NAMES)

    def test_incomplete_source_cannot_create_a_complete_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, completed=1)
            with self.assertRaises(ValueError):
                numeric.run(root, expected_shards=self.NAMES)
            self.assertFalse((root / "tokens_numeric").exists())
            self.assertFalse((root / "token_frequencies.numeric.txt").exists())

    def test_watch_consumes_only_manifest_completed_shards_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, completed=1)
            # Stop the watcher at its first wait, simulating an interruption.
            with patch.object(numeric.time, "sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    numeric.run(root, watch=True, expected_shards=self.NAMES)
            state = json.loads((root / "reports/step6-numeric-state.json").read_text())
            self.assertEqual(list(state["completed"]), ["part_00.txt"])
            self.assertTrue((root / "tokens_numeric/part_00.txt").exists())
            self.assertFalse((root / "tokens_numeric/part_01.txt").exists())
            self.assertFalse((root / "token_frequencies.numeric.txt").exists())
            # Complete the untouched second source and publish the raw dictionary.
            path = root / "tokens/part_01.txt"
            _, stats = frequencies.count_shard(path)
            step5_path = root / "reports/step5-state.json"
            step5 = json.loads(step5_path.read_text())
            step5["completed"]["part_01.txt"] = {**{key: stats[key] for key in frequencies.COUNT_KEYS},
                                                   "input": {}, "output": identity(path)}
            step5["status"] = "complete"
            step5_path.write_text(json.dumps(step5))
            summary = {"status": "complete", "config": step5["config"], "shard_count": 2,
                       **{key: sum(row[key] for row in step5["completed"].values()) for key in frequencies.COUNT_KEYS},
                       "shards": [{"file": name, **row} for name, row in step5["completed"].items()]}
            (root / "reports/step5-summary.json").write_text(json.dumps(summary))
            frequencies.run(root, expected_shards=self.NAMES, preview=False)
            result = numeric.run(root, resume=True, expected_shards=self.NAMES)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["shard_count"], 2)

    def test_corrupt_raw_dictionary_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            frequencies.run(root, expected_shards=self.NAMES, preview=False)
            table = root / "token_frequencies.txt"
            with table.open("ab") as stream:
                stream.write(b"# changed\n")
            with self.assertRaises(ValueError):
                numeric.run(root, expected_shards=self.NAMES)
            self.assertFalse((root / "token_frequencies.numeric.txt").exists())


if __name__ == "__main__":
    unittest.main()
