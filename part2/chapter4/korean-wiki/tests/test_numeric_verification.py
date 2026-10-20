"""Tiny actual-file checks for the independent numeric verifier; no corpus scan."""

import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_numeric_tokens as verify


NAMES = ["part_00.txt", "part_01.txt"]


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{zlib.crc32(data):08x}"}


def independent_counts(data: bytes) -> dict:
    stats = dict.fromkeys((*verify.COUNT_KEYS, "lines", "blank_lines"), 0)
    inside = False
    for raw in data.splitlines(keepends=True):
        line = raw.decode().rstrip("\r\n")
        stats["lines"] += 1
        if not line:
            stats["blank_lines"] += 1
            inside = False
            continue
        tokens = line.split(" ")
        stats["sentences"] += 1
        stats["tokens"] += len(tokens)
        stats["space_tokens"] += tokens.count("<sp>/WS")
        if not inside:
            stats["paragraphs"] += 1
            inside = True
    return stats


def direct_fixture(root: Path, raw: bytes | None = None):
    if raw is None:
        raw = ("서울/NNP <sp>/WS <sp>/WS 2010/SN ./SF <sp>/WS\r\n"
               "2-1/W_SERIAL ./SF\r\n\r\n996,000/SN ./SF\r\n\r\n").encode()
    source, numeric = root / "part_00.txt", root / "numeric.txt"
    source.write_bytes(raw)
    numeric.write_bytes(b"".join(verify.reference_numeric_line(line)
                                  for line in raw.splitlines(keepends=True)))
    raw_counts, numeric_counts = independent_counts(raw), independent_counts(numeric.read_bytes())
    step5_record = {key: raw_counts[key] for key in verify.COUNT_KEYS}
    step5_record["output"] = identity(source)
    record = {**numeric_counts, "input_tokens": raw_counts["tokens"],
              "added_tokens": numeric_counts["tokens"] - raw_counts["tokens"],
              "input": identity(source), "output": identity(numeric)}
    return source, numeric, record, step5_record


def full_fixture(root: Path) -> None:
    """Exercise real producer manifests on a two-shard, three-sentence fixture."""
    import step6_count_tokens as frequencies
    import step6_split_numbers as numeric
    (root / "tokens").mkdir()
    (root / "reports").mkdir()
    texts = ["서울/NNP <sp>/WS 2010/SN ./SF <sp>/WS\r\n\r\n",
             "차/NNG <sp>/WS 2-1/W_SERIAL ./SF\r\n996,000/SN ./SF\r\n\r\n"]
    rows = {}
    for name, text in zip(NAMES, texts):
        path = root / "tokens" / name
        path.write_bytes(text.encode())
        counts = independent_counts(path.read_bytes())
        rows[name] = {key: counts[key] for key in verify.COUNT_KEYS}
        rows[name].update({"input": {}, "output": identity(path)})
    config = {"model_type": "cong-global", "kiwipiepy_version": "fixture-only"}
    (root / "reports/step5-state.json").write_text(json.dumps(
        {"config": config, "status": "complete", "completed": rows}))
    (root / "reports/step5-summary.json").write_text(json.dumps(
        {"status": "complete", "config": config, "shard_count": len(NAMES),
         **{key: sum(row[key] for row in rows.values()) for key in verify.COUNT_KEYS},
         "shards": [{"file": name, **row} for name, row in rows.items()]}))
    with redirect_stdout(io.StringIO()):
        frequencies.run(root, expected_shards=NAMES, preview=False)
        numeric.run(root, expected_shards=NAMES)


def kinds(result: dict) -> set:
    return {row["kind"] for row in result["errors"]}


class NumericReferenceTests(unittest.TestCase):
    def test_independent_rule_matches_supplied_original_helper(self):
        tree = ast.parse((ROOT / "processed-reference/step6a_build_vocab.py").read_text())
        names = {"_SPLIT_POS_TAGS", "_FW_TO_HW"}
        body = [node for node in tree.body
                if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                                        for target in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "_split_number_token")]
        namespace = {}
        exec(compile(ast.Module(body=body, type_ignores=[]), "original-reference", "exec"), namespace)
        for form in ("2010", "996,000", "2-1", "12/3", "０１２", "²", "١", "A_2", "-", ""):
            for tag in ("SN", "W_SERIAL", "SL", "NNG", "WS", "SN-extra"):
                token = f"{form}/{tag}"
                self.assertEqual(verify.reference_numeric_pieces(token),
                                 tuple(namespace["_split_number_token"](token)))

    def test_independent_sample_preserves_endings_and_ordinary_keys(self):
        for ending in (b"\r\n", b"\n", b""):
            raw = "차/NNG <sp>/WS ２０１０/SN <sp>/WS".encode() + ending
            expected = "차/NNG <sp>/WS 2/SN 0/SN 1/SN 0/SN <sp>/WS".encode() + ending
            self.assertEqual(verify.reference_numeric_line(raw), expected)
            self.assertEqual(verify.reference_numeric_line(ending), ending)
        for malformed in (b"/SN\n", b"12/SN  ./SF\n", b"12/SN\t./SF\n"):
            with self.assertRaises(ValueError):
                verify.reference_numeric_line(malformed)


class NumericShardVerifierTests(unittest.TestCase):
    def test_full_hashes_structures_ws_and_small_sample_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            source, numeric, record, step5 = direct_fixture(Path(directory))
            result = verify.verify_shard(source, numeric, record, step5)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["input"], identity(source))
            self.assertEqual(result["output"], identity(numeric))
            self.assertEqual(result["source_counts"]["space_tokens"], 3)
            self.assertEqual(result["numeric_counts"]["space_tokens"], 3)
            self.assertEqual(result["sampled_sentences"], 3)
            self.assertTrue(all(row["passed"] for row in result["samples"]))

    def test_truncated_numeric_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, numeric, record, step5 = direct_fixture(Path(directory))
            numeric.write_bytes(numeric.read_bytes().splitlines(keepends=True)[0])
            result = verify.verify_shard(source, numeric, record, step5)
            self.assertFalse(result["passed"])
            self.assertTrue({"line_count", "identity", "numeric_count"}.issubset(kinds(result)))

    def test_changed_raw_source_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, numeric, record, step5 = direct_fixture(Path(directory))
            source.write_bytes(source.read_bytes().replace(b"2010/SN", b"2011/SN"))
            result = verify.verify_shard(source, numeric, record, step5)
            self.assertFalse(result["passed"])
            self.assertIn("identity", kinds(result))
            self.assertIn("sample_transform", kinds(result))

    def test_blank_boundary_and_line_ending_changes_are_detected(self):
        for replacement, expected in ((b"./SF\r\n", "paragraph_boundary"), (b"\n", "line_ending")):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                source, numeric, record, step5 = direct_fixture(Path(directory))
                numeric.write_bytes(numeric.read_bytes().replace(b"\r\n\r\n", b"\r\n" + replacement, 1))
                result = verify.verify_shard(source, numeric, record, step5)
                self.assertFalse(result["passed"])
                self.assertIn(expected, kinds(result))

    def test_wrong_digit_is_detected_even_with_updated_record_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source, numeric, record, step5 = direct_fixture(Path(directory))
            numeric.write_bytes(numeric.read_bytes().replace(b"2/SN", b"8/SN", 1))
            record["output"] = identity(numeric)
            result = verify.verify_shard(source, numeric, record, step5)
            self.assertFalse(result["passed"])
            self.assertIn("sample_transform", kinds(result))
            self.assertNotIn("identity", kinds(result))

    def test_unsplit_numeric_sample_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, numeric, record, step5 = direct_fixture(Path(directory), b"12/SN ./SF\r\n\r\n")
            numeric.write_bytes(b"12/SN ./SF\r\n\r\n")
            record.update({**independent_counts(numeric.read_bytes()), "output": identity(numeric), "added_tokens": 0})
            result = verify.verify_shard(source, numeric, record, step5)
            self.assertFalse(result["passed"])
            self.assertIn("sample_transform", kinds(result))

    def test_reservoir_is_bounded_deterministic_and_samples_beyond_first_32(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = "".join(f"{index}/SN ./SF\r\n" for index in range(120)).encode() + b"\r\n"
            source, numeric, record, step5 = direct_fixture(Path(directory), raw)
            first = verify.verify_shard(source, numeric, record, step5, samples_per_shard=32)
            second = verify.verify_shard(source, numeric, record, step5, samples_per_shard=32)
            self.assertTrue(first["passed"], first["errors"])
            self.assertEqual(first["samples"], second["samples"])
            self.assertEqual(first["sampled_sentences"], 32)
            self.assertTrue(any(row["line"] > 32 for row in first["samples"]))


class NumericFullVerifierTests(unittest.TestCase):
    def test_completed_two_shard_fixture_checks_full_tables_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            before = {str(path.relative_to(root)): identity(path) for folder in ("tokens", "tokens_numeric")
                      for path in (root / folder).glob("*.txt")}
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=1)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["scope"], "full")
            self.assertEqual(result["shard_count"], 2)
            self.assertEqual(result["sampled_sentences"], 3)
            self.assertTrue(result["frequency_verification"]["checked"])
            self.assertIn("not exhaustive", result["transformation_scope"])
            self.assertEqual(result["source_totals"]["sentences"], result["numeric_totals"]["sentences"])
            for name, original in before.items():
                self.assertEqual(identity(root / name), original)
            self.assertTrue((root / "reports/step6-numeric-verification.json").is_file())

    def test_two_worker_streaming_verifier_matches_sequential_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=2, write_report=False)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual([row["file"] for row in result["shards"]], NAMES)

    def test_mismatched_unicode_runtime_cannot_claim_reference_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            path = root / "reports/step6-numeric-state.json"
            state = json.loads(path.read_text())
            state["config"]["unicode_version"] = "different-unicode-version"
            path.write_text(json.dumps(state))
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=1, write_report=False)
            self.assertFalse(result["passed"])
            self.assertIn("unicode_runtime", kinds(result))

    def test_missing_shard_and_incomplete_default_full_scope_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            default = verify.verify_outputs(root, workers=1, write_report=False)
            self.assertFalse(default["passed"])
            self.assertIn("incomplete_full_scope", kinds(default))
            (root / "tokens_numeric/part_01.txt").unlink()
            missing = verify.verify_outputs(root, expected_shards=NAMES, workers=1, write_report=False)
            self.assertFalse(missing["passed"])
            self.assertIn("missing_shard", kinds(missing))

    def test_partial_scope_uses_only_recorded_completed_shards_and_skips_final_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            path = root / "reports/step6-numeric-state.json"
            state = json.loads(path.read_text())
            state["status"] = "waiting_for_source"
            del state["completed"]["part_01.txt"]
            path.write_text(json.dumps(state))
            result = verify.verify_outputs(root, partial=True, expected_shards=NAMES, workers=1)
            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["scope"], "partial")
            self.assertEqual(result["shard_count"], 1)
            self.assertFalse(result["frequency_verification"]["checked"])
            self.assertTrue((root / "reports/step6-numeric-verification.partial.json").is_file())

    def test_step5_summary_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            path = root / "reports/step5-summary.json"
            summary = json.loads(path.read_text())
            summary["sentences"] += 1
            path.write_text(json.dumps(summary))
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=1, write_report=False)
            self.assertFalse(result["passed"])
            self.assertIn("step5_summary_count", kinds(result))

    def test_summary_and_raw_frequency_identity_mismatches_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            for name in ("step6-numeric-summary.json", "numeric-token-frequency-summary.json"):
                path = root / "reports" / name
                summary = json.loads(path.read_text())
                summary["input_tokens"] += 1
                summary["raw_frequency_input"]["sha256"] = "0" * 64
                path.write_text(json.dumps(summary))
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=1, write_report=False)
            self.assertFalse(result["passed"])
            self.assertIn("numeric_summary_source_count", kinds(result))
            self.assertIn("identity", kinds(result))

    def test_final_frequency_tampering_cannot_pass_even_with_unchanged_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_fixture(root)
            path = root / "token_frequencies.numeric.txt"
            path.write_bytes(path.read_bytes().replace("서울/NNP".encode(), "서을/NNP".encode()))
            result = verify.verify_outputs(root, expected_shards=NAMES, workers=1, write_report=False)
            self.assertFalse(result["passed"])
            self.assertIn("frequency_failed", kinds(result))

    def test_nonisolated_and_fullwidth_numeric_keys_in_frequency_table_are_rejected(self):
        import step6_count_tokens as frequencies
        for token in ("12/SN", "１/SN", "2/W_SERIAL"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "numeric.txt"
                frequencies.write_frequency_table(path, Counter({token: 3}), {"status": "complete"})
                result = verify.verify_frequency_table(path, identity(path), 3, 1, numeric=True)
                self.assertFalse(result["passed"])
                self.assertIn("frequency_row", kinds(result))


if __name__ == "__main__":
    unittest.main()
