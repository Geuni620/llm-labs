"""Focused Kiwi integration checks plus independent persisted-data validation."""

import hashlib
import importlib.util
import io
from itertools import islice
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from kiwipiepy import Kiwi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step5_tokenize as step5
import verify_tokenization as verify


def load_reference():
    spec = importlib.util.spec_from_file_location("lecture_step5a", ROOT / "processed-reference/step5a_tokenize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = load_reference()


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kiwi = Kiwi(num_workers=1, model_type="cong-global")

    def test_synthetic_outputs_match_lecture(self):
        sentences = [
            "서울은 대한민국의 수도이다.",
            "1987년에 태어났다.",
            "제임스 D. Watson은 DNA를 연구했다.",
            "경계에 공백이 있다. ",
            "나는 도로다.",
            "차를 타고 서울에 갔다.",
            "따뜻한 차를 마셨다.",
            "공을 차면 멀리 날아간다.",
            "3.14는 소수이고 1987은 정수다.",
            "서울 (약 100만명)에 관한 설명이다.",
        ]
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                self.assertEqual(step5.tokenize_sentence(self.kiwi, sentence),
                                 REFERENCE.tokenize_sentence(self.kiwi, sentence))

    def test_numbers_and_punctuation_are_not_discarded(self):
        result = step5.tokenize_sentence(self.kiwi, "1987년에 태어났다.")
        self.assertIn("1987/SN", result.split(" "))
        self.assertIn("./SF", result.split(" "))

    def test_noun_homonyms_remain_same_pos_but_verb_differs(self):
        car = step5.tokenize_sentence(self.kiwi, "차를 타고 서울에 갔다.").split(" ")
        tea = step5.tokenize_sentence(self.kiwi, "따뜻한 차를 마셨다.").split(" ")
        kick = step5.tokenize_sentence(self.kiwi, "공을 차면 멀리 날아간다.").split(" ")
        self.assertIn("차/NNG", car)
        self.assertIn("차/NNG", tea)
        self.assertIn("차/VV", kick)

    def test_overlapping_offsets_do_not_create_false_space(self):
        analysis = [SimpleNamespace(form="가나", tag="NNG", start=0, len=2),
                    SimpleNamespace(form="나다", tag="NNG", start=1, len=2),
                    SimpleNamespace(form="X", tag="SL", start=4, len=1)]
        self.assertEqual(step5.format_tokens("가나다 X ", analysis),
                         "가나/NNG 나다/NNG <sp>/WS X/SL <sp>/WS")

    def test_multiword_forms_escape_spaces_and_empty_analysis_fails(self):
        analysis = [SimpleNamespace(form="New York", tag="NNP", start=0, len=8)]
        self.assertEqual(step5.format_tokens("New York", analysis), "New_York/NNP")
        with self.assertRaises(ValueError):
            step5.format_tokens("분명히 문장이 있다.", [])

    def test_bulk_and_serial_match_for_100_actual_sentences(self):
        with (ROOT / "sentences_dedup/part_00.txt").open(encoding="utf-8") as handle:
            sentences = list(islice((line.rstrip("\n") for line in handle if line.strip()), 100))
        self.assertEqual(len(sentences), 100)
        inputs = [sentence + (" " if index % 10 != 9 else "") for index, sentence in enumerate(sentences)]
        actual = [step5.format_tokens(sentence, analysis)
                  for sentence, analysis in zip(inputs, self.kiwi.tokenize(inputs), strict=True)]
        expected = [REFERENCE.tokenize_sentence(self.kiwi, sentence) for sentence in inputs]
        self.assertEqual(actual, expected)

    def test_file_batches_preserve_paragraphs_and_unterminated_eof(self):
        source_text = ("서울은 대한민국의 수도이다.\n두 번째 문장이다.\n1987년에 태어났다.\n\n"
                       "제임스 D. Watson은 DNA를 연구했다.\n\n마지막 문장이다.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, expected = root / "source.txt", root / "tokens.txt", root / "reference.txt"
            source.write_bytes(source_text.replace("\n", "\r\n").encode())
            before = source.read_bytes()
            self.assertEqual([len(paragraph) for paragraph in step5.iter_paragraphs(source)], [3, 1, 1])
            counts = step5.process_file(self.kiwi, source, output, batch_size=2)
            ref_counts = REFERENCE._process_one_file(self.kiwi, str(source), str(expected))
            self.assertEqual((counts["sentences"], counts["tokens"]), ref_counts)
            self.assertEqual(counts["paragraphs"], 3)
            self.assertEqual(output.read_text(), expected.read_text())
            self.assertEqual(source.read_bytes(), before)
            data = output.read_bytes()
            self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
            self.assertEqual(counts["output"]["sha256"], hashlib.sha256(data).hexdigest())
            lines = output.read_text().splitlines()
            self.assertTrue(lines[0].endswith("<sp>/WS"))
            self.assertTrue(lines[1].endswith("<sp>/WS"))
            self.assertFalse(lines[2].endswith("<sp>/WS"))
            self.assertEqual(sum(bool(line) for line in lines), 5)
            with self.assertRaises(FileExistsError):
                step5.process_file(self.kiwi, source, output, batch_size=2)

    def test_missing_bulk_analysis_does_not_publish_partial_file(self):
        class IncompleteAnalyzer:
            def tokenize(self, _):
                return iter(())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.txt", root / "tokens.txt"
            source.write_text("문장이 있다.\n\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                step5.process_file(IncompleteAnalyzer(), source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_independent_round_trip_matches_reference_restoration(self):
        for sentence in ("서울은 대한민국의 수도이다.", "1987년에 태어났다.", "나는 도로다."):
            token_line = step5.tokenize_sentence(self.kiwi, sentence)
            self.assertEqual(verify.restore_surface(token_line, self.kiwi), step5.detokenize(token_line, self.kiwi).rstrip())
        samples = [{"original": "서울은 대한민국의 수도이다.",
                    "tokens": step5.tokenize_sentence(self.kiwi, "서울은 대한민국의 수도이다.")}]
        result = verify.verify_round_trips(samples, self.kiwi)
        self.assertEqual(result["sentences"], 1)
        self.assertEqual(result["exact"] + result["mismatch"] + result["join_errors"], 1)
        self.assertEqual(result["join_errors"], 0)


class IndependentVerifierTests(unittest.TestCase):
    def test_canonical_alignment_and_space_counts(self):
        source = "서울이다.\r\n다음이다.\r\n\r\n끝이다.\r\n\r\n"
        tokens = "서울/NNP 이/VCP 다/EF ./SF <sp>/WS\r\n다음/NNG 이/VCP 다/EF ./SF\r\n\r\n끝/NNG 이/VCP 다/EF ./SF\r\n\r\n"
        result = verify.verify_streams(io.StringIO(source), io.StringIO(tokens), require_crlf=True)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["counts"]["source_sentences"], 3)
        self.assertEqual(result["counts"]["token_paragraphs"], 2)
        self.assertEqual(result["counts"]["tokens"], 13)
        self.assertEqual(result["counts"]["space_tokens"], 1)
        self.assertEqual(result["counts"]["trailing_space_tokens"], 1)

    def test_truncated_and_extra_lines_are_not_silently_ignored(self):
        for source, tokens in [("첫째다.\n둘째다.\n\n", "첫째/NNG 다/EF ./SF <sp>/WS\n"),
                               ("첫째다.\n\n", "첫째/NNG 다/EF ./SF\n\n남/NNG 다/EF ./SF\n\n")]:
            with self.subTest(source=source):
                result = verify.verify_streams(io.StringIO(source), io.StringIO(tokens))
                self.assertGreater(result["errors"].get("line_count_mismatches", 0), 0)

    def test_empty_token_line_and_shifted_paragraph_are_errors(self):
        result = verify.verify_streams(io.StringIO("서울이다.\n\n"), io.StringIO("\n서울/NNP 이/VCP 다/EF ./SF\n"))
        self.assertEqual(result["errors"]["blank_line_mismatches"], 2)

    def test_invalid_tokens_and_spaces_are_rejected_without_losing_slashes(self):
        for line in ("서울/", "/NNG", "서울", "서울/NNG  ./SF", "<sp>/NNG", "공백/WS", "<sp>/WS 서울/NNP"):
            with self.subTest(line=line):
                _, bad = verify.parse_tokens(line)
                self.assertTrue(bad)
        self.assertEqual(verify.parse_tokens("//SW V/VV-R 서울/NNP <sp>/WS")[1], [])

    def test_missing_trailing_paragraph_space_is_detected(self):
        result = verify.verify_streams(io.StringIO("처음이다.\n둘째다.\n\n"),
                                       io.StringIO("처음/NNG 이/VCP 다/EF ./SF\n둘째/NNG 다/EF ./SF\n\n"))
        self.assertEqual(result["errors"]["trailing_space_mismatches"], 1)

    def test_round_trip_sampling_is_deterministic_and_bounded(self):
        source = "".join(f"문장{index}다.\n\n" for index in range(100))
        tokens = "".join(f"문장/NNG {index}/SN 다/EF ./SF\n\n" for index in range(100))
        first = verify.verify_streams(io.StringIO(source), io.StringIO(tokens), sample_size=8, seed="fixed")
        second = verify.verify_streams(io.StringIO(source), io.StringIO(tokens), sample_size=8, seed="fixed")
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(len(first["samples"]), 8)
        self.assertTrue(any(sample["line"] > 100 for sample in first["samples"]))

    def test_source_identity_detects_same_size_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "sentences_dedup"
            folder.mkdir()
            path = folder / "part_00.txt"
            path.write_bytes(b"original\n")
            stat = path.stat()
            snapshot = {"sentences_dedup/part_00.txt": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            measure = {}
            self.assertEqual(list(verify.measured_lines(path, measure)), ["original\n"])
            self.assertEqual(verify.check_identity(root, snapshot, {"sentences_dedup/part_00.txt": measure})["mismatches"], [])
            path.write_bytes(b"mutated!\n")
            changed = {}
            list(verify.measured_lines(path, changed))
            result = verify.check_identity(root, snapshot, {"sentences_dedup/part_00.txt": changed})
            self.assertIn("sha256", [value["field"] for value in result["mismatches"]])

    def test_writer_summary_counts_and_digests_are_independently_compared(self):
        actual = {"file": "part_00.txt", "counts": {"token_sentences": 2, "token_paragraphs": 1,
                                                      "tokens": 8, "space_tokens": 1},
                  "input": {"bytes": 10, "sha256": "source", "crc32": "12345678"},
                  "output": {"bytes": 30, "sha256": "output", "crc32": "abcdef12"}}
        summary = {"shard_count": 1, "sentences": 2, "paragraphs": 1, "tokens": 8, "space_tokens": 1,
                   "shards": [{"file": "part_00.txt", "sentences": 2, "paragraphs": 1, "tokens": 8,
                               "space_tokens": 1, "input": actual["input"].copy(), "output": actual["output"].copy()}]}
        self.assertEqual(verify.compare_summary(summary, [actual], partial=False), [])
        summary["tokens"] = 9
        summary["shards"][0]["output"]["sha256"] = "changed"
        fields = {value["field"] for value in verify.compare_summary(summary, [actual], partial=False)}
        self.assertEqual(fields, {"tokens", "output.sha256"})

    def test_serial_and_parallel_shard_scans_match_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sentences_dedup").mkdir()
            (root / "tokens").mkdir()
            names = ["part_01.txt", "part_00.txt"]
            for name in names:
                (root / "sentences_dedup" / name).write_bytes("처음이다.\r\n다음이다.\r\n\r\n끝이다.\r\n\r\n".encode())
                (root / "tokens" / name).write_bytes(("처음/NNG 이/VCP 다/EF ./SF <sp>/WS\r\n"
                                                     "다음/NNG 이/VCP 다/EF ./SF\r\n\r\n"
                                                     "끝/NNG 이/VCP 다/EF ./SF\r\n\r\n").encode())
            serial = list(verify.verify_shards(root, names, sample_size=2, workers=1))
            parallel = list(verify.verify_shards(root, names, sample_size=2, workers=2))
            self.assertEqual(serial, parallel)
            self.assertEqual([value["file"] for value in parallel], names)
            self.assertEqual(sum(value["error_count"] for value in parallel), 0)
            self.assertTrue(all(len(value["samples"]) == 2 for value in parallel))


if __name__ == "__main__":
    unittest.main()
