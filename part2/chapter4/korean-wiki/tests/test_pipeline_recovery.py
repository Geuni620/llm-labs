"""Recovery reuses passed evidence, never partial checks or changed artifacts."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continue_experiment import reusable_tokenization_verification


def measure(path):
    data = path.read_bytes()
    return {"bytes": len(data), "mtime_ns": path.stat().st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{zlib.crc32(data):08x}"}


def fixture(root):
    (root / "reports").mkdir()
    (root / "tokens").mkdir()
    (root / "sentences_dedup").mkdir()
    state = {"status": "complete", "config": {"model_type": "cong-global"}, "inputs": {}, "completed": {}}
    shards, snapshot = [], {}
    for i in range(32):
        name = f"part_{i:02d}.txt"
        source, tokens = root / "sentences_dedup" / name, root / "tokens" / name
        source.write_text("문장이다.\n\n")
        tokens.write_text("문장/NNG 이/VCP 다/EF ./SF\n\n")
        incoming, outgoing = measure(source), measure(tokens)
        state["inputs"][name] = incoming
        state["completed"][name] = {"output": outgoing}
        shards.append({"file": name, "input": incoming, "output": outgoing})
        snapshot["sentences_dedup/" + name] = incoming
    reference = root / "reference.py"
    reference.write_text("reference")
    snapshot["reference.py"] = {"sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    report = {"passed": True, "scope": "full", "verified_shards": 32, "failures": 0,
              "input_identity": {"checked_files": 33, "mismatches": [], "sha256_unverified": [],
                                 "sha256_missing_from_snapshot": []}, "shards": shards,
              "counts": {"token_sentences": 32, "token_paragraphs": 32, "tokens": 128, "space_tokens": 0}}
    summary = {"status": "complete", "config": state["config"], "sentences": 32,
               "paragraphs": 32, "tokens": 128, "space_tokens": 0}
    for name, data in [("step5-state.json", state), ("step5-summary.json", summary),
                       ("step5-verification.json", report), ("step5-input-snapshot.json", snapshot)]:
        (root / "reports" / name).write_text(json.dumps(data))


class RecoveryTests(unittest.TestCase):
    def test_unchanged_fully_verified_inputs_are_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            self.assertEqual(reusable_tokenization_verification(root)["verified_shards"], 32)

    def test_partial_verification_is_never_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            path = root / "reports/step5-verification.json"
            report = json.loads(path.read_text()); report["scope"] = "partial"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                reusable_tokenization_verification(root)

    def test_changed_token_or_source_is_rejected_even_with_equal_size(self):
        for directory in ("tokens", "sentences_dedup"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture(root)
                path = root / directory / "part_00.txt"
                previous = path.stat()
                data = path.read_bytes()
                path.write_bytes(b"X" + data[1:])
                os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1000000))
                with self.assertRaisesRegex(ValueError, "file changed"):
                    reusable_tokenization_verification(root)

    def test_manifest_digest_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            path = root / "reports/step5-state.json"
            state = json.loads(path.read_text()); state["completed"]["part_00.txt"]["output"]["sha256"] = "0" * 64
            path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                reusable_tokenization_verification(root)

    def test_inconsistent_verification_totals_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            path = root / "reports/step5-summary.json"
            summary = json.loads(path.read_text()); summary["tokens"] += 1
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "totals differ"):
                reusable_tokenization_verification(root)

    def test_changed_reference_with_hash_only_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            (root / "reference.py").write_text("different")
            with self.assertRaisesRegex(ValueError, "reference changed"):
                reusable_tokenization_verification(root)


if __name__ == "__main__":
    unittest.main()
