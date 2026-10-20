"""Compatibility and corruption checks for NumPy loading; no model is run."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npy_corpus import NpyArrayCorpus
from training_corpus import ArrayCorpus, PARAGRAPH_DTYPE, RESERVED, WindowSampler, identity


class NpyCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "corpus"
        self.directory.mkdir()
        self.report = Path(self.temporary.name) / "export.json"
        tokens = RESERVED + ["서울/NNP", "은/JX", "<sp>/WS", "수도/NNG", "./SF", "1/SN"]
        first, second = [2, 2, 4, 5, 6, 7, 8, 3], [2, 2, 9, 8, 3]
        rows = [(0, len(first), 0, 1, 1), (len(first), len(second), 2 ** 31, 1, 2)]
        outputs = {}
        for split in ("train", "validation", "test"):
            np.asarray(first + second, dtype="<u2").tofile(self.directory / f"{split}.ids.u16")
            np.save(self.directory / f"{split}.paragraphs.npy", np.asarray(rows, dtype=PARAGRAPH_DTYPE),
                    allow_pickle=False)
            for suffix in ("ids.u16", "paragraphs.npy"):
                name = f"{split}.{suffix}"
                outputs[name] = identity(self.directory / name)
        (self.directory / "vocabulary.json").write_text(json.dumps({
            "tokens": tokens, "reserved_ids": dict(zip(RESERVED, range(4)))}))
        outputs["vocabulary.json"] = identity(self.directory / "vocabulary.json")
        (self.directory / "manifest.json").write_text(json.dumps({
            "status": "complete", "context_size": 2, "vocab_size": len(tokens), "outputs": outputs}))
        subprocess.run([sys.executable, str(ROOT / "export_integer_npy.py"), "--corpus",
                        str(self.directory), "--report", str(self.report)],
                       check=True, capture_output=True, text=True)

    def load(self, split="train", fraction=1.0, **kwargs):
        return NpyArrayCorpus(self.directory, split, 2, fraction=fraction,
                              export_report=self.report, **kwargs)

    def test_all_windows_and_resumed_sampler_match_legacy_for_each_split_and_subset(self):
        for split in ("train", "validation", "test"):
            for fraction in (.1, .25, 1.0):
                with self.subTest(split=split, fraction=fraction):
                    old = ArrayCorpus(self.directory, split, 2, fraction=fraction)
                    new = self.load(split, fraction)
                    self.assertIsInstance(new.ids, np.memmap)
                    self.assertIsInstance(new.records, np.memmap)
                    self.assertFalse(new.ids.flags.writeable)
                    self.assertEqual(new.windows, old.windows)
                    for left, right in zip(old.batch(np.arange(old.windows)),
                                           new.batch(np.arange(new.windows))):
                        np.testing.assert_array_equal(left, right)
                    sampler = WindowSampler(new.windows, seed=42)
                    consumed = new.windows - 1
                    sampler.take(consumed)
                    resumed = WindowSampler(old.windows, seed=42, consumed=consumed)
                    for left, right in zip(new.batch(sampler.take(new.windows + 2)),
                                           old.batch(resumed.take(old.windows + 2))):
                        np.testing.assert_array_equal(left, right)

    def test_numpy_reader_does_not_require_raw_integer_files(self):
        for path in self.directory.glob("*.ids.u16"):
            path.unlink()
        for split in ("train", "validation", "test"):
            self.assertEqual(self.load(split).batch([0])[1].tolist(), [4])

    def test_bound_files_reject_corruption(self):
        for name in ("train.ids.npy", "train.paragraphs.npy", "vocabulary.json", "manifest.json"):
            with self.subTest(name=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaises(ValueError):
                    self.load()
                path.write_bytes(original)

    def test_headers_id_bounds_and_paragraph_boundaries_still_checked_without_array_hashes(self):
        path = self.directory / "train.ids.npy"
        original = path.read_bytes()
        for changed in (np.zeros((1, 13), dtype="<u2"), np.zeros(13, dtype="<u4"),
                        np.asarray([2, 2, 10, 5, 6, 7, 8, 3, 2, 2, 9, 8, 3], dtype="<u2"),
                        np.asarray([2, 2, 4, 5, 6, 7, 8, 4, 2, 2, 9, 8, 3], dtype="<u2")):
            np.save(path, changed, allow_pickle=False)
            with self.assertRaises(ValueError):
                self.load(verify=False)
        path.write_bytes(original)
        records_path = self.directory / "train.paragraphs.npy"
        records = np.load(records_path, allow_pickle=False)
        records[1]["start"] -= 1
        np.save(records_path, records, allow_pickle=False)
        with self.assertRaises(ValueError):
            self.load(verify=False)

    def test_configuration_and_export_split_binding_are_checked(self):
        for fraction in (0, 2, float("nan")):
            with self.assertRaises(ValueError):
                self.load(fraction=fraction)
        with self.assertRaises(ValueError):
            NpyArrayCorpus(self.directory, "train", 3, export_report=self.report)
        report = json.loads(self.report.read_text())
        report["arrays"].append(report["arrays"][0])
        self.report.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
