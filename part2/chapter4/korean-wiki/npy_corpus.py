"""Read the exported NumPy corpus with the original paragraph/window semantics.

The legacy trainer remains unchanged.  This additive reader uses only the .npy
arrays, paragraph metadata, dictionary, manifest, and bound export report; it
does not need the old raw .u16 files or start any model computation.
"""

import json
from pathlib import Path

import numpy as np

from training_corpus import ArrayCorpus, PARAGRAPH_DTYPE, RESERVED, identity


def _record_identity(record):
    return {key: record[key] for key in ("bytes", "sha256")}


class NpyArrayCorpus(ArrayCorpus):
    """A memory-mapped .npy replacement for ``ArrayCorpus``.

    ``batch`` is inherited unchanged, so context boundaries, target indexing,
    subset selection and ``WindowSampler`` positions match the legacy reader.
    ``verify=False`` skips large array/paragraph file hashes only; dictionary
    and manifest binding, array headers, ID bounds and boundaries remain checked.
    Pass ``export_report`` when moving the corpus without the default directory
    layout. Absolute source paths in the report are provenance, not load paths.
    """

    def __init__(self, directory, split, context_size, fraction=1.0, verify=True,
                 *, export_report=None):
        directory = Path(directory).resolve()
        if split not in {"train", "validation", "test"} or not 0 < fraction <= 1:
            raise ValueError("Invalid split/fraction")
        if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size < 1:
            raise ValueError("Require a positive integer context size")
        report_path = (directory.parent.parent / "reports/integer-encoding-npy.json"
                       if export_report is None else Path(export_report))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest_path = directory / "manifest.json"
        vocabulary_path = directory / "vocabulary.json"
        if report["status"] != "complete":
            raise ValueError("Incomplete NumPy export report")
        if identity(manifest_path) != _record_identity(report["source_manifest"]):
            raise ValueError("Original manifest differs from the bound NumPy export")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["context_size"] != context_size:
            raise ValueError("Incomplete corpus or context-size mismatch")
        vocabulary_identity = identity(vocabulary_path)
        if (vocabulary_identity != manifest["outputs"]["vocabulary.json"]
                or vocabulary_identity != _record_identity(report["vocabulary"])):
            raise ValueError("Vocabulary differs from the bound corpus/export")
        dictionary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        vocabulary = dictionary["tokens"]
        if (not 5 <= len(vocabulary) <= 65536
                or len(vocabulary) != manifest["vocab_size"]
                or len(vocabulary) != report["vocabulary"]["size"]
                or len(set(vocabulary)) != len(vocabulary)
                or vocabulary[:4] != RESERVED
                or dictionary["reserved_ids"] != dict(zip(RESERVED, range(4)))):
            raise ValueError("Invalid vocabulary size or reserved ID mapping")
        entries = [entry for entry in report["arrays"] if entry["split"] == split]
        if len(entries) != 1:
            raise ValueError("Require exactly one NumPy export entry for the split")
        entry = entries[0]
        if _record_identity(entry["source"]) != manifest["outputs"][f"{split}.ids.u16"]:
            raise ValueError("NumPy source identity differs from the original corpus")
        ids_path, records_path = directory / f"{split}.ids.npy", directory / f"{split}.paragraphs.npy"
        if verify:
            if identity(ids_path) != entry["output"]:
                raise ValueError("NumPy integer array identity mismatch")
            if identity(records_path) != manifest["outputs"][records_path.name]:
                raise ValueError("Paragraph metadata identity mismatch")
        self.ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        self.records = np.load(records_path, mmap_mode="r", allow_pickle=False)
        if (self.ids.ndim != 1 or self.ids.dtype != np.dtype("<u2")
                or list(self.ids.shape) != entry["shape"]
                or self.ids.size * 2 != entry["source"]["bytes"] or not self.ids.size):
            raise ValueError("Incorrect NumPy integer-array header/shape")
        if (self.records.ndim != 1 or self.records.dtype != PARAGRAPH_DTYPE
                or not len(self.records)):
            raise ValueError("Incorrect paragraph metadata header")
        if int(self.ids.min()) < 2 or int(self.ids.max()) >= len(vocabulary):
            raise ValueError("Integer ID is outside the retained vocabulary")
        starts = self.records["start"]
        lengths = self.records["length"]
        if np.any(starts > len(self.ids)) or np.any(lengths > len(self.ids)):
            raise ValueError("Paragraph points outside the integer array")
        starts = starts.astype(np.int64)
        lengths = lengths.astype(np.int64)
        ends = starts + lengths
        if (starts[0] != 0 or np.any(starts[1:] != ends[:-1])
                or ends[-1] != len(self.ids) or np.any(lengths <= context_size)):
            raise ValueError("Paragraph offsets/lengths do not cover the integer array")
        if (np.any(self.ids[ends - 1] != 3)
                or any(np.any(self.ids[starts + offset] != 2) for offset in range(context_size))
                or np.count_nonzero(self.ids == 2) != len(self.records) * context_size
                or np.count_nonzero(self.ids == 3) != len(self.records)):
            raise ValueError("BOS/EOS paragraph boundaries are inconsistent")
        self.context_size, self.manifest = context_size, manifest
        self.vocabulary, self.export_report = vocabulary, report
        self.selected = np.flatnonzero(self.records["subset_key"] < int(fraction * 2 ** 32))
        if not len(self.selected):
            raise ValueError("Selected corpus has no paragraphs")
        self.counts = self.records["length"][self.selected].astype(np.int64) - context_size
        self.cumulative = np.cumsum(self.counts, dtype=np.int64)
        self.windows = int(self.cumulative[-1])
