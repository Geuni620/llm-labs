"""Export the existing integer corpus to NumPy arrays without changing its IDs."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def identity(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--corpus", type=Path, default=base / "training-corpora/wiki-k35328")
    parser.add_argument("--report", type=Path, default=base / "reports/integer-encoding-npy.json")
    args = parser.parse_args()
    corpus, report_path = args.corpus.resolve(), args.report.resolve()
    splits = ("train", "validation", "test")
    destinations = [corpus / f"{split}.ids.npy" for split in splits]
    for path in [*destinations, report_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    manifest_path, vocabulary_path = corpus / "manifest.json", corpus / "vocabulary.json"
    manifest_identity = identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("The source corpus is incomplete")
    vocabulary_identity = identity(vocabulary_path)
    if vocabulary_identity != manifest["outputs"]["vocabulary.json"]:
        raise ValueError("Vocabulary differs from the bound source manifest")
    vocabulary = json.loads(vocabulary_path.read_text())["tokens"]
    if len(vocabulary) != manifest["vocab_size"]:
        raise ValueError("Vocabulary size mismatch")

    report = {
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_manifest": {"path": str(manifest_path), **manifest_identity},
        "vocabulary": {"path": str(vocabulary_path), "size": len(vocabulary), **vocabulary_identity},
        "dtype": "<u2",
        "format": "NumPy .npy, one-dimensional uint16 token IDs",
        "id_policy": "Existing IDs are unchanged; vocabulary.tokens[id] resolves each token.",
        "boundary_metadata": "Existing split.paragraphs.npy records start/length and document provenance; BOS/EOS remain in the ID arrays.",
        "scope": f"Existing train-only K={len(vocabulary):,} vocabulary and OOV-filtered train/validation/test corpus; not every original token is retained.",
        "training_restarted": False,
        "arrays": [],
    }
    chunk_size = 1_048_576
    with tempfile.TemporaryDirectory(prefix=".npy-export-", dir=corpus) as temporary:
        staged = Path(temporary)
        for split in splits:
            source_path = corpus / f"{split}.ids.u16"
            source_identity = identity(source_path)
            if source_identity != manifest["outputs"][source_path.name]:
                raise ValueError(f"Source differs from its manifest: {source_path}")
            if source_identity["bytes"] % 2:
                raise ValueError(f"Invalid uint16 payload length: {source_path}")
            source = np.memmap(source_path, dtype="<u2", mode="r")
            target_path = staged / f"{split}.ids.npy"
            target = np.lib.format.open_memmap(target_path, mode="w+", dtype="<u2", shape=source.shape)
            for start in range(0, len(source), chunk_size):
                target[start:start + chunk_size] = source[start:start + chunk_size]
            target.flush()
            del target

            restored = np.load(target_path, mmap_mode="r", allow_pickle=False)
            if restored.shape != source.shape or restored.dtype != np.dtype("<u2"):
                raise ValueError(f"Incorrect NumPy array header: {split}")
            low, high = len(vocabulary), -1
            for start in range(0, len(source), chunk_size):
                block = restored[start:start + chunk_size]
                if not np.array_equal(block, source[start:start + chunk_size]):
                    raise ValueError(f"Integer round trip failed: {split}, offset {start}")
                if block.size:
                    low, high = min(low, int(block.min())), max(high, int(block.max()))
            if high >= len(vocabulary):
                raise ValueError(f"Out-of-vocabulary integer ID: {split}")
            paragraph_path = corpus / f"{split}.paragraphs.npy"
            if identity(paragraph_path) != manifest["outputs"][paragraph_path.name]:
                raise ValueError(f"Paragraph metadata changed: {split}")
            paragraphs = np.load(paragraph_path, mmap_mode="r", allow_pickle=False)
            if not len(paragraphs) or int(paragraphs[-1]["start"]) + int(paragraphs[-1]["length"]) != len(restored):
                raise ValueError(f"Paragraph offsets do not match the integer array: {split}")
            sample_ids = restored[:24].tolist()
            report["arrays"].append({
                "split": split,
                "path": str(corpus / target_path.name),
                "shape": list(restored.shape),
                "dtype": str(restored.dtype),
                "id_min": low,
                "id_max": high,
                "source": {"path": str(source_path), **source_identity},
                "output": identity(target_path),
                "header_bytes": int(restored.offset),
                "all_integer_values_equal": True,
                "sample_ids": sample_ids,
                "sample_tokens": [vocabulary[index] for index in sample_ids],
            })
            del source, restored, paragraphs
            print(f"Verified {split}: {report['arrays'][-1]['shape'][0]:,} uint16 IDs", flush=True)

        if identity(manifest_path) != manifest_identity or identity(vocabulary_path) != vocabulary_identity:
            raise ValueError("Source manifest or vocabulary changed during export")
        report["verification"] = "All ID values compared against the original arrays; source hashes, vocabulary, and paragraph-end offsets checked."
        report_path.parent.mkdir(parents=True, exist_ok=True)
        staged_report = staged / "export-report.json"
        staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        for destination in destinations:
            os.link(staged / destination.name, destination)
        os.link(staged_report, report_path)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
