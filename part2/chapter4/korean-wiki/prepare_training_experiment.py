"""Prepare article-separated, train-only-vocabulary Wiki experiment inputs.

Original text/token files are read-only. Rebuild article provenance from Parquet
and check every retained paragraph against the existing deduplicated text. Only
after this check do we count training tokens and encode numeric-split sentences.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import fcntl
import itertools
import json
from pathlib import Path
import shutil
import struct
import tempfile
import time

import numpy as np

from training_corpus import (PARAGRAPH_DTYPE, atomic_json, choose_vocabulary,
                             document_split, encode_paragraph, identity, paragraphs,
                             stable_int, subset_key, token_line_key, tokenized_sentence_key)


ROOT = Path(__file__).resolve().parent


def rebuild_provenance(root, staging, seed):
    import pyarrow.parquet as pq
    from step3_extract_paragraphs import (PARQUET_FILES, evaluate_line, has_formula,
                                         split_sentences)
    from step4_dedup_sentences import hash_key, normalize

    files = [pq.ParquetFile(root / name) for name in PARQUET_FILES]
    tasks = [(i, group) for i, file in enumerate(files) for group in range(file.num_row_groups)]
    counts = Counter()
    for shard in range(32):
        name = f"part_{shard:02d}"
        source = root / "sentences_dedup" / f"{name}.txt"
        wanted = {hash_key(normalize(line)) for paragraph in paragraphs(source) for line in paragraph}
        actual = iter(paragraphs(source))
        output = staging / "provenance" / f"{name}.jsonl.gz"
        with gzip.open(output, "wt", encoding="utf-8", newline="\n") as target:
            for file_index, row_group in tasks[shard::32]:
                rows = files[file_index].read_row_group(row_group, columns=["id", "title", "text"]).to_pylist()
                for row_index, row in enumerate(rows):
                    text = row["text"]
                    if text is None or has_formula(text):
                        continue
                    for line_number, line in enumerate(text.split("\n"), 1):
                        _, cleaned, rejected = evaluate_line(line)
                        if rejected is not None:
                            continue
                        kept = []
                        for sentence in split_sentences(cleaned):
                            key = hash_key(normalize(sentence))
                            if key in wanted:
                                wanted.remove(key)
                                kept.append(sentence)
                        if not kept:
                            continue
                        if next(actual, None) != kept:
                            raise ValueError(f"Provenance alignment failed: {name}, article {row['id']}, line {line_number}")
                        doc_id = str(row["id"])
                        record = {"doc_id": doc_id, "title": row["title"],
                                  "source": [file_index, row_group, row_index, line_number],
                                  "sentences": len(kept),
                                  "text_sha256": token_line_key("\n".join(kept)).hex(),
                                  "split": document_split(doc_id, seed),
                                  "subset_key": subset_key(doc_id, seed),
                                  "doc_key": stable_int(doc_id, seed, "document")}
                        target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts["paragraphs"] += 1
                        counts["sentences"] += len(kept)
        if wanted or next(actual, None) is not None:
            raise ValueError(f"Unmapped deduplicated input: {name}")
        print(f"provenance {shard + 1}/32: {counts['paragraphs']:,} paragraphs", flush=True)
    return dict(counts)


def iter_records(root, provenance):
    sentinel = object()
    for shard in range(32):
        name = f"part_{shard:02d}"
        with gzip.open(provenance / f"{name}.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = (json.loads(line) for line in stream)
            for lines, record in itertools.zip_longest(paragraphs(root / "tokens_numeric" / f"{name}.txt"), records, fillvalue=sentinel):
                if lines is sentinel or record is sentinel or len(lines) != record["sentences"]:
                    raise ValueError(f"Numeric token/provenance boundary mismatch: {name}")
                yield record, lines


def holdout_keys(root, provenance):
    heldout = {"validation": set(), "test": set()}
    for record, lines in iter_records(root, provenance):
        if record["split"] in heldout:
            heldout[record["split"]].update(map(tokenized_sentence_key, lines))
    overlap = heldout["test"] & heldout["validation"]
    heldout["test"].difference_update(overlap)
    return heldout, len(overlap)


def eligible_records(root, provenance, heldout, stats):
    seen = {"train": set(), "validation": set(), "test": set()}
    excluded = heldout["validation"] | heldout["test"]
    for record, lines in iter_records(root, provenance):
        split = record["split"]
        stats[split + "_source_paragraphs"] += 1
        stats[split + "_source_sentences"] += len(lines)
        result = []
        for line in lines:
            key = tokenized_sentence_key(line)
            if split == "train" and key in excluded:
                stats["train_heldout_overlap_removed_sentences"] += 1
            elif split == "test" and key not in heldout["test"]:
                stats["test_validation_overlap_removed_sentences"] += 1
            elif key in seen[split]:
                stats[split + "_tokenized_duplicates_removed_sentences"] += 1
            else:
                seen[split].add(key)
                result.append(line)
        if result:
            yield record, result


def build_encoded_corpus(root, staging, vocabulary, heldout, context_size):
    lookup = {token: index for index, token in enumerate(vocabulary)}
    stats, offsets, docs = Counter(), Counter(), {split: set() for split in ("train", "validation", "test")}
    handles = {split: ((staging / f"{split}.ids.u16").open("xb"),
                       (staging / f"{split}.paragraphs.bin").open("xb")) for split in docs}
    example_rows = []
    try:
        for record, lines in eligible_records(root, staging / "provenance", heldout, stats):
            split = record["split"]
            stats[split + "_eligible_sentences"] += len(lines)
            stats[split + "_eligible_tokens"] += sum(len(line.split(" ")) for line in lines)
            encoded, kept = encode_paragraph(lines, lookup, context_size)
            stats[split + "_oov_removed_sentences"] += len(lines) - kept
            if encoded is None:
                stats[split + "_oov_removed_paragraphs"] += 1
                continue
            if kept != len(lines):
                stats[split + "_partially_trimmed_paragraphs"] += 1
            start = offsets[split]
            handles[split][0].write(encoded.tobytes())
            handles[split][1].write(struct.pack("<QIIIQ", start, len(encoded), record["subset_key"], kept, record["doc_key"]))
            offsets[split] += len(encoded)
            docs[split].add(record["doc_id"])
            stats[split + "_kept_sentences"] += kept
            stats[split + "_kept_paragraphs"] += 1
            stats[split + "_target_windows"] += len(encoded) - context_size
            if split == "train":
                for fraction in (10, 25, 100):
                    if record["subset_key"] < int(fraction / 100 * 2 ** 32):
                        stats[f"train_{fraction}pct_paragraphs"] += 1
                        stats[f"train_{fraction}pct_sentences"] += kept
                        stats[f"train_{fraction}pct_target_windows"] += len(encoded) - context_size
            if len(example_rows) < 12:
                example_rows.append({"source": record, "before": lines,
                                     "kept_token_lines": [line for line in lines if all(token in lookup for token in line.split(" "))],
                                     "ids_first_80": encoded[:80].tolist()})
    finally:
        for pair in handles.values():
            for stream in pair:
                stream.close()
    if any(docs[a] & docs[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Article leakage across splits")
    for split in docs:
        raw = staging / f"{split}.paragraphs.bin"
        data = np.fromfile(raw, dtype=PARAGRAPH_DTYPE)
        if not len(data) or not stats[split + "_target_windows"]:
            raise ValueError(f"Empty encoded split: {split}")
        np.save(staging / f"{split}.paragraphs.npy", data, allow_pickle=False)
        raw.unlink()
        stats[split + "_documents"] = len(docs[split])
    atomic_json(staging / "examples.json", example_rows)
    return dict(stats)


def validate_source(root):
    from benchmark_nplm import assert_pipeline_idle
    assert_pipeline_idle(root)
    identities = {}
    for name in ("step5-summary.json", "token-frequency-summary.json", "step6-numeric-summary.json",
                 "numeric-token-frequency-summary.json", "vocabulary-coverage.json"):
        path = root / "reports" / name
        payload = json.loads(path.read_text())
        if payload.get("status", "complete") != "complete":
            raise ValueError(f"Incomplete source: {name}")
        identities["reports/" + name] = identity(path)
    for name in ("step5-verification.json", "token-frequency-verification.json", "step6-numeric-verification.json"):
        path = root / "reports" / name
        result = json.loads(path.read_text())
        if result.get("passed") is not True or result.get("scope") != "full":
            raise ValueError(f"Source verification has not passed in full: {name}")
        identities["reports/" + name] = identity(path)
    download = json.loads((root / "download_manifest.json").read_text())
    for row in download["files"]:
        measured = identity(root / row["file"])
        if measured["sha256"] != row["sha256"] or measured["bytes"] != row["bytes"]:
            raise ValueError("Original Parquet changed")
        identities[row["file"]] = measured
    numeric = json.loads((root / "reports/step6-numeric-state.json").read_text())
    for name, record in numeric["completed"].items():
        for relative, expected in [("tokens_numeric/" + name, record["output"])]:
            measured = identity(root / relative)
            if any(measured[key] != expected[key] for key in ("bytes", "sha256")):
                raise ValueError(f"Numeric source changed: {name}")
            identities[relative] = measured
    for path in sorted((root / "sentences_dedup").glob("part_*.txt")):
        identities[str(path.relative_to(root))] = identity(path)
    return identities


def prepare(root, output, vocab_size, context_size=8, seed=42):
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment inputs: {output}")
    if not 1 <= context_size <= 64:
        raise ValueError("Context size must be between 1 and 64")
    inputs = validate_source(root)
    if shutil.disk_usage(root).free < 4 * 1024 ** 3:
        raise RuntimeError("Need at least 4 GiB free to prepare a separate corpus")
    staging = Path(tempfile.mkdtemp(prefix=".training-prep-", dir=root))
    (staging / "provenance").mkdir()
    started = time.monotonic()
    provenance = rebuild_provenance(root, staging, seed)
    heldout, heldout_overlap = holdout_keys(root, staging / "provenance")
    counts, count_stats = Counter(), Counter()
    for record, lines in eligible_records(root, staging / "provenance", heldout, count_stats):
        if record["split"] == "train":
            for line in lines:
                counts.update(line.split(" "))
    vocabulary = choose_vocabulary(counts, vocab_size)
    atomic_json(staging / "vocabulary.json", {"tokens": vocabulary, "reserved_ids": dict(zip(vocabulary[:4], range(4))),
                                             "source": "training articles only, after tokenized dedup and heldout overlap exclusion"})
    with (staging / "train-token-frequencies.txt").open("x", encoding="utf-8") as stream:
        stream.write("rank\ttoken\tcount\n")
        for rank, (token, count) in enumerate(sorted(counts.items(), key=lambda row: (-row[1], row[0])), 1):
            stream.write(f"{rank}\t{token}\t{count}\n")
    frequency_totals = {"tokens": counts.total(), "types": len(counts)}
    del counts
    stats = build_encoded_corpus(root, staging, vocabulary, heldout, context_size)
    outputs = {str(path.relative_to(staging)): identity(path) for path in sorted(staging.rglob("*")) if path.is_file()}
    scripts = {name: identity(root / name) for name in ("prepare_training_experiment.py", "training_corpus.py",
                                                       "step3_extract_paragraphs.py", "step4_dedup_sentences.py")}
    for relative, expected in inputs.items():
        if identity(root / relative) != expected:
            raise ValueError(f"Source changed during preparation: {relative}")
    manifest = {"status": "complete", "schema_version": 1, "vocab_size": vocab_size,
                "context_size": context_size, "seed": seed, "inputs": inputs, "outputs": outputs,
                "scripts": scripts, "provenance": provenance, "training_frequency": frequency_totals,
                "statistics": stats, "heldout_tokenized_overlap_between_validation_test": heldout_overlap,
                "split_policy": "article ID hash, train 98% / validation 1% / test 1%; source sentences globally deduplicated",
                "leakage_policy": "all heldout tokenized sentences excluded from train; sentence-edge WS ignored for comparison only; validation wins overlap with test",
                "subset_policy": "independent article hash creates nested 10%, 25%, 100% of the same eligible training pool",
                "oov_policy": "drop entire sentence; retain paragraph if at least one sentence remains; no UNK replacements",
                "numeric_policy": "pre-split SN and W_SERIAL into digit/SN and separator/SW using existing tokens_numeric inputs",
                "paragraph_encoding": "C BOS context tokens, retained sentence tokens, EOS; no context crosses paragraphs",
                "elapsed_seconds": time.monotonic() - started}
    atomic_json(staging / "manifest.json", manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    print(output / "manifest.json", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with (args.root / "reports/.nplm-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.root.resolve(), args.output.resolve(), args.vocab_size, args.context_size, args.seed)


if __name__ == "__main__":
    main()
