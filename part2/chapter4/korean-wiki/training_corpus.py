"""Shared, deterministic corpus utilities for the controlled Wiki experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


RESERVED = ["<pad>", "<unk>", "<bos>", "<eos>"]
PARAGRAPH_DTYPE = np.dtype([
    ("start", "<u8"), ("length", "<u4"), ("subset_key", "<u4"),
    ("sentences", "<u4"), ("doc_key", "<u8"),
])


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def paragraphs(path):
    paragraph = []
    with Path(path).open(encoding="utf-8", newline=None) as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                paragraph.append(line)
            elif paragraph:
                yield paragraph
                paragraph = []
    if paragraph:
        yield paragraph


def stable_int(value, seed, purpose, size=8):
    value = f"{seed}:{purpose}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=size).digest(), "big")


def document_split(doc_id, seed=42):
    # These thresholds are fixed before inspecting any token frequencies.
    bucket = stable_int(str(doc_id), seed, "split") % 10000
    return "train" if bucket < 9800 else "validation" if bucket < 9900 else "test"


def subset_key(doc_id, seed=42):
    # Separate hash purpose keeps subset selection independent of split choice.
    return stable_int(str(doc_id), seed, "subset", size=4)


def token_line_key(line):
    return hashlib.sha256(line.encode("utf-8")).digest()


def tokenized_sentence_key(line):
    """Ignore sentence-edge WS only when comparing duplicates, not in outputs.

    The lecture adds a trailing WS when another sentence follows in a paragraph.
    This boundary marker must not hide an otherwise identical heldout sentence.
    """
    while line.startswith("<sp>/WS "):
        line = line[len("<sp>/WS "):]
    while line.endswith(" <sp>/WS"):
        line = line[:-len(" <sp>/WS")]
    return token_line_key("" if line == "<sp>/WS" else line)


def choose_vocabulary(counts, size):
    if not 5 <= size <= 65536:
        raise ValueError("Vocabulary includes four reserved IDs; require 5 <= K <= 65,536")
    ordinary = [token for token, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:size - 4]]
    if len(ordinary) != size - 4 or any(token in RESERVED for token in ordinary):
        raise ValueError("Insufficient ordinary types, or reserved-token collision")
    return RESERVED + ordinary


def encode_paragraph(lines, vocabulary, context_size):
    """Lecture policy: discard an OOV sentence, retain any remaining sentences."""
    kept = [line.split(" ") for line in lines if all(token in vocabulary for token in line.split(" "))]
    if not kept:
        return None, 0
    ids = [2] * context_size
    for sentence in kept:
        ids.extend(vocabulary[token] for token in sentence)
    ids.append(3)
    return np.asarray(ids, dtype="<u2"), len(kept)


class ArrayCorpus:
    """Memory-mapped paragraphs; contexts never cross a paragraph boundary."""

    def __init__(self, directory, split, context_size, fraction=1.0, verify=True):
        directory = Path(directory)
        if split not in {"train", "validation", "test"} or not 0 < fraction <= 1:
            raise ValueError("Invalid split/fraction")
        self.context_size = context_size
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["context_size"] != context_size:
            raise ValueError("Incomplete corpus or context-size mismatch")
        self.manifest = manifest
        for suffix in ("ids.u16", "paragraphs.npy"):
            name = f"{split}.{suffix}"
            if verify and identity(directory / name) != manifest["outputs"][name]:
                raise ValueError(f"Corpus identity mismatch: {name}")
        self.ids = np.memmap(directory / f"{split}.ids.u16", dtype="<u2", mode="r")
        self.records = np.load(directory / f"{split}.paragraphs.npy", mmap_mode="r", allow_pickle=False)
        self.selected = np.flatnonzero(self.records["subset_key"] < int(fraction * 2 ** 32))
        if not len(self.selected):
            raise ValueError("Selected corpus has no paragraphs")
        self.counts = self.records["length"][self.selected].astype(np.int64) - context_size
        if np.any(self.counts < 1):
            raise ValueError("Paragraph does not contain a target after its BOS context")
        self.cumulative = np.cumsum(self.counts, dtype=np.int64)
        self.windows = int(self.cumulative[-1])

    def batch(self, windows):
        windows = np.asarray(windows, dtype=np.int64)
        if np.any(windows < 0) or np.any(windows >= self.windows):
            raise IndexError("Target window outside corpus")
        selected_index = np.searchsorted(self.cumulative, windows, side="right")
        before = np.where(selected_index == 0, 0, self.cumulative[np.maximum(selected_index - 1, 0)])
        paragraph_ids = self.selected[selected_index]
        starts = self.records["start"][paragraph_ids].astype(np.int64) + windows - before
        contexts = self.ids[starts[:, None] + np.arange(self.context_size)].astype(np.int64)
        targets = self.ids[starts + self.context_size].astype(np.int64)
        return contexts, targets, paragraph_ids


class WindowSampler:
    """Reproducible permutation per epoch without an O(number of tokens) array.

    An affine permutation visits every target exactly once before repeating.
    This is deterministic mixing, not a uniform draw from all possible orders.
    """

    def __init__(self, windows, seed=42, consumed=0):
        if windows < 1 or consumed < 0:
            raise ValueError("Require a positive corpus and nonnegative progress")
        self.windows, self.seed, self.consumed = int(windows), seed, int(consumed)

    def take(self, count):
        if count < 1:
            raise ValueError("Batch size must be positive")
        pieces = []
        while count:
            epoch, offset = divmod(self.consumed, self.windows)
            amount = min(count, self.windows - offset)
            multiplier = stable_int(epoch, self.seed, "permutation-a") % self.windows or 1
            while math.gcd(multiplier, self.windows) != 1:
                multiplier = (multiplier + 1) % self.windows or 1
            shift = stable_int(epoch, self.seed, "permutation-b") % self.windows
            # At production scale N < 2**31, so multiplication fits signed int64.
            if self.windows >= 2 ** 31:
                raise ValueError("This bounded sampler supports fewer than 2**31 windows")
            positions = np.arange(offset, offset + amount, dtype=np.int64)
            pieces.append((multiplier * positions + shift) % self.windows)
            self.consumed += amount
            count -= amount
        return np.concatenate(pieces)
