"""Shared exact C initializer and real-Wiki affine sampling for CPU/MPS checks.

No Torch import, implicit compilation, mmap, model execution or source writes.
Root builds init_reference.dylib before using this module. Corpus arrays are
owned RAM copies; generated batches have int64 IDs without changing their values.
"""

from __future__ import annotations

import argparse
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
FNV_OFFSET = 14695981039346656037
MASK32 = (1 << 32) - 1
PARAGRAPH_DTYPE = np.dtype([
    ("start", "<u8"), ("length", "<u4"), ("subset_key", "<u4"),
    ("sentences", "<u4"), ("doc_key", "<u8"),
], align=False)
FULL_CONFIG = {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64,
               "hidden_dim": 128, "batch_size": 128, "seed": 42}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    bounds = {"vocab_size": (2, 65536), "context_size": (1, 32), "embedding_dim": (1, 512),
              "hidden_dim": (1, 1024), "batch_size": (1, 1024), "seed": (0, MASK32)}
    for name, (low, high) in bounds.items():
        value = config.get(name, 42 if name == "seed" else None)
        require(type(value) is int and low <= value <= high, f"Invalid initializer config: {name}")


def parameter_layout(config):
    validate_config(config)
    vocab, context, dim, hidden = (config[key] for key in
                                   ("vocab_size", "context_size", "embedding_dim", "hidden_dim"))
    shapes = [("embedding", [vocab, dim]), ("hidden_weight", [hidden, context * dim]),
              ("hidden_bias", [hidden]), ("output_weight", [vocab, hidden]), ("output_bias", [vocab])]
    offset, result = 0, []
    for name, shape in shapes:
        length = math.prod(shape)
        result.append({"name": name, "shape": shape, "offset": offset, "length": length})
        offset += length
    return result


def parameter_count(config):
    layout = parameter_layout(config)
    return layout[-1]["offset"] + layout[-1]["length"]


@lru_cache(maxsize=4)
def _load_library(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Compile the approved BLAS-free C helper first: {path}; no automatic fallback/compilation is performed")
    library = ctypes.CDLL(str(path))
    library.init_reference_abi_version.argtypes = []
    library.init_reference_abi_version.restype = ctypes.c_uint32
    require(library.init_reference_abi_version() == 1, "Unsupported initializer helper ABI")
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    u32 = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
    library.fill_init.argtypes = [f32, ctypes.c_size_t, ctypes.c_uint32]
    library.fill_init.restype = ctypes.c_uint32
    library.fnv32_floats.argtypes = [f32, ctypes.c_size_t]
    library.fnv32_floats.restype = ctypes.c_uint64
    library.fnv32.argtypes = [u32, ctypes.c_size_t, ctypes.c_uint64]
    library.fnv32.restype = ctypes.c_uint64
    library.fill_synthetic.argtypes = [u32, u32, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    library.fill_synthetic.restype = ctypes.c_uint32
    return library


def helper_library_path():
    return Path(os.environ.get("KOWIKI_INIT_REFERENCE_LIBRARY", str(BASE / "init_reference.dylib"))).resolve()


def create_initial(config):
    """Return (flat float32, five-entry layout) with the C engine's exact bits."""
    layout = parameter_layout(config)
    count = layout[-1]["offset"] + layout[-1]["length"]
    flat = np.empty(count, dtype=np.float32)
    _load_library(str(helper_library_path())).fill_init(flat, count, config.get("seed", 42))
    require(bool(np.isfinite(flat).all()), "The initialization helper returned non-finite values")
    return flat, layout


def checksum_f32(values):
    values = np.asarray(values)
    require(values.dtype == np.dtype(np.float32), "Float checksum requires float32; no implicit precision conversion")
    flat = np.ascontiguousarray(values.reshape(-1))
    return f"{_load_library(str(helper_library_path())).fnv32_floats(flat, flat.size):016x}"


def _as_u32(values):
    values = np.asarray(values)
    require(values.dtype.kind in "iu" and bool(np.all(values >= 0)) and bool(np.all(values <= MASK32)),
            "Checksum IDs must be uint32-representable integers")
    return np.ascontiguousarray(values.reshape(-1), dtype=np.uint32)


def checksum_batch(contexts, targets):
    """Hash all row-major context IDs first, followed by all target IDs (C ABI)."""
    contexts, targets = np.asarray(contexts), np.asarray(targets)
    require(contexts.ndim == 2 and targets.ndim == 1 and contexts.shape[0] == targets.shape[0], "Invalid batch shapes")
    words = np.concatenate((_as_u32(contexts), _as_u32(targets)))
    return f"{_load_library(str(helper_library_path())).fnv32(words, words.size, FNV_OFFSET):016x}"


def synthetic_batches(config, batches=4):
    """C-compatible synthetic stream after the initializer's PRNG draws."""
    validate_config(config)
    require(type(batches) is int and 1 <= batches <= 64, "Synthetic batch count must be 1..64")
    batch, context = config["batch_size"], config["context_size"]
    contexts = np.empty(batches * batch * context, dtype=np.uint32)
    targets = np.empty(batches * batch, dtype=np.uint32)
    _load_library(str(helper_library_path())).fill_synthetic(
        contexts, targets, parameter_count(config), batches, batch, context,
        config["vocab_size"], config.get("seed", 42))
    contexts = contexts.reshape(batches, batch, context)
    targets = targets.reshape(batches, batch)
    return [(np.array(contexts[index], dtype=np.int64, copy=True),
             np.array(targets[index], dtype=np.int64, copy=True)) for index in range(batches)]


create_synthetic_batches = synthetic_batches


def _next_random(state):
    state = (state ^ ((state << 13) & MASK32)) & MASK32
    state = (state ^ (state >> 17)) & MASK32
    return (state ^ ((state << 5) & MASK32)) & MASK32


class Corpus:
    def __init__(self, path, context_size, vocab_size):
        self.path = Path(path).resolve()
        self.context_size, self.vocab_size = context_size, vocab_size
        manifest_bytes = (self.path / "manifest.json").read_bytes()
        self.manifest = json.loads(manifest_bytes)
        require(self.manifest.get("status") == "complete", "A complete retained corpus is required")
        require(self.manifest.get("context_size") == context_size and self.manifest.get("vocab_size") == vocab_size,
                "Corpus vocabulary/context differs from the model")
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.ids = np.array(np.load(self.path / "train.ids.npy", allow_pickle=False), copy=True, order="C")
        self.paragraphs = np.array(np.load(self.path / "train.paragraphs.npy", allow_pickle=False), copy=True, order="C")
        self._validate_and_index()
        require(self.windows == self.manifest["statistics"]["train_target_windows"], "Window count differs from manifest")

    @classmethod
    def from_arrays(cls, ids, paragraphs, context_size, vocab_size):
        """Small in-memory fixture constructor; never reads a Wiki source."""
        self = cls.__new__(cls)
        self.path, self.manifest, self.manifest_sha256 = None, None, None
        self.context_size, self.vocab_size = context_size, vocab_size
        self.ids, self.paragraphs = np.array(ids, copy=True, order="C"), np.array(paragraphs, copy=True, order="C")
        self._validate_and_index()
        return self

    def _validate_and_index(self):
        require(type(self.context_size) is int and 1 <= self.context_size <= 32, "Invalid context size")
        require(type(self.vocab_size) is int and 2 <= self.vocab_size <= 65536, "Invalid vocabulary size")
        require(self.ids.ndim == 1 and self.ids.size > 0 and self.ids.dtype == np.dtype("<u2"), "Require one-dimensional uint16 IDs")
        require(self.paragraphs.ndim == 1 and self.paragraphs.size > 0
                and self.paragraphs.dtype == PARAGRAPH_DTYPE and self.paragraphs.dtype.itemsize == 28,
                "Require packed 28-byte paragraph records")
        starts = self.paragraphs["start"].astype(np.uint64)
        lengths = self.paragraphs["length"].astype(np.uint64)
        require(bool(np.all(lengths > self.context_size)), "Paragraph does not contain target windows")
        end_offsets = starts + lengths
        require(starts[0] == 0 and bool(np.all(starts[1:] == end_offsets[:-1]))
                and int(end_offsets[-1]) == self.ids.size and bool(np.all(end_offsets <= self.ids.size)),
                "Paragraph records are not contiguous/in range")
        self.prefix = np.cumsum(lengths - self.context_size, dtype=np.uint64)
        self.windows = int(self.prefix[-1])
        require(0 < self.windows < 2147483648, "C reference requires 0 < windows < 2^31")
        for offset in range(self.context_size):
            require(bool(np.all(self.ids[starts + offset] == 2)), "Paragraph BOS context mismatch")
        require(bool(np.all(self.ids[end_offsets - 1] == 3)), "Paragraph EOS mismatch")
        bos_count = eos_count = 0
        for offset in range(0, self.ids.size, 1048576):
            chunk = self.ids[offset:offset + 1048576]
            require(bool(np.all((chunk >= 2) & (chunk < self.vocab_size))), "Corpus ID outside retained vocabulary")
            bos_count += int(np.count_nonzero(chunk == 2))
            eos_count += int(np.count_nonzero(chunk == 3))
        require(bos_count == self.paragraphs.size * self.context_size and eos_count == self.paragraphs.size,
                "Corpus BOS/EOS counts differ from C contract")
        self.ids.flags.writeable = self.paragraphs.flags.writeable = self.prefix.flags.writeable = False

    @property
    def target_windows(self):
        return self.windows

    @property
    def paragraph_count(self):
        return int(self.paragraphs.size)

    @property
    def owned_ram_bytes(self):
        return int(self.ids.nbytes + self.paragraphs.nbytes + self.prefix.nbytes)

    def sampler(self, seed):
        return Sampler(self, seed)

    def first_samples(self, seed=42, count=4):
        return self.sampler(seed).first_samples(count)

    @property
    def metadata(self):
        return {"path": str(self.path) if self.path is not None else None, "split": "train",
                "stored_ids": int(self.ids.size), "paragraphs": self.paragraph_count,
                "target_windows": self.windows, "owned_ram_bytes": self.owned_ram_bytes,
                "context_size": self.context_size, "vocab_size": self.vocab_size,
                "manifest_sha256": self.manifest_sha256,
                "sampling_policy": "C-compatible affine permutation with coprime multiplier and exact uint32 xorshift state"}


class Sampler:
    def __init__(self, corpus, seed):
        self.corpus = corpus
        self.reset(seed)

    def _next_pass(self):
        self.position = 0
        self.random = _next_random(self.random)
        self.multiplier = self.random % self.corpus.windows or 1
        while math.gcd(self.multiplier, self.corpus.windows) != 1:
            self.multiplier = (self.multiplier + 1) % self.corpus.windows or 1
        self.random = _next_random(self.random)
        self.shift = self.random % self.corpus.windows

    def reset(self, seed):
        require(type(seed) is int and 0 <= seed <= MASK32, "Sampler seed must fit uint32")
        self.seed, self.position, self.consumed = seed, 0, 0
        self.random = (seed ^ 0x85EBCA6B) or 0x9E3779B9
        self._next_pass()
        return self

    def snapshot(self):
        return {key: int(getattr(self, key)) for key in ("position", "multiplier", "shift", "consumed", "random")}

    def _indices(self, batch_size):
        require(type(batch_size) is int and 1 <= batch_size <= 1048576, "Invalid gather batch size")
        chunks, remaining = [], batch_size
        while remaining:
            if self.position == self.corpus.windows:
                self._next_pass()
            count = min(remaining, self.corpus.windows - self.position)
            positions = np.arange(self.position, self.position + count, dtype=np.uint64)
            windows = (positions * np.uint64(self.multiplier) + np.uint64(self.shift)) % np.uint64(self.corpus.windows)
            chunks.append(windows)
            self.position += count
            self.consumed += count
            remaining -= count
        windows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        paragraphs = np.searchsorted(self.corpus.prefix, windows, side="right")
        before = np.zeros(windows.size, dtype=np.uint64)
        previous = paragraphs > 0
        before[previous] = self.corpus.prefix[paragraphs[previous] - 1]
        starts = self.corpus.paragraphs["start"][paragraphs] + windows - before
        return windows, paragraphs, starts

    def next_batch(self, batch_size):
        _, _, starts = self._indices(batch_size)
        positions = starts[:, None] + np.arange(self.corpus.context_size, dtype=np.uint64)[None, :]
        contexts = np.array(self.corpus.ids[positions], dtype=np.int64, copy=True, order="C")
        targets = np.array(self.corpus.ids[starts + self.corpus.context_size], dtype=np.int64, copy=True, order="C")
        return contexts, targets

    def first_samples(self, count=4):
        require(type(count) is int and 1 <= count <= 1024, "Invalid first-sample count")
        clone = Sampler(self.corpus, self.seed)
        windows, paragraphs, starts = clone._indices(count)
        return [{"window_index": int(window), "paragraph_index": int(paragraph),
                 "contexts": self.corpus.ids[int(start):int(start) + self.corpus.context_size].astype(int).tolist(),
                 "target": int(self.corpus.ids[int(start) + self.corpus.context_size])}
                for window, paragraph, start in zip(windows, paragraphs, starts)]


def validate_cpu_reference(corpus_path, reference_report, *, steps=285):
    """No model execution: compare shared fixtures with preserved CPU metadata."""
    report = json.loads(Path(reference_report).read_text(encoding="utf-8"))
    trials = [row["measurement"] for row in report["trials"] if row.get("passed")]
    require(trials and all(row["steps"] == steps for row in trials), "Reference scope differs from requested fixed work")
    initial, layout = create_initial(FULL_CONFIG)
    initial_checksum = checksum_f32(initial)
    corpus = Corpus(corpus_path, FULL_CONFIG["context_size"], FULL_CONFIG["vocab_size"])
    sampler = corpus.sampler(FULL_CONFIG["seed"])
    start = sampler.snapshot()
    first = sampler.first_samples()
    contexts, targets = sampler.next_batch(FULL_CONFIG["batch_size"])
    first_checksum = checksum_batch(contexts, targets)
    for _ in range(1, steps):
        contexts, targets = sampler.next_batch(FULL_CONFIG["batch_size"])
    last_checksum = checksum_batch(contexts, targets)
    final = sampler.snapshot()
    for row in trials:
        require(row["initial_parameter_checksum"] == initial_checksum, "C initializer checksum mismatch")
        require(row["input_checksum"] == first_checksum and row["final_input_checksum"] == last_checksum,
                "C first/final batch checksum mismatch")
        require(row["sampler_initial"] == start and row["sampler_final"] == final, "C sampler snapshots differ")
        require(row["corpus"]["first_windows"] == first and row["corpus"]["target_windows"] == corpus.windows,
                "C first-window or full-window metadata differs")
    return {"passed": True, "reference_trial_count": len(trials), "steps": steps,
            "events": steps * FULL_CONFIG["batch_size"], "parameter_count": initial.size,
            "parameter_layout": layout, "initial_parameter_checksum": initial_checksum,
            "input_checksum": first_checksum, "final_input_checksum": last_checksum,
            "sampler_initial": start, "sampler_final": final, "first_windows": first,
            "corpus": corpus.metadata, "helper_library": str(helper_library_path()),
            "scope": "Initialization and integer gathering only; no CPU/MPS model, training, or timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training-corpora/wiki-k35328")
    parser.add_argument("--reference", type=Path, default=ROOT / "cpu-benchmark/results/cpu-fixedwork-mac-20261018-v1.json")
    parser.add_argument("--steps", type=int, default=285)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_cpu_reference(args.corpus, args.reference, steps=args.steps)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
