"""Materialize the lecture's numeric-token representation separately.

Only SN and W_SERIAL forms are expanded to individual digit/SN or separator/SW
keys. Fullwidth ０–９ become ASCII digits, exactly as in the supplied reference.
Every sentence, paragraph boundary, ordinary token and <sp>/WS is preserved.
This shrinks numeric vocabulary; it does not establish arithmetic ability.

Use --watch while Step 5 is running. Only manifest-complete, hash-verified raw
shards are consumed. --resume requires unchanged code, configuration and files.
No native Kiwi model is loaded and no raw tokens are overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
import unicodedata
import zlib

from step6_count_tokens import (
    COUNT_KEYS, EXPECTED_SHARDS, IDENTITY_KEYS, SPACE_TOKEN,
    _atomic_json, _require_identity, file_identity, tokenizer_fingerprint,
    write_frequency_table,
)


ROOT = Path(__file__).resolve().parent
SPLIT_POS_TAGS = frozenset({"SN", "W_SERIAL"})
FULLWIDTH_TO_ASCII = str.maketrans("０１２３４５６７８９", "0123456789")
NUMERIC_COUNT_KEYS = (
    "input_tokens", "tokens", "sentences", "paragraphs", "space_tokens",
    "numeric_source_occurrences", "numeric_expansion_tokens",
    "numeric_changed_occurrences", "added_tokens", "lines", "blank_lines",
)


@lru_cache(maxsize=262144)
def _cached_numeric_expansion(token: str) -> tuple[str, ...]:
    surface, _, _ = token.rpartition("/")
    surface = surface.translate(FULLWIDTH_TO_ASCII)
    return tuple(f"{char}/SN" if char.isdigit() else f"{char}/SW" for char in surface)


def _is_numeric_token(token: str) -> bool:
    if not token.endswith(("/SN", "/W_SERIAL")):
        return False
    surface, _, tag = token.rpartition("/")
    return bool(surface) and tag in SPLIT_POS_TAGS


def split_numeric_token(token: str) -> tuple[str, ...]:
    """Pure reference-compatible expansion; only numeric keys enter the cache."""
    if not _is_numeric_token(token):
        return (token,)
    return _cached_numeric_expansion(token)


def _body_and_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1:]
    return line, ""


def transform_token_line(line: str) -> str:
    """Expand a serialized sentence, preserving its optional exact line ending."""
    body, ending = _body_and_ending(line)
    if not body:
        return ending
    tokens = body.split(" ")
    if "" in tokens or any(char.isspace() and char != " " for char in body):
        raise ValueError("Malformed serialized token separator")
    return " ".join(piece for token in tokens for piece in split_numeric_token(token)) + ending


def transform_frequency_counts(counts: Counter) -> Counter:
    """Distribute every source occurrence to every expanded piece, including repeats."""
    expanded = Counter()
    for token, count in counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"Invalid frequency: {token!r}: {count!r}")
        for piece in split_numeric_token(token):
            expanded[piece] += count
    return expanded


def transform_shard(
    input_path: str | Path,
    output_path: str | Path,
    expected_identity: dict,
    expected_counts: dict | None = None,
) -> dict:
    """Verify source bytes while writing one atomic, independently hashed output."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite numeric output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = input_path.stat()
    input_sha, output_sha = hashlib.sha256(), hashlib.sha256()
    input_crc = output_crc = input_bytes = output_bytes = 0
    stats = dict.fromkeys(NUMERIC_COUNT_KEYS, 0)
    source_types, expanded_types, examples = set(), set(), []
    in_paragraph = False
    temporary = None
    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".numeric-{output_path.stem}-",
                                         suffix=".partial", dir=output_path.parent,
                                         delete=False) as target:
            temporary = Path(target.name)
            with input_path.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    input_sha.update(raw)
                    input_crc = zlib.crc32(raw, input_crc)
                    input_bytes += len(raw)
                    body, ending = _body_and_ending(raw.decode("utf-8"))
                    stats["lines"] += 1
                    if not body:
                        stats["blank_lines"] += 1
                        in_paragraph = False
                        result = raw
                    else:
                        tokens = body.split(" ")
                        if "" in tokens or any(char.isspace() and char != " " for char in body):
                            raise ValueError(f"Malformed token separator: {input_path}:{line_number}")
                        pieces = []
                        for token in tokens:
                            if not _is_numeric_token(token):
                                pieces.append(token)
                                continue
                            expansion = _cached_numeric_expansion(token)
                            pieces.extend(expansion)
                            stats["numeric_source_occurrences"] += 1
                            stats["numeric_expansion_tokens"] += len(expansion)
                            source_types.add(token)
                            expanded_types.update(expansion)
                            if expansion != (token,):
                                stats["numeric_changed_occurrences"] += 1
                            if len(examples) < 8 and not any(row["before"] == token for row in examples):
                                examples.append({"kind": "actual", "line": line_number,
                                                 "before": token, "after": list(expansion)})
                        stats["input_tokens"] += len(tokens)
                        stats["tokens"] += len(pieces)
                        stats["space_tokens"] += tokens.count(SPACE_TOKEN)
                        stats["sentences"] += 1
                        if not in_paragraph:
                            stats["paragraphs"] += 1
                            in_paragraph = True
                        result = (" ".join(pieces) + ending).encode("utf-8")
                    target.write(result)
                    output_sha.update(result)
                    output_crc = zlib.crc32(result, output_crc)
                    output_bytes += len(result)
            target.flush()
            os.fsync(target.fileno())
        after = input_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during numeric splitting: {input_path}")
        identity = {"bytes": input_bytes, "mtime_ns": after.st_mtime_ns,
                    "sha256": input_sha.hexdigest(), "crc32": f"{input_crc & 0xffffffff:08x}"}
        _require_identity(identity, expected_identity, input_path)
        if expected_counts is not None:
            for key in COUNT_KEYS:
                actual = stats["input_tokens"] if key == "tokens" else stats[key]
                if actual != expected_counts[key]:
                    raise ValueError(f"Step 5 count mismatch ({key}): {input_path}")
        stats["added_tokens"] = stats["tokens"] - stats["input_tokens"]
        if stats["added_tokens"] != stats["numeric_expansion_tokens"] - stats["numeric_source_occurrences"]:
            raise RuntimeError("Numeric expansion total is inconsistent")
        # link() is atomic and refuses a destination that appeared in the meantime.
        os.link(temporary, output_path)
        temporary.unlink()
        temporary = None
        output = {"bytes": output_bytes, "mtime_ns": output_path.stat().st_mtime_ns,
                  "sha256": output_sha.hexdigest(), "crc32": f"{output_crc & 0xffffffff:08x}"}
        return {**stats, "input": identity, "output": output,
                "numeric_source_types": len(source_types),
                "numeric_expansion_types": len(expanded_types),
                "numeric_expansion_type_keys": sorted(expanded_types), "examples": examples,
                "elapsed_seconds": time.monotonic() - started}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_raw_frequencies(path: Path, expected_identity: dict) -> tuple[Counter, dict]:
    """Read the verified full raw dictionary once, not the multi-GB corpus again."""
    before = path.stat()
    sha, crc, byte_count = hashlib.sha256(), 0, 0
    counts, metadata = Counter(), {}
    source_tokens = source_types = numeric_occurrences = numeric_types = expanded_tokens = 0
    numeric_output_types, examples = set(), []
    previous, header_seen = None, False
    with path.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            byte_count += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("#"):
                if ": " in line:
                    key, value = line[2:].split(": ", 1)
                    metadata[key] = json.loads(value)
                continue
            if not header_seen:
                if line != "rank\ttoken\tcount\tpercent\tcumulative_percent":
                    raise ValueError("Unrecognized raw frequency table header")
                header_seen = True
                continue
            rank_text, token, count_text, _, _ = line.split("\t")
            count = int(count_text)
            source_types += 1
            if int(rank_text) != source_types or count < 1:
                raise ValueError("Invalid raw frequency rank or count")
            ordering = (-count, token)
            if previous is not None and ordering <= previous:
                raise ValueError("Raw frequency table is not uniquely frequency-ranked")
            previous = ordering
            source_tokens += count
            pieces = split_numeric_token(token)
            for piece in pieces:
                counts[piece] += count
            if _is_numeric_token(token):
                numeric_occurrences += count
                numeric_types += 1
                expanded_tokens += count * len(pieces)
                numeric_output_types.update(pieces)
                if len(examples) < 12:
                    examples.append({"kind": "actual", "before": token,
                                     "after": list(pieces), "source_count": count})
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Raw frequency dictionary changed while reading")
    identity = {"bytes": byte_count, "mtime_ns": after.st_mtime_ns,
                "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}
    _require_identity(identity, expected_identity, path)
    if (not header_seen or metadata.get("status") != "complete"
            or metadata.get("row_limit") is not None
            or metadata.get("total_tokens") != source_tokens
            or metadata.get("unique_types") != source_types):
        raise ValueError("A complete, internally consistent raw frequency dictionary is required")
    return counts, {"input": identity, "metadata": metadata,
                    "input_tokens": source_tokens, "input_unique_types": source_types,
                    "numeric_source_occurrences": numeric_occurrences,
                    "numeric_source_types": numeric_types,
                    "numeric_expansion_tokens": expanded_tokens,
                    "numeric_expansion_types": len(numeric_output_types),
                    "numeric_expansion_type_keys": sorted(numeric_output_types), "examples": examples}


def _config(step5: dict, names: list[str]) -> dict:
    return {"schema_version": 1, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_tokenizer_fingerprint": tokenizer_fingerprint(step5["config"]),
            "python_version": platform.python_version(), "unicode_version": unicodedata.unidata_version,
            "source_directory": "tokens", "output_directory": "tokens_numeric",
            "expected_shards": names, "split_pos_tags": sorted(SPLIT_POS_TAGS),
            "fullwidth_digits": "ASCII normalization of U+FF10-U+FF19 only",
            "digits": "char.isdigit(): char/SN; every other numeric-form char: char/SW",
            "line_endings": "preserved exactly (production source CRLF)",
            "sentence_or_paragraph_filtering": False}


def _log(root: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    with (root / "reports/step6-numeric-progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _validate_step5(state: dict, names: list[str]) -> None:
    if set(state["completed"]) - set(names):
        raise ValueError("Unexpected Step 5 shard names")
    for name, row in state["completed"].items():
        if not all(key in row["output"] for key in IDENTITY_KEYS):
            raise ValueError(f"Incomplete Step 5 output identity: {name}")


def _finalize(root: Path, state: dict, step5: dict, names: list[str]) -> dict:
    reports = root / "reports"
    raw = json.loads((reports / "token-frequency-summary.json").read_text(encoding="utf-8"))
    if (raw.get("status") != "complete" or raw.get("shard_count") != len(names)
            or raw.get("tokenizer_fingerprint") != state["config"]["source_tokenizer_fingerprint"]):
        raise ValueError("Raw frequency summary does not match numeric source scope")
    source_shards = {row["file"]: row for row in raw["source_shards"]}
    if len(raw["source_shards"]) != len(names) or set(source_shards) != set(names):
        raise ValueError("Raw frequency source shards do not cover the numeric scope")
    for name in names:
        if source_shards[name]["input"] != state["completed"][name]["input"]:
            raise ValueError(f"Raw frequency source identity differs: {name}")
    totals = {key: sum(row[key] for row in state["completed"].values()) for key in NUMERIC_COUNT_KEYS}
    counts, numeric = _read_raw_frequencies(root / "token_frequencies.txt", raw["output"])
    if numeric["input_tokens"] != raw["tokens"] or numeric["input_unique_types"] != raw["unique_types"]:
        raise ValueError("Raw frequency dictionary totals differ from its summary")
    for key in ("input_tokens", "numeric_source_occurrences", "numeric_expansion_tokens"):
        if numeric[key] != totals[key]:
            raise ValueError(f"Weighted dictionary differs from materialized numeric files: {key}")
    if sum(counts.values()) != totals["tokens"] or counts[SPACE_TOKEN] != totals["space_tokens"]:
        raise ValueError("Weighted numeric frequencies do not equal actual numeric output totals")
    step5_summary = json.loads((reports / "step5-summary.json").read_text(encoding="utf-8"))
    for key in COUNT_KEYS:
        actual = totals["input_tokens"] if key == "tokens" else totals[key]
        if actual != step5_summary[key]:
            raise ValueError(f"Numeric source totals differ from Step 5 summary: {key}")
    table = write_frequency_table(
        root / "token_frequencies.numeric.txt", counts,
        {"status": "complete", "model_type": step5["config"].get("model_type"),
         "tokenizer_fingerprint": state["config"]["source_tokenizer_fingerprint"],
         "numeric_fingerprint": tokenizer_fingerprint(state["config"]),
         "source_directory": "tokens_numeric", "shard_count": len(names),
         "expected_shard_count": len(names), "source_shards": names,
         "raw_frequency_sha256": numeric["input"]["sha256"]},
    )
    summary = {"status": "complete", "config": state["config"],
               "tokenizer_fingerprint": state["config"]["source_tokenizer_fingerprint"],
               "numeric_fingerprint": tokenizer_fingerprint(state["config"]),
               "model_type": step5["config"].get("model_type"),
               "shard_count": len(names), "expected_shard_count": len(names),
               "source_shards": [{"file": name, **state["completed"][name]} for name in names],
               **totals, **{key: value for key, value in numeric.items() if key not in {"metadata", "input"}},
               "raw_frequency_input": numeric["input"], "unique_types": len(counts),
               "removed_unique_types": raw["unique_types"] - len(counts),
               "removed_unique_types_percent": (raw["unique_types"] - len(counts)) / max(raw["unique_types"], 1) * 100,
               "top30": table["top30"], "output": table["output"],
               "frequency_sum_matches_materialized_outputs": True,
               "sentence_count_unchanged": True, "paragraph_count_unchanged": True,
               "space_token_count_unchanged": True,
               "caveat": "Digit patterns become representable; arithmetic ability and generalization require training and evaluation."}
    _atomic_json(reports / "numeric-token-frequency-summary.json", summary)
    _atomic_json(reports / "step6-numeric-summary.json", summary)
    state["status"] = "complete"
    state["frequency_output"] = table["output"]
    state["raw_frequency_input"] = numeric["input"]
    _atomic_json(reports / "step6-numeric-state.json", state)
    _log(root, f"COMPLETE: {totals['input_tokens']:,} → {totals['tokens']:,} occurrences; "
               f"{raw['unique_types']:,} → {len(counts):,} types; all sentences/paragraphs preserved.")
    return summary


def run(root: str | Path = ROOT, watch: bool = False, resume: bool = False,
        poll_seconds: float = 12, expected_shards: list[str] | None = None) -> dict:
    root = Path(root)
    names = list(EXPECTED_SHARDS if expected_shards is None else expected_shards)
    if (not names or len(set(names)) != len(names)
            or any(re.fullmatch(r"part_\d{2}\.txt", name) is None for name in names)):
        raise ValueError("Expected unique part_NN.txt shard names")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    reports, output_dir = root / "reports", root / "tokens_numeric"
    reports.mkdir(parents=True, exist_ok=True)
    state_path, step5_path = reports / "step6-numeric-state.json", reports / "step5-state.json"
    with (reports / ".step6-numeric.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another numeric materialization process is running") from error
        while not step5_path.exists():
            if not watch:
                raise FileNotFoundError("Step 5 completion manifest is required")
            time.sleep(poll_seconds)
        step5 = json.loads(step5_path.read_text(encoding="utf-8"))
        _validate_step5(step5, names)
        if not watch and (step5.get("status") != "complete" or set(step5["completed"]) != set(names)):
            raise ValueError("Step 5 is incomplete; use --watch for completed shards")
        config = _config(step5, names)
        if state_path.exists():
            if not resume:
                raise FileExistsError("Numeric state exists; use --resume after verifying its configuration")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["config"] != config:
                raise ValueError("Resume refused: numeric code/configuration changed")
            if set(state["completed"]) - set(names):
                raise ValueError("Resume refused: unexpected numeric completed shards")
            for name, row in state["completed"].items():
                if (name not in step5["completed"] or row["input"] != step5["completed"][name]["output"]
                        or file_identity(root / "tokens" / name) != row["input"]
                        or file_identity(output_dir / name) != row["output"]):
                    raise ValueError(f"Resume refused: source/output identity changed: {name}")
            if state.get("status") == "complete":
                if (file_identity(root / "token_frequencies.numeric.txt") != state["frequency_output"]
                        or file_identity(root / "token_frequencies.txt") != state["raw_frequency_input"]):
                    raise ValueError("Resume refused: final frequency dictionary changed")
                return json.loads((reports / "numeric-token-frequency-summary.json").read_text(encoding="utf-8"))
        else:
            if (output_dir.exists() or (root / "token_frequencies.numeric.txt").exists()
                    or (reports / "numeric-token-frequency-summary.json").exists()
                    or (reports / "step6-numeric-summary.json").exists()):
                raise FileExistsError("Existing numeric outputs lack a matching completion state")
            state = {"config": config, "status": "running", "completed": {},
                     "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            output_dir.mkdir()
            _atomic_json(state_path, state)
        unexpected = {path.name for path in output_dir.glob("*.txt")} - set(state["completed"])
        if unexpected:
            raise ValueError(f"Refusing unrecorded numeric output files: {sorted(unexpected)}")
        for partial in output_dir.glob(".numeric-*.partial"):
            partial.unlink()
        last_wait = None
        while True:
            step5 = json.loads(step5_path.read_text(encoding="utf-8"))
            _validate_step5(step5, names)
            if _config(step5, names) != config:
                raise ValueError("Step 5 or numeric configuration changed while running")
            for name, row in state["completed"].items():
                if name not in step5["completed"] or step5["completed"][name]["output"] != row["input"]:
                    raise ValueError(f"Previously completed numeric source record changed: {name}")
                for directory, identity in ((root / "tokens", row["input"]), (output_dir, row["output"])):
                    current = (directory / name).stat()
                    if (current.st_size, current.st_mtime_ns) != (identity["bytes"], identity["mtime_ns"]):
                        raise ValueError(f"Previously completed numeric source/output changed: {directory / name}")
            for name in names:
                if name in state["completed"] or name not in step5["completed"]:
                    continue
                _log(root, f"Splitting {name}; {len(state['completed'])}/{len(names)} shards complete.")
                row = transform_shard(root / "tokens" / name, output_dir / name,
                                      step5["completed"][name]["output"], step5["completed"][name])
                state["completed"][name] = row
                state["status"] = "running"
                _atomic_json(state_path, state)
                totals = {key: sum(item[key] for item in state["completed"].values()) for key in NUMERIC_COUNT_KEYS}
                _atomic_json(reports / "step6-numeric-progress.json",
                             {"status": "partial", "shard_count": len(state["completed"]),
                              "expected_shard_count": len(names), **totals,
                              "source_tokenizer_fingerprint": config["source_tokenizer_fingerprint"]})
                _log(root, f"Completed {name}: {row['input_tokens']:,} → {row['tokens']:,} tokens; "
                           f"{row['sentences']:,} sentences unchanged.")
            all_done = len(state["completed"]) == len(names) and step5.get("status") == "complete"
            prerequisites = all((reports / name).exists()
                                for name in ("step5-summary.json", "token-frequency-summary.json"))
            if all_done and prerequisites and (root / "token_frequencies.txt").exists():
                summary = _finalize(root, state, step5, names)
                _atomic_json(reports / "step6-numeric-progress.json",
                             {"status": "complete", "shard_count": len(names), "expected_shard_count": len(names),
                              **{key: summary[key] for key in NUMERIC_COUNT_KEYS}})
                return summary
            if not watch:
                raise ValueError("Raw frequencies/Step 5 are incomplete; no final numeric dictionary published")
            waiting = "waiting_for_raw_frequencies" if all_done else "waiting_for_source"
            if waiting != last_wait:
                _log(root, waiting)
                last_wait = waiting
                state["status"] = waiting
                _atomic_json(state_path, state)
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=12)
    args = parser.parse_args()
    run(watch=args.watch, resume=args.resume, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
