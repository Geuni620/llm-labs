"""Count every actual serialized Kiwi form/POS token without vocabulary filtering.

Run after Step 5, or use --watch to follow completed immutable shards. Full
frequency output is published only after all 32 shards and Step 5's completion
summary agree. This is a frequency dictionary, not a top-K model vocabulary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zlib


ROOT = Path(__file__).resolve().parent
SPACE_TOKEN = "<sp>/WS"
EXPECTED_SHARDS = [f"part_{index:02d}.txt" for index in range(32)]
SCHEMA_VERSION = 1
IDENTITY_KEYS = ("bytes", "mtime_ns", "sha256", "crc32")
COUNT_KEYS = ("sentences", "paragraphs", "tokens", "space_tokens")


def tokenizer_fingerprint(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    before = path.stat()
    sha, crc = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while reading: {path}")
    return {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}


def _require_identity(actual: dict, expected: dict | None, path: Path) -> None:
    if expected is None:
        return
    for key in IDENTITY_KEYS:
        if key in expected and expected[key] != actual[key]:
            raise ValueError(f"Source identity mismatch ({key}): {path}")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     prefix=f".{path.name}-", suffix=".partial",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_counts(counts: Counter) -> None:
    for token, count in counts.items():
        if not isinstance(token, str) or not token or any(char.isspace() for char in token):
            raise ValueError(f"Invalid serialized token key: {token!r}")
        form, separator, tag = token.rpartition("/")
        if not separator or not form or not tag:
            raise ValueError(f"Invalid form/POS token key: {token!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"Invalid token frequency for {token!r}: {count!r}")


def count_shard(path: str | Path, expected_identity: dict | None = None) -> tuple[Counter, dict]:
    """Count exact ASCII-space-delimited keys while hashing the actual bytes."""
    path = Path(path)
    before = path.stat()
    counts = Counter()
    stats = dict.fromkeys(COUNT_KEYS, 0)
    sha, crc, byte_count = hashlib.sha256(), 0, 0
    in_paragraph = False
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            byte_count += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                in_paragraph = False
                continue
            tokens = line.split(" ")
            if "" in tokens:
                raise ValueError(f"Malformed token separator at {path}:{line_number}")
            counts.update(tokens)
            stats["sentences"] += 1
            stats["tokens"] += len(tokens)
            stats["space_tokens"] += tokens.count(SPACE_TOKEN)
            if not in_paragraph:
                stats["paragraphs"] += 1
                in_paragraph = True
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while counting: {path}")
    identity = {"bytes": byte_count, "mtime_ns": after.st_mtime_ns,
                "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}
    _require_identity(identity, expected_identity, path)
    # Validate unique keys once, rather than rechecking every repeated token.
    _validate_counts(counts)
    if sum(counts.values()) != stats["tokens"] or counts[SPACE_TOKEN] != stats["space_tokens"]:
        raise RuntimeError(f"Internal frequency consistency failure: {path}")
    return counts, {**stats, "input": identity, "unique_types": len(counts)}


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1, newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_count_shard(
    path: str | Path,
    expected_identity: dict,
    tokenizer_fingerprint: str,
    cache_dir: str | Path,
) -> tuple[Counter, dict, bool]:
    """Reuse only a cache tied to both verified source bytes and tokenizer config."""
    path, cache_dir = Path(path), Path(cache_dir)
    cache_path = cache_dir / f"{path.stem}.json.gz"
    current = None
    if cache_path.exists():
        # Hashing cached sources is cheap compared with reparsing every token.
        current = file_identity(path)
        _require_identity(current, expected_identity, path)
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if (payload["schema_version"] == SCHEMA_VERSION
                    and payload["tokenizer_fingerprint"] == tokenizer_fingerprint
                    and payload["source"] == current):
                counts = Counter(payload["counts"])
                _validate_counts(counts)
                stats = payload["stats"]
                if (stats["input"] == current and sum(counts.values()) == stats["tokens"]
                        and counts[SPACE_TOKEN] == stats["space_tokens"]
                        and len(counts) == stats["unique_types"]):
                    return counts, {**stats, "cache": {"path": str(cache_path), **file_identity(cache_path)}}, True
        except (OSError, EOFError, UnicodeError, ValueError, KeyError, TypeError):
            # Corrupt or incompatible caches are regenerated from verified source.
            pass
    counts, stats = count_shard(path, expected_identity)
    _write_cache(cache_path, {"schema_version": SCHEMA_VERSION,
                             "tokenizer_fingerprint": tokenizer_fingerprint,
                             "source": stats["input"], "stats": stats,
                             "counts": dict(counts)})
    return counts, {**stats, "cache": {"path": str(cache_path), **file_identity(cache_path)}}, False


def write_frequency_table(
    path: str | Path,
    counts: Counter,
    metadata: dict,
    limit: int | None = None,
    allow_replace: bool = False,
) -> dict:
    """Write all types by descending count, lexical ties, with exact count sums.

    A limited table is explicitly partial. Percent denominators always include
    all counted tokens, including those outside a partial top-1,000 preview.
    Existing different final tables are preserved; identical reruns are safe.
    """
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    _validate_counts(counts)
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    meta = {**metadata, "total_tokens": total, "unique_types": len(counts),
            "row_limit": limit, "status": "partial" if limit is not None else metadata.get("status", "complete"),
            "percent_denominator": "all tokens in the covered source shards",
            "includes": "all form/POS keys, including <sp>/WS, numbers, punctuation and function morphemes"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    cumulative, rows, top30 = 0, 0, []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{path.name}-", suffix=".partial",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("# Kiwi morphology/POS token frequency dictionary\n")
            for key, value in meta.items():
                stream.write(f"# {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}\n")
            stream.write("rank\ttoken\tcount\tpercent\tcumulative_percent\n")
            for rank, (token, count) in enumerate(ranked, 1):
                if limit is not None and rank > limit:
                    break
                cumulative += count
                percent = count / total * 100
                cumulative_percent = cumulative / total * 100
                stream.write(f"{rank}\t{token}\t{count}\t{percent:.9f}\t{cumulative_percent:.9f}\n")
                rows += 1
                if rank <= 30:
                    top30.append({"rank": rank, "token": token, "count": count,
                                  "percent": percent, "cumulative_percent": cumulative_percent})
            stream.flush()
            os.fsync(stream.fileno())
        identity = file_identity(temporary)
        if path.exists() and not allow_replace:
            existing = file_identity(path)
            if existing["sha256"] != identity["sha256"] or existing["bytes"] != identity["bytes"]:
                raise FileExistsError(f"Refusing to overwrite a different frequency table: {path}")
            temporary.unlink()
            temporary = None
            identity = existing
        else:
            os.replace(temporary, path)
            temporary = None
            identity["mtime_ns"] = path.stat().st_mtime_ns
        return {"tokens": total, "unique_types": len(counts), "rows": rows,
                "written_tokens": cumulative, "top30": top30, "output": identity}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_state(path: Path, expected_names: list[str]) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    completed = state["completed"]
    unexpected = set(completed) - set(expected_names)
    if unexpected:
        raise ValueError(f"Unexpected Step 5 completed shard names: {sorted(unexpected)}")
    for name, shard in completed.items():
        if not all(key in shard["output"] for key in IDENTITY_KEYS):
            raise ValueError(f"Incomplete output identity in Step 5 manifest: {name}")
    return state


def _log(root: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    with (root / "reports/token-frequency-progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run(
    root: str | Path = ROOT,
    watch: bool = False,
    poll_seconds: float = 12,
    preview: bool = True,
    expected_shards: list[str] | None = None,
) -> dict:
    root = Path(root)
    names = list(EXPECTED_SHARDS if expected_shards is None else expected_shards)
    if not names or len(set(names)) != len(names):
        raise ValueError("expected_shards must contain unique shard names")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    reports, cache_dir = root / "reports", root / "frequency-cache"
    reports.mkdir(parents=True, exist_ok=True)
    state_path = reports / "step5-state.json"
    summary_path = reports / "step5-summary.json"
    processed, totals, counts = {}, dict.fromkeys(COUNT_KEYS, 0), Counter()
    fingerprint, config = None, None
    started = time.monotonic()
    logged_wait = False
    while True:
        if not state_path.exists():
            if not watch:
                raise FileNotFoundError("Step 5 completion manifest is required")
            if not logged_wait:
                _log(root, "Waiting for Step 5 manifest; no partial token files will be read.")
                logged_wait = True
            time.sleep(poll_seconds)
            continue
        state = _load_state(state_path, names)
        current_fingerprint = tokenizer_fingerprint(state["config"])
        if fingerprint is None:
            fingerprint, config = current_fingerprint, state["config"]
        elif current_fingerprint != fingerprint:
            raise ValueError("Step 5 tokenizer fingerprint changed while counting")
        if not watch and (set(state["completed"]) != set(names)
                          or state.get("status") != "complete" or not summary_path.exists()):
            raise ValueError("Full Step 5 completion is required; use --watch to follow completed shards")
        for name, prior in processed.items():
            if name not in state["completed"] or state["completed"][name]["output"] != prior["input"]:
                raise ValueError(f"A previously counted completion record changed: {name}")
            current = (root / "tokens" / name).stat()
            if (current.st_size, current.st_mtime_ns) != (prior["input"]["bytes"], prior["input"]["mtime_ns"]):
                raise ValueError(f"A previously counted token file changed: {name}")
        for name in names:
            if name in processed or name not in state["completed"]:
                continue
            shard = state["completed"][name]
            shard_counts, stats, hit = load_or_count_shard(
                root / "tokens" / name, shard["output"], fingerprint, cache_dir,
            )
            for key in COUNT_KEYS:
                if stats[key] != shard[key]:
                    raise ValueError(f"Token count differs from Step 5 ({name}, {key})")
            counts.update(shard_counts)
            del shard_counts
            processed[name] = stats
            for key in COUNT_KEYS:
                totals[key] += stats[key]
            _log(root, f"Counted {name} ({'verified cache' if hit else 'source scan'}): "
                       f"{stats['tokens']:,} tokens; {len(processed)}/{len(names)} shards, "
                       f"{totals['tokens']:,} total tokens, {len(counts):,} types.")
            progress = {"status": "partial", "tokenizer_fingerprint": fingerprint,
                        "model_type": config.get("model_type"), **totals,
                        "unique_types": len(counts), "shard_count": len(processed),
                        "expected_shard_count": len(names),
                        "processed_sources": [{"file": item, **processed[item]} for item in names if item in processed]}
            _atomic_json(reports / "token-frequency-state.json", progress)
            if preview:
                preview_metadata = {"status": "partial", "model_type": config.get("model_type"),
                                    "tokenizer_fingerprint": fingerprint,
                                    "shard_count": len(processed), "expected_shard_count": len(names),
                                    "source_shards": [item for item in names if item in processed]}
                write_frequency_table(reports / "token_frequencies.partial.all.txt", counts,
                                      preview_metadata, allow_replace=True)
                write_frequency_table(reports / "token_frequencies.partial.txt", counts,
                                      preview_metadata,
                                      limit=1000, allow_replace=True)
        if len(processed) == len(names) and state.get("status") == "complete" and summary_path.exists():
            break
        if not watch:
            raise ValueError("Step 5 is not complete; no final frequency table was published")
        time.sleep(poll_seconds)
    step5 = json.loads(summary_path.read_text(encoding="utf-8"))
    if (step5.get("status") != "complete" or step5["shard_count"] != len(names)
            or tokenizer_fingerprint(step5["config"]) != fingerprint):
        raise ValueError("Step 5 final summary does not match the counted source scope")
    if "shards" in step5:
        summary_shards = {shard["file"]: shard for shard in step5["shards"]}
        if len(step5["shards"]) != len(names) or set(summary_shards) != set(names):
            raise ValueError("Step 5 final summary shard identities do not cover the full scope")
        for name in names:
            if summary_shards[name]["output"] != processed[name]["input"]:
                raise ValueError(f"Step 5 final summary source identity differs: {name}")
    for key in COUNT_KEYS:
        if totals[key] != step5[key]:
            raise ValueError(f"Frequency totals differ from Step 5 summary: {key}")
    if sum(counts.values()) != totals["tokens"] or counts[SPACE_TOKEN] != totals["space_tokens"]:
        raise RuntimeError("Global frequency sum or whitespace-token count does not match Step 5")
    table = write_frequency_table(root / "token_frequencies.txt", counts,
                                  {"status": "complete", "model_type": config.get("model_type"),
                                   "kiwipiepy_version": config.get("kiwipiepy_version"),
                                   "kiwipiepy_model_version": config.get("kiwipiepy_model_version"),
                                   "tokenizer_fingerprint": fingerprint,
                                   "source_directory": "tokens", "shard_count": len(names),
                                   "expected_shard_count": len(names), "source_shards": names})
    summary = {"status": "complete", "model_type": config.get("model_type"),
               "tokenizer_fingerprint": fingerprint, "config": config,
               "counter_runtime": {"python_version": platform.python_version(), "platform": platform.platform(),
                                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "shard_count": len(names), "source_shards": [{"file": name, **processed[name]} for name in names],
               **totals, "unique_types": len(counts), "top30": table["top30"],
               "frequency_sum_matches_step5": True, "space_token_count_matches_step5": True,
               "output": table["output"], "elapsed_seconds": time.monotonic() - started}
    _atomic_json(reports / "token-frequency-summary.json", summary)
    _atomic_json(reports / "token-frequency-state.json", {**summary, "expected_shard_count": len(names)})
    _log(root, f"COMPLETE: {totals['tokens']:,} tokens, {len(counts):,} types; token_frequencies.txt contains every type.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Follow completed Step 5 shards until full scope is complete")
    parser.add_argument("--poll-seconds", type=float, default=12)
    parser.add_argument("--no-preview", action="store_true", help="Skip explicitly partial frequency tables")
    args = parser.parse_args()
    run(watch=args.watch, poll_seconds=args.poll_seconds, preview=not args.no_preview)


if __name__ == "__main__":
    main()
