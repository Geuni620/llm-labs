"""Build an exact, compact rank-frequency chart from the counted Kiwi tokens.

The graph covers every serialized form/POS type, rather than just frequent
words. Equal-frequency ranks are stored as [last_rank, count] runs, preserving
the frequency of every integer rank without embedding millions of token names.

By default the final, complete frequency table and completion summary are
required. --partial permits an untruncated table for completed shards only;
the top-1,000 preview is deliberately rejected because it cannot show the
covered vocabulary's whole rank distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zlib


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "reports/token-frequency-plot.template.html"
PLACEHOLDER = "__TOKEN_FREQUENCY_PLOT_DATA__"
HEADER = "rank\ttoken\tcount\tpercent\tcumulative_percent"
POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
MAX_HTML_BYTES = 1_000_000


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _read_table(path: Path) -> tuple[dict, dict]:
    metadata, runs, top_tokens = {}, [], []
    sha, crc, byte_count = hashlib.sha256(), 0, 0
    rank, total, previous_count, previous_token = 0, 0, None, None
    header_seen = False
    before = path.stat()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            byte_count += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("#") and not header_seen:
                key, separator, value = line[1:].strip().partition(":")
                if separator:
                    if key in metadata:
                        raise ValueError(f"Duplicate metadata field: {key}")
                    metadata[key] = json.loads(value.strip())
                continue
            if not header_seen:
                if line != HEADER:
                    raise ValueError(f"Unexpected frequency table header at line {line_number}")
                header_seen = True
                continue
            columns = line.split("\t")
            if len(columns) != 5 or not POSITIVE_INTEGER.fullmatch(columns[0]) or not POSITIVE_INTEGER.fullmatch(columns[2]):
                raise ValueError(f"Malformed frequency row at line {line_number}")
            row_rank, token, count = int(columns[0]), columns[1], int(columns[2])
            if row_rank != rank + 1:
                raise ValueError(f"Non-contiguous frequency rank at line {line_number}")
            if not token or any(character.isspace() for character in token):
                raise ValueError(f"Invalid serialized token at line {line_number}")
            if previous_count is not None and (count > previous_count or (count == previous_count and token <= previous_token)):
                raise ValueError(f"Frequency rows are not count-descending with lexical ties at line {line_number}")
            rank, total = row_rank, total + count
            if count != previous_count:
                runs.append([rank, count])
            else:
                runs[-1][0] = rank
            if rank <= 30:
                top_tokens.append([rank, token, count])
            previous_count, previous_token = count, token
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Frequency table changed while the graph was being built")
    if not header_seen or not runs:
        raise ValueError("Frequency table must contain at least one token type")
    return metadata, {"vocabulary_size": rank, "total_tokens": total,
                      "runs": runs, "top_tokens": top_tokens,
                      "minimum_count": runs[-1][1], "maximum_count": runs[0][1],
                      "frequency_sha256": sha.hexdigest(), "frequency_crc32": f"{crc & 0xffffffff:08x}",
                      "frequency_bytes": byte_count}


def _summary_source_names(summary: dict, partial: bool) -> list[str]:
    sources = summary.get("processed_sources" if partial else "source_shards")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Summary does not identify the covered source shards")
    names = [item if isinstance(item, str) else item.get("file") for item in sources]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Summary source shard names must be unique nonempty strings")
    return names


def singleton_statistics(runs: list[list[int]], total_tokens: int) -> dict:
    """Read the frequency-one tail exactly from the final compressed run.

    The two percentages answer different questions: the fraction of token
    types seen once, and their fraction of all actual token occurrences.
    """
    _positive_integer(total_tokens, "total_tokens")
    if not runs:
        raise ValueError("Frequency runs must contain at least one token type")
    vocabulary_size = _positive_integer(runs[-1][0], "vocabulary_size")
    _positive_integer(runs[-1][1], "minimum_frequency")
    previous_end = runs[-2][0] if len(runs) > 1 else 0
    singleton_types = vocabulary_size - previous_end if runs[-1][1] == 1 else 0
    if singleton_types < 0 or singleton_types > total_tokens:
        raise ValueError("Invalid frequency-one tail boundaries")
    return {"singleton_types": singleton_types,
            "singleton_type_percent": singleton_types / vocabulary_size * 100,
            "singleton_occurrence_percent": singleton_types / total_tokens * 100,
            "singleton_rank_interval": [vocabulary_size - singleton_types + 1, vocabulary_size] if singleton_types else None}


def load_plot_data(frequency: str | Path, summary_path: str | Path, allow_partial: bool = False) -> dict:
    frequency, summary_path = Path(frequency), Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata, data = _read_table(frequency)
    status = metadata.get("status")
    if status not in {"complete", "partial"} or summary.get("status") != status:
        raise ValueError("Frequency table and summary completion status must agree")
    partial = status == "partial"
    if partial and not allow_partial:
        raise ValueError("Complete frequency results are required; use --partial explicitly for completed shards")
    if "row_limit" not in metadata or metadata["row_limit"] is not None:
        raise ValueError("A truncated top-N preview cannot represent the full rank distribution; supply the untruncated all-types table")
    for meta_key, data_key, summary_key in [("unique_types", "vocabulary_size", "unique_types"), ("total_tokens", "total_tokens", "tokens")]:
        _positive_integer(metadata.get(meta_key), meta_key)
        _positive_integer(summary.get(summary_key), f"summary.{summary_key}")
        if metadata[meta_key] != data[data_key] or summary[summary_key] != data[data_key]:
            raise ValueError(f"Frequency row totals do not match {meta_key}")
    shard_count = _positive_integer(metadata.get("shard_count"), "shard_count")
    expected = _positive_integer(metadata.get("expected_shard_count"), "expected_shard_count")
    if shard_count > expected or summary.get("shard_count") != shard_count:
        raise ValueError("Frequency table and summary shard counts do not agree")
    if summary.get("expected_shard_count", expected) != expected:
        raise ValueError("Frequency table and summary expected shard counts do not agree")
    names = metadata.get("source_shards")
    if not isinstance(names, list) or len(names) != shard_count or names != _summary_source_names(summary, partial):
        raise ValueError("Frequency table and summary source shard scope do not agree")
    if not partial and (shard_count != expected or expected != 32):
        raise ValueError("The final graph requires all 32 source shards")
    if metadata.get("model_type") != "cong-global" or summary.get("model_type") != metadata["model_type"]:
        raise ValueError("The graph requires the requested cong-global tokenizer model")
    fingerprint = metadata.get("tokenizer_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint or fingerprint != summary.get("tokenizer_fingerprint"):
        raise ValueError("Frequency table and summary tokenizer fingerprints do not agree")
    output = summary.get("output")
    if not partial:
        if summary.get("frequency_sum_matches_step5") is not True or summary.get("space_token_count_matches_step5") is not True:
            raise ValueError("Final frequency summary is missing Step 5 count consistency checks")
        if not isinstance(output, dict) or not all(key in output for key in ("sha256", "bytes", "crc32")):
            raise ValueError("Final frequency summary is missing the table's verified file identity")
    if isinstance(output, dict):
        for identity_key, data_key in [("sha256", "frequency_sha256"), ("bytes", "frequency_bytes"), ("crc32", "frequency_crc32")]:
            if identity_key in output and output[identity_key] != data[data_key]:
                raise ValueError(f"Frequency table identity mismatch: {identity_key}")
    # Each run has a distinct positive count. Its minimum possible sum is
    # n(n+1)/2, bounding the exact representation independently of type count.
    if len(data["runs"]) * (len(data["runs"]) + 1) // 2 > data["total_tokens"]:
        raise RuntimeError("Invalid frequency-run compression")
    return {"schema_version": 1, "status": status, "model_type": metadata["model_type"],
            "shard_count": shard_count, "expected_shard_count": expected,
            "source_shards": names, "tokenizer_fingerprint": fingerprint, **data,
            **singleton_statistics(data["runs"], data["total_tokens"])}


def frequency_at_rank(runs: list[list[int]], rank: int) -> int:
    """Exact lower-bound lookup, mirrored by the browser's tooltip lookup."""
    if not runs or isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or rank > runs[-1][0]:
        raise ValueError("Rank is outside the plotted vocabulary")
    low, high = 0, len(runs)
    while low < high:
        middle = (low + high) // 2
        if runs[middle][0] < rank:
            low = middle + 1
        else:
            high = middle
    return runs[low][1]


def build_plot(frequency: str | Path, summary: str | Path, output: str | Path, allow_partial: bool = False, template: str | Path = TEMPLATE) -> dict:
    data = load_plot_data(frequency, summary, allow_partial)
    markup = Path(template).read_text(encoding="utf-8")
    if markup.count(PLACEHOLDER) != 1:
        raise ValueError("Plot template must contain exactly one data placeholder")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    markup = markup.replace(PLACEHOLDER, payload)
    if len(markup.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError("Exact plot exceeds the 1 MB visualization limit")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{output.name}-", suffix=".partial", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(markup)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": data["status"], "vocabulary_size": data["vocabulary_size"],
            "total_tokens": data["total_tokens"], "exact_frequency_runs": len(data["runs"]),
            "singleton_types": data["singleton_types"],
            "singleton_type_percent": data["singleton_type_percent"],
            "singleton_occurrence_percent": data["singleton_occurrence_percent"],
            "singleton_rank_interval": data["singleton_rank_interval"],
            "shards": f"{data['shard_count']}/{data['expected_shard_count']}",
            "html_bytes": output.stat().st_size, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, default=ROOT / "token_frequencies.txt")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/token-frequency-summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/token-frequency-plot.html")
    parser.add_argument("--partial", action="store_true", help="Allow an explicitly partial, untruncated completed-shard table")
    args = parser.parse_args()
    if args.partial and args.output == ROOT / "reports/token-frequency-plot.html":
        args.output = ROOT / "reports/token-frequency-plot.partial.html"
    print(json.dumps(build_plot(args.frequency, args.summary, args.output, args.partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
