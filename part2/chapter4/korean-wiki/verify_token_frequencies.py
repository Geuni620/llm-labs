"""Verify the complete frequency table against digest-bound shard counts.

This checks every cached type/count and every final TSV row. It reuses the
successful Step 5 all-line/hash verification instead of splitting all token
occurrences again. Source contexts are bounded examples, not a second census.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
from itertools import islice, zip_longest
import json
import math
from pathlib import Path
import re
import time
import zlib


ROOT = Path(__file__).resolve().parent
SPACE_TOKEN = "<sp>/WS"
HEADER = "rank\ttoken\tcount\tpercent\tcumulative_percent"
PERCENT_TEXT = re.compile(r"[0-9]+\.[0-9]{9}\Z")


def valid_rounded_percent(text, numerator, denominator):
    """Nine decimal places with an explicit binary64 arithmetic allowance.

    Counts remain exact integers. The writer evaluates numerator/denominator*100
    using two float operations before formatting. Near a decimal half-way point,
    the stored value can exceed the exact-rational half-unit by a few float ULPs.
    Four ULPs conservatively cover those two operations; they are not a relative
    tolerance and are far smaller than one displayed 1e-9 percentage-point unit.
    https://docs.python.org/3.13/library/math.html#math.ulp
    """
    if denominator <= 0 or not 0 <= numerator <= denominator or not PERCENT_TEXT.fullmatch(text):
        return False
    try:
        actual = Decimal(text)
        expected = Decimal(numerator) * 100 / Decimal(denominator)
        binary_allowance = Decimal.from_float(4 * math.ulp(float(expected)))
        tolerance = Decimal("0.0000000005") + binary_allowance
        return actual.is_finite() and 0 <= actual <= 100 and abs(actual - expected) <= tolerance
    except (InvalidOperation, ValueError, OverflowError):
        return False


def verify_table(path, expected_counts=None, total_tokens=None):
    """Check every rank/count/key, sort order, and nine-decimal percentage."""
    errors, metadata, seen, top = [], {}, set(), []
    rows = cumulative = 0
    prior = None
    found_header = False
    digest, crc, size = hashlib.sha256(), 0, 0
    if total_tokens is None and expected_counts is not None:
        total_tokens = sum(expected_counts.values())

    def error(kind, line, detail):
        if len(errors) < 40:
            errors.append({"kind": kind, "line": line, "detail": detail})

    with Path(path).open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            digest.update(raw)
            crc = zlib.crc32(raw, crc)
            size += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("#"):
                key, separator, value = line.lstrip("# ").partition(": ")
                if separator:
                    try:
                        metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[key] = value
                continue
            if not line:
                continue
            if not found_header:
                if line != HEADER:
                    error("header", line_no, f"Expected {HEADER!r}")
                found_header = True
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                error("columns", line_no, f"Expected five columns, got {len(fields)}")
                continue
            rank_text, token, count_text, percent_text, cumulative_text = fields
            try:
                rank, count = int(rank_text), int(count_text)
            except ValueError:
                error("integer", line_no, "Rank/count must be integers")
                continue
            rows += 1
            if rank != rows:
                error("rank", line_no, f"Expected {rows}, got {rank}")
            if count <= 0:
                error("count", line_no, "Counts must be positive")
            form, slash, tag = token.rpartition("/")
            if not slash or not form or not tag or any(character.isspace() for character in token):
                error("token_format", line_no, token[:100])
            if token in seen:
                error("duplicate_token", line_no, token)
            seen.add(token)
            order = (-count, token)
            if prior is not None and order < prior:
                error("sort", line_no, "Expected descending count and ascending exact token for ties")
            prior = order
            if expected_counts is not None and expected_counts.get(token) != count:
                error("cached_count", line_no, f"{token}: cache={expected_counts.get(token)}, table={count}")
            cumulative += count
            if total_tokens and total_tokens > 0:
                for kind, text, numerator in (("percent", percent_text, count),
                                               ("cumulative_percent", cumulative_text, cumulative)):
                    if not valid_rounded_percent(text, numerator, total_tokens):
                        error(kind, line_no, f"Invalid rounded percentage: {text}")
            if len(top) < 30:
                top.append({"rank": rank, "token": token, "count": count,
                            "percent": percent_text, "cumulative_percent": cumulative_text})
    if not found_header:
        error("header", 0, "Missing TSV header")
    if total_tokens is not None and cumulative != total_tokens:
        error("sum", 0, f"Expected {total_tokens}, got {cumulative}")
    missing_types = 0
    if expected_counts is not None:
        missing_types = sum(token not in seen for token in expected_counts)
        if missing_types:
            error("missing_types", 0, f"{missing_types} cached token types are absent")
        if rows != len(expected_counts):
            error("type_count", 0, f"Expected {len(expected_counts)} rows, got {rows}")
    return {"rows": rows, "unique_types": len(seen), "sum_counts": cumulative,
            "missing_types": missing_types, "errors": errors, "metadata": metadata, "top30": top,
            "percentage_validation": "Exact Decimal ratio, half a 1e-9 display unit plus four binary64 ULPs; ranks/counts/hashes remain exact",
            "output": {"bytes": size, "sha256": digest.hexdigest(), "crc32": f"{crc:08x}"}}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_matches(expected, actual, fields=("bytes", "sha256", "crc32")):
    return all(expected.get(field) == actual.get(field) for field in fields)


def sample_source_contexts(root, names, top_rows, limit_per_shard=256):
    """Find bounded original-sentence examples containing frequent tokens."""
    wanted = {row["token"] for row in top_rows[:15]}
    results, inspected = [], 0
    for name in names:
        source = root / "sentences_dedup" / name
        tokens = root / "tokens" / name
        if not source.exists():
            continue
        with source.open(encoding="utf-8") as source_stream, tokens.open(encoding="utf-8") as token_stream:
            for line_no, pair in enumerate(islice(zip_longest(source_stream, token_stream), limit_per_shard), 1):
                original, token_line = pair
                inspected += 1
                if original is None or token_line is None:
                    continue
                original, token_line = original.rstrip("\r\n"), token_line.rstrip("\r\n")
                present = wanted.intersection(token_line.split(" "))
                for token in sorted(present):
                    results.append({"token": token, "file": name, "line": line_no,
                                    "original": original, "tokens": token_line})
                    wanted.remove(token)
                if not wanted:
                    break
        if not wanted:
            break
    return {"maximum_physical_lines_per_shard": limit_per_shard, "inspected_line_pairs": inspected,
            "examples": results, "not_found_in_bounded_sample": sorted(wanted)}


def verify_outputs(root=ROOT, expected_shards=None):
    root = Path(root)
    names = list(expected_shards) if expected_shards is not None else [f"part_{index:02d}.txt" for index in range(32)]
    reports = root / "reports"
    summary = _read_json(reports / "token-frequency-summary.json")
    step5 = _read_json(reports / "step5-summary.json")
    state = _read_json(reports / "step5-state.json")
    token_check = _read_json(reports / "step5-verification.json")
    errors, aggregate, shard_reports = [], Counter(), []

    def error(kind, detail, file=None):
        if len(errors) < 60:
            errors.append({"kind": kind, "file": file, "detail": detail})

    if summary.get("status") != "complete":
        error("scope", "Frequency summary is not complete")
    if step5.get("status") != "complete" or state.get("status") != "complete":
        error("scope", "Step 5 completion records are not complete")
    if token_check.get("scope") != "full" or not token_check.get("passed"):
        error("scope", "Requires a passed full independent Step 5 verification")
    if summary.get("config") != step5.get("config") or step5.get("config") != state.get("config"):
        error("model_config", "Frequency and tokenization configurations differ")
    fingerprint = hashlib.sha256(json.dumps(summary.get("config"), ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    if fingerprint != summary.get("tokenizer_fingerprint"):
        error("model_fingerprint", "Tokenizer fingerprint is not the digest of its recorded configuration")
    entries = {value["file"]: value for value in summary.get("source_shards", [])}
    writer_entries = {value["file"]: value for value in step5.get("shards", [])}
    verified_entries = {value["file"]: value for value in token_check.get("shards", [])}
    if set(entries) != set(names) or summary.get("shard_count") != len(names):
        error("inventory", "Frequency source shards differ from the exact expected set")
    if set(state.get("completed", {})) != set(names) or set(writer_entries) != set(names) or set(verified_entries) != set(names):
        error("inventory", "Step 5 manifest/summary/verification shard sets disagree")

    for name in names:
        entry = entries.get(name)
        writer = writer_entries.get(name)
        verified = verified_entries.get(name)
        completed = state.get("completed", {}).get(name)
        if not all((entry, writer, verified, completed)):
            continue
        expected = completed["output"]
        if not (_identity_matches(expected, entry.get("input", {}))
                and _identity_matches(expected, writer.get("output", {}))
                and _identity_matches(expected, verified.get("output", {}))):
            error("source_identity", "Digest-bound token sources disagree", name)
        current = (root / "tokens" / name).stat()
        if (current.st_size, current.st_mtime_ns) != (expected.get("bytes"), expected.get("mtime_ns")):
            error("source_changed", "Token file metadata changed after the full hash verification", name)
        cache_identity = entry.get("cache", {})
        cache_path = Path(cache_identity.get("path", f"frequency-cache/{Path(name).stem}.json.gz"))
        if not cache_path.is_absolute():
            cache_path = root / cache_path
        data = cache_path.read_bytes()
        actual_cache = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "crc32": f"{zlib.crc32(data):08x}"}
        if not _identity_matches(cache_identity, actual_cache):
            error("cache_identity", "Compressed cache bytes/digests differ from final summary", name)
        cached = json.loads(gzip.decompress(data))
        if cached.get("tokenizer_fingerprint") != summary.get("tokenizer_fingerprint"):
            error("cache_model", "Cache tokenizer fingerprint differs", name)
        if not _identity_matches(cached.get("source", {}), expected):
            error("cache_source", "Cache source bytes/digests differ", name)
        values = cached.get("counts", {})
        if not isinstance(values, dict) or any(type(value) is not int or value < 1 for value in values.values()):
            error("cache_counts", "Cache must contain positive integer exact-token counts", name)
            continue
        total = sum(values.values())
        for field, actual in (("tokens", total), ("space_tokens", values.get(SPACE_TOKEN, 0))):
            if actual != writer.get(field) or actual != entry.get(field) or actual != cached.get("stats", {}).get(field):
                error("cache_total", f"{field} differs between cache and tokenization counts", name)
        aggregate.update(values)
        shard_reports.append({"file": name, "types": len(values), "tokens": total,
                              "source_sha256": expected["sha256"], "cache_sha256": actual_cache["sha256"]})

    total = sum(aggregate.values())
    if total != step5.get("tokens") or total != summary.get("tokens"):
        error("token_total", "All cached counts do not add up to Step 5 and frequency totals")
    if len(aggregate) != summary.get("unique_types"):
        error("unique_types", "Merged exact token-type count differs from frequency summary")
    if aggregate.get(SPACE_TOKEN, 0) != step5.get("space_tokens") or aggregate.get(SPACE_TOKEN, 0) != summary.get("space_tokens"):
        error("space_tokens", "Whitespace-token counts differ")
    table = verify_table(root / "token_frequencies.txt", aggregate, total_tokens=total)
    errors.extend(table["errors"])
    metadata = table["metadata"]
    if (metadata.get("status") != "complete" or metadata.get("row_limit") is not None
            or metadata.get("shard_count") != len(names)
            or metadata.get("source_shards") != names
            or metadata.get("total_tokens") != total
            or metadata.get("unique_types") != len(aggregate)):
        error("table_scope", "Table metadata does not describe the complete, untruncated source scope")
    if not _identity_matches(summary.get("output", {}), table["output"]):
        error("table_identity", "Final table bytes/digests differ from its summary")
    for row in table["top30"]:
        listed = next((value for value in summary.get("top30", []) if value.get("rank") == row["rank"]), None)
        if listed is None or listed.get("token") != row["token"] or listed.get("count") != row["count"]:
            error("top30", f"Rank {row['rank']} differs from summary")
    pos_counts = Counter()
    for token, count in aggregate.items():
        _, _, tag = token.rpartition("/")
        pos_counts[tag] += count
    contexts = sample_source_contexts(root, names, table["top30"])
    return {"scope": "full", "passed": not errors, "shard_count": len(shard_reports),
            "tokens": total, "unique_types": len(aggregate), "space_tokens": aggregate.get(SPACE_TOKEN, 0),
            "errors": errors, "table": table, "shards": shard_reports,
            "pos_token_totals": dict(pos_counts.most_common()), "source_contexts": contexts,
            "validation_scope": "Every final TSV row and every cached type/count verified. Cache digests bind counts to the completed token sources; Step 5 full independent line/SHA verification is reused. Token occurrences were not fully recounted a second time."}


def main():
    started = time.monotonic()
    report = verify_outputs()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    path = ROOT / "reports/token-frequency-verification.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['shard_count']} shards, "
          f"{report['tokens']:,} tokens, {report['unique_types']:,} types, {len(report['errors'])} errors")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
