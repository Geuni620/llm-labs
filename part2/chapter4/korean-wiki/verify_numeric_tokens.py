"""Independently verify the separately materialized numeric-token corpus.

All selected bytes, sentence/paragraph boundaries, occurrence totals, WS counts,
and recorded SHA256/CRC32 identities are checked. Transformation correctness is
checked on 32 deterministic reservoir-sampled actual sentences per shard using
the supplied lecture's numeric rule, independently written here. It is sampled
verification, not a claim that every token's transformation was re-executed.

Default scope is all 32 completed shards. --partial explicitly checks only the
immutable completed numeric shards and never labels that result full-corpus.
No Kiwi model is loaded, no source data is changed, and no corpus Counter is built.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import time
import unicodedata
import zlib


ROOT = Path(__file__).resolve().parent
EXPECTED_SHARDS = [f"part_{index:02d}.txt" for index in range(32)]
IDENTITY_KEYS = ("bytes", "mtime_ns", "sha256", "crc32")
COUNT_KEYS = ("sentences", "paragraphs", "tokens", "space_tokens")
TABLE_HEADER = "rank\ttoken\tcount\tpercent\tcumulative_percent"
WS_PATTERN = re.compile(rb"(?:^| )<sp>/WS(?= |$)")
MAX_ERRORS = 64


def _problem(problems: list, kind: str, **details) -> None:
    if len(problems) < MAX_ERRORS:
        problems.append({"kind": kind, **details})


def _split_ending(raw: bytes) -> tuple[bytes, bytes]:
    if raw.endswith(b"\r\n"):
        return raw[:-2], b"\r\n"
    if raw.endswith((b"\n", b"\r")):
        return raw[:-1], raw[-1:]
    return raw, b""


def reference_numeric_pieces(token: str) -> tuple[str, ...]:
    """Independent equivalent of original step6a_build_vocab.py:82-100.

    Only SN/W_SERIAL are expanded. Only fullwidth ASCII digits are translated;
    other Unicode digits keep their glyph and use Python's isdigit(), as in the
    original. Ordinary tokens and POS suffixes remain unchanged.
    """
    form, separator, tag = token.rpartition("/")
    if not separator or not form or tag not in {"SN", "W_SERIAL"}:
        return (token,)
    normalized = "".join(chr(ord(char) - 0xFEE0) if "０" <= char <= "９" else char
                         for char in form)
    return tuple(f"{char}/{'SN' if char.isdigit() else 'SW'}" for char in normalized)


def reference_numeric_line(raw: bytes) -> bytes:
    body, ending = _split_ending(raw)
    if not body:
        return raw
    line = body.decode("utf-8")
    tokens = line.split(" ")
    if "" in tokens or any(char.isspace() and char != " " for char in line):
        raise ValueError("Malformed serialized token separator")
    for token in tokens:
        form, separator, tag = token.rpartition("/")
        if not separator or not form or not tag:
            raise ValueError(f"Malformed form/POS token: {token!r}")
    pieces = (piece for token in tokens for piece in reference_numeric_pieces(token))
    return " ".join(pieces).encode("utf-8") + ending


def _identity_for(path: Path, sha, crc: int, byte_count: int, before) -> tuple[dict, bool]:
    after = path.stat()
    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    return {"bytes": byte_count, "mtime_ns": after.st_mtime_ns,
            "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}, changed


def _check_identity(actual: dict, expected: dict | None, errors: list, label: str) -> None:
    if expected is None:
        _problem(errors, "missing_identity", source=label)
        return
    for key in IDENTITY_KEYS:
        if key not in expected or actual.get(key) != expected[key]:
            _problem(errors, "identity", source=label, field=key,
                     expected=expected.get(key), actual=actual.get(key))


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    before, sha, crc, size = path.stat(), hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    result, changed = _identity_for(path, sha, crc, size, before)
    if changed:
        raise ValueError(f"File changed during verification: {path}")
    return result


def _count_body(body: bytes, stats: dict, in_paragraph: bool, errors: list,
                label: str, line_number: int) -> bool:
    stats["lines"] += 1
    if not body:
        stats["blank_lines"] += 1
        return False
    if (body.startswith(b" ") or body.endswith(b" ") or b"  " in body
            or any(char in body for char in (b"\t", b"\r", b"\n", b"\v", b"\f"))):
        _problem(errors, "token_separator", source=label, line=line_number)
    stats["sentences"] += 1
    stats["tokens"] += body.count(b" ") + 1
    stats["space_tokens"] += sum(1 for _ in WS_PATTERN.finditer(body))
    if not in_paragraph:
        stats["paragraphs"] += 1
    return True


def verify_shard(raw_path: str | Path, numeric_path: str | Path,
                 expected_record: dict, step5_record: dict,
                 samples_per_shard: int = 32, seed: int = 20261017) -> dict:
    """Stream both files once; retain at most samples_per_shard whole sentences."""
    if samples_per_shard < 1:
        raise ValueError("samples_per_shard must be positive")
    raw_path, numeric_path = Path(raw_path), Path(numeric_path)
    errors, samples = [], []
    keys = (*COUNT_KEYS, "lines", "blank_lines")
    source_stats, numeric_stats = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
    raw_before, numeric_before = raw_path.stat(), numeric_path.stat()
    hashes, crcs, sizes = [hashlib.sha256(), hashlib.sha256()], [0, 0], [0, 0]
    in_source_paragraph = in_numeric_paragraph = False
    seen_sentences = 0
    shard_seed = seed ^ int.from_bytes(hashlib.sha256(raw_path.name.encode()).digest()[:8], "big")
    rng = random.Random(shard_seed)
    with raw_path.open("rb") as source, numeric_path.open("rb") as numeric:
        for line_number, (raw, output) in enumerate(zip_longest(source, numeric), 1):
            for index, content in enumerate((raw, output)):
                if content is not None:
                    hashes[index].update(content)
                    crcs[index] = zlib.crc32(content, crcs[index])
                    sizes[index] += len(content)
            if raw is None or output is None:
                _problem(errors, "line_count", line=line_number,
                         missing="source" if raw is None else "numeric")
            source_body, source_ending = _split_ending(raw) if raw is not None else (None, None)
            output_body, output_ending = _split_ending(output) if output is not None else (None, None)
            if source_body is not None:
                in_source_paragraph = _count_body(source_body, source_stats, in_source_paragraph,
                                                   errors, "source", line_number)
            if output_body is not None:
                in_numeric_paragraph = _count_body(output_body, numeric_stats, in_numeric_paragraph,
                                                    errors, "numeric", line_number)
            if raw is not None and output is not None:
                if bool(source_body) != bool(output_body):
                    _problem(errors, "paragraph_boundary", line=line_number)
                if source_ending != output_ending:
                    _problem(errors, "line_ending", line=line_number)
            if source_body:
                seen_sentences += 1
                row = (line_number, raw, output)
                if len(samples) < samples_per_shard:
                    samples.append(row)
                else:
                    slot = rng.randrange(seen_sentences)
                    if slot < samples_per_shard:
                        samples[slot] = row
    source_identity, source_changed = _identity_for(raw_path, hashes[0], crcs[0], sizes[0], raw_before)
    output_identity, output_changed = _identity_for(numeric_path, hashes[1], crcs[1], sizes[1], numeric_before)
    if source_changed or output_changed:
        _problem(errors, "changed_during_read", source=source_changed, numeric=output_changed)
    _check_identity(source_identity, expected_record.get("input"), errors, "numeric_state.input")
    _check_identity(output_identity, expected_record.get("output"), errors, "numeric_state.output")
    _check_identity(source_identity, step5_record.get("output"), errors, "step5_state.output")
    for key in COUNT_KEYS:
        if source_stats[key] != step5_record.get(key):
            _problem(errors, "step5_count", field=key, expected=step5_record.get(key), actual=source_stats[key])
        target_key = "input_tokens" if key == "tokens" else key
        if source_stats[key] != expected_record.get(target_key):
            _problem(errors, "source_count", field=target_key,
                     expected=expected_record.get(target_key), actual=source_stats[key])
        if numeric_stats[key] != expected_record.get(key):
            _problem(errors, "numeric_count", field=key,
                     expected=expected_record.get(key), actual=numeric_stats[key])
    for key in ("lines", "blank_lines"):
        if numeric_stats[key] != expected_record.get(key):
            _problem(errors, "numeric_count", field=key,
                     expected=expected_record.get(key), actual=numeric_stats[key])
    for key in ("sentences", "paragraphs", "space_tokens", "lines", "blank_lines"):
        if source_stats[key] != numeric_stats[key]:
            _problem(errors, "boundary_or_ws_count", field=key,
                     source=source_stats[key], numeric=numeric_stats[key])
    added = numeric_stats["tokens"] - source_stats["tokens"]
    if expected_record.get("added_tokens") != added:
        _problem(errors, "added_tokens", expected=expected_record.get("added_tokens"), actual=added)
    checked_samples = []
    for line_number, raw, output in sorted(samples):
        row = {"line": line_number, "input_sha256": hashlib.sha256(raw).hexdigest(),
               "output_sha256": hashlib.sha256(output).hexdigest() if output is not None else None}
        try:
            expected = reference_numeric_line(raw)
            row["reference_sha256"] = hashlib.sha256(expected).hexdigest()
            row["passed"] = expected == output
            if not row["passed"]:
                _problem(errors, "sample_transform", line=line_number,
                         expected=expected[:400].decode("utf-8", errors="replace"),
                         actual=output[:400].decode("utf-8", errors="replace") if output else None)
        except (ValueError, UnicodeError) as error:
            row["passed"] = False
            _problem(errors, "sample_invalid", line=line_number, error=str(error))
        checked_samples.append(row)
    return {"file": raw_path.name, "passed": not errors, "errors": errors,
            "input": source_identity, "output": output_identity,
            "source_counts": source_stats, "numeric_counts": numeric_stats,
            "added_tokens": added, "sampled_sentences": len(checked_samples),
            "sampling": {"method": "deterministic reservoir over nonempty source lines",
                         "requested_per_shard": samples_per_shard, "seed": seed},
            "samples": checked_samples}


def verify_frequency_table(path: str | Path, expected_identity: dict,
                           expected_tokens: int, expected_types: int,
                           numeric: bool = False) -> dict:
    """Check table bytes/totals without building any full vocabulary Counter."""
    path = Path(path)
    before, sha, crc, size = path.stat(), hashlib.sha256(), 0, 0
    errors, metadata, numeric_piece_types = [], {}, set()
    header, rows, total, ws = False, 0, 0, 0
    previous = None
    derived = dict.fromkeys(("input_tokens", "numeric_source_occurrences", "numeric_source_types",
                             "numeric_expansion_tokens", "numeric_changed_occurrences", "tokens"), 0)
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            size += len(raw)
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("#"):
                    if ": " in line:
                        key, value = line[2:].split(": ", 1)
                        metadata[key] = json.loads(value)
                    continue
                if not header:
                    if line != TABLE_HEADER:
                        raise ValueError("Unrecognized frequency table header")
                    header = True
                    continue
                rank_text, token, count_text, percent_text, cumulative_text = line.split("\t")
                rank, count = int(rank_text), int(count_text)
                rows += 1
                if rank != rows or count < 1:
                    raise ValueError("Invalid rank or occurrence count")
                form, separator, tag = token.rpartition("/")
                if not separator or not form or not tag or any(char.isspace() for char in token):
                    raise ValueError("Invalid form/POS key")
                ordering = (-count, token)
                if previous is not None and ordering <= previous:
                    raise ValueError("Non-unique frequency/lexical ordering")
                previous = ordering
                total += count
                if token == "<sp>/WS":
                    ws += count
                for observed, expected in ((float(percent_text), count / max(expected_tokens, 1) * 100),
                                           (float(cumulative_text), total / max(expected_tokens, 1) * 100)):
                    if not math.isfinite(observed) or not math.isclose(observed, expected, abs_tol=1e-7):
                        raise ValueError("Frequency percentage differs from occurrence totals")
                pieces = reference_numeric_pieces(token)
                if numeric:
                    if tag == "W_SERIAL" or (tag == "SN" and
                            (len(form) != 1 or not form.isdigit() or "０" <= form <= "９")):
                        raise ValueError("Non-isolated or unnormalized numeric output key")
                else:
                    derived["input_tokens"] += count
                    derived["tokens"] += len(pieces) * count
                    if tag in {"SN", "W_SERIAL"}:
                        derived["numeric_source_occurrences"] += count
                        derived["numeric_source_types"] += 1
                        derived["numeric_expansion_tokens"] += len(pieces) * count
                        derived["numeric_changed_occurrences"] += count if pieces != (token,) else 0
                        numeric_piece_types.update(pieces)
            except (ValueError, UnicodeError, TypeError) as error:
                _problem(errors, "frequency_row", line=line_number, error=str(error))
    identity, changed = _identity_for(path, sha, crc, size, before)
    if changed:
        _problem(errors, "changed_during_read", source=str(path))
    _check_identity(identity, expected_identity, errors, str(path.name))
    if not header or rows != expected_types or total != expected_tokens:
        _problem(errors, "frequency_totals", expected_tokens=expected_tokens, tokens=total,
                 expected_types=expected_types, types=rows, header=header)
    if (metadata.get("status") != "complete" or metadata.get("row_limit") is not None
            or metadata.get("total_tokens") != total or metadata.get("unique_types") != rows):
        _problem(errors, "frequency_metadata", metadata=metadata)
    derived["numeric_expansion_types"] = len(numeric_piece_types)
    derived["added_tokens"] = derived["tokens"] - derived["input_tokens"]
    return {"passed": not errors, "errors": errors, "input": identity,
            "tokens": total, "unique_types": rows, "space_tokens": ws,
            "metadata": metadata, "derived_numeric_totals": derived if not numeric else None}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _fingerprint(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _worker(args):
    return verify_shard(*args)


def _atomic_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{path.name}-", suffix=".partial",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_outputs(root: str | Path = ROOT, *, partial: bool = False, workers: int = 4,
                   samples_per_shard: int = 32, expected_shards: list[str] | None = None,
                   write_report: bool = True) -> dict:
    if workers < 1 or samples_per_shard < 1:
        raise ValueError("workers and samples_per_shard must be positive")
    root, started = Path(root), time.monotonic()
    names = list(EXPECTED_SHARDS if expected_shards is None else expected_shards)
    if (not names or len(set(names)) != len(names)
            or any(re.fullmatch(r"part_\d{2}\.txt", name) is None for name in names)):
        raise ValueError("Expected unique part_NN.txt names")
    errors, shards = [], []
    report = {"passed": False, "scope": "partial" if partial else "full",
              "expected_shards": names, "shard_count": 0, "errors": errors,
              "structural_hash_scope": "all bytes and physical lines in selected immutable shards",
              "transformation_scope": "deterministic reservoir sample; not exhaustive token-by-token transformation verification",
              "samples_per_shard": samples_per_shard, "shards": shards,
              "verification_runtime": {"python_version": platform.python_version(),
                                       "unicode_version": unicodedata.unidata_version,
                                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "frequency_verification": {"checked": False, "reason": "requires completed full-scope outputs"}}
    report_path = root / "reports" / ("step6-numeric-verification.partial.json" if partial
                                       else "step6-numeric-verification.json")
    try:
        reports = root / "reports"
        state = _read_json(reports / "step6-numeric-state.json")
        step5 = _read_json(reports / "step5-state.json")
        completed, source_completed = state["completed"], step5["completed"]
        config_names = state["config"].get("expected_shards")
        if config_names != names:
            _problem(errors, "expected_scope", expected=names, actual=config_names)
        if set(completed) - set(names) or set(source_completed) - set(names):
            _problem(errors, "unexpected_shards")
        if partial:
            selected = sorted(completed)
            if not selected:
                _problem(errors, "no_completed_shards")
        else:
            selected = names
            if (state.get("status") != "complete" or step5.get("status") != "complete"
                    or set(completed) != set(names) or set(source_completed) != set(names)):
                _problem(errors, "incomplete_full_scope", numeric_status=state.get("status"),
                         numeric_completed=len(completed), step5_completed=len(source_completed))
        report["selected_shards"] = selected
        source_fp = _fingerprint(step5["config"])
        numeric_fp = _fingerprint(state["config"])
        if state["config"].get("source_tokenizer_fingerprint") != source_fp:
            _problem(errors, "tokenizer_fingerprint")
        if (state["config"].get("unicode_version", unicodedata.unidata_version)
                != unicodedata.unidata_version):
            _problem(errors, "unicode_runtime", expected=state["config"].get("unicode_version"),
                     actual=unicodedata.unidata_version)
        for name in selected:
            if name not in source_completed:
                _problem(errors, "missing_source_record", file=name)
            for directory in ("tokens", "tokens_numeric"):
                if not (root / directory / name).is_file():
                    _problem(errors, "missing_shard", file=name, directory=directory)
        for directory in ("tokens", "tokens_numeric"):
            unexpected = {path.name for path in (root / directory).glob("*.txt")} - set(names)
            if unexpected:
                _problem(errors, "unrecorded_shard", directory=directory, files=sorted(unexpected))
        if not errors:
            tasks = [(root / "tokens" / name, root / "tokens_numeric" / name,
                      completed[name], source_completed[name], samples_per_shard) for name in selected]
            if workers == 1 or len(tasks) == 1:
                shards.extend(_worker(task) for task in tasks)
            else:
                with ProcessPoolExecutor(max_workers=min(workers, 4, len(tasks))) as pool:
                    shards.extend(pool.map(_worker, tasks))
            report["shard_count"] = len(shards)
            for row in shards:
                if not row["passed"]:
                    _problem(errors, "shard_failed", file=row["file"], errors=row["errors"])
            source_totals = {key: sum(row["source_counts"][key] for row in shards)
                             for key in (*COUNT_KEYS, "lines", "blank_lines")}
            numeric_totals = {key: sum(row["numeric_counts"][key] for row in shards)
                              for key in (*COUNT_KEYS, "lines", "blank_lines")}
            report.update({"source_totals": source_totals, "numeric_totals": numeric_totals,
                           "sampled_sentences": sum(row["sampled_sentences"] for row in shards)})
            # Completion records already selected must remain immutable even if
            # a legitimate partial producer adds subsequent shards concurrently.
            end_state = _read_json(reports / "step6-numeric-state.json")
            end_step5 = _read_json(reports / "step5-state.json")
            if end_state["config"] != state["config"] or end_step5["config"] != step5["config"]:
                _problem(errors, "configuration_changed")
            if not partial and (end_state.get("status") != "complete" or end_step5.get("status") != "complete"):
                _problem(errors, "completion_status_changed")
            for name in selected:
                if (end_state["completed"].get(name) != completed[name]
                        or end_step5["completed"].get(name) != source_completed[name]):
                    _problem(errors, "completion_record_changed", file=name)
            if not partial:
                summary = _read_json(reports / "step6-numeric-summary.json")
                alias = _read_json(reports / "numeric-token-frequency-summary.json")
                raw_summary = _read_json(reports / "token-frequency-summary.json")
                step5_summary = _read_json(reports / "step5-summary.json")
                if alias != summary:
                    _problem(errors, "numeric_summary_alias")
                for label, value in (("numeric", summary), ("raw_frequency", raw_summary)):
                    if (value.get("status") != "complete" or value.get("shard_count") != len(names)
                            or value.get("tokenizer_fingerprint") != source_fp):
                        _problem(errors, "summary_scope", source=label)
                if (summary.get("config") != state["config"]
                        or summary.get("numeric_fingerprint") != numeric_fp):
                    _problem(errors, "numeric_summary_configuration")
                if (step5_summary.get("status") != "complete"
                        or step5_summary.get("shard_count") != len(names)
                        or _fingerprint(step5_summary.get("config", {})) != source_fp):
                    _problem(errors, "step5_summary_scope")
                by_name = {row["file"]: row for row in summary.get("source_shards", [])}
                raw_by_name = {row["file"]: row for row in raw_summary.get("source_shards", [])}
                if (len(by_name) != len(names) or len(summary.get("source_shards", [])) != len(names)
                        or set(by_name) != set(names) or set(raw_by_name) != set(names)
                        or len(raw_summary.get("source_shards", [])) != len(names)):
                    _problem(errors, "summary_source_shards")
                for row in shards:
                    name = row["file"]
                    if by_name.get(name) != {"file": name, **completed[name]}:
                        _problem(errors, "numeric_summary_shard", file=name)
                    _check_identity(row["input"], raw_by_name.get(name, {}).get("input"), errors,
                                    f"raw_frequency_summary:{name}")
                for key in COUNT_KEYS:
                    if source_totals[key] != step5_summary.get(key):
                        _problem(errors, "step5_summary_count", field=key,
                                 expected=step5_summary.get(key), actual=source_totals[key])
                    if source_totals[key] != raw_summary.get(key):
                        _problem(errors, "raw_summary_count", field=key,
                                 expected=raw_summary.get(key), actual=source_totals[key])
                    source_key = "input_tokens" if key == "tokens" else key
                    if source_totals[key] != summary.get(source_key):
                        _problem(errors, "numeric_summary_source_count", field=source_key)
                    if numeric_totals[key] != summary.get(key):
                        _problem(errors, "numeric_summary_output_count", field=key)
                for key in ("lines", "blank_lines"):
                    if numeric_totals[key] != summary.get(key):
                        _problem(errors, "numeric_summary_output_count", field=key)
                raw_table = verify_frequency_table(root / "token_frequencies.txt", raw_summary["output"],
                                                   source_totals["tokens"], raw_summary["unique_types"])
                numeric_table = verify_frequency_table(root / "token_frequencies.numeric.txt", summary["output"],
                                                       numeric_totals["tokens"], summary["unique_types"], numeric=True)
                report["frequency_verification"] = {"checked": True, "raw": raw_table, "numeric": numeric_table}
                for label, table in (("raw", raw_table), ("numeric", numeric_table)):
                    if not table["passed"]:
                        _problem(errors, "frequency_failed", source=label, errors=table["errors"])
                    if table["space_tokens"] != source_totals["space_tokens"]:
                        _problem(errors, "frequency_ws_count", source=label)
                    metadata = table["metadata"]
                    if (metadata.get("shard_count") != len(names)
                            or metadata.get("expected_shard_count") != len(names)
                            or metadata.get("source_shards") != names
                            or metadata.get("tokenizer_fingerprint") != source_fp):
                        _problem(errors, "frequency_scope", source=label)
                for label, actual, expected in (("state.raw_frequency_input", raw_table["input"], state.get("raw_frequency_input")),
                                                 ("summary.raw_frequency_input", raw_table["input"], summary.get("raw_frequency_input")),
                                                 ("state.frequency_output", numeric_table["input"], state.get("frequency_output"))):
                    _check_identity(actual, expected, errors, label)
                if numeric_table["metadata"].get("raw_frequency_sha256") != raw_table["input"]["sha256"]:
                    _problem(errors, "frequency_source_sha256")
                if (numeric_table["metadata"].get("numeric_fingerprint") != numeric_fp
                        or numeric_table["metadata"].get("tokenizer_fingerprint") != source_fp):
                    _problem(errors, "frequency_fingerprint")
                for key, value in raw_table["derived_numeric_totals"].items():
                    if summary.get(key) != value:
                        _problem(errors, "weighted_numeric_total", field=key,
                                 expected=value, actual=summary.get(key))
                if raw_table["derived_numeric_totals"]["tokens"] != numeric_totals["tokens"]:
                    _problem(errors, "weighted_vs_materialized_tokens")
                if (summary.get("input_unique_types") != raw_table["unique_types"]
                        or summary.get("removed_unique_types") != raw_table["unique_types"] - numeric_table["unique_types"]):
                    _problem(errors, "numeric_unique_type_totals")
    except (OSError, ValueError, KeyError, TypeError) as error:
        _problem(errors, "verification_error", error=f"{type(error).__name__}: {error}")
    report["passed"] = not errors
    report["elapsed_seconds"] = time.monotonic() - started
    if write_report:
        _atomic_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=32)
    args = parser.parse_args()
    result = verify_outputs(args.root, partial=args.partial, workers=args.workers,
                            samples_per_shard=args.samples_per_shard)
    print(json.dumps({"passed": result["passed"], "scope": result["scope"],
                      "shards": result["shard_count"], "sampled_sentences": result.get("sampled_sentences", 0),
                      "elapsed_seconds": result["elapsed_seconds"], "errors": result["errors"]},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
