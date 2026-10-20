"""Independently check every token line and sample Kiwi surface reconstruction.

Unlike the lecture's zip-based checker, this fails on missing/extra lines, empty
token lines, and shifted paragraph boundaries. Round trips are sampled because
Kiwi's morphological normalization is intentionally not always lossless.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import repeat, zip_longest
import json
from pathlib import Path
import random
import re
import time
import zlib


ROOT = Path(__file__).resolve().parent
SPACE_TOKEN = "<sp>/WS"
TAG_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*(?:-[IR])?\Z")
WHITESPACE_PATTERN = re.compile(r"\s")
MISSING = object()


def parse_tokens(line):
    """Validate the persisted format; the final slash separates form and tag."""
    values = line.split(" ")
    bad = []
    for index, token in enumerate(values):
        form, slash, tag = token.rpartition("/")
        if (not slash or not form or not TAG_PATTERN.fullmatch(tag)
                or WHITESPACE_PATTERN.search(token)
                or (tag == "WS" and token != SPACE_TOKEN)
                or (form == "<sp>" and tag != "WS")
                or (token == SPACE_TOKEN and (index == 0 or values[index - 1] == SPACE_TOKEN))):
            bad.append(token)
    return values, bad


def _without_newline(value):
    return value.removesuffix("\n").removesuffix("\r")


def verify_streams(source, tokens, sample_size=32, seed=0, require_crlf=False):
    """Stream two line iterators with independent counts and bounded samples."""
    counts, errors = Counter(), Counter()
    examples, samples = [], []
    generator = random.Random(seed)
    previous_source_content = previous_token_content = False
    pending_trailing = None

    def error(kind, line, detail):
        errors[kind] += 1
        if len(examples) < 15:
            examples.append({"kind": kind, "line": line, "detail": detail})

    for line_no, (raw_source, raw_tokens) in enumerate(zip_longest(source, tokens, fillvalue=MISSING), 1):
        source_line = None if raw_source is MISSING else _without_newline(raw_source)
        token_line = None if raw_tokens is MISSING else _without_newline(raw_tokens)
        if pending_trailing is not None:
            previous_line, actual_trailing = pending_trailing
            if actual_trailing != bool(source_line):
                error("trailing_space_mismatches", previous_line,
                      "Nonfinal paragraph sentences require trailing <sp>/WS; final ones do not.")
            pending_trailing = None
        if source_line is None or token_line is None:
            error("line_count_mismatches", line_no,
                  "Missing source line" if source_line is None else "Missing token line")
        elif bool(source_line) != bool(token_line):
            error("blank_line_mismatches", line_no, "Sentence/paragraph boundary differs")

        if source_line is not None:
            counts["source_lines"] += 1
            if source_line:
                counts["source_sentences"] += 1
                counts["source_paragraphs"] += int(not previous_source_content)
            else:
                counts["source_blank_lines"] += 1
            previous_source_content = bool(source_line)
        if token_line is None:
            continue
        counts["token_lines"] += 1
        if require_crlf and not raw_tokens.endswith("\r\n"):
            error("output_newline_mismatches", line_no, "Expected CRLF line terminator")
        if not token_line:
            counts["token_blank_lines"] += 1
            previous_token_content = False
            continue
        counts["token_sentences"] += 1
        counts["token_paragraphs"] += int(not previous_token_content)
        previous_token_content = True
        values, bad = parse_tokens(token_line)
        counts["tokens"] += len(values)
        counts["space_tokens"] += values.count(SPACE_TOKEN)
        counts["trailing_space_tokens"] += int(values[-1] == SPACE_TOKEN)
        if bad:
            counts["invalid_tokens"] += len(bad)
            error("invalid_token_lines", line_no, repr(bad[:3])[:300])
        if source_line:
            pending_trailing = (line_no, values[-1] == SPACE_TOKEN)
            candidate = {"line": line_no, "original": source_line, "tokens": token_line}
            if len(samples) < sample_size:
                samples.append(candidate)
            elif sample_size:
                index = generator.randrange(counts["source_sentences"])
                if index < sample_size:
                    samples[index] = candidate
    if pending_trailing is not None and pending_trailing[1]:
        error("trailing_space_mismatches", pending_trailing[0], "Final sentence has trailing <sp>/WS")
    count_names = ("source_lines", "token_lines", "source_sentences", "token_sentences",
                   "source_paragraphs", "token_paragraphs", "source_blank_lines", "token_blank_lines",
                   "tokens", "space_tokens", "trailing_space_tokens", "invalid_tokens")
    return {"counts": {name: counts[name] for name in count_names},
            "errors": dict(errors), "error_count": sum(errors.values()),
            "examples": examples, "samples": sorted(samples, key=lambda value: value["line"])}


def restore_surface(token_line, kiwi):
    """Reconstruct independently from the on-disk form/tag representation."""
    values, bad = parse_tokens(token_line)
    if bad:
        raise ValueError(f"Invalid tokens: {bad[:3]!r}")
    word_start = True
    morphemes = []
    for value in values:
        if value == SPACE_TOKEN:
            word_start = True
            continue
        form, _, tag = value.rpartition("/")
        morphemes.append((form.replace("_", " "), tag, word_start))
        word_start = False
    return kiwi.join(morphemes).rstrip()


def verify_round_trips(samples, kiwi, max_examples=15):
    counts = Counter()
    examples = []
    for sample in samples:
        counts["sentences"] += 1
        try:
            restored = restore_surface(sample["tokens"], kiwi)
        except Exception as exc:
            counts["join_errors"] += 1
            restored = f"{type(exc).__name__}: {exc}"
            kind = "join_error"
        else:
            kind = "exact" if restored == sample["original"] else "mismatch"
            counts[kind] += 1
        if kind != "exact" and len(examples) < max_examples:
            examples.append({**sample, "kind": kind, "restored": restored})
    return {"sentences": counts["sentences"], "exact": counts["exact"],
            "mismatch": counts["mismatch"], "join_errors": counts["join_errors"],
            "examples": examples}


def measured_lines(path, measure):
    """Hash original bytes in the same pass used for independent line checks."""
    digest, checksum, size = hashlib.sha256(), 0, 0
    before = path.stat()
    with path.open("rb", buffering=1024 * 1024) as handle:
        for value in handle:
            digest.update(value)
            checksum = zlib.crc32(value, checksum)
            size += len(value)
            yield value.decode("utf-8")
    after = path.stat()
    measure.update(bytes=size, sha256=digest.hexdigest(), crc32=f"{checksum:08x}",
                   mtime_ns=after.st_mtime_ns,
                   stable_during_read=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))


def _verify_shard(root, name, sample_size=32):
    """Scan one shard without constructing a Kiwi model in the worker."""
    root = Path(root)
    source_measure, token_measure = {}, {}
    result = verify_streams(measured_lines(root / "sentences_dedup" / name, source_measure),
                            measured_lines(root / "tokens" / name, token_measure),
                            sample_size=sample_size, seed=f"kiwi-roundtrip-v1:{name}", require_crlf=True)
    result.update(file=name, input=source_measure, output=token_measure)
    if not source_measure["stable_during_read"] or not token_measure["stable_during_read"]:
        result["error_count"] += 1
        result["errors"]["modified_during_scan"] = 1
    return result


def verify_shards(root, names, sample_size=32, workers=4):
    """Yield deterministic input-order results from serial or bounded scans."""
    if workers < 1:
        raise ValueError("workers must be positive")
    names = list(names)
    if workers == 1 or len(names) < 2:
        for name in names:
            yield _verify_shard(root, name, sample_size)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(names))) as executor:
        yield from executor.map(_verify_shard, repeat(root), names, repeat(sample_size))


def check_identity(root, snapshot, measures):
    """Verify source metadata and every available original SHA256."""
    mismatches, sha_unverified, sha_missing = [], [], []
    for relative, expected in snapshot.items():
        if relative.startswith("sentences_dedup/") and "sha256" not in expected:
            sha_missing.append(relative)
        path = root / relative
        if not path.is_file():
            mismatches.append({"file": relative, "field": "exists", "expected": True, "actual": False})
            continue
        stat = path.stat()
        actual = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if relative in measures:
            actual.update(measures[relative])
        elif not relative.startswith("sentences_dedup/") and "sha256" in expected:
            actual["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for field, value in expected.items():
            if field not in actual:
                if field == "sha256":
                    sha_unverified.append(relative)
                continue
            if value != actual[field]:
                mismatches.append({"file": relative, "field": field, "expected": value, "actual": actual[field]})
    return {"checked_files": len(snapshot), "mismatches": mismatches,
            "sha256_unverified": sha_unverified,
            "sha256_missing_from_snapshot": sha_missing,
            "sha256_checked_source_shards": sum(name.startswith("sentences_dedup/")
                                                 and "sha256" in snapshot.get(name, {}) for name in measures)}


def compare_summary(summary, shards, partial):
    """Cross-check the writer's counts and digests against this separate scan."""
    mismatches = []
    expected_shards = {value["file"]: value for value in summary.get("shards", [])}
    fields = {"sentences": "token_sentences", "paragraphs": "token_paragraphs",
              "tokens": "tokens", "space_tokens": "space_tokens"}
    for shard in shards:
        expected = expected_shards.get(shard["file"])
        if expected is None:
            mismatches.append({"file": shard["file"], "field": "summary_entry", "actual": "missing"})
            continue
        for expected_name, count_name in fields.items():
            if expected_name in expected and expected[expected_name] != shard["counts"][count_name]:
                mismatches.append({"file": shard["file"], "field": expected_name,
                                   "expected": expected[expected_name], "actual": shard["counts"][count_name]})
        for side in ("input", "output"):
            for field in ("bytes", "sha256", "crc32"):
                value = expected.get(side, {}).get(field)
                if value is not None and value != shard[side].get(field):
                    mismatches.append({"file": shard["file"], "field": f"{side}.{field}",
                                       "expected": value, "actual": shard[side].get(field)})
    if not partial:
        for expected_name, count_name in fields.items():
            actual = sum(value["counts"][count_name] for value in shards)
            if expected_name in summary and summary[expected_name] != actual:
                mismatches.append({"field": expected_name, "expected": summary[expected_name], "actual": actual})
        if summary.get("shard_count") != len(shards):
            mismatches.append({"field": "shard_count", "expected": summary.get("shard_count"), "actual": len(shards)})
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="Check completed token shards only; explicitly not a full result")
    parser.add_argument("--samples-per-shard", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="Parallel streaming shard checks; Kiwi joins run only in the parent")
    args = parser.parse_args()
    if args.samples_per_shard < 1:
        parser.error("--samples-per-shard must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.monotonic()
    reports = ROOT / "reports"
    expected_names = {f"part_{index:02d}.txt" for index in range(32)}
    source_names = {path.name for path in (ROOT / "sentences_dedup").glob("part_*.txt")}
    token_names = {path.name for path in (ROOT / "tokens").glob("part_*.txt")}
    inventory_errors = []
    if source_names != expected_names:
        inventory_errors.append({"kind": "source_inventory", "missing": sorted(expected_names - source_names),
                                 "extra": sorted(source_names - expected_names)})
    if not args.partial and token_names != expected_names:
        inventory_errors.append({"kind": "token_inventory", "missing": sorted(expected_names - token_names),
                                 "extra": sorted(token_names - expected_names)})
    if token_names - expected_names:
        inventory_errors.append({"kind": "unexpected_token_shards", "files": sorted(token_names - expected_names)})
    names = sorted(source_names & token_names)
    if not names:
        raise SystemExit("No completed token shards are available; nothing was verified.")

    snapshot = json.loads((reports / "step5-input-snapshot.json").read_text())
    shards, source_measures, samples = [], {}, []
    for result in verify_shards(ROOT, names, sample_size=args.samples_per_shard, workers=args.workers):
        name = result["file"]
        samples.extend({"file": name, **value} for value in result.pop("samples"))
        shards.append(result)
        source_measures[f"sentences_dedup/{name}"] = result["input"]
        print(f"{name}: {result['counts']['token_sentences']:,} sentences, {result['counts']['tokens']:,} tokens, "
              f"{result['error_count']} errors", flush=True)

    from kiwipiepy import Kiwi, __version__
    kiwi = Kiwi(num_workers=1, model_type="cong-global")
    round_trips = verify_round_trips(samples, kiwi)
    round_trips.update(method="Deterministic seeded reservoir sample within every verified shard",
                       samples_per_shard=args.samples_per_shard,
                       note="Sample only. Morphological spelling normalization and underscore escaping can change surface text.")
    identity = check_identity(ROOT, snapshot, source_measures)
    summary_path = reports / "step5-summary.json"
    summary_mismatches = compare_summary(json.loads(summary_path.read_text()), shards, args.partial) if summary_path.exists() else []
    source_count_mismatches = []
    step4 = reports / "step4-summary.json"
    if step4.exists():
        expected_counts = {value["file"]: value["kept_sentences"] for value in json.loads(step4.read_text())["shards"]}
        for shard in shards:
            actual = shard["counts"]["source_sentences"]
            if expected_counts.get(shard["file"]) != actual:
                source_count_mismatches.append({"file": shard["file"], "expected": expected_counts.get(shard["file"]), "actual": actual})
    totals = Counter()
    for shard in shards:
        totals.update(shard["counts"])
    failures = (len(inventory_errors) + sum(shard["error_count"] for shard in shards)
                + len(identity["mismatches"]) + len(summary_mismatches) + len(source_count_mismatches)
                + round_trips["join_errors"] + len(identity["sha256_missing_from_snapshot"])
                + (0 if args.partial else len(identity["sha256_unverified"])))
    report = {"scope": "partial" if args.partial else "full", "passed": failures == 0,
              "verified_shards": len(shards), "expected_shards": 32,
              "scan_workers": min(args.workers, len(names)),
              "unverified_shards": sorted(expected_names - set(names)), "kiwipiepy_version": __version__,
              "model_type": "cong-global", "counts": dict(totals), "inventory_errors": inventory_errors,
              "input_identity": identity, "writer_summary_available": summary_path.exists(),
              "writer_summary_mismatches": summary_mismatches, "step4_count_mismatches": source_count_mismatches,
              "round_trip": round_trips, "shards": shards, "failures": failures,
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    reports.mkdir(exist_ok=True)
    destination = reports / "step5-verification.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"{'PARTIAL' if args.partial else 'FULL'}: {len(shards)}/32 shards; "
          f"round trip {round_trips['exact']}/{round_trips['sentences']} exact, "
          f"{round_trips['mismatch']} normalized, {round_trips['join_errors']} errors; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
