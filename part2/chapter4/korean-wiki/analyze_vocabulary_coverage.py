"""Read-only, exact top-K vocabulary coverage for raw and numeric-split tokens.

K includes four reserved IDs. Ordinary tokens are ranked by descending count,
then lexical token order. A sentence survives only when every token is among
the first K-4 ordinary types. No training data or vocabulary is written here.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections import Counter
import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import time
import unicodedata
import zlib

from step6_split_numbers import split_numeric_token
from step6_count_tokens import _atomic_json, _require_identity, file_identity, tokenizer_fingerprint


ROOT = Path(__file__).resolve().parent
RESERVED_SLOTS = 4
SELECTED_K = (16384, 32768, 49152, 65000, 65536)
EXPECTED_SHARDS = [f"part_{index:02d}.txt" for index in range(32)]
HISTOGRAM_KEYS = (
    "sentence_required_k", "paragraph_any_required_k", "paragraph_all_required_k",
    "sentence_tokens_by_required_k", "paragraph_tokens_by_all_required_k",
)
TOTAL_KEYS = ("total_tokens", "total_sentences", "total_paragraphs")


def rank_counts(counts) -> list[tuple[str, int]]:
    for token, count in counts.items():
        if not isinstance(token, str) or not token or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Token frequencies must have nonempty keys and positive integer counts")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_rank_map(counts) -> dict[str, int]:
    return {token: rank for rank, (token, _) in enumerate(rank_counts(counts), 1)}


def _new_histogram() -> dict:
    return {**dict.fromkeys(TOTAL_KEYS, 0), **{key: Counter() for key in HISTOGRAM_KEYS}}


def _add_sentence(hist: dict, required_k: int, tokens: int) -> None:
    hist["total_sentences"] += 1
    hist["total_tokens"] += tokens
    hist["sentence_required_k"][required_k] += 1
    hist["sentence_tokens_by_required_k"][required_k] += tokens


def _add_paragraph(hist: dict, minimum: int | None, maximum: int, tokens: int) -> None:
    if minimum is not None:
        hist["total_paragraphs"] += 1
        hist["paragraph_any_required_k"][minimum] += 1
        hist["paragraph_all_required_k"][maximum] += 1
        hist["paragraph_tokens_by_all_required_k"][maximum] += tokens


def analyze_paragraphs(paragraphs, ranks: dict[str, int], splitter=None) -> dict:
    """Pure one-mode histogram analysis for token lists or serialized sentences."""
    hist = _new_histogram()
    for paragraph in paragraphs:
        minimum, maximum, paragraph_tokens = None, 0, 0
        for sentence in paragraph:
            tokens = sentence.split(" ") if isinstance(sentence, str) else sentence
            maximum_rank, token_count = 0, 0
            for token in tokens:
                if not token:
                    raise ValueError("Empty serialized token in sentence")
                pieces = (token,) if splitter is None else splitter(token)
                for piece in pieces:
                    maximum_rank = max(maximum_rank, ranks[piece])
                    token_count += 1
            if not token_count:
                continue
            required_k = maximum_rank + RESERVED_SLOTS
            _add_sentence(hist, required_k, token_count)
            minimum = required_k if minimum is None else min(minimum, required_k)
            maximum = max(maximum, required_k)
            paragraph_tokens += token_count
        _add_paragraph(hist, minimum, maximum, paragraph_tokens)
    return hist


def _prepare_metrics(hist: dict, ranked_counts) -> dict:
    prefix = array("Q", [0])
    running = 0
    for _, count in ranked_counts:
        running += count
        prefix.append(running)
    if running != hist["total_tokens"]:
        raise ValueError("Frequency token total and source histogram token total differ")
    cdfs = {}
    for key in HISTOGRAM_KEYS:
        thresholds, cumulative = array("Q"), array("Q")
        subtotal = 0
        for threshold, value in sorted(hist[key].items()):
            subtotal += value
            thresholds.append(int(threshold))
            cumulative.append(subtotal)
        cdfs[key] = (thresholds, cumulative)
    expected = {
        "sentence_required_k": hist["total_sentences"],
        "paragraph_any_required_k": hist["total_paragraphs"],
        "paragraph_all_required_k": hist["total_paragraphs"],
        "sentence_tokens_by_required_k": hist["total_tokens"],
        "paragraph_tokens_by_all_required_k": hist["total_tokens"],
    }
    for key, total in expected.items():
        cumulative = cdfs[key][1]
        if (cumulative[-1] if cumulative else 0) != total:
            raise ValueError(f"Histogram consistency failure: {key}")
    return {"prefix": prefix, "cdfs": cdfs, "hist": hist}


def _cdf(prepared: dict, key: str, k: int) -> int:
    thresholds, cumulative = prepared["cdfs"][key]
    index = bisect_right(thresholds, k)
    return cumulative[index - 1] if index else 0


def _point(prepared: dict, k: int) -> dict:
    if not isinstance(k, int) or isinstance(k, bool) or k < RESERVED_SLOTS:
        raise ValueError("K must be an integer including at least four reserved slots")
    hist = prepared["hist"]
    total_tokens, sentences, paragraphs = (hist[key] for key in TOTAL_KEYS)
    covered = prepared["prefix"][min(k - RESERVED_SLOTS, len(prepared["prefix"]) - 1)]
    kept = _cdf(prepared, "sentence_required_k", k)
    any_kept = _cdf(prepared, "paragraph_any_required_k", k)
    all_kept = _cdf(prepared, "paragraph_all_required_k", k)
    training_tokens = _cdf(prepared, "sentence_tokens_by_required_k", k)
    strict_tokens = _cdf(prepared, "paragraph_tokens_by_all_required_k", k)
    return {
        "k": k, "ordinary_slots": k - RESERVED_SLOTS,
        "covered_tokens": covered, "token_coverage_percent": covered / total_tokens * 100 if total_tokens else 0,
        "kept_sentences": kept, "removed_sentences": sentences - kept,
        "sentence_keep_percent": kept / sentences * 100 if sentences else 0,
        "lecture_kept_paragraphs": any_kept, "lecture_removed_paragraphs": paragraphs - any_kept,
        "lecture_paragraph_keep_percent": any_kept / paragraphs * 100 if paragraphs else 0,
        "strict_kept_paragraphs": all_kept, "strict_removed_paragraphs": paragraphs - all_kept,
        "strict_paragraph_keep_percent": all_kept / paragraphs * 100 if paragraphs else 0,
        "retained_training_tokens": training_tokens, "lost_training_tokens": total_tokens - training_tokens,
        "retained_training_token_percent": training_tokens / total_tokens * 100 if total_tokens else 0,
        "strict_retained_training_tokens": strict_tokens, "strict_lost_training_tokens": total_tokens - strict_tokens,
        "strict_retained_training_token_percent": strict_tokens / total_tokens * 100 if total_tokens else 0,
        "vocabulary_token_cost_proxy": k * training_tokens,
    }


def metrics_at_k(hist: dict, ranked_counts, k: int) -> dict:
    return _point(_prepare_metrics(hist, ranked_counts), k)


def _curve_grid(unique_types: int) -> list[int]:
    maximum = unique_types + RESERVED_SLOTS
    points = {RESERVED_SLOTS, maximum, *SELECTED_K}
    points.update(range(512, min(maximum, 16384) + 1, 512))
    points.update(range(16384, min(maximum, 65536) + 1, 256))
    points.update(range(67584, min(maximum, 200000) + 1, 2048))
    points.add(min(maximum, 200000))
    return sorted(point for point in points if point >= RESERVED_SLOTS)


def build_curve(hist: dict, ranked_counts, ks=None) -> list[dict]:
    prepared = _prepare_metrics(hist, ranked_counts)
    points = _curve_grid(len(ranked_counts)) if ks is None else sorted(set(ks))
    curve = [_point(prepared, k) for k in points]
    for index, point in enumerate(curve):
        prior = curve[index - 1] if index else None
        point["sentence_gain_pp_per_1000_from_previous"] = (
            (point["sentence_keep_percent"] - prior["sentence_keep_percent"]) / (point["k"] - prior["k"]) * 1000
            if prior is not None else None
        )
    return curve


def retention_thresholds(hist: dict, targets=(50, 60, 70, 80, 90)) -> dict:
    result = {}
    for label, key, total_key in (
        ("sentence_keep", "sentence_required_k", "total_sentences"),
        ("lecture_paragraph_keep", "paragraph_any_required_k", "total_paragraphs"),
        ("strict_paragraph_keep", "paragraph_all_required_k", "total_paragraphs"),
    ):
        ordered = sorted(hist[key].items())
        output = []
        for target in targets:
            if target < 0 or target > 100:
                raise ValueError("Retention target must be between zero and 100")
            needed, cumulative = math.ceil(hist[total_key] * target / 100), 0
            selected = RESERVED_SLOTS if needed == 0 else None
            for k, value in ordered:
                if selected is not None:
                    break
                cumulative += value
                if cumulative >= needed:
                    selected = int(k)
            output.append({"target_percent": target, "k": selected,
                           "within_uint16": selected is not None and selected <= 65536})
        result[label] = output
    return result


def choose_knee(points: list[dict], low: int = 16384, high: int = 65536) -> dict:
    metric = "sentence_keep_percent"
    candidates = sorted((point for point in points if low <= point["k"] <= high), key=lambda point: point["k"])
    if not candidates:
        return {"k": None, "method": "no measured point in the requested range", "metric": metric,
                "range": [low, high], "operational_only": True}
    first, last = candidates[0], candidates[-1]
    x_span, y_span = last["k"] - first["k"], last[metric] - first[metric]
    distances = []
    for index, point in enumerate(candidates):
        x = (point["k"] - first["k"]) / x_span if x_span else 0
        y = (point[metric] - first[metric]) / y_span if y_span else 0
        previous = candidates[max(0, index - 4)]
        following = candidates[min(len(candidates) - 1, index + 4)]
        before = ((point[metric] - previous[metric]) / (point["k"] - previous["k"]) * 1000
                  if point["k"] != previous["k"] else 0)
        after = ((following[metric] - point[metric]) / (following["k"] - point["k"]) * 1000
                 if following["k"] != point["k"] else 0)
        distances.append({"k": point["k"], "sentence_keep_percent": point[metric],
                          "chord_distance": y - x, "gain_before_pp_per_1000": before,
                          "gain_after_pp_per_1000": after, "slope_drop_pp_per_1000": before - after})
    selected = max(distances, key=lambda point: (point["chord_distance"], -point["k"]))
    return {**selected, "method": "maximum normalized distance above the endpoint chord",
            "metric": metric, "range": [first["k"], last["k"]], "operational_only": True,
            "quality_optimum_proven": False,
            "candidates": sorted(distances, key=lambda point: (-point["chord_distance"], point["k"]))[:5]}


def read_ranked_frequencies(path: str | Path, expected_identity: dict | None = None) -> dict:
    """Read and hash a full-type ranked table, retaining exact integer counts."""
    path = Path(path)
    before = path.stat()
    sha, crc, byte_count = hashlib.sha256(), 0, 0
    metadata, ranked, header, previous, total = {}, [], False, None, 0
    with path.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            byte_count += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("#"):
                if line.startswith("# ") and ": " in line:
                    key, value = line[2:].split(": ", 1)
                    metadata[key] = json.loads(value)
                continue
            if not header:
                if line != "rank\ttoken\tcount\tpercent\tcumulative_percent":
                    raise ValueError(f"Invalid frequency table header: {path}")
                header = True
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(f"Invalid frequency table row: {path}")
            rank, token, count = int(fields[0]), fields[1], int(fields[2])
            order = (-count, token)
            if rank != len(ranked) + 1 or count < 1 or not token or (previous is not None and order <= previous):
                raise ValueError(f"Invalid count/order/rank at frequency row {rank}: {path}")
            ranked.append((token, count))
            previous = order
            total += count
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Frequency table changed while reading: {path}")
    identity = {"bytes": byte_count, "mtime_ns": after.st_mtime_ns,
                "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}
    _require_identity(identity, expected_identity, path)
    if (not header or metadata.get("row_limit") is not None
            or metadata.get("total_tokens") != total or metadata.get("unique_types") != len(ranked)):
        raise ValueError(f"An untruncated all-type frequency table with matching totals is required: {path}")
    return {"metadata": metadata, "ranked_counts": ranked, "tokens": total,
            "unique_types": len(ranked), "identity": identity, "path": str(path)}


def _make_requirements(raw_ranked, numeric_ranked) -> dict[str, tuple[int, int, int]]:
    numeric_ranks = {token: rank for rank, (token, _) in enumerate(numeric_ranked, 1)}
    requirements = {}
    for raw_rank, (token, _) in enumerate(raw_ranked, 1):
        pieces = split_numeric_token(token)
        if not pieces:
            raise ValueError(f"Numeric expansion returned no tokens: {token!r}")
        numeric_rank = max(numeric_ranks[piece] for piece in pieces)
        requirements[token] = (raw_rank + RESERVED_SLOTS, numeric_rank + RESERVED_SLOTS, len(pieces))
    return requirements


def analyze_shard(path: str | Path, requirements: dict, expected_identity: dict | None = None) -> dict:
    """Scan raw serialized tokens once and collect both modes' exact histograms."""
    path = Path(path)
    before, started = path.stat(), time.monotonic()
    raw_hist, numeric_hist = _new_histogram(), _new_histogram()
    raw_min, raw_max, raw_mass = None, 0, 0
    numeric_min, numeric_max, numeric_mass = None, 0, 0
    sha, crc, byte_count = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            sha.update(raw)
            crc = zlib.crc32(raw, crc)
            byte_count += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                _add_paragraph(raw_hist, raw_min, raw_max, raw_mass)
                _add_paragraph(numeric_hist, numeric_min, numeric_max, numeric_mass)
                raw_min, raw_max, raw_mass = None, 0, 0
                numeric_min, numeric_max, numeric_mass = None, 0, 0
                continue
            tokens = line.split(" ")
            required_raw = required_numeric = numeric_tokens = 0
            for token in tokens:
                try:
                    token_raw, token_numeric, token_length = requirements[token]
                except KeyError as error:
                    raise ValueError(f"Source token missing from the scoped frequency tables at {path}:{line_number}: {token!r}") from error
                if token_raw > required_raw:
                    required_raw = token_raw
                if token_numeric > required_numeric:
                    required_numeric = token_numeric
                numeric_tokens += token_length
            _add_sentence(raw_hist, required_raw, len(tokens))
            _add_sentence(numeric_hist, required_numeric, numeric_tokens)
            raw_min = required_raw if raw_min is None else min(raw_min, required_raw)
            raw_max = max(raw_max, required_raw)
            raw_mass += len(tokens)
            numeric_min = required_numeric if numeric_min is None else min(numeric_min, required_numeric)
            numeric_max = max(numeric_max, required_numeric)
            numeric_mass += numeric_tokens
    _add_paragraph(raw_hist, raw_min, raw_max, raw_mass)
    _add_paragraph(numeric_hist, numeric_min, numeric_max, numeric_mass)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Raw token source changed during analysis: {path}")
    identity = {"bytes": byte_count, "mtime_ns": after.st_mtime_ns,
                "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}
    _require_identity(identity, expected_identity, path)
    return {"file": path.name, "input": identity, "raw": raw_hist, "numeric": numeric_hist,
            "elapsed_seconds": time.monotonic() - started}


_WORKER_REQUIREMENTS = None


def _initialize_worker(requirements: dict) -> None:
    global _WORKER_REQUIREMENTS
    _WORKER_REQUIREMENTS = requirements


def _analyze_task(task) -> dict:
    return analyze_shard(task[0], _WORKER_REQUIREMENTS, task[1])


def _merge_histograms(target: dict, source: dict) -> None:
    for key in TOTAL_KEYS:
        target[key] += source[key]
    for key in HISTOGRAM_KEYS:
        target[key].update(source[key])


def _write_histograms(path: Path, payload: dict) -> dict:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_sources(root: Path, partial: bool, expected_names: list[str]) -> dict:
    reports = root / "reports"
    step5 = _read_json(reports / "step5-state.json")
    raw_summary = _read_json(reports / ("token-frequency-state.json" if partial else "token-frequency-summary.json"))
    raw_table = read_ranked_frequencies(root / ("reports/token_frequencies.partial.all.txt" if partial else "token_frequencies.txt"),
                                        None if partial else raw_summary["output"])
    metadata = raw_table["metadata"]
    names = metadata.get("source_shards")
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or not set(names) <= set(expected_names):
        raise ValueError("Raw frequency scope contains invalid or repeated shard names")
    if not partial and (names != expected_names or step5.get("status") != "complete"
                        or raw_summary.get("status") != "complete" or metadata.get("status") != "complete"):
        raise ValueError("Full analysis requires all completed raw shards and the final raw frequency table")
    if (metadata.get("shard_count") != len(names) or raw_summary.get("shard_count") != len(names)
            or metadata.get("expected_shard_count") != len(expected_names)
            or raw_summary["tokens"] != raw_table["tokens"] or raw_summary["unique_types"] != raw_table["unique_types"]):
        raise ValueError("Raw frequency table and source summary scopes/totals differ; retry after an atomic update")
    fingerprint = tokenizer_fingerprint(step5["config"])
    if raw_summary.get("tokenizer_fingerprint") != fingerprint or metadata.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("Raw frequency tokenizer fingerprint differs from Step 5")
    source_list = raw_summary.get("processed_sources", raw_summary.get("source_shards", []))
    raw_sources = {row["file"]: row for row in source_list}
    if len(source_list) != len(names) or set(raw_sources) != set(names):
        raise ValueError("Raw frequency source identity records do not match the table scope")
    for name in names:
        if name not in step5["completed"] or raw_sources[name]["input"] != step5["completed"][name]["output"]:
            raise ValueError(f"Raw source identity differs from Step 5: {name}")
    print("실제 빈도표에서 숫자 분리 전·후의 정확한 순위를 준비합니다.", flush=True)
    numeric_counts = Counter()
    for token, count in raw_table["ranked_counts"]:
        for piece in split_numeric_token(token):
            numeric_counts[piece] += count
    derived_ranked = rank_counts(numeric_counts)
    del numeric_counts
    numeric_summary, numeric_sources = None, None
    if partial:
        numeric_table = {"ranked_counts": derived_ranked, "tokens": sum(count for _, count in derived_ranked),
                         "unique_types": len(derived_ranked), "identity": None,
                         "origin": "exact weighted numeric expansion of the same covered raw frequency table"}
    else:
        numeric_summary = _read_json(reports / "numeric-token-frequency-summary.json")
        numeric_table = read_ranked_frequencies(root / "token_frequencies.numeric.txt", numeric_summary["output"])
        numeric_sources = {row["file"]: row for row in numeric_summary["source_shards"]}
        if (numeric_summary.get("status") != "complete" or numeric_summary.get("shard_count") != len(names)
                or len(numeric_summary["source_shards"]) != len(names) or set(numeric_sources) != set(names)
                or numeric_summary.get("tokenizer_fingerprint") != fingerprint
                or numeric_summary.get("tokens") != numeric_table["tokens"]
                or numeric_summary.get("unique_types") != numeric_table["unique_types"]
                or numeric_table["metadata"].get("status") != "complete"
                or numeric_table["metadata"].get("source_shards") != names
                or numeric_table["ranked_counts"] != derived_ranked):
            raise ValueError("Numeric frequencies do not exactly match the full raw source and pinned splitting rule")
        numeric_config = numeric_summary.get("config", {})
        numeric_script_sha = hashlib.sha256((ROOT / "step6_split_numbers.py").read_bytes()).hexdigest()
        if (("script_sha256" in numeric_config and numeric_config["script_sha256"] != numeric_script_sha)
                or ("unicode_version" in numeric_config and numeric_config["unicode_version"] != unicodedata.unidata_version)):
            raise ValueError("The numeric implementation or Unicode digit rules differ from the pinned numeric run")
        for name in names:
            if numeric_sources[name]["input"] != raw_sources[name]["input"]:
                raise ValueError(f"Numeric and raw sources differ: {name}")
        numeric_table["origin"] = "final numeric frequency table, verified against weighted raw-source expansion"
    source_scope = [{"file": name, "input": raw_sources[name]["input"]} for name in names]
    scope_fingerprint = tokenizer_fingerprint({"tokenizer_fingerprint": fingerprint, "sources": source_scope})
    return {"names": names, "step5": step5, "raw_summary": raw_summary, "raw_sources": raw_sources,
            "numeric_summary": numeric_summary, "numeric_sources": numeric_sources,
            "raw_table": raw_table, "numeric_table": numeric_table,
            "tokenizer_fingerprint": fingerprint, "source_fingerprint": scope_fingerprint}


def _write_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_human_reports(root: Path, stem: str, result: dict) -> None:
    lines = ["어휘 수 K와 실제 데이터 보존율: 정확한 순위 기반 분석", f"상태: {result['status']}",
             f"범위: {result['scope']['shard_count']}/{result['scope']['expected_shard_count']} 완료 샤드",
             "K는 pad/unk/bos/eos용 4개 예약 슬롯을 포함합니다. 실제 보통 토큰은 상위 K-4개입니다.",
             "숫자 분리 모드는 SN/W_SERIAL을 문자 단위로 변환한 뒤 빈도·순위·OOV를 계산합니다.",
             "문장 보존은 모든 토큰이 어휘에 있을 때만 성립합니다. 문단 ANY는 한 문장 이상 남김, ALL은 문단 전체를 보존합니다.",
             "모든 수치는 원본 위키 문서가 아닌 실제 토큰화 문장/문단을 분모로 사용합니다.", ""]
    columns = ["variant", "k", "token_coverage_percent", "sentence_keep_percent", "lecture_paragraph_keep_percent",
               "strict_paragraph_keep_percent", "kept_sentences", "removed_sentences", "lecture_kept_paragraphs",
               "strict_kept_paragraphs", "retained_training_tokens", "lost_training_tokens",
               "strict_retained_training_tokens", "strict_lost_training_tokens", "vocabulary_token_cost_proxy"]
    selected_tsv, curve_tsv = ["\t".join(columns)], ["\t".join(columns)]
    for variant in result["variants"]:
        lines.extend([variant["label"], f"빈도 총 토큰 {variant['total_tokens']:,} / 종류 {variant['unique_types']:,} / "
                      f"문장 {variant['total_sentences']:,} / 문단 {variant['total_paragraphs']:,}"])
        for point in variant["selected_points"]:
            lines.append(f"K={point['k']:,}: 토큰 빈도 커버 {point['token_coverage_percent']:.4f}%, "
                         f"문장 {point['sentence_keep_percent']:.4f}%, 문단 ANY {point['lecture_paragraph_keep_percent']:.4f}%, "
                         f"문단 ALL {point['strict_paragraph_keep_percent']:.4f}%; "
                         f"실제 남는 학습 토큰 {point['retained_training_tokens']:,}")
        knee = variant["candidate_knee"]
        lines.extend([f"운영 비교용 무릎 후보 K={knee['k']}: {knee['method']}",
                      "무릎 후보는 데이터 보존율과 어휘 크기만 비교한 휴리스틱이며 모델 품질의 최적값이 아닙니다.",
                      "K × 남는 학습 토큰은 단순 비용 대리값이며 CPU 시간이나 처리량을 측정한 결과가 아닙니다.",
                      "문장/문단 50·60·70·80·90%를 처음 만족하는 정확한 K:",
                      json.dumps(variant["thresholds"], ensure_ascii=False), ""])
        for key, output in (("selected_points", selected_tsv), ("curve_points", curve_tsv)):
            for point in variant[key]:
                values = [variant["id"]] + [point[column] for column in columns[1:]]
                output.append("\t".join(f"{value:.9f}" if isinstance(value, float) else str(value) for value in values))
    _write_text(root / "reports" / f"{stem}.txt", "\n".join(lines) + "\n")
    _write_text(root / "reports" / f"{stem}.selected.tsv", "\n".join(selected_tsv) + "\n")
    _write_text(root / "reports" / f"{stem}.curve.tsv", "\n".join(curve_tsv) + "\n")


def run(root: str | Path = ROOT, workers: int = 4, partial: bool = False, expected_shards=None) -> dict:
    root = Path(root)
    names_expected = list(EXPECTED_SHARDS if expected_shards is None else expected_shards)
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.monotonic()
    sources = _read_sources(root, partial, names_expected)
    names, raw_table, numeric_table = sources["names"], sources["raw_table"], sources["numeric_table"]
    requirements = _make_requirements(raw_table["ranked_counts"], numeric_table["ranked_counts"])
    totals = {"raw": _new_histogram(), "numeric": _new_histogram()}
    source_results = {}
    tasks = [(str(root / "tokens" / name), sources["raw_sources"][name]["input"]) for name in names]
    workers = min(workers, len(tasks), os.cpu_count() or 1)
    print(f"원시 토큰 {len(names)}개 샤드를 한 번 읽어 두 표현의 정확한 CDF를 계산합니다 (워커 {workers}).", flush=True)

    def accept(result: dict) -> None:
        name = result["file"]
        source = sources["raw_sources"][name]
        for key, source_key in (("total_tokens", "tokens"), ("total_sentences", "sentences"), ("total_paragraphs", "paragraphs")):
            if result["raw"][key] != source[source_key]:
                raise ValueError(f"Scanned raw source totals differ: {name}, {source_key}")
            if sources["numeric_sources"] is not None and result["numeric"][key] != sources["numeric_sources"][name][source_key]:
                raise ValueError(f"Scanned numeric totals differ from materialized output: {name}, {source_key}")
        for mode in totals:
            _merge_histograms(totals[mode], result[mode])
        source_results[name] = {"file": name, "input": result["input"], "elapsed_seconds": result["elapsed_seconds"]}
        print(f"분석 {len(source_results)}/{len(names)}: {name} / {result['raw']['total_sentences']:,}문장", flush=True)

    if workers == 1:
        for path, identity in tasks:
            accept(analyze_shard(path, requirements, identity))
    else:
        # The analyzer has no Kiwi model or native thread pool. Fork lets these
        # read-only rank requirements remain shared instead of four model copies.
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        context = multiprocessing.get_context(method)
        with context.Pool(workers, initializer=_initialize_worker, initargs=(requirements,)) as pool:
            for result in pool.imap_unordered(_analyze_task, tasks):
                accept(result)
    del requirements
    if (totals["raw"]["total_tokens"] != raw_table["tokens"]
            or totals["numeric"]["total_tokens"] != numeric_table["tokens"]
            or totals["raw"]["total_sentences"] != totals["numeric"]["total_sentences"]
            or totals["raw"]["total_paragraphs"] != totals["numeric"]["total_paragraphs"]):
        raise ValueError("Global source/frequency/normalization totals differ")
    if not partial:
        step5_summary = _read_json(root / "reports/step5-summary.json")
        for key, source_key in (("total_tokens", "tokens"), ("total_sentences", "sentences"), ("total_paragraphs", "paragraphs")):
            if totals["raw"][key] != step5_summary[source_key]:
                raise ValueError(f"Full coverage denominators differ from Step 5: {source_key}")
    stem = "vocabulary-coverage.partial" if partial else "vocabulary-coverage"
    variants = []
    for mode, table, label in (("raw", raw_table, "숫자 분리 전: 실제 Kiwi 형태소/품사"),
                               ("numeric", numeric_table, "숫자 분리 후: 강의의 SN/W_SERIAL 문자 분리")):
        hist = totals[mode]
        curve = build_curve(hist, table["ranked_counts"])
        indexed = {point["k"]: point for point in curve}
        full = indexed[len(table["ranked_counts"]) + RESERVED_SLOTS]
        if (full["kept_sentences"] != hist["total_sentences"]
                or full["strict_kept_paragraphs"] != hist["total_paragraphs"]
                or full["covered_tokens"] != hist["total_tokens"]):
            raise RuntimeError("Full vocabulary must preserve every source sentence, paragraph and token")
        variants.append({"id": mode, "label": label, **{key: hist[key] for key in TOTAL_KEYS},
                         "unique_types": table["unique_types"], "full_vocabulary_k": table["unique_types"] + RESERVED_SLOTS,
                         "frequency_table_identity": table["identity"], "frequency_origin": table.get("origin", "actual full-type frequency table"),
                         "source_fingerprint": sources["source_fingerprint"],
                         "selected_points": [indexed[k] for k in SELECTED_K], "curve_points": curve,
                         "thresholds": retention_thresholds(hist), "candidate_knee": choose_knee(curve)})
    histogram_path = root / "reports" / f"{stem}.histograms.json.gz"
    histogram_identity = _write_histograms(histogram_path, {"reserved_slots": RESERVED_SLOTS,
                                                           "source_fingerprint": sources["source_fingerprint"],
                                                           "variants": totals})
    result = {
        "status": "partial" if partial else "complete",
        "scope": {"source_shards": names, "source_fingerprint": sources["source_fingerprint"],
                  "shard_count": len(names), "expected_shard_count": len(names_expected)},
        "reserved_slots": RESERVED_SLOTS, "reserved_ids": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
        "tokenizer_fingerprint": sources["tokenizer_fingerprint"],
        "ranking": "descending frequency, then token lexical order; exact ordinary rank <= K-4",
        "sentence_rule": "every token must be in the ordinary top-(K-4) vocabulary",
        "lecture_paragraph_rule": "at least one complete sentence survives; remaining sentences form the nonempty paragraph",
        "strict_paragraph_rule": "every sentence in the original tokenized paragraph survives",
        "numeric_rule": "SN/W_SERIAL -> character tokens before counting, ranking and OOV; fullwidth digits become ASCII; other tokens unchanged",
        "denominator_note": "Actual tokenized sentences and paragraphs in the covered shards, not original Wikipedia documents.",
        "cost_proxy_note": "K * retained_training_tokens is an analytic output-vocabulary-work proxy, not a CPU benchmark or training-time prediction.",
        "quality_note": "Retention/size knees do not establish a model-quality optimum; 3-hour/8-hour training comparisons require actual controlled runs.",
        "curve_note": "Every published grid point is exact. Intermediate K can be recomputed from saved rank histograms and full frequency tables without rescanning source tokens.",
        "runtime": {"python_version": platform.python_version(), "unicode_version": unicodedata.unidata_version,
                    "platform": platform.platform(), "workers": workers,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "numeric_script_sha256": hashlib.sha256((ROOT / "step6_split_numbers.py").read_bytes()).hexdigest()},
        "sources": [source_results[name] for name in names],
        "histogram_source": {"path": str(histogram_path), **histogram_identity},
        "variants": variants, "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(root / "reports" / f"{stem}.json", result)
    _write_human_reports(root, stem, result)
    print(f"완료: reports/{stem}.json / .txt / .selected.tsv / .curve.tsv", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--partial", action="store_true", help="Analyze only the explicitly labeled covered raw frequency scope")
    args = parser.parse_args()
    run(workers=args.workers, partial=args.partial)


if __name__ == "__main__":
    main()
