"""Plot exact vocabulary-coverage analysis without rereading token shards.

The analysis must contain matched original and digit-split source scopes.
Full results require all 32 shards; a partial snapshot requires --partial and
is visibly labelled. Counts, percentages and paragraph policies are validated
before producing the visualization fragment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "reports/vocabulary-coverage-plot.template.html"
PLACEHOLDER = "__VOCABULARY_COVERAGE_PLOT_DATA__"
SELECTED_KS = [16384, 32768, 49152, 65000, 65536]
METRICS = [
    {"key": "token_coverage_percent", "count_key": "covered_tokens", "total_key": "total_tokens", "label": "토큰 커버리지", "short_label": "토큰", "unit": "회"},
    {"key": "sentence_keep_percent", "count_key": "kept_sentences", "total_key": "total_sentences", "label": "문장 보존", "short_label": "문장", "unit": "문장"},
    {"key": "lecture_paragraph_keep_percent", "count_key": "lecture_kept_paragraphs", "total_key": "total_paragraphs", "label": "문단 보존 · 남은 문장 있음", "short_label": "일부 문단", "unit": "문단"},
    {"key": "strict_paragraph_keep_percent", "count_key": "strict_kept_paragraphs", "total_key": "total_paragraphs", "label": "문단 보존 · 모든 문장 통과", "short_label": "완전 문단", "unit": "문단"},
]
MAX_HTML_BYTES = 1_000_000


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _point(point: dict, variant: dict) -> list[int]:
    k = _integer(point.get("k"), "point.k", 4)
    counts = []
    for metric in METRICS:
        count = _integer(point.get(metric["count_key"]), metric["count_key"])
        total = variant[metric["total_key"]]
        if count > total:
            raise ValueError(f"{metric['count_key']} exceeds its denominator at K={k}")
        supplied = point.get(metric["key"])
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)) or not math.isfinite(supplied):
            raise ValueError(f"Missing finite percentage {metric['key']} at K={k}")
        if not math.isclose(supplied, count / total * 100, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"Wrong denominator or percentage for {metric['key']} at K={k}")
        counts.append(count)
    if counts[3] > counts[2]:
        raise ValueError(f"Strict whole-paragraph retention exceeds nonempty-paragraph retention at K={k}")
    for removed_key, kept, total_key in [("removed_sentences", counts[1], "total_sentences"), ("lecture_removed_paragraphs", counts[2], "total_paragraphs"), ("strict_removed_paragraphs", counts[3], "total_paragraphs")]:
        if removed_key in point and point[removed_key] != variant[total_key] - kept:
            raise ValueError(f"Inconsistent {removed_key} at K={k}")
    return [k, *counts]


def display_domains(variants: list[dict]) -> dict:
    """Keep the small-dictionary view independent of a distant full endpoint."""
    ks = [point[0] for variant in variants for point in variant["curve"]]
    minimum, maximum = min(ks), max(ks)
    return {"focus": {"linear": [0, 200000], "log": [minimum, 200000]},
            "all": {"linear": [0, maximum], "log": [minimum, maximum]}}


def load_coverage_data(path: str | Path, allow_partial: bool = False) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw)
    status = report.get("status")
    if status not in {"complete", "partial"}:
        raise ValueError("Coverage analysis must declare complete or partial status")
    if status == "partial" and not allow_partial:
        raise ValueError("Complete coverage analysis is required; use --partial explicitly for a matched partial snapshot")
    scope = report.get("scope", {})
    names = scope.get("source_shards")
    shard_count = _integer(scope.get("shard_count"), "scope.shard_count", 1)
    expected = _integer(scope.get("expected_shard_count"), "scope.expected_shard_count", 1)
    if not isinstance(names, list) or len(names) != shard_count or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != shard_count:
        raise ValueError("Coverage analysis must identify unique matched source shards")
    if expected != 32 or shard_count > expected or (status == "complete" and shard_count != expected):
        raise ValueError("Final coverage plots require all 32 matched source shards")
    fingerprint = scope.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Coverage analysis is missing its shared source fingerprint")
    if report.get("reserved_slots", report.get("special_token_count", 4)) != 4 or report.get("special_token_count", 4) != 4:
        raise ValueError("K must include the four reserved special tokens")
    variants = report.get("variants")
    if not isinstance(variants, list) or len(variants) != 2 or {variant.get("id") for variant in variants} != {"raw", "numeric"}:
        raise ValueError("Matched raw and numeric variants are both required")
    normalized = []
    for variant in sorted(variants, key=lambda item: item["id"] == "numeric"):
        for key in ("total_tokens", "total_sentences", "total_paragraphs", "unique_types"):
            _integer(variant.get(key), f"{variant['id']}.{key}", 1)
        if variant.get("source_fingerprint") != fingerprint:
            raise ValueError("Original and digit-split variants must share the exact source fingerprint")
        identity = variant.get("frequency_table_identity")
        origin = variant.get("frequency_origin", "actual full-type frequency table")
        derived_frequency = (status == "partial" and variant["id"] == "numeric" and identity is None
                             and origin == "exact weighted numeric expansion of the same covered raw frequency table")
        if not derived_frequency and (not isinstance(identity, dict) or not isinstance(identity.get("sha256"), str) or not identity["sha256"]):
            raise ValueError("Each variant must identify the counted frequency table")
        curve = variant.get("curve_points")
        selected = variant.get("selected_points")
        if not isinstance(curve, list) or not curve or not isinstance(selected, list):
            raise ValueError("Both exact measured curve points and selected K points are required")
        curve_rows = [_point(point, variant) for point in curve]
        if any(previous[0] >= current[0] for previous, current in zip(curve_rows, curve_rows[1:])):
            raise ValueError("Measured K curve points must be strictly increasing")
        if any(previous[index] > current[index] for previous, current in zip(curve_rows, curve_rows[1:]) for index in range(1, 5)):
            raise ValueError("Retention counts must be nondecreasing as vocabulary size grows")
        by_k = {point[0]: point for point in curve_rows}
        full_k = variant["unique_types"] + 4
        if by_k.get(full_k) != [full_k, variant["total_tokens"], variant["total_sentences"], variant["total_paragraphs"], variant["total_paragraphs"]]:
            raise ValueError("The full vocabulary must preserve all source tokens, sentences and paragraphs")
        selected_rows = [_point(point, variant) for point in selected]
        if len({point[0] for point in selected_rows}) != len(selected_rows):
            raise ValueError("Selected K points must not repeat")
        if any(k not in {point[0] for point in selected_rows} for k in SELECTED_KS):
            raise ValueError("Exact points for 16,384 / 32,768 / 49,152 / 65,000 / 65,536 are required")
        if any(by_k.get(point[0]) != point for point in selected_rows):
            raise ValueError("Selected points must exactly match their measured curve snapshot")
        knee = variant.get("candidate_knee")
        if not isinstance(knee, dict) or knee.get("operational_only") is not True or knee.get("k") not in by_k or not isinstance(knee.get("method"), str) or not knee["method"]:
            raise ValueError("A measured, explicitly operational tradeoff candidate is required")
        label = variant.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Each numeric variant requires a readable label")
        normalized.append({"id": variant["id"], "label": label,
                           "totals": [variant["total_tokens"], variant["total_sentences"], variant["total_paragraphs"], variant["total_paragraphs"]],
                           "unique_types": variant["unique_types"], "curve": curve_rows,
                           "selected": sorted(selected_rows),
                           "candidate_knee": {"k": knee["k"], "method": knee["method"], "metric": knee.get("metric", "sentence_keep_percent"), "operational_only": True},
                           "frequency_sha256": identity["sha256"] if isinstance(identity, dict) else None,
                           "frequency_origin": origin, "derived_frequency": derived_frequency})
    if normalized[0]["totals"][1:] != normalized[1]["totals"][1:]:
        raise ValueError("Numeric comparison must cover the same source sentences and paragraphs")
    return {"schema_version": 1, "status": status, "scope": {"source_shards": names, "shard_count": shard_count, "expected_shard_count": expected, "source_fingerprint": fingerprint},
            "special_token_count": 4, "selected_ks": SELECTED_KS, "metrics": METRICS,
            "variants": normalized, "display_ranges": display_domains(normalized),
            "analysis_sha256": hashlib.sha256(raw).hexdigest()}


def build_plot(analysis: str | Path, output: str | Path, allow_partial: bool = False, template: str | Path = TEMPLATE) -> dict:
    data = load_coverage_data(analysis, allow_partial)
    markup = Path(template).read_text(encoding="utf-8")
    if markup.count(PLACEHOLDER) != 1:
        raise ValueError("Coverage template must contain exactly one data placeholder")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    markup = markup.replace(PLACEHOLDER, payload)
    if len(markup.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError("Exact coverage chart data exceeds the 1 MB visualization limit")
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
    return {"status": data["status"], "shards": f"{data['scope']['shard_count']}/{data['scope']['expected_shard_count']}",
            "selected_ks": SELECTED_KS, "candidate_ks": {variant["id"]: variant["candidate_knee"]["k"] for variant in data["variants"]},
            "analysis_sha256": data["analysis_sha256"], "html_bytes": output.stat().st_size, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "reports/vocabulary-coverage.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vocabulary-coverage-plot.html")
    parser.add_argument("--partial", action="store_true", help="Allow a visibly labelled matched partial-scope analysis")
    args = parser.parse_args()
    if args.partial and args.output == ROOT / "reports/vocabulary-coverage-plot.html":
        args.output = ROOT / "reports/vocabulary-coverage-plot.partial.html"
    print(json.dumps(build_plot(args.analysis, args.output, args.partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
