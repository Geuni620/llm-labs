"""강의 Step 3: 문서 필터 1개 + 원본문단 행 필터 7개, 이후 문장 분리.

필터 기준은 processed-reference/step3_extract_paragraphs.py와 동일하다.
원본은 읽기만 하며, 결과·제거 사유·실제 사례를 별도로 저장한다.
실행: uv run --no-project --with pyarrow python step3_extract_paragraphs.py
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
PARQUET_FILES = ["parquet/data/kowiki/000_00000.parquet", "parquet/data/kowiki/000_00001.parquet"]
OUTPUT_DIRS = ["paragraphs_raw", "paragraphs", "sentences"]
N_SHARDS = 32
# 참고 압축본의 TXT와 같은 줄바꿈을 사용한다. 파이썬으로 읽으면 보통의 \n이다.
OUTPUT_NEWLINE = "\r\n"
MATH_SYMBOLS = frozenset("±∓≠≤≥∈∪∑√∞")
HANJA_RATIO_MAX = 0.30
LETTER_RATIO_MIN = 0.20
HANGUL_RATIO_MIN = 0.30
_BRACKET_RE = re.compile(r"\(([^()]*)\)")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_INITIAL_END_RE = re.compile(r"(?:^|\s)[A-Z]\.$")
_LIST_MARKER_RE = re.compile(r"^\d+\.$")
FILTERS = ["has_formula", "is_structural", "has_math_symbol", "hanja_ratio",
           "clean_paragraph", "letter_ratio", "hangul_ratio", "is_sentence_shape"]


def has_formula(text):
    return "{\\displaystyle" in text


def is_structural(line):
    value = line.strip()
    if not value or value[0] in "#|":
        return True
    if value[0] in "-*" and len(value) >= 2 and value[1] == " ":
        return True
    index = 0
    while index < len(value) and value[index].isdigit():
        index += 1
    return 0 < index < len(value) - 1 and value[index:index + 2] == ". "


def has_math_symbol(line):
    return any(character in MATH_SYMBOLS for character in line)


def _is_hangul(code):
    return 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F


def _is_hanja(code):
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def hanja_ratio(text):
    hangul = hanja = 0
    for character in text:
        code = ord(character)
        if _is_hangul(code):
            hangul += 1
        elif _is_hanja(code):
            hanja += 1
    return hanja / (hangul + hanja) if hangul + hanja else 0.0


def _has_non_korean_letter(text):
    return any(character.isalpha() and not _is_hangul(ord(character)) for character in text)


def clean_paragraph(text):
    placeholders = {}
    counter = 0
    while True:
        match = _BRACKET_RE.search(text)
        if match is None:
            break
        if _has_non_korean_letter(match.group(1)):
            text = text[:match.start()] + text[match.end():]
        else:
            key = f"\x00{counter}\x00"
            placeholders[key] = match.group(0)
            text = text[:match.start()] + key + text[match.end():]
            counter += 1
    for key, original in reversed(placeholders.items()):
        text = text.replace(key, original)
    text = _WS_RE.sub(" ", text).strip()
    while ". ." in text:
        text = text.replace(". .", ".")
    return text


def letter_ratio(text):
    return sum(character.isalpha() for character in text) / len(text) if text else 0.0


def hangul_ratio(text):
    return sum(_is_hangul(ord(character)) for character in text) / len(text) if text else 0.0


def is_sentence_shape(text):
    text = text.rstrip()
    return len(text) >= 2 and text[-1] in ".?!"


def split_sentences(paragraph):
    parts = [part for part in _SENT_SPLIT_RE.split(paragraph.strip()) if part]
    if not parts:
        return parts
    merged = [parts[0]]
    for following in parts[1:]:
        if _INITIAL_END_RE.search(merged[-1]):
            merged[-1] += " " + following
        else:
            merged.append(following)
    return [sentence for sentence in merged if not _LIST_MARKER_RE.match(sentence)]


def evaluate_line(line):
    """비교용 원문, 괄호·공백 정리 결과, 처음 탈락한 필터를 반환한다."""
    if is_structural(line):
        return None, None, "is_structural"
    raw = line.strip()
    if has_math_symbol(line):
        return raw, None, "has_math_symbol"
    if hanja_ratio(line) > HANJA_RATIO_MAX:
        return raw, None, "hanja_ratio"
    cleaned = clean_paragraph(line)
    if not cleaned:
        return raw, cleaned, "clean_paragraph"
    if letter_ratio(cleaned) < LETTER_RATIO_MIN:
        return raw, cleaned, "letter_ratio"
    if hangul_ratio(cleaned) < HANGUL_RATIO_MIN:
        return raw, cleaned, "hangul_ratio"
    if not is_sentence_shape(cleaned):
        return raw, cleaned, "is_sentence_shape"
    return raw, cleaned, None


def _remember(cases, key, source, before, after=None, line_no=0):
    if cases is None or key in cases:
        return
    cases[key] = {"source": dict(source or {}), "line": line_no,
                  "before": before, "after": after}


def _write_paragraphs(out_raw, out_clean, out_sent, texts, counts=None, cases=None, sources=None):
    """강의 함수와 같은 출력·반환값. 추가 인자는 진단 기록만 담당한다."""
    counts = counts if counts is not None else Counter()
    docs = raw_paras = raw_chars = clean_paras = clean_chars = sentences = 0
    for index, text in enumerate(texts):
        docs += 1
        source = sources[index] if sources is not None else {"document_index": index}
        if text is None:
            counts["null_documents"] += 1
            continue
        line_count = text.count("\n") + 1
        counts["source_lines"] += line_count
        if has_formula(text):
            counts["rejected_has_formula"] += 1
            counts["source_lines_in_formula_documents"] += line_count
            _remember(cases, "has_formula", source, text)
            continue
        counts["documents_without_formula"] += 1
        for line_no, line in enumerate(text.split("\n"), 1):
            counts["lines_after_document_filter"] += 1
            raw, cleaned, rejected = evaluate_line(line)
            if raw is not None:
                out_raw.write(raw + "\n\n")
                raw_paras += 1
                raw_chars += len(raw)
            if cleaned is not None:
                counts["lines_entering_clean_paragraph"] += 1
                if cleaned != raw:
                    counts["lines_changed_by_clean_paragraph"] += 1
                    counts["characters_removed_by_clean_paragraph"] += len(raw) - len(cleaned)
                    _remember(cases, "clean_paragraph_changed", source, line, cleaned, line_no)
                if _BRACKET_RE.search(cleaned):
                    _remember(cases, "parentheses_preserved", source, line, cleaned, line_no)
            if rejected is not None:
                counts["rejected_" + rejected] += 1
                _remember(cases, rejected, source, line, cleaned, line_no)
                continue
            out_clean.write(cleaned + "\n\n")
            clean_paras += 1
            clean_chars += len(cleaned)
            split = split_sentences(cleaned)
            if re.search(r"(?:^|\s)[A-Z]\.\s+", cleaned):
                _remember(cases, "initials_preserved", source, cleaned, split, line_no)
            for sentence in split:
                out_sent.write(sentence + "\n")
                sentences += 1
            out_sent.write("\n")
    values = (docs, raw_paras, raw_chars, clean_paras, clean_chars, sentences)
    counts.update(dict(zip(["documents", "raw_paragraphs", "raw_characters", "clean_paragraphs",
                            "clean_characters", "sentences"], values)))
    return values


def _process_shard(job):
    shard, tasks, root_string, output_string = job
    root, output = Path(root_string), Path(output_string)
    counts, cases = Counter(), {}
    files = {}
    with (
        (output / "paragraphs_raw" / f"part_{shard:02d}.txt").open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as raw,
        (output / "paragraphs" / f"part_{shard:02d}.txt").open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as clean,
        (output / "sentences" / f"part_{shard:02d}.txt").open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as sentences,
    ):
        for file_index, row_group in tasks:
            if file_index not in files:
                files[file_index] = pq.ParquetFile(root / PARQUET_FILES[file_index])
            rows = files[file_index].read_row_group(row_group, columns=["id", "title", "text"]).to_pylist()
            sources = [{"id": row["id"], "title": row["title"], "file": PARQUET_FILES[file_index],
                        "row_group": row_group, "row_in_group": i,
                        "order": [file_index, row_group, i]} for i, row in enumerate(rows)]
            _write_paragraphs(raw, clean, sentences, [row["text"] for row in rows], counts, cases, sources)
    return {"shard": shard, "counts": dict(counts), "cases": cases}


def filter_summary(counts):
    total_docs = counts.get("documents", 0)
    all_lines = counts.get("lines_after_document_filter", 0)
    raw_lines = counts.get("raw_paragraphs", 0)
    result = []
    for index, name in enumerate(FILTERS, 1):
        removed = counts.get("rejected_" + name, 0)
        denominator = total_docs if index == 1 else all_lines if index == 2 else raw_lines
        unit = "전체 문서" if index == 1 else "수식 문서 제외 후 전체 행" if index == 2 else "구조 행 제외 후 본문행"
        row = {"number": index, "filter": name, "removed": removed,
               "denominator": denominator, "denominator_label": unit,
               "removed_percent": removed / denominator * 100 if denominator else 0.0}
        if index == 5:
            row.update(changed_lines=counts.get("lines_changed_by_clean_paragraph", 0),
                       changed_input_lines=counts.get("lines_entering_clean_paragraph", 0),
                       characters_removed=counts.get("characters_removed_by_clean_paragraph", 0))
        result.append(row)
    return result


def run_full(root=ROOT, workers=4):
    root = Path(root)
    for name in OUTPUT_DIRS:
        if (root / name).exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {root / name}")
    tasks = []
    inputs = []
    for index, relative in enumerate(PARQUET_FILES):
        parquet = pq.ParquetFile(root / relative)
        inputs.append({"file": relative, "documents": parquet.metadata.num_rows,
                       "row_groups": parquet.num_row_groups})
        tasks.extend((index, group) for group in range(parquet.num_row_groups))
    assigned = [[] for _ in range(N_SHARDS)]
    for index, task in enumerate(tasks):
        assigned[index % N_SHARDS].append(task)
    staging = Path(tempfile.mkdtemp(prefix=".step3-", dir=root))
    for name in OUTPUT_DIRS:
        (staging / name).mkdir()
    started = time.monotonic()
    results = []
    try:
        jobs = [(i, assigned[i], str(root), str(staging)) for i in range(N_SHARDS)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_shard, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"샤드 {result['shard']:02d} 완료: 문단 {result['counts'].get('clean_paragraphs', 0):,} / "
                      f"문장 {result['counts'].get('sentences', 0):,}", flush=True)
        totals, cases = Counter(), {}
        for result in sorted(results, key=lambda item: item["shard"]):
            totals.update(result["counts"])
            for key, candidate in result["cases"].items():
                order = candidate["source"]["order"] + [candidate["line"]]
                if key not in cases or order < cases[key]["source"]["order"] + [cases[key]["line"]]:
                    cases[key] = candidate
        rejected_content = sum(totals.get("rejected_" + name, 0) for name in FILTERS[2:])
        assert totals["raw_paragraphs"] - rejected_content == totals["clean_paragraphs"]
        assert totals["source_lines"] == totals["source_lines_in_formula_documents"] + totals["lines_after_document_filter"]
        assert totals["lines_after_document_filter"] == totals["rejected_is_structural"] + totals["raw_paragraphs"]
        for name in OUTPUT_DIRS:
            (staging / name).rename(root / name)
        reference = root / "processed-reference/step3_extract_paragraphs.py"
        summary = {"inputs": inputs, "workers": workers, "shards": N_SHARDS,
                   "elapsed_seconds": round(time.monotonic() - started, 2),
                   "output_newline": "CRLF (matches supplied archive)",
                   "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None,
                   "counts": dict(totals), "filters": filter_summary(totals), "cases": cases,
                   "shard_counts": [{"shard": r["shard"], "counts": r["counts"]} for r in sorted(results, key=lambda r: r["shard"])]}
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "step3-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps({"step": 3, "counts": dict(totals)}, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def analyze_document(text):
    buffers = [io.StringIO(), io.StringIO(), io.StringIO()]
    counts, cases = Counter(), {}
    _write_paragraphs(*buffers, [text], counts, cases)
    rows, snapshots = [], {number: [] for number in range(2, 9)}
    rejected_document = has_formula(text)
    if not rejected_document:
        for number, original in enumerate(text.split("\n"), 1):
            raw, cleaned, rejected = evaluate_line(original)
            rejection_number = FILTERS.index(rejected) + 1 if rejected else 9
            for stage in range(2, 9):
                if rejection_number > stage:
                    snapshots[stage].append(cleaned if stage >= 5 else raw)
            rows.append({"line": number, "before": original, "after": cleaned,
                         "rejected_by": rejected,
                         "sentences": split_sentences(cleaned) if rejected is None else []})
    trace = [{"number": 1, "filter": FILTERS[0], "text": "" if rejected_document else text}]
    for stage in range(2, 9):
        content = snapshots[stage]
        trace.append({"number": stage, "filter": FILTERS[stage - 1],
                      "text": "\n\n".join(content) + ("\n\n" if content else "")})
    return {"before": text, "document_rejected": rejected_document,
            "document_rejected_by": "has_formula" if rejected_document else None,
            "raw_paragraphs": buffers[0].getvalue(), "paragraphs": buffers[1].getvalue(),
            "sentences": buffers[2].getvalue(), "counts": dict(counts),
            "filter_snapshots": trace, "line_decisions": rows}


def write_sample_reports(root=ROOT):
    root = Path(root)
    ids = [282586, 1792712, 3110350, 660723, 303062]
    found = {}
    for relative in PARQUET_FILES:
        offset = 0
        for batch in pq.ParquetFile(root / relative).iter_batches(batch_size=256):
            rows = batch.to_pylist()
            for index, row in enumerate(rows):
                if row["page_id"] in ids:
                    found[row["page_id"]] = {"row": row, "file": relative, "row_index": offset + index}
            offset += len(rows)
            if len(found) == len(ids):
                break
        if len(found) == len(ids):
            break
    if set(found) != set(ids):
        raise ValueError("샘플 문서를 모두 찾지 못했습니다")
    output = root / "reports/examples"
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for page_id in ids:
        record = found[page_id]
        row = record["row"]
        sample_path = root / f"examples/kowiki-{page_id}.text.txt"
        unchanged = sample_path.read_bytes() == row["text"].encode("utf-8")
        if not unchanged:
            raise ValueError(f"원문과 다른 Step 2 샘플: {sample_path}")
        trace = analyze_document(row["text"])
        for key in ["raw_paragraphs", "paragraphs", "sentences"]:
            with (output / f"kowiki-{page_id}.{key}.txt").open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as file:
                file.write(trace[key])
        sample = {"id": row["id"], "title": row["title"], "source_file": record["file"],
                  "source_row_index": record["row_index"], "step1_row": row,
                  "step2_file": str(sample_path.relative_to(root)), "step2_equals_source_text": unchanged,
                  "step3": trace}
        (output / f"kowiki-{page_id}.trace.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2))
        samples.append(sample)
    summary_path = root / "reports/step3-summary.json"
    comparison = {"step1": json.loads((root / "download_manifest.json").read_text()),
                  "step2": {"sample_documents": len(samples), "format": "UTF-8 TXT, original text only",
                            "all_samples_match_parquet_text": all(s["step2_equals_source_text"] for s in samples)},
                  "step3": json.loads(summary_path.read_text()) if summary_path.exists() else None,
                  "samples": samples}
    (root / "reports/step1-3-comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("샘플 5개 변경 내역 저장: reports/step1-3-comparison.json", flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers는 1 이상이어야 합니다")
    if not args.samples_only:
        run_full(workers=args.workers)
    write_sample_reports()


if __name__ == "__main__":
    main()
