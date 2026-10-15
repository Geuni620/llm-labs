"""일상 대화의 답변을 N-그램 실습용 문장 목록으로 정리합니다."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import unicodedata


DATA_DIR = Path(__file__).resolve().parent / "data"
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
INVISIBLE_PATTERN = re.compile(r"[\u200b\ufeff]")


def clean_text(text: str) -> str:
    """글자 표현을 통일하고 불필요한 공백과 제어 문자를 정리합니다."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_PATTERN.sub("", text)
    text = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in text
    )
    return " ".join(text.split())


def prepare_data() -> dict:
    source_path = DATA_DIR / "ChatbotData.csv"
    source_bytes = source_path.read_bytes()
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))
    if reader.fieldnames != ["Q", "A", "label"]:
        raise ValueError("CSV 컬럼은 Q, A, label이어야 합니다.")

    rows = list(reader)
    for data_row, row in enumerate(rows, start=1):
        if (
            set(row) != {"Q", "A", "label"}
            or any(value is None for value in row.values())
            or row["label"].strip() not in {"0", "1", "2"}
        ):
            raise ValueError(f"CSV 데이터 {data_row}행의 형식을 확인해 주세요.")

    answers = [row["A"] for row in rows if row["label"].strip() == "0"]
    counts = Counter()
    normalized_rows = 0
    empty_rows = 0
    for answer in answers:
        cleaned = clean_text(answer)
        normalized_rows += cleaned != answer
        if cleaned:
            counts[cleaned] += 1
        else:
            empty_rows += 1

    # 첫 등장 순서를 유지하고, 학습/평가 분리 전에 같은 답변을 하나로 만듭니다.
    sentences = list(counts)
    if not sentences:
        raise ValueError("정제 후 사용할 답변이 없습니다.")
    output_bytes = ("\n".join(sentences) + "\n").encode("utf-8")

    report = {
        "source": {
            "file": source_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(rows),
            "labels_with_outer_spaces": sum(
                row["label"] != row["label"].strip() for row in rows
            ),
        },
        "selection": {"label": "0", "column": "A", "rows": len(answers)},
        "rules": [
            "Normalize label whitespace when selecting rows.",
            "Normalize answer text to Unicode NFC.",
            "Remove zero-width spaces (U+200B) and BOM characters (U+FEFF).",
            "Replace control characters with spaces; trim and collapse whitespace.",
            "Drop empty answers and deduplicate normalized answers in source order.",
        ],
        "quality_checks_before_cleaning": {
            "empty_answers": sum(not answer.strip() for answer in answers),
            "answers_with_urls": sum(bool(URL_PATTERN.search(a)) for a in answers),
            "answers_with_html_tags": sum(bool(HTML_PATTERN.search(a)) for a in answers),
            "answers_with_invisible_characters": sum(
                bool(INVISIBLE_PATTERN.search(a)) for a in answers
            ),
            "answers_with_control_characters": sum(
                any(unicodedata.category(c) == "Cc" for c in a) for a in answers
            ),
            "answers_with_outer_spaces": sum(a != a.strip() for a in answers),
            "answers_with_repeated_whitespace": sum(
                bool(re.search(r"\s{2,}", a)) for a in answers
            ),
        },
        "result": {
            "normalized_answer_rows": normalized_rows,
            "empty_answer_rows_removed": empty_rows,
            "duplicate_answer_rows_removed": sum(counts.values()) - len(sentences),
            "output_sentences": len(sentences),
            "file": "daily_answers.txt",
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
        },
        "duplicate_examples": [
            {"text": text, "source_count": count, "removed": count - 1}
            for text, count in counts.most_common(10)
            if count > 1
        ],
    }
    (DATA_DIR / "daily_answers.txt").write_bytes(output_bytes)
    (DATA_DIR / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    report = prepare_data()
    selected = report["selection"]["rows"]
    removed = report["result"]["duplicate_answer_rows_removed"]
    output = report["result"]["output_sentences"]
    print(f"일상 답변 {selected:,}행 → 중복 {removed:,}행 정리 → {output:,}문장")
    print(f"학습용 문장: {DATA_DIR / 'daily_answers.txt'}")
    print(f"정제 보고서: {DATA_DIR / 'cleaning_report.json'}")
