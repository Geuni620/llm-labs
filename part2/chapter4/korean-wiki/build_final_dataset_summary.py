"""Summarize existing dataset artifacts and recorded timings without processing data."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CORPUS = ROOT / "training-corpora/wiki-k35328"


def read(path):
    return json.loads(path.read_text())


def recorded_timing(label, filename, *, note="", kind="elapsed"):
    return {
        "label": label,
        "seconds": read(ROOT / filename)["elapsed_seconds"],
        "source": filename + ":elapsed_seconds",
        "kind": kind,
        "note": note,
    }


def build():
    extraction = read(REPORTS / "step3-summary.json")
    dedup = read(REPORTS / "step4-summary.json")
    kiwi = read(REPORTS / "step5-summary.json")
    numeric = read(REPORTS / "step6-numeric-summary.json")
    numeric_state = read(REPORTS / "step6-numeric-state.json")
    manifest = read(CORPUS / "manifest.json")
    export = read(REPORTS / "integer-encoding-npy.json")
    assert manifest["status"] == export["status"] == "complete"
    stats, context = manifest["statistics"], manifest["context_size"]
    splits = {}
    for split, label in (("train", "학습"), ("validation", "검증"), ("test", "테스트")):
        ids_path = CORPUS / f"{split}.ids.npy"
        ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        assert ids.ndim == 1 and ids.dtype == np.dtype("<u2")
        paragraphs = stats[f"{split}_kept_paragraphs"]
        targets = int(ids.size) - context * paragraphs
        assert targets == stats[f"{split}_target_windows"]
        assert ids_path.stat().st_size == ids.size * 2 + 128
        splits[split] = {
            "label": label,
            "eligible_sentences": stats[f"{split}_eligible_sentences"],
            "source_paragraphs": stats[f"{split}_source_paragraphs"],
            "sentences": stats[f"{split}_kept_sentences"],
            "paragraphs": paragraphs,
            "body_tokens": targets - paragraphs,
            "target_windows": targets,
            "stored_ids": int(ids.size),
            "bos_ids": context * paragraphs,
            "eos_ids": paragraphs,
            "id_npy_bytes": ids_path.stat().st_size,
            "paragraph_npy_bytes": (CORPUS / f"{split}.paragraphs.npy").stat().st_size,
        }
    numeric_fields = [key for key in splits["train"] if key != "label"]
    splits["all"] = {"label": "3분할 합계", **{
        key: sum(value[key] for value in splits.values()) for key in numeric_fields
    }}
    for split in splits.values():
        split["sentence_keep_percent"] = 100 * split["sentences"] / split["eligible_sentences"]
        split["paragraph_any_keep_percent"] = 100 * split["paragraphs"] / split["source_paragraphs"]
        split["id_and_paragraph_bytes"] = split["id_npy_bytes"] + split["paragraph_npy_bytes"]
    completed = numeric_state["completed"]
    completed = list(completed.values()) if isinstance(completed, dict) else completed
    assert len(completed) == 32
    numeric_transform_seconds = sum(row["elapsed_seconds"] for row in completed)
    timings = [
        recorded_timing("문단·문장 추출", "reports/step3-summary.json"),
        recorded_timing("전역 중복 제거", "reports/step4-summary.json"),
        recorded_timing("전처리 독립 검증", "reports/verification.json"),
        recorded_timing("Kiwi 토큰화", "reports/step5-summary.json", note="cong-global · 워커 8개 · 전체 32파일"),
        recorded_timing("Kiwi 독립 검증", "reports/step5-verification.json"),
        recorded_timing("빈도표 독립 검증", "reports/token-frequency-verification.json", note="복구 후 최종 통과한 재검증 1회"),
        {"label": "숫자 파일 변환*", "seconds": numeric_transform_seconds,
         "source": "reports/step6-numeric-state.json:completed[*].elapsed_seconds sum",
         "kind": "partial_sum", "note": "32개 파일 변환 함수의 경과시간 합계. 소스 대기와 최종 숫자 빈도표 생성 제외."},
        recorded_timing("숫자 독립 검증", "reports/step6-numeric-verification.json"),
        recorded_timing("사전별 보존율 분석", "reports/vocabulary-coverage.json"),
        recorded_timing("사전·OOV·정수 준비†", "training-corpora/wiki-k35328/manifest.json",
                        note="출처 복원·train 빈도·사전·OOV·정수 코퍼스 저장을 묶은 내부 측정. 최초 원본 해시 확인 제외.", kind="bundled_elapsed"),
    ]
    watcher = read(REPORTS / "token-frequency-summary.json")
    summary = {
        "status": "complete", "created_at": datetime.now().astimezone().isoformat(),
        "vocabulary_size": manifest["vocab_size"], "reserved_ids": 4,
        "context_size": context,
        "source": {
            "snapshot": "2025-08", "documents": extraction["counts"]["documents"],
            "clean_paragraphs": extraction["counts"]["clean_paragraphs"],
            "extracted_sentences": dedup["input_sentences"],
            "deduplicated_sentences": dedup["kept_sentences"],
            "deduplicated_paragraphs": kiwi["paragraphs"],
            "raw_tokens": kiwi["tokens"], "numeric_tokens": numeric["tokens"],
            "numeric_unique_types": numeric["unique_types"],
        },
        "splits": splits,
        "timings": timings,
        "watcher_timing": {"label": "원본 빈도 집계", "seconds": watcher["elapsed_seconds"],
                           "source": "reports/token-frequency-summary.json:elapsed_seconds", "includes_waiting": True},
        "unrecorded_timings": ["Hugging Face 다운로드", "TXT 샘플 확인", "NumPy 추가 내보내기", "숫자 빈도표 생성 단독 시간"],
        "timing_policy": "Reported program elapsed times and explicitly marked partial sums; concurrent watch processes overlap. Do not sum into an overall total or infer CPU time/percentage allocation.",
        "storage_policy": "MB=1,000,000 bytes; MiB=1,048,576 bytes. ID arrays and optional paragraph metadata exclude vocabulary/provenance/retained .u16/checkpoints.",
        "vocabulary_policy": "Final train-only vocabulary and OOV-filtered corpus. Exploratory full-frequency K curves are separate.",
        "reuse_policy": "When K changes, reuse cleaning, deduplication, Kiwi and numeric token files; rebuild train vocabulary selection, OOV filtering and integer encoding.",
        "training_status": "deferred_by_user", "model_training_started_by_summary": False,
    }
    (REPORTS / "final-dataset-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    template = (REPORTS / "final-dataset-summary.template.html").read_text()
    fragment = template.replace("__DATA__", json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c"))
    (REPORTS / "final-dataset-summary.html").write_text(fragment)
    shell = (REPORTS / "visualization-shell.html").read_text()
    assert shell.count("__CONTENT__") == 1
    (REPORTS / "final-dataset-summary.preview.html").write_text(shell.replace("__CONTENT__", fragment))
    print(json.dumps({"status": "complete", "train": splits["train"], "numeric_transform_seconds": numeric_transform_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    build()
