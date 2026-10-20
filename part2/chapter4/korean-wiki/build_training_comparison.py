"""Collect measured checkpoints only; missing 3h/8h outcomes stay empty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_corpus import atomic_json


def read_metrics(path):
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]  # A concurrent writer has not completed its last row.
    return [json.loads(line) for line in lines]


def collect(run_root, corpus, destination):
    run_root, corpus, destination = Path(run_root), Path(corpus), Path(destination)
    runs, shared, curves = {}, None, {}
    manifest = json.loads((corpus / "manifest.json").read_text()) if (corpus / "manifest.json").exists() else None
    controlled_fields = ["device", "cpu_threads", "context_size", "embedding_dim", "hidden_dim", "batch_size",
                         "learning_rate", "weight_decay", "gradient_clip_norm", "seed", "evaluation_windows",
                         "generation_tokens", "temperature", "top_k"]
    for name in ("data-10pct", "data-25pct", "data-100pct"):
        directory = run_root / name
        if not (directory / "run.json").exists():
            continue
        meta = json.loads((directory / "run.json").read_text())
        condition = {key: meta[key] for key in ("corpus_identity", "vocabulary_identity", "initialization_sha256",
                                                 "evaluation_window_sha256", "scripts", "runtime")}
        condition["config"] = {key: meta["config"][key] for key in controlled_fields}
        if shared is None:
            shared = condition
        elif condition != shared:
            raise ValueError("Training runs do not share evaluation, vocabulary, initialization and model conditions")
        records = read_metrics(directory / "metrics.jsonl")
        runs[name] = {"meta": meta, "metrics": records}
        curves[name] = [{"elapsed_seconds": row["actual_elapsed_seconds"], "steps": row["steps"],
                          "epochs_equivalent": row["epochs_equivalent"], "validation": row["validation"]}
                         for row in records]
    rows = []
    for fraction, seconds in [(10, 10800), (25, 10800), (100, 10800), (100, 28800)]:
        name = f"data-{fraction}pct"
        available = runs.get(name, {}).get("metrics", [])
        result = next((row for row in available if row["nominal_seconds"] == seconds), None)
        if result is not None and result["actual_elapsed_seconds"] < seconds:
            raise ValueError("A timed checkpoint was recorded before its stated duration")
        rows.append({"data_percent": fraction, "nominal_hours": seconds / 3600,
                     "status": "measured" if result else "pending", "result": result,
                     "training_paragraphs": runs.get(name, {}).get("meta", {}).get("training_paragraphs"),
                     "training_target_windows": runs.get(name, {}).get("meta", {}).get("training_target_windows")})
    report = {"kind": "ACTUAL_CHECKPOINT_COMPARISON", "status": "complete" if all(row["result"] for row in rows) else "in_progress",
              "shared_conditions": shared, "rows": rows, "curves": curves,
              "corpus_statistics": manifest["statistics"] if manifest else None,
              "note": "No estimates fill missing results. The same validation targets, vocabulary and initialization are required. Test split remains unused; one seed only."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination.with_suffix(".json"), report)
    text = ["한국어 위키 실제 학습 비교", "", "데이터\t시간\t실제 경과(초)\t평가 loss\t평가 PPL\t공백/EOS 제외 PPL\t상태"]
    for row in rows:
        result = row["result"]
        if result:
            val = result["validation"]
            text.append(f"{row['data_percent']}%\t{row['nominal_hours']:g}h\t{result['actual_elapsed_seconds']:.1f}\t{val['loss']:.6f}\t{val['perplexity']:.4f}\t{val['content_perplexity']:.4f}\t실측")
        else:
            text.append(f"{row['data_percent']}%\t{row['nominal_hours']:g}h\t—\t—\t—\t—\t대기 중")
    for row in rows:
        if row["result"]:
            text += ["", f"데이터 {row['data_percent']}%, {row['nominal_hours']:g}시간 생성 예시"]
            for sample in row["result"]["generation"]:
                text += [f"입력: {sample['prompt']}", f"복원: {sample['restored_text']}",
                         "토큰: " + " ".join(sample["generated_tokens"])]
    destination.with_suffix(".txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.runs, args.corpus, args.output)
