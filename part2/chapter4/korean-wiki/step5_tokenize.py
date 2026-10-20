"""Step 5a: deduplicated sentences -> Kiwi cong-global morpheme/POS tokens.

The formatting follows processed-reference/step5a_tokenize.py. One Kiwi model
uses a bounded native thread pool; no model is copied into 32 processes. This
stage emits text tokens, not integer IDs or embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Callable, Iterable
import zlib

from kiwipiepy import Kiwi
import kiwipiepy_model


ROOT = Path(__file__).resolve().parent
KIWI_MODEL = "cong-global"
SPACE_TOKEN = "<sp>/WS"
SCHEMA_VERSION = 1
EXPECTED_SHARDS = [f"part_{index:02d}.txt" for index in range(32)]
COUNT_KEYS = ("sentences", "paragraphs", "tokens", "space_tokens")


def format_tokens(sentence: str, analysis: Iterable) -> str:
    """Format one already computed analysis, including original space markers."""
    tokens = list(analysis)
    if sentence and not tokens:
        raise ValueError(f"Kiwi returned no tokens for a nonempty sentence: {sentence!r}")
    out: list[str] = []
    for index, token in enumerate(tokens):
        out.append(f"{token.form.replace(' ', '_')}/{token.tag}")
        current_end = token.start + token.len
        if index + 1 < len(tokens):
            next_start = tokens[index + 1].start
            if next_start >= current_end and any(
                char.isspace() for char in sentence[current_end:next_start]
            ):
                out.append(SPACE_TOKEN)
        elif current_end < len(sentence) and any(
            char.isspace() for char in sentence[current_end:]
        ):
            out.append(SPACE_TOKEN)
    return " ".join(out)


def tokenize_sentence(kiwi: Kiwi, sentence: str) -> str:
    return format_tokens(sentence, kiwi.tokenize(sentence))


def detokenize(tokens_str: str, kiwi: Kiwi) -> str:
    """Lecture's surface restoration; normalized forms need not be byte exact."""
    morphs: list[tuple[str, str, bool]] = []
    word_start = True
    for token in tokens_str.split(" "):
        if token == SPACE_TOKEN:
            word_start = True
            continue
        form, separator, tag = token.rpartition("/")
        if not separator or not tag:
            raise ValueError(f"Malformed token: {token!r}")
        morphs.append((form.replace("_", " "), tag, word_start))
        word_start = False
    return kiwi.join(morphs)


def iter_paragraphs(path: str | Path):
    """Yield nonempty paragraphs; normalize CRLF on input just like the lecture."""
    paragraph: list[str] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            sentence = line.rstrip("\n")
            if sentence:
                paragraph.append(sentence)
            elif paragraph:
                yield paragraph
                paragraph = []
    if paragraph:
        yield paragraph


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while computing its identity: {path}")
    return {
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": sha.hexdigest(),
        "crc32": f"{crc & 0xffffffff:08x}",
    }


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_file(
    kiwi: Kiwi,
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 1024,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Tokenize one shard atomically, preserving sentence/paragraph ordering.

    A batch can end in the middle of a paragraph. Its end markers retain the
    boundary, and each non-last sentence is analyzed with one trailing space.
    The callback receives cumulative counts for the current shard each batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial = input_path.stat()
    started = time.monotonic()
    counts = dict.fromkeys(COUNT_KEYS, 0)
    pending: list[tuple[str, bool]] = []
    sha = hashlib.sha256()
    crc = 0
    written = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".step5-{output_path.stem}-", suffix=".partial",
            dir=output_path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)

            def flush_batch() -> None:
                nonlocal crc, written
                if not pending:
                    return
                analyses = iter(kiwi.tokenize([item[0] for item in pending]))
                for sentence, paragraph_end in pending:
                    try:
                        analysis = next(analyses)
                    except StopIteration as error:
                        raise RuntimeError("Kiwi returned fewer analyses than inputs") from error
                    token_line = format_tokens(sentence, analysis)
                    parts = token_line.split(" ")
                    counts["sentences"] += 1
                    counts["tokens"] += len(parts)
                    counts["space_tokens"] += parts.count(SPACE_TOKEN)
                    if paragraph_end:
                        counts["paragraphs"] += 1
                    data = (token_line + "\r\n" + ("\r\n" if paragraph_end else "")).encode("utf-8")
                    stream.write(data)
                    sha.update(data)
                    crc = zlib.crc32(data, crc)
                    written += len(data)
                if next(analyses, None) is not None:
                    raise RuntimeError("Kiwi returned more analyses than inputs")
                pending.clear()
                if progress is not None:
                    progress({**counts, "elapsed_seconds": time.monotonic() - started})

            for paragraph in iter_paragraphs(input_path):
                for index, sentence in enumerate(paragraph):
                    paragraph_end = index == len(paragraph) - 1
                    pending.append((sentence if paragraph_end else sentence + " ", paragraph_end))
                    if len(pending) >= batch_size:
                        flush_batch()
            flush_batch()
            stream.flush()
            os.fsync(stream.fileno())
        source = file_identity(input_path)
        if (source["bytes"], source["mtime_ns"]) != (initial.st_size, initial.st_mtime_ns):
            raise RuntimeError(f"Input changed during tokenization: {input_path}")
        if output_path.exists():
            raise FileExistsError(f"Output appeared while processing: {output_path}")
        os.replace(temporary, output_path)
        temporary = None
        output = {
            "bytes": written,
            "mtime_ns": output_path.stat().st_mtime_ns,
            "sha256": sha.hexdigest(),
            "crc32": f"{crc & 0xffffffff:08x}",
        }
        return {
            **counts, "input": source, "output": output,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def model_config(workers: int, batch_size: int) -> dict:
    model_path = Path(kiwipiepy_model.get_model_path())
    model_files = {
        path.name: {key: value for key, value in file_identity(path).items() if key != "mtime_ns"}
        for path in sorted(model_path.iterdir()) if path.is_file() and path.name != "__init__.py"
        and path.suffix in {".mdl", ".dict", ".morph", ".txt"}
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": KIWI_MODEL,
        "kiwipiepy_version": importlib.metadata.version("kiwipiepy"),
        "kiwipiepy_model_version": importlib.metadata.version("kiwipiepy_model"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "workers": workers,
        "batch_size": batch_size,
        "newline": "CRLF",
        "space_token": SPACE_TOKEN,
        "model_files": model_files,
        "tokenizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _sample_source(path: Path, limit: int):
    paragraph: list[tuple[str, int]] = []
    remaining = limit
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            sentence = line.rstrip("\n")
            if sentence:
                paragraph.append((sentence, line_number))
                continue
            if not paragraph:
                continue
            for index, (sentence, number) in enumerate(paragraph):
                yield sentence, sentence + (" " if index < len(paragraph) - 1 else ""), number
                remaining -= 1
                if not remaining:
                    return
            paragraph = []
    for index, (sentence, number) in enumerate(paragraph):
        yield sentence, sentence + (" " if index < len(paragraph) - 1 else ""), number
        remaining -= 1
        if not remaining:
            return


def write_sample_reports(kiwi: Kiwi, root: Path = ROOT) -> dict:
    """Review 32 first sentences per shard (1,024 total), separate from full run."""
    rows = []
    for path in sorted((root / "sentences_dedup").glob("part_*.txt")):
        for sentence, analysis_input, number in _sample_source(path, 32):
            rows.append({"source": {"file": path.name, "line": number},
                         "sentence": sentence, "tokenization_input": analysis_input})
    synthetic_sentences = [
        ("서울과 어절 공백", "서울은 대한민국의 수도이다."),
        ("숫자 원문 보존", "1987년에 태어났다."),
        ("영문 이니셜", "제임스 D. Watson은 DNA를 연구했다."),
        ("명사 차: 자동차", "차를 타고 서울에 갔다."),
        ("명사 차: 마시는 차", "따뜻한 차를 마셨다."),
        ("동사 차다", "공을 차면 멀리 날아간다."),
    ]
    synthetic = [{"label": label, "sentence": sentence, "tokenization_input": sentence}
                 for label, sentence in synthetic_sentences]
    for collection in (rows, synthetic):
        analyses = kiwi.tokenize([row["tokenization_input"] for row in collection])
        for row, analysis in zip(collection, analyses, strict=True):
            row["tokens"] = format_tokens(row["tokenization_input"], analysis)
            try:
                row["restored"] = detokenize(row["tokens"], kiwi).rstrip()
                row["status"] = "exact" if row["restored"] == row["sentence"] else "mismatch"
            except Exception as error:
                row["restored"] = None
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
    status = {name: sum(row["status"] == name for row in rows)
              for name in ("exact", "mismatch", "error")}
    result = {
        "model_type": KIWI_MODEL,
        "kiwipiepy_version": importlib.metadata.version("kiwipiepy"),
        "sample_method": "First 32 source sentences in each sorted shard; source line numbers are 1-based.",
        "sample_count": len(rows), **status,
        "exact_percent": status["exact"] / max(len(rows), 1) * 100,
        "note": "Kiwi.join can normalize contracted/expanded forms. A mismatch is recorded, never discarded.",
        "samples": rows, "synthetic_examples": synthetic,
    }
    atomic_json(root / "reports/step5-samples.json", result)
    lines = ["Kiwi cong-global 형태소/품사 토큰화", "아래 6개는 설명을 위한 합성 예시입니다.", ""]
    for row in synthetic:
        lines.extend([row["label"], f"입력: {row['sentence']}", f"토큰: {row['tokens']}",
                      f"복원: {row['restored']}", f"상태: {row['status']}", ""])
    lines.extend(["아래는 실제 말뭉치입니다. 전체 1,024개 표본은 step5-samples.json에 있습니다.", ""])
    for row in rows[:8] + [row for row in rows if row["status"] != "exact"][:8]:
        lines.extend([f"{row['source']['file']}:{row['source']['line']}",
                      f"입력: {row['sentence']}", f"토큰: {row['tokens']}",
                      f"복원: {row['restored']}", f"상태: {row['status']}", ""])
    (root / "reports/step5-examples.txt").write_text("\n".join(lines), encoding="utf-8")
    return result


def run_full(workers: int = 8, batch_size: int = 1024, resume: bool = False, root: Path = ROOT) -> dict:
    workers = min(max(workers, 1), os.cpu_count() or 1)
    source_dir, output_dir, report_dir = root / "sentences_dedup", root / "tokens", root / "reports"
    paths = sorted(source_dir.glob("part_*.txt"))
    if [path.name for path in paths] != EXPECTED_SHARDS:
        raise ValueError("Expected exactly sentences_dedup/part_00.txt through part_31.txt")
    state_path = report_dir / "step5-state.json"
    summary_path = report_dir / "step5-summary.json"
    if not resume and (output_dir.exists() or state_path.exists() or summary_path.exists()):
        raise FileExistsError("Step 5 output/metadata already exists; use --resume after verifying its configuration")
    print("Kiwi cong-global: 모델과 입력 파일의 무결성을 확인합니다.", flush=True)
    config = model_config(workers, batch_size)
    inputs = {path.name: file_identity(path) for path in paths}
    if resume:
        if not state_path.exists():
            raise FileNotFoundError("Cannot resume without reports/step5-state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != config or state["inputs"] != inputs:
            raise ValueError("Resume refused: tokenizer/model/configuration/input identity changed")
        for name, result in state["completed"].items():
            if file_identity(output_dir / name) != result["output"]:
                raise ValueError(f"Resume refused: completed output changed: {name}")
        unexpected = {path.name for path in output_dir.glob("*.txt")} - set(state["completed"])
        if unexpected:
            raise ValueError(f"Resume refused: output files without completion records: {sorted(unexpected)}")
    else:
        state = {"config": config, "inputs": inputs, "completed": {}, "status": "running",
                 "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_seconds": 0.0}
        output_dir.mkdir(parents=True)
        atomic_json(state_path, state)
    # Interrupted partial files are never treated as completed shards.
    for partial in output_dir.glob(".step5-*.partial"):
        partial.unlink()
    kiwi = Kiwi(num_workers=workers, model_type=KIWI_MODEL)
    print(f"Kiwi 1개 모델 / 네이티브 워커 {workers}개 / 배치 {batch_size:,}문장", flush=True)
    started = time.monotonic()
    prior_elapsed = state.get("elapsed_seconds", 0.0)
    base = {key: sum(item[key] for item in state["completed"].values()) for key in COUNT_KEYS}
    last_print = 0.0
    for path in paths:
        if path.name in state["completed"]:
            continue

        def progress(counts: dict) -> None:
            nonlocal last_print
            now = time.monotonic()
            if now - last_print < 10:
                return
            elapsed = now - started
            done = base["sentences"] + counts["sentences"]
            progress_record = {
                "status": "running", "current_shard": path.name,
                "completed_shards": len(state["completed"]),
                "sentences": done, "current_shard_counts": counts,
                "elapsed_seconds": prior_elapsed + elapsed,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_json(report_dir / "step5-progress.json", progress_record)
            print(f"{path.name}: {counts['sentences']:,}문장 / 전체 {done:,}문장 "
                  f"({len(state['completed'])}/32샤드 완료)", flush=True)
            last_print = now

        result = process_file(kiwi, path, output_dir / path.name, batch_size, progress)
        if result["input"] != inputs[path.name]:
            raise RuntimeError(f"Input identity changed during run: {path.name}")
        state["completed"][path.name] = result
        state["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        atomic_json(state_path, state)
        for key in COUNT_KEYS:
            base[key] += result[key]
        print(f"완료 {path.name}: {result['sentences']:,}문장 / {result['tokens']:,}토큰", flush=True)
    state["status"] = "complete"
    atomic_json(state_path, state)
    summary = {
        "status": "complete", "config": config, **base,
        "shard_count": len(state["completed"]),
        "input_bytes": sum(item["input"]["bytes"] for item in state["completed"].values()),
        "output_bytes": sum(item["output"]["bytes"] for item in state["completed"].values()),
        "elapsed_seconds": state["elapsed_seconds"],
        "average_tokens_per_sentence": base["tokens"] / max(base["sentences"], 1),
        "shards": [{"file": name, **state["completed"][name]} for name in EXPECTED_SHARDS],
    }
    atomic_json(summary_path, summary)
    atomic_json(report_dir / "step5-progress.json", {"status": "complete", **base,
                "completed_shards": 32, "elapsed_seconds": state["elapsed_seconds"]})
    print(f"전체 완료: {base['sentences']:,}문장 / {base['tokens']:,}토큰", flush=True)
    write_sample_reports(kiwi, root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    workers = min(args.workers, os.cpu_count() or 1)
    if args.samples_only:
        result = write_sample_reports(Kiwi(num_workers=workers, model_type=KIWI_MODEL))
        print(f"표본 {result['sample_count']:,}문장: 정확 복원 {result['exact']:,}, "
              f"표면형 차이 {result['mismatch']:,}, 복원 오류 {result['error']:,}", flush=True)
    else:
        run_full(workers, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
