"""강의 Step 4: NFKC + 공백 정리 후 동일 문장을 전 파일에서 한 번만 보존.

비교 키만 정규화하며, 출력 문장 자체는 원문을 보존한다.
실행: python step4_dedup_sentences.py
"""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import unicodedata


ROOT = Path(__file__).resolve().parent
OUTPUT_NEWLINE = "\r\n"
_WS_RE = re.compile(r"\s+")


def normalize(text):
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def hash_key(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def dedup_one_file(input_path, output_path, seen, duplicates, shard, first_repeat_cases):
    total = kept = 0
    paragraph_kept = False
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as target:
        for line_number, line in enumerate(source, 1):
            line = line.rstrip("\n")
            if not line:
                if paragraph_kept:
                    target.write("\n")
                paragraph_kept = False
                continue
            total += 1
            key = hash_key(normalize(line))
            if key in seen:
                duplicates[key] += 1
                if len(first_repeat_cases) < 8 and duplicates[key] == 1:
                    first_repeat_cases.append({"key": key.hex(), "removed": line,
                                               "removed_source": {"file": input_path.name, "line": line_number},
                                               "first_position": seen[key]})
                continue
            seen[key] = (shard << 32) | line_number
            target.write(line + "\n")
            kept += 1
            paragraph_kept = True
        if paragraph_kept:
            target.write("\n")
    return total, kept


def _write_removed_log(path, first_texts, duplicates, shards, total, kept):
    removed = sum(duplicates.values())
    with path.open("w", encoding="utf-8", newline=OUTPUT_NEWLINE) as file:
        file.write("# 문장 단위 정규화 + Exact dedup 결과\n")
        file.write(f"# 입력 샤드 수:        {shards}\n")
        file.write(f"# 입력 문장 수:        {total:,}\n")
        file.write(f"# 출력 문장 수:        {kept:,}\n")
        file.write(f"# 제거된 문장 수:      {removed:,}")
        if total:
            file.write(f"  ({removed / total * 100:.2f}%)")
        file.write("\n")
        file.write(f"# 유니크 중복 종류:    {len(duplicates):,}\n")
        file.write("#\n")
        file.write("# 형식: [Nx] <첫 등장 원문>\n")
        file.write("# N = 코퍼스 전체에서 그 문장이 등장한 총 횟수 (원본 1 + 중복 N-1)\n")
        file.write("# 정렬: N desc (가장 자주 등장한 boilerplate가 맨 위)\n")
        file.write("# 전역 dedup이므로 샤드 경계와 무관\n\n")
        for key, extra in sorted(duplicates.items(), key=lambda item: item[1], reverse=True):
            file.write(f"[{extra + 1}x] {first_texts[key]}\n")


def run_full(root=ROOT):
    root = Path(root)
    paths = sorted((root / "sentences").glob("part_*.txt"))
    if not paths:
        raise FileNotFoundError("먼저 Step 3을 실행해야 합니다")
    output = root / "sentences_dedup"
    removed_path = root / "sentences_dedup_removed.txt"
    if output.exists() or removed_path.exists():
        raise FileExistsError("기존 중복 제거 결과를 덮어쓰지 않습니다")
    staging = Path(tempfile.mkdtemp(prefix=".step4-", dir=root))
    (staging / "sentences_dedup").mkdir()
    seen, duplicates = {}, Counter()
    examples, shards = [], []
    total = kept = 0
    started = time.monotonic()
    try:
        for shard, path in enumerate(paths):
            incoming, outgoing = dedup_one_file(path, staging / "sentences_dedup" / path.name,
                                                seen, duplicates, shard, examples)
            total += incoming
            kept += outgoing
            shards.append({"file": path.name, "input_sentences": incoming, "kept_sentences": outgoing})
            if shard % 4 == 3 or shard == len(paths) - 1:
                print(f"중복 제거 {shard + 1}/{len(paths)}: {total:,} → {kept:,}", flush=True)
        assert kept == len(seen) and total - kept == sum(duplicates.values())
        # 전체 문장을 메모리에 보관하지 않고, 중복 그룹의 첫 원문만 한 번 더 읽는다.
        first_refs = {key: seen[key] for key in duplicates}
        del seen
        gc.collect()
        requests = defaultdict(dict)
        for key, position in first_refs.items():
            requests[position >> 32][position & 0xFFFFFFFF] = key
        first_texts = {}
        for shard, wanted in requests.items():
            last = max(wanted)
            with paths[shard].open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    if number in wanted:
                        first_texts[wanted[number]] = line.rstrip("\n")
                    if number >= last:
                        break
        assert len(first_texts) == len(duplicates)
        _write_removed_log(staging / removed_path.name, first_texts, duplicates, len(paths), total, kept)
        for example in examples:
            key = bytes.fromhex(example.pop("key"))
            position = example.pop("first_position")
            example["kept"] = first_texts[key]
            example["kept_source"] = {"file": paths[position >> 32].name, "line": position & 0xFFFFFFFF}
            example["total_occurrences"] = duplicates[key] + 1
        top = [{"text": first_texts[key], "occurrences_before": extra + 1, "occurrences_after": 1,
                "removed": extra,
                "first_source": {"file": paths[first_refs[key] >> 32].name, "line": first_refs[key] & 0xFFFFFFFF}}
               for key, extra in duplicates.most_common(15)]
        (staging / "sentences_dedup").rename(output)
        (staging / removed_path.name).rename(removed_path)
        summary = {"input_sentences": total, "kept_sentences": kept, "removed_sentences": total - kept,
                   "removed_percent": (total - kept) / total * 100 if total else 0.0,
                   "duplicate_groups": len(duplicates), "normalization": "NFKC + whitespace; English case preserved",
                   "examples": examples, "top_duplicates": top, "shards": shards,
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports/step4-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k not in ["examples", "top_duplicates", "shards"]}, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_full()
