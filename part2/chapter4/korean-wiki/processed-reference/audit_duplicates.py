"""읽기 전용 중복 검사. 원본 문장, 토큰 및 ID 배열은 수정하지 않는다.

실행: uv run --no-project --with numpy python audit_duplicates.py
결과: audit/duplicates.json. 숫자/표기 변형은 중복 확정이 아닌 후보로 분리한다.
"""

import argparse
import gc
import hashlib
import json
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


START = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - START:.1f}s] {message}", flush=True)


def digest(value):
    return hashlib.sha256(value).digest()


def normalize_sentence(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def location(shard, line):
    return (shard << 32) | line


def reference(stage, names, encoded, kind="line"):
    shard, offset = encoded >> 32, encoded & 0xFFFFFFFF
    return {"file": f"{stage}/{names[shard]}", kind: offset}


def read_text_locations(root, stage, names, locations):
    wanted = defaultdict(set)
    for encoded in locations:
        wanted[encoded >> 32].add(encoded & 0xFFFFFFFF)
    result = {}
    for shard, line_numbers in wanted.items():
        with (root / stage / names[shard]).open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if line_no in line_numbers:
                    result[location(shard, line_no)] = line.rstrip("\r\n")
                if line_no >= max(line_numbers):
                    break
    return result


def sentence_audit(root):
    """전역 정규화 문장 집합을 전후에 대조하며, 출력 중복도 직접 센다."""
    seen = {}
    counts = Counter()
    names = [p.name for p in sorted((root / "sentences").glob("part_*.txt"))]
    out_names = [p.name for p in sorted((root / "sentences_dedup").glob("part_*.txt"))]
    if names != out_names or not names:
        raise ValueError("문장 샤드 구성이 일치하지 않습니다")
    before = 0
    for shard, name in enumerate(names):
        with (root / "sentences" / name).open(encoding="utf-8") as source:
            for line_no, text in enumerate(source, 1):
                text = text.rstrip("\r\n")
                if not text:
                    continue
                before += 1
                key = digest(normalize_sentence(text).encode())
                if key in seen:
                    counts[key] += 1
                else:
                    seen[key] = location(shard, line_no)
        if shard % 8 == 7 or shard == len(names) - 1:
            log(f"원본 문장 검사 {shard + 1}/{len(names)} 샤드: {before:,}개")
    input_unique, input_groups = len(seen), len(counts)
    top = counts.most_common(10)
    selected = [seen[key] for key, _ in top]
    texts = read_text_locations(root, "sentences", names, selected)
    source_examples = [
        {"occurrences": extra + 1, "first": reference("sentences", names, seen[key]),
         "text": texts[seen[key]]}
        for key, extra in top
    ]
    counts.clear()
    total = duplicate_extra = introduced = 0
    output_groups = set()
    examples = []
    for shard, name in enumerate(out_names):
        with (root / "sentences_dedup" / name).open(encoding="utf-8") as source:
            for line_no, text in enumerate(source, 1):
                text = text.rstrip("\r\n")
                if not text:
                    continue
                total += 1
                key = digest(normalize_sentence(text).encode())
                previous = seen.get(key)
                current = location(shard, line_no)
                if previous is None:
                    introduced += 1
                    seen[key] = -current
                elif previous < 0:
                    duplicate_extra += 1
                    output_groups.add(key)
                    if len(examples) < 12:
                        examples.append({"first": reference("sentences_dedup", names, -previous),
                                         "repeat": reference("sentences_dedup", names, current),
                                         "text": text})
                else:
                    seen[key] = -current
        if shard % 8 == 7 or shard == len(names) - 1:
            log(f"중복 제거 문장 검사 {shard + 1}/{len(names)} 샤드: {total:,}개")
    missing = sum(value > 0 for value in seen.values())
    return {
        "rule": "Unicode NFKC + 모든 공백을 한 칸으로 정리; 영문 대소문자 보존",
        "hash": "SHA-256 of normalized UTF-8 text",
        "source": {"sentences": before, "unique_sentences": input_unique,
                   "duplicate_extra_occurrences": before - input_unique,
                   "duplicate_groups": input_groups, "top_examples": source_examples},
        "sentences_dedup": {"sentences": total, "unique_sentences": total - duplicate_extra,
                            "duplicate_extra_occurrences": duplicate_extra,
                            "duplicate_groups": len(output_groups), "examples": examples},
        "missing_source_unique_sentences": missing,
        "introduced_sentences": introduced,
        "same_normalized_sentence_set": missing == 0 and introduced == 0,
    }


def token_audit(root):
    names = [p.name for p in sorted((root / "tokens").glob("part_*.txt"))]
    seen = {}
    canonical_counts = Counter()
    raw_groups = set()
    canonical_examples = []
    total = raw_duplicates = canonical_duplicates = 0
    suffix = b" <sp>/WS"
    for shard, name in enumerate(names):
        with (root / "tokens" / name).open("rb") as source:
            for line_no, line in enumerate(source, 1):
                raw = line.rstrip(b"\r\n")
                if not raw:
                    continue
                total += 1
                trailing = int(raw.endswith(suffix))
                canonical = raw.removesuffix(suffix)
                key = digest(canonical)
                value = seen.get(key)
                current = location(shard, line_no)
                flag = 1 << trailing
                if value is None:
                    # 아래 2비트는 관찰한 끝 공백 종류, 나머지는 최초 위치.
                    seen[key] = (current << 2) | flag
                    continue
                canonical_duplicates += 1
                canonical_counts[key] += 1
                if value & flag:
                    raw_duplicates += 1
                    raw_groups.add((key, trailing))
                else:
                    seen[key] = value | flag
                if canonical_counts[key] == 1 and len(canonical_examples) < 12:
                    canonical_examples.append({"first_encoded": value >> 2,
                                               "repeat_encoded": current,
                                               "tokens": canonical.decode()})
        if shard % 8 == 7 or shard == len(names) - 1:
            log(f"형태소 토큰 검사 {shard + 1}/{len(names)} 샤드: {total:,}개")
    refs = [x[k] for x in canonical_examples for k in ["first_encoded", "repeat_encoded"]]
    source_texts = read_text_locations(root, "sentences_dedup", names, refs)
    examples = []
    for item in canonical_examples:
        first, repeat = item["first_encoded"], item["repeat_encoded"]
        examples.append({"first": reference("tokens", names, first),
                         "repeat": reference("tokens", names, repeat),
                         "first_source_sentence": source_texts[first],
                         "repeat_source_sentence": source_texts[repeat],
                         "tokens": item["tokens"]})
    return {"sentences": total,
            "exact_token_line_duplicate_extra_occurrences": raw_duplicates,
            "exact_token_line_duplicate_groups": len(raw_groups),
            "ignoring_final_space_token_duplicate_extra_occurrences": canonical_duplicates,
            "ignoring_final_space_token_duplicate_groups": len(canonical_counts),
            "examples": examples,
            "note": "문장 끝 <sp>/WS 유무를 제외한 비교를 별도로 집계한다. 원문이 달라도 형태소 열은 같아질 수 있다."}


def variant_audit(root, mode):
    names = [p.name for p in sorted((root / "sentences_dedup").glob("part_*.txt"))]
    seen = {}
    counts = Counter()
    pairs = []
    total = duplicates = 0
    style_pattern = re.compile(r"[\W_]+")
    number_pattern = re.compile(r"\d+(?:[.,]\d+)*")
    for shard, name in enumerate(names):
        with (root / "sentences_dedup" / name).open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                text = line.rstrip("\r\n")
                if not text:
                    continue
                total += 1
                normalized = normalize_sentence(text)
                canonical = (style_pattern.sub("", normalized.casefold()) if mode == "style"
                             else number_pattern.sub("<NUMBER>", normalized))
                key = digest(canonical.encode())
                current = location(shard, line_no)
                previous = seen.get(key)
                if previous is None:
                    seen[key] = current
                    continue
                duplicates += 1
                counts[key] += 1
                if counts[key] == 1 and len(pairs) < 12 and len(canonical) >= 20:
                    pairs.append((previous, current))
        if shard % 8 == 7 or shard == len(names) - 1:
            log(f"변형 후보({mode}) {shard + 1}/{len(names)} 샤드: {total:,}개")
    texts = read_text_locations(root, "sentences_dedup", names, [x for pair in pairs for x in pair])
    return {"sentences": total, "candidate_extra_occurrences": duplicates,
            "candidate_groups": len(counts),
            "rule": ("NFKC 후 공백·문장부호 제거 및 영문 casefold" if mode == "style"
                     else "NFKC 및 공백 정리 후 숫자열을 <NUMBER>로 치환"),
            "classification": "유사/서식 후보이며 동일 데이터로 확정한 수치가 아님",
            "examples": [{"first": reference("sentences_dedup", names, a),
                          "repeat": reference("sentences_dedup", names, b),
                          "first_text": texts[a], "repeat_text": texts[b]} for a, b in pairs]}


def id_paragraph_audit(root, size):
    stage = f"token_ids_{size}"
    names = [p.name for p in sorted((root / stage).glob("part_*.npy"))]
    seen = {}
    counts = Counter()
    total = duplicates = token_count = 0
    for shard, name in enumerate(names):
        ids = np.load(root / stage / name, mmap_mode="r", allow_pickle=False)
        if ids.dtype != np.dtype("<u2") or ids.ndim != 1:
            raise ValueError("잘못된 ID 배열 형식")
        token_count += ids.size
        ends = np.flatnonzero(ids == 3)
        starts = np.flatnonzero(ids == 2)
        if len(starts) != len(ends) or (ids.size and (not len(ends) or ends[-1] != ids.size - 1)):
            raise ValueError("문단 시작·끝 마커가 일치하지 않습니다")
        expected_start = 0
        for start, end in zip(starts, ends):
            start, end = int(start), int(end)
            if start != expected_start or end <= start:
                raise ValueError("문단 마커 순서가 잘못되었습니다")
            expected_start = end + 1
            total += 1
            key = digest(ids[start:end + 1].tobytes())
            if key in seen:
                duplicates += 1
                counts[key] += 1
            else:
                seen[key] = location(shard, start)
        if shard % 16 == 15 or shard == len(names) - 1:
            log(f"ID 문단({size}) {shard + 1}/{len(names)} 샤드: {total:,}개")
    vocab_path = root / f"vocab_{size}.txt"
    vocab = [line.split("\t", 1)[0] for line in vocab_path.read_text().splitlines()] if vocab_path.exists() else None
    examples = []
    for key, extra in counts.most_common(8):
        encoded = seen[key]
        shard, start = encoded >> 32, encoded & 0xFFFFFFFF
        ids = np.load(root / stage / names[shard], mmap_mode="r", allow_pickle=False)
        tail = ids[start:]
        end = int(np.flatnonzero(tail == 3)[0]) + 1
        prefix = tail[:min(end, 64)].tolist()
        examples.append({"occurrences": extra + 1,
                         "first": reference(stage, names, encoded, "array_offset"),
                         "paragraph_tokens": end,
                         "token_ids_prefix": prefix,
                         "tokens_prefix": [vocab[i] for i in prefix] if vocab else None})
    return {"vocab_size": size, "tokens": int(token_count), "paragraphs": total,
            "unique_paragraphs": len(seen), "duplicate_extra_occurrences": duplicates,
            "duplicate_groups": len(counts), "examples": examples,
            "rule": "동일 vocab 내 <bos>부터 <eos>까지의 전체 ID 시퀀스 완전 일치"}


def self_check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for stage in ["sentences", "sentences_dedup", "tokens", "token_ids_16"]:
            (root / stage).mkdir()
        (root / "sentences/part_00.txt").write_text("서울은 수도다.\n서울은   수도다.\n서울은 수도이다.\n부산은 항구다.\n")
        (root / "sentences_dedup/part_00.txt").write_text("서울은 수도다.\n서울은 수도이다.\n부산은 항구다.\n")
        same = "서울/NNP 은/JX 수도/NNG 이/VCP 다/EF ./SF"
        (root / "tokens/part_00.txt").write_text(same + " <sp>/WS\n" + same + "\n부산/NNP 항구/NNG ./SF\n")
        source = sentence_audit(root)
        assert source["source"]["duplicate_extra_occurrences"] == 1
        assert source["sentences_dedup"]["duplicate_extra_occurrences"] == 0
        assert source["same_normalized_sentence_set"]
        tokens = token_audit(root)
        assert tokens["exact_token_line_duplicate_extra_occurrences"] == 0
        assert tokens["ignoring_final_space_token_duplicate_extra_occurrences"] == 1
        np.save(root / "token_ids_16/part_00.npy", np.array([2, 10, 3, 2, 11, 3, 2, 10, 3], dtype="<u2"))
        ids = id_paragraph_audit(root, 16)
        assert ids["paragraphs"] == 3 and ids["duplicate_extra_occurrences"] == 1
    log("검사기 검증 데이터 통과: 정규화 중복, 공백 토큰 변형, ID 문단 중복")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    root = args.dataset.resolve()
    output = root / "audit/duplicates.json"
    output.parent.mkdir(exist_ok=True)
    result = {"dataset": str(root), "status": "running", "modified_corpus": False}

    def save():
        result["elapsed_seconds"] = round(time.monotonic() - START, 2)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(output)

    result["sentences"] = sentence_audit(root)
    save()
    gc.collect()
    result["tokens"] = token_audit(root)
    save()
    gc.collect()
    result["style_variants"] = variant_audit(root, "style")
    save()
    gc.collect()
    result["number_templates"] = variant_audit(root, "numbers")
    save()
    gc.collect()
    result["id_paragraphs"] = []
    for size in [16384, 32768, 49152, 65536]:
        result["id_paragraphs"].append(id_paragraph_audit(root, size))
        save()
        gc.collect()
    result["status"] = "complete"
    save()
    log(f"완료: {output}")
    print(json.dumps({"sentences": result["sentences"]["sentences_dedup"],
                      "token_duplicate_extra": result["tokens"]["exact_token_line_duplicate_extra_occurrences"],
                      "token_boundary_ignored_duplicate_extra": result["tokens"]["ignoring_final_space_token_duplicate_extra_occurrences"],
                      "style_candidates": result["style_variants"]["candidate_extra_occurrences"],
                      "number_template_candidates": result["number_templates"]["candidate_extra_occurrences"],
                      "ids": [{k: v for k, v in item.items() if k != "examples"} for item in result["id_paragraphs"]]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
