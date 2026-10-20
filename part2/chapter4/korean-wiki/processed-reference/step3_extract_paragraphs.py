"""
FineWiki kowiki parquet → 문단/문장 추출 (1차 청소).

필터 파이프라인 (실행 순서):

  1. has_formula       '{\\displaystyle' 포함 문서 전체 제외  (문서 단위)
  2. is_structural     헤딩·불릿·번호·표·빈 줄 제거           (라인 단위)
  3. has_math_symbol   수학 전용 기호(±∓≠≤≥∈∪∑√∞) 포함 제거
  4. hanja_ratio       한자/(한자+한글) 비율 30% 초과 제거
  5. clean_paragraph   비한글 letter 포함 괄호 제거, 공백 정리, '. .' → '.'
  6. letter_ratio      isalpha 비율 20% 미만 제거
  7. hangul_ratio      한글 비율 30% 미만 제거
  8. is_sentence_shape 종결 부호 '.', '?', '!' 로 끝나야 함

출력 (3종):
  paragraphs_raw/part_NN.txt  step 1-2 적용, 라인 content filter 전 (비교용)
  paragraphs/part_NN.txt      step 1-8 모두 적용, 문단 단위 최종본
  sentences/part_NN.txt       step 1-8 + 문장 분할, 한 줄 = 한 문장

포맷:
  - paragraphs*: 한 줄 = 한 문단, 문단 사이 빈 줄 1개
  - sentences:   한 줄 = 한 문장, 같은 문단 문장끼리 인접, 문단 사이 빈 줄 1개
  - 문서 경계 표시 없음, 샤드 간 문단 수 균등 (round-robin 배분)
"""

import argparse
import os
import random
import re
import sys
import time
from multiprocessing import Pool

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILES = [
    "parquet/data/kowiki/000_00000.parquet",
    "parquet/data/kowiki/000_00001.parquet",
]
OUT_DIR_RAW = os.path.join(DATA_DIR, "paragraphs_raw")
OUT_DIR = os.path.join(DATA_DIR, "paragraphs")
OUT_DIR_SENT = os.path.join(DATA_DIR, "sentences")
N_SHARDS = 32

# 수학 전용 Unicode 기호 (일상 기호 ×÷°½→←πασ 등은 제외)
MATH_SYMBOLS = frozenset("±∓≠≤≥∈∪∑√∞")
_BRACKET_RE = re.compile(r"\(([^()]*)\)")  # 안쪽(비중첩) 괄호 매칭용
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # 구두점 보존 lookbehind
# 분할 후 머지용: 끝이 ' [A-Z].' 인 조각 (외국 인물 미들 이니셜)
_INITIAL_END_RE = re.compile(r"(?:^|\s)[A-Z]\.$")
# 분할 후 삭제용: 항목 번호 노이즈 (e.g., "1.", "2.", "10.")
_LIST_MARKER_RE = re.compile(r"^\d+\.$")
HANJA_RATIO_MAX = 0.30
LETTER_RATIO_MIN = 0.20
HANGUL_RATIO_MIN = 0.30


def is_structural(line: str) -> bool:
    """헤딩/리스트/표/빈 줄이면 True."""
    s = line.strip()
    if not s:
        return True
    c0 = s[0]
    if c0 == "#":
        return True
    if c0 == "|":
        return True
    if c0 in "-*" and len(s) >= 2 and s[1] == " ":
        return True
    # 번호 리스트 "1. item" (각주 "1. ↑ ..." 포함)
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if 0 < i < len(s) - 1 and s[i] == "." and s[i + 1] == " ":
        return True
    return False


def has_formula(s: str) -> bool:
    """LaTeX displaystyle 수식 포함 여부 (문서 단위)."""
    return "{\\displaystyle" in s


def has_math_symbol(line: str) -> bool:
    """수학 전용 Unicode 기호(MATH_SYMBOLS) 포함 여부."""
    return any(ch in MATH_SYMBOLS for ch in line)


def _is_hangul(code: int) -> bool:
    return (
        0xAC00 <= code <= 0xD7A3  # Hangul Syllables
        or 0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x3130 <= code <= 0x318F  # Hangul Compatibility Jamo
    )


def _is_hanja(code: int) -> bool:
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
    )


def hanja_ratio(s: str) -> float:
    """한자 / (한자 + 한글) 비율. 둘 다 없으면 0.0."""
    n_hangul = n_hanja = 0
    for ch in s:
        code = ord(ch)
        if _is_hangul(code):
            n_hangul += 1
        elif _is_hanja(code):
            n_hanja += 1
    total = n_hangul + n_hanja
    return n_hanja / total if total else 0.0


def _has_non_korean_letter(s: str) -> bool:
    """한글 외 letter(한자/라틴/그리스 등) 포함 여부. 숫자/기호는 해당 없음."""
    return any(ch.isalpha() and not _is_hangul(ord(ch)) for ch in s)


def clean_paragraph(s: str) -> str:
    """비한글 letter 포함 괄호를 안쪽부터 반복 제거. 공백 정리, '. .' → '.'

    순수 한글/숫자/기호 괄호(예: '(약 100명)')는 보존. 중첩 괄호는 placeholder로
    치환해 안쪽→바깥쪽 순으로 처리하고 역순으로 복원한다.
    """
    placeholders: dict[str, str] = {}
    counter = 0
    while True:
        m = _BRACKET_RE.search(s)
        if m is None:
            break
        if _has_non_korean_letter(m.group(1)):
            s = s[: m.start()] + s[m.end() :]
        else:
            key = f"\x00{counter}\x00"
            placeholders[key] = m.group(0)
            s = s[: m.start()] + key + s[m.end() :]
            counter += 1
    # reversed: 바깥쪽 placeholder를 먼저 복원해야 중첩 안쪽 키가 단계별로 노출됨
    for key, original in reversed(placeholders.items()):
        s = s.replace(key, original)
    s = _WS_RE.sub(" ", s).strip()
    while ". ." in s:
        s = s.replace(". .", ".")
    return s


def letter_ratio(s: str) -> float:
    """isalpha 비율. 빈 문자열은 0.0."""
    return sum(ch.isalpha() for ch in s) / len(s) if s else 0.0


def hangul_ratio(s: str) -> float:
    """한글 / 전체 문자 비율. 빈 문자열은 0.0.

    hanja_ratio 는 분모가 한자+한글인 것과 다름.
    """
    return sum(_is_hangul(ord(ch)) for ch in s) / len(s) if s else 0.0


def is_sentence_shape(s: str) -> bool:
    """종결 부호 '.?!' 로 끝나고 2자 이상인지."""
    s = s.rstrip()
    return len(s) >= 2 and s[-1] in ".?!"


def split_sentences(paragraph: str) -> list[str]:
    """종결 부호 뒤 공백에서 문장 분할. '3.14' 같은 소수점은 분할하지 않음.

    1차로 단순 분할 후 두 단계 후처리:
      (a) 끝이 ' [A-Z].' 인 조각(미들 이니셜로 잘린 fragment)은 다음 조각과 머지
      (b) bare digit marker ('1.', '2.', '10.' …) 는 항목 번호 노이즈로 보고 삭제

      "1953년 제임스 D. Watson은..."
      → 1차: ["1953년 제임스 D.", "Watson은..."]
      → 머지: ["1953년 제임스 D. Watson은..."]

      "방법은 다음과 같다. 1. 준비물을 갖춘다. 2. 작업을 시작한다."
      → 1차: ["방법은 다음과 같다.", "1.", "준비물을 갖춘다.", "2.", "작업을 시작한다."]
      → 삭제: ["방법은 다음과 같다.", "준비물을 갖춘다.", "작업을 시작한다."]

    'James D. R. Smith' 같이 연속 이니셜도 자동 처리됨 (in-place 갱신).
    """
    parts = [p for p in _SENT_SPLIT_RE.split(paragraph.strip()) if p]
    if not parts:
        return parts
    merged: list[str] = [parts[0]]
    for nxt in parts[1:]:
        if _INITIAL_END_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + nxt
        else:
            merged.append(nxt)
    # 항목 번호 노이즈 제거 (e.g., "1.", "2.", "10.")
    return [s for s in merged if not _LIST_MARKER_RE.match(s)]


def _write_paragraphs(
    out_raw, out_clean, out_sent, texts
) -> tuple[int, int, int, int, int, int]:
    """texts 를 순회해 3개 출력 파일에 기록.

    out_raw:   step 1-2 통과 문단 (비교용)
    out_clean: step 1-8 통과 문단 (최종본)
    out_sent:  clean 문단을 문장 단위로 분할
    반환: (n_docs, n_paras_raw, n_chars_raw, n_paras_clean, n_chars_clean, n_sents)
    """
    n_docs = n_paras_raw = n_chars_raw = n_paras_clean = n_chars_clean = n_sents = 0
    for text in texts:
        n_docs += 1
        if text is None:
            continue
        if has_formula(text):                       # step 1: 문서 전체 제외
            continue
        for line in text.split("\n"):
            if is_structural(line):                 # step 2: 구조 라인 제거
                continue

            stripped = line.strip()
            out_raw.write(stripped)
            out_raw.write("\n\n")
            n_paras_raw += 1
            n_chars_raw += len(stripped)

            # step 3-8: clean/sentences 출력 전용
            if has_math_symbol(line):               # step 3
                continue
            if hanja_ratio(line) > HANJA_RATIO_MAX: # step 4
                continue
            cleaned = clean_paragraph(line)         # step 5
            if not cleaned:
                continue
            if letter_ratio(cleaned) < LETTER_RATIO_MIN:  # step 6
                continue
            if hangul_ratio(cleaned) < HANGUL_RATIO_MIN:  # step 7
                continue
            if not is_sentence_shape(cleaned):      # step 8
                continue

            out_clean.write(cleaned)
            out_clean.write("\n\n")
            n_paras_clean += 1
            n_chars_clean += len(cleaned)

            for sent in split_sentences(cleaned):
                out_sent.write(sent)
                out_sent.write("\n")
                n_sents += 1
            out_sent.write("\n")
    return n_docs, n_paras_raw, n_chars_raw, n_paras_clean, n_chars_clean, n_sents


def _process_shard(args):
    """Worker: 할당받은 row group들을 처리해 3개 샤드 파일에 기록."""
    shard_idx, tasks = args  # tasks: list[tuple[file_idx, rg_idx]]
    out_path_raw = os.path.join(OUT_DIR_RAW, f"part_{shard_idx:02d}.txt")
    out_path_clean = os.path.join(OUT_DIR, f"part_{shard_idx:02d}.txt")
    out_path_sent = os.path.join(OUT_DIR_SENT, f"part_{shard_idx:02d}.txt")

    by_file: dict[int, list[int]] = {}
    for file_idx, rg in tasks:
        by_file.setdefault(file_idx, []).append(rg)

    n_docs = n_paras_raw = n_chars_raw = n_paras_clean = n_chars_clean = n_sents = 0

    with (
        open(out_path_raw, "w", encoding="utf-8") as out_raw,
        open(out_path_clean, "w", encoding="utf-8") as out_clean,
        open(out_path_sent, "w", encoding="utf-8") as out_sent,
    ):
        for file_idx, rgs in by_file.items():
            pf = pq.ParquetFile(os.path.join(DATA_DIR, PARQUET_FILES[file_idx]))
            for rg in sorted(rgs):
                texts = (
                    pf.read_row_group(rg, columns=["text"])
                    .column("text")
                    .to_pylist()
                )
                d, pr, cr, pc, cc, ns = _write_paragraphs(
                    out_raw, out_clean, out_sent, texts
                )
                n_docs += d
                n_paras_raw += pr
                n_chars_raw += cr
                n_paras_clean += pc
                n_chars_clean += cc
                n_sents += ns
    return (
        shard_idx,
        n_docs,
        n_paras_raw,
        n_chars_raw,
        n_paras_clean,
        n_chars_clean,
        n_sents,
    )


def _run_test(n_rgs_req: int, shuffle_seed: int | None = None) -> None:
    """테스트 모드: 단일 프로세스로 앞 N개 row group만 처리.

    출력: paragraphs_test_raw.txt / paragraphs_test.txt / sentences_test.txt
    shuffle_seed 지정 시 문서 순서를 해당 seed 로 셔플해 출력.
    """
    out_path_raw = os.path.join(DATA_DIR, "paragraphs_test_raw.txt")
    out_path_clean = os.path.join(DATA_DIR, "paragraphs_test.txt")
    out_path_sent = os.path.join(DATA_DIR, "sentences_test.txt")

    print("=" * 60)
    print(f"FineWiki kowiki → 문단 추출 [TEST MODE: 앞 {n_rgs_req} row group]")
    print("=" * 60)

    fpath_rel = PARQUET_FILES[0]
    fpath = os.path.join(DATA_DIR, fpath_rel)
    pf = pq.ParquetFile(fpath)
    total_rgs = pf.num_row_groups
    n_rgs = min(n_rgs_req, total_rgs)
    print(
        f"  {fpath_rel}: {pf.metadata.num_rows:,} rows, "
        f"{total_rgs} row groups → 앞 {n_rgs}개 사용"
    )

    t0 = time.time()

    all_texts = []
    for rg in range(n_rgs):
        all_texts.extend(
            pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        )
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(all_texts)

    with (
        open(out_path_raw, "w", encoding="utf-8") as out_raw,
        open(out_path_clean, "w", encoding="utf-8") as out_clean,
        open(out_path_sent, "w", encoding="utf-8") as out_sent,
    ):
        n_docs, n_paras_raw, n_chars_raw, n_paras_clean, n_chars_clean, n_sents = (
            _write_paragraphs(out_raw, out_clean, out_sent, all_texts)
        )

    elapsed = time.time() - t0
    sz_mb_raw = os.path.getsize(out_path_raw) / 1024 / 1024
    sz_mb_clean = os.path.getsize(out_path_clean) / 1024 / 1024
    sz_mb_sent = os.path.getsize(out_path_sent) / 1024 / 1024
    drop_paras = (1 - n_paras_clean / max(n_paras_raw, 1)) * 100
    drop_chars = (1 - n_chars_clean / max(n_chars_raw, 1)) * 100
    sents_per_para = n_sents / max(n_paras_clean, 1)

    print("\n" + "=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"문서 수:      {n_docs:,}")
    print()
    print(f"{'':12s}{'raw':>14s}  {'clean':>14s}  {'Δ':>10s}")
    print(f"{'문단 수':12s}{n_paras_raw:>14,}  {n_paras_clean:>14,}  {-drop_paras:>9.1f}%")
    print(f"{'문자 수':12s}{n_chars_raw:>14,}  {n_chars_clean:>14,}  {-drop_chars:>9.1f}%")
    print(f"{'파일 크기':12s}{sz_mb_raw:>13.1f}M  {sz_mb_clean:>13.1f}M")
    print()
    print(f"문장 수:      {n_sents:,}  ({sents_per_para:.1f} 문장/문단)")
    print(f"문장 파일:    {sz_mb_sent:.1f}M")
    print()
    print("출력: paragraphs_test_raw.txt / paragraphs_test.txt / sentences_test.txt")
    print("=" * 60)



def _run_full() -> None:
    """전체 변환: 모든 parquet의 모든 row group을 32 샤드로 멀티프로세싱 처리."""
    os.makedirs(OUT_DIR_RAW, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_SENT, exist_ok=True)

    print("=" * 60)
    print(f"FineWiki kowiki → 문단 추출 ({N_SHARDS} shards, multiprocessing)")
    print("=" * 60)

    all_tasks = []
    for file_idx, fpath_rel in enumerate(PARQUET_FILES):
        pf = pq.ParquetFile(os.path.join(DATA_DIR, fpath_rel))
        n_rgs = pf.num_row_groups
        print(f"  {fpath_rel}: {pf.metadata.num_rows:,} rows, {n_rgs} row groups")
        for rg in range(n_rgs):
            all_tasks.append((file_idx, rg))

    # 샤드에 round-robin 배분 (샤드별 row group 수 거의 균등)
    shard_tasks = [[] for _ in range(N_SHARDS)]
    for i, task in enumerate(all_tasks):
        shard_tasks[i % N_SHARDS].append(task)

    print(f"\nLaunching {N_SHARDS} workers (total {len(all_tasks)} row groups) ...")
    t0 = time.time()
    with Pool(N_SHARDS) as pool:
        results = pool.map(_process_shard, [(i, shard_tasks[i]) for i in range(N_SHARDS)])
    elapsed = time.time() - t0

    total_docs = sum(r[1] for r in results)
    total_paras_raw = sum(r[2] for r in results)
    total_chars_raw = sum(r[3] for r in results)
    total_paras_clean = sum(r[4] for r in results)
    total_chars_clean = sum(r[5] for r in results)
    total_sents = sum(r[6] for r in results)
    drop_paras = (1 - total_paras_clean / max(total_paras_raw, 1)) * 100
    drop_chars = (1 - total_chars_clean / max(total_chars_raw, 1)) * 100

    print("\n" + "=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"문서 수:      {total_docs:,}")
    print()
    print(f"{'':12s}{'raw':>14s}  {'clean':>14s}  {'Δ':>10s}")
    print(
        f"{'문단 수':12s}{total_paras_raw:>14,}  "
        f"{total_paras_clean:>14,}  {-drop_paras:>9.1f}%"
    )
    print(
        f"{'문자 수':12s}{total_chars_raw:>14,}  "
        f"{total_chars_clean:>14,}  {-drop_chars:>9.1f}%"
    )
    print()
    avg_len_clean = total_chars_clean / max(total_paras_clean, 1)
    sents_per_para = total_sents / max(total_paras_clean, 1)
    print(f"평균 clean 문단 길이: {avg_len_clean:.1f} 자")
    print(f"문장 수:      {total_sents:,}  ({sents_per_para:.1f} 문장/문단)")
    print()
    print("샤드별 (raw → clean 문단 / 문장 수, clean 파일 크기):")
    for shard_idx, _, pr, _, pc, _, ns in sorted(results):
        fp = os.path.join(OUT_DIR, f"part_{shard_idx:02d}.txt")
        sz_mb = os.path.getsize(fp) / 1024 / 1024
        print(
            f"  part_{shard_idx:02d}.txt: {pr:>8,} → {pc:>8,} 문단 / "
            f"{ns:>8,} 문장  {sz_mb:>7.1f} MB"
        )
    print("=" * 60)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FineWiki kowiki parquet → 문단/문장 추출"
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help=(
            "테스트 모드: 첫 parquet의 앞 N개 row group만 단일 프로세스로 처리 "
            "(1 row group ≈ 1,000 문서, 기본 N=1)"
        ),
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="--test 전용: 문서 순서를 SEED 로 셔플 (예: --shuffle 1)",
    )
    args = parser.parse_args()

    if args.test is not None:
        _run_test(args.test, shuffle_seed=args.shuffle)
    else:
        _run_full()


if __name__ == "__main__":
    main()
