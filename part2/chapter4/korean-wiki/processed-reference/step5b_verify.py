"""
step5a 토큰화 round-trip 검증.

토큰열을 Kiwi.join() 으로 표면형 복원하여 원본과 비교.

분류:
  exact      완전 일치
  mismatch   불일치 (형태소 분석기의 표면형 정규화 등)

참고:
  Kiwi 의 원형 복원 ("도로다" → "도로이다") 과 축약 ("되어" → "돼") 때문에
  100% 일치는 원리적으로 불가능. 참고용 품질 지표로 사용.

입력:
  - sentences_dedup/part_NN.txt   (원본)
  - tokens/part_NN.txt            (step5a 출력)

테스트 모드:
  - sentences_dedup/part_NN.txt (앞 N 샤드)  vs  tokens_test.txt (step5a --test 출력)
"""

import argparse
import difflib
import os
import sys
import time
from multiprocessing import Pool

from kiwipiepy import Kiwi

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(DATA_DIR, "sentences_dedup")
TOK_DIR = os.path.join(DATA_DIR, "tokens")
TEST_TOK_PATH = os.path.join(DATA_DIR, "tokens_test.txt")
MISMATCH_LOG_PATH = os.path.join(DATA_DIR, "tokens_verify_mismatches.txt")
TEST_MISMATCH_LOG_PATH = os.path.join(DATA_DIR, "tokens_verify_mismatches_test.txt")
KIWI_MODEL = "cong-global"
N_SHARDS = 32
DEFAULT_MISMATCH_SAMPLES = 30

SPACE_TOKEN = "<sp>/WS"


def detokenize(tokens_str: str, kiwi: Kiwi) -> str:
    """토큰 시퀀스 → 원문 표면형 복원.

    <sp>/WS 위치를 어절 시작 마커로 변환하여 Kiwi.join 호출.
    form 안의 '_' 는 ' ' 로 복원 (step5a 의 escape 역변환).
    """
    tokens = tokens_str.split(" ")
    morphs: list[tuple[str, str, bool]] = []
    next_is_word_start = True
    for tok in tokens:
        if tok == SPACE_TOKEN:
            next_is_word_start = True
            continue
        slash_idx = tok.rfind("/")
        form = tok[:slash_idx].replace("_", " ")
        tag = tok[slash_idx + 1 :]
        morphs.append((form, tag, next_is_word_start))
        next_is_word_start = False
    return kiwi.join(morphs)


def find_first_diff(a: str, b: str) -> int:
    """첫 번째 차이 문자 위치."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def _format_diffs(orig: str, rest: str) -> str:
    """두 문자열의 차이 부분만 추출하여 한 줄 요약.

    예: '되어서' → '돼서' (pos 3), '이다' → '다' (pos 15)
    """
    sm = difflib.SequenceMatcher(None, orig, rest)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        o = orig[i1:i2]
        r = rest[j1:j2]
        if tag == "replace":
            parts.append(f"'{o}' → '{r}' (pos {i1})")
        elif tag == "delete":
            parts.append(f"'{o}' 삭제 (pos {i1})")
        elif tag == "insert":
            parts.append(f"'{r}' 삽입 (pos {i1})")
    return ", ".join(parts)


def _verify_streaming(
    src_iter,
    tok_iter,
    kiwi: Kiwi,
    mismatches: list,
    max_samples: int,
    log_file=None,
) -> tuple[int, int, int, int]:
    """src/tok 라인 이터레이터를 streaming 비교.

    mismatches: 콘솔 미리보기용 (max_samples 개까지)
    log_file:   모든 불일치를 기록할 파일 핸들 (None 이면 기록 안 함)
    반환: (n_total, n_exact, n_mismatch, n_error)
    """
    n_total = n_exact = n_mismatch = n_error = 0
    line_no = 0

    for src_line, tok_line in zip(src_iter, tok_iter):
        line_no += 1
        src_line = src_line.rstrip("\n")
        tok_line = tok_line.rstrip("\n")

        if not src_line:
            continue
        if not tok_line:
            continue

        n_total += 1
        try:
            # rstrip: 비마지막 문장의 trailing <sp>/WS가 복원되면
            # 공백이 생기므로 원본 비교 전에 제거
            restored = detokenize(tok_line, kiwi).rstrip()
        except Exception as e:
            n_error += 1
            msg = f"<JOIN ERROR: {e}>"
            if len(mismatches) < max_samples:
                mismatches.append((line_no, src_line, msg, "error"))
            if log_file:
                log_file.write(f"[line {line_no}] error\n")
                log_file.write(f"  ORIG: {src_line}\n")
                log_file.write(f"  REST: {msg}\n\n")
            continue

        if src_line == restored:
            n_exact += 1
        else:
            n_mismatch += 1
            if len(mismatches) < max_samples:
                mismatches.append((line_no, src_line, restored, "mismatch"))
            if log_file:
                log_file.write(f"[line {line_no}]\n")
                log_file.write(f"  ORIG: {src_line}\n")
                log_file.write(f"  REST: {restored}\n")
                log_file.write(f"  DIFF: {_format_diffs(src_line, restored)}\n\n")

    return n_total, n_exact, n_mismatch, n_error


def _print_mismatches(mismatches: list) -> None:
    if not mismatches:
        print("\n(불일치 없음)")
        return
    print(f"\n=== 불일치 사례 ({len(mismatches)}개) ===")
    for i, (line_no, orig, rest, kind) in enumerate(mismatches, 1):
        print()
        print(f"[{i}] line {line_no} ({kind})")
        print(f"  ORIG: {orig}")
        print(f"  REST: {rest}")
        if kind == "mismatch":
            print(f"  DIFF: {_format_diffs(orig, rest)}")


def _print_summary(
    elapsed: float,
    n_total: int,
    n_exact: int,
    n_mismatch: int,
    n_error: int,
) -> None:
    def pct(n: int) -> str:
        return f"{n / max(n_total, 1) * 100:>6.2f}%"

    print()
    print("=" * 60)
    print(f"소요 시간:     {elapsed:.1f}s")
    print(f"검사 문장 수:   {n_total:,}")
    print()
    print(f"완전 일치:    {n_exact:>10,}  {pct(n_exact)}")
    print(f"불일치:       {n_mismatch:>10,}  {pct(n_mismatch)}")
    if n_error:
        print(f"join 오류:    {n_error:>10,}  {pct(n_error)}")
    print("=" * 60)


def _list_src_shards() -> list[str]:
    return sorted(
        os.path.join(SRC_DIR, f)
        for f in os.listdir(SRC_DIR)
        if f.endswith(".txt")
    )


def _verify_shard(args) -> tuple[int, str, int, int, int, int, list, str]:
    """워커: 샤드 1개 검증. 반환: (idx, name, total, exact, mismatch, error, mismatches, log_text)"""
    shard_idx, src_path, tok_path, max_samples = args
    name = os.path.basename(src_path)
    kiwi = Kiwi(num_workers=1, model_type=KIWI_MODEL)
    mismatches: list = []

    import io
    log_buf = io.StringIO()
    log_buf.write(f"# {name}\n")

    with (
        open(src_path, encoding="utf-8") as fs,
        open(tok_path, encoding="utf-8") as ft,
    ):
        n, ex, ms, er = _verify_streaming(
            fs, ft, kiwi, mismatches, max_samples, log_file=log_buf
        )

    return shard_idx, name, n, ex, ms, er, mismatches, log_buf.getvalue()


def _run_full(max_samples: int) -> None:
    src_paths = _list_src_shards()
    if not src_paths:
        print(f"입력 없음: {SRC_DIR}/*.txt")
        return
    if not os.path.isdir(TOK_DIR):
        print(f"토큰 디렉토리 없음: {TOK_DIR}  (먼저 step5a_tokenize.py 실행)")
        return

    # 토큰 파일 존재하는 샤드만 필터
    tasks = []
    for i, src_path in enumerate(src_paths):
        name = os.path.basename(src_path)
        tok_path = os.path.join(TOK_DIR, name)
        if os.path.exists(tok_path):
            tasks.append((i, src_path, tok_path, max_samples))
        else:
            print(f"  {name}: 토큰 파일 없음 — 건너뜀")

    n_workers = min(N_SHARDS, len(tasks))

    print("=" * 60)
    print(f"Step 5 round-trip 검증 ({len(tasks)} shards, {n_workers} workers)")
    print("=" * 60)
    print(f"원본:        {SRC_DIR}")
    print(f"토큰:        {TOK_DIR}")
    print()

    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_verify_shard, tasks)
    elapsed = time.time() - t0

    total = exact = mismatch = error = 0
    all_mismatches: list = []

    with open(MISMATCH_LOG_PATH, "w", encoding="utf-8") as log:
        for shard_idx, name, n, ex, ms, er, shard_mm, log_text in sorted(results):
            total += n
            exact += ex
            mismatch += ms
            error += er
            all_mismatches.extend(shard_mm)
            log.write(log_text)
            print(
                f"  {name}: {n:>8,} / "
                f"exact {ex:>8,} / mismatch {ms:>5,}"
                + (f" / err {er}" if er else "")
            )

    _print_summary(elapsed, total, exact, mismatch, error)
    _print_mismatches(all_mismatches[:max_samples])
    log_mb = os.path.getsize(MISMATCH_LOG_PATH) / 1024 / 1024
    print(f"불일치 로그:   {log_mb:.1f}M  {MISMATCH_LOG_PATH}")


def _run_test(n_shards: int, max_samples: int) -> None:
    """앞 N 입력 샤드 vs tokens_test.txt 비교."""
    src_paths = _list_src_shards()[:n_shards]
    if not src_paths:
        print(f"입력 없음: {SRC_DIR}/*.txt")
        return
    if not os.path.exists(TEST_TOK_PATH):
        print(
            f"토큰 파일 없음: {TEST_TOK_PATH}  "
            f"(먼저 step5a_tokenize.py --test {n_shards} 실행)"
        )
        return

    print("=" * 60)
    print(f"Step 5 round-trip 검증 [TEST MODE: 앞 {len(src_paths)} 샤드]")
    print("=" * 60)
    print(f"원본:        {SRC_DIR}")
    print(f"토큰:        tokens_test.txt")
    print()

    kiwi = Kiwi(num_workers=1, model_type=KIWI_MODEL)
    mismatches: list = []
    total = exact = mismatch = error = 0

    t0 = time.time()
    with (
        open(TEST_TOK_PATH, encoding="utf-8") as ft,
        open(TEST_MISMATCH_LOG_PATH, "w", encoding="utf-8") as log,
    ):
        for src_path in src_paths:
            name = os.path.basename(src_path)
            print(f"  verifying {name} ...")
            log.write(f"# {name}\n")
            with open(src_path, encoding="utf-8") as fs:
                n, ex, ms, er = _verify_streaming(
                    fs, ft, kiwi, mismatches, max_samples, log_file=log
                )
            total += n
            exact += ex
            mismatch += ms
            error += er
    elapsed = time.time() - t0

    _print_summary(elapsed, total, exact, mismatch, error)
    _print_mismatches(mismatches)
    log_mb = os.path.getsize(TEST_MISMATCH_LOG_PATH) / 1024 / 1024
    print(f"불일치 로그:   {log_mb:.1f}M  {TEST_MISMATCH_LOG_PATH}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5 토큰화 round-trip 검증 (Kiwi.join 으로 복원 후 원본과 비교)"
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="테스트 모드: 앞 N 샤드 vs tokens_test.txt 비교 (기본 N=1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_MISMATCH_SAMPLES,
        metavar="M",
        help=f"불일치 사례 출력 개수 (기본 {DEFAULT_MISMATCH_SAMPLES})",
    )
    args = parser.parse_args()

    if args.test is not None:
        _run_test(args.test, args.samples)
    else:
        _run_full(args.samples)


if __name__ == "__main__":
    main()
