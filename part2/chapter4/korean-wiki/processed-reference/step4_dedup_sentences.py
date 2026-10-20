"""
sentences/part_*.txt → 문장 단위 정규화 + Exact 중복 제거.

정규화: NFKC + 공백 정리 (영문 case 보존)
해싱:   blake2b 16 byte digest (충돌 확률 ~1e-19)
정규화는 비교 전용, 출력은 원문 그대로.

전역 dedup: 32 샤드를 같은 seen dict 로 처리 → 샤드 간 중복도 잡힘.
문단 경계(빈 줄) 보존, 문단 전체가 dup이면 빈 줄도 함께 제거.

제거 로그: '[Nx] 첫 등장 원문' 형식, 빈도 desc 정렬.

입력: sentences/part_NN.txt   (한 줄 = 한 문장, 빈 줄 = 문단 경계)
출력:
  sentences_dedup/part_NN.txt    중복 제거 후 본문
  sentences_dedup_removed.txt    제거된 문장 빈도표 (검증용)
"""

import argparse
import hashlib
import os
import re
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(DATA_DIR, "sentences")
OUT_DIR = os.path.join(DATA_DIR, "sentences_dedup")
REMOVED_PATH = os.path.join(DATA_DIR, "sentences_dedup_removed.txt")
TEST_OUT_PATH = os.path.join(DATA_DIR, "sentences_dedup_test.txt")
TEST_REMOVED_PATH = os.path.join(DATA_DIR, "sentences_dedup_test_removed.txt")
TOP_N = 30

_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """NFKC + 공백 정리. dedup 비교 키 전용 (출력엔 안 씀).

    영문 case 는 보존 — "Apple" 과 "apple" 은 다른 문장으로 취급.
    """
    s = unicodedata.normalize("NFKC", s)
    return _WS_RE.sub(" ", s).strip()


def hash_key(s: str) -> bytes:
    """blake2b 16 byte digest. 충돌 무시 가능 (1e9 문장에서 ~1e-19)."""
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()


def _dedup_one_file(
    in_path: str,
    out_path: str,
    seen: dict[bytes, str],
    dup_count: dict[bytes, int],
) -> tuple[int, int]:
    """샤드 1개 dedup. seen / dup_count 는 호출자 소유 전역 dict 이며 in-place 갱신.

    문단 경계 보존:
      - 문장 1개라도 살아남은 문단 끝에서만 빈 줄 출력
      - 문단 전체가 dup → 빈 줄도 안 씀 → 연속 빈 줄 발생 X
    반환: (입력 문장 수, 출력 문장 수)
    """
    in_sents = out_sents = 0
    para_has_kept = False
    with (
        open(in_path, encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                if para_has_kept:
                    fout.write("\n")
                para_has_kept = False
                continue
            in_sents += 1
            key = hash_key(normalize(line))
            if key in seen:
                dup_count[key] = dup_count.get(key, 0) + 1
                continue
            seen[key] = line
            fout.write(line)
            fout.write("\n")
            out_sents += 1
            para_has_kept = True
        # 파일 끝에 trailing 빈 줄이 없는 경우 대비
        if para_has_kept:
            fout.write("\n")
    return in_sents, out_sents


def _list_input_shards() -> list[str]:
    return sorted(
        os.path.join(IN_DIR, f)
        for f in os.listdir(IN_DIR)
        if f.endswith(".txt")
    )


def _write_removed_log(
    removed_path: str,
    seen: dict[bytes, str],
    dup_count: dict[bytes, int],
    n_shards: int,
    total_in: int,
    total_out: int,
) -> None:
    """제거된 문장 빈도표 작성. 각 줄: '[Nx] 첫 등장 원문' (빈도 desc).

    N = 코퍼스 전체 등장 횟수 (= dup_count[key] + 1, 원본 1회 포함).
    """
    n_unique_dups = len(dup_count)
    n_removed = sum(dup_count.values())
    sorted_items = sorted(dup_count.items(), key=lambda kv: kv[1], reverse=True)

    with open(removed_path, "w", encoding="utf-8") as fout:
        fout.write("# 문장 단위 정규화 + Exact dedup 결과\n")
        fout.write(f"# 입력 샤드 수:        {n_shards}\n")
        fout.write(f"# 입력 문장 수:        {total_in:,}\n")
        fout.write(f"# 출력 문장 수:        {total_out:,}\n")
        fout.write(f"# 제거된 문장 수:      {n_removed:,}")
        if total_in:
            fout.write(f"  ({n_removed / total_in * 100:.2f}%)")
        fout.write("\n")
        fout.write(f"# 유니크 중복 종류:    {n_unique_dups:,}\n")
        fout.write("#\n")
        fout.write("# 형식: [Nx] <첫 등장 원문>\n")
        fout.write("# N = 코퍼스 전체에서 그 문장이 등장한 총 횟수 (원본 1 + 중복 N-1)\n")
        fout.write("# 정렬: N desc (가장 자주 등장한 boilerplate가 맨 위)\n")
        fout.write("# 전역 dedup이므로 샤드 경계와 무관\n")
        fout.write("\n")
        for key, dup_n in sorted_items:
            total_n = dup_n + 1  # 원본 자신도 카운트
            fout.write(f"[{total_n}x] {seen[key]}\n")


def _print_summary(
    elapsed: float,
    n_unique: int,
    total_in: int,
    total_out: int,
    in_size: int,
    out_size: int,
) -> None:
    drop = (1 - total_out / max(total_in, 1)) * 100
    in_mb = in_size / 1024 / 1024
    out_mb = out_size / 1024 / 1024
    drop_size = (1 - out_mb / max(in_mb, 1e-9)) * 100

    print()
    print("=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"unique key:   {n_unique:,}")
    print()
    print(f"{'':12s}{'in':>14s}  {'out':>14s}  {'Δ':>10s}")
    print(f"{'문장 수':12s}{total_in:>14,}  {total_out:>14,}  {-drop:>9.1f}%")
    print(
        f"{'파일 크기':12s}{in_mb:>13.1f}M  {out_mb:>13.1f}M  {-drop_size:>9.1f}%"
    )
    print("=" * 60)


def _print_top_dups(
    seen: dict[bytes, str],
    dup_count: dict[bytes, int],
    n: int = TOP_N,
) -> None:
    """가장 자주 잡힌 중복 문장 TOP N 을 콘솔에 출력."""
    if not dup_count:
        print("\n(중복 없음)")
        return
    print(f"\nTOP {n} duplicates (총 등장 횟수 desc):")
    sorted_dups = sorted(dup_count.items(), key=lambda kv: kv[1], reverse=True)[:n]
    for key, cnt in sorted_dups:
        original = seen[key]
        if len(original) > 80:
            original = original[:77] + "..."
        print(f"  [{cnt + 1:>5}x] {original}")  # +1: 원본도 카운트


def _run_full() -> None:
    in_paths = _list_input_shards()
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"문장 정규화 + Exact dedup ({len(in_paths)} shards)")
    print("=" * 60)
    print(f"입력:        {IN_DIR}")
    print(f"출력:        {OUT_DIR}")
    print(f"제거 로그:    sentences_dedup_removed.txt")
    print()

    seen: dict[bytes, str] = {}
    dup_count: dict[bytes, int] = {}
    shard_stats: list[tuple[str, int, int]] = []
    total_in = total_out = 0

    t0 = time.time()
    for in_path in in_paths:
        name = os.path.basename(in_path)
        out_path = os.path.join(OUT_DIR, name)
        in_sents, out_sents = _dedup_one_file(in_path, out_path, seen, dup_count)
        shard_stats.append((name, in_sents, out_sents))
        total_in += in_sents
        total_out += out_sents
        removed_pct = (1 - out_sents / max(in_sents, 1)) * 100
        print(
            f"  {name}: {in_sents:>8,} → {out_sents:>8,} "
            f"({removed_pct:>5.1f}% removed)"
        )
    elapsed = time.time() - t0

    _write_removed_log(
        REMOVED_PATH, seen, dup_count, len(in_paths), total_in, total_out
    )

    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = sum(
        os.path.getsize(os.path.join(OUT_DIR, name))
        for name, _, _ in shard_stats
    )
    removed_size = os.path.getsize(REMOVED_PATH)
    _print_summary(elapsed, len(seen), total_in, total_out, in_size, out_size)
    print(f"제거 로그:    {removed_size / 1024 / 1024:>6.1f}M  {REMOVED_PATH}")
    _print_top_dups(seen, dup_count)


def _run_test(n_shards: int) -> None:
    """앞 N개 입력 샤드만 dedup. 출력은 단일 파일.

    출력:
      sentences_dedup_test.txt          중복 제거 본문
      sentences_dedup_test_removed.txt  제거된 문장 빈도표
    """
    in_paths = _list_input_shards()[:n_shards]
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    print("=" * 60)
    print(f"문장 dedup [TEST MODE: 앞 {len(in_paths)} 샤드]")
    print("=" * 60)
    print(f"입력:        {IN_DIR}")
    print(f"출력:        sentences_dedup_test.txt")
    print(f"제거 로그:    sentences_dedup_test_removed.txt")
    print()

    seen: dict[bytes, str] = {}
    dup_count: dict[bytes, int] = {}
    in_sents = out_sents = 0
    para_has_kept = False

    t0 = time.time()
    with open(TEST_OUT_PATH, "w", encoding="utf-8") as fout:
        for in_path in in_paths:
            print(f"  reading {os.path.basename(in_path)} ...")
            with open(in_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        if para_has_kept:
                            fout.write("\n")
                        para_has_kept = False
                        continue
                    in_sents += 1
                    key = hash_key(normalize(line))
                    if key in seen:
                        dup_count[key] = dup_count.get(key, 0) + 1
                        continue
                    seen[key] = line
                    fout.write(line)
                    fout.write("\n")
                    out_sents += 1
                    para_has_kept = True
            # 샤드 경계도 문단 경계로 처리
            if para_has_kept:
                fout.write("\n")
            para_has_kept = False
    elapsed = time.time() - t0

    _write_removed_log(
        TEST_REMOVED_PATH, seen, dup_count, len(in_paths), in_sents, out_sents
    )

    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = os.path.getsize(TEST_OUT_PATH)
    removed_size = os.path.getsize(TEST_REMOVED_PATH)
    _print_summary(elapsed, len(seen), in_sents, out_sents, in_size, out_size)
    print(f"제거 로그:    {removed_size / 1024 / 1024:>6.1f}M  {TEST_REMOVED_PATH}")
    _print_top_dups(seen, dup_count)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="sentences/part_*.txt → 정규화 + exact dedup"
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="테스트 모드: 앞 N개 샤드만 dedup → sentences_dedup_test.txt (기본 N=1)",
    )
    args = parser.parse_args()

    if args.test is not None:
        _run_test(args.test)
    else:
        _run_full()


if __name__ == "__main__":
    main()
