"""
tokens_filtered_{VS}/part_*.txt → 정수 인코딩.

vocab_{VS}.txt를 읽어 {토큰→인덱스} 사전을 만들고,
tokens_filtered_{VS}/part_*.txt의 각 토큰을 정수로 변환하여
token_ids_{VS}/part_NN.npy로 저장한다.

각 .npy 파일은 1차원 uint16 배열 — 해당 샤드의 모든 토큰을
순서대로 이어붙인 것이다. <bos>/<eos>가 문단 경계를 표시한다.

dtype 선택: 세 vocab size(32768, 49152, 65536) 모두 uint16 범위(0–65535)에 맞음.

입력: vocab_{VS}.txt + tokens_filtered_{VS}/part_NN.txt
출력: token_ids_{VS}/part_NN.npy (uint16, 1-D)

사용:
  python step7_encode.py                     # VS=32768 (기본값)
  python step7_encode.py --vocab-size 49152
  python step7_encode.py --vocab-size 65536
  python step7_encode.py --test              # 앞 1개 샤드만
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_SHARDS = 32


# ──────────────────────────────────────────────
# Vocab 로딩
# ──────────────────────────────────────────────


def load_vocab(vocab_path: str) -> dict[str, int]:
    """vocab 파일 → {토큰: 인덱스} 사전."""
    token2id: dict[str, int] = {}
    with open(vocab_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.split("\t", 1)[0]
            token2id[token] = idx
    return token2id


# ──────────────────────────────────────────────
# 샤드 인코딩
# ──────────────────────────────────────────────

_TOKEN2ID: dict[str, int] | None = None


def _init_worker(token2id: dict[str, int]) -> None:
    global _TOKEN2ID
    _TOKEN2ID = token2id


def _encode_shard(args: tuple[int, str, str]) -> tuple[int, int, int]:
    """워커: 샤드 1개를 정수 배열로 변환.

    반환: (shard_idx, n_lines, n_tokens)
    """
    shard_idx, in_path, out_path = args
    token2id = _TOKEN2ID

    ids: list[int] = []
    n_lines = 0
    for line in open(in_path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        n_lines += 1
        for tok in line.split(" "):
            ids.append(token2id[tok])

    arr = np.array(ids, dtype=np.uint16)
    np.save(out_path, arr)
    return shard_idx, n_lines, len(ids)


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────


def _list_input_shards(in_dir: str) -> list[str]:
    return sorted(
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.endswith(".txt")
    )


def _run(vocab_size: int, test_shards: int | None) -> None:
    vocab_path = os.path.join(DATA_DIR, f"vocab_{vocab_size}.txt")
    in_dir = os.path.join(DATA_DIR, f"tokens_filtered_{vocab_size}")
    out_dir = os.path.join(DATA_DIR, f"token_ids_{vocab_size}")

    if not os.path.isfile(vocab_path):
        print(f"vocab 파일 없음: {vocab_path}")
        return
    if not os.path.isdir(in_dir):
        print(f"입력 디렉토리 없음: {in_dir}")
        return

    in_paths = _list_input_shards(in_dir)
    if test_shards is not None:
        in_paths = in_paths[:test_shards]

    mode = f"TEST: 앞 {len(in_paths)} 샤드" if test_shards else f"{len(in_paths)} shards"

    print("=" * 60)
    print(f"정수 인코딩 [{mode}]")
    print("=" * 60)
    print(f"vocab:  {vocab_path}")
    print(f"입력:   {in_dir}")
    print(f"출력:   {out_dir}")
    print()

    # Vocab 로딩
    print("Vocab 로딩 ...")
    t0 = time.time()
    token2id = load_vocab(vocab_path)
    print(f"  {len(token2id):,} 토큰 ({time.time() - t0:.1f}s)")

    if len(token2id) > 65536:
        print(f"오류: vocab size {len(token2id):,} > uint16 범위(65536)")
        return

    # 인코딩
    os.makedirs(out_dir, exist_ok=True)
    tasks = [
        (i, p, os.path.join(out_dir, os.path.basename(p).replace(".txt", ".npy")))
        for i, p in enumerate(in_paths)
    ]

    print(f"\n인코딩 시작 ...")
    t1 = time.time()

    if test_shards is not None:
        # 테스트 모드: 단일 프로세스
        _init_worker(token2id)
        results = [_encode_shard(t) for t in tasks]
    else:
        n_workers = min(N_SHARDS, len(tasks))
        with Pool(n_workers, initializer=_init_worker, initargs=(token2id,)) as pool:
            results = pool.map(_encode_shard, tasks)

    elapsed = time.time() - t0

    # 결과 출력
    total_lines = total_tokens = 0
    total_bytes = 0
    print()
    for shard_idx, n_lines, n_tokens in sorted(results):
        total_lines += n_lines
        total_tokens += n_tokens
        out_path = tasks[shard_idx][2]
        sz = os.path.getsize(out_path)
        total_bytes += sz
        print(
            f"  part_{shard_idx:02d}.npy: "
            f"{n_lines:>8,} lines  {n_tokens:>10,} tokens  "
            f"{sz / 1024 / 1024:>7.1f} MB"
        )

    print()
    print("=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"총 문장 수:   {total_lines:,}")
    print(f"총 토큰 수:   {total_tokens:,}")
    print(f"총 파일 크기: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"dtype:        uint16 ({total_tokens * 2 / 1024 / 1024:.1f} MB raw)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tokens_filtered → 정수 인코딩 (numpy uint16)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        metavar="VS",
        help="vocab size (기본값 32768). vocab_{VS}.txt 와 tokens_filtered_{VS}/ 를 사용",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="테스트 모드: 앞 N개 샤드만 단일 프로세스로 처리 (기본 N=1)",
    )
    args = parser.parse_args()
    _run(args.vocab_size, args.test)


if __name__ == "__main__":
    main()
