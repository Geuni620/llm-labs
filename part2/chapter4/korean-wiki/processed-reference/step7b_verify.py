"""
step7 정수 인코딩 round-trip 검증.

token_ids_{VS}/part_NN.npy를 역변환하여 tokens_filtered_{VS}/part_NN.txt의
토큰 시퀀스와 정확히 일치하는지 확인한다.

검증 방식:
  1. .npy → id2token 으로 토큰 시퀀스 복원
  2. .txt → 빈 줄 제외, 모든 토큰을 순서대로 이어붙인 시퀀스
  3. 두 시퀀스를 비교 → 하나라도 다르면 불일치 위치 보고

참고:
  .npy는 flat 1D 배열이므로 줄바꿈 정보는 없다.
  토큰 시퀀스 자체가 동일한지만 검증한다.

사용:
  python step7b_verify.py                     # VS=32768 (기본값)
  python step7b_verify.py --vocab-size 49152
  python step7b_verify.py --test              # 앞 1개 샤드만
"""

import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_id2token(vocab_path: str) -> list[str]:
    """vocab 파일 → [인덱스→토큰] 리스트."""
    id2token: list[str] = []
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            id2token.append(line.split("\t", 1)[0])
    return id2token


def tokens_from_txt(path: str) -> list[str]:
    """tokens_filtered .txt 에서 모든 토큰을 순서대로 추출."""
    tokens: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tokens.extend(line.split(" "))
    return tokens


def verify_shard(
    shard_idx: int,
    npy_path: str,
    txt_path: str,
    id2token: list[str],
) -> tuple[int, bool, int, str]:
    """샤드 1개 검증.

    반환: (shard_idx, passed, n_tokens, detail)
    """
    ids = np.load(npy_path)
    decoded = [id2token[i] for i in ids]
    expected = tokens_from_txt(txt_path)

    if decoded == expected:
        return shard_idx, True, len(decoded), "OK"

    # 불일치 — 첫 번째 위치 찾기
    min_len = min(len(decoded), len(expected))
    first_diff = min_len  # 길이가 다르고 공통 부분은 같은 경우
    for i in range(min_len):
        if decoded[i] != expected[i]:
            first_diff = i
            break

    detail = (
        f"MISMATCH at token {first_diff}: "
        f"npy={decoded[first_diff] if first_diff < len(decoded) else 'EOF'} "
        f"txt={expected[first_diff] if first_diff < len(expected) else 'EOF'} "
        f"(npy={len(decoded):,} vs txt={len(expected):,} tokens)"
    )
    return shard_idx, False, len(decoded), detail


def _run(vocab_size: int, test_shards: int | None) -> None:
    vocab_path = os.path.join(DATA_DIR, f"vocab_{vocab_size}.txt")
    npy_dir = os.path.join(DATA_DIR, f"token_ids_{vocab_size}")
    txt_dir = os.path.join(DATA_DIR, f"tokens_filtered_{vocab_size}")

    for path, label in [(vocab_path, "vocab"), (npy_dir, "npy dir"), (txt_dir, "txt dir")]:
        if not os.path.exists(path):
            print(f"{label} 없음: {path}")
            return

    npy_files = sorted(f for f in os.listdir(npy_dir) if f.endswith(".npy"))
    if test_shards is not None:
        npy_files = npy_files[:test_shards]

    mode = f"TEST: 앞 {len(npy_files)} 샤드" if test_shards else f"{len(npy_files)} shards"

    print("=" * 60)
    print(f"step7 인코딩 검증 [{mode}]")
    print("=" * 60)
    print(f"vocab: {vocab_path}")
    print(f"npy:   {npy_dir}")
    print(f"txt:   {txt_dir}")
    print()

    id2token = load_id2token(vocab_path)
    print(f"Vocab: {len(id2token):,} 토큰")
    print()

    t0 = time.time()
    n_pass = n_fail = total_tokens = 0

    for npy_file in npy_files:
        shard_idx = int(npy_file.replace("part_", "").replace(".npy", ""))
        npy_path = os.path.join(npy_dir, npy_file)
        txt_file = npy_file.replace(".npy", ".txt")
        txt_path = os.path.join(txt_dir, txt_file)

        if not os.path.isfile(txt_path):
            print(f"  part_{shard_idx:02d}: txt 없음 — SKIP")
            continue

        idx, passed, n_tokens, detail = verify_shard(
            shard_idx, npy_path, txt_path, id2token
        )
        total_tokens += n_tokens

        status = "PASS" if passed else "FAIL"
        print(f"  part_{idx:02d}: {n_tokens:>10,} tokens  {status}  {detail}")

        if passed:
            n_pass += 1
        else:
            n_fail += 1

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"소요 시간:  {elapsed:.1f}s")
    print(f"총 토큰:    {total_tokens:,}")
    print(f"PASS:       {n_pass}  /  FAIL: {n_fail}")
    if n_fail == 0:
        print("결과: 모든 샤드 round-trip 검증 통과")
    else:
        print("결과: 불일치 발견!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="step7 정수 인코딩 round-trip 검증"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        metavar="VS",
        help="vocab size (기본값 32768)",
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="테스트 모드: 앞 N개 샤드만 (기본 N=1)",
    )
    args = parser.parse_args()
    _run(args.vocab_size, args.test)


if __name__ == "__main__":
    main()
