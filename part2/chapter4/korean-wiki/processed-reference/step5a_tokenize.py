"""
sentences_dedup/part_*.txt → Kiwi 형태소/품사 토큰화.

토큰 형식:
  - 일반:   형태소/품사    (예: 국밥/NNG, 먹/VV, 었/EP)
  - 숫자:   형태소/SN      (예: 1987/SN — 원본 보존, 숫자 분리는 이후 단계)
  - 공백:   <sp>/WS        (Kiwi SP(쉼표 등)/SF(종결부호)와 충돌 방지용 커스텀 태그)

출력: 한 줄 = 한 문장의 토큰열, 빈 줄 = 문단 경계, 샤드 구조 유지.
특수 토큰 (<bos>, <eos> 등) 미삽입 — 이후 단계에서 모델별로 추가.

Kiwi 특성:
  - 원형 복원용 토큰 추가 가능 (예: "도로다" → 도로/NNG + 이/VCP + 다/EF)
    → "도로이다"와 동일 시퀀스가 되며, 이 표면 변이 통일은 의도적 손실.
  - 동형이의어가 품사 태그로 분리됨 (예: 차/NNG 자동차·tea  vs  차/VV 차다)
  - 자모 단독 토큰 (ᆫ/ETM, ᆯ/ETM 등) 그대로 보존 → 표면 변이 통일
  - 받침 합성 시 start 위치 겹침 → next_tok.start >= cur_end 조건으로 공백 검출
  - multi-word 고유명사 → form 안 공백을 '_' 로 escape (토큰 구분자 충돌 방지)

입력: sentences_dedup/part_NN.txt   (한 줄 = 한 문장, 빈 줄 = 문단 경계)
출력: tokens/part_NN.txt             (한 줄 = 한 문장의 토큰열, 빈 줄 = 문단 경계)
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

from kiwipiepy import Kiwi

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(DATA_DIR, "sentences_dedup")
OUT_DIR = os.path.join(DATA_DIR, "tokens")
TEST_OUT_PATH = os.path.join(DATA_DIR, "tokens_test.txt")
N_SHARDS = 32
KIWI_MODEL = "cong-global"

SPACE_TOKEN = "<sp>/WS"


def tokenize_sentence(kiwi: Kiwi, sentence: str) -> str:
    """문장 1개를 형태소/품사 토큰 시퀀스로 변환.

    규칙:
      - form 안의 공백은 '_' 로 escape (multi-word NNP 대비)
      - 형태소 사이 원문 공백 → <sp>/WS 삽입
      - 받침 합성으로 두 토큰의 원문 위치가 겹치면 공백 없음
      - 토큰은 단일 스페이스로 join
    """
    tokens = kiwi.tokenize(sentence)
    out: list[str] = []
    for i, tok in enumerate(tokens):
        form = tok.form
        if " " in form:
            form = form.replace(" ", "_")
        out.append(f"{form}/{tok.tag}")
        # 다음 토큰과의 공백 검출
        if i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            cur_end = tok.start + tok.len
            if next_tok.start >= cur_end:
                gap = sentence[cur_end : next_tok.start]
                if any(c.isspace() for c in gap):
                    out.append(SPACE_TOKEN)
            # else: 두 토큰이 원문에서 겹침 (받침 합성) → 공백 없음
    # 마지막 토큰 뒤 trailing space 검출.
    # _flush_para()가 비마지막 문장에 trailing space를 추가하므로,
    # 이 코드가 없으면 해당 공백이 <sp>/WS로 변환되지 않는다.
    if tokens:
        last = tokens[-1]
        last_end = last.start + last.len
        if last_end < len(sentence) and any(
            c.isspace() for c in sentence[last_end:]
        ):
            out.append(SPACE_TOKEN)
    return " ".join(out)


def _flush_para(
    kiwi: Kiwi, para: list[str], fout
) -> tuple[int, int]:
    """버퍼링된 문단의 문장들을 토큰화하여 출력.

    비마지막 문장 끝에 공백을 추가해서 토크나이저에 전달한다.
    step3에서 문장 분리 시 소멸된 문장 간 공백을 이 시점에 복원.
    반환: (문장 수, 토큰 수)
    """
    n_sents = 0
    n_tokens = 0
    for i, sent in enumerate(para):
        if i < len(para) - 1:
            sent = sent + " "  # 비마지막 문장: 공백 복원
        tokens_str = tokenize_sentence(kiwi, sent)
        fout.write(tokens_str)
        fout.write("\n")
        n_sents += 1
        n_tokens += tokens_str.count(" ") + 1
    fout.write("\n")  # 문단 경계
    return n_sents, n_tokens


def _process_one_file(
    kiwi: Kiwi, in_path: str, out_path: str
) -> tuple[int, int]:
    """샤드 1개 토큰화. 문단 단위로 버퍼링하여 비마지막 문장에 공백 추가.

    반환: (문장 수, 토큰 수)
    """
    n_sents = 0
    n_tokens = 0
    para: list[str] = []
    with (
        open(in_path, encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                if para:
                    s, t = _flush_para(kiwi, para, fout)
                    n_sents += s
                    n_tokens += t
                    para = []
                continue
            para.append(line)
        if para:
            s, t = _flush_para(kiwi, para, fout)
            n_sents += s
            n_tokens += t
    return n_sents, n_tokens


def _process_shard(args):
    """Worker: 샤드 1개를 단일 Kiwi 인스턴스로 토큰화."""
    shard_idx, in_path, out_path = args
    kiwi = Kiwi(num_workers=1, model_type=KIWI_MODEL)
    n_sents, n_tokens = _process_one_file(kiwi, in_path, out_path)
    return shard_idx, n_sents, n_tokens


def _list_input_shards() -> list[str]:
    return sorted(
        os.path.join(IN_DIR, f)
        for f in os.listdir(IN_DIR)
        if f.endswith(".txt")
    )


def _print_summary(
    elapsed: float,
    total_sents: int,
    total_tokens: int,
    in_size: int,
    out_size: int,
) -> None:
    in_mb = in_size / 1024 / 1024
    out_mb = out_size / 1024 / 1024
    avg_tokens = total_tokens / max(total_sents, 1)

    print()
    print("=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"문장 수:      {total_sents:,}")
    print(f"토큰 수:      {total_tokens:,}  ({avg_tokens:.1f} 토큰/문장)")
    print()
    print(f"{'':12s}{'in':>14s}  {'out':>14s}")
    print(f"{'파일 크기':12s}{in_mb:>13.1f}M  {out_mb:>13.1f}M")
    print("=" * 60)


def _run_full() -> None:
    in_paths = _list_input_shards()
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Kiwi 형태소/품사 토큰화 ({len(in_paths)} shards, multiprocessing)")
    print("=" * 60)
    print(f"입력:        {IN_DIR}")
    print(f"출력:        {OUT_DIR}")
    print()

    tasks = [
        (i, in_path, os.path.join(OUT_DIR, os.path.basename(in_path)))
        for i, in_path in enumerate(in_paths)
    ]
    n_workers = min(N_SHARDS, len(tasks))

    print(f"Launching {n_workers} workers ...")
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_process_shard, tasks)
    elapsed = time.time() - t0

    total_sents = sum(r[1] for r in results)
    total_tokens = sum(r[2] for r in results)

    print()
    print("샤드별 (문장 수 / 토큰 수, 출력 파일 크기):")
    for shard_idx, n_sents, n_tokens in sorted(results):
        out_path = os.path.join(OUT_DIR, f"part_{shard_idx:02d}.txt")
        sz_mb = os.path.getsize(out_path) / 1024 / 1024
        print(
            f"  part_{shard_idx:02d}.txt: {n_sents:>8,} 문장 / "
            f"{n_tokens:>11,} 토큰  {sz_mb:>7.1f} MB"
        )

    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = sum(
        os.path.getsize(os.path.join(OUT_DIR, os.path.basename(p)))
        for p in in_paths
    )
    _print_summary(elapsed, total_sents, total_tokens, in_size, out_size)


def _run_test(n_shards: int) -> None:
    """앞 N개 입력 샤드만 단일 프로세스로 토큰화. 출력은 단일 파일."""
    in_paths = _list_input_shards()[:n_shards]
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    print("=" * 60)
    print(f"Kiwi 형태소/품사 토큰화 [TEST MODE: 앞 {len(in_paths)} 샤드]")
    print("=" * 60)
    print(f"입력:        {IN_DIR}")
    print(f"출력:        tokens_test.txt")
    print()

    kiwi = Kiwi(num_workers=1, model_type=KIWI_MODEL)

    total_sents = 0
    total_tokens = 0
    t0 = time.time()

    with open(TEST_OUT_PATH, "w", encoding="utf-8") as fout:
        for in_path in in_paths:
            print(f"  tokenizing {os.path.basename(in_path)} ...")
            para: list[str] = []
            with open(in_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        if para:
                            s, t = _flush_para(kiwi, para, fout)
                            total_sents += s
                            total_tokens += t
                            para = []
                        continue
                    para.append(line)
                if para:
                    s, t = _flush_para(kiwi, para, fout)
                    total_sents += s
                    total_tokens += t

    elapsed = time.time() - t0
    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = os.path.getsize(TEST_OUT_PATH)
    _print_summary(elapsed, total_sents, total_tokens, in_size, out_size)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sentences_dedup/part_*.txt → Kiwi 형태소/품사 토큰화"
    )
    parser.add_argument(
        "--test",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="테스트 모드: 앞 N개 샤드만 단일 프로세스로 → tokens_test.txt (기본 N=1)",
    )
    args = parser.parse_args()

    if args.test is not None:
        _run_test(args.test)
    else:
        _run_full()


if __name__ == "__main__":
    main()
