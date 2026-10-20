"""
tokens/part_*.txt → 빈도 기반 어휘 구축 + OOV 문장 필터링.

Pass 1: 전체 토큰 빈도 카운팅 (단일 프로세스)
  - SN/W_SERIAL 토큰을 개별 자릿수 + 구분자로 분리 후 카운팅
Pass 2: vocab 외 토큰이 포함된 문장 제거 + <bos>/<eos> 삽입 (멀티프로세싱)
  - 출력에도 분리된 숫자 토큰이 반영됨

어휘 구축:
  - count >= min_count 인 토큰만 포함
  - 인덱스 0–3 특수 토큰: <pad>, <unk>, <bos>, <eos>
  - 빈도 내림차순 정렬 (동률 시 사전순)

문단 경계:
  - 입력의 빈 줄(문단 경계) → <bos>/<eos> 토큰으로 대체
  - <bos>: 문단 첫 문장 앞, <eos>: 문단 마지막 문장 뒤
  - 문단 내 모든 문장이 제거되면 <bos>/<eos>도 생략

입력: tokens/part_NN.txt              (한 줄 = 토큰열, 빈 줄 = 문단 경계)
출력:
  vocab_{VS}.txt                      어휘 (토큰\\t빈도, 행 번호 = 인덱스)
  tokens_filtered_{VS}/part_NN.txt    OOV 문장 제거 + <bos>/<eos> 삽입
"""

import argparse
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(DATA_DIR, "tokens")
N_SHARDS = 32

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
BOS = "<bos>"
EOS = "<eos>"


# ──────────────────────────────────────────────
# 숫자 토큰 분리 (SN, W_SERIAL → 개별 자릿수 + 구분자)
# ──────────────────────────────────────────────
#
# 동기:
#   SN(숫자)과 W_SERIAL(일련번호) 토큰이 vocab의 약 5.7%(~86,000개)를 차지하며
#   대부분 빈도 1인 고유 숫자(예: 99994845/SN, 2010-11/W_SERIAL)다.
#   개별 자릿수(0–9) + 구분자로 분리하면 ~86,000개 → ~16개로 통합된다.
#
# 예:
#   "2010/SN"       → "2/SN 0/SN 1/SN 0/SN"
#   "2-1/W_SERIAL"  → "2/SN -/SW 1/SN"
#   "996,000/SN"    → "9/SN 9/SN 6/SN ,/SW 0/SN 0/SN 0/SN"
#
# 현대 LLM과의 관계:
#   LLaMA 계열(SentencePiece)과 GPT 계열(BPE) 모두 숫자를 1–3자리 단위로
#   분리한다. 자릿수 수준의 토큰화는 현대 LLM 토크나이저의 표준 관행이다.
#
# 트레이드오프 — 의미 분산:
#   "2010"(연도)의 의미가 4개 자릿수 토큰에 분산된다.
#   현대 Transformer 기반 LLM은 self-attention과 위치 인코딩을 통해
#   분산된 자릿수 토큰의 의미를 내부 레이어에서 합성(compose)한다.
#   다만 이 합성은 완벽하지 않다 — 특히 다자릿수 산술(arithmetic)에서
#   오류가 발생하며, 이는 LLM의 알려진 약점이다. CoT(chain-of-thought)
#   프롬프팅이 산술 성능을 개선하는 이유도 내부 합성만으로는 부족하기
#   때문이다. 그러나 의미론적 이해(예: "2010은 연도다")는 비교적 잘
#   학습된다.
#   이 프로젝트(Bengio NPLM / Word2Vec 데모)에서는 숫자의 완전한 의미
#   학습까지 목표하지 않으므로, 자릿수 분리의 vocab 축소 효과가 우선이다.
#
# 구분자 POS 태그:
#   숫자 내부 구분자(,.:-/_)는 /SW(기타 기호)로 태깅한다.
#   동일 문자가 독립 토큰으로 이미 존재(,/SP, ./SF, -/SO 등)하지만,
#   숫자 내부 맥락과 문장 부호 맥락은 다르므로 별도 토큰으로 유지한다.

_SPLIT_POS_TAGS = frozenset({"SN", "W_SERIAL"})
_FW_TO_HW = str.maketrans("０１２３４５６７８９", "0123456789")


def _split_number_token(token: str) -> list[str]:
    """SN/W_SERIAL 토큰을 개별 자릿수 + 구분자로 분리.

    해당하지 않는 토큰은 [token] 그대로 반환.
    전각 숫자(０–９)는 반각(0–9)으로 정규화.
    """
    slash_idx = token.rfind("/")
    if slash_idx <= 0:
        return [token]
    pos = token[slash_idx + 1:]
    if pos not in _SPLIT_POS_TAGS:
        return [token]
    surface = token[:slash_idx].translate(_FW_TO_HW)
    if len(surface) == 1:
        return [f"{surface}/SN" if surface.isdigit() else f"{surface}/SW"]
    return [
        f"{ch}/SN" if ch.isdigit() else f"{ch}/SW"
        for ch in surface
    ]


def _expand_line(tokens: list[str]) -> list[str]:
    """한 줄의 토큰 리스트에서 SN/W_SERIAL 토큰을 자릿수 단위로 확장."""
    return [sub for tok in tokens for sub in _split_number_token(tok)]


def _list_input_shards() -> list[str]:
    return sorted(
        os.path.join(IN_DIR, f)
        for f in os.listdir(IN_DIR)
        if f.endswith(".txt")
    )


# ──────────────────────────────────────────────
# Pass 1: 토큰 빈도 카운팅
# ──────────────────────────────────────────────


def count_tokens(paths: list[str]) -> Counter:
    """전체 샤드의 토큰 빈도 카운팅. 빈 줄 건너뜀."""
    counter: Counter = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                counter.update(_expand_line(line.split(" ")))
    return counter


# ──────────────────────────────────────────────
# Vocab 구축
# ──────────────────────────────────────────────


def build_vocab(
    counter: Counter,
    min_count: int = 0,
    top_k: int | None = None,
) -> tuple[list[tuple[str, int]], set[str]]:
    """빈도 >= min_count 인 토큰으로 vocab 구축. top_k 지정 시 상위 K개로 제한.

    반환: (vocab_list, vocab_set)
      vocab_list: [(token, count), ...] 특수 토큰 포함, 인덱스 순
      vocab_set:  set of token strings
    """
    filtered = [
        (tok, cnt) for tok, cnt in counter.items() if cnt >= min_count
    ]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    if top_k is not None:
        filtered = filtered[: top_k - len(SPECIAL_TOKENS)]

    vocab_list = [(tok, 0) for tok in SPECIAL_TOKENS] + filtered
    vocab_set = {tok for tok, _ in vocab_list}
    return vocab_list, vocab_set


def write_vocab(vocab_list: list[tuple[str, int]], path: str) -> None:
    """vocab 파일 작성. 형식: 토큰\\t빈도 (행 번호 = 인덱스)."""
    with open(path, "w", encoding="utf-8") as f:
        for tok, cnt in vocab_list:
            f.write(f"{tok}\t{cnt}\n")


# ──────────────────────────────────────────────
# Pass 2: 필터링 + <bos>/<eos> 삽입
# ──────────────────────────────────────────────

_VOCAB_SET: set[str] | None = None


def _init_worker(vocab_set: set[str]) -> None:
    global _VOCAB_SET
    _VOCAB_SET = vocab_set


def _write_para(fout, para_kept: list[str]) -> None:
    """문단의 kept 문장들에 <bos>/<eos> 붙여서 출력.

    문장 간 공백(<sp>/WS)은 step5a에서 이미 삽입됨.
    이 함수에서 <sp>/WS를 추가하면 중복이 되므로 하지 않는다.
    """
    para_kept[0] = BOS + " " + para_kept[0]
    para_kept[-1] = para_kept[-1] + " " + EOS
    for s in para_kept:
        fout.write(s + "\n")


def _filter_shard(args) -> tuple[int, int, int, int, int]:
    """워커: 샤드 1개 필터링.

    반환: (shard_idx, n_sent_in, n_sent_out, n_para_in, n_para_out)
    """
    shard_idx, in_path, out_path = args
    vocab_set = _VOCAB_SET
    n_sent_in = n_sent_out = n_para_in = n_para_out = 0
    para_kept: list[str] = []
    para_has_content = False

    with (
        open(in_path, encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                if para_has_content:
                    n_para_in += 1
                    if para_kept:
                        n_para_out += 1
                        _write_para(fout, para_kept)
                    para_kept = []
                    para_has_content = False
                continue

            para_has_content = True
            n_sent_in += 1
            tokens = _expand_line(line.split(" "))
            if all(t in vocab_set for t in tokens):
                para_kept.append(" ".join(tokens))
                n_sent_out += 1

        # EOF — 마지막 문단 flush
        if para_has_content:
            n_para_in += 1
            if para_kept:
                n_para_out += 1
                _write_para(fout, para_kept)

    return shard_idx, n_sent_in, n_sent_out, n_para_in, n_para_out


# ──────────────────────────────────────────────
# 통계 출력
# ──────────────────────────────────────────────


def _print_freq_distribution(counter: Counter, min_count: int) -> None:
    """빈도 분포 + min_count별 vocab size 표시."""
    total_unique = len(counter)

    freq_dist: Counter = Counter()
    for cnt in counter.values():
        freq_dist[cnt] += 1

    print(f"\n전체 유니크 토큰:  {total_unique:,}")
    print()
    print("빈도 분포:")
    for c in range(1, 11):
        n = freq_dist.get(c, 0)
        pct = n / max(total_unique, 1) * 100
        print(f"  count={c:<5d} {n:>10,} tokens  ({pct:>5.1f}%)")
    n_above = sum(n for c, n in freq_dist.items() if c > 10)
    pct_above = n_above / max(total_unique, 1) * 100
    print(f"  count>10    {n_above:>10,} tokens  ({pct_above:>5.1f}%)")

    print()
    print("min_count별 vocab size (특수 토큰 4개 포함):")
    for mc in [1, 2, 3, 5, 10, 20, 50, 100]:
        n = sum(1 for cnt in counter.values() if cnt >= mc) + len(SPECIAL_TOKENS)
        marker = "  ◀" if mc == min_count else ""
        print(f"  min_count={mc:<5d} → {n:>10,}{marker}")

    vocab_size = (
        sum(1 for cnt in counter.values() if cnt >= min_count)
        + len(SPECIAL_TOKENS)
    )
    print()
    print(f"선택: min_count={min_count}  →  vocab size = {vocab_size:,}")


def _print_summary(
    elapsed: float,
    n_sent_in: int,
    n_sent_out: int,
    n_para_in: int,
    n_para_out: int,
    in_size: int,
    out_size: int,
    vocab_size: int,
) -> None:
    sent_drop = (1 - n_sent_out / max(n_sent_in, 1)) * 100
    para_drop = (1 - n_para_out / max(n_para_in, 1)) * 100
    in_mb = in_size / 1024 / 1024
    out_mb = out_size / 1024 / 1024

    print()
    print("=" * 60)
    print(f"소요 시간:    {elapsed:.1f}s")
    print(f"vocab size:   {vocab_size:,}")
    print()
    print(f"{'':12s}{'in':>14s}  {'out':>14s}  {'제거':>10s}")
    print(f"{'문장 수':12s}{n_sent_in:>14,}  {n_sent_out:>14,}  {sent_drop:>9.1f}%")
    print(f"{'문단 수':12s}{n_para_in:>14,}  {n_para_out:>14,}  {para_drop:>9.1f}%")
    print(f"{'파일 크기':12s}{in_mb:>13.1f}M  {out_mb:>13.1f}M")
    print("=" * 60)


# ──────────────────────────────────────────────
# Full / Test
# ──────────────────────────────────────────────


def _run_full(min_count: int, top_k: int | None = None) -> None:
    in_paths = _list_input_shards()
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    # Pass 1
    print("=" * 60)
    print(f"어휘 구축 + OOV 문장 필터링 ({len(in_paths)} shards)")
    print("=" * 60)
    print(f"min_count:   {min_count}")
    if top_k is not None:
        print(f"top_k:       {top_k:,}")
    print()

    print("Pass 1: 토큰 빈도 카운팅 ...")
    t0 = time.time()
    counter = count_tokens(in_paths)
    t1 = time.time()
    print(f"  완료 ({t1 - t0:.1f}s)")

    _print_freq_distribution(counter, min_count)

    # Vocab (경로 결정을 위해 보고서보다 먼저 구축)
    vocab_list, vocab_set = build_vocab(counter, min_count, top_k)
    vs = len(vocab_list)
    tag = f"_{vs}"
    out_dir = os.path.join(DATA_DIR, f"tokens_filtered{tag}")
    vocab_path = os.path.join(DATA_DIR, f"vocab{tag}.txt")

    write_vocab(vocab_list, vocab_path)
    vocab_mb = os.path.getsize(vocab_path) / 1024 / 1024
    print(f"\nvocab 저장:  {vocab_mb:.1f}M  {vocab_path}")

    # Pass 2
    os.makedirs(out_dir, exist_ok=True)
    tasks = [
        (i, in_path, os.path.join(out_dir, os.path.basename(in_path)))
        for i, in_path in enumerate(in_paths)
    ]
    n_workers = min(N_SHARDS, len(tasks))

    print(f"\nPass 2: 문장 필터링 ({n_workers} workers) ...")
    print(f"  출력: {out_dir}")
    t2 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(vocab_set,)) as pool:
        results = pool.map(_filter_shard, tasks)
    elapsed = time.time() - t0

    total_si = total_so = total_pi = total_po = 0
    print()
    print("샤드별 (문장 in → out):")
    for shard_idx, si, so, pi, po in sorted(results):
        total_si += si
        total_so += so
        total_pi += pi
        total_po += po
        pct = (1 - so / max(si, 1)) * 100
        out_path = os.path.join(out_dir, f"part_{shard_idx:02d}.txt")
        sz_mb = os.path.getsize(out_path) / 1024 / 1024
        print(
            f"  part_{shard_idx:02d}.txt: {si:>8,} → {so:>8,} "
            f"({pct:>5.1f}% 제거)  {sz_mb:>7.1f} MB"
        )

    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = sum(
        os.path.getsize(os.path.join(out_dir, os.path.basename(p)))
        for p in in_paths
    )
    _print_summary(
        elapsed, total_si, total_so, total_pi, total_po,
        in_size, out_size, len(vocab_list),
    )


def _run_test(n_shards: int, min_count: int, top_k: int | None = None) -> None:
    """앞 N개 샤드만 단일 프로세스로 처리."""
    in_paths = _list_input_shards()[:n_shards]
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    # Pass 1
    print("=" * 60)
    print(f"어휘 구축 + OOV 문장 필터링 [TEST: 앞 {len(in_paths)} 샤드]")
    print("=" * 60)
    print(f"min_count:   {min_count}")
    if top_k is not None:
        print(f"top_k:       {top_k:,}")
    print()

    print("Pass 1: 토큰 빈도 카운팅 ...")
    t0 = time.time()
    counter = count_tokens(in_paths)
    t1 = time.time()
    print(f"  완료 ({t1 - t0:.1f}s)")

    _print_freq_distribution(counter, min_count)

    # Vocab
    vocab_list, vocab_set = build_vocab(counter, min_count, top_k)
    vs = len(vocab_list)
    tag = f"_{vs}_test"
    vocab_path = os.path.join(DATA_DIR, f"vocab{tag}.txt")
    out_path = os.path.join(DATA_DIR, f"tokens_filtered{tag}.txt")

    write_vocab(vocab_list, vocab_path)
    vocab_mb = os.path.getsize(vocab_path) / 1024 / 1024
    print(f"\nvocab 저장:  {vocab_mb:.1f}M  {vocab_path}")

    # Pass 2 (single process)
    print("\nPass 2: 문장 필터링 ...")
    n_si = n_so = n_pi = n_po = 0
    para_kept: list[str] = []
    para_has_content = False

    with open(out_path, "w", encoding="utf-8") as fout:
        for in_path in in_paths:
            print(f"  filtering {os.path.basename(in_path)} ...")
            with open(in_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        if para_has_content:
                            n_pi += 1
                            if para_kept:
                                n_po += 1
                                _write_para(fout, para_kept)
                            para_kept = []
                            para_has_content = False
                        continue

                    para_has_content = True
                    n_si += 1
                    tokens = _expand_line(line.split(" "))
                    if all(t in vocab_set for t in tokens):
                        para_kept.append(" ".join(tokens))
                        n_so += 1

            # 샤드 경계 = 문단 경계
            if para_has_content:
                n_pi += 1
                if para_kept:
                    n_po += 1
                    _write_para(fout, para_kept)
                para_kept = []
                para_has_content = False

    elapsed = time.time() - t0
    in_size = sum(os.path.getsize(p) for p in in_paths)
    out_size = os.path.getsize(out_path)
    _print_summary(
        elapsed, n_si, n_so, n_pi, n_po,
        in_size, out_size, len(vocab_list),
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tokens/part_*.txt → 빈도 기반 어휘 구축 + OOV 문장 필터링"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        metavar="N",
        help="최소 빈도 (기본값 0). count < N 인 토큰은 어휘에서 제외",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="vocab size 상한 (특수 토큰 포함). 빈도순 상위 K개만 포함",
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

    if args.test is not None:
        _run_test(args.test, args.min_count, args.top_k)
    else:
        _run_full(args.min_count, args.top_k)


if __name__ == "__main__":
    main()
