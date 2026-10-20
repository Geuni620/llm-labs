"""
vocab_*.txt 파일들을 비교하는 시각화 보고서 생성.

step6a_build_vocab.py로 생성된 여러 vocab 설정(vocab_32768.txt, vocab_65536.txt 등)을
자동 탐색하여 비교/개별 보고서를 생성.

입력: tokens/part_*.txt (원본, 빈도 카운팅용) + vocab_*.txt (설정별)
출력: vocab_comparison.png       (비교 보고서)
      vocab_report_full.png      (전체 범위 보고서)
      vocab_report_{VS}.png      (설정별 보고서)

사용법:
  python step6b_vocab_report.py
"""

import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(DATA_DIR, "tokens")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# ──────────────────────────────────────────────
# 숫자 토큰 분리 (step6a와 동일)
# ──────────────────────────────────────────────

_SPLIT_POS_TAGS = frozenset({"SN", "W_SERIAL"})
_FW_TO_HW = str.maketrans("０１２３４５６７８９", "0123456789")


def _split_number_token(token: str) -> list[str]:
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
    return [sub for tok in tokens for sub in _split_number_token(tok)]


# ──────────────────────────────────────────────
# Vocab 파일 탐색 / 로드
# ──────────────────────────────────────────────

_VOCAB_RE = re.compile(r"^vocab_(\d+)\.txt$")


def _discover_vocabs() -> list[tuple[int, str]]:
    """vocab_NNNNN.txt 파일 탐색. (vocab_size, path) 리스트, 오름차순."""
    vocabs = []
    for f in os.listdir(DATA_DIR):
        m = _VOCAB_RE.match(f)
        if m:
            vocabs.append((int(m.group(1)), os.path.join(DATA_DIR, f)))
    vocabs.sort()
    return vocabs


def _load_vocab_set(path: str) -> set[str]:
    tokens = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens.add(line.split("\t", 1)[0])
    return tokens


# ──────────────────────────────────────────────
# 데이터 분석 (단일 패스)
# ──────────────────────────────────────────────


_VOCAB_SETS: list[set[str]] | None = None


def _init_analyze_worker(vocab_sets: list[set[str]]) -> None:
    global _VOCAB_SETS
    _VOCAB_SETS = vocab_sets


def _analyze_shard(path: str) -> tuple[Counter, int, list[int]]:
    counter: Counter = Counter()
    n_sents = 0
    n_kept = [0] * len(_VOCAB_SETS)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tokens = _expand_line(line.split(" "))
            counter.update(tokens)
            n_sents += 1
            for i, vs in enumerate(_VOCAB_SETS):
                if all(t in vs for t in tokens):
                    n_kept[i] += 1
    return counter, n_sents, n_kept


def _analyze(
    in_paths: list[str],
    vocab_sets: list[set[str]],
) -> tuple[Counter, int, list[int]]:
    """토큰 빈도 카운팅 + 각 vocab에 대한 문장 유지 수 (멀티프로세싱)."""
    n_workers = min(32, len(in_paths))
    with Pool(n_workers, initializer=_init_analyze_worker, initargs=(vocab_sets,)) as pool:
        results = pool.map(_analyze_shard, in_paths)

    counter: Counter = Counter()
    n_sents = 0
    n_kept = [0] * len(vocab_sets)
    for shard_counter, shard_n, shard_k in results:
        counter += shard_counter
        n_sents += shard_n
        for i in range(len(vocab_sets)):
            n_kept[i] += shard_k[i]
    return counter, n_sents, n_kept


# ──────────────────────────────────────────────
# 보고서 생성
# ──────────────────────────────────────────────


def _generate_report(
    vocabs: list[tuple[int, str]],
    counter: Counter,
    n_sents: int,
    n_kept: list[int],
    sent_min_freqs: list[int],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    report_path = os.path.join(DATA_DIR, "vocab_comparison.png")

    # 빈도 내림차순 정렬
    sorted_counts = np.array(sorted(counter.values(), reverse=True))
    ranks = np.arange(1, len(sorted_counts) + 1)
    total_occ = sorted_counts.sum()
    cumulative = np.cumsum(sorted_counts) / total_occ * 100

    # 색상
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = [f"{vs:,}" for vs, _ in vocabs]
    fig.suptitle(
        f"Step 6 Vocab 비교 ({', '.join(names)})",
        fontsize=14,
        fontweight="bold",
    )

    # 각 vocab 기본 수치
    config_info = []
    for i, (vs, vpath) in enumerate(vocabs):
        n_regular = vs - len(SPECIAL_TOKENS)
        freq = int(sorted_counts[n_regular - 1])
        cov = float(cumulative[n_regular - 1])
        ret = n_kept[i] / n_sents * 100
        config_info.append({
            "vs": vs, "n_regular": n_regular, "freq": freq,
            "cov": cov, "ret": ret, "kept": n_kept[i], "path": vpath,
        })

    # ── Panel 1: 순위-빈도 + vocab 경계 영역 ──
    ax = axes[0, 0]
    max_n = config_info[-1]["n_regular"]
    ax.plot(ranks[:max_n], sorted_counts[:max_n],
            color="steelblue", linewidth=1)
    # 각 config 간 영역 색칠
    prev_n = 0
    for i, ci in enumerate(config_info):
        c = colors[i % len(colors)]
        ax.fill_between(
            ranks[prev_n:ci["n_regular"]],
            sorted_counts[prev_n:ci["n_regular"]],
            alpha=0.15, color=c,
            label=f"{ci['vs']:,} (freq>={ci['freq']})",
        )
        ax.axvline(ci["n_regular"], color=c, linestyle="--",
                   linewidth=1.5, alpha=0.7)
        prev_n = ci["n_regular"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("토큰 순위")
    ax.set_ylabel("등장 횟수")
    ax.set_title("순위-빈도 분포 + vocab 경계")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 핵심 지표 그룹 막대 ──
    ax = axes[0, 1]
    x_arr = np.arange(len(vocabs))
    w = 0.35
    bars1 = ax.bar(x_arr - w / 2, [ci["cov"] for ci in config_info],
                   w, color="steelblue", label="토큰 커버리지 (%)")
    bars2 = ax.bar(x_arr + w / 2, [ci["ret"] for ci in config_info],
                   w, color="mediumseagreen", label="문장 유지율 (%)")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"{ci['vs']:,}" for ci in config_info])
    ax.set_xlabel("vocab size")
    ax.set_ylabel("%")
    ax.set_title("토큰 커버리지 vs 문장 유지율")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: vocab size–문장 유지율 연속 곡선 ──
    ax = axes[1, 0]
    sent_min_arr = np.array(sent_min_freqs)
    # vocab size 후보를 빈도 기반으로 계산
    vs_points = sorted(set(
        list(range(1000, 20001, 1000))
        + list(range(20000, 100001, 2000))
        + list(range(100000, 200001, 10000))
    ))
    vs_points = [v for v in vs_points if v <= len(sorted_counts)]
    ret_curve = []
    for v in vs_points:
        n_reg = v - len(SPECIAL_TOKENS)
        if n_reg <= 0 or n_reg > len(sorted_counts):
            continue
        mc = int(sorted_counts[n_reg - 1])
        ret_curve.append((v, float(np.sum(sent_min_arr >= mc)) / n_sents * 100))
    if ret_curve:
        vs_x, ret_y = zip(*ret_curve)
        ax.plot(vs_x, ret_y, color="steelblue", linewidth=1.5)
    for i, ci in enumerate(config_info):
        c = colors[i % len(colors)]
        ax.plot(ci["vs"], ci["ret"], "o", color=c, markersize=10, zorder=5)
        ax.annotate(
            f"{ci['vs']:,}\n{ci['ret']:.1f}%",
            xy=(ci["vs"], ci["ret"]),
            xytext=(10, 5), textcoords="offset points",
            fontsize=9, color=c, fontweight="bold",
        )
    ax.set_xlabel("vocab size")
    ax.set_ylabel("문장 유지율 (%)")
    ax.set_title("vocab size vs 문장 유지율")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: 구간별 POS 증분 (한국어 라벨) ──
    ax = axes[1, 1]
    pos_keys = ["NNG", "NNP", "VV", "VA", "SL", "NNB", "XS*", "기타"]
    pos_kr = {
        "NNG": "일반명사", "NNP": "고유명사", "VV": "동사",
        "VA": "형용사", "SL": "외국어", "NNB": "의존명사",
        "XS*": "접미사", "기타": "기타",
    }
    pos_clr = {
        "NNG": "#3498db", "NNP": "#e74c3c", "VV": "#2ecc71",
        "VA": "#9b59b6", "SL": "#f39c12", "NNB": "#1abc9c",
        "XS*": "#e67e22", "기타": "#95a5a6",
    }

    def _count_pos(vpath):
        cnt: Counter = Counter()
        with open(vpath, encoding="utf-8") as f:
            for line in f:
                tok = line.split("\t", 1)[0]
                if tok in SPECIAL_TOKENS:
                    continue
                slash = tok.rfind("/")
                if slash > 0:
                    pos = tok[slash + 1:]
                    if pos.startswith("XS"):
                        cnt["XS*"] += 1
                    elif pos in ("NNG", "NNP", "VV", "VA", "SL", "NNB"):
                        cnt[pos] += 1
                    else:
                        cnt["기타"] += 1
                else:
                    cnt["기타"] += 1
        return cnt

    all_pos = [_count_pos(ci["path"]) for ci in config_info]
    segment_labels = [f"~{config_info[0]['vs']:,}"]
    segment_data = [all_pos[0]]
    for i in range(1, len(config_info)):
        segment_labels.append(
            f"{config_info[i-1]['vs']:,}\n~{config_info[i]['vs']:,}")
        diff = Counter()
        for p in pos_keys:
            diff[p] = all_pos[i].get(p, 0) - all_pos[i - 1].get(p, 0)
        segment_data.append(diff)

    x_seg = np.arange(len(segment_labels))
    bottom = np.zeros(len(segment_labels))
    for pk in pos_keys:
        vals = [sd.get(pk, 0) for sd in segment_data]
        ax.bar(x_seg, vals, 0.7, bottom=bottom,
               color=pos_clr[pk], label=pos_kr[pk])
        bottom += np.array(vals)
    ax.set_xticks(x_seg)
    ax.set_xticklabels(segment_labels, fontsize=9)
    ax.set_xlabel("vocab 구간")
    ax.set_ylabel("추가된 토큰 수")
    ax.set_title("구간별 추가 토큰 품사 분포")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n비교 보고서: {report_path}")


def _generate_full_report(
    counter: Counter,
    n_sents: int,
    sent_min_freqs: list[int],
) -> None:
    """전체 vocab 범위에 대한 보고서 생성."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    report_path = os.path.join(DATA_DIR, "vocab_report_full.png")

    all_counts = np.array(sorted(counter.values(), reverse=True))
    ranks = np.arange(1, len(all_counts) + 1)
    total_occ = all_counts.sum()
    cumulative = np.cumsum(all_counts) / total_occ * 100
    max_count = int(all_counts[0])
    sent_min_arr = np.array(sent_min_freqs)

    # min_count 후보 범위
    mc_range = sorted(set(
        list(range(1, 21))
        + list(range(25, 101, 5))
        + [150, 200, 300, 500, 1000]
    ))
    mc_range = [mc for mc in mc_range if mc <= max_count]

    # 논문 참조선
    ref_models = [
        ("Bengio 2003", 17_964, "orange"),
        ("Word2Vec", 30_000, "purple"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"전체 Vocab 보고서 ({len(all_counts):,} unique tokens, "
        f"{n_sents:,} sentences)",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel 1: 순위-빈도 (Zipf, 전체) ──
    ax = axes[0, 0]
    ax.plot(ranks, all_counts, color="steelblue", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("토큰 순위 (빈도 내림차순)")
    ax.set_ylabel("등장 횟수")
    ax.set_title("순위-빈도 분포 (Zipf)")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 빈도 히스토그램 (전체) ──
    ax = axes[0, 1]
    bins = np.logspace(0, np.log10(max_count + 1), 60)
    ax.hist(all_counts, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("등장 횟수 (코퍼스 전체)")
    ax.set_ylabel("토큰 종류 수")
    ax.set_title("토큰 빈도 분포")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: min_count별 vocab size (전체) ──
    ax = axes[1, 0]
    vocab_sizes = [
        int(np.sum(all_counts >= mc)) + len(SPECIAL_TOKENS) for mc in mc_range
    ]
    ax.plot(mc_range, vocab_sizes, "b-o", markersize=3, linewidth=1.5)
    for name, ref_vocab, color in ref_models:
        ax.axhline(
            ref_vocab, color=color, linestyle="--", linewidth=1.5,
            label=f"{name} (~{ref_vocab:,})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("min_count")
    ax.set_ylabel("vocab size")
    ax.set_title("min_count별 vocab size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: min_count별 문장 유지율 (전체) ──
    ax = axes[1, 1]
    retentions = [
        float(np.sum(sent_min_arr >= mc)) / n_sents * 100 for mc in mc_range
    ]
    ax.plot(mc_range, retentions, "g-o", markersize=3, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("min_count")
    ax.set_ylabel("문장 유지율 (%)")
    ax.set_title("min_count별 문장 유지율")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  전체 보고서: {report_path}")


_COUNTER: Counter | None = None


def _init_freq_worker(counter: Counter) -> None:
    global _COUNTER
    _COUNTER = counter


def _freq_shard(path: str) -> list[int]:
    freqs: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tokens = _expand_line(line.split(" "))
            freqs.append(min(_COUNTER[t] for t in tokens))
    return freqs


def _compute_sent_min_freqs(
    in_paths: list[str], counter: Counter
) -> list[int]:
    """각 문장의 최소 토큰 빈도 계산 (멀티프로세싱 2nd pass)."""
    n_workers = min(32, len(in_paths))
    with Pool(n_workers, initializer=_init_freq_worker, initargs=(counter,)) as pool:
        results = pool.map(_freq_shard, in_paths)
    return [f for shard in results for f in shard]


def _generate_individual_reports(
    vocabs: list[tuple[int, str]],
    counter: Counter,
    n_sents: int,
    n_kept: list[int],
    sent_min_freqs: list[int],
) -> None:
    """각 vocab 설정별 개별 보고서 생성.

    원본 vocab_report.png와 동일한 4패널 구성:
      1. 순위-빈도 (Zipf)      2. 빈도 히스토그램
      3. min_count별 vocab size  4. min_count별 문장 유지율
    vocab 범위에 맞춰 축 범위를 조정.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    all_counts = np.array(sorted(counter.values(), reverse=True))
    sent_min_arr = np.array(sent_min_freqs)

    # 논문 참조선
    ref_models = [
        ("Bengio 2003", 17_964, "orange"),
        ("Word2Vec", 30_000, "purple"),
    ]

    for idx, (vs, _) in enumerate(vocabs):
        report_path = os.path.join(DATA_DIR, f"vocab_report_{vs}.png")
        n_regular = vs - len(SPECIAL_TOKENS)
        freq_boundary = int(all_counts[n_regular - 1])
        retention = n_kept[idx] / n_sents * 100

        # vocab 포함 토큰만
        v_counts = all_counts[:n_regular]
        v_ranks = np.arange(1, n_regular + 1)

        # min_count 범위: 경계 빈도부터 (vocab 포함 범위만)
        mc_range = sorted(set(
            list(range(max(1, freq_boundary), freq_boundary + 20))
            + list(range(freq_boundary, freq_boundary * 5, max(1, freq_boundary // 10)))
            + list(range(freq_boundary * 5, freq_boundary * 20 + 1, max(1, freq_boundary // 2)))
        ))
        mc_max = int(all_counts[0])
        mc_range = [mc for mc in mc_range if mc <= mc_max]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Vocab {vs:,} 보고서 (경계 빈도={freq_boundary:,}, "
            f"문장 유지율={retention:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )

        # ── Panel 1: 순위-빈도 (Zipf, vocab 범위만) ──
        ax = axes[0, 0]
        ax.plot(v_ranks, v_counts, color="steelblue", linewidth=1)
        ax.axhline(
            freq_boundary, color="red", linestyle="--", linewidth=1.5,
            label=f"경계 빈도={freq_boundary:,}",
        )
        ax.annotate(
            f"vocab={vs:,}",
            xy=(n_regular, freq_boundary),
            xytext=(-15, 12),
            textcoords="offset points",
            fontsize=10, color="red", fontweight="bold",
            ha="right",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("토큰 순위 (빈도 내림차순)")
        ax.set_ylabel("등장 횟수")
        ax.set_title("순위-빈도 분포 (Zipf)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 2: 빈도 히스토그램 (vocab 범위만) ──
        ax = axes[0, 1]
        bins = np.logspace(
            np.log10(max(freq_boundary, 1)),
            np.log10(int(v_counts[0]) + 1),
            50,
        )
        ax.hist(v_counts, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(
            freq_boundary, color="red", linestyle="--", linewidth=1.5,
            label=f"경계 빈도={freq_boundary:,}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("등장 횟수")
        ax.set_ylabel("토큰 종류 수")
        ax.set_title("토큰 빈도 분포 (vocab 내)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 3: min_count별 vocab size (경계부터만) ──
        ax = axes[1, 0]
        vocab_sizes = [
            int(np.sum(all_counts >= mc)) + len(SPECIAL_TOKENS)
            for mc in mc_range
        ]
        ax.plot(mc_range, vocab_sizes, "b-o", markersize=3, linewidth=1.5)
        ax.plot(freq_boundary, vs, "ro", markersize=9, zorder=5)
        ax.annotate(
            f"{vs:,}", xy=(freq_boundary, vs),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, color="red", fontweight="bold",
        )
        # 논문 참조선 (이 vocab 범위 이하만)
        for name, ref_vocab, color in ref_models:
            if ref_vocab <= vs:
                ax.axhline(
                    ref_vocab, color=color, linestyle="--", linewidth=1.5,
                    label=f"{name} (~{ref_vocab:,})",
                )
        ax.set_xscale("log")
        ax.set_xlabel("min_count (경계 빈도)")
        ax.set_ylabel("vocab size")
        ax.set_title("min_count별 vocab size")
        ax.set_ylim(0, vs * 1.1)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

        # ── Panel 4: min_count별 문장 유지율 (경계부터만) ──
        ax = axes[1, 1]
        retentions_curve = [
            float(np.sum(sent_min_arr >= mc)) / n_sents * 100
            for mc in mc_range
        ]
        ax.plot(mc_range, retentions_curve, "g-o", markersize=3, linewidth=1.5)
        ax.plot(freq_boundary, retention, "ro", markersize=9, zorder=5)
        ax.annotate(
            f"{retention:.1f}%", xy=(freq_boundary, retention),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, color="red", fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_xlabel("min_count (경계 빈도)")
        ax.set_ylabel("문장 유지율 (%)")
        ax.set_title("min_count별 문장 유지율")
        ax.set_ylim(0, min(retention * 1.3, 100))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(report_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  개별 보고서: {report_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    vocabs = _discover_vocabs()
    if not vocabs:
        print("vocab_*.txt 파일 없음. step6a_build_vocab.py를 먼저 실행하세요.")
        return

    print(f"발견된 vocab 설정: {len(vocabs)}")
    for vs, path in vocabs:
        print(f"  {vs:>8,}  {os.path.basename(path)}")

    # vocab set 로드
    print("\nvocab 로드 ...")
    vocab_sets = [_load_vocab_set(path) for _, path in vocabs]

    # 입력 샤드
    in_paths = sorted(
        os.path.join(IN_DIR, f)
        for f in os.listdir(IN_DIR)
        if f.endswith(".txt")
    )
    if not in_paths:
        print(f"입력 없음: {IN_DIR}/*.txt")
        return

    # 단일 패스: 빈도 카운팅 + 문장 유지율
    print(f"\n분석 중 ({len(in_paths)} shards) ...")
    t0 = time.time()
    counter, n_sents, n_kept = _analyze(in_paths, vocab_sets)
    elapsed = time.time() - t0
    print(f"  완료 ({elapsed:.1f}s, {len(counter):,} unique, {n_sents:,} sentences)")

    for i, (vs, _) in enumerate(vocabs):
        ret = n_kept[i] / n_sents * 100
        print(f"  vocab {vs:>6,}: {n_kept[i]:>10,} / {n_sents:,} ({ret:.1f}%)")

    print("\n문장별 최소 빈도 계산 (2nd pass) ...")
    t1 = time.time()
    sent_min_freqs = _compute_sent_min_freqs(in_paths, counter)
    print(f"  완료 ({time.time() - t1:.1f}s)")

    _generate_report(vocabs, counter, n_sents, n_kept, sent_min_freqs)
    _generate_full_report(counter, n_sents, sent_min_freqs)
    _generate_individual_reports(vocabs, counter, n_sents, n_kept, sent_min_freqs)


if __name__ == "__main__":
    main()