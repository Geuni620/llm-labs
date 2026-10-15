"""같은 답변 데이터에서 N별 글자 묶음의 빈도와 조건부 확률을 저장합니다."""

from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "daily_answers.txt"


def count_ngrams(answers, n):
    if n < 1:
        raise ValueError("N은 1 이상이어야 합니다.")
    counts = Counter()
    for answer in answers:
        # 답변 하나 안에서 한 글자씩 이동합니다. 다른 답변과 연결하지 않습니다.
        for start in range(len(answer) - n + 1):
            counts[answer[start : start + n]] += 1
    return counts


def context_totals(counts):
    totals = Counter()
    for gram, count in counts.items():
        totals[gram[:-1]] += count
    return totals


def save_counts(counts, n, answers, source_bytes):
    folders = {1: "unigram", 2: "bigram", 3: "trigram"}
    folder = ROOT / folders[n] if n in folders else ROOT / "ngram" / f"n{n}"
    folder.mkdir(parents=True, exist_ok=True)
    totals = context_totals(counts)
    with (folder / "counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["ngram", "context", "next", "count", "context_total", "probability", "percent"],
        )
        writer.writeheader()
        for gram, count in sorted(counts.items(), key=lambda item: (item[0][:-1], -item[1])):
            total = totals[gram[:-1]]
            writer.writerow({
                "ngram": gram,
                "context": gram[:-1],
                "next": gram[-1],
                "count": count,
                "context_total": total,
                "probability": count / total,
                "percent": count / total * 100,
            })
    summary = {
        "source": "data/daily_answers.txt",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "answer_count": len(answers),
        "unit": "Unicode character; spaces and punctuation included",
        "n": n,
        "context_length": n - 1,
        "ngram_occurrences": sum(counts.values()),
        "distinct_ngrams": len(counts),
        "distinct_contexts": len(totals),
        "boundary_tokens_added": False,
        "smoothing_applied": False,
        "probability": "count / sum of counts with the same context",
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"N={n}: {sum(counts.values()):,}회 / {len(counts):,}개 조합 → {folder.relative_to(ROOT)}/counts.csv")


def show_context(counts, context, top):
    total = context_totals(counts)[context]
    print(f"문맥 {context!r} 뒤의 전체 횟수: {total:,}회")
    print("묶음\t횟수\t분모\t비율")
    candidates = [(gram, count) for gram, count in counts.most_common() if gram[:-1] == context]
    for gram, count in candidates[:top]:
        print(f"{gram.replace(' ', '␣')}\t{count}\t{total}\t{count / total:.2%}")
    if not candidates:
        print("이 문맥 뒤의 글자 묶음은 관측되지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, help="묶는 글자 수. 생략하면 1, 2, 3, 4를 저장합니다.")
    parser.add_argument("--context", help="조회할 앞 글자 N-1개. N=1은 빈 문자열입니다.")
    parser.add_argument("--top", type=int, default=10, help="조회할 후보 수. 기본 10개.")
    args = parser.parse_args()
    if args.n is not None and args.n < 1:
        parser.error("--n은 1 이상이어야 합니다.")
    if args.top < 1:
        parser.error("--top은 1 이상이어야 합니다.")
    if args.context is not None and (args.n is None or len(args.context) != args.n - 1):
        parser.error("--context는 --n과 함께 지정하며, 길이가 N-1이어야 합니다.")

    source_bytes = SOURCE.read_bytes()
    answers = source_bytes.decode("utf-8").splitlines()
    if not answers or any(not answer for answer in answers):
        parser.error("학습용 답변 파일은 비어 있지 않은 답변을 한 줄에 하나씩 담아야 합니다.")
    orders = [args.n] if args.n is not None else [1, 2, 3, 4]
    if max(orders) > max(map(len, answers)):
        parser.error("N이 모든 답변의 길이보다 커서 묶음을 만들 수 없습니다.")
    for n in orders:
        counts = count_ngrams(answers, n)
        save_counts(counts, n, answers, source_bytes)
        if args.context is not None or n == 1:
            show_context(counts, args.context if args.context is not None else "", args.top)


if __name__ == "__main__":
    main()
