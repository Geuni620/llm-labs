"""N별 글자 빈도표에서 최대 빈도 또는 가중 추첨으로 문장을 생성합니다."""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import unicodedata


ROOT = Path(__file__).resolve().parent.parent
COUNTS_FILE = ROOT / "bigram" / "counts.csv"
OUTPUT_DIR = ROOT / "bigram" / "generation"
END_PUNCTUATION = ".!?"


def model_folder(n):
    folders = {1: "unigram", 2: "bigram", 3: "trigram"}
    return ROOT / folders[n] if n in folders else ROOT / "ngram" / f"n{n}"


def load_ngram(n, path=None):
    if n < 1:
        raise ValueError("N은 1 이상이어야 합니다.")
    path = model_folder(n) / "counts.csv" if path is None else path
    model = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"context", "next", "count"}.issubset(reader.fieldnames or []):
            raise ValueError("빈도표에는 context, next, count 컬럼이 필요합니다.")
        for row in reader:
            context, following = row["context"], row["next"]
            count = int(row["count"])
            if context is None or following is None or len(context) != n - 1 or len(following) != 1 or count < 1:
                raise ValueError(f"N={n}의 표는 {n-1}글자 문맥, 다음 한 글자, 양수 횟수로 구성됩니다.")
            if (context, following) in seen:
                raise ValueError("빈도표에 같은 N-그램 행이 두 번 들어 있습니다.")
            seen.add((context, following))
            model[context].append((following, count))
    if not model:
        raise ValueError("N-그램 빈도표가 비어 있습니다.")
    return dict(model)


def load_bigrams(path=COUNTS_FILE):
    return load_ngram(2, path)


def sample_candidate(candidates, rng):
    """횟수만큼의 추첨 구간을 만들고 정수 티켓 하나로 후보를 고릅니다."""
    total = sum(candidate["count"] for candidate in candidates)
    ticket = rng.randrange(total)
    cumulative = 0
    for candidate in candidates:
        cumulative += candidate["count"]
        if ticket < cumulative:
            return candidate, ticket
    raise ValueError("추첨 구간에 해당하는 후보가 없습니다.")


def generate_ngram(model, prefix, n=2, method="greedy", max_chars=100, rng=None):
    if n < 1 or method not in {"greedy", "sample"}:
        raise ValueError("N은 1 이상이고 선택 방법은 greedy 또는 sample이어야 합니다.")
    prefix = unicodedata.normalize("NFC", prefix)
    if len(prefix) < n - 1:
        raise ValueError(f"N={n}에는 공백을 포함해 {n-1}글자 이상의 접두어가 필요합니다.")
    if max_chars < 1 or len(prefix) > max_chars:
        raise ValueError("최대 길이는 1 이상이고 접두어 길이 이상이어야 합니다.")

    rng = random.Random(42) if rng is None else rng
    text = prefix
    steps = []
    while True:
        if text and text[-1] in END_PUNCTUATION:
            stop_reason = "punctuation"
            break
        if len(text) >= max_chars:
            stop_reason = "max_length"
            break

        # 문맥은 마지막 N-1글자입니다. 유니그램의 문맥은 빈 문자열입니다.
        context = text[-(n - 1):] if n > 1 else ""
        row = model.get(context, [])
        if not row:
            stop_reason = "unseen_context"
            break
        total = sum(count for _, count in row)
        candidates = [
            {"next": character, "count": count, "probability": count / total}
            for character, count in sorted(row, key=lambda item: -item[1])
        ]
        # 같은 빈도에서는 CSV의 첫 등장 순서를 유지합니다.
        if method == "greedy":
            selected, ticket = candidates[0], None
        else:
            selected, ticket = sample_candidate(candidates, rng)
        text += selected["next"]
        steps.append({
            "step": len(steps) + 1,
            "context": context,
            "context_total": total,
            "candidates": candidates,
            "selected": selected["next"],
            "selected_count": selected["count"],
            "selected_model_probability": selected["probability"],
            "text": text,
        })
        if method == "sample":
            steps[-1]["sampling_ticket"] = ticket

    return {
        "n": n,
        "method": method,
        "prefix": prefix,
        "max_chars": max_chars,
        "text": text,
        "stop_reason": stop_reason,
        "steps": steps,
    }


def generate_greedy(model, prefix, max_chars=100, n=2):
    return generate_ngram(model, prefix, n=n, max_chars=max_chars)


def generate_batch(model, prefix, n=2, method="sample", seed=42, count=10, max_chars=100):
    if count < 1:
        raise ValueError("생성 개수는 1 이상이어야 합니다.")
    # 배치 전체에서 하나의 난수 흐름을 사용합니다.
    rng = random.Random(seed)
    return [
        generate_ngram(model, prefix, n=n, method=method, max_chars=max_chars, rng=rng)
        for _ in range(count)
    ]


def show_trace(result, top=10):
    print(f"접두어: {result['prefix']}")
    for step in result["steps"]:
        context = step["context"].replace(" ", "␣")
        print(f"\n[{step['step']}] 문맥 {context!r} · 분모 {step['context_total']}회")
        print("후보\t횟수\t데이터 비율\t선택")
        for candidate in step["candidates"][:top]:
            character = "공백(␣)" if candidate["next"] == " " else candidate["next"]
            chosen = "선택" if candidate["next"] == step["selected"] else ""
            print(f"{character}\t{candidate['count']}\t{candidate['probability']:.2%}\t{chosen}")
        print(f"누적 텍스트: {step['text']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="N-그램의 N. 문맥은 마지막 N-1글자입니다.")
    parser.add_argument("--prefix", default="하", help="시작할 접두어. 기본 '하'.")
    parser.add_argument("--method", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--seed", type=int, default=42, help="추첨 시드. 배치 시작에 한 번 설정합니다.")
    parser.add_argument("--count", type=int, default=1, help="생성 횟수. 중복된 결과도 기록합니다.")
    parser.add_argument("--max-chars", type=int, default=100, help="접두어를 포함한 최대 글자 수.")
    parser.add_argument("--trace", action="store_true", help="첫 문장의 선택 과정을 출력합니다. 파일에는 모든 문장의 기록을 저장합니다.")
    parser.add_argument("--top", type=int, default=10, help="로그에 표시할 후보 수. 모델의 후보는 모두 유지합니다.")
    args = parser.parse_args()
    if args.n < 1 or args.count < 1 or args.top < 1:
        parser.error("--n, --count, --top은 1 이상이어야 합니다.")

    try:
        folder = model_folder(args.n)
        counts_file = folder / "counts.csv"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        source_bytes = (ROOT / "data" / "daily_answers.txt").read_bytes()
        if summary["n"] != args.n or hashlib.sha256(source_bytes).hexdigest() != summary["source_sha256"]:
            raise ValueError("답변 데이터와 빈도표의 기준이 다릅니다. count_ngrams.py로 다시 집계해 주세요.")
        results = generate_batch(
            load_ngram(args.n), args.prefix, n=args.n, method=args.method,
            seed=args.seed, count=args.count, max_chars=args.max_chars,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    input_info = {
        "counts_file": str(counts_file.relative_to(ROOT)),
        "counts_sha256": hashlib.sha256(counts_file.read_bytes()).hexdigest(),
        "answers_file": "data/daily_answers.txt",
        "answers_sha256": summary["source_sha256"],
        "answer_count": summary["answer_count"],
    }
    texts = [result["text"] for result in results]
    answers = source_bytes.decode("utf-8").splitlines()
    answer_set = set(answers)
    batch = {
        "n": args.n,
        "context_length": args.n - 1,
        "method": args.method,
        "prefix": results[0]["prefix"],
        "seed": args.seed if args.method == "sample" else None,
        "count": args.count,
        "max_chars": args.max_chars,
        "input": input_info,
        "unique_sentences": len(set(texts)),
        "exact_answer_matches": sum(text in answer_set for text in texts),
        "source_fragment_matches": sum(any(text in answer for answer in answers) for text in texts),
        "results": results,
    }
    output_dir = folder / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "greedy" and args.count == 1:
        text_file, trace_file = output_dir / "result.txt", output_dir / "trace.json"
        payload = {**results[0], "input": input_info}
    else:
        stem = f"sample_seed{args.seed}" if args.method == "sample" else "greedy"
        text_file, trace_file = output_dir / f"{stem}_sentences.txt", output_dir / f"{stem}_traces.json"
        payload = batch
    text_file.write_text("\n".join(texts) + "\n", encoding="utf-8")
    trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    method_label = "최대 빈도 선택" if args.method == "greedy" else "횟수 비례 추첨"
    print(f"N={args.n} · 문맥 {args.n-1}글자 · {method_label} · 답변 {summary['answer_count']:,}개 기준")
    if args.trace:
        show_trace(results[0], args.top)
    reasons = {
        "punctuation": "문장부호 선택",
        "max_length": "최대 길이 도달",
        "unseen_context": "다음 후보가 없는 문맥",
    }
    for index, result in enumerate(results, start=1):
        print(f"{index:02d}. {result['text']} [{reasons[result['stop_reason']]}]")
    print(f"서로 다른 결과: {batch['unique_sentences']}/{args.count}")
    print(f"원본 답변과 완전 일치: {batch['exact_answer_matches']}/{args.count}")
    print(f"저장: {text_file}")
    print(f"추적 기록: {trace_file}")


if __name__ == "__main__":
    main()
