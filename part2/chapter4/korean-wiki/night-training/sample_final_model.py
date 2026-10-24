"""Read-only final-model samples and exhaustive training-membership checks.

Run prepare/decode with .venv-kiwi; run infer/match with .venv-training.
Kiwi and the C engine never share a process. No training step or checkpoint save.
Each phase refuses to replace an existing result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import mmap
from pathlib import Path
import re
import sys
import time

BASE = Path(__file__).resolve().parent
WIKI = BASE.parent
sys.path.insert(0, str(WIKI))
PREFIX = WIKI / "reports/nplm-final-test20-20261024"
SELECTED = WIKI / "reports/resume-20261023-r2-selected-model.json"
PROMPTS = [
    "서울은", "대한민국의 수도는", "부산은", "제주도는", "한강은",
    "한국의 역사는", "조선 시대에는", "이 도시는", "이 지역의 인구는", "이 학교는",
    "이 회사는", "이 작품은", "이 영화는", "이 음악은", "과학 기술은",
    "컴퓨터는", "그는", "1987년에는", "2025년에는", "2037년에는",
]


def now():
    return datetime.now().astimezone().isoformat()


def identity(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def path_for(suffix):
    return Path(str(PREFIX) + suffix)


def read(suffix):
    return json.loads(path_for(suffix).read_text(encoding="utf-8"))


def write(suffix, value):
    path = path_for(suffix)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def normalize_surface(value):
    return " ".join(value.split())


def strip_edge_spaces(tokens):
    tokens = list(tokens)
    while tokens and tokens[0] == "<sp>/WS":
        tokens.pop(0)
    while tokens and tokens[-1] == "<sp>/WS":
        tokens.pop()
    return tokens


def prepare():
    from importlib.metadata import version
    from kiwipiepy import Kiwi
    from step5_tokenize import tokenize_sentence
    from step6_split_numbers import split_numeric_token

    selected = json.loads(SELECTED.read_text())
    vocabulary = json.loads(Path(selected["vocabulary"]).read_text())["tokens"]
    lookup = {token: index for index, token in enumerate(vocabulary)}
    kiwi = Kiwi(num_workers=1, model_type="cong-global")
    examples = []
    for index, prompt in enumerate(PROMPTS, 1):
        raw = tokenize_sentence(kiwi, prompt + " ").split(" ")
        tokens = [piece for token in raw for piece in split_numeric_token(token)]
        examples.append({"id": index, "prompt": prompt, "prompt_tokens": tokens,
                         "prompt_ids": [lookup.get(token, 1) for token in tokens],
                         "prompt_oov_tokens": [token for token in tokens if token not in lookup],
                         "seed": 4200 + index})
    write(".inputs.json", {"kind": "FIXED_20_FINAL_MODEL_INPUTS", "at": now(),
          "selection": selected, "selected_descriptor_identity": identity(SELECTED),
          "sampling": {"top_k": 40, "temperature": .8, "max_new_tokens": 160,
                       "stop": "first generated SF ending, EOS, or token cap",
                       "selection": "One sample per predeclared prompt; no rejection or retries."},
          "tokenizer": {"kiwipiepy": version("kiwipiepy"), "model_type": "cong-global",
                        "numeric_policy": "SN/W_SERIAL to individual digit/SN and separator/SW"},
          "examples": examples})
    print(json.dumps({"phase": "prepared", "examples": len(examples),
                      "prompt_oov_tokens": sum(len(x["prompt_oov_tokens"]) for x in examples)}), flush=True)


def state_identity(engine):
    return {"arrays": {name: hashlib.sha256(engine.arrays[name].tobytes()).hexdigest()
                       for name in ("parameters", "moment1", "moment2")},
            "visited": hashlib.sha256(engine.visited.tobytes()).hexdigest(),
            "sampler": engine.sampler, "info": engine.info}


def infer():
    import fcntl
    from contextlib import ExitStack
    import numpy as np
    import checkpoint
    from engine import Engine, LIBRARY
    from run import corpus_identity

    inputs = read(".inputs.json")
    selected = inputs["selection"]
    checkpoint_path = Path(selected["checkpoint"]["path"])
    if identity(checkpoint_path) != selected["checkpoint"]["identity"]:
        raise ValueError("Selected checkpoint changed")
    run = json.loads((checkpoint_path.parent / "run.json").read_text())
    if run["config"] != selected["config"]:
        raise ValueError("Selected model configuration changed")
    data, _ = corpus_identity(Path(selected["corpus"]))
    if data != run["identity"]["data"] or identity(LIBRARY) != run["identity"]["library"]:
        raise ValueError("Bound corpus/library changed")
    for name, expected in run["identity"]["scripts"].items():
        if identity(BASE / name) != expected:
            raise ValueError("Frozen trainer source changed: " + name)
    tokens = json.loads(Path(selected["vocabulary"]).read_text())["tokens"]
    sampling = inputs["sampling"]
    began = time.monotonic()
    with ExitStack() as stack:
        for name in ("cpu-benchmark/.timing.lock", "reports/.nplm-benchmark.lock"):
            lock = stack.enter_context((WIKI / name).open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(selected["corpus"], selected["config"])
        stack.callback(engine.close)
        checkpoint.restore(checkpoint_path, engine, run["identity"])
        before = state_identity(engine)
        if (before["info"]["updates"] != selected["updates"]
                or before["info"]["consumed"] != selected["target_predictions"]):
            raise ValueError("Wrong final model counters")
        examples = []
        for item in inputs["examples"]:
            rng = np.random.default_rng(item["seed"])
            context = [2] * engine.config["context_size"] + item["prompt_ids"]
            generated, top_five = [], None
            stop_reason = "token_cap"
            for _ in range(sampling["max_new_tokens"]):
                logits = engine.logits(np.asarray([context[-engine.config["context_size"]:]],
                                                  dtype=np.uint32))[0].astype(np.float64)
                if not np.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite generation logits")
                logits[:3] = -np.inf
                order = np.argsort(-logits, kind="stable")[:sampling["top_k"]]
                if top_five is None:
                    full = np.exp(logits - logits.max()); full /= full.sum()
                    top_five = [{"token": tokens[int(i)], "probability": float(full[i])}
                                for i in order[:5]]
                values = logits[order] / sampling["temperature"]
                probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
                token_id = int(rng.choice(order, p=probabilities))
                if token_id == 3:
                    stop_reason = "eos"
                    break
                generated.append(token_id)
                context.append(token_id)
                form, _, tag = tokens[token_id].rpartition("/")
                if tag == "SF" and any(char in form for char in ".?!"):
                    stop_reason = "sentence_ending"
                    break
            sequence = [tokens[i] for i in generated]
            trigrams = list(zip(sequence, sequence[1:], sequence[2:]))
            examples.append({**item, "generated_ids": generated, "generated_tokens": sequence,
                             "generation_terminated_by": stop_reason,
                             "sentence_ending_reached": stop_reason == "sentence_ending",
                             "repeated_trigram_fraction": 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.,
                             "first_next_token_candidates": top_five})
            print(json.dumps({"phase": "inference", "sample": item["id"], "tokens": len(generated),
                              "stop": stop_reason}), flush=True)
        after = state_identity(engine)
        if before != after:
            raise AssertionError("Inference changed weights, optimizer or sampler")
    after_file = identity(checkpoint_path)
    if after_file != selected["checkpoint"]["identity"]:
        raise AssertionError("Checkpoint file changed")
    write(".inference.json", {"kind": "ACTUAL_READ_ONLY_FINAL_CHECKPOINT_INFERENCE", "at": now(),
          "inputs_identity": identity(path_for(".inputs.json")), "selection": selected,
          "sampling": sampling, "seconds": time.monotonic() - began,
          "state_before": before, "state_after": after, "state_unchanged": before == after,
          "checkpoint_identity_after": after_file, "training_steps": 0, "examples": examples})


def decode():
    from kiwipiepy import Kiwi
    from step5_tokenize import detokenize

    data = read(".inference.json")
    kiwi = Kiwi(num_workers=1, model_type="cong-global")
    for item in data["examples"]:
        item["full_surface"] = detokenize(" ".join(item["prompt_tokens"] + item["generated_tokens"]), kiwi)
        item["generated_surface"] = detokenize(" ".join(item["generated_tokens"]), kiwi) if item["generated_tokens"] else ""
        item["normalized_full_surface"] = normalize_surface(item["full_surface"])
        item["normalized_generated_surface"] = normalize_surface(item["generated_surface"])
    write(".decoded.json", {**data, "decoded_at": now(),
          "surface_note": "Kiwi.join restores form/POS tokens; it can normalize contractions and spaces."})
    print(json.dumps({"phase": "decoded", "samples": len(data["examples"])}), flush=True)


def find_id_sequence(buffer, offset, records, token_ids, context_size=8):
    """Count aligned contiguous matches inside actual paragraph content only."""
    import numpy as np

    if not token_ids or any(token_id < 4 for token_id in token_ids):
        return {"count": 0, "examples": []}
    needle = np.asarray(token_ids, dtype="<u2").tobytes()
    starts = records["start"]
    count, witnesses, position = 0, [], offset
    while (position := buffer.find(needle, position)) != -1:
        relative = position - offset
        if relative >= 0 and relative % 2 == 0:
            token_position = relative // 2
            paragraph = int(np.searchsorted(starts, token_position, side="right")) - 1
            if paragraph >= 0:
                record = records[paragraph]
                start, length = int(record["start"]), int(record["length"])
                if start + context_size <= token_position and token_position + len(token_ids) <= start + length - 1:
                    count += 1
                    if len(witnesses) < 3:
                        witnesses.append({"train_paragraph_index": paragraph, "doc_key": int(record["doc_key"]),
                                          "array_token_offset": token_position,
                                          "paragraph_token_offset": token_position - start})
        position += 1
    return {"count": count, "examples": witnesses}


def match():
    import numpy as np
    from training_corpus import paragraphs, document_split

    began = time.monotonic()
    data = read(".decoded.json")
    selected = data["selection"]
    corpus = Path(selected["corpus"])
    manifest = json.loads((corpus / "manifest.json").read_text())
    run = json.loads((Path(selected["checkpoint"]["path"]).parent / "run.json").read_text())
    ids_path = corpus / "train.ids.npy"
    records_path = corpus / "train.paragraphs.npy"
    bound = {path.name: identity(path) for path in (ids_path, records_path)}
    if any(value != run["identity"]["data"][key] for key, value in bound.items()):
        raise ValueError("Actual training arrays changed")
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    records = np.load(records_path, mmap_mode="r", allow_pickle=False)
    vocabulary = json.loads((corpus / "vocabulary.json").read_text())["tokens"]
    lookup = {token: index for index, token in enumerate(vocabulary)}
    patterns, raw_candidates = {}, defaultdict(list)
    for item in data["examples"]:
        for label in ("full", "generated"):
            surface = item["normalized_" + label + "_surface"]
            if surface:
                patterns[(item["id"], label)] = surface
    expression = re.compile("|".join(re.escape(x) for x in sorted(set(patterns.values()), key=len, reverse=True)))
    source_identities, source_sentences = {}, 0
    for shard in range(32):
        name = f"part_{shard:02d}"
        source = WIKI / "sentences_dedup" / (name + ".txt")
        sha, byte_count = hashlib.sha256(), 0
        paragraph, sentence_index, in_paragraph = -1, 0, False
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                sha.update(raw); byte_count += len(raw)
                surface = normalize_surface(raw.decode("utf-8"))
                if not surface:
                    in_paragraph = False
                    continue
                if not in_paragraph:
                    paragraph += 1; sentence_index = 0; in_paragraph = True
                source_sentences += 1
                if expression.search(surface):
                    for key, query in patterns.items():
                        if query in surface:
                            raw_candidates[(name, paragraph)].append({"key": key, "sentence_index": sentence_index,
                                "source_line": line_number, "source_surface": surface, "surface_equals_source_sentence": query == surface})
                sentence_index += 1
        actual = {"bytes": byte_count, "sha256": sha.hexdigest()}
        if actual != manifest["inputs"][str(source.relative_to(WIKI))]:
            raise ValueError("Source shard changed: " + name)
        source_identities[name] = actual
        print(json.dumps({"phase": "source_search", "shard": shard + 1, "sentences": source_sentences}), flush=True)
    if source_sentences != manifest["provenance"]["sentences"]:
        raise ValueError("Source sentence scope incomplete")
    surface_results = {key: {"count": 0, "examples": []} for key in patterns}
    with ids_path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
        for item in data["examples"]:
            for label in ("full", "generated"):
                tokens = item["generated_tokens"] if label == "generated" else item["prompt_tokens"] + item["generated_tokens"]
                tokens = strip_edge_spaces(tokens)
                token_ids = [lookup.get(token, 1) for token in tokens]
                item[label + "_token_match"] = find_id_sequence(buffer, ids.offset, records, token_ids)
        # Only source candidates need provenance and numeric-token restoration.
        # A source-text hit counts as TRAIN only if that source sentence's exact
        # original numeric token sequence is also present in the actual train IDs.
        for shard in range(32):
            name = f"part_{shard:02d}"
            wanted = {paragraph: values for (shard_name, paragraph), values in raw_candidates.items() if shard_name == name}
            if not wanted:
                continue
            numeric_path = WIKI / "tokens_numeric" / (name + ".txt")
            if identity(numeric_path) != manifest["inputs"][str(numeric_path.relative_to(WIKI))]:
                raise ValueError("Numeric source changed")
            with gzip.open(corpus / "provenance" / (name + ".jsonl.gz"), "rt", encoding="utf-8") as provenance:
                sentinel = object()
                for paragraph, (lines, record_raw) in enumerate(itertools.zip_longest(paragraphs(numeric_path), provenance, fillvalue=sentinel)):
                    if paragraph not in wanted:
                        continue
                    if lines is sentinel or record_raw is sentinel:
                        raise ValueError("Provenance alignment changed")
                    record = json.loads(record_raw)
                    if len(lines) != record["sentences"] or document_split(record["doc_id"]) != record["split"]:
                        raise ValueError("Provenance metadata mismatch")
                    for hit in wanted[paragraph]:
                        key = tuple(hit["key"])
                        if record["split"] != "train":
                            continue
                        tokens = strip_edge_spaces(lines[hit["sentence_index"]].split(" "))
                        if any(token not in lookup for token in tokens):
                            continue
                        source_match = find_id_sequence(buffer, ids.offset, records, [lookup[token] for token in tokens])
                        if not source_match["count"]:
                            continue
                        result = surface_results[key]
                        result["count"] += 1
                        if len(result["examples"]) < 3:
                            result["examples"].append({"shard": name, "source_line": hit["source_line"],
                                "source_sentence_index": hit["sentence_index"], "source_paragraph_index": paragraph,
                                "doc_id": record["doc_id"], "title": record["title"],
                                "source_sentence": hit["source_surface"],
                                "surface_equals_source_sentence": hit["surface_equals_source_sentence"],
                                "train_array_witness": source_match["examples"][0]})
    for item in data["examples"]:
        for label in ("full", "generated"):
            result = surface_results.get((item["id"], label), {"count": 0, "examples": []})
            item[label + "_surface_match"] = result
            item[label + "_found_in_train"] = bool(result["count"] or item[label + "_token_match"]["count"])
    if identity(Path(selected["checkpoint"]["path"])) != selected["checkpoint"]["identity"]:
        raise AssertionError("Final checkpoint changed during test")
    totals = {"samples": len(data["examples"]),
              "sentence_ending_reached": sum(x["sentence_ending_reached"] for x in data["examples"]),
              "full_found_in_train": sum(x["full_found_in_train"] for x in data["examples"]),
              "generated_only_found_in_train": sum(x["generated_found_in_train"] for x in data["examples"]),
              "prompt_oov_tokens": sum(len(x["prompt_oov_tokens"]) for x in data["examples"])}
    result = {**data, "kind": "ACTUAL_FINAL_MODEL_20_SAMPLES_WITH_EXHAUSTIVE_TRAIN_CHECK", "checked_at": now(),
              "membership_seconds": time.monotonic() - began,
              "membership_scope": {"split": "train", "paragraphs": len(records),
                  "sentences": manifest["statistics"]["train_kept_sentences"], "array_ids": int(ids.size),
                  "train_array_identities": bound, "dedup_source_shards_checked": 32,
                  "dedup_source_sentences_checked": source_sentences, "dedup_source_identities": source_identities,
                  "token_rule": "Exact aligned contiguous ID sequence within actual train paragraph content; edge WS ignored. No BOS/EOS or cross-paragraph matches.",
                  "surface_rule": "Whitespace-normalized full query occurs inside a source sentence; candidate must be train-split, in-vocabulary, and its original numeric token line must occur in actual train IDs.",
                  "limitations": "No exact match does not prove absence of memorization or semantic copying. Samples are not a held-out quality benchmark; no rejection sampling."},
              "totals": totals}
    write(".results.json", result)
    lines = ["NPLM 최종 4.074373에폭 — 실제 생성 20개와 학습 데이터 대조", "",
             f"검사 범위: train {len(records):,}문단 / {manifest['statistics']['train_kept_sentences']:,}문장 전체",
             "일치: 실제 train 토큰열 일치 또는 원문 표면형 일치(공백 정규화, 실제 train 포함 확인).",
             "입력+생성 전체와 생성 부분만을 따로 검색. 유사한 의미/암기 여부를 판정하는 검사는 아님.",
             "학습 추가 0회. 가중치/Adam/샘플러 및 체크포인트 해시 불변 확인.",
             f"문장 종결 도달: {totals['sentence_ending_reached']}/20; 전체 일치: {totals['full_found_in_train']}/20; 생성 부분 일치: {totals['generated_only_found_in_train']}/20", ""]
    for item in result["examples"]:
        lines.extend([f"[{item['id']:02d}] 입력: {item['prompt']}",
                      "전체 결과: " + item["full_surface"], "새로 생성: " + item["generated_surface"],
                      "학습 데이터(입력+생성): " + ("있음" if item["full_found_in_train"] else "일치 없음"),
                      "학습 데이터(생성 부분만): " + ("있음" if item["generated_found_in_train"] else "일치 없음"),
                      f"종료: {item['generation_terminated_by']} / 생성 토큰 {len(item['generated_ids'])}개",
                      "생성 토큰열: " + " ".join(item["generated_tokens"]), ""])
    with path_for(".results.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"phase": "complete", **totals}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "infer", "decode", "match"])
    arguments = parser.parse_args()
    globals()[arguments.phase]()


if __name__ == "__main__":
    main()
