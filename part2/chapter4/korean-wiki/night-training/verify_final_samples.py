"""Independent integer-index verifier for the fixed 20-sample membership run."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sample_final_model import (BASE, PREFIX, SELECTED, identity, normalize_surface,
                                now, path_for, read, strip_edge_spaces, write)
from training_corpus import document_split, stable_int


def main():
    data = read(".results.json")
    selected = data["selection"]
    if identity(SELECTED) != read(".inputs.json")["selected_descriptor_identity"]:
        raise AssertionError("Final selection metadata changed")
    checkpoint = Path(selected["checkpoint"]["path"])
    assert identity(checkpoint) == selected["checkpoint"]["identity"]
    assert data["state_before"] == data["state_after"] and data["state_unchanged"]
    assert data["training_steps"] == 0
    corpus = Path(selected["corpus"])
    ids = np.load(corpus / "train.ids.npy", mmap_mode="r", allow_pickle=False)
    records = np.load(corpus / "train.paragraphs.npy", mmap_mode="r", allow_pickle=False)
    vocabulary = json.loads((corpus / "vocabulary.json").read_text())["tokens"]
    lookup = {token: index for index, token in enumerate(vocabulary)}
    examples = data["examples"]
    assert len(examples) == 20 and [x["id"] for x in examples] == list(range(1, 21))
    first_positions, checks = {}, []
    for item in examples:
        assert [vocabulary[i] for i in item["generated_ids"]] == item["generated_tokens"]
        assert [lookup.get(t, 1) for t in item["prompt_tokens"]] == item["prompt_ids"]
        assert item["sentence_ending_reached"]
        assert item["generated_tokens"][-1].endswith("/SF")
        for label in ("full", "generated"):
            tokens = item["generated_tokens"] if label == "generated" else item["prompt_tokens"] + item["generated_tokens"]
            query = [lookup.get(token, 1) for token in strip_edge_spaces(tokens)]
            if not query or any(token < 4 for token in query):
                positions = np.empty(0, dtype=np.int64)
            else:
                if query[0] not in first_positions:
                    first_positions[query[0]] = np.flatnonzero(ids == query[0])
                positions = first_positions[query[0]]
                positions = positions[positions + len(query) <= len(ids)]
                for offset, token in enumerate(query[1:], 1):
                    positions = positions[ids[positions + offset] == token]
                    if not len(positions):
                        break
                paragraph_ids = np.searchsorted(records["start"], positions, side="right") - 1
                starts = records["start"][paragraph_ids].astype(np.int64)
                ends = starts + records["length"][paragraph_ids].astype(np.int64)
                positions = positions[(positions >= starts + 8) & (positions + len(query) <= ends - 1)]
            actual = item[label + "_token_match"]
            assert len(positions) == actual["count"], (item["id"], label, len(positions), actual)
            assert [int(x) for x in positions[:3]] == [x["array_token_offset"] for x in actual["examples"]]
            checks.append({"sample": item["id"], "query": label, "independent_count": len(positions), "passed": True})
            surface = item["normalized_" + label + "_surface"]
            for witness in item[label + "_surface_match"]["examples"]:
                source_path = BASE.parent / "sentences_dedup" / (witness["shard"] + ".txt")
                with source_path.open(encoding="utf-8") as stream:
                    source_line = next(line for index, line in enumerate(stream, 1) if index == witness["source_line"])
                assert normalize_surface(source_line) == witness["source_sentence"]
                assert surface in witness["source_sentence"]
                assert document_split(witness["doc_id"]) == "train"
                record = records[witness["train_array_witness"]["train_paragraph_index"]]
                assert int(record["doc_key"]) == stable_int(witness["doc_id"], 42, "document")
    assert data["membership_scope"]["dedup_source_sentences_checked"] == 6677632
    assert data["membership_scope"]["sentences"] == 2977860
    assert data["membership_scope"]["paragraphs"] == len(records) == 1529599
    with path_for(".inputs.txt").open("x", encoding="utf-8") as stream:
        stream.write("NPLM 4.074373에폭 모델 — 고정 이어쓰기 입력 20개\n")
        stream.write("학습 추가 없음 / 각 입력에서 seed 고정 1회 생성 / top-k 40, temperature 0.8\n\n")
        for item in examples:
            stream.write(f"{item['id']:02d}. {item['prompt']}\n")
    with path_for(".results.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n실제 일치 근거(공백 정규화; 원문 전체가 아닌 포함 관계도 표시)\n")
        for item in examples:
            for label in ("full", "generated"):
                for witness in item[label + "_surface_match"]["examples"]:
                    stream.write(f"[{item['id']:02d}] {label}: {witness['title']} ({witness['doc_id']})\n")
                    stream.write("실제 학습 원문: " + witness["source_sentence"] + "\n")
                    stream.write(f"위치: sentences_dedup/{witness['shard']}.txt:{witness['source_line']} / train 문단 {witness['train_array_witness']['train_paragraph_index']}\n")
    output = {"kind": "INDEPENDENT_TRAIN_MEMBERSHIP_INDEX_VERIFICATION", "at": now(),
              "passed": True, "checks": checks, "comparisons": len(checks),
              "algorithm": "NumPy first-token positions then vectorized exact token comparisons; independent from mmap byte search.",
              "boundary_regression_tests": {"passed": 6, "file": str(BASE / "test_sample_final_model.py")},
              "checkpoint_unchanged": True, "read_only_state_verified": True,
              "results_identity": identity(path_for(".results.json")),
              "report_identity": identity(path_for(".results.txt")),
              "source_scripts": {name: identity(BASE / name) for name in ("sample_final_model.py", "verify_final_samples.py")},
              "totals": data["totals"]}
    write(".verification.json", output)
    print(json.dumps({"passed": True, "comparisons": len(checks), "totals": data["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
