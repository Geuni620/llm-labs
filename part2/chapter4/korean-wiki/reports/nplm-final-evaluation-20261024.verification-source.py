"""Evaluate the selected, stopped checkpoint without any optimizer updates.

The ``evaluate`` and ``oracle`` commands must run in separate processes: only
the former loads the C ABI, and only the latter imports Torch. Outputs are new
dated reports; the selected descriptor and every checkpoint remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

BASE = Path(__file__).resolve().parent
WIKI = BASE.parent
sys.path.insert(0, str(WIKI))
SELECTION = WIKI / "reports/resume-20261023-r2-selected-model.json"
PREFIX = WIKI / "reports/nplm-final-evaluation-20261024"
ARTIFACT = BASE / ".artifacts/final-evaluation-20261024.npz"
TOLERANCE = {"logits_atol": 2e-6, "logits_rtol": 2e-4,
             "mean_nll_absolute_error": 2e-6, "declared_before_execution": True}


def now():
    return datetime.now().astimezone().isoformat()


def output_path(suffix):
    return Path(str(PREFIX) + suffix)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(suffix, data):
    with output_path(suffix).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("Input changed while hashing: " + str(path))
    return {"bytes": before.st_size, "sha256": digest}


def metrics(totals):
    """Turn pooled count/NLL/correct sums into token-weighted metrics."""
    result = {}
    for prefix, offset in (("", 0), ("content_", 3)):
        count, nll, correct = map(float, totals[offset:offset + 3])
        if not (count > 0 and count.is_integer() and math.isfinite(nll) and nll >= 0
                and correct.is_integer() and 0 <= correct <= count):
            raise ValueError("Invalid evaluation totals")
        loss = nll / count
        result.update({prefix + "targets": int(count), prefix + "negative_log_likelihood_sum": nll,
                       prefix + "loss": loss, prefix + "perplexity": math.exp(loss),
                       prefix + "accuracy": correct / count})
    return result


def fixed_windows(count, seed, expected_hash):
    import numpy as np

    windows = np.random.default_rng(seed + 7000).choice(count, min(16384, count), replace=False)
    actual = hashlib.sha256(windows.astype("<i8").tobytes()).hexdigest()
    if actual != expected_hash:
        raise ValueError("Fixed validation target selection differs from the training run")
    return windows


def state_identity(engine):
    return {"arrays": {name: hashlib.sha256(engine.arrays[name].tobytes()).hexdigest()
                       for name in ("parameters", "moment1", "moment2")},
            "visited": hashlib.sha256(engine.visited.tobytes()).hexdigest(),
            "sampler": engine.sampler, "info": engine.info}


def evaluate():
    import numpy as np
    import checkpoint
    from engine import Engine, LIBRARY
    from npy_corpus import NpyArrayCorpus
    from run import corpus_identity
    from summarize import assert_not_running
    from thermal_guard import NightGuard

    if output_path(".raw.json").exists() or ARTIFACT.exists():
        raise FileExistsError("Preserve existing evaluation outputs")
    selection = read(SELECTION)
    selected_path = Path(selection["checkpoint"]["path"])
    run_dir = selected_path.parent
    assert_not_running(run_dir, WIKI / "reports/resume-20261023-r2-control.json")
    run = read(run_dir / "run.json")
    corpus = Path(selection["corpus"])
    if identity(selected_path) != selection["checkpoint"]["identity"]:
        raise ValueError("Selected final checkpoint changed")
    bound_data, manifest = corpus_identity(corpus)
    if bound_data != run["identity"]["data"] or identity(LIBRARY) != run["identity"]["library"]:
        raise ValueError("Bound corpus/library changed")
    for name, expected in run["identity"]["scripts"].items():
        if identity(BASE / name) != expected:
            raise ValueError("Frozen trainer changed: " + name)
    vocabulary = read(corpus / "vocabulary.json")["tokens"]
    space_id = vocabulary.index("<sp>/WS")
    validation = NpyArrayCorpus(corpus, "validation", 8, verify=True)
    test = NpyArrayCorpus(corpus, "test", 8, verify=True)
    train_records = np.load(corpus / "train.paragraphs.npy", mmap_mode="r", allow_pickle=False)
    overlap = {"train_validation": int(np.intersect1d(train_records["doc_key"], validation.records["doc_key"]).size),
               "train_test": int(np.intersect1d(train_records["doc_key"], test.records["doc_key"]).size),
               "validation_test": int(np.intersect1d(validation.records["doc_key"], test.records["doc_key"]).size)}
    if any(overlap.values()):
        raise ValueError("Document split overlap")
    windows = fixed_windows(validation.windows, run["config"]["seed"], run["evaluation_window_sha256"])
    started_at, started = now(), time.monotonic()
    events = []
    start = datetime.now().astimezone()
    guard = NightGuard(start + timedelta(minutes=15), start + timedelta(minutes=16), emit=events.append)
    engine = None
    try:
        with ExitStack() as stack:
            for relative in ("cpu-benchmark/.timing.lock", "reports/.nplm-benchmark.lock"):
                handle = stack.enter_context((WIKI / relative).open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard.start()
            if not guard.wait_ready(10):
                raise RuntimeError("Evaluation monitor not ready: " + str(guard.reason))
            engine = Engine(corpus, selection["config"])
            checkpoint.restore(selected_path, engine, run["identity"])
            before = state_identity(engine)
            if (before["info"]["consumed"] != selection["target_predictions"]
                    or before["info"]["updates"] != selection["updates"]):
                raise ValueError("Final checkpoint counters differ")
            rows = []
            for name, source, selected in (("fixed_validation", validation, windows),
                                           ("full_validation", validation, None), ("full_test", test, None)):
                count = len(selected) if selected is not None else source.windows
                totals = np.zeros(6, dtype=np.float64)
                phase_started, last_progress = time.monotonic(), 0
                for offset in range(0, count, engine.config["batch_size"]):
                    if guard.should_stop:
                        raise RuntimeError("Evaluation stopped: " + str(guard.reason))
                    indices = (selected[offset:offset + engine.config["batch_size"]] if selected is not None
                               else np.arange(offset, min(offset + engine.config["batch_size"], count), dtype=np.int64))
                    x, y, _ = source.batch(indices)
                    totals += engine.evaluate(x, y, space_id)
                    if name == "fixed_validation" and offset == 0:
                        inputs = {"x": x[:256], "y": y[:256], "logits": engine.logits(x[:256]),
                                  "stats": engine.evaluate(x[:256], y[:256], space_id)}
                        with ARTIFACT.open("xb") as stream:
                            np.savez(stream, **inputs)
                    if time.monotonic() - last_progress >= 20 or offset + len(y) == count:
                        print(json.dumps({"phase": name, "completed": offset + len(y), "total": count,
                                          "percent": 100 * (offset + len(y)) / count,
                                          "thermal": guard.metadata()["last_state"]}), flush=True)
                        last_progress = time.monotonic()
                if int(totals[0]) != count:
                    raise ValueError("Evaluation target coverage incomplete")
                rows.append({"name": name, "evaluated_at": now(), "evaluation_seconds": time.monotonic() - phase_started,
                             "split": "test" if name == "full_test" else "validation", "metrics": metrics(totals),
                             "coverage": "entire split" if selected is None else "original fixed validation targets"})
            after = state_identity(engine)
            if before != after:
                raise AssertionError("Evaluation changed trained state")
            engine.close(); engine = None
            guard.mark_stopped(); guard.close()
            if identity(selected_path) != selection["checkpoint"]["identity"]:
                raise AssertionError("Selected checkpoint file changed")
            write(".raw.json", {"kind": "READ_ONLY_FINAL_CHECKPOINT_EVALUATION_RAW", "status": "awaiting_independent_oracle",
                  "started_at": started_at, "finished_at": now(), "wall_seconds": time.monotonic() - started,
                  "selection": selection, "selection_descriptor_identity": identity(SELECTION),
                  "input_bindings": {**bound_data, "test.ids.npy": identity(corpus / "test.ids.npy"),
                                     "test.paragraphs.npy": identity(corpus / "test.paragraphs.npy")},
                  "corpus_statistics": manifest["statistics"], "config": run["config"],
                  "fixed_validation_window_sha256": run["evaluation_window_sha256"],
                  "document_key_overlap": overlap, "state_before": before, "state_after": after,
                  "state_unchanged": before == after, "training_updates_added": 0,
                  "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "guard": guard.metadata(), "guard_events": events, "rows": rows,
                  "numerical_fixture": {"path": str(ARTIFACT), **identity(ARTIFACT)},
                  "predeclared_oracle_tolerances": TOLERANCE,
                  "source_identity": identity(Path(__file__)),
                  "scope": "Selected final weights, same model/vocabulary/data split. Full validation and full test contain only retained in-vocabulary sentences. Accuracy is next-token top-1, not factual or sentence accuracy."})
    except BaseException as error:
        if engine is not None:
            engine.close()
        guard.mark_stopped(); guard.close()
        write(".failure.json", {"at": now(), "error": repr(error), "guard": guard.metadata()})
        raise


def oracle():
    import numpy as np
    import torch
    import torch.nn.functional as functional

    raw = read(output_path(".raw.json"))
    if raw["predeclared_oracle_tolerances"] != TOLERANCE:
        raise ValueError("Oracle tolerances changed")
    selected = raw["selection"]
    checkpoint_path = Path(selected["checkpoint"]["path"])
    if identity(checkpoint_path) != selected["checkpoint"]["identity"]:
        raise ValueError("Final checkpoint changed before independent verification")
    if identity(ARTIFACT) != {k: raw["numerical_fixture"][k] for k in ("bytes", "sha256")}:
        raise ValueError("Numerical fixture changed")
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        parameters = archive["parameters"]
    if hashlib.sha256(parameters.tobytes()).hexdigest() != raw["state_before"]["arrays"]["parameters"]:
        raise ValueError("Oracle loaded different weights")
    config = selected["config"]
    vocab, context, dim, hidden = [config[key] for key in ("vocab_size", "context_size", "embedding_dim", "hidden_dim")]
    offset, arrays = 0, []
    for shape in ((vocab, dim), (hidden, context * dim), (hidden,), (vocab, hidden), (vocab,)):
        size = math.prod(shape)
        arrays.append(torch.from_numpy(parameters[offset:offset + size].reshape(shape)))
        offset += size
    if offset != len(parameters):
        raise ValueError("Parameter layout does not cover actual weights")
    with np.load(ARTIFACT, allow_pickle=False) as fixture:
        x = torch.from_numpy(fixture["x"].astype(np.int64))
        y = torch.from_numpy(fixture["y"].astype(np.int64))
        actual_logits, actual_stats = fixture["logits"], fixture["stats"]
    with torch.no_grad():
        embedding, hidden_weight, hidden_bias, output_weight, output_bias = arrays
        encoded = embedding[x].reshape(len(x), context * dim)
        # Match the frozen C graph: GEMM(beta=0), then a separate FP32 bias add.
        # F.linear(..., bias) can fuse the addition into GEMM and round differently.
        hidden_values = torch.tanh(functional.linear(encoded, hidden_weight) + hidden_bias)
        logits = functional.linear(hidden_values, output_weight) + output_bias
        # The C evaluator accumulates/normalizes NLL in double precision. Keep the
        # model forward FP32, but use FP64 for the independent reference loss.
        losses = functional.cross_entropy(logits.double(), y, reduction="none").numpy()
        correct = (logits.argmax(dim=-1) == y).numpy()
    reference = logits.numpy()
    error = np.abs(actual_logits.astype(np.float64) - reference.astype(np.float64))
    allowed = TOLERANCE["logits_atol"] + TOLERANCE["logits_rtol"] * np.abs(reference.astype(np.float64))
    vocabulary = read(Path(selected["vocabulary"]))["tokens"]
    mask = (y.numpy() != vocabulary.index("<sp>/WS")) & (y.numpy() != 3)
    expected_stats = np.asarray([len(y), losses.sum(), correct.sum(), mask.sum(), losses[mask].sum(), correct[mask].sum()], dtype=np.float64)
    nll_error = [abs(float(actual_stats[i] - expected_stats[i])) / expected_stats[i - 1] for i in (1, 4)]
    passed = bool(np.isfinite(error).all() and np.all(error <= allowed)
                  and max(nll_error) <= TOLERANCE["mean_nll_absolute_error"]
                  and np.array_equal(actual_stats[[0, 2, 3, 5]], expected_stats[[0, 2, 3, 5]]))
    result = {"kind": "ISOLATED_TORCH_CPU_FINAL_WEIGHTS_ORACLE", "at": now(), "passed": passed,
              "targets": len(y), "compared_logits": int(error.size), "max_absolute_logit_error": float(error.max()),
              "logits_outside_tolerance": int((error > allowed).sum()), "mean_nll_errors": nll_error,
              "c_totals": actual_stats.tolist(), "torch_totals": expected_stats.tolist(),
              "tolerances": TOLERANCE, "checkpoint_identity_after": identity(checkpoint_path),
              "raw_report_identity": identity(output_path(".raw.json")), "torch_version": torch.__version__}
    result["reference_policy"] = "FP32 separate GEMM/bias operations matching C; independent FP64 NLL. Original tolerances unchanged."
    result["source_identity"] = identity(Path(__file__))
    initial_path = output_path(".oracle.json")
    result["initial_reference_attempt"] = {"identity": identity(initial_path), "passed": read(initial_path)["passed"]} if initial_path.exists() else None
    write(".oracle-aligned.json", result)
    print(json.dumps({"passed": passed, "targets": len(y), "max_logit_error": result["max_absolute_logit_error"],
                      "mean_nll_errors": nll_error}), flush=True)
    if not passed:
        raise AssertionError("Independent final-checkpoint numerical comparison failed")


def comparison_rows(baselines, final_epoch, final_metrics):
    if any(row["validation"]["targets"] != final_metrics["targets"] for row in baselines):
        raise ValueError("Cannot compare different fixed validation target counts")
    rows = [{"epochs": row["epochs"], "validation": row["validation"]} for row in baselines]
    return rows + [{"epochs": final_epoch, "validation": final_metrics}]


def report():
    raw, oracle_result = read(output_path(".raw.json")), read(output_path(".oracle-aligned.json"))
    if not oracle_result["passed"] or oracle_result["raw_report_identity"] != identity(output_path(".raw.json")):
        raise ValueError("Independent evaluation gate missing/failed")
    if identity(Path(raw["selection"]["checkpoint"]["path"])) != raw["selection"]["checkpoint"]["identity"]:
        raise ValueError("Final checkpoint changed")
    baselines = read(WIKI / "reports/resume-20261023-r2-final-234-comparison.json")["rows"]
    comparison = comparison_rows(baselines, raw["selection"]["epochs"], raw["rows"][0]["metrics"])
    snapshot = output_path(".execution-source.py")
    if identity(snapshot) != raw["source_identity"]:
        raise ValueError("Original evaluation source snapshot differs")
    result = {**raw, "kind": "ACTUAL_FINAL_CHECKPOINT_EVALUATION", "status": "complete",
              "independent_verification": oracle_result, "fixed_validation_comparison": comparison,
              "execution_source_snapshot": {"path": str(snapshot), **identity(snapshot)},
              "postprocessor_source_identity": identity(Path(__file__)),
              "postprocessor_note": "After evaluation, the report adapter was aligned with the historical rows' epochs field. Evaluation math, oracle tolerances and frozen trainer sources were unchanged; exact execution source is preserved.",
              "historical_selection_descriptor_preserved": identity(SELECTION) == raw["selection_descriptor_identity"],
              "test_policy": "First recorded full test evaluation after the final checkpoint was selected; do not tune this checkpoint against the test result.",
              "interpretation": "Loss/PPL evaluate next-token prediction. Top-1 accuracy counts predicted tokens, not correct facts. Generation and memorization results remain separate evidence."}
    write(".json", result)
    lines = ["최종 NPLM 4.074373에폭 — 실제 읽기 전용 성능 검증", "",
             "학습 추가 0회 / 기존 latest.npz 사용 / 전체 가중치·Adam·샘플러·방문 벡터·파일 해시 불변", "",
             "평가 범위\t타깃 수\tLoss\tPPL\t정확도\t공백·EOS 제외 타깃\t제외 Loss\t제외 PPL\t제외 정확도"]
    for row in raw["rows"]:
        m = row["metrics"]
        lines.append(f"{row['name']}\t{m['targets']}\t{m['loss']:.9f}\t{m['perplexity']:.6f}\t{m['accuracy']:.6%}\t{m['content_targets']}\t{m['content_loss']:.9f}\t{m['content_perplexity']:.6f}\t{m['content_accuracy']:.6%}")
    lines.extend(["", "기존 고정 검증 16,384 타깃과 같은 조건의 에폭별 비교", "epoch\tLoss\tPPL\t공백·EOS 제외 Loss\t제외 PPL"])
    for row in comparison:
        m = row["validation"]
        lines.append(f"{row['epochs']:.6f}\t{m['loss']:.9f}\t{m['perplexity']:.6f}\t{m['content_loss']:.9f}\t{m['content_perplexity']:.6f}")
    lines.extend(["", raw["scope"], result["interpretation"],
                  "수치 검증: 실제 최종 가중치의 256 타깃을 별도 Torch CPU로 계산해 전체 logits/NLL/정답 수 대조.",
                  "생성 20개의 사실 오류와 문맥 혼선을 수치 검증 통과로 해결했다고 해석하지 않는다."])
    with output_path(".txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(output_path(".json")), "oracle_passed": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("evaluate", "oracle", "report"))
    globals()[parser.parse_args().phase]()


if __name__ == "__main__":
    main()
