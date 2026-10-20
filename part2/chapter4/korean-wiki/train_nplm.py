"""Actual, bounded Wiki NPLM training with comparable timed checkpoints.

This is separate from benchmark_nplm.py: it updates weights using prepared Wiki
paragraphs, evaluates a fixed validation sample, and saves generated examples.
The NPLM has a short fixed context; it is not a long-context Transformer LLM.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_nplm import BenchmarkConfig, assert_pipeline_idle, build_model, parameter_count
from benchmark_nplm import _peak_rss_bytes, _mps_snapshot
from training_corpus import ArrayCorpus, WindowSampler, atomic_json, identity


ROOT = Path(__file__).resolve().parent
PROMPTS = [
    ("서울은", ["서울/NNP", "은/JX", "<sp>/WS"]),
    ("대한민국의", ["대한민국/NNP", "의/JKG", "<sp>/WS"]),
    ("역사는", ["역사/NNG", "는/JX", "<sp>/WS"]),
    ("그는", ["그/NP", "는/JX", "<sp>/WS"]),
    ("2025년", ["2/SN", "0/SN", "2/SN", "5/SN", "년/NNB", "<sp>/WS"]),
    ("2037년", ["2/SN", "0/SN", "3/SN", "7/SN", "년/NNB", "<sp>/WS"]),
]


@dataclass(frozen=True)
class TrainingConfig:
    fraction: float = 1.0
    device: str = "cpu"
    cpu_threads: int = 4
    context_size: int = 8
    embedding_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 42
    budget_seconds: float = 28800
    checkpoints_seconds: tuple = (3600, 10800, 28800)
    evaluation_seconds: float = 600
    evaluation_windows: int = 16384
    generation_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    max_steps: int | None = None

    def validate(self):
        if self.fraction not in (0.1, 0.25, 1.0) or self.device not in ("cpu", "mps"):
            raise ValueError("Require a documented 10%, 25% or 100% corpus and CPU/MPS")
        for value in (self.cpu_threads, self.context_size, self.embedding_dim, self.hidden_dim,
                      self.batch_size, self.evaluation_windows, self.generation_tokens, self.top_k):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Model/evaluation dimensions must be positive integers")
        for value in (self.budget_seconds, self.evaluation_seconds, self.learning_rate, self.temperature, self.gradient_clip_norm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Time/rate/temperature settings must be positive and finite")
        if self.budget_seconds > 8 * 3600 or self.generation_tokens > 256:
            raise ValueError("Bounded runner permits at most 8 hours per run and 256 generated tokens")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 or x > self.budget_seconds for x in self.checkpoints_seconds):
            raise ValueError("Checkpoint times must lie inside the run budget")
        if tuple(sorted(set(self.checkpoints_seconds))) != self.checkpoints_seconds:
            raise ValueError("Checkpoint times must be unique and increasing")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def evaluate(model, corpus, windows, device, batch_size, space_id):
    model.eval()
    totals = {"targets": 0, "negative_log_likelihood_sum": 0.0, "correct": 0,
              "content_targets": 0, "content_negative_log_likelihood_sum": 0.0, "content_correct": 0}
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            contexts, target_ids, _ = corpus.batch(windows[start:start + batch_size])
            x, y = torch.from_numpy(contexts).to(device), torch.from_numpy(target_ids).to(device)
            logits = model(x)
            losses = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
            correct = (logits.argmax(-1) == y).cpu().numpy()
            # Report all targets and targets excluding custom WS and EOS separately.
            content = (target_ids != space_id) & (target_ids != 3)
            totals["targets"] += len(target_ids)
            totals["negative_log_likelihood_sum"] += float(losses.astype(np.float64).sum())
            totals["correct"] += int(correct.sum())
            totals["content_targets"] += int(content.sum())
            totals["content_negative_log_likelihood_sum"] += float(losses[content].astype(np.float64).sum())
            totals["content_correct"] += int(correct[content].sum())
    for prefix in ("", "content_"):
        count = totals[prefix + "targets"]
        mean = totals[prefix + "negative_log_likelihood_sum"] / count if count else None
        totals[prefix + "loss"] = mean
        totals[prefix + "perplexity"] = math.exp(mean) if mean is not None and mean < 700 else None
        totals[prefix + "accuracy"] = totals[prefix + "correct"] / count if count else None
    model.train()
    return totals


def generate_examples(model, tokens, config, kiwi):
    lookup = {token: i for i, token in enumerate(tokens)}
    model.eval()
    results = []
    for index, (label, prompt) in enumerate(PROMPTS):
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1000 + index)
        generated, context = [], [2] * config.context_size + [lookup.get(token, 1) for token in prompt]
        with torch.no_grad():
            for _ in range(config.generation_tokens):
                logits = model(torch.tensor([context[-config.context_size:]], device=config.device))[0].float().cpu()
                logits[:3] = -float("inf")
                values, indices = torch.topk(logits / config.temperature, min(config.top_k, len(tokens) - 3))
                sampled = int(indices[torch.multinomial(torch.softmax(values, dim=0), 1, generator=generator)])
                if sampled == 3:
                    break
                generated.append(tokens[sampled])
                context.append(sampled)
        try:
            # Same explicit whitespace handling as the lecture restoration code.
            from step5_tokenize import detokenize
            restored = detokenize(" ".join(prompt + generated), kiwi)
            restoration_error = None
        except Exception as error:
            restored, restoration_error = None, f"{type(error).__name__}: {error}"
        trigrams = list(zip(generated, generated[1:], generated[2:]))
        repetition = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        results.append({"prompt": label, "prompt_tokens": prompt,
                        "prompt_oov_tokens": [token for token in prompt if token not in lookup],
                        "generated_tokens": generated, "restored_text": restored,
                        "restoration_error": restoration_error, "repeated_trigram_fraction": repetition,
                        "seed": config.seed + 1000 + index})
    model.train()
    return results


def save_checkpoint(path, model, optimizer, sampler, config, state, visited, corpus_identity):
    synchronize(config.device)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "config": asdict(config), "state": state, "sampler_consumed": sampler.consumed,
               "visited_paragraphs": torch.from_numpy(visited.copy()), "corpus_identity": corpus_identity,
               "torch_rng": torch.get_rng_state(),
               "mps_rng": torch.mps.get_rng_state() if config.device == "mps" else None}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(corpus_dir, output, config, resume=False):
    config.validate()
    assert_pipeline_idle(ROOT)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if output.exists() and not resume:
        raise FileExistsError(f"Preserving existing training run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/.nplm-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(corpus_dir, output, config, resume)


def _run_locked(corpus_dir, output, config, resume):
    from kiwipiepy import Kiwi
    torch.set_num_threads(config.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config.seed)
    train = ArrayCorpus(corpus_dir, "train", config.context_size, config.fraction)
    validation = ArrayCorpus(corpus_dir, "validation", config.context_size)
    corpus_identity = identity(corpus_dir / "manifest.json")
    vocabulary = json.loads((corpus_dir / "vocabulary.json").read_text())["tokens"]
    if identity(corpus_dir / "vocabulary.json") != train.manifest["outputs"]["vocabulary.json"]:
        raise ValueError("Vocabulary changed")
    model_config = BenchmarkConfig(vocab_size=len(vocabulary), context_size=config.context_size,
                                   embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
                                   batch_size=config.batch_size, device="cpu")
    model_config.validate()
    model = build_model(model_config)
    initialization = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initialization.update(name.encode())
        initialization.update(tensor.detach().numpy().tobytes())
    initialization_sha256 = initialization.hexdigest()
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  foreach=False, fused=False)
    visited = np.zeros(len(train.records), dtype=np.bool_)
    sampler = WindowSampler(train.windows, config.seed)
    state = {"status": "running", "steps": 0, "elapsed_seconds": 0.0, "update_seconds": 0.0,
             "evaluation_seconds": 0.0, "checkpoint_seconds": 0.0, "completed_checkpoints": [],
             "last_periodic_evaluation_seconds": 0.0}
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        expected = asdict(config)
        if saved["config"] != expected or saved["corpus_identity"] != corpus_identity:
            raise ValueError("Resume configuration/corpus differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        visited = saved["visited_paragraphs"].numpy().copy()
        sampler.consumed = saved["sampler_consumed"]
        torch.set_rng_state(saved["torch_rng"])
        if config.device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        if state["status"] == "complete":
            raise ValueError("This training run is already complete")
        state["status"] = "running"
    rng = np.random.default_rng(config.seed + 7000)
    validation_windows = rng.choice(validation.windows, min(config.evaluation_windows, validation.windows), replace=False)
    evaluation_identity = hashlib.sha256(validation_windows.astype("<i8").tobytes()).hexdigest()
    kiwi = Kiwi(num_workers=1, model_type="cong-global")
    metadata = {"config": asdict(config), "corpus": str(corpus_dir), "corpus_identity": corpus_identity,
                "vocabulary_identity": identity(corpus_dir / "vocabulary.json"),
                "initialization_sha256": initialization_sha256,
                "evaluation_window_sha256": evaluation_identity, "evaluation_windows": len(validation_windows),
                "training_target_windows": train.windows, "training_paragraphs": len(train.selected),
                "parameters": parameter_count(len(vocabulary), config.context_size, config.embedding_dim, config.hidden_dim),
                "runtime": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
                "scripts": {name: identity(ROOT / name) for name in ("train_nplm.py", "training_corpus.py", "benchmark_nplm.py")},
                "kind": "ACTUAL_WIKI_TRAINING", "test_split_used": False,
                "evaluation_scope": "fixed seeded sample of validation target windows; excludes OOV sentences using the common vocabulary",
                "limitations": "One seed and a fixed-context NPLM; generation examples and perplexity do not establish long-paragraph LLM quality."}
    if resume:
        previous = json.loads((output / "run.json").read_text())
        for field in ("scripts", "evaluation_window_sha256", "vocabulary_identity", "initialization_sha256"):
            if metadata[field] != previous[field]:
                raise ValueError(f"Resume identity mismatch: {field}")
    else:
        atomic_json(output / "run.json", metadata)
    recent_losses = deque(maxlen=100)
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    accumulated, started = state["elapsed_seconds"], time.monotonic()
    def elapsed():
        return accumulated + time.monotonic() - started
    def snapshot(nominal=None, label="periodic"):
        triggered = elapsed()
        begun = time.monotonic()
        result = {"kind": "ACTUAL_WIKI_CHECKPOINT", "label": label, "nominal_seconds": nominal,
                  "actual_elapsed_seconds": triggered, "steps": state["steps"],
                  "target_prediction_events": sampler.consumed,
                  "unique_target_windows_visited": min(sampler.consumed, train.windows),
                  "epochs_equivalent": sampler.consumed / train.windows,
                  "unique_training_paragraphs_visited": int(visited.sum()),
                  "peak_rss_bytes": _peak_rss_bytes(), "mps_memory": _mps_snapshot(torch, config.device),
                  "recent_training_loss": float(np.mean(recent_losses)) if recent_losses else None,
                  "validation": evaluate(model, validation, validation_windows, config.device, config.batch_size,
                                         vocabulary.index("<sp>/WS") if "<sp>/WS" in vocabulary else -1),
                  "generation": generate_examples(model, vocabulary, config, kiwi),
                  "evaluation_window_sha256": evaluation_identity}
        state["evaluation_seconds"] += time.monotonic() - begun
        if nominal is not None:
            state["completed_checkpoints"].append(nominal)
        state["elapsed_seconds"] = elapsed()
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        if nominal is not None or label in {"initial", "final", "stopped"}:
            save_started = time.monotonic()
            save_checkpoint(output / "latest.pt", model, optimizer, sampler, config, state, visited, corpus_identity)
            if nominal is not None or label == "initial":
                # Preserve the exact evaluated weights without another optimizer copy.
                path = output / f"model-{0 if nominal is None else nominal:g}s.pt"
                temporary = path.with_name(path.name + ".tmp")
                torch.save({"model": model.state_dict(), "config": asdict(config), "metrics": result,
                            "vocabulary_identity": metadata["vocabulary_identity"]}, temporary)
                temporary.replace(path)
            state["checkpoint_seconds"] += time.monotonic() - save_started
        atomic_json(output / "progress.json", {**state, "elapsed_seconds": elapsed(),
                    "target_prediction_events": sampler.consumed, "epochs_equivalent": sampler.consumed / train.windows,
                    "unique_training_paragraphs_visited": int(visited.sum()), "last_validation": result["validation"]})
        print(json.dumps({key: result[key] for key in ("label", "nominal_seconds", "actual_elapsed_seconds", "steps", "validation")}), flush=True)
        return result
    if not resume:
        snapshot(label="initial")
        # Initial evaluation is setup: the requested training budget starts here.
        accumulated, started = 0.0, time.monotonic()
        state["evaluation_seconds"] = 0.0
    last_progress = 0.0
    model.train()
    try:
        while elapsed() < config.budget_seconds:
            if stop["requested"] or (output / "STOP").exists():
                state["status"] = "stopped"
                break
            if config.max_steps is not None and state["steps"] >= config.max_steps:
                break
            begun = time.monotonic()
            contexts, target_ids, paragraph_ids = train.batch(sampler.take(config.batch_size))
            x, y = torch.from_numpy(contexts).to(config.device), torch.from_numpy(target_ids).to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            synchronize(config.device)
            recent_losses.append(float(loss.detach().cpu()))
            state["update_seconds"] += time.monotonic() - begun
            state["steps"] += 1
            visited[paragraph_ids] = True
            due = [seconds for seconds in config.checkpoints_seconds
                   if seconds <= elapsed() and seconds not in state["completed_checkpoints"]]
            if due:
                for nominal in due:
                    snapshot(nominal=nominal, label="timed")
            elif elapsed() - state["last_periodic_evaluation_seconds"] >= config.evaluation_seconds:
                state["last_periodic_evaluation_seconds"] = elapsed()
                snapshot()
            if elapsed() - last_progress >= 30:
                atomic_json(output / "progress.json", {**state, "elapsed_seconds": elapsed(),
                            "target_prediction_events": sampler.consumed, "epochs_equivalent": sampler.consumed / train.windows,
                            "recent_training_loss": float(np.mean(recent_losses))})
                last_progress = elapsed()
        if state["status"] == "running":
            state["status"] = "complete" if config.max_steps is None else "step-limited-pilot-complete"
        # A final batch can cross the budget boundary; persist every crossed nominal time.
        for nominal in config.checkpoints_seconds:
            if nominal <= elapsed() and nominal not in state["completed_checkpoints"]:
                snapshot(nominal=nominal, label="timed")
        final = snapshot(label="stopped" if state["status"] == "stopped" else "final")
        atomic_json(output / "summary.json", {"status": state["status"], "kind": "ACTUAL_WIKI_TRAINING",
                    "run": metadata, "state": state, "final": final})
        return final
    except BaseException:
        state["status"] = "failed"
        state["elapsed_seconds"] = elapsed()
        atomic_json(output / "progress.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, choices=(0.1, 0.25, 1.0), default=1.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--checkpoint-hours", type=float, nargs="+", default=[1, 3, 8])
    parser.add_argument("--evaluation-windows", type=int, default=16384)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(fraction=args.fraction, device=args.device, budget_seconds=args.hours * 3600,
                            checkpoints_seconds=tuple(x * 3600 for x in args.checkpoint_hours),
                            evaluation_windows=args.evaluation_windows, max_steps=args.max_steps)
    run(args.corpus.resolve(), args.output.resolve(), config, args.resume)


if __name__ == "__main__":
    main()
