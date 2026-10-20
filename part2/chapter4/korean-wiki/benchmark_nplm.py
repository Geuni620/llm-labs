"""Bounded synthetic NPLM microbenchmark; this does not train the Wiki corpus.

Run only after preprocessing/counting/coverage jobs finish. Each configuration
runs in a fresh child process, with float32, AdamW and a full-vocabulary output.
CPU and MPS use identical shapes/seeds for a given vocabulary. MPS is explicitly
synchronized around timing. Synthetic throughput extrapolations are not actual
3-hour/8-hour training results, nor evidence of language-model quality.

--smoke-test permits a tiny correctness check while preprocessing is busy; it
reports neither timing nor budget estimates. --plan-only writes the separate
controlled-experiment design without starting model training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_VOCABS = (16384, 32768, 49152, 65000, 65536)
BUSY_REPORTS = (
    "step5-progress.json", "step5-state.json", "token-frequency-state.json",
    "step6-numeric-progress.json", "step6-numeric-state.json",
    "vocabulary-coverage-progress.json", "vocabulary-coverage-state.json",
)
PIPELINE_PROGRAMS = (
    "step3_extract_paragraphs.py", "step4_dedup_sentences.py", "step5_tokenize.py",
    "step6_count_tokens.py", "step6_split_numbers.py", "analyze_vocabulary_coverage.py",
    "build_vocabulary_coverage_plot.py",
)
OFFICIAL_REFERENCES = {
    "mps": "https://docs.pytorch.org/docs/2.14/notes/mps.html",
    "synchronization": "https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html",
    "driver_memory": "https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html",
}
EXPERIMENT_DESIGN = """3시간·8시간 학습 결과를 비교하기 위한 실험 설계

이 파일은 실험 계획입니다. 마이크로벤치마크나 토큰 보존율은 학습 품질 결과가 아닙니다.

1. 시간 비교: 같은 8시간 학습 경로에서 3시간과 8시간 체크포인트를 저장합니다.
   모델·사전·데이터 순서·초기 가중치 seed·옵티마이저·배치를 고정합니다.
   두 독립 실행 대신 같은 경로를 사용하면 3시간 이후의 추가 학습 효과를 비교할 수 있습니다.
   시작 직전 모델(0시간)과 짧은 시험 체크포인트도 보존합니다.
   학습 시작 후 경과한 벽시계 시간을 기준으로 저장하고, 실제 업데이트 시간,
   평가/저장 시간, 업데이트 횟수, 예측한 target 수, 고유 학습 데이터 수를 각각 기록합니다.

2. 누수 방지: 원본 문서 ID 기준으로 train/validation/test를 먼저 고정합니다.
   한 문서의 문단은 같은 split에 배치하고, exact 중복이 split을 넘지 않도록 확인합니다.
   사전과 빈도는 train split에서만 만듭니다. 현재 전체 코퍼스 빈도는 탐색 자료입니다.
   같은 held-out 원문·토큰화·숫자 분리 정책·프롬프트 목록과 평가 seed를 모든 실험에 사용합니다.
   사전별 OOV 때문에 평가에서 빠진 문장/토큰 비율을 반드시 함께 기록합니다.
   서로 다른 사전의 perplexity만 단순 비교하면 평가 대상이 달라질 수 있습니다.

3. 사전 크기 비교: 16,384 / 32,768 / 49,152 / 65,000 / 65,536을 후보로 둡니다.
   사전 크기 V에 따라 embedding과 full-softmax 비용이 바뀝니다.
   이 축에서는 모델 나머지 설정과 원문 train pool을 고정하고, OOV 정책을 명시합니다.
   공통으로 표현 가능한 동일 held-out 영역의 loss와 전체 영역 커버리지를 따로 보고합니다.
   각자 유리한 평가 문장만 남긴 loss를 전체 품질 우위로 해석하지 않습니다.

4. 데이터 양 비교: 사전을 한 개로 고정한 뒤 train 문서를 seed로 섞어
   10% / 25% / 100%의 중첩 부분집합을 만듭니다. 빈도 높은 문단만 고르지 않습니다.
   같은 업데이트 수 비교와 같은 시간 예산 비교를 나누면 데이터 효과와 계산량 효과를 구분할 수 있습니다.
   시간과 데이터 양을 동시에 바꾼 한 쌍으로 어느 요인이 좋아졌다고 단정하지 않습니다.

5. 관측 결과: 같은 프롬프트/생성 seed/temperature에서 생성 문장, 반복률,
   held-out loss와 OOV율, 숫자·날짜 형식 보존, 학습 시간·메모리를 나란히 기록합니다.
   가능하면 seed를 둘 이상 반복해 우연한 초기화 효과도 확인합니다.
   context=8 NPLM은 짧은 다음 토큰 예측 모델이므로 장문 문단 품질을 보장하지 않습니다.

6. 3시간·8시간 예산 환산: 이 도구는 합성 미니배치의 target 예측/초를 측정합니다.
   예산 수치는 측정 속도 × 시간 × utilization margin의 외삽일 뿐입니다.
   고유 토큰 수나 최적 문단 수가 아닙니다. 문단별 실제 학습 창 수와 epoch 수로 다시 환산해야 합니다.
   실제 데이터 로딩, 평가, 체크포인트 저장, 장시간 발열과 앱 사용은 짧은 타이밍에 포함되지 않습니다.
   따라서 최종 선택은 보존율·실측 속도·검증 품질의 절충으로 하며 단일 최적값을 미리 확정하지 않습니다.
"""


@dataclass(frozen=True)
class BenchmarkConfig:
    vocab_size: int = 16384
    context_size: int = 8
    embedding_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 128
    device: str = "cpu"
    cpu_threads: int = 4
    interop_threads: int = 1
    seed: int = 42
    learning_rate: float = 0.001
    warmup_steps: int = 3
    repeats: int = 5
    steps_per_repeat: int = 5
    max_seconds: float = 120
    utilization_margin: float = 0.70

    def validate(self) -> None:
        for key in ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size",
                    "cpu_threads", "interop_threads", "warmup_steps", "repeats", "steps_per_repeat"):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.vocab_size < 2 or self.device not in {"cpu", "mps"}:
            raise ValueError("vocab_size >= 2 and device cpu/mps are required")
        if self.warmup_steps > 20 or self.repeats > 20 or self.steps_per_repeat > 100:
            raise ValueError("Microbenchmark limits: warmup <= 20, repeats <= 20, steps/repeat <= 100")
        if not 0 < self.utilization_margin <= 1 or not 0 < self.max_seconds <= 300:
            raise ValueError("Require 0 < utilization_margin <= 1 and 0 < max_seconds <= 300")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if parameter_count(self.vocab_size, self.context_size, self.embedding_dim, self.hidden_dim) * 16 > 2 * 1024 ** 3:
            raise ValueError("This bounded microbenchmark limits parameter/gradient/Adam lower bound to 2 GiB")
        if max(self.batch_size * self.vocab_size * 4,
               self.batch_size * self.context_size * self.embedding_dim * 4) > 256 * 1024 ** 2:
            raise ValueError("This bounded microbenchmark limits each logits/embedding activation tensor to 256 MiB")


def parameter_count(vocab_size: int, context_size: int, embedding_dim: int, hidden_dim: int) -> int:
    """Embedding V*D + hidden C*D*H+H + output H*V+V; no tied weights."""
    return vocab_size * (embedding_dim + hidden_dim + 1) + context_size * embedding_dim * hidden_dim + hidden_dim


def theoretical_memory(config: BenchmarkConfig) -> dict:
    parameters = parameter_count(config.vocab_size, config.context_size, config.embedding_dim, config.hidden_dim)
    return {"parameter_count": parameters, "float32_parameter_bytes": parameters * 4,
            "gradient_bytes_lower_bound": parameters * 4,
            "adam_first_second_moments_bytes_lower_bound": parameters * 8,
            "parameter_gradient_adam_bytes_lower_bound": parameters * 16,
            "single_logits_tensor_bytes": config.batch_size * config.vocab_size * 4,
            "scope": "Theoretical lower bound, not peak memory; excludes allocator/framework, activations, logits gradients and optimizer temporaries."}


def build_model(config: BenchmarkConfig):
    import torch

    class NPLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
            self.hidden = torch.nn.Linear(config.context_size * config.embedding_dim, config.hidden_dim)
            self.output = torch.nn.Linear(config.hidden_dim, config.vocab_size)

        def forward(self, context_ids):
            embeddings = self.embedding(context_ids)
            hidden = torch.tanh(self.hidden(embeddings.flatten(start_dim=1)))
            return self.output(hidden)

    return NPLM().to(device=config.device, dtype=torch.float32)


def train_step(model, optimizer, contexts, targets):
    import torch.nn.functional as functional
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(contexts), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def pipeline_idle_snapshot(root: str | Path, check_processes: bool = True) -> dict:
    """Fail closed on incomplete manifests or running preprocessing/coverage jobs."""
    root = Path(root)
    reports, blockers, inspected = root / "reports", [], []
    paths = {reports / name for name in BUSY_REPORTS}
    paths.update(reports.glob("*coverage*progress*.json"))
    paths.update(reports.glob("*coverage*state*.json"))
    for path in sorted(paths):
        if not path.exists():
            continue
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            status = payload.get("status")
            inspected.append({"file": path.name, "status": status, "sha256": hashlib.sha256(raw).hexdigest()})
            if status != "complete":
                blockers.append(f"{path.name}: {status!r}")
        except (OSError, ValueError, TypeError) as error:
            blockers.append(f"{path.name}: unreadable ({type(error).__name__})")
    active = []
    if check_processes:
        try:
            listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                                     text=True, check=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Cannot verify that preprocessing processes are idle") from error
        pattern = re.compile(r"(?<![\w.-])(?:" + "|".join(re.escape(name) for name in PIPELINE_PROGRAMS) + r")(?:\s|$)")
        for line in listing.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) != os.getpid() and pattern.search(fields[1]):
                active.append({"pid": int(fields[0]), "program": pattern.search(fields[1]).group(0).strip()})
        blockers.extend(f"process {row['pid']}: {row['program']}" for row in active)
    return {"idle": not blockers, "blockers": blockers, "reports": inspected,
            "active_processes": active, "process_check_performed": check_processes}


def assert_pipeline_idle(root: str | Path) -> dict:
    snapshot = pipeline_idle_snapshot(root)
    if not snapshot["idle"]:
        raise RuntimeError("Benchmark refused while data jobs are incomplete/running: " + "; ".join(snapshot["blockers"]))
    return snapshot


def extrapolated_budgets(targets_per_second: float, batch_size: int, utilization_margin: float) -> dict:
    if not math.isfinite(targets_per_second) or targets_per_second <= 0 or batch_size < 1 or not 0 < utilization_margin <= 1:
        raise ValueError("Positive finite throughput, batch size and 0 < utilization_margin <= 1 required")
    rates = {}
    for hours in (3, 8):
        steps = math.floor(targets_per_second * hours * 3600 * utilization_margin / batch_size)
        rates[f"{hours}h"] = {"hours": hours, "optimizer_steps": steps,
                              "target_prediction_events": steps * batch_size}
    return {"kind": "EXTRAPOLATION_ONLY", "utilization_margin": utilization_margin,
            "budgets": rates,
            "unit": "Target prediction events, including repeated epochs; not unique tokens or paragraphs.",
            "excluded": "Real data loading, evaluation/checkpoints, long-run thermal throttling and other app activity.",
            "quality_results": None}


def _configure_torch(config: BenchmarkConfig):
    import torch
    torch.set_num_threads(config.cpu_threads)
    if torch.get_num_interop_threads() != config.interop_threads:
        torch.set_num_interop_threads(config.interop_threads)
    torch.manual_seed(config.seed)
    return torch


def _batches(torch, config: BenchmarkConfig):
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    batches, digest = [], hashlib.sha256()
    for _ in range(4):
        contexts = torch.randint(config.vocab_size, (config.batch_size, config.context_size), generator=generator)
        targets = torch.randint(config.vocab_size, (config.batch_size,), generator=generator)
        digest.update(contexts.numpy().tobytes())
        digest.update(targets.numpy().tobytes())
        batches.append((contexts.to(config.device), targets.to(config.device)))
    return batches, digest.hexdigest()


def _synchronize(torch, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _mps_snapshot(torch, device: str) -> dict | None:
    if device != "mps":
        return None
    return {"current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory()}


def _peak_rss_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def benchmark_single(config: BenchmarkConfig, root: str | Path = ROOT) -> dict:
    """Called in a fresh child by the suite; refuse contended pipeline timing."""
    config.validate()
    idle = assert_pipeline_idle(root)
    torch = _configure_torch(config)
    import numpy
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
               "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
               "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset (disabled by default)"),
               "dtype": "float32", "optimizer": "AdamW(foreach=False, fused=False)",
               "loss": "full vocabulary cross_entropy", "pid": os.getpid()}
    base = {"config": asdict(config), "runtime": runtime, "memory_theoretical": theoretical_memory(config),
            "pipeline_idle": idle, "kind": "SYNTHETIC_MICROBENCHMARK", "corpus_training_performed": False,
            "quality_results": None}
    if config.device == "mps" and not torch.backends.mps.is_available():
        return {**base, "status": "unavailable", "reason": "Installed PyTorch reports MPS unavailable"}
    if config.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS timing requires CPU fallback disabled so device comparison stays explicit")
    started = time.perf_counter()
    model = build_model(config)
    actual_params = sum(parameter.numel() for parameter in model.parameters())
    if actual_params != base["memory_theoretical"]["parameter_count"]:
        raise RuntimeError("Model parameter count differs from theoretical formula")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False, fused=False)
    batches, batch_sha = _batches(torch, config)
    mps_snapshots = [{"stage": "model_and_data", "memory": _mps_snapshot(torch, config.device)}]
    for index in range(config.warmup_steps):
        loss = train_step(model, optimizer, *batches[index % len(batches)])
        if time.perf_counter() - started > config.max_seconds:
            raise TimeoutError("Microbenchmark warmup exceeded bounded time limit")
    _synchronize(torch, config.device)
    if not math.isfinite(float(loss.item())):
        raise RuntimeError("Non-finite warmup loss")
    mps_snapshots.append({"stage": "after_warmup", "memory": _mps_snapshot(torch, config.device)})
    durations, process_cpu_times = [], []
    for repeat in range(config.repeats):
        _synchronize(torch, config.device)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for index in range(config.steps_per_repeat):
            loss = train_step(model, optimizer, *batches[(repeat * config.steps_per_repeat + index) % len(batches)])
        _synchronize(torch, config.device)
        durations.append(time.perf_counter() - start_wall)
        process_cpu_times.append(time.process_time() - start_cpu)
        if not math.isfinite(float(loss.item())):
            raise RuntimeError("Non-finite measured-step loss")
        if time.perf_counter() - started > config.max_seconds:
            raise TimeoutError("Microbenchmark exceeded bounded time limit")
    mps_snapshots.append({"stage": "after_measurement", "memory": _mps_snapshot(torch, config.device)})
    per_repeat_targets = config.batch_size * config.steps_per_repeat
    median_seconds = statistics.median(durations)
    rate = per_repeat_targets / median_seconds
    memory = {"peak_host_rss_bytes": _peak_rss_bytes(),
              "peak_host_rss_scope": "High-water mark of this fresh subprocess including imports/model/optimizer/batches; MPS unified memory is not separately additive.",
              "mps_snapshots": mps_snapshots if config.device == "mps" else [],
              "mps_snapshot_scope": "Point-in-time allocator/driver samples; these are not a measured device peak."}
    return {**base, "status": "complete", "actual_parameter_count": actual_params,
            "batch_ids_sha256": batch_sha, "input_distribution": "four resident uniform random mini-batches; reused only to measure compute",
            "timing": {"warmup_excluded": True, "mps_synchronized": config.device == "mps",
                       "repeat_wall_seconds": durations, "repeat_process_cpu_seconds": process_cpu_times,
                       "median_repeat_seconds": median_seconds, "min_repeat_seconds": min(durations),
                       "max_repeat_seconds": max(durations), "median_step_seconds": median_seconds / config.steps_per_repeat,
                       "measured_steps": config.repeats * config.steps_per_repeat,
                       "measured_target_prediction_events": per_repeat_targets * config.repeats,
                       "measured_wall_seconds": sum(durations),
                       "median_target_predictions_per_second": rate,
                       "aggregate_target_predictions_per_second": per_repeat_targets * config.repeats / sum(durations),
                       "context_token_lookups_per_second": rate * config.context_size},
            "memory": memory, "finite_loss_check": True,
            "loss_check_scope": "Synthetic correctness only; loss values are not model-quality comparisons.",
            "time_budgets": extrapolated_budgets(rate, config.batch_size, config.utilization_margin)}


def smoke_test(devices: list[str] | None = None) -> dict:
    """Tiny correctness exercise; intentionally omit every performance estimate."""
    import torch
    rows = []
    for device in (["cpu"] if devices is None else devices):
        config = BenchmarkConfig(vocab_size=17, context_size=3, embedding_dim=4, hidden_dim=8,
                                 batch_size=4, device=device, cpu_threads=1, warmup_steps=1, repeats=1, steps_per_repeat=1)
        config.validate()
        _configure_torch(config)
        if device == "mps" and not torch.backends.mps.is_available():
            rows.append({"device": device, "status": "unavailable"})
            continue
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
        batches, batch_sha = _batches(torch, config)
        before = next(model.parameters()).detach().clone()
        loss = train_step(model, optimizer, *batches[0])
        _synchronize(torch, device)
        if not math.isfinite(float(loss.item())) or not bool(torch.any(before != next(model.parameters())).item()):
            raise RuntimeError("Smoke test did not produce a finite loss and a parameter update")
        rows.append({"device": device, "status": "passed", "finite_loss": True, "parameter_updated": True,
                     "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                     "batch_ids_sha256": batch_sha})
    return {"status": "smoke-test-only", "kind": "CORRECTNESS_ONLY", "results": rows,
            "timing_comparison_valid": False, "throughput": None, "time_budgets": None,
            "corpus_training_performed": False}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_suite(configs: list[BenchmarkConfig], output: str | Path, root: str | Path = ROOT) -> dict:
    root, output = Path(root), Path(output)
    if not configs or len(configs) > 40:
        raise ValueError("Require 1–40 bounded benchmark configurations")
    for config in configs:
        config.validate()
    idle = assert_pipeline_idle(root)
    if output.exists():
        raise FileExistsError(f"Preserving existing benchmark result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".nplm-benchmark.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another NPLM benchmark is already running") from error
        report = {"status": "running", "kind": "SYNTHETIC_MICROBENCHMARK", "corpus_training_performed": False,
                  "actual_3h_or_8h_results": None, "quality_results": None,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "pipeline_idle": idle, "configuration_count": len(configs), "results": [],
                  "experiment_design": EXPERIMENT_DESIGN, "official_references": OFFICIAL_REFERENCES,
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        _atomic_json(output, report)
        for config in configs:
            assert_pipeline_idle(root)
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": str(config.cpu_threads), "MKL_NUM_THREADS": str(config.cpu_threads),
                                "VECLIB_MAXIMUM_THREADS": str(config.cpu_threads), "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
            with tempfile.TemporaryDirectory(prefix="nplm-microbenchmark-") as directory:
                child_output = Path(directory) / "result.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--child-config", json.dumps(asdict(config)),
                           "--child-output", str(child_output), "--root", str(root)]
                try:
                    child = subprocess.run(command, capture_output=True, text=True, env=environment,
                                           timeout=config.max_seconds + 60, check=False)
                    if child.returncode == 0 and child_output.exists():
                        result = json.loads(child_output.read_text(encoding="utf-8"))
                    else:
                        result = {"status": "failed", "config": asdict(config), "returncode": child.returncode,
                                  "error": (child.stderr or child.stdout)[-4000:]}
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout", "config": asdict(config),
                              "error": "Fresh child exceeded bounded microbenchmark timeout"}
            report["results"].append(result)
            _atomic_json(output, report)
            print(f"{config.device} threads={config.cpu_threads} V={config.vocab_size}: {result['status']}", flush=True)
        report["status"] = "complete" if all(row["status"] in {"complete", "unavailable"} for row in report["results"]) else "completed_with_failures"
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_json(output, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=list(DEFAULT_VOCABS))
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--cpu-threads", nargs="+", type=int, default=[4])
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps-per-repeat", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utilization-margin", type=float, default=0.70)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child-config", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_config:
        if not args.child_output:
            parser.error("child output required")
        _atomic_json(args.child_output, benchmark_single(BenchmarkConfig(**json.loads(args.child_config)), args.root))
        return
    if args.plan_only:
        output = args.output or args.root / "reports/training-comparison-plan.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(EXPERIMENT_DESIGN)
        print(output)
        return
    if args.smoke_test:
        output = args.output or args.root / "reports/nplm-benchmark-smoke.json"
        if output.exists():
            raise FileExistsError(f"Preserving existing smoke result: {output}")
        _atomic_json(output, smoke_test(args.devices))
        print(output)
        return
    configs = [BenchmarkConfig(vocab_size=vocab, context_size=args.context_size, embedding_dim=args.embedding_dim,
                               hidden_dim=args.hidden_dim, batch_size=args.batch_size, device=device,
                               cpu_threads=threads, seed=args.seed, warmup_steps=args.warmup_steps,
                               repeats=args.repeats, steps_per_repeat=args.steps_per_repeat,
                               max_seconds=args.max_seconds, utilization_margin=args.utilization_margin)
               for device in dict.fromkeys(args.devices)
               for threads in (list(dict.fromkeys(args.cpu_threads)) if device == "cpu" else [args.cpu_threads[0]])
               for vocab in dict.fromkeys(args.vocab_sizes)]
    run_suite(configs, args.output or args.root / "reports/nplm-benchmark.json", args.root)


if __name__ == "__main__":
    main()
