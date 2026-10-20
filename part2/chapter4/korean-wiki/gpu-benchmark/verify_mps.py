#!/usr/bin/env python3
"""Independent CPU-PyTorch numerical gates for the FP32 MPS experiment.

Small gates consume the engine's four-update check JSON and independently
replay every update, including all gradients, parameters and Adam moments.
The full-vocabulary gate uses the public engine tensor API in a fresh child;
all large arrays stay in memory and only comparison summaries are serialized.
No timing comparison, training checkpoint or complete epoch is produced.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
LOCK = ROOT / "cpu-benchmark/.timing.lock"
ENGINE = BASE / "benchmark_mps.py"
FIXTURES = {
    "tiny": {"vocab_size": 11, "context_size": 2, "embedding_dim": 3, "hidden_dim": 5, "batch_size": 4},
    "irregular": {"vocab_size": 73, "context_size": 3, "embedding_dim": 7, "hidden_dim": 67, "batch_size": 71},
}
CASES = {
    "torch_cpu/default": ("cpu", "default"),
    "mps/default": ("mps", "default"),
    "mps/preload_defer": ("mps", "preload_defer"),
    "mps/preload_defer_foreach": ("mps", "preload_defer_foreach"),
    "mps/preload_defer_foreach_metal": ("mps", "preload_defer_foreach_metal"),
}
TOLERANCES = {
    "declared_before_execution": True,
    "rule": "abs(actual-reference) <= atol + rtol*abs(reference)",
    "ordinary": {"atol": 2e-6, "rtol": 2e-4},
    "moment1": {"atol": 2e-9, "rtol": 2e-4},
    "moment2": {"atol": 2e-11, "rtol": 5e-4},
    "policy": "Same predeclared limits as the fixed-work CPU gate. Failures are retained, never relaxed after observing a device result.",
}
OPTIMIZER = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_json(value):
    def object_pairs(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = item
        return result
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(value, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def identity(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"Source changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def numerical_lock(externally_held=False):
    with LOCK.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        if externally_held:
            if acquired:
                fcntl.flock(stream, fcntl.LOCK_UN)
                raise RuntimeError("--lock-held requires the coordinator to own the shared numerical lock")
            yield
        else:
            if not acquired:
                raise RuntimeError("Another CPU/MPS model process owns cpu-benchmark/.timing.lock")
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def environment(variant):
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0",
               OMP_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
    if variant.endswith("_metal"):
        env["PYTORCH_MPS_PREFER_METAL"] = "1"
    else:
        env.pop("PYTORCH_MPS_PREFER_METAL", None)
    return env


def launch(command, variant, timeout=180):
    process = subprocess.Popen(command, env=environment(variant), cwd=BASE,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise TimeoutError("Bounded numerical child exceeded its timeout; process group terminated")
    result = parse_json(stdout)
    evidence = {"command": command, "exit_code": process.returncode, "stderr": stderr,
                "child_json_sha256": hashlib.sha256(stdout.encode()).hexdigest()}
    return result, evidence


def finite_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Non-finite/non-numeric {label}")


def finite_list(value, length, label):
    require(isinstance(value, list) and len(value) == length, f"Incorrect {label} length")
    for item in value:
        finite_number(item, label)


def validate_devices(devices, backend):
    for name in ("parameters", "inputs", "gradients", "moments"):
        entries = devices.get(name)
        require(isinstance(entries, list) and entries and all(str(value).split(":")[0] == backend for value in entries),
                f"Actual {name} device is not {backend}: {entries!r}")
    # capturable=False AdamW keeps bookkeeping step scalars on CPU. This is
    # explicit bookkeeping, not permission to put model math/state on CPU.
    step_devices = devices.get("optimizer_step_state", [])
    require(isinstance(step_devices, list) and all(str(value).split(":")[0] in {"cpu", backend} for value in step_devices),
            "Unexpected optimizer bookkeeping device")


def validate_runtime(runtime, variant):
    require(runtime.get("fallback_env") == "0", "CPU fallback must be explicitly disabled")
    require(runtime.get("mps_fast_math_env") == "0", "MPS fast math must be explicitly disabled")
    require(runtime.get("prefer_metal_env") == ("1" if variant.endswith("_metal") else None),
            "PREFER_METAL must be absent for baseline; presence='0' is not baseline")
    require(runtime.get("cpu_threads") == 4 and runtime.get("cpu_interop_threads") == 1,
            "Reference/host CPU thread budget changed")


def compare(torch, name, actual, expected, family="ordinary"):
    atol, rtol = TOLERANCES[family]["atol"], TOLERANCES[family]["rtol"]
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().to(torch.float64).reshape(-1)
    else:
        actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    expected = expected.detach().cpu().to(torch.float64).reshape(-1) if isinstance(expected, torch.Tensor) else torch.as_tensor(expected, dtype=torch.float64).reshape(-1)
    require(actual.shape == expected.shape and actual.numel() > 0, f"Shape mismatch: {name}")
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), f"Non-finite array: {name}")
    absolute = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    failures = absolute > allowed
    worst = int(torch.argmax(absolute))
    row = {"name": name, "family": family, "passed": not bool(failures.any()), "elements": actual.numel(),
           "max_absolute_error": float(absolute.max()),
           "max_relative_error": float((absolute / expected.abs().clamp_min(1e-30)).max()),
           "failed_elements": int(failures.sum()), "actual_at_worst": float(actual[worst]),
           "expected_at_worst": float(expected[worst]), "worst_absolute_error_index": worst,
           "atol": atol, "rtol": rtol}
    if family in {"moment1", "moment2"}:
        if family == "moment2":
            require(bool(torch.all(actual >= 0)) and bool(torch.all(expected >= 0)), "Second moments must be nonnegative")
        strong = expected.abs() > allowed
        erased = int(((actual == 0) & strong).sum())
        all_zero = bool((expected != 0).any()) and not bool((actual != 0).any())
        row.update(expected_nonzero_elements=int((expected != 0).sum()), actual_nonzero_elements=int((actual != 0).sum()),
                   erased_strong_nonzero_elements=erased, entire_vector_incorrectly_zero=all_zero)
        row["passed"] = row["passed"] and erased == 0 and not all_zero
    return row


def build_reference(torch, config, initial, layout):
    from torch import nn
    class IndependentNplm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
            self.hidden = nn.Linear(config["context_size"] * config["embedding_dim"], config["hidden_dim"])
            self.output = nn.Linear(config["hidden_dim"], config["vocab_size"])

        def forward(self, contexts):
            inputs = self.embedding(contexts).reshape(contexts.shape[0], config["context_size"] * config["embedding_dim"])
            return self.output(torch.tanh(self.hidden(inputs)))
    model = IndependentNplm().to(device="cpu", dtype=torch.float32)
    parameters = [model.embedding.weight, model.hidden.weight, model.hidden.bias, model.output.weight, model.output.bias]
    flat = torch.as_tensor(initial, dtype=torch.float32)
    require(bool(torch.isfinite(flat).all()), "Reference initial values overflow float32")
    with torch.no_grad():
        for entry, parameter in zip(layout, parameters):
            require(list(parameter.shape) == entry["shape"], "Reference parameter order/shape mismatch")
            parameter.copy_(flat[entry["offset"]:entry["offset"] + entry["length"]].reshape(parameter.shape))
    optimizer = torch.optim.AdamW(parameters, **OPTIMIZER, foreach=False, fused=False, capturable=False)
    return model, parameters, optimizer


def reference_update(torch, reference, contexts, targets, clip_norm):
    from torch.nn import functional as functional
    model, parameters, optimizer = reference
    contexts = torch.as_tensor(contexts, dtype=torch.int64, device="cpu")
    targets = torch.as_tensor(targets, dtype=torch.int64, device="cpu")
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(contexts), targets, reduction="mean")
    loss.backward()
    raw = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, foreach=False, error_if_nonfinite=True)
    scale = torch.clamp(clip_norm / (norm.detach() + 1e-6), max=1.0)
    clipped = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
    optimizer.step()
    captured = {"loss": loss.detach(), "gradient_norm": norm.detach(), "clipping_scale": scale,
                "raw_gradients": raw, "clipped_gradients": clipped,
                "updated_params": torch.cat([parameter.detach().reshape(-1) for parameter in parameters]),
                "moment1": torch.cat([optimizer.state[parameter]["exp_avg"].detach().reshape(-1) for parameter in parameters]),
                "moment2": torch.cat([optimizer.state[parameter]["exp_avg_sq"].detach().reshape(-1) for parameter in parameters])}
    require(all(bool(torch.isfinite(value).all()) for value in captured.values()), "Non-finite independent CPU oracle")
    return captured


def validate_optimizer(optimizer, variant):
    for name in ("lr", "eps", "weight_decay"):
        finite_number(optimizer.get(name), f"optimizer.{name}")
        require(math.isclose(optimizer[name], OPTIMIZER[name], rel_tol=1e-12), f"Optimizer {name} changed")
    require(tuple(optimizer.get("betas", [])) == OPTIMIZER["betas"], "Optimizer betas changed")
    require(optimizer.get("foreach") is ("foreach" in variant) and optimizer.get("fused") is False
            and optimizer.get("capturable") is False, "Optimizer execution policy changed")


def configure_reference_threads():
    import torch
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return torch


def check_small(data, backend, variant, fixture, clip_norm):
    import numpy as np
    from corpus_fixture import create_initial, parameter_layout, checksum_f32, synthetic_batches
    require(data.get("kind") == "MPS_NPLM_NUMERICAL_CHECK" and data.get("status") == "complete", "Unexpected child check schema")
    require(data.get("backend") == backend and data.get("variant") == variant and data.get("fixture") == fixture,
            "Child check did not apply requested variant/fixture/backend")
    require(data.get("dtype") == "float32" and str(data.get("actual_parameter_device")).split(":")[0] == backend,
            "Actual parameter dtype/device differs from requested backend")
    validate_devices(data["actual_devices"], backend)
    validate_runtime(data["runtime"], variant)
    validate_optimizer(data["optimizer"], variant)
    require(data.get("clip_norm") == clip_norm, "Requested clipping policy changed")
    config = data["config"]
    expected_config = {**FIXTURES[fixture], "seed": 42, "threads": 4, "warmup_steps": 0, "steps": 4}
    require(all(type(config.get(key)) is int and config[key] == value for key, value in expected_config.items()), "Synthetic check shape/count changed")
    initial, layout = create_initial(config)
    require(data["parameter_layout"] == layout, "Initial parameter order/offset/layout changed")
    count = initial.size
    finite_list(data["initial_params"], count, "initial_params")
    actual_initial = np.asarray(data["initial_params"], dtype=np.float32)
    require(actual_initial.tobytes() == initial.tobytes(), "Child initializer bits differ from C helper")
    require(data["initial_parameter_checksum"] == checksum_f32(initial), "Initializer FNV evidence differs")
    history = data["history"]
    require(isinstance(history, list) and len(history) == 4, "Four consecutive updates are required")
    batches = synthetic_batches(config)
    torch = configure_reference_threads()
    reference = build_reference(torch, config, initial, layout)
    comparisons, input_digest, ever_used = [], hashlib.sha256(), set()
    fields = ("raw_gradients", "clipped_gradients", "updated_params", "moment1", "moment2")
    for step, (row, (contexts, targets)) in enumerate(zip(history, batches), 1):
        require(row.get("step") == step, "Update counter/order changed")
        require(row["contexts"] == contexts.tolist() and row["targets"] == targets.tolist(), "Integer fixture IDs/order differ from C synthetic stream")
        input_digest.update(contexts.astype("<u4").tobytes())
        input_digest.update(targets.astype("<u4").tobytes())
        for key in fields:
            finite_list(row[key], count, f"step{step}.{key}")
        for key in ("loss", "gradient_norm", "clipping_scale"):
            finite_number(row[key], f"step{step}.{key}")
        require(row["loss"] >= 0 and row["gradient_norm"] >= 0 and 0 < row["clipping_scale"] <= 1,
                "Invalid loss/norm/scale")
        require(type(row.get("clip_active")) is bool and row["clip_active"] == (row["clipping_scale"] < 1), "Invalid clipping-active evidence")
        expected = reference_update(torch, reference, contexts, targets, clip_norm)
        require(row["clip_active"] == (float(expected["clipping_scale"]) < 1), "CPU and device clipping activation differ")
        for key in (*fields, "loss", "gradient_norm", "clipping_scale"):
            actual = row[key] if key in fields else [row[key]]
            family = key if key in ("moment1", "moment2") else "ordinary"
            comparisons.append(compare(torch, f"update{step}.{key}", actual, expected[key], family))
        unused = sorted(set(range(config["vocab_size"])) - set(contexts.reshape(-1).tolist()))
        ever_used.update(contexts.reshape(-1).tolist())
        indices = [token * config["embedding_dim"] + dimension for token in unused for dimension in range(config["embedding_dim"])]
        require(all(row[key][index] == 0 for key in ("raw_gradients", "clipped_gradients") for index in indices),
                "Unused embedding rows have nonzero device gradients")
        never_used = sorted(set(range(config["vocab_size"])) - ever_used)
        never_used_indices = [token * config["embedding_dim"] + dimension for token in never_used for dimension in range(config["embedding_dim"])]
        require(all(row[key][index] == 0 for key in ("moment1", "moment2") for index in never_used_indices),
                "Never-used embedding rows have nonzero device moments")
        for parameter in reference[1]:
            require(float(reference[2].state[parameter]["step"]) == step, "Independent optimizer update counter changed")
    for key, history_key in (("grads", "raw_gradients"), ("optimizer_grads", "clipped_gradients"),
                              ("updated_params", "updated_params"), ("moment1", "moment1"), ("moment2", "moment2")):
        require(data[key] == history[-1][history_key], f"Final alias differs from history: {key}")
    return {"status": "complete", "passed": all(row["passed"] for row in comparisons),
            "fixture": fixture, "clip_norm": clip_norm, "updates": 4, "parameter_count": int(count),
            "backend": backend, "variant": variant, "runtime": data["runtime"], "actual_devices": data["actual_devices"],
            "initial_parameter_checksum": data["initial_parameter_checksum"], "input_sequence_sha256": input_digest.hexdigest(),
            "initial_parameter_bits_match_C": True, "synthetic_ID_stream_matches_C": True,
            "all_history_arrays_finite_checked_on_CPU": True, "all_four_updates_compared": True,
            "comparison_count": len(comparisons), "failed_comparisons": sum(not row["passed"] for row in comparisons),
            "comparisons": comparisons,
            "scope": "Every one of four synthetic updates, including full raw/clipped gradients, parameters and moments; CPU step bookkeeping is explicitly permitted for capturable=False AdamW."}


def full_size_child(backend, variant, clip_norm, corpus_path):
    """Fresh child only: compare resident tensors, never serialize full arrays."""
    import benchmark_mps as engine
    engine.configure_environment(variant)
    torch = configure_reference_threads()
    import numpy as np
    from corpus_fixture import Corpus, FULL_CONFIG, checksum_batch, checksum_f32, parameter_layout
    config = {**FULL_CONFIG, "threads": 4, "warmup_steps": 0, "steps": 1}
    corpus = Corpus(corpus_path, config["context_size"], config["vocab_size"])
    require(corpus.windows == 124232535, "Full geometry must use the preserved real-Wiki target pool")
    sampler = corpus.sampler(42)
    sampler_initial = sampler.snapshot()
    contexts, targets = sampler.next_batch(config["batch_size"])
    require(checksum_batch(contexts, targets) == "1a55aed564d5b308", "Full-size first batch differs from the preserved C run")
    bundle = engine.build_model_optimizer(config, backend, variant)
    layout, initial = parameter_layout(config), bundle["initial_flat"]
    require(bundle["layout"] == layout and initial.size == 6883968, "Full-size model shape/parameter order changed")
    initial_checksum = checksum_f32(initial)
    require(initial_checksum == "b5b7b03ca89a72af", "Full-size initializer differs from the preserved C run")
    actual_initial = torch.cat([parameter.detach().reshape(-1) for parameter in bundle["parameters"]]).cpu().numpy()
    require(actual_initial.dtype == np.float32 and actual_initial.tobytes() == initial.tobytes(), "Actual device model initial bits differ from C initializer")
    del actual_initial
    reference = build_reference(torch, config, initial, layout)
    device_contexts = torch.from_numpy(contexts).to(bundle["device"])
    device_targets = torch.from_numpy(targets).to(bundle["device"])
    captured = engine.train_update(bundle, device_contexts, device_targets, clip_norm, capture=True)
    if backend == "mps":
        torch.mps.synchronize()
    for key, value in captured.items():
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and str(value.device).split(":")[0] == backend, f"Actual full-size {key} dtype/device differs")
    # Transfer each result once; subsequent comparisons are all CPU operations.
    actual = {key: value.detach().cpu() for key, value in captured.items()}
    require(all(bool(torch.isfinite(value).all()) for value in actual.values()), "Non-finite full-size device state")
    expected = reference_update(torch, reference, contexts, targets, clip_norm)
    comparisons = [compare(torch, "fullV.update1." + key, actual[key], expected[key],
                           key if key in ("moment1", "moment2") else "ordinary") for key in expected]
    runtime, devices = engine.runtime_metadata(), engine.actual_devices(bundle, device_contexts)
    validate_runtime(runtime, variant)
    validate_devices(devices, backend)
    for parameter in bundle["parameters"]:
        counter = bundle["optimizer"].state[parameter]["step"]
        require(math.isfinite(float(counter.item())) and float(counter.item()) == 1, "Full-size optimizer counter differs")
    return {"status": "complete", "passed": all(row["passed"] for row in comparisons),
            "kind": "FULL_GEOMETRY_REAL_WIKI_ONE_UPDATE_GATE", "backend": backend, "variant": variant,
            "clip_norm": clip_norm, "updates": 1, "config": config, "parameter_count": int(initial.size),
            "runtime": runtime, "actual_devices": devices, "all_arrays_finite_checked_on_CPU": True,
            "all_large_arrays_kept_in_memory": True, "initial_parameter_checksum": initial_checksum,
            "input_checksum": checksum_batch(contexts, targets), "initial_parameter_bits_match_C": True,
            "sampler_initial": sampler_initial, "sampler_after_one_batch": sampler.snapshot(),
            "first_windows": corpus.first_samples(), "comparison_count": len(comparisons),
            "failed_comparisons": sum(not row["passed"] for row in comparisons), "comparisons": comparisons,
            "scope": "One actual real-Wiki batch at V35328/C8/D64/H128/B128; not four full-size updates or a complete training-quality evaluation."}


def is_unsupported(error):
    return isinstance(error, NotImplementedError) or (isinstance(error, RuntimeError) and any(
        message in str(error).lower() for message in ("not implemented for", "not supported on mps", "does not support mps", "mps unavailable")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--corpus", type=Path, default=ROOT / "training-corpora/wiki-k35328")
    parser.add_argument("--full-clip-norms", nargs="+", type=float, default=[1.0])
    parser.add_argument("--report", type=Path, default=BASE / "mps-model-verification.json")
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--full-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "mps"), help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=tuple(value[1] for value in CASES.values()), help=argparse.SUPPRESS)
    parser.add_argument("--clip-norm", type=float, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.full_child:
        with numerical_lock(externally_held=True):
            try:
                result = full_size_child(args.backend, args.variant, args.clip_norm, args.corpus)
            except Exception as error:
                result = {"status": "unsupported" if is_unsupported(error) or type(error).__name__ == "UnsupportedCase" else "failed",
                          "passed": False, "error": f"{type(error).__name__}: {error}",
                          "backend": args.backend, "variant": args.variant, "cpu_fallback_used": False}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("passed") else 2 if result["status"] == "unsupported" else 1
    if args.report.exists():
        raise FileExistsError(f"Preserving prior numerical report: {args.report}")
    names = list(dict.fromkeys(args.case or CASES))
    report = {"kind": "TORCH_CPU_MPS_INDEPENDENT_NUMERICAL_GATES", "status": "running", "passed": False,
              "started_at": datetime.now().astimezone().isoformat(), "cases": {},
              "tolerances_predeclared": TOLERANCES,
              "scope": "Four synthetic updates at tiny/irregular shapes and one actual full-vocabulary Wiki update; no throughput or learned-quality claim.",
              "default_full_size_clips": args.full_clip_norms, "oracle_threads": 4,
              "failure_policy": "No implicit CPU fallback, no tolerance relaxation, unsupported options remain explicit skips.",
              "checkpoint_policy": "No trained checkpoint or five-epoch result is produced."}
    try:
        from corpus_fixture import helper_library_path
        require(all(math.isfinite(value) and value > 0 for value in args.full_clip_norms), "Invalid full-size clip norm")
        sources = [ENGINE, BASE / "corpus_fixture.py", BASE / "init_reference.c", helper_library_path(), Path(__file__)]
        report["sources_before"] = {str(path.resolve()): identity(path) for path in sources}
        reference_path = ROOT / "reports/mps-corpus-reference-verification.json"
        reference = parse_json(reference_path.read_text(encoding="utf-8"))
        require(reference.get("passed") is True and reference.get("initial_parameter_checksum") == "b5b7b03ca89a72af"
                and reference.get("input_checksum") == "1a55aed564d5b308", "Exact C initialization/integer fixture identity gate is required first")
        report["corpus_reference"] = {**identity(reference_path), "passed": True}
        atomic_json(args.report, report)
        # The parent only runs CPU oracle math; each MPS policy is initialized
        # in a fresh child with its own before-import environment.
        os.environ.update({key: value for key, value in environment("default").items()
                           if key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")})
        os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
        with numerical_lock(args.lock_held):
            for name in names:
                backend, variant = CASES[name]
                entry = {"backend": backend, "variant": variant, "status": "running", "passed": False,
                         "small_checks": [], "full_size_checks": [], "small_gate_passed": False,
                         "full_size_gate_passed": False, "sources": report["sources_before"]}
                report["cases"][name] = entry
                unsupported = False
                for fixture in FIXTURES:
                    for clip in (1.0, 0.05):
                        command = [sys.executable, str(ENGINE), "--check", "--backend", backend, "--variant", variant,
                                   "--fixture", fixture, "--clip-norm", str(clip), "--check-updates", "4"]
                        row = {"fixture": fixture, "clip_norm": clip, "updates": 4, "passed": False}
                        try:
                            data, evidence = launch(command, variant)
                            row.update(evidence)
                            if data.get("status") == "unsupported":
                                row.update(status="unsupported", reason=data.get("error"), cpu_fallback_used=False)
                                unsupported = True
                            else:
                                require(evidence["exit_code"] == 0, f"Device check child failed: {data.get('error', 'unknown error')}")
                                row.update(check_small(data, backend, variant, fixture, clip))
                        except Exception as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        entry["small_checks"].append(row)
                        atomic_json(args.report, report)
                        if unsupported:
                            break
                    if unsupported:
                        break
                if not unsupported:
                    for clip in args.full_clip_norms:
                        command = [sys.executable, str(Path(__file__).resolve()), "--full-child", "--backend", backend,
                                   "--variant", variant, "--clip-norm", str(clip), "--corpus", str(args.corpus.resolve())]
                        row = {"updates": 1, "clip_norm": clip, "passed": False}
                        try:
                            data, evidence = launch(command, variant, timeout=240)
                            row.update(evidence)
                            row.update(data)
                            unsupported |= data.get("status") == "unsupported"
                        except Exception as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        entry["full_size_checks"].append(row)
                        atomic_json(args.report, report)
                entry["small_gate_passed"] = len(entry["small_checks"]) == 4 and all(row["passed"] for row in entry["small_checks"])
                entry["full_size_gate_passed"] = (len(entry["full_size_checks"]) == len(args.full_clip_norms)
                                                  and all(row["passed"] for row in entry["full_size_checks"]))
                entry["passed"] = entry["small_gate_passed"] and entry["full_size_gate_passed"]
                numerical_failure = any(row.get("status") == "failed" or (row.get("status") == "complete" and not row["passed"])
                                        for row in entry["small_checks"] + entry["full_size_checks"])
                entry["status"] = "complete" if entry["passed"] else "failed" if numerical_failure else "unsupported" if unsupported else "failed"
                entry["unsupported"] = unsupported
                entry["comparison_count"] = sum(row.get("comparison_count", 0) for row in entry["small_checks"] + entry["full_size_checks"])
                entry["failed_comparisons"] = sum(row.get("failed_comparisons", 0) for row in entry["small_checks"] + entry["full_size_checks"])
                atomic_json(args.report, report)
            report["sources_after"] = {str(path.resolve()): identity(path) for path in sources}
            require(report["sources_before"] == report["sources_after"], "Engine/helper/verifier changed during numerical checks")
        report["passed"] = all(entry["passed"] for entry in report["cases"].values())
        supported = [entry for entry in report["cases"].values() if entry["status"] != "unsupported"]
        report["all_supported_cases_passed"] = bool(supported) and all(entry["passed"] for entry in supported)
        report["unsupported_cases"] = [name for name, entry in report["cases"].items() if entry["status"] == "unsupported"]
        report["failed_cases"] = [name for name, entry in report["cases"].items() if entry["status"] == "failed"]
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
    report["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_json(args.report, report)
    if args.quiet:
        rendered = {key: report[key] for key in ("kind", "status", "passed", "unsupported_cases", "failed_cases", "error") if key in report}
        rendered["cases"] = {name: {key: entry.get(key) for key in ("status", "passed", "small_gate_passed", "full_size_gate_passed", "comparison_count", "failed_comparisons")}
                             for name, entry in report["cases"].items()}
        rendered["report"] = str(args.report.resolve())
    else:
        rendered = report
    print(json.dumps(rendered, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
