#!/usr/bin/env python3
"""Strict independent CPU-oracle gates for the new MPS optimization candidates.

One fresh child owns each candidate. A geometry's model/compiler wrapper is
reused for clip 1/.05, with two real warmups and an exact in-place reset before
each four-update check. Full-model tensors stay in memory; only summaries are
serialized. Existing sources, data and tolerances are never modified.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


BASE = Path(__file__).resolve().parent
GPU = BASE.parent
ROOT = GPU.parent
sys.path.insert(0, str(GPU))
sys.path.insert(0, str(BASE))
import verify_mps as oracle


CASES = ("control", "native_fused", "native_fused_zero_none", "metal_clip_fused", "compiled_model_fused")
OPTIONS = {
    "control": {"foreach": True, "fused": False, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": False},
    "native_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": False},
    "native_fused_zero_none": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": False, "metal_clip": False},
    "metal_clip_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": False, "metal_clip": True},
    "compiled_model_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": True, "metal_clip": False},
}
FIELDS = ("loss", "gradient_norm", "clipping_scale", "raw_gradients", "clipped_gradients", "updated_params", "moment1", "moment2")
GEOMETRIES = ("tiny", "irregular", "full_wiki")
CACHE_VARIABLES = ("TORCHINDUCTOR_COMPILE_THREADS", "MAX_JOBS", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_COMPILE_DEBUG_DIR")


def require(condition, message):
    oracle.require(condition, message)


def unsupported(error):
    if oracle.is_unsupported(error) or type(error).__name__ == "UnsupportedCase":
        return True
    text = str(error).lower()
    return (isinstance(error, ModuleNotFoundError) and getattr(error, "name", "") in {"triton", "triton.language"}) or (
        isinstance(error, RuntimeError) and any(value in text for value in (
            "unsupported device type", "mps is not supported", "mps backend is not supported", "not supported on macos")))


def source_paths():
    from corpus_fixture import helper_library_path
    return [BASE / "engine.py", BASE / "compiled_path.py", BASE / "metal_clip.py", Path(__file__),
            GPU / "benchmark_mps.py", GPU / "verify_mps.py", GPU / "corpus_fixture.py",
            GPU / "init_reference.c", helper_library_path()]


def identities():
    return {str(path.resolve()): oracle.identity(path) for path in source_paths()}


def parameter_ids(bundle):
    expected = tuple(id(value) for value in bundle["parameters"])
    require(tuple(id(value) for value in bundle["model"].parameters()) == expected, "Compiler changed model Parameter order/identity")
    require(tuple(id(value) for group in bundle["optimizer"].param_groups for value in group["params"]) == expected,
            "Optimizer Parameter order/identity differs")
    return expected


def validate_policy(torch, bundle, case):
    expected = OPTIONS[case]
    require(bundle.get("case") == case and bundle.get("backend") == "mps", "Wrong candidate/backend")
    require(bundle.get("zero_grad_set_to_none") is expected["zero_grad_set_to_none"], "Gradient clearing policy differs")
    parameter_ids(bundle)
    for parameter in bundle["parameters"]:
        require(parameter.device.type == "mps" and parameter.dtype == torch.float32, "Model must stay FP32 on MPS")
    for group in bundle["optimizer"].param_groups:
        require(group.get("foreach") is expected["foreach"] and group.get("fused") is expected["fused"], "Actual optimizer flags differ")
        require(group.get("capturable") is False, "Capturable AdamW is outside this comparison")
        for key in ("lr", "eps", "weight_decay"):
            require(group[key] == oracle.OPTIMIZER[key], f"Optimizer {key} changed")
        require(tuple(group["betas"]) == oracle.OPTIMIZER["betas"], "Optimizer betas changed")
    metadata = bundle["optimization_metadata"]
    require(all(metadata["requested_options"].get(key) is value for key, value in expected.items()), "Candidate metadata differs from actual request")
    if expected["compiled_model"]:
        from torch._dynamo import config as dynamo
        compiled = metadata["compile"]
        require(compiled.get("enabled") is True and compiled.get("options") == {
            "backend": "inductor", "fullgraph": True, "dynamic": False, "mode": "default"}, "Strict model-only compilation options changed")
        require(hasattr(bundle["model"], "_orig_mod") and dynamo.suppress_errors is False
                and not dynamo.disable and not dynamo.debug_backend_override, "Eager fallback/compiler override is forbidden")
    return {"requested_options": dict(expected), "actual_optimizer_flags_verified": True,
            "original_parameter_objects_and_order_preserved": True}


def compiler_snapshot(torch):
    result = {"counters": {}, "generated_kernel_count": None, "available": True}
    try:
        from torch._dynamo.utils import counters
        for name in ("frames", "stats", "graph_break", "unimplemented", "recompiles", "inductor", "aot_autograd"):
            result["counters"][name] = {str(key): int(value) if isinstance(value, int) else str(value)
                                        for key, value in counters.get(name, {}).items()}
        from torch._inductor import metrics
        value = getattr(metrics, "generated_kernel_count", None)
        result["generated_kernel_count"] = int(value) if value is not None else None
    except (ImportError, AttributeError) as error:
        result.update(available=False, unavailable_reason=f"{type(error).__name__}: {error}")
    return result


def compiler_delta(before, after):
    deltas = {}
    for section in set(before["counters"]) | set(after["counters"]):
        old, new = before["counters"].get(section, {}), after["counters"].get(section, {})
        deltas[section] = {key: new.get(key, 0) - old.get(key, 0)
                           for key in set(old) | set(new)
                           if isinstance(new.get(key, 0), int) and isinstance(old.get(key, 0), int)}
    old, new = before.get("generated_kernel_count"), after.get("generated_kernel_count")
    return {"counters": deltas, "generated_kernel_count": new - old if old is not None and new is not None else None}


def reset_and_prove(torch, engine, bundle, initial_bytes, optimizer_id, expected_ids, expected_step=2):
    before_state = []
    for parameter in bundle["parameters"]:
        state = bundle["optimizer"].state[parameter]
        require(all(key in state for key in ("step", "exp_avg", "exp_avg_sq")), "Warmup did not allocate complete Adam state")
        require(float(state["step"].detach().cpu().item()) == expected_step, "Warmup optimizer counter differs")
        for key in ("exp_avg", "exp_avg_sq"):
            require(bool(torch.isfinite(state[key].detach().cpu()).all()), f"Non-finite warmup {key}")
        require(bool(torch.isfinite(parameter.detach().cpu()).all()), "Non-finite warmup parameters")
        before_state.append(tuple(id(state[key]) for key in ("step", "exp_avg", "exp_avg_sq")))
    engine.reset_after_warmup(bundle)
    torch.mps.synchronize()
    require(id(bundle["optimizer"]) == optimizer_id and parameter_ids(bundle) == expected_ids, "Reset recreated optimizer or parameters")
    initial = torch.cat([value.detach().reshape(-1) for value in bundle["parameters"]]).cpu().numpy()
    require(initial.tobytes() == initial_bytes, "Reset did not restore exact C initializer bits")
    none_count = zero_count = 0
    step_devices = []
    for index, parameter in enumerate(bundle["parameters"]):
        state = bundle["optimizer"].state[parameter]
        require(tuple(id(state[key]) for key in ("step", "exp_avg", "exp_avg_sq")) == before_state[index], "Reset replaced Adam state buffers")
        require(float(state["step"].detach().cpu().item()) == 0, "Adam step was not reset")
        step_devices.append(str(state["step"].device))
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key].detach().cpu()
            require(bool(torch.isfinite(value).all()) and bool(torch.all(value == 0)), f"Reset did not clear {key}")
        if parameter.grad is None:
            none_count += 1
        else:
            require(bool(torch.all(parameter.grad.detach().cpu() == 0)), "Reset did not clear gradients")
            zero_count += 1
    return {"passed": True, "warmup_optimizer_steps": expected_step, "weights_exact_initial_bits": True,
            "adam_moment1_zero": True, "adam_moment2_zero": True, "adam_step_zero": True,
            "optimizer_recreated": False, "parameter_objects_preserved": True, "state_buffer_objects_preserved": True,
            "cleared_gradient_none_count": none_count, "cleared_gradient_zero_tensor_count": zero_count,
            "actual_step_state_devices": sorted(set(step_devices))}


def input_factory(geometry, config, corpus_path):
    from corpus_fixture import Corpus, synthetic_batches
    if geometry != "full_wiki":
        batches = synthetic_batches(config)
        def reset():
            return iter(batches)
        return reset, {"kind": "C-compatible synthetic four-batch stream"}, None
    corpus = Corpus(corpus_path, config["context_size"], config["vocab_size"])
    require(corpus.windows == 124232535, "Real Wiki window count differs")
    sampler = corpus.sampler(42)
    def reset():
        sampler.reset(42)
        return (sampler.next_batch(config["batch_size"]) for _ in range(4))
    return reset, {"kind": "real retained Wiki affine stream", "corpus": corpus.metadata,
                   "first_windows": sampler.first_samples(), "sampler_initial": sampler.snapshot()}, sampler


def validate_embedding_rows(torch, actual, expected, contexts, config, seen):
    vocab, dim = config["vocab_size"], config["embedding_dim"]
    current = set(int(value) for value in contexts.reshape(-1))
    previous_only = seen - current
    seen.update(current)
    unused = torch.ones(vocab, dtype=torch.bool)
    unused[list(current)] = False
    never = torch.ones(vocab, dtype=torch.bool)
    never[list(seen)] = False
    size = vocab * dim
    for name in ("raw_gradients", "clipped_gradients"):
        require(bool(torch.all(actual[name][:size].reshape(vocab, dim)[unused] == 0)), "Unused embedding gradient is not zero")
    for name in ("moment1", "moment2"):
        require(bool(torch.all(actual[name][:size].reshape(vocab, dim)[never] == 0)), "Never-used embedding moment is not zero")
    prior_momentum = 0
    if previous_only:
        prior_momentum = int(torch.count_nonzero(expected["moment1"][:size].reshape(vocab, dim)[list(previous_only)]))
    return {"unused_embedding_gradients_exact_zero": True, "never_used_embedding_moments_exact_zero": True,
            "previously_used_now_unused_rows": len(previous_only), "reference_prior_momentum_nonzero_elements": prior_momentum,
            "previously_used_rows_compared_in_full_parameter_and_moment_vectors": True}


def check_clip(torch, engine, bundle, case, geometry, config, clip_norm, reset_inputs, input_metadata, sampler, geometry_compiler_before):
    from corpus_fixture import checksum_batch
    initial_bytes = bundle["initial_flat"].tobytes()
    optimizer_id, expected_ids = id(bundle["optimizer"]), parameter_ids(bundle)
    engine.reset_after_warmup(bundle)
    torch.mps.synchronize()
    compile_case = OPTIONS[case]["compiled_model"]
    before = compiler_snapshot(torch) if compile_case else None
    warmups, snapshots = [], []
    warm_inputs = reset_inputs()
    for step in range(2):
        contexts, targets = next(warm_inputs)
        device_contexts = torch.from_numpy(contexts).to(bundle["device"])
        device_targets = torch.from_numpy(targets).to(bundle["device"])
        begun = time.perf_counter()
        live = engine.train_update(bundle, device_contexts, device_targets, clip_norm, capture=False)
        torch.mps.synchronize()
        values = {key: float(value.detach().cpu().item()) for key, value in live.items()}
        require(all(math.isfinite(value) for value in values.values()), "Non-finite warmup scalar")
        warmups.append({"step": step + 1, "synchronized_wall_seconds_including_lazy_compile": time.perf_counter() - begun,
                        "scalars": values})
        if compile_case:
            snapshots.append(compiler_snapshot(torch))
    reset_evidence = reset_and_prove(torch, engine, bundle, initial_bytes, optimizer_id, expected_ids)
    validate_policy(torch, bundle, case)
    reference = oracle.build_reference(torch, config, bundle["initial_flat"], bundle["layout"])
    batches = reset_inputs()
    comparisons, scalar_refs, scalar_cpu, input_checksums, row_semantics, state_steps = [], [], [], [], [], []
    seen = set()
    for step in range(1, 5):
        contexts, targets = next(batches)
        input_checksums.append(checksum_batch(contexts, targets))
        if geometry == "full_wiki" and step == 1:
            require(input_checksums[-1] == "1a55aed564d5b308", "First actual Wiki batch differs from C reference")
        device_contexts = torch.from_numpy(contexts).to(bundle["device"])
        device_targets = torch.from_numpy(targets).to(bundle["device"])
        if OPTIONS[case]["metal_clip"]:
            # Verification-only scratch poison: production kernels must fully
            # overwrite reused partial reductions, including the last tail.
            partials = bundle["_metal_clip_state"]["partials"]
            require(partials.dtype == torch.float32 and partials.device.type == "mps", "Unexpected Metal reduction scratch")
            partials.fill_(float("nan"))
        captured = engine.train_update(bundle, device_contexts, device_targets, clip_norm, capture=True)
        torch.mps.synchronize()
        require(set(captured) == set(FIELDS), "Capture ABI changed")
        for key, value in captured.items():
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.device.type == "mps",
                    f"Actual {key} must remain FP32/MPS")
        for parameter in bundle["parameters"]:
            require(parameter.grad is not None and parameter.grad.layout == torch.strided
                    and not parameter.grad.is_sparse, "Every parameter needs a dense gradient")
        actual = {key: value.detach().cpu() for key, value in captured.items()}
        require(all(bool(torch.isfinite(value).all()) for value in actual.values()), "Non-finite captured model state")
        expected = oracle.reference_update(torch, reference, contexts, targets, clip_norm)
        for key in FIELDS:
            comparisons.append(oracle.compare(torch, f"update{step}.{key}", actual[key], expected[key],
                                               key if key in ("moment1", "moment2") else "ordinary"))
        row_semantics.append(validate_embedding_rows(torch, actual, expected, contexts, config, seen))
        scales = [float(actual["clipping_scale"]), float(expected["clipping_scale"])]
        require(all(0 < scale <= 1 for scale in scales), "Invalid clipping scale")
        require((scales[0] < 1) == (scales[1] < 1), "CPU/device clipping activation differs")
        if clip_norm == 0.05:
            require(scales[0] < 1, "The .05 check did not exercise active clipping")
        # Keep original device scalar references, not clones, to detect a reused
        # Metal stats buffer corrupting later deferred loss/norm/scale history.
        scalar_refs.append({key: captured[key] for key in ("loss", "gradient_norm", "clipping_scale")})
        scalar_cpu.append({key: actual[key].clone() for key in scalar_refs[-1]})
        counters = [float(bundle["optimizer"].state[parameter]["step"].detach().cpu().item())
                    for parameter in bundle["parameters"]]
        require(all(value == step for value in counters), "Actual optimizer update counter differs")
        state_steps.append(counters)
        del actual, expected, captured
    torch.mps.synchronize()
    for original, saved in zip(scalar_refs, scalar_cpu):
        require(all(torch.equal(original[key].detach().cpu(), saved[key]) for key in original),
                "A device scalar buffer was overwritten; deferred history aliases across updates")
    runtime, devices = engine.runtime_metadata(case), engine.actual_devices(bundle, device_contexts)
    oracle.validate_runtime(runtime, "preload_defer_foreach")
    oracle.validate_devices(devices, "mps")
    final_compiler = compiler_snapshot(torch) if compile_case else None
    compilation = None
    if compile_case:
        delta = compiler_delta(before, final_compiler)
        geometry_delta = compiler_delta(geometry_compiler_before, final_compiler)
        require(not any(value > 0 for value in delta["counters"].get("graph_break", {}).values()), "Strict compiled model had a graph break")
        stats = geometry_delta["counters"].get("stats", {})
        inductor = geometry_delta["counters"].get("inductor", {})
        execution_counter_evidence = (stats.get("unique_graphs", 0) > 0
                                      or any(value > 0 for key, value in inductor.items() if "cache_hit" in key or "cache_miss" in key)
                                      or (geometry_delta["generated_kernel_count"] or 0) > 0)
        require(execution_counter_evidence, "Compiled wrapper has no graph/codegen/cache execution evidence")
        compilation = {"before_first_warmup": before, "after_first_warmup": snapshots[0],
                       "after_second_warmup": snapshots[1], "after_four_checked_updates": final_compiler,
                       "delta": delta, "geometry_delta_from_before_setup": geometry_delta,
                       "execution_counter_evidence_verified": execution_counter_evidence,
                       "factory_metadata_after_gate": copy.deepcopy(bundle["optimization_metadata"].get("compile")),
                       "actual_forward_and_backward_completed": True,
                       "counter_scope": "Per-clip snapshots in one geometry's reused wrapper; a later clip may reuse cached compiled graphs without incrementing counters.",
                       "fusion_or_speedup_proven": False}
    return {"geometry": geometry, "clip_norm": clip_norm, "updates": 4, "warmup_updates": 2,
            "status": "complete", "passed": all(row["passed"] for row in comparisons),
            "parameter_count": int(bundle["initial_flat"].size), "config": config,
            "runtime": runtime, "actual_devices": devices, "warmup": warmups, "reset": reset_evidence,
            "optimizer_step_counters": state_steps, "input_checksums": input_checksums,
            "input_metadata": input_metadata, "sampler_after_four_batches": sampler.snapshot() if sampler is not None else None,
            "embedding_row_checks": row_semantics, "dense_gradients_verified": True,
            "deferred_scalar_reference_values_preserved": True, "deferred_scalar_checks": 12,
            "metal_partial_scratch_nan_poisoned_before_checked_updates": OPTIONS[case]["metal_clip"],
            "all_captured_arrays_finite_checked_on_CPU": True, "large_arrays_never_serialized": True,
            "compile_execution_evidence": compilation,
            "comparison_count": len(comparisons), "failed_comparisons": sum(not row["passed"] for row in comparisons),
            "comparisons": comparisons,
            "initialization_and_warmup_scope": "Host setup and synchronized warmups may include lazy compilation; these are not isolated compiler-time or steady-state performance measurements."}


def check_case(case, corpus_path):
    import engine
    engine.configure_environment(case)
    torch = oracle.configure_reference_threads()
    from corpus_fixture import FULL_CONFIG, checksum_f32, parameter_layout
    require(case in engine.CASES, "Engine does not expose the requested case")
    report = {"case": case, "backend": "mps", "status": "running", "passed": False, "checks": [],
              "expected_gate_count": 6, "cache_environment": {key: os.environ.get(key) for key in CACHE_VARIABLES}}
    for geometry in GEOMETRIES:
        config = {**(FULL_CONFIG if geometry == "full_wiki" else oracle.FIXTURES[geometry]),
                  "seed": 42, "threads": 4, "warmup_steps": 2, "steps": 4}
        try:
            setup_counter = compiler_snapshot(torch) if OPTIONS[case]["compiled_model"] else None
            begun = time.perf_counter()
            bundle = engine.build_model_optimizer(config, case)
            torch.mps.synchronize()
            setup_wall = time.perf_counter() - begun
            policy = validate_policy(torch, bundle, case)
            require(bundle["layout"] == parameter_layout(config), "Five-array parameter order/layout differs from C")
            initial = bundle["initial_flat"]
            initial_values = torch.cat([parameter.detach().reshape(-1) for parameter in bundle["parameters"]]).cpu().numpy()
            require(initial_values.tobytes() == initial.tobytes(), "Actual initial parameter bits differ from C")
            require(geometry != "full_wiki" or checksum_f32(initial) == "b5b7b03ca89a72af", "Full model initializer differs from C reference")
            del initial_values
            reset_inputs, input_metadata, sampler = input_factory(geometry, config, corpus_path)
            for clip in (1.0, 0.05):
                try:
                    row = check_clip(torch, engine, bundle, case, geometry, config, clip, reset_inputs, input_metadata, sampler, setup_counter)
                    row.update(policy=policy, initial_parameter_checksum=checksum_f32(initial),
                               synchronized_model_setup_wall_seconds=setup_wall,
                               same_wrapper_reused_for_both_clips=True,
                               geometry_compiler_snapshot_before_setup=setup_counter,
                               optimization_metadata=copy.deepcopy(bundle["optimization_metadata"]))
                except Exception as error:
                    row = {"geometry": geometry, "clip_norm": clip, "updates": 4, "passed": False,
                           "status": "unsupported" if unsupported(error) else "failed",
                           "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                           "optimization_metadata": copy.deepcopy(bundle.get("optimization_metadata", {}))}
                report["checks"].append(row)
                if row["status"] == "unsupported":
                    break
            if report["checks"][-1]["status"] == "unsupported":
                break
        except Exception as error:
            report["checks"].append({"geometry": geometry, "clip_norm": None, "passed": False,
                                     "status": "unsupported" if unsupported(error) else "failed",
                                     "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
            if report["checks"][-1]["status"] == "unsupported":
                break
    failures = any(row["status"] == "failed" or (row["status"] == "complete" and not row["passed"]) for row in report["checks"])
    unsupported_case = any(row["status"] == "unsupported" for row in report["checks"])
    report["passed"] = len(report["checks"]) == 6 and all(row["passed"] for row in report["checks"])
    report["status"] = "complete" if report["passed"] else "failed" if failures else "unsupported" if unsupported_case else "failed"
    report["small_gate_passed"] = all(any(row.get("geometry") == geometry and row.get("clip_norm") == clip and row["passed"]
                                           for row in report["checks"]) for geometry in ("tiny", "irregular") for clip in (1.0, 0.05))
    report["full_size_gate_passed"] = all(any(row.get("geometry") == "full_wiki" and row.get("clip_norm") == clip and row["passed"]
                                               for row in report["checks"]) for clip in (1.0, 0.05))
    report["comparison_count"] = sum(row.get("comparison_count", 0) for row in report["checks"])
    report["failed_comparisons"] = sum(row.get("failed_comparisons", 0) for row in report["checks"])
    report["optimizer_warmup_reset_gate_passed"] = len(report["checks"]) == 6 and all(row.get("reset", {}).get("passed") for row in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training-corpora/wiki-k35328")
    parser.add_argument("--report", type=Path, default=BASE / "verification.json")
    parser.add_argument("--child-timeout", type=float, default=900)
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--child-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_case:
        with oracle.numerical_lock(externally_held=True):
            try:
                result = check_case(args.child_case, args.corpus)
            except Exception as error:
                result = {"case": args.child_case, "status": "unsupported" if unsupported(error) else "failed",
                          "passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["passed"] else 2 if result["status"] == "unsupported" else 1
    if args.report.exists():
        raise FileExistsError(f"Preserving prior optimization gate: {args.report}")
    report = {"kind": "MPS_OPTIMIZATION_INDEPENDENT_CPU_ORACLE_GATES", "status": "running", "passed": False,
              "started_at": datetime.now().astimezone().isoformat(), "cases": {},
              "tolerances_predeclared": oracle.TOLERANCES,
              "scope": "Five candidate paths; tiny/irregular/full Wiki geometries, clip 1/.05, four consecutive updates after two warmups and exact state reset. No throughput or learned-quality claim.",
              "expected_cases": list(dict.fromkeys(args.case or CASES)), "expected_gates_per_case": 6,
              "compile_policy": "Strict model-only Inductor; wrapper creation is not proof of execution, fusion, or speedup. Actual counter snapshots and successful forward/backward are recorded separately.",
              "optimizer_step_state_policy": "Native fused step bookkeeping may be CPU or MPS. Parameters, gradients and moments must remain MPS.",
              "cache_environment": {key: os.environ.get(key) for key in CACHE_VARIABLES}}
    try:
        require(math.isfinite(args.child_timeout) and args.child_timeout > 0, "Child timeout must be finite and positive")
        report["sources_before"] = identities()
        oracle.atomic_json(args.report, report)
        with oracle.numerical_lock(args.lock_held):
            for case in report["expected_cases"]:
                command = [sys.executable, str(Path(__file__).resolve()), "--child-case", case, "--corpus", str(args.corpus.resolve())]
                process = subprocess.Popen(command, cwd=BASE, env=oracle.environment("preload_defer_foreach"), text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    stdout, stderr = process.communicate(timeout=args.child_timeout)
                    result = oracle.parse_json(stdout)
                    require(result.get("case") == case, "Child case mismatch")
                    result.update(command=command, exit_code=process.returncode, stderr=stderr,
                                  child_json_sha256=hashlib.sha256(stdout.encode()).hexdigest())
                    require(not result.get("passed") or process.returncode == 0, "Passing child returned nonzero exit")
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    result = {"case": case, "status": "failed", "passed": False, "timed_out": True,
                              "error": "Candidate numerical child exceeded root-configured timeout", "stderr": stderr,
                              "partial_stdout": stdout, "command": command}
                except Exception as error:
                    result = {"case": case, "status": "failed", "passed": False,
                              "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                              "exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "command": command}
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    raise
                report["cases"][case] = result
                oracle.atomic_json(args.report, report)
            report["sources_after"] = identities()
            require(report["sources_before"] == report["sources_after"], "Optimization engine/helper/verifier changed during gates")
        report["passed"] = bool(report["cases"]) and all(row["passed"] for row in report["cases"].values())
        report["unsupported_cases"] = [name for name, row in report["cases"].items() if row["status"] == "unsupported"]
        report["failed_cases"] = [name for name, row in report["cases"].items() if row["status"] == "failed"]
        report["completed_gates"] = sum(len(row.get("checks", [])) for row in report["cases"].values())
        report["comparison_count"] = sum(row.get("comparison_count", 0) for row in report["cases"].values())
        report["failed_comparisons"] = sum(row.get("failed_comparisons", 0) for row in report["cases"].values())
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["completed_at"] = datetime.now().astimezone().isoformat()
    oracle.atomic_json(args.report, report)
    if args.quiet:
        result = {key: report[key] for key in ("kind", "status", "passed", "unsupported_cases", "failed_cases", "completed_gates", "comparison_count", "failed_comparisons", "error") if key in report}
        result["cases"] = {name: {key: row.get(key) for key in ("status", "passed", "small_gate_passed", "full_size_gate_passed", "comparison_count", "failed_comparisons")}
                           for name, row in report["cases"].items()}
        result["report"] = str(args.report.resolve())
    else:
        result = report
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
