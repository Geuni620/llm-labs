"""One bounded, independently gated MPS optimization observation.

Caller owns the shared numerical lock and process timeout. Compilation/setup,
warmups/reset and final all-array audits are outside the timed updates.
"""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from optimization import engine_refined as engine
from corpus_fixture import Corpus, FULL_CONFIG, checksum_batch, checksum_f32
from verify_mps import numerical_lock


def compile_snapshot():
    from torch._dynamo.utils import counters, guard_failures
    from torch._inductor import metrics
    groups = ("stats", "frames", "inductor", "aot_autograd", "graph_break", "unimplemented")
    return {"counters": {name: {str(key): int(value) for key, value in counters[name].items()}
                         for name in groups},
            "guard_failures": sum(len(value) for value in guard_failures.values()),
            "generated_kernel_count": int(metrics.generated_kernel_count)}


def compiler_work(snapshot):
    counters = snapshot["counters"]
    return (counters["stats"].get("unique_graphs", 0),
            counters["aot_autograd"].get("total", 0), snapshot["guard_failures"],
            snapshot["generated_kernel_count"], counters["inductor"], counters["graph_break"])


def run(args):
    engine.configure_environment(args.case)
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = {**FULL_CONFIG, "steps": args.steps, "warmup_steps": 2, "threads": 4}
    corpus = Corpus(args.corpus, 8, 35328)
    if corpus.windows != 124232535:
        raise ValueError("The retained target pool changed")
    sampler = corpus.sampler(42)
    bundle = engine.build_model_optimizer(config, args.case)
    warmup_seconds, warmup_values = [], []
    for _ in range(2):
        contexts, targets = sampler.next_batch(128)
        x, y = torch.from_numpy(contexts).to("mps"), torch.from_numpy(targets).to("mps")
        torch.mps.synchronize()
        begun = time.perf_counter()
        live = engine.train_update(bundle, x, y, 1)
        torch.mps.synchronize()
        warmup_seconds.append(time.perf_counter() - begun)
        values = {key: float(value.item()) for key, value in live.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite warmup")
        warmup_values.append(values)
    engine.reset_after_warmup(bundle)
    sampler.reset(42)
    initial_sampler = sampler.snapshot()
    first_windows = sampler.first_samples(4)
    actual_initial = torch.cat([p.detach().reshape(-1) for p in bundle["parameters"]]).cpu().numpy()
    if actual_initial.tobytes() != bundle["initial_flat"].tobytes():
        raise ValueError("Warmup did not restore the exact C parameter bits")
    del actual_initial
    for parameter in bundle["parameters"]:
        state = bundle["optimizer"].state[parameter]
        if float(state["step"].item()) != 0:
            raise ValueError("Adam step not reset")
        for key in ("exp_avg", "exp_avg_sq"):
            if not bool(torch.all(state[key].detach().cpu() == 0)):
                raise ValueError("Adam moment not reset")
        if bundle["zero_grad_set_to_none"]:
            if parameter.grad is not None:
                raise ValueError("None-gradient reset policy changed")
        elif parameter.grad is None or not bool(torch.all(parameter.grad.detach().cpu() == 0)):
            raise ValueError("Dense gradient not reset")
    before_compile = compile_snapshot()
    if args.case == "compiled_model":
        counters = before_compile["counters"]
        inductor = counters["inductor"]
        if (counters["stats"].get("unique_graphs", 0) < 1 or counters["graph_break"]
                or counters["aot_autograd"].get("ok", 0) < 1
                or not (before_compile["generated_kernel_count"] > 0
                        or inductor.get("fxgraph_cache_hit", 0) > 0)):
            raise ValueError("No complete compiled graph execution evidence")
        bundle["optimization_metadata"]["compile"].update(
            status="forward_backward_executed_and_counter_evidence_verified",
            graph_support_verified=True,
            actual_execution_evidence_scope="Successful forward/backward plus Dynamo/AOT/Inductor/codegen or compiled-cache counters, MPS tensors and fallback disabled.",
        )
        from torch._dynamo import config as dynamo_config
        dynamo_config.error_on_recompile = True
    losses, norms, scales = [], [], []
    torch.mps.synchronize()
    cpu_before = time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    begun = time.perf_counter()
    batches = [sampler.next_batch(128) for _ in range(args.steps)]
    xs = torch.from_numpy(np.stack([batch[0] for batch in batches])).to("mps")
    ys = torch.from_numpy(np.stack([batch[1] for batch in batches])).to("mps")
    preparation_host_seconds = time.perf_counter() - begun
    for index in range(args.steps):
        live = engine.train_update(bundle, xs[index], ys[index], 1)
        losses.append(live["loss"])
        norms.append(live["gradient_norm"])
        scales.append(live["clipping_scale"])
    loss_values = torch.stack(losses).cpu().tolist()
    norm_values = torch.stack(norms).cpu().tolist()
    if not all(math.isfinite(value) for value in loss_values + norm_values):
        raise ValueError("Nonfinite loss or norm in at least one timed update")
    torch.mps.synchronize()
    elapsed = time.perf_counter() - begun
    cpu_seconds = time.process_time() - cpu_before
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    after_compile = compile_snapshot()
    if args.case == "compiled_model" and compiler_work(before_compile) != compiler_work(after_compile):
        raise ValueError("Compilation, guard failure or graph break occurred inside the timed window")
    audit_begun = time.perf_counter()
    engine.original.frozen._check_final_finite(bundle)
    scale_values = torch.stack(scales).cpu().tolist()
    if not all(math.isfinite(value) and 0 < value <= 1 for value in scale_values):
        raise ValueError("Invalid clipping scale")
    if len({value.data_ptr() for value in norms}) != args.steps:
        raise ValueError("Deferred norm history aliases a reused result buffer")
    audit_seconds = time.perf_counter() - audit_begun
    devices = engine.actual_devices(bundle, xs)
    for key in ("parameters", "inputs", "gradients", "moments"):
        if not devices[key] or any(not value.startswith("mps") for value in devices[key]):
            raise ValueError("The actual learning tensors are not MPS")
    steps = [float(bundle["optimizer"].state[p]["step"].item()) for p in bundle["parameters"]]
    if steps != [float(args.steps)]*5:
        raise ValueError("Actual Adam update counts differ")
    return {"kind": "REAL_WIKI_MPS_OPTIMIZATION_FIXED_WORK", "status": "complete", "case": args.case,
            "config": config, "steps": args.steps, "target_prediction_events": args.steps*128,
            "elapsed_seconds": elapsed, "targets_per_second": args.steps*128/elapsed,
            "process_cpu_seconds": cpu_seconds, "rusage": {
                "user_seconds": usage_after.ru_utime-usage_before.ru_utime,
                "system_seconds": usage_after.ru_stime-usage_before.ru_stime},
            "dtype": "float32", "warmup_reset": True, "warmup_full_reset_audited": True,
            "warmup_seconds": warmup_seconds, "warmup_scalars": warmup_values,
            "warmup_timing_scope": "First actual update may include compilation plus execution; not pure compiler time.",
            "corpus": corpus.metadata, "first_windows": first_windows,
            "initial_parameter_checksum": checksum_f32(bundle["initial_flat"]),
            "input_checksum": checksum_batch(*batches[0]), "final_input_checksum": checksum_batch(*batches[-1]),
            "sampler_initial": initial_sampler, "sampler_final": sampler.snapshot(),
            "actual_devices": devices, "actual_adam_step_values": steps,
            "runtime": engine.runtime_metadata(args.case), "optimization_metadata": bundle["optimization_metadata"],
            "losses": loss_values, "gradient_norms": norm_values, "clipping_scales": scale_values,
            "finite_checks_passed": True, "clip_active_updates": sum(value < 1 for value in scale_values),
            "deferred_norm_buffers_distinct": True,
            "compile_before_timed": before_compile, "compile_after_timed": after_compile,
            "no_compile_inside_timing": args.case != "compiled_model" or compiler_work(before_compile) == compiler_work(after_compile),
            "host_preparation_and_transfer_observed_seconds": preparation_host_seconds,
            "final_full_state_audit_seconds": audit_seconds,
            "memory_snapshot_after_audit": {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                                             "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                                             "scope": "End snapshot, not peak; unified memory must not be added to RSS."},
            "timing_scope": "Synchronized wall includes real batch gather/preload/transfer, every update and all loss/norm readbacks/checks. Setup/warmups/reset/compile/final full-state audit excluded.",
            "quality_scope": "Short timing and independent numerical agreement do not establish long-run quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=engine.CASES, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--corpus", type=Path, default=engine.original.frozen.WIKI / "training-corpora/wiki-k35328")
    args = parser.parse_args()
    if not 8 <= args.steps <= 1024:
        parser.error("Bounded observation requires 8..1024 updates")
    result = {"status": "failed", "case": args.case}
    try:
        with numerical_lock(externally_held=True):
            result = run(args)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    main()
