"""Diagnostic synchronized phase timings, never a training-speed forecast.

Phase barriers perturb execution. Root owns the shared numerical lock and
launches this script alone with a bounded process deadline.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
import benchmark_mps as baseline
from corpus_fixture import Corpus, FULL_CONFIG, checksum_batch, checksum_f32
from verify_mps import identity, numerical_lock


def profile():
    baseline.configure_environment("preload_defer_foreach")
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = {**FULL_CONFIG, "threads": 4, "steps": 8, "warmup_steps": 2}
    corpus = Corpus(baseline.WIKI / "training-corpora/wiki-k35328", 8, 35328)
    sampler = corpus.sampler(42)
    bundle = baseline.build_model_optimizer(config, "mps", "preload_defer_foreach")
    for _ in range(2):
        contexts, targets = sampler.next_batch(128)
        baseline.train_update(bundle, torch.from_numpy(contexts).to("mps"),
                              torch.from_numpy(targets).to("mps"), 1)
    baseline.reset_after_warmup(bundle)
    sampler.reset(42)
    batches = [sampler.next_batch(128) for _ in range(8)]
    contexts = torch.from_numpy(np.stack([row[0] for row in batches])).to("mps")
    targets = torch.from_numpy(np.stack([row[1] for row in batches])).to("mps")
    torch.mps.synchronize()
    phase_times, host_calls, losses, norms = {}, {}, [], []

    def phase(name, function):
        torch.mps.synchronize()
        begun = time.perf_counter()
        value = function()
        host_calls.setdefault(name, []).append(time.perf_counter() - begun)
        torch.mps.synchronize()
        phase_times.setdefault(name, []).append((time.perf_counter() - begun)*1000)
        return value

    model, optimizer = bundle["model"], bundle["optimizer"]
    begun = time.perf_counter()
    for index in range(8):
        phase("gradient_clear", lambda: optimizer.zero_grad(set_to_none=False))
        hidden = phase("embedding_hidden_tanh", lambda: torch.tanh(model.hidden(
            model.embedding(contexts[index]).reshape(128, -1))))
        logits = phase("output_projection", lambda: model.output(hidden))
        loss = phase("cross_entropy", lambda: torch.nn.functional.cross_entropy(logits, targets[index]))
        phase("backward", loss.backward)

        def clip():
            norm = torch.nn.utils.clip_grad_norm_(bundle["parameters"], 1.0,
                                                  foreach=False, error_if_nonfinite=False)
            scale = torch.clamp(1.0 / (norm.detach() + 1e-6), max=1.0)
            return norm, scale

        norm, scale = phase("global_norm_clip", clip)
        phase("adamw_foreach", optimizer.step)
        losses.append(loss.detach())
        norms.append(norm.detach())
    torch.mps.synchronize()
    elapsed = time.perf_counter() - begun
    streams = phase_times
    loss_values = torch.stack(losses).cpu().tolist()
    norm_values = torch.stack(norms).cpu().tolist()
    if not all(math.isfinite(x) for x in loss_values + norm_values):
        raise ValueError("Nonfinite diagnostic update")
    baseline._check_final_finite(bundle)
    total_event_ms = sum(sum(row) for row in streams.values())
    phases = [{"phase": name, "synchronized_wall_ms": streams[name],
               "median_synchronized_wall_ms": statistics.median(streams[name]),
               "total_synchronized_wall_ms": sum(streams[name]),
               "phase_wall_fraction": sum(streams[name])/total_event_ms,
               "host_call_ms": [value*1000 for value in host_calls[name]],
               "median_host_call_ms": statistics.median(host_calls[name])*1000}
              for name in streams]
    return {"status": "complete", "config": config, "phases": phases,
            "instrumented_total_wall_seconds": elapsed, "sum_phase_wall_ms": total_event_ms,
            "first_input_checksum": checksum_batch(*batches[0]),
            "initial_parameter_checksum": checksum_f32(bundle["initial_flat"]),
            "runtime": baseline.runtime_metadata(), "actual_devices": baseline.actual_devices(bundle, contexts),
            "losses": loss_values, "gradient_norms": norm_values,
            "scope": "Diagnostic 8 real-Wiki updates with synchronize before/after every phase. Forced barriers perturb overlap and include dispatch/wait costs; not pure kernel busy time, GPU utilization, or the uninstrumented throughput benchmark. CPU call durations may contain implicit waits.",
            "input_scope": "Full corpus RAM loaded; eight preloaded B128 batches. Data preload excluded only for this phase diagnostic, not the main benchmark."}


def main():
    output = baseline.WIKI / "reports/mps-optimization-phase-profile-20261018-v3.json"
    if output.exists():
        raise FileExistsError("Preserve earlier phase profile")
    paths = [Path(__file__), Path(baseline.__file__), BASE.parent / "corpus_fixture.py"]
    report = {"kind": "MPS_SYNCHRONIZED_PHASE_DIAGNOSTIC", "status": "running",
              "started_at": datetime.now().astimezone().isoformat(),
              "source_before": {str(p): identity(p) for p in paths}}
    try:
        with numerical_lock(externally_held=True):
            report.update(profile())
        report["source_after"] = {str(p): identity(p) for p in paths}
        if report["source_after"] != report["source_before"]:
            raise ValueError("A source changed during phase profiling")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["completed_at"] = datetime.now().astimezone().isoformat()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key in ("status","error","phases","instrumented_total_wall_seconds")},ensure_ascii=False),flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
