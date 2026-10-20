"""Serial, fixed-work, gated observations of independent non-fused MPS options."""

from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import re
import statistics
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from optimization_round2 import engine
from run_mps_comparison import (
    CPU, WIKI, EPOCH5_TARGETS, _read_gate, atomic_json, capture, identity,
    parse_json, require, work_fingerprint,
)

OUTPUT = WIKI / "reports/mps-combinations-round2-comparison-20261018-v1.json"
GATE = WIKI / "reports/mps-combinations-round2-verification-20261018-v1.json"
C_BINARY = CPU / "experiments/cpu_fixedwork_combined"
STEPS, BATCH = 512, 128


def main():
    require(not OUTPUT.exists(), "Preserve previous optimization timing evidence")
    gate = parse_json(GATE.read_text())
    require(gate["status"] == "complete" and gate["sources_before"] == gate["sources_after"],
            "Completed, source-bound numerical gate required")
    for path, wanted in gate["sources_after"].items():
        require(identity(Path(path)) == {key: wanted[key] for key in ("bytes", "sha256")},
                f"A numerical-gate source changed: {path}")
    require(gate["cases"]["metal_clip"]["passed"], "The existing Metal-clip control must pass")
    cases = ["c_cpu"] + [name for name in engine.CASES if gate["cases"][name]["passed"]]
    c_gate = _read_gate(CPU / "experiments/fixedwork-model-verification.json", "c_cpu_optimized", C_BINARY)
    require(c_gate["passed"], "C reference gate must pass")
    frozen = {str(path.resolve()): identity(path) for path in (
        Path(__file__), BASE / "benchmark.py", BASE / "engine.py", BASE.parent / "optimization/engine_refined.py",
        BASE.parent / "optimization/engine.py", BASE.parent / "optimization/compiled_path.py",
        BASE.parent / "optimization/metal_clip.py", C_BINARY)}
    logs = OUTPUT.parent / (OUTPUT.stem + "-logs")
    require(not logs.exists(), "Preserve prior timing logs")
    logs.mkdir()
    report = {"kind": "MPS_OPTIMIZATION_SAME_WORK_COMPARISON", "status": "running",
              "created_at": datetime.now().astimezone().isoformat(), "source_before": frozen,
              "gate": {"path": str(GATE), **identity(GATE)},
              "gate_cases": {name: {key:value for key,value in row.items() if key in (
                  "status", "passed", "small_gate_passed", "full_size_gate_passed", "comparison_count", "failed_comparisons")}
                             for name,row in gate["cases"].items()},
              "excluded_candidates": {name: row for name,row in gate["cases"].items() if not row["passed"]},
              "prior_fused_failures": "reports/mps-optimization-verification-20261018-v1.json",
              "steps": STEPS, "batch_size": BATCH, "events_per_trial": STEPS*BATCH,
              "planned_order": [], "trials": [], "summary": [],
              "policy": {"same_model_data_initial_bits": True, "same_optimizer": "AdamW foreach=True, fused=False",
                         "dtype": "FP32", "seed": 42, "warmups": 2, "repeats": 3,
                         "scope": "Short real-Wiki timing, full RAM input. Whole-five-epoch time extrapolated, not actually trained.",
                         "setup": "First-update compile+execution and warmup/reset excluded and separately recorded.",
                         "timer": "Input gather/preload/transfer, updates, every-step loss/norm checks and GPU completion included.",
                         "quality": "Numerical agreement is not long-run quality equivalence. Original training remains deferred."}}
    reference = None
    env = dict(os.environ, OMP_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0",
               PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0",
               TORCHINDUCTOR_COMPILE_THREADS="1", MAX_JOBS="1",
               TORCHINDUCTOR_CACHE_DIR=str((BASE.parent / ".artifacts/mps-optimization-inductor").resolve()))
    env.pop("PYTORCH_MPS_PREFER_METAL", None)
    atomic_json(OUTPUT, report)
    try:
        with (CPU / ".timing.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for repeat in (1, 2, 3):
                order = cases if repeat != 2 else list(reversed(cases))
                report["planned_order"].append({"repeat": repeat, "cases": order})
                for name in order:
                    for path, wanted in frozen.items():
                        require(identity(Path(path)) == wanted, f"A frozen source changed: {path}")
                    if name == "c_cpu":
                        command = [str(C_BINARY), "--corpus", str(WIKI / "training-corpora/wiki-k35328"),
                                   "--steps", str(STEPS), "--threads", "4", "--warmup-steps", "2", "--seed", "42",
                                   "--batch-size", str(BATCH), "--clip-norm", "1", "--softmax-parallel",
                                   "--reduction-parallel", "--adam-parallel"]
                    else:
                        command = [str(WIKI / ".venv-training/bin/python"), "-u", str(BASE / "benchmark.py"),
                                   "--case", name, "--steps", str(STEPS)]
                    label = f"repeat-{repeat}-{name}"
                    print(f"{label}: {STEPS} updates / {STEPS*BATCH:,} prediction events", flush=True)
                    before_load = list(os.getloadavg())
                    raw = capture(["/usr/bin/time", "-l", *command], timeout=180 if name != "c_cpu" and engine.CASE_OPTIONS[name]["compiled_model"] else 90,
                                  env=env, cwd=BASE)
                    stdout, stderr = logs/(label+".stdout.json"), logs/(label+".stderr.txt")
                    stdout.write_text(raw["stdout"])
                    stderr.write_text(raw["stderr"])
                    trial = {"case": name, "repeat": repeat, "passed": False, "command": command,
                             "exit_code": raw["exit_code"], "timed_out": raw["timed_out"],
                             "stdout_log": str(stdout), "stderr_log": str(stderr),
                             "process_wall_seconds": raw["process_wall_seconds"],
                             "load_before": before_load, "load_after": list(os.getloadavg())}
                    report["trials"].append(trial)
                    try:
                        require(raw["exit_code"] == 0 and not raw["timed_out"], "Timing child failed or timed out")
                        row = parse_json(raw["stdout"])
                        trial["measurement"] = row
                        require(row.get("status", "complete") == "complete", row.get("error", "Child failed"))
                        require(row["steps"] == STEPS and row["target_prediction_events"] == STEPS*BATCH,
                                "Fixed work differs")
                        for key,value in {"vocab_size":35328,"context_size":8,"embedding_dim":64,"hidden_dim":128,
                                          "batch_size":128,"seed":42,"warmup_steps":2}.items():
                            require(row["config"][key] == value, f"Configuration changed: {key}")
                        require(math.isclose(row["targets_per_second"], STEPS*BATCH/row["elapsed_seconds"], rel_tol=1e-12),
                                "Throughput arithmetic differs")
                        fingerprint = work_fingerprint(row)
                        if reference is None:
                            reference = fingerprint
                        require(fingerprint == reference, "Initial bits or actual first/final input/sampler differs")
                        if name != "c_cpu":
                            require(row["finite_checks_passed"] and row["warmup_full_reset_audited"]
                                    and row["deferred_norm_buffers_distinct"] and row["no_compile_inside_timing"],
                                    "State/timing audit failed")
                            require(row["runtime"]["fallback_env"] == "0" and row["runtime"]["mps_fast_math_env"] == "0"
                                    and row["runtime"]["prefer_metal_env"] is None, "MPS environment changed")
                            require(row["optimization_metadata"]["optimizer"]["fused"] is False
                                    and row["optimization_metadata"]["optimizer"]["foreach"] is True, "Optimizer changed")
                        rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$",raw["stderr"])
                        require(rss is not None, "Missing macOS RSS")
                        trial.update(passed=True, inputs_equal=True, peak_rss_bytes=int(rss.group(1)),
                                     outside_10_30_seconds=not 10 <= row["elapsed_seconds"] <= 30)
                        print(f"  {row['elapsed_seconds']:.6f}s, {row['targets_per_second']:.2f} events/s", flush=True)
                    except (ValueError, KeyError, TypeError) as error:
                        trial["error"] = str(error)
                        print(f"  retained failure: {error}", flush=True)
                    atomic_json(OUTPUT, report)
            for name in cases:
                rows = [trial["measurement"] for trial in report["trials"] if trial["case"] == name and trial["passed"]]
                summary = {"case":name,"successful_repeats":len(rows),"passed":len(rows)==3}
                if len(rows)==3:
                    seconds=[row["elapsed_seconds"] for row in rows]
                    rates=[row["targets_per_second"] for row in rows]
                    hours=[EPOCH5_TARGETS/rate/3600 for rate in rates]
                    summary.update(actual_seconds=seconds,median_seconds=statistics.median(seconds),rates=rates,
                                   median_targets_per_second=statistics.median(rates),
                                   epoch5_hours_extrapolated=statistics.median(hours),epoch5_hours_range=[min(hours),max(hours)])
                report["summary"].append(summary)
            report["source_after"]={path:identity(Path(path)) for path in frozen}
            require(report["source_before"]==report["source_after"],"A source changed during timing")
            report["status"]="complete" if all(row["passed"] for row in report["summary"]) else "complete_with_failures"
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["completed_at"]=datetime.now().astimezone().isoformat()
        atomic_json(OUTPUT,report)


if __name__=="__main__":
    main()
