"""Short, serial thread-budget comparison; no full training or source rebuild.

The unchanged combined C executable and independent numerical checker are
bound by hashes. Every trial performs the same 512 updates after full reset.
"""

from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
WIKI = HERE.parent
CPU = WIKI / "cpu-benchmark"
sys.path.insert(0, str(WIKI / "gpu-benchmark"))
from run_mps_comparison import capture, parse_json, atomic_json, identity, work_fingerprint, require

BINARY = CPU / "experiments/cpu_fixedwork_combined"
GATE = WIKI / "reports/cpu-thread-round2-verification-20261018-v1.json"
OUTPUT = WIKI / "reports/cpu-thread-round2-comparison-20261018-v1.json"
TARGETS = 124232535
STEPS = 512


def main():
    require(not OUTPUT.exists(), "Preserve existing timing results")
    gate = parse_json(GATE.read_text())
    require(gate["passed"] and gate["passed_cases"] == 24 and gate["failed_comparisons"] == 0,
            "All thread-budget correctness cases must pass before timing")
    require(gate["variants"]["combined"]["executable_before"] ==
            gate["variants"]["combined"]["executable_after"], "Verification binary changed")
    require(identity(BINARY) == {k: gate["variants"]["combined"]["executable_after"][k]
                                 for k in ("bytes", "sha256")}, "Verified binary changed")
    sources = [Path(__file__), BINARY, GATE,
               *(CPU / "experiments").glob("*.c"), *(CPU / "experiments").glob("*.h")]
    frozen = {str(path.resolve()): identity(path) for path in sources}
    logs = OUTPUT.parent / (OUTPUT.stem + "-logs")
    require(not logs.exists(), "Preserve previous logs")
    logs.mkdir()
    report = {"kind": "CPU_THREAD_BUDGET_SAME_WORK_COMPARISON", "status": "running",
              "started_at": datetime.now().astimezone().isoformat(), "sources_before": frozen,
              "steps": STEPS, "events_per_trial": STEPS * 128,
              "epoch_target_windows": TARGETS,
              "order": [[4, 6, 8], [8, 4, 6], [6, 8, 4]], "trials": [], "summary": [],
              "policy": "Same FP32 model/data/initialization/B128/optimizer/clipping. Only OpenMP and BLAS requested thread budget changes. Accelerate internal worker count and UI responsiveness are not measured. Full-epoch hours are extrapolations; long training remains deferred."}
    atomic_json(OUTPUT, report)
    reference = None
    with (CPU / ".timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for repeat, order in enumerate(report["order"], 1):
            for threads in order:
                name = f"repeat-{repeat}-threads-{threads}"
                env = dict(os.environ, **{key: str(threads) for key in (
                    "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", OMP_DYNAMIC="FALSE",
                    OMP_MAX_ACTIVE_LEVELS="1", OMP_THREAD_LIMIT=str(threads))
                command = [str(BINARY), "--corpus", str(WIKI / "training-corpora/wiki-k35328"),
                           "--steps", str(STEPS), "--threads", str(threads), "--warmup-steps", "2",
                           "--seed", "42", "--batch-size", "128", "--clip-norm", "1",
                           "--softmax-parallel", "--reduction-parallel", "--adam-parallel"]
                print(name, flush=True)
                raw = capture(["/usr/bin/time", "-l", *command], timeout=120, env=env, cwd=HERE)
                (logs / (name + ".stdout.json")).write_text(raw["stdout"])
                (logs / (name + ".stderr.txt")).write_text(raw["stderr"])
                trial = {"threads": threads, "repeat": repeat, "command": command,
                         "exit_code": raw["exit_code"], "passed": False,
                         "process_wall_seconds": raw["process_wall_seconds"],
                         "stdout_log": str(logs / (name + ".stdout.json")),
                         "stderr_log": str(logs / (name + ".stderr.txt"))}
                report["trials"].append(trial)
                try:
                    require(raw["exit_code"] == 0 and not raw["timed_out"], "Child failed or timed out")
                    row = parse_json(raw["stdout"])
                    require(row["steps"] == STEPS and row["target_prediction_events"] == STEPS * 128,
                            "Completed work differs")
                    for key, value in {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64,
                                       "hidden_dim": 128, "batch_size": 128, "seed": 42,
                                       "warmup_steps": 2, "threads": threads}.items():
                        require(row["config"][key] == value, f"Configuration changed: {key}")
                    require(row["finite_flag"] and row["warmup_reset"] and not row["fast_math_build"],
                            "Finite/reset/FP32 checks failed")
                    require(row["corpus"]["target_windows"] == TARGETS, "Corpus windows changed")
                    require(all(row["features"].values()), "Existing optimizations changed")
                    require(math.isclose(row["targets_per_second"], STEPS * 128 / row["elapsed_seconds"],
                                         rel_tol=1e-12), "Throughput arithmetic differs")
                    fingerprint = work_fingerprint(row)
                    if reference is None:
                        reference = fingerprint
                    require(reference == fingerprint, "Actual initialization/input/sampler differs")
                    rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", raw["stderr"])
                    require(rss is not None, "RSS not recorded")
                    trial.update(passed=True, measurement=row, peak_rss_bytes=int(rss.group(1)),
                                 outside_10_30_seconds=not 10 <= row["elapsed_seconds"] <= 30)
                    print(f"  {row['elapsed_seconds']:.6f}s / {row['targets_per_second']:.2f} targets/s", flush=True)
                except Exception as error:
                    trial["error"] = f"{type(error).__name__}: {error}"
                    print(trial["error"], flush=True)
                atomic_json(OUTPUT, report)
        for threads in (4, 6, 8):
            rows = [r["measurement"] for r in report["trials"] if r["threads"] == threads and r["passed"]]
            summary = {"threads": threads, "successful_repeats": len(rows), "passed": len(rows) == 3}
            if summary["passed"]:
                seconds = [r["elapsed_seconds"] for r in rows]
                rate = statistics.median(r["targets_per_second"] for r in rows)
                summary.update(actual_seconds=seconds, median_seconds=statistics.median(seconds),
                               median_targets_per_second=rate, epoch3_hours=3 * TARGETS / rate / 3600,
                               epoch5_hours=5 * TARGETS / rate / 3600,
                               epoch5_observed_range=[5 * TARGETS / max(r["targets_per_second"] for r in rows) / 3600,
                                                      5 * TARGETS / min(r["targets_per_second"] for r in rows) / 3600])
            report["summary"].append(summary)
        report["sources_after"] = {path: identity(Path(path)) for path in frozen}
        require(frozen == report["sources_after"], "A source changed during timing")
        report.update(status="complete" if all(r["passed"] for r in report["summary"]) else "failed",
                      completed_at=datetime.now().astimezone().isoformat(), identical_work_fingerprint=reference)
        atomic_json(OUTPUT, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
