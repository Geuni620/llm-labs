"""Bounded follow-up timings; frozen numerical engines remain unchanged.

The B128 cases have identical updates and target events. B512 has identical
target events but fewer Adam updates and is a separate training condition.
"""

from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

from run_mps_comparison import (
    CASES, CPU, EPOCH5_TARGETS, WIKI, _read_gate, atomic_json, capture,
    environment, identity, parse_json, require, validate_measurement,
    work_fingerprint,
)


BASE = Path(__file__).resolve().parent
OUTPUT = WIKI / "reports/mps-confirmation-20261018-v1.json"
BATCH_GATE = WIKI / "reports/mps-batch512-verification-20261018-v1.json"
MAIN_GATE = WIKI / "reports/mps-model-verification-20261018-v1.json"
C_GATE = CPU / "experiments/fixedwork-model-verification.json"
C_BINARY = CPU / "experiments/cpu_fixedwork_combined"
PYTHON = WIKI / ".venv-training/bin/python"
WRAPPER = BASE / "confirm_mps.py"
CONDITIONS = {
    "c_cpu_optimized_b128": ("c_cpu_optimized", 128, 512),
    "mps_default_b128": ("mps_default", 128, 512),
    "mps_foreach_b128": ("mps_preload_defer_foreach", 128, 512),
    "mps_foreach_b512": ("mps_preload_defer_foreach", 512, 128),
}


def main():
    require(not OUTPUT.exists(), "Preserve existing confirmation evidence")
    batch_gate = parse_json(BATCH_GATE.read_text())
    require(batch_gate.get("passed") is True, "Independent B512 gate must pass")
    require(batch_gate.get("variant") == "preload_defer_foreach", "Wrong B512 variant")
    require(batch_gate.get("sources_before") == batch_gate.get("sources_after")
            and batch_gate.get("sources_before"), "B512 gate source proof is incomplete")
    for path, value in batch_gate["sources_before"].items():
        require(identity(Path(path)) == {key: value[key] for key in ("bytes", "sha256")},
                f"B512 gate source changed: {path}")
    reset = batch_gate["warmup_reset_verified"]
    require(all(reset.get(key) is True for key in (
        "initial_parameter_bits_restored", "adam_step_zero", "moment1_zero", "moment2_zero",
        "gradients_zero", "sampler_restored")), "B512 warmup/reset verification did not pass")
    sources = [Path(__file__), WRAPPER, BASE / "benchmark_mps.py", BASE / "corpus_fixture.py",
               BASE / "init_reference.c", BASE / "init_reference.dylib", BASE / "verify_mps.py",
               BASE / "verify_batch512.py", BASE / "run_mps_comparison.py", C_BINARY]
    frozen = {str(path.resolve()): identity(path) for path in sources}
    gates = {name: _read_gate(C_GATE if name == "c_cpu_optimized" else MAIN_GATE, name, C_BINARY)
             for name in {condition[0] for condition in CONDITIONS.values()}}
    require(all(gate["passed"] for gate in gates.values()), "A core numerical gate did not pass")
    logs = OUTPUT.parent / (OUTPUT.stem + "-logs")
    require(not logs.exists(), "Preserve existing confirmation logs")
    logs.mkdir()
    report = {"status": "running", "kind": "REAL_WIKI_SHORT_TIMING_CONFIRMATION",
              "created_at": datetime.now().astimezone().isoformat(), "source_before": frozen,
              "gates": gates, "batch512_gate": {"path": str(BATCH_GATE), **identity(BATCH_GATE)},
              "conditions": CONDITIONS, "target_events_per_trial": 65536, "trials": [],
              "summary": [], "policy": {
                  "repeats": 3, "order": "forward/reverse/forward", "dtype": "FP32",
                  "scope": "Short RAM-resident Wiki timings, not a whole epoch or learned-quality result.",
                  "batch512": "Same sampled target stream; 128 Adam updates versus 512 at B128. Different training condition.",
                  "duration": "N selected once from the preserved main timings. Durations outside 10-30s are retained.",
                  "timer": "Frozen engines include gather/transfer/updates/per-step checks; final full-state audit excluded.",
              }}
    reference_b128 = None
    common_stream = None
    try:
        with (CPU / ".timing.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for repeat in (1, 2, 3):
                order = list(CONDITIONS) if repeat != 2 else list(reversed(CONDITIONS))
                for name in order:
                    for path, wanted in frozen.items():
                        require(identity(Path(path)) == wanted, f"Frozen source changed: {path}")
                    case_id, batch, steps = CONDITIONS[name]
                    case = CASES[case_id]
                    if case_id == "c_cpu_optimized":
                        command = [str(C_BINARY), "--corpus", str(WIKI / "training-corpora/wiki-k35328"),
                                   "--steps", str(steps), "--threads", "4", "--warmup-steps", "2",
                                   "--seed", "42", "--batch-size", str(batch), "--clip-norm", "1",
                                   "--softmax-parallel", "--reduction-parallel", "--adam-parallel"]
                    else:
                        command = [str(PYTHON), str(WRAPPER), "--backend", case["backend"],
                                   "--variant", case["variant"], "--steps", str(steps),
                                   "--batch-size", str(batch), "--corpus", str(WIKI / "training-corpora/wiki-k35328")]
                    label = f"repeat-{repeat}-{name}"
                    print(f"{label}: {steps} updates / {steps*batch:,} prediction events", flush=True)
                    load_before = list(os.getloadavg())
                    completed = capture(["/usr/bin/time", "-l", *command], timeout=90,
                                        env=environment(case), cwd=BASE)
                    stdout, stderr = logs / (label + ".stdout.json"), logs / (label + ".stderr.txt")
                    stdout.write_text(completed["stdout"])
                    stderr.write_text(completed["stderr"])
                    trial = {"case": name, "base_case": case_id, "repeat": repeat, "passed": False,
                             "batch_size": batch, "steps": steps, "command": command,
                             "exit_code": completed["exit_code"], "timed_out": completed["timed_out"],
                             "process_wall_seconds": completed["process_wall_seconds"],
                             "stdout_log": str(stdout), "stderr_log": str(stderr),
                             "load_before": load_before, "load_after": list(os.getloadavg())}
                    report["trials"].append(trial)
                    require(completed["exit_code"] == 0 and not completed["timed_out"], "Confirmation child failed")
                    data = parse_json(completed["stdout"])
                    trial["measurement"] = data
                    validate_measurement(data, case_id, steps, batch)
                    fingerprint = work_fingerprint(data)
                    stream = {key: fingerprint[key] for key in (
                        "initial_parameter_checksum", "sampler_initial", "sampler_final", "first_windows", "events")}
                    if common_stream is None:
                        common_stream = stream
                    require(stream == common_stream, "Sampled target stream or initial bits changed")
                    if batch == 128:
                        if reference_b128 is None:
                            reference_b128 = fingerprint
                        require(fingerprint == reference_b128, "B128 fixed work or first/final input differs")
                    rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", completed["stderr"])
                    require(rss is not None, "Missing macOS RSS record")
                    trial.update(passed=True, peak_rss_bytes=int(rss.group(1)),
                                 outside_10_30_seconds=not 10 <= data["elapsed_seconds"] <= 30)
                    atomic_json(OUTPUT, report)
                    print(f"  {data['elapsed_seconds']:.6f}s, {data['targets_per_second']:.2f} events/s", flush=True)
        for name, (_, batch, steps) in CONDITIONS.items():
            rows = [row["measurement"] for row in report["trials"] if row["case"] == name and row["passed"]]
            require(len(rows) == 3, "Every condition needs three observations")
            rates = [row["targets_per_second"] for row in rows]
            seconds = [row["elapsed_seconds"] for row in rows]
            hours = [EPOCH5_TARGETS / rate / 3600 for rate in rates]
            require(all(math.isfinite(value) for value in rates + seconds + hours), "Nonfinite summary")
            report["summary"].append({"case": name, "batch_size": batch, "steps": steps,
                                      "actual_seconds": seconds, "rates": rates,
                                      "median_seconds": statistics.median(seconds),
                                      "median_targets_per_second": statistics.median(rates),
                                      "epoch5_hours_extrapolated": statistics.median(hours),
                                      "min_epoch5_hours": min(hours), "max_epoch5_hours": max(hours)})
        report["source_after"] = {path: identity(Path(path)) for path in frozen}
        require(report["source_before"] == report["source_after"], "Sources changed during confirmation")
        report.update(status="complete", identical_b128_fixed_work=True, identical_sampled_target_stream=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["completed_at"] = datetime.now().astimezone().isoformat()
        atomic_json(OUTPUT, report)


if __name__ == "__main__":
    main()
