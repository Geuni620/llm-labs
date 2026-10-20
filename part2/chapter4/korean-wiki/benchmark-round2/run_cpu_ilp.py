"""Same-condition ILP trials plus explicitly separate batch-size experiments.

All cases consume 65,536 real Wiki targets. B256/B512 have fewer Adam updates;
their timing cannot establish equal learning quality to B128.
"""

from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

WIKI = Path(__file__).resolve().parent.parent
EXP = WIKI / "cpu-benchmark/experiments"
sys.path.insert(0, str(WIKI / "gpu-benchmark"))
from run_mps_comparison import capture, parse_json, atomic_json, identity, require, work_fingerprint

REPORT = WIKI / "reports/cpu-ilp-round2-comparison-20261018-v1.json"
BUILD = WIKI / "reports/cpu-ilp-round2-build-20261018-v1.json"
GATE = WIKI / "reports/cpu-ilp-round2-verification-20261018-v1.json"
UNIT = WIKI / "reports/cpu-ilp-round2-unit-20261018-v1.json"
CASES = {
    "baseline_t4": (EXP / "cpu_fixedwork_combined", 4, 128),
    "ilp_t4": (EXP / "round2/cpu_fixedwork_ilp", 4, 128),
    "baseline_t6": (EXP / "cpu_fixedwork_combined", 6, 128),
    "ilp_t6": (EXP / "round2/cpu_fixedwork_ilp", 6, 128),
    "batch256_t6": (EXP / "cpu_fixedwork_combined", 6, 256),
    "batch512_t6": (EXP / "cpu_fixedwork_combined", 6, 512),
}
TARGETS = 124232535
EVENTS = 65536


def main():
    require(not REPORT.exists(), "Preserve prior measurements")
    gate, unit, build = [json.loads(p.read_text()) for p in (GATE, UNIT, BUILD)]
    require(gate["passed"] and gate["failed_comparisons"] == 0, "ILP independent model checks must pass")
    require(unit["finite_gate_passed"] and unit["nonfinite_probe"]["bitwise_identical"], "ILP bitwise unit gate must pass")
    require(build["status"] == "complete", "New build manifest required")
    require(identity(CASES["ilp_t4"][0]) == {k: build["binary"][k] for k in ("bytes", "sha256")},
            "Candidate binary changed after build")
    expected_binary = gate["variants"]["combined"]["executable_after"]
    require(identity(CASES["ilp_t4"][0]) == {k: expected_binary[k] for k in ("bytes", "sha256")},
            "Candidate binary changed after numerical verification")
    sources = [Path(__file__), BUILD, GATE, UNIT, *{v[0] for v in CASES.values()},
               *(EXP / "round2").glob("*.c"), *EXP.glob("*.c"), *EXP.glob("*.h")]
    before = {str(p): identity(p) for p in sources}
    names = list(CASES)
    orders = [names, list(reversed(names)), names[2:] + names[:2]]
    logs = REPORT.parent / (REPORT.stem + "-logs")
    require(not logs.exists(), "Preserve prior logs")
    logs.mkdir()
    report = {"kind": "CPU_ILP_SAME_CONDITION_AND_BATCH_EXPERIMENTS", "status": "running",
              "started_at": datetime.now().astimezone().isoformat(), "sources_before": before,
              "events_per_trial": EVENTS, "epoch_target_windows": TARGETS, "planned_order": orders,
              "trials": [], "summary": [],
              "policy": {"timing": "Serial real Wiki RAM input; full reset; finite audit. No full training.",
                         "same_condition": "B128 ILP and baseline compared at matching thread budget; byte-identical end weights/moments required.",
                         "batch_experiments": "B256/B512 change optimizer update count and batch statistics. Throughput only; quality not established.",
                         "epoch_hours": "Short-throughput extrapolation, excludes evaluation/checkpoints/long-run thermal changes.",
                         "batch_numeric_scope": "Existing C engine; new bitwise softmax units include B256/B512 actual V35328; full-state finiteness checked. No independent full-model B256/B512 reference gate."}}
    atomic_json(REPORT, report)
    fingerprints = {}
    sampling_reference = None
    with (WIKI / "cpu-benchmark/.timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for repeat, order in enumerate(orders, 1):
            for name in order:
                binary, threads, batch = CASES[name]
                steps = EVENTS // batch
                env = dict(os.environ, **{key: str(threads) for key in (
                    "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
                env.pop("OMP_THREAD_LIMIT", None)
                command = [str(binary), "--corpus", str(WIKI / "training-corpora/wiki-k35328"),
                           "--steps", str(steps), "--threads", str(threads), "--warmup-steps", "2",
                           "--seed", "42", "--batch-size", str(batch), "--clip-norm", "1",
                           "--softmax-parallel", "--reduction-parallel", "--adam-parallel"]
                label = f"repeat-{repeat}-{name}"
                print(label, flush=True)
                raw = capture(["/usr/bin/time", "-l", *command], timeout=120, env=env, cwd=WIKI)
                (logs / (label + ".stdout.json")).write_text(raw["stdout"])
                (logs / (label + ".stderr.txt")).write_text(raw["stderr"])
                trial = {"case": name, "repeat": repeat, "threads": threads, "batch": batch,
                         "steps": steps, "command": command, "passed": False,
                         "exit_code": raw["exit_code"], "process_wall_seconds": raw["process_wall_seconds"],
                         "stdout_log": str(logs / (label + ".stdout.json")),
                         "stderr_log": str(logs / (label + ".stderr.txt"))}
                report["trials"].append(trial)
                try:
                    require(raw["exit_code"] == 0 and not raw["timed_out"], "Timing child failed")
                    row = parse_json(raw["stdout"])
                    require(row["steps"] == steps and row["target_prediction_events"] == EVENTS, "Actual work differs")
                    require(row["config"]["batch_size"] == batch and row["requested_threads"] == threads,
                            "Actual batch/thread budget differs")
                    require(row["finite_flag"] and row["warmup_reset"] and not row["fast_math_build"], "State/FP32 audit failed")
                    require(row["corpus"]["target_windows"] == TARGETS, "Corpus changed")
                    require(row["parameter_count"] == 6883968 and all(row["features"].values()), "Model/optimizations changed")
                    require(math.isclose(row["targets_per_second"], EVENTS / row["elapsed_seconds"], rel_tol=1e-12),
                            "Throughput arithmetic differs")
                    fingerprint = work_fingerprint(row)
                    if batch not in fingerprints:
                        fingerprints[batch] = fingerprint
                    require(fingerprint == fingerprints[batch], "Same-B initialization/first/final batches/sampler differ")
                    sampling = {key: row[key] for key in ("initial_parameter_checksum", "sampler_initial", "sampler_final")}
                    if sampling_reference is None:
                        sampling_reference = sampling
                    require(sampling_reference == sampling, "Cross-batch initialization or target stream endpoints differ")
                    trial.update(passed=True, measurement=row,
                                 outside_10_30_seconds=not 10 <= row["elapsed_seconds"] <= 30)
                    print(f"  {row['elapsed_seconds']:.6f}s / {row['targets_per_second']:.2f} targets/s", flush=True)
                except Exception as error:
                    trial["error"] = f"{type(error).__name__}: {error}"
                    print(trial["error"], flush=True)
                atomic_json(REPORT, report)
        for name, (_, threads, batch) in CASES.items():
            rows = [r["measurement"] for r in report["trials"] if r["case"] == name and r["passed"]]
            item = {"case": name, "threads": threads, "batch_size": batch,
                    "same_training_conditions_as_B128": batch == 128,
                    "passed": len(rows) == 3, "successful_repeats": len(rows)}
            if len(rows) == 3:
                seconds = [r["elapsed_seconds"] for r in rows]
                rate = statistics.median(r["targets_per_second"] for r in rows)
                item.update(actual_seconds=seconds, median_seconds=statistics.median(seconds),
                            median_targets_per_second=rate, epoch2_hours=2 * TARGETS / rate / 3600,
                            epoch3_hours=3 * TARGETS / rate / 3600,
                            epoch5_hours=5 * TARGETS / rate / 3600,
                            epoch5_observed_range=[5 * TARGETS * min(seconds) / EVENTS / 3600,
                                                   5 * TARGETS * max(seconds) / EVENTS / 3600])
            report["summary"].append(item)
        checksums = {}
        for name in CASES:
            checksums[name] = sorted({tuple(r["measurement"][key] for key in (
                "final_parameter_checksum", "final_moment1_checksum", "final_moment2_checksum"))
                for r in report["trials"] if r["case"] == name and r["passed"]})
        bit_checks = {str(t): len(checksums[f"baseline_t{t}"]) == 1 and
                      checksums[f"baseline_t{t}"] == checksums[f"ilp_t{t}"] for t in (4, 6)}
        report.update(final_checksums_by_case=checksums, ilp_final_state_bit_identical=bit_checks,
                      batch_fingerprints=fingerprints, shared_sampler_and_initialization=sampling_reference)
        report["sources_after"] = {p: identity(Path(p)) for p in before}
        require(before == report["sources_after"], "Source/binary changed during timing")
        report.update(status="complete" if all(r["passed"] for r in report["summary"]) and all(bit_checks.values())
                      else "complete_with_failures", completed_at=datetime.now().astimezone().isoformat())
        atomic_json(REPORT, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "ilp_final_state_bit_identical": bit_checks}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
