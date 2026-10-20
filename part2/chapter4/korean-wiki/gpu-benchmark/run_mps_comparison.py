"""Prepare, or explicitly run, checked fixed-work C/PyTorch CPU/MPS trials.

No Torch import, compilation, package installation, training restart or remote
compute occurs here. Fresh children receive their environment before import.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys


BASE = Path(__file__).resolve().parent
WIKI = BASE.parent
CPU = WIKI / "cpu-benchmark"
sys.path.insert(0, str(CPU / "experiments"))
from run_experiments import THREAD_ENVIRONMENT, capture, identity, parse_json, atomic_json


CASES = {
    "c_cpu_optimized": {"backend": "c", "variant": "combined"},
    "torch_cpu": {"backend": "cpu", "variant": "default", "gate_key": "torch_cpu/default"},
    "mps_default": {"backend": "mps", "variant": "default", "gate_key": "mps/default"},
    "mps_preload_defer": {"backend": "mps", "variant": "preload_defer", "gate_key": "mps/preload_defer"},
    "mps_preload_defer_foreach": {"backend": "mps", "variant": "preload_defer_foreach", "gate_key": "mps/preload_defer_foreach"},
    "mps_preload_defer_foreach_metal": {"backend": "mps", "variant": "preload_defer_foreach_metal", "gate_key": "mps/preload_defer_foreach_metal"},
}
TRAIN_TARGETS = 124232535
EPOCH5_TARGETS = 621162675
ENGINE = BASE / "benchmark_mps.py"
DEFAULT_OUTPUT = WIKI / "reports/mps-comparison-20261018.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment(case: dict) -> dict:
    result = dict(os.environ, **THREAD_ENVIRONMENT,
                  PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0")
    if case["variant"].endswith("_metal"):
        result["PYTORCH_MPS_PREFER_METAL"] = "1"
    else:
        result.pop("PYTORCH_MPS_PREFER_METAL", None)
    return result


def select_steps(pilot: dict, target_seconds=15) -> int:
    require(pilot["steps"] == 4 and pilot["elapsed_seconds"] > 0, "Need a four-update MPS default pilot")
    raw = target_seconds * 4 / pilot["elapsed_seconds"]
    return max(8, min(64, int(math.floor(raw / 4 + .5)) * 4))


def work_fingerprint(row: dict) -> dict:
    return {key: row[key] for key in ("initial_parameter_checksum", "input_checksum", "final_input_checksum",
                                     "sampler_initial", "sampler_final")} | {
        "first_windows": row.get("first_windows", row.get("corpus", {}).get("first_windows")),
        "steps": row["steps"], "events": row["target_prediction_events"]}


def validate_measurement(row: dict, case_id: str, steps: int, batch_size=128) -> None:
    case = CASES[case_id]
    require(row["steps"] == steps and row["target_prediction_events"] == steps * batch_size,
            "All main cases must complete exactly the same updates and prediction events")
    for key, expected in {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64, "hidden_dim": 128,
                          "batch_size": batch_size, "seed": 42, "warmup_steps": 2}.items():
        require(row["config"].get(key) == expected, f"Model/data configuration changed: {key}")
    require(row["elapsed_seconds"] > 0 and math.isclose(row["targets_per_second"], steps * batch_size / row["elapsed_seconds"], rel_tol=1e-9),
            "Invalid synchronized end-to-end rate")
    require(row.get("warmup_reset") is True, "Weights/optimizer/sampler must reset after warmup")
    require(row["sampler_initial"]["consumed"] == 0 and row["sampler_final"]["consumed"] == steps * batch_size,
            "Sampler start/end differs from the fixed-work definition")
    require(row["process_cpu_seconds"] >= 0, "Missing timed process CPU resource measurement")
    if case["backend"] == "c":
        require(row["kind"] == "REAL_WIKI_C_MICROBENCHMARK" and row.get("finite_flag") is True,
                "C optimized reference must be the frozen finite real-Wiki benchmark")
        require(all(row["features"].get(feature) is True for feature in ("fixed_norm", "softmax_inv_batch", "minimal_zero")),
                "The C reference must use the checked combined settings")
    else:
        require(row["kind"] == "REAL_WIKI_TORCH_CPU_MPS_FIXED_WORK" and row.get("status") == "complete"
                and row["backend"] == case["backend"] and row["variant"] == case["variant"], "Wrong child backend/policy")
        require(row.get("finite_checks_passed") is True and row["dtype"] == "float32", "FP32 and complete finite checks are required")
        runtime = row["runtime"]
        require(runtime["fallback_env"] == "0" and runtime["mps_fast_math_env"] == "0", "CPU fallback or MPS fast math changed")
        require(runtime.get("prefer_metal_env") == ("1" if case["variant"].endswith("_metal") else None),
                "Metal preference must be absent for controls and present only for the explicit candidate")
        require(runtime["cpu_threads"] == 4 and runtime["cpu_interop_threads"] == 1, "CPU thread policy changed")
        require(len(row["losses"]) == steps and len(row["gradient_norms"]) == steps, "Every step loss/norm must be retained and checked")
        for key in ("parameters", "inputs", "gradients", "moments"):
            devices = row["actual_devices"][key]
            require(devices and all(device.startswith(case["backend"]) for device in devices), "Wrong actual device or hidden CPU fallback")


def _read_gate(path: Path, case_id: str, c_binary: Path) -> dict:
    report = parse_json(path.read_text())
    if case_id == "c_cpu_optimized":
        record = report["variants"]["combined"]
        require(report.get("passed") is True and record.get("passed") is True
                and record.get("multiupdate_poison_passed") is True, "C combined numerical gate did not pass")
        require(identity(c_binary) == {key: record["executable_after"][key] for key in ("bytes", "sha256")}, "C binary differs from its gate")
        return {"path": str(path), **identity(path), "status": "complete", "passed": True, "record": record}
    record = report["cases"][CASES[case_id]["gate_key"]]
    require(record["backend"] == CASES[case_id]["backend"] and record["variant"] == CASES[case_id]["variant"], "Wrong gate backend/variant")
    source_records = record.get("sources", report.get("sources_before", {}))
    require(isinstance(source_records, dict) and source_records, "Numerical gate must bind actual engine/helper source hashes")
    verified_sources = {}
    for key, value in source_records.items():
        require(isinstance(value, dict) and "sha256" in value and "bytes" in value, "Incomplete numerical source identity")
        filename = value.get("path", key)
        require(isinstance(filename, str) and Path(filename).is_absolute(), "Numerical source path must be explicit")
        wanted = {field: value[field] for field in ("bytes", "sha256")}
        require(identity(Path(filename)) == wanted, f"Numerical-gate source changed: {filename}")
        verified_sources[filename] = wanted
    require(all(str(filename.resolve()) in verified_sources for filename in (ENGINE, BASE / "corpus_fixture.py", BASE / "init_reference.c")),
            "Gate proof must include this engine and both exact-C fixture sources")
    if record.get("status") == "unsupported":
        return {"path": str(path), **identity(path), "status": "unsupported", "passed": False,
                "record": record, "verified_sources": verified_sources}
    require(record.get("status") == "complete" and record.get("passed") is True
            and record.get("small_gate_passed") is True and record.get("full_size_gate_passed") is True,
            "Both four-update small and real-size one-step numerical gates must pass")
    return {"path": str(path), **identity(path), "status": "complete", "passed": True,
            "record": record, "verified_sources": verified_sources}


def summarize(trials: list[dict], names: list[str]) -> list[dict]:
    result = []
    for name in names:
        rows = [trial for trial in trials if trial["case"] == name and trial.get("passed")]
        if len(rows) != 3:
            result.append({"case": name, "status": "not_complete", "passed": False, "repeats": len(rows)})
            continue
        rates = [row["measurement"]["targets_per_second"] for row in rows]
        times = [row["measurement"]["elapsed_seconds"] for row in rows]
        median_rate = statistics.median(rates)
        result.append({"case": name, "status": "complete", "passed": True, "repeats": 3,
                       "elapsed_seconds": times, "targets_per_second": rates,
                       "median_elapsed_seconds": statistics.median(times), "min_elapsed_seconds": min(times), "max_elapsed_seconds": max(times),
                       "median_targets_per_second": median_rate, "min_targets_per_second": min(rates), "max_targets_per_second": max(rates),
                       "same_case_max_min_rate_ratio": max(rates) / min(rates),
                       "process_cpu_seconds": [row["measurement"]["process_cpu_seconds"] for row in rows],
                       "extrapolation_only": {"epoch5_prediction_events": EPOCH5_TARGETS,
                                               "epoch5_hours": EPOCH5_TARGETS / median_rate / 3600,
                                               "per_trial_epoch5_hours": [EPOCH5_TARGETS / rate / 3600 for rate in rates],
                                               "caveat": "Short synchronized end-to-end wall throughput only; not a measured full training run, convergence or learned-quality result."}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gate", action="append", default=[], help="caseID=passed verifier JSON; same all-case report may be reused")
    parser.add_argument("--python", type=Path, default=WIKI / ".venv-training/bin/python")
    parser.add_argument("--c-binary", type=Path, default=CPU / "experiments/cpu_fixedwork_combined")
    parser.add_argument("--corpus", type=Path, default=WIKI / "training-corpora/wiki-k35328")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, help="Optional explicit same N, 8..64 and a multiple of four")
    parser.add_argument("--target-seconds", type=float, default=15)
    args = parser.parse_args()
    require(sys.platform == "darwin", "This comparison targets the local Mac")
    require(args.steps is None or (8 <= args.steps <= 64 and args.steps % 4 == 0), "Common steps must be 8..64 and divisible by four")
    require(10 <= args.target_seconds <= 30, "Pilot target must remain bounded")
    output = args.output.absolute()
    artifact = output if args.run else output.with_name(output.stem + ".prepare.json")
    require(not artifact.exists(), f"Preserving existing evidence: {artifact}")
    output.parent.mkdir(parents=True, exist_ok=True)
    gate_paths = {}
    for value in args.gate:
        name, separator, path = value.partition("=")
        require(separator and name in CASES and path and name not in gate_paths, "Use one caseID=PATH per gate")
        gate_paths[name] = Path(path).absolute()
    sources = [ENGINE, BASE / "corpus_fixture.py", BASE / "init_reference.c", Path(__file__).resolve(),
               CPU / "experiments/run_experiments.py"]
    frozen = {str(path): identity(path) for path in sources}
    c_binary = args.c_binary.absolute()
    c_identity = identity(c_binary)
    report = {"kind": "REAL_WIKI_LOCAL_CPU_MPS_FIXED_WORK_COMPARISON", "status": "preparing",
              "created_at": datetime.now().astimezone().isoformat(), "source_before": frozen,
              "c_binary": {"path": str(c_binary), **c_identity}, "gates": {}, "gate_errors": {},
              "planned_cases": CASES, "main_steps": args.steps, "batch_size": 128, "repeats": 3,
              "trial_order": [], "pilot": None, "trials": [], "summary": [],
              "policy": {"dtype": "float32", "optimizer": "AdamW", "seed": 42, "cpu_threads": 4, "cpu_interop_threads": 1,
                         "warmups": 2, "fallback": False, "mps_fast_math": False, "fresh_child_per_trial": True,
                         "preload_transfer_scope": "Batch gathering/preloading and transfers are inside the synchronized timed window.",
                         "finite_scope": "Every update loss/norm checked; defer variant changes detection timing only. Final full-state finite audit is separate outside timer.",
                         "optional_batch512": "Not performed automatically; requires a later separate root-approved numerical gate and experiment.",
                         "quality": "Small numerical agreement and short-run rates do not establish equal long-run learned quality."}}
    report["data_identity_evidence"] = []
    for filename in ("mps-helper-build-20261018.json", "mps-corpus-reference-verification.json", "mps-parent-audit-20261018.json"):
        proof = WIKI / "reports" / filename
        if proof.exists():
            report["data_identity_evidence"].append({"path": str(proof), **identity(proof),
                                                      "report": parse_json(proof.read_text())})
    for name, path in gate_paths.items():
        try:
            report["gates"][name] = _read_gate(path, name, c_binary)
            for filename, wanted in report["gates"][name].get("verified_sources", {}).items():
                require(filename not in frozen or frozen[filename] == wanted, "Gate sources disagree across cases")
                frozen[filename] = wanted
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["gate_errors"][name] = str(error)
    report["missing_gates"] = [name for name in CASES if name not in gate_paths]
    if not args.run:
        report["status"] = "prepared"
        report["model_or_compiler_executed"] = False
        atomic_json(artifact, report)
        print(json.dumps({"status": "prepared", "report": str(artifact), "missing_gates": report["missing_gates"], "gate_errors": report["gate_errors"]}))
        return
    logs = output.parent / (output.stem + "-logs")
    require(not logs.exists(), "Preserving existing logs")
    logs.mkdir()
    progress = output.with_name(output.stem + ".progress.json")
    reference = None

    def save():
        atomic_json(output, report)
        atomic_json(progress, {"status": report["status"], "main_steps": report["main_steps"], "trials": len(report["trials"]),
                               "last_case": report["trials"][-1]["case"] if report["trials"] else None, "error": report.get("error")})

    def unchanged():
        for path, wanted in frozen.items():
            require(identity(Path(path)) == wanted, f"Frozen engine/helper/coordinator source changed: {path}")
        require(identity(c_binary) == c_identity, "Frozen optimized C executable changed")
        for gate in report["gates"].values():
            require(identity(Path(gate["path"])) == {key: gate[key] for key in ("bytes", "sha256")}, "Numerical gate evidence changed")

    def execute(name, steps, repeat, pilot=False):
        nonlocal reference
        unchanged()
        case = CASES[name]
        if name == "c_cpu_optimized":
            command = [str(c_binary), "--corpus", str(args.corpus.absolute()), "--steps", str(steps), "--threads", "4",
                       "--warmup-steps", "2", "--seed", "42", "--batch-size", "128", "--clip-norm", "1",
                       "--softmax-parallel", "--reduction-parallel", "--adam-parallel"]
        else:
            command = [str(args.python.absolute()), str(ENGINE), "--backend", case["backend"], "--variant", case["variant"],
                       "--corpus", str(args.corpus.absolute()), "--steps", str(steps), "--batch-size", "128",
                       "--threads", "4", "--warmup-steps", "2", "--seed", "42", "--clip-norm", "1"]
        label = f"{'pilot' if pilot else f'repeat-{repeat}'}-{name}"
        print(f"{label}: {steps} updates / {steps * 128:,} prediction events", flush=True)
        before_load = list(os.getloadavg())
        completed = capture(["/usr/bin/time", "-l", *command], timeout=90, env=environment(case), cwd=BASE)
        stdout, stderr = logs / (label + ".stdout.json"), logs / (label + ".stderr.txt")
        stdout.write_text(completed["stdout"], encoding="utf-8")
        stderr.write_text(completed["stderr"], encoding="utf-8")
        record = {"case": name, "repeat": repeat, "passed": False, "command": command,
                  "process_wall_seconds": completed["process_wall_seconds"], "exit_code": completed["exit_code"], "timed_out": completed["timed_out"],
                  "stdout_log": {"path": str(stdout), **identity(stdout)}, "stderr_log": {"path": str(stderr), **identity(stderr)},
                  "load_before": before_load, "load_after": list(os.getloadavg())}
        try:
            require(not completed["timed_out"], "Child process group exceeded its 90-second cap")
            data = parse_json(completed["stdout"])
            record["measurement"] = data
            require(completed["exit_code"] == 0 and data.get("status", "complete") == "complete", "Checked child failed or became unsupported during timing")
            validate_measurement(data, name, steps)
            key = work_fingerprint(data)
            if not pilot:
                if reference is None:
                    reference = key
                require(key == reference, "C/CPU/MPS initial bits or first/final input/sampler work differs")
            rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", completed["stderr"])
            require(rss is not None, "macOS time -l RSS missing")
            record.update(passed=True, inputs_equal=True, peak_rss_bytes=int(rss.group(1)),
                          outside_10_30_seconds=not 10 <= data["elapsed_seconds"] <= 30)
        except (ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
        if pilot:
            report["pilot"] = record
        else:
            report["trials"].append(record)
        save()
        require(record["passed"], f"Trial failed: {name}: {record.get('error')}")
        print(f"  {record['measurement']['elapsed_seconds']:.6f}s, {record['measurement']['targets_per_second']:.2f} events/s", flush=True)
        return record["measurement"]

    def canceled(signum, frame):
        raise KeyboardInterrupt(f"Canceled by signal {signum}")
    signal.signal(signal.SIGTERM, canceled)
    report["status"] = "running"
    save()
    try:
        require(not report["missing_gates"] and not report["gate_errors"], "All planned cases need explicit numerical gate results")
        require(report["gates"]["mps_default"].get("passed") is True, "A numerically validated MPS default is required for the common-N pilot")
        supported = [name for name in CASES if report["gates"][name].get("passed")]
        report["skipped"] = {name: gate for name, gate in report["gates"].items() if not gate["passed"]}
        with (CPU / ".timing.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.steps is None:
                pilot = execute("mps_default", 4, 0, pilot=True)
                steps = select_steps(pilot, args.target_seconds)
            else:
                steps = args.steps
            report["main_steps"], report["prediction_events_per_trial"] = steps, steps * 128
            save()
            for repeat in (1, 2, 3):
                order = supported if repeat != 2 else list(reversed(supported))
                report["trial_order"].append({"repeat": repeat, "cases": order})
                for name in order:
                    execute(name, steps, repeat)
            unchanged()
            report["summary"] = summarize(report["trials"], supported)
            require(all(row["passed"] for row in report["summary"]), "Not every supported case completed three observations")
            report["source_after"] = {path: identity(Path(path)) for path in frozen}
            report["status"] = "complete_with_unsupported_cases" if report["skipped"] else "complete"
    except BaseException as error:
        report["status"] = "canceled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["completed_at"] = datetime.now().astimezone().isoformat()
        report["successful_trials"] = sum(record["passed"] for record in report["trials"])
        save()


if __name__ == "__main__":
    main()
