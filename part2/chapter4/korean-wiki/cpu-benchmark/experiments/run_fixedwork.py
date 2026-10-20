"""Prepare or explicitly run checked fixed-work Mac CPU comparisons.

This coordinator never compiles a binary. Root supplies five independently
checked executables and their gate reports. The default is --prepare-only;
only --run executes pilots or benchmark children. Every main trial uses the
same step count, selected once before the four local ABBA blocks.
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

from run_experiments import THREAD_ENVIRONMENT, atomic_json, capture, identity, parse_json


BASE = Path(__file__).resolve().parent
BENCHMARK = BASE.parent
WIKI = BENCHMARK.parent
CANDIDATES = ("fixed_norm", "softmax_inv_batch", "minimal_zero", "combined")
VARIANTS = ("baseline", *CANDIDATES)
FEATURES = {
    "baseline": {"fixed_norm": False, "softmax_inv_batch": False, "minimal_zero": False},
    "fixed_norm": {"fixed_norm": True, "softmax_inv_batch": False, "minimal_zero": False},
    "softmax_inv_batch": {"fixed_norm": False, "softmax_inv_batch": True, "minimal_zero": False},
    "minimal_zero": {"fixed_norm": False, "softmax_inv_batch": False, "minimal_zero": True},
    "combined": {"fixed_norm": True, "softmax_inv_batch": True, "minimal_zero": True},
}
FIXED_CONFIG = {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64,
                "hidden_dim": 128, "batch_size": 128, "seed": 42,
                "threads": 4, "warmup_steps": 2}
PHASES = ("ram_gather", "matrix_multiply", "softmax", "bias_reduce", "gradient_clip", "adamw")
PHASE_CALLS = {"ram_gather": 1, "matrix_multiply": 6, "softmax": 1,
               "bias_reduce": 2, "gradient_clip": 1, "adamw": 1}
OMP_ALL = ("--softmax-parallel", "--reduction-parallel", "--adam-parallel")
TRAIN_TARGETS = 124232535
EPOCH5_TARGETS = 621162675
TIMEOUT_SECONDS = 60
DEFAULT_OUTPUT = BENCHMARK / "results/cpu-fixedwork-mac-20261018-v1.json"
SOURCE_NAMES = ("fixedwork_benchmark.c", "fixedwork_api.h", "fixedwork_norm.c",
                "fixedwork_softmax.c", "vector_vforce.c", "gemm_kernels.c",
                "experiment_kernels.h", "fast_exp.h", "check_fixedwork.py",
                "run_fixedwork.py", "run_experiments.py")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition, message) -> None:
    if not condition:
        raise ValueError(message)


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, filename = value.partition("=")
        require(separator and name in VARIANTS and filename, f"Use variant=PATH for one of {VARIANTS}")
        require(name not in result, f"Duplicate variant path: {name}")
        result[name] = Path(filename).absolute()
    return result


def select_main_steps(pilots: list[dict], target_seconds=15.0, min_steps=128, max_steps=4096) -> dict:
    """Use the last successful baseline pilot once; never retune by candidate."""
    require(bool(pilots), "A successful baseline pilot is required")
    last = pilots[-1].get("measurement", pilots[-1])
    seconds, steps = last["elapsed_seconds"], last["steps"]
    require(seconds > 0 and type(steps) is int and steps > 0, "Invalid baseline pilot timing")
    raw = int(math.floor(target_seconds * steps / seconds + 0.5))
    selected = max(min_steps, min(max_steps, raw))
    return {"steps": selected, "method": "last successful baseline fixed-step pilot, nearest integer then common bounds",
            "pilot_steps": steps, "pilot_elapsed_seconds": seconds,
            "target_baseline_seconds": target_seconds, "unbounded_steps": raw,
            "min_steps": min_steps, "max_steps": max_steps,
            "predicted_baseline_seconds": selected * seconds / steps,
            "candidate_specific_retuning": False}


def work_fingerprint(data: dict) -> dict:
    """Final parameter/moment checksums are intentionally absent from this key."""
    return {key: data[key] for key in (
        "initial_parameter_checksum", "input_checksum", "final_input_checksum",
        "sampler_initial", "sampler_final",
    )} | {"first_windows": data["corpus"]["first_windows"],
         "steps": data["steps"], "target_prediction_events": data["target_prediction_events"]}


def validate_measurement(data: dict, variant: str, steps: int) -> dict:
    require(variant in FEATURES, "Unknown fixed-work variant")
    require(data.get("kind") == "REAL_WIKI_C_MICROBENCHMARK", "Real Wiki timing output is required")
    require(data.get("fixed_work") is True and data.get("poison_gradients") is False and data.get("check_updates") == 0,
            "Timing must use the production fixed-work path, not a poisoned correctness fixture")
    for key, expected in FIXED_CONFIG.items():
        require(data["config"].get(key) == expected, f"Changed common model configuration: {key}")
    require(data["config"].get("steps") == steps and data["config"].get("seconds") == 0,
            "Fixed steps only: elapsed-duration mode must remain disabled")
    require(data["steps"] == steps and data["completed_steps"] == steps
            and data["target_prediction_events"] == steps * 128, "Completed work differs from the fixed N")
    require(data.get("requested_seconds", 0) == 0, "Elapsed-duration work is not comparable here")
    require(data.get("finite_flag") is True and data.get("warmup_reset") is True,
            "Finite result and complete warmup reset are required")
    require(data.get("gradient_max_norm") == 1 and data.get("fast_math_build") is False,
            "Norm-1 clipping and strict global FP flags are required")
    require(data["corpus"].get("target_windows") == TRAIN_TARGETS
            and data["corpus"].get("owned_ram_bytes", 0) > 0, "Unexpected RAM-resident Wiki source")
    require("vvexpf" in data.get("exponential_implementation", ""), "Every variant must use CPU vForce")
    for key, value in FEATURES[variant].items():
        require(data["features"].get(key) is value, f"Incorrect compiled feature: {key}")
    kernel = data["kernel_options"]
    require(all(kernel.get(key) is True for key in ("softmax_parallel", "reduction_parallel", "adam_parallel"))
            and kernel.get("tiled_gemm") == 0 and kernel.get("fast_exp") is False,
            "Common vForce/OpenMP/CBLAS kernel policy changed")
    omp = data["openmp"]
    require(omp.get("compiled") is True and omp.get("probe_team_size") == 4
            and omp.get("nested_disabled") is True, "OpenMP must be compiled, nonnested, with an actual four-thread probe")
    wall = data["elapsed_seconds"]
    process_cpu = data["process_cpu_seconds"]
    require(wall > 0 and process_cpu >= 0, "Invalid compute wall/process CPU time")
    require(math.isclose(data["targets_per_second"], steps * 128 / wall, rel_tol=1e-10), "Incorrect target throughput")
    require(data.get("thread_clock_available") in {True, False}, "Thread CPU availability must be explicit")
    if data["thread_clock_available"]:
        require(isinstance(data.get("thread_cpu_seconds"), (int, float)) and data["thread_cpu_seconds"] >= 0,
                "Invalid main-thread CPU measurement")
    else:
        require(data.get("thread_cpu_seconds") is None, "Unavailable thread CPU time must be null")
    usage = data["rusage"]
    require(usage["user_seconds"] >= 0 and usage["system_seconds"] >= 0, "Invalid timed-loop getrusage CPU delta")
    phases = data["phase_wall_seconds"]
    require(all(key in phases and phases[key] >= 0 for key in (*PHASES, "other")), "Invalid aggregate phase timers")
    tolerance = max(1e-6, wall * 1e-7)
    require(sum(phases[key] for key in PHASES) <= wall + tolerance, "Phase timers overlap or exceed elapsed wall time")
    require(math.isclose(sum(phases.values()), wall, rel_tol=1e-7, abs_tol=1e-6), "Aggregate residual phase does not reconcile")
    for key, calls in PHASE_CALLS.items():
        require(data["phase_calls"].get(key) == calls * steps, f"Incorrect fixed-work phase call count: {key}")
    updates = data["per_update"]
    for key in ("wall_seconds", "gradient_norm", "clipping_scale", "clip_active"):
        require(isinstance(updates[key], list) and len(updates[key]) == steps, f"Incomplete per-update {key}")
    require(all(value >= 0 for value in updates["gradient_norm"])
            and all(0 < value <= 1 for value in updates["clipping_scale"]), "Invalid per-update clipping evidence")
    require(all(type(value) is bool for value in updates["clip_active"]), "Clip-active evidence must use booleans")
    require(all(active == (scale < 1) for active, scale in zip(updates["clip_active"], updates["clipping_scale"])),
            "Clip-active count disagrees with clipping scales")
    per_phases = updates["phase_wall_seconds"]
    for key in (*PHASES, "other"):
        require(isinstance(per_phases[key], list) and len(per_phases[key]) == steps
                and all(value >= 0 for value in per_phases[key]), f"Incomplete per-update phase: {key}")
        if key != "other":
            require(math.isclose(sum(per_phases[key]), phases[key], rel_tol=2e-5, abs_tol=tolerance),
                    f"Per-update and aggregate phases differ: {key}")
        else:
            require(sum(per_phases[key]) <= phases[key] + tolerance,
                    "Per-update residuals exceed the overall residual, including loop bookkeeping")
    for index, update_wall in enumerate(updates["wall_seconds"]):
        require(update_wall >= 0 and sum(per_phases[key][index] for key in PHASES) <= update_wall + tolerance,
                "Per-update measured phases exceed the update wall time")
        require(math.isclose(sum(per_phases[key][index] for key in (*PHASES, "other")), update_wall,
                             rel_tol=1e-7, abs_tol=1e-6), "Per-update phase residual does not reconcile")
    require(sum(updates["wall_seconds"]) <= wall + tolerance, "Per-update wall sum exceeds the timed loop")
    bookkeeping = wall - sum(updates["wall_seconds"])
    require(math.isclose(phases["other"] - sum(per_phases["other"]), bookkeeping,
                         rel_tol=1e-5, abs_tol=tolerance), "Aggregate residual does not account for outer record bookkeeping")
    require(data["sampler_initial"]["consumed"] == 0 and data["sampler_final"]["consumed"] == steps * 128,
            "Warmup sampler was not reset, or the measured stream consumed different work")
    for key in ("initial_parameter_checksum", "input_checksum", "final_input_checksum", "final_parameter_checksum",
                "final_moment1_checksum", "final_moment2_checksum"):
        require(isinstance(data.get(key), str) and re.fullmatch(r"[0-9a-f]{16}", data[key]), f"Missing audit checksum: {key}")
    return {"passed": True, "fixed_work_inputs": work_fingerprint(data), "phase_accounting_passed": True,
            "clip_active_updates": sum(updates["clip_active"]), "clip_active_percent": sum(updates["clip_active"]) / steps * 100,
            "median_update_wall_seconds": statistics.median(updates["wall_seconds"]),
            "outer_loop_bookkeeping_wall_seconds": max(0, wall - sum(updates["wall_seconds"])),
            "aggregate_cpu_cores_during_timed_loop": process_cpu / wall,
            "process_cpu_scope": "Timed-loop aggregate CPU consumption of the process and its workers; not wall duration.",
            "final_parameter_checksum_equal_required": False}


def summarize_abba(trials: list[dict], events: int) -> list[dict]:
    output = []
    for candidate in CANDIDATES:
        rows = sorted((row for row in trials if row["block"] == candidate), key=lambda row: row["position"])
        entry = {"candidate": candidate, "status": "not_complete", "passed": False}
        if len(rows) != 4 or any(not row.get("passed") for row in rows):
            entry["completed_trial_ids"] = [row["trial_id"] for row in rows]
            output.append(entry)
            continue
        require([row["variant"] for row in rows] == ["baseline", candidate, candidate, "baseline"], "ABBA order changed")
        controls, variants = [rows[0], rows[3]], [rows[1], rows[2]]
        baseline_times = [row["measurement"]["elapsed_seconds"] for row in controls]
        control_elapsed = statistics.mean(baseline_times)
        control_cpu = statistics.mean(row["measurement"]["process_cpu_seconds"] for row in controls)
        baseline_rates = [row["measurement"]["targets_per_second"] for row in controls]
        times = [row["measurement"]["elapsed_seconds"] for row in variants]
        cpu = [row["measurement"]["process_cpu_seconds"] for row in variants]
        rates = [row["measurement"]["targets_per_second"] for row in variants]
        rate_drift = abs(baseline_rates[1] - baseline_rates[0]) / baseline_rates[0] * 100
        elapsed_drift = abs(baseline_times[1] - baseline_times[0]) / baseline_times[0] * 100
        per_trial = [{"trial_id": row["trial_id"], "time_speedup": control_elapsed / seconds,
                      "wall_seconds_saved": control_elapsed - seconds,
                      "wall_saved_percent": (1 - seconds / control_elapsed) * 100,
                      "process_cpu_saved_percent": (1 - row["measurement"]["process_cpu_seconds"] / control_cpu) * 100
                      if control_cpu else None} for row, seconds in zip(variants, times)]
        savings = [row["wall_saved_percent"] for row in per_trial]
        median_rate = statistics.median(rates)
        entry.update({
            "status": "complete", "passed": True,
            "control_trial_ids": [row["trial_id"] for row in controls],
            "candidate_trial_ids": [row["trial_id"] for row in variants],
            "control_elapsed_seconds": control_elapsed, "control_endpoint_elapsed_seconds": baseline_times,
            "control_effective_targets_per_second": events / control_elapsed,
            "control_endpoint_targets_per_second": baseline_rates,
            "control_endpoint_rate_arithmetic_mean": statistics.mean(baseline_rates),
            "control_process_cpu_seconds": control_cpu,
            "control_rate_drift_percent": rate_drift, "control_elapsed_drift_percent": elapsed_drift,
            "control_drift_percent": max(rate_drift, elapsed_drift),
            "high_control_drift": max(rate_drift, elapsed_drift) > 20,
            "candidate_elapsed_seconds": times, "candidate_process_cpu_seconds": cpu,
            "candidate_targets_per_second": rates, "candidate_median_elapsed_seconds": statistics.median(times),
            "candidate_min_elapsed_seconds": min(times), "candidate_max_elapsed_seconds": max(times),
            "candidate_median_targets_per_second": median_rate,
            "candidate_min_targets_per_second": min(rates), "candidate_max_targets_per_second": max(rates),
            "per_trial": per_trial, "wall_saved_percent_min": min(savings), "wall_saved_percent_max": max(savings),
            "wall_saved_percent_median": statistics.median(savings),
            "process_cpu_saved_percent_median": statistics.median(row["process_cpu_saved_percent"] for row in per_trial)
            if control_cpu else None,
            "savings_method": "Each B is compared with arithmetic mean elapsed of its two surrounding A controls; range is the two B savings, not a 2x2 cross-product or interpolated control.",
            "control_rate_method": "Primary control rps = events/mean(A elapsed), the harmonic mean of endpoint rps. Endpoint arithmetic mean rps is descriptive only.",
            "drift_method": "max(abs(A2_rps-A1_rps)/A1_rps, abs(A2_time-A1_time)/A1_time)*100; >20% flags variability without deleting observations.",
            "performance_improvement_proven": False,
            "extrapolation_only": {
                "epoch5_targets": EPOCH5_TARGETS,
                "candidate_epoch5_hours_per_trial": [EPOCH5_TARGETS / rate / 3600 for rate in rates],
                "candidate_epoch5_hours_from_median_rate": EPOCH5_TARGETS / median_rate / 3600,
                "control_epoch5_hours": EPOCH5_TARGETS / (events / control_elapsed) / 3600,
                "caveat": "Short-run wall-throughput arithmetic only; no epoch or learned-quality result. Aggregate process CPU seconds are not used in wall-hour projections.",
            },
        })
        output.append(entry)
    return output


def process_snapshot() -> dict:
    """Inspect names/CPU usage, retaining no arbitrary unrelated command args."""
    completed = capture(["ps", "-axo", "pid=,ppid=,pcpu=,comm="], timeout=5)
    require(not completed["timed_out"] and completed["exit_code"] == 0, "Cannot inspect competing processes")
    rows, known = [], []
    own_pid = os.getpid()
    for line in completed["stdout"].splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid, parent, cpu, command = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
        if pid == own_pid:
            continue
        row = {"pid": pid, "parent_pid": parent, "cpu_percent": cpu, "executable": command}
        rows.append(row)
        basename = Path(command).name.lower()
        if basename.startswith(("cpu_fixedwork_", "cpu_experiment_", "cpu_vector_", "cpu_norm_simd_")):
            known.append({**row, "classification": "another CPU benchmark"})
        elif "python" in basename:
            args = capture(["ps", "-p", str(pid), "-o", "args="], timeout=5)
            # Inspect only to identify model/benchmark scripts. Never log generic
            # argument strings, which may contain unrelated application secrets.
            pattern = r"(?:^|[ /])((?:train[^ /]*|run_(?:experiments|vector_exp|norm_simd|cumulative|benchmarks)|check_(?:fixedwork|variants))\.py)(?:\s|$)"
            match = re.search(pattern, args["stdout"])
            if match:
                known.append({**row, "classification": "another model/benchmark/checker worker", "script": match.group(1)})
    return {"captured_at": now(), "load_average": list(os.getloadavg()),
            "top_cpu_processes": sorted(rows, key=lambda row: -row["cpu_percent"])[:20],
            "known_training_or_benchmark_processes": known,
            "policy": "No processes are terminated. Known concurrent training/checker/benchmark work aborts the study; unrelated host activity is recorded, not silently removed."}


def _source_rows(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [{"path": path, **row} for path, row in value.items()]
    return []


def validate_gate_scope(report: dict, variant: str) -> dict:
    """Require the full predeclared 80-case Cartesian gate, not true flags alone."""
    require(report.get("kind") == "FIXEDWORK_MODEL_INDEPENDENT_TORCH_VERIFICATION"
            and report.get("status") == "complete" and report.get("passed") is True, "A complete fixed-work model gate is required")
    require(set(report.get("variants", {})) == set(VARIANTS), "All five build variants must share the full model gate")
    specification = report["validation_spec"]
    require(len(specification["fixtures"]) == 2 and set(specification["fixtures"]) == {"tiny", "irregular"}
            and len(specification["threads"]) == 2 and set(specification["threads"]) == {1, 4}
            and len(specification["clip_norms"]) == 2 and set(specification["clip_norms"]) == {1.0, 0.05},
            "Reduced fixture, thread or clipping scope cannot authorize timing")
    require(specification.get("expected_cases_per_variant") == 16 and specification.get("expected_total_cases") == 80,
            "Exactly 16 model checks per build and 80 total are required")
    scenarios = specification["scenarios"]
    require(len(scenarios) == 2 and {(row["check_updates"], row["poison_gradients"]) for row in scenarios.values()}
            == {(1, False), (4, True)}, "Both clean one-step and four-update poison scenarios are required")
    require(specification.get("runtime_kernel_arguments") == list(OMP_ALL), "Gate did not use the common OpenMP kernel options")
    cases = report["cases"]
    require(len(cases) == 80 and report.get("passed_cases") == 80 and report.get("failed_cases") == 0
            and report.get("failed_comparisons") == 0, "Not all 80 predeclared numerical cases passed")
    tolerances = report["tolerances_predeclared"]
    require(tolerances.get("declared_before_execution") is True, "Numerical tolerances must be fixed before checking")
    for family, atol, rtol in (("ordinary_loss_norm_scale_gradients_parameters", 2e-6, 2e-4),
                              ("moment1", 2e-9, 2e-4), ("moment2", 2e-11, 5e-4)):
        require(tolerances[family] == {"atol": atol, "rtol": rtol}, f"Changed numerical tolerance: {family}")
    expected = {(fixture, threads, clip, updates, poison)
                for fixture in ("tiny", "irregular") for threads in (1, 4) for clip in (1.0, 0.05)
                for updates, poison in ((1, False), (4, True))}
    all_indices = []
    for name, selected in report["variants"].items():
        indices = selected["case_indices"]
        require(len(indices) == 16 and len(set(indices)) == 16
                and all(type(index) is int and 0 <= index < 80 for index in indices), "Gate case indices are incomplete or duplicated")
        observed = set()
        for index in indices:
            row = cases[index]
            require(row.get("feature") == name and row.get("passed") is True, "Gate case assigned to wrong feature or failed")
            observed.add((row["fixture"], row["threads"], row["clip_norm"], row["check_updates"], row["poison_gradients"]))
        require(observed == expected, f"Incomplete clean/poison Cartesian coverage: {name}")
        require(selected.get("passed") is True and selected.get("passed_cases") == 16, f"Variant model gate failed: {name}")
        all_indices.extend(indices)
    require(sorted(all_indices) == list(range(80)), "Model cases must cover all five variants exactly once")
    same = report["same_work_validation"]
    require(same.get("identical_initial_parameter_bits") is True
            and same.get("identical_input_sequences_for_matching_fixture_and_scenario") is True,
            "Gate input/initial-parameter identity proof is missing")
    if report.get("legacy_crosscheck", {}).get("requested"):
        require(report["legacy_crosscheck"].get("passed") is True, "Requested legacy baseline crosscheck did not pass")
    return report["variants"][variant]


def validate_gate(path: Path, variant: str, binary: Path) -> dict:
    report = parse_json(path.read_text(encoding="utf-8"))
    selected = validate_gate_scope(report, variant)
    for key in ("one_step_passed", "multi_update_passed", "poison_passed", "multiupdate_poison_passed"):
        require(selected.get(key) is True, f"Required clean/multiupdate/poison gate missing: {variant}.{key}")
    for key, expected in FEATURES[variant].items():
        require(selected["features"].get(key) is expected, f"Gate compiled feature mismatch: {variant}.{key}")
    actual = identity(binary)
    for key in ("executable_before", "executable_after"):
        checked = selected[key]
        require(actual == {name: checked[name] for name in ("bytes", "sha256")}, f"Binary differs from its numerical gate: {variant}")
    before, after = _source_rows(report.get("sources_before")), _source_rows(report.get("sources_after"))
    require(before and after, "Gate must preserve source fingerprints before and after validation")
    index_after = {row["path"]: row for row in after}
    for row in before:
        wanted = {name: row[name] for name in ("bytes", "sha256")}
        require(row["path"] in index_after and wanted == {name: index_after[row["path"]][name] for name in wanted},
                "Gate sources changed during numerical validation")
        require(identity(Path(row["path"])) == wanted, f"Gate source is no longer frozen: {row['path']}")
    checker = report.get("checker")
    require(isinstance(checker, dict) and checker.get("path"), "Gate must preserve the independent checker identity")
    require(identity(Path(checker["path"])) == {name: checker[name] for name in ("bytes", "sha256")}, "Independent checker changed after the gate")
    return {"path": str(path), **identity(path), "variant": variant, "passed": True,
            "selected_gate": selected, "sources_before": before, "sources_after": after, "checker": checker}


def validate_build_manifest(path: Path, binaries: dict[str, Path]) -> dict:
    manifest = parse_json(path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete", "Build manifest must record a successful completed build")
    require(set(manifest.get("variants", {})) == set(VARIANTS), "Build manifest must cover exactly five variants")
    sources = _source_rows(manifest.get("sources"))
    after = {row["path"]: row for row in _source_rows(manifest.get("sources_after"))}
    require(sources and after, "Build manifest needs sources and sources_after fingerprints")
    for row in sources:
        wanted = {key: row[key] for key in ("bytes", "sha256")}
        require(identity(Path(row["path"])) == wanted and row["path"] in after
                and {key: after[row["path"]][key] for key in wanted} == wanted,
                f"Build source differs from the frozen build manifest: {row['path']}")
    macros = {"fixed_norm": "FW_FIXED_NORM", "softmax_inv_batch": "FW_SOFTMAX_INV_BATCH", "minimal_zero": "FW_MINIMAL_ZERO"}
    for variant, row in manifest["variants"].items():
        require(row.get("exit_code") == 0, f"Build did not pass: {variant}")
        require(identity(binaries[variant]) == {key: row["binary"][key] for key in ("bytes", "sha256")},
                f"Actual executable differs from build manifest: {variant}")
        command = row["command"]
        require(all(flag in command for flag in ("-O3", "-mcpu=native", "-ffp-contract=off", "-fno-math-errno",
                                                 "-DEXPERIMENT_VFORCE_EXP=1", "-fopenmp", "-lomp", "Accelerate")),
                f"Build flags do not match the common strict native/vForce/OpenMP policy: {variant}")
        require(not any(flag in command for flag in ("-ffast-math", "-Ofast", "-funsafe-math-optimizations", "-ffinite-math-only")),
                "Global relaxed FP flags are outside this fixed-work comparison")
        for feature, macro in macros.items():
            expected = f"-D{macro}={int(FEATURES[variant][feature])}"
            definitions = [flag for flag in command if flag.startswith(f"-D{macro}=")]
            require(definitions == [expected], f"Build macro is missing or redefined: {variant}.{feature}")
            require(row["features"].get(feature) is FEATURES[variant][feature], f"Build feature metadata differs: {variant}.{feature}")
    return {"path": str(path), **identity(path), "passed": True, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true", help="Metadata/plan only; the default")
    action.add_argument("--run", action="store_true", help="Explicitly execute checked fixed-work pilots and ABBA trials")
    parser.add_argument("--binary", action="append", default=[], help="variant=prebuilt executable")
    parser.add_argument("--gate", action="append", default=[], help="variant=passed model-validation JSON; one shared all-build report is allowed")
    parser.add_argument("--evidence", action="append", type=Path, default=[], help="Additional passed kernel verification JSON")
    parser.add_argument("--build-manifest", type=Path, help="Completed root-owned build manifest to verify and reuse; no rebuilding")
    parser.add_argument("--corpus", type=Path, default=WIKI / "training-corpora/wiki-k35328")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, help="One explicit common N, otherwise select once from baseline pilot")
    parser.add_argument("--target-seconds", type=float, default=15)
    parser.add_argument("--min-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4096)
    args = parser.parse_args()
    require(sys.platform == "darwin", "This experiment targets Mac CPU / Apple Accelerate")
    require(10 <= args.target_seconds <= 30 and 1 <= args.min_steps <= args.max_steps <= 4096, "Invalid pilot/common-N bounds")
    require(args.steps is None or 1 <= args.steps <= 4096, "Explicit common steps must be between 1 and 4096")
    binaries = {variant: BASE / f"cpu_fixedwork_{variant}" for variant in VARIANTS}
    binaries.update(named_paths(args.binary))
    gates = named_paths(args.gate)
    corpus, output = args.corpus.absolute(), args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = output if args.run else output.with_name(output.stem + ".prepare.json")
    require(not artifact.exists(), f"Preserving existing evidence: {artifact}")
    source_paths = [BASE / name for name in SOURCE_NAMES]
    existing_sources = {str(path): identity(path) for path in source_paths if path.exists()}
    report = {
        "kind": "REAL_WIKI_MAC_FIXED_WORK_ABBA", "status": "preparing", "created_at": now(),
        "execution_authorization": "explicit --run" if args.run else "prepare-only; no binary execution",
        "config": {**FIXED_CONFIG, "optimizer": "AdamW", "gradient_max_norm": 1.0},
        "environment": THREAD_ENVIRONMENT, "main_steps": args.steps, "events_per_trial": None,
        "pilot_policy": {"initial_steps": 64, "secondary_steps_if_first_under_2_seconds": 128,
                         "target_seconds": args.target_seconds, "min_main_steps": args.min_steps,
                         "max_main_steps": args.max_steps, "candidate_specific_N": False},
        "design": {"blocks": list(CANDIDATES), "within_block_order": ["baseline", "candidate", "candidate", "baseline"],
                   "main_trial_count": 16, "warmup_updates": 2, "process_timeout_seconds": TIMEOUT_SECONDS,
                   "baseline_control": "Arithmetic mean elapsed of the two local endpoint baselines; both B observations remain separate.",
                   "drift_policy": "Endpoint drift >20% flags variability; do not discard observations or change N.",
                   "duration_policy": "10-30s is a pilot target only; every actual duration is preserved, including outside-range trials."},
        "scope": "Same real-Wiki target windows and same model/optimizer work. No full epoch, production trainer, GPU, package installation or application/process modification.",
        "cpu_metric_scope": "Timed process CPU sums workers; timed main-thread CPU and getrusage are separate from wall time. Epoch-hour projection uses wall target throughput only.",
        "parameter_checksum_policy": "Record final parameters/m/v as audit values; differences do not substitute for or fail the independent numerical gate.",
        "source_before": existing_sources, "binaries": {}, "gates": {}, "evidence": [],
        "readiness": {"missing_sources": [str(path) for path in source_paths if not path.exists()],
                      "missing_binaries": [variant for variant, path in binaries.items() if not path.exists()],
                      "missing_gates": [variant for variant in VARIANTS if variant not in gates or not gates[variant].exists()]},
        "preflight_errors": [], "build_manifest": None,
        "pilots": [], "trials": [], "comparisons": [], "process_snapshots": [],
    }
    for variant, path in binaries.items():
        if path.exists():
            report["binaries"][variant] = {"path": str(path), **identity(path)}
    for variant, path in gates.items():
        if path.exists() and binaries[variant].exists():
            try:
                report["gates"][variant] = validate_gate(path, variant, binaries[variant])
            except (OSError, ValueError, KeyError, TypeError) as error:
                report["preflight_errors"].append(f"Gate {variant}: {error}")
    if args.build_manifest is not None:
        try:
            report["build_manifest"] = validate_build_manifest(args.build_manifest.absolute(), binaries)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["preflight_errors"].append(f"Build manifest: {error}")
    for path in args.evidence:
        try:
            evidence = parse_json(path.read_text(encoding="utf-8"))
            require(evidence.get("passed") is True, f"Kernel verification was not passed: {path}")
            report["evidence"].append({"path": str(path.absolute()), **identity(path), "report": evidence})
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["preflight_errors"].append(f"Kernel evidence: {error}")
    if not args.run:
        report["status"] = "prepared"
        report["readiness"]["runnable"] = not any(report["readiness"].values()) and not report["preflight_errors"]
        atomic_json(artifact, report)
        print(json.dumps({"status": "prepared", "executed_model_or_compiler": False,
                          "runnable": report["readiness"]["runnable"], "report": str(artifact)}, indent=2))
        return
    logs = output.parent / (output.stem + "-logs")
    require(not logs.exists(), f"Preserving existing raw logs: {logs}")
    logs.mkdir()
    progress_path = output.with_name(output.stem + ".progress.json")

    def save() -> None:
        atomic_json(output, report)
        atomic_json(progress_path, {"status": report["status"], "updated_at": now(), "main_steps": report["main_steps"],
                                    "passed_main_trials": sum(row.get("passed", False) for row in report["trials"]),
                                    "planned_main_trials": 16, "last_trial": report["trials"][-1]["trial_id"] if report["trials"] else None,
                                    "error": report.get("error")})

    frozen_binaries = {variant: {key: row[key] for key in ("bytes", "sha256")} for variant, row in report["binaries"].items()}
    manifest = corpus / "manifest.json"
    export_path = WIKI / "reports/integer-encoding-npy.json"
    data_files = {}
    references = {}
    initial_reference = None

    def unchanged() -> None:
        for name, wanted in existing_sources.items():
            require(identity(Path(name)) == wanted, f"Frozen source changed: {name}")
        for variant, wanted in frozen_binaries.items():
            require(identity(binaries[variant]) == wanted, f"Validated binary changed: {variant}")
        for variant, gate in report["gates"].items():
            require(identity(Path(gate["path"])) == {key: gate[key] for key in ("bytes", "sha256")}, f"Gate evidence changed: {variant}")
        if report["build_manifest"] is not None:
            build = report["build_manifest"]
            require(identity(Path(build["path"])) == {key: build[key] for key in ("bytes", "sha256")}, "Build manifest changed")
        for name, info in data_files.items():
            stat = Path(name).stat()
            require((stat.st_size, stat.st_mtime_ns) == (info["bytes"], info["mtime_ns"]), f"Read-only corpus metadata changed: {name}")

    def execute(variant: str, steps: int, label: str, block: str, position: int, pilot=False) -> dict:
        nonlocal initial_reference
        unchanged()
        snapshot = process_snapshot()
        report["process_snapshots"].append({"before": label, **snapshot})
        require(not snapshot["known_training_or_benchmark_processes"], "Another live training/checker/benchmark worker was found; no processes were terminated")
        command = [str(binaries[variant]), "--corpus", str(corpus), "--steps", str(steps), "--warmup-steps", "2",
                   "--threads", "4", "--seed", "42", "--vocab-size", "35328", "--context-size", "8",
                   "--embedding-dim", "64", "--hidden-dim", "128", "--batch-size", "128", "--clip-norm", "1", *OMP_ALL]
        print(f"{label}: {variant}, fixed {steps} updates / {steps * 128:,} targets", flush=True)
        result = capture(["/usr/bin/time", "-l", *command], timeout=TIMEOUT_SECONDS, env=dict(os.environ, **THREAD_ENVIRONMENT))
        stdout_path, stderr_path = logs / f"{label}.stdout.json", logs / f"{label}.stderr.txt"
        stdout_path.write_text(result["stdout"], encoding="utf-8")
        stderr_path.write_text(result["stderr"], encoding="utf-8")
        row = {"trial_id": label, "block": block, "position": position, "variant": variant, "pilot": pilot,
               "passed": False, "command": command, "process_wall_seconds": result["process_wall_seconds"],
               "exit_code": result["exit_code"], "timed_out": result["timed_out"],
               "load_before": snapshot["load_average"], "load_after": list(os.getloadavg()),
               "stdout_log": {"path": str(stdout_path), **identity(stdout_path)},
               "stderr_log": {"path": str(stderr_path), **identity(stderr_path)}}
        try:
            require(not result["timed_out"] and result["exit_code"] == 0, "Child failed or its entire process group exceeded the 60-second cap")
            data = parse_json(result["stdout"])
            row["measurement"] = data
            validation = validate_measurement(data, variant, steps)
            common_initial = {key: data[key] for key in ("initial_parameter_checksum", "input_checksum")}
            common_initial["first_windows"] = data["corpus"]["first_windows"]
            if initial_reference is None:
                initial_reference = common_initial
            require(common_initial == initial_reference, "Initial parameter bits or first Wiki windows differ")
            fingerprint = work_fingerprint(data)
            if steps not in references:
                references[steps] = fingerprint
            require(fingerprint == references[steps], "Same-N first/final inputs or sampler start/end differ across variants")
            rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", result["stderr"])
            require(rss is not None, "macOS time -l peak RSS missing")
            row.update({"passed": True, "inputs_equal": True, "phase_accounting_passed": True,
                        "validation": validation, "peak_rss_bytes": int(rss.group(1)),
                        "outside_10_30_seconds": not 10 <= data["elapsed_seconds"] <= 30})
        except (ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
        (report["pilots"] if pilot else report["trials"]).append(row)
        save()
        require(row["passed"], f"Fixed-work trial not passed: {label}: {row.get('error')}")
        print(f"  {row['measurement']['elapsed_seconds']:.6f}s wall, {row['measurement']['process_cpu_seconds']:.6f}s process CPU, "
              f"{row['measurement']['targets_per_second']:.2f} targets/s; clip active {row['validation']['clip_active_updates']}/{steps}", flush=True)
        return row

    def canceled(signum, frame):
        raise KeyboardInterrupt(f"Canceled by signal {signum}")
    signal.signal(signal.SIGTERM, canceled)
    report["status"] = "running"
    save()
    try:
        require(not any(report["readiness"].values()) and len(report["gates"]) == len(VARIANTS)
                and not report["preflight_errors"], "All sources, binaries and passed clean/poison gates must be valid before --run: "
                + "; ".join(report["preflight_errors"]))
        with (BENCHMARK / ".timing.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata = parse_json(manifest.read_text(encoding="utf-8"))
            export = parse_json(export_path.read_text(encoding="utf-8"))
            require(metadata.get("status") == "complete" and metadata.get("vocab_size") == 35328
                    and metadata["statistics"]["train_target_windows"] == TRAIN_TARGETS, "Wrong preserved Wiki training corpus")
            require(identity(manifest) == {key: export["source_manifest"][key] for key in ("bytes", "sha256")}, "Corpus manifest differs from the verified NumPy export")
            expected_ids = next(row["output"] for row in export["arrays"] if row["split"] == "train")
            expected = {"train.ids.npy": expected_ids, "train.paragraphs.npy": metadata["outputs"]["train.paragraphs.npy"]}
            for name, wanted in expected.items():
                path = corpus / name
                actual = identity(path)
                require(actual == wanted, f"Actual Wiki array/index differs from preserved verification: {name}")
                data_files[str(path)] = {**actual, "mtime_ns": path.stat().st_mtime_ns}
            data_files[str(manifest)] = {**identity(manifest), "mtime_ns": manifest.stat().st_mtime_ns}
            report["corpus"] = {"path": str(corpus), "full_train_target_windows": TRAIN_TARGETS,
                                "epoch5_target_events": EPOCH5_TARGETS, "verified_files_before": data_files.copy(),
                                "numpy_export_report": {"path": str(export_path), **identity(export_path)}}
            report["host"] = {"load_average_before": list(os.getloadavg()), "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
                              "logical_cpus": os.cpu_count()}
            if args.steps is None:
                first = execute("baseline", 64, "pilot-64", "pilot", 0, pilot=True)
                if first["measurement"]["elapsed_seconds"] < 2:
                    execute("baseline", 128, "pilot-128", "pilot", 0, pilot=True)
                selection = select_main_steps(report["pilots"], args.target_seconds, args.min_steps, args.max_steps)
                main_steps = selection["steps"]
            else:
                main_steps = args.steps
                selection = {"steps": main_steps, "method": "explicit one common --steps value", "candidate_specific_retuning": False}
            report["main_steps"], report["events_per_trial"] = main_steps, main_steps * 128
            report["step_selection"] = selection
            save()
            for block in CANDIDATES:
                for position, variant in enumerate(("baseline", block, block, "baseline"), 1):
                    execute(variant, main_steps, f"{block}-{position:02d}-{variant}", block, position)
                report["comparisons"] = summarize_abba(report["trials"], main_steps * 128)
                save()
            unchanged()
            report["corpus"]["verified_files_after"] = {}
            for name, wanted in data_files.items():
                actual = identity(Path(name))
                require(actual == {key: wanted[key] for key in ("bytes", "sha256")}, f"Input/index bytes changed during study: {name}")
                report["corpus"]["verified_files_after"][name] = actual
            report["source_after"] = {name: identity(Path(name)) for name in existing_sources}
            report["comparisons"] = summarize_abba(report["trials"], main_steps * 128)
            require(len(report["trials"]) == 16 and all(row["passed"] for row in report["trials"])
                    and all(row["passed"] for row in report["comparisons"]), "Not every planned fixed-work observation completed")
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "canceled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if report["events_per_trial"] is not None:
            report["comparisons"] = summarize_abba(report["trials"], report["events_per_trial"])
        raise
    finally:
        report["completed_at"] = now()
        report["successful_trial_count"] = sum(row.get("passed", False) for row in report["trials"])
        report["all_planned_work_completed"] = report["status"] == "complete"
        save()
    print(json.dumps({"status": report["status"], "main_steps": report["main_steps"],
                      "events_per_trial": report["events_per_trial"], "trials": len(report["trials"]),
                      "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
