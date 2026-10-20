"""Run bounded cumulative CPU experiments with phase wall-time instrumentation.

Every numerical check and real-Wiki trial runs serially under .timing.lock.
Trials measure a fixed compute duration, hence completed update counts differ.
Final trained losses from different step counts are not quality comparisons.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parent
BENCHMARK = BASE.parent
WIKI = BENCHMARK.parent
COMMON_FLAGS = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                "-ffp-contract=off", "-fno-math-errno"]
COMPILERS = {
    "O3": [],
    "native": ["-mcpu=native"],
    "relaxed_simd": ["-mcpu=native", "-ffast-math"],
}
OMP_ALL = ["--softmax-parallel", "--reduction-parallel", "--adam-parallel"]
CASES = [
    {"id": "baseline", "compiler": "O3", "checker_variant": "baseline", "args": [], "reference": "baseline"},
    {"id": "native_all_omp", "compiler": "native", "checker_variant": "baseline", "args": OMP_ALL, "reference": "baseline"},
    {"id": "native_all_omp_fast_exp", "compiler": "native", "checker_variant": "baseline", "args": [*OMP_ALL, "--fast-exp"], "reference": "native_all_omp"},
    {"id": "relaxed_all_omp", "compiler": "relaxed_simd", "checker_variant": "baseline", "args": OMP_ALL, "reference": "native_all_omp"},
    {"id": "relaxed_all_omp_tile64", "compiler": "relaxed_simd", "checker_variant": "baseline", "args": [*OMP_ALL, "--tiled-gemm", "64"], "reference": "relaxed_all_omp"},
]
SOURCE_NAMES = ("cumulative_benchmark.c", "benchmark.c", "run_cumulative.py", "experiment_kernels.h", "gemm_kernels.c", "vector_kernels.c", "fast_exp.h")
FIXED_CONFIG = {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64,
                "hidden_dim": 128, "batch_size": 128, "seed": 42, "warmup_steps": 2}
THREAD_ENVIRONMENT = {
    "VECLIB_MAXIMUM_THREADS": "4", "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "BLIS_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0", "OMP_DYNAMIC": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1", "OMP_NESTED": "FALSE", "OMP_THREAD_LIMIT": "4",
}


def identity(path: str | Path) -> dict:
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest}


def assert_finite_numbers(value, label="payload") -> None:
    """Python finite checks remain meaningful for binaries built with fast-math."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite JSON number at {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_numbers(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_numbers(item, f"{label}[{index}]")


def parse_json(payload: str) -> dict:
    def invalid_constant(constant):
        raise ValueError(f"Nonstandard/nonfinite JSON constant: {constant}")
    result = json.loads(payload, parse_constant=invalid_constant)
    assert_finite_numbers(result)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result


def expected_kernel_options(arguments: list[str]) -> dict:
    options = {"softmax_parallel": False, "reduction_parallel": False,
               "adam_parallel": False, "tiled_gemm": 0, "fast_exp": False}
    toggles = {"--softmax-parallel": "softmax_parallel", "--reduction-parallel": "reduction_parallel",
               "--adam-parallel": "adam_parallel", "--fast-exp": "fast_exp"}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in toggles:
            options[toggles[argument]] = True
        elif argument == "--tiled-gemm":
            index += 1
            if index == len(arguments) or arguments[index] not in {"32", "64"}:
                raise ValueError("A tile backend must be 32 or 64")
            options["tiled_gemm"] = int(arguments[index])
        else:
            raise ValueError(f"Unknown experiment kernel option: {argument}")
        index += 1
    return options


def capture(command: list[str], *, timeout=90, env=None, cwd=BASE) -> dict:
    """Kill the entire wrapper/benchmark process group on every timeout."""
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return {"command": command, "exit_code": process.returncode, "timed_out": timed_out,
            "process_wall_seconds": time.monotonic() - started,
            "stdout": stdout, "stderr": stderr}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(trials: list[dict], cases: list[dict], repeats: int, targets: int) -> list[dict]:
    summary = []
    by_case = {case["id"]: [row for row in trials if row["case"] == case["id"] and row.get("passed")]
               for case in cases}
    baseline = by_case.get("baseline", [])
    baseline_rates = {row["repeat"]: row["targets_per_second"] for row in baseline}
    baseline_rate = statistics.median(baseline_rates.values()) if len(baseline_rates) == repeats else None
    for case in cases:
        rows = by_case[case["id"]]
        entry = {"case": case["id"], "status": "passed" if len(rows) == repeats else "not_passed",
                 "passed": len(rows) == repeats, "repeats_completed": len(rows),
                 "compiler_variant": case["compiler"], "kernel_args": case["args"],
                 "reference_case": case["reference"],
                 "precision_relaxed": case["compiler"] == "relaxed_simd",
                 "approximate_exponential": "--fast-exp" in case["args"]}
        if not entry["passed"]:
            summary.append(entry)
            continue
        rates = [row["targets_per_second"] for row in rows]
        rate = statistics.median(rates)
        reference = by_case.get(case["reference"], [])
        reference_rates = {row["repeat"]: row["targets_per_second"] for row in reference}
        reference_rate = statistics.median(reference_rates.values()) if len(reference_rates) == repeats else None
        entry.update({
            "median_targets_per_second": rate, "min_targets_per_second": min(rates), "max_targets_per_second": max(rates),
            "median_elapsed_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
            "completed_steps": [row["steps"] for row in rows],
            "target_prediction_events": [row["target_prediction_events"] for row in rows],
            "maximum_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
            "median_corpus_load_seconds": statistics.median(row["corpus"]["corpus_load_seconds"] for row in rows),
            "median_process_wall_seconds": statistics.median(row["process_wall_seconds"] for row in rows),
            "speedup_relative_to_baseline": rate / baseline_rate if baseline_rate else None,
            "speedup_relative_to_reference": rate / reference_rate if reference_rate else None,
            "paired_speedups_relative_to_baseline": [row["targets_per_second"] / baseline_rates[row["repeat"]]
                                                      for row in rows if row["repeat"] in baseline_rates],
            "same_duration_losses_are_not_quality_comparisons": True,
            "extrapolation_only": {
                "target_windows": targets, "epoch1_hours": targets / rate / 3600,
                "epoch5_hours": 5 * targets / rate / 3600,
                "caveat": "Short RAM-resident fixed-duration throughput extrapolation; no full epoch, 5-epoch training, quality or sustained-speed result.",
            },
        })
        entry["eligible_for_combination"] = bool(case["id"] != "baseline" and baseline_rate
            and rate >= baseline_rate * 1.02 and len(entry["paired_speedups_relative_to_baseline"]) == repeats
            and all(speedup > 1 for speedup in entry["paired_speedups_relative_to_baseline"]))
        summary.append(entry)
    return summary


def choose_combination(summary: list[dict]) -> dict | None:
    eligible = {row["case"]: row for row in summary if row.get("eligible_for_combination")}
    compiler = max((row for name, row in eligible.items() if name in {"native", "relaxed_simd"}),
                   key=lambda row: row["median_targets_per_second"], default=None)
    args, components = [], []
    if compiler:
        components.append(compiler["case"])
    for name in ("softmax_omp", "reduction_omp", "adamw_omp", "fast_exp"):
        if name in eligible:
            args.extend(eligible[name]["kernel_args"])
            components.append(name)
    tile = max((row for name, row in eligible.items() if name in {"tiled32", "tiled64"}),
               key=lambda row: row["median_targets_per_second"], default=None)
    if tile:
        args.extend(tile["kernel_args"])
        components.append(tile["case"])
    if not components:
        return None
    return {"id": "verified_combination", "compiler": compiler["compiler_variant"] if compiler else "O3",
            "checker_variant": "baseline", "args": args, "reference": "baseline",
            "components": components}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--corpus", type=Path, default=WIKI / "training-corpora/wiki-k35328")
    parser.add_argument("--python", type=Path, default=WIKI / ".venv-training/bin/python")
    parser.add_argument("--openmp-prefix", type=Path, default=Path("/opt/homebrew/opt/libomp"))
    parser.add_argument("--seconds", type=int, default=20, choices=range(10, 31))
    parser.add_argument("--repeats", type=int, default=2, choices=(2, 3))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-test-report", type=Path, action="append", default=[])
    parser.add_argument("--no-combined", action="store_true")
    args = parser.parse_args()
    def canceled(signum, frame):
        raise KeyboardInterrupt(f"Experiment canceled by signal {signum}")
    signal.signal(signal.SIGTERM, canceled)
    if platform.system() != "Darwin":
        parser.error("This experiment explicitly targets Mac CPU and Apple Accelerate")
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Preserving existing experiment evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem + "-logs")
    logs.mkdir(exist_ok=False)
    corpus, python = args.corpus.absolute(), args.python.absolute()
    checker = BASE / "check_variants.py"
    source_paths = [BASE / name for name in SOURCE_NAMES]
    source_paths += [checker, Path(__file__).resolve(), BENCHMARK / "check_correctness.py", BENCHMARK / "cpu_benchmark.c", BENCHMARK / "Makefile"]
    if (BASE / "Makefile").exists():
        source_paths.append(BASE / "Makefile")
    frozen = {str(path): identity(path) for path in source_paths}
    manifest_path = corpus / "manifest.json"
    manifest = parse_json(manifest_path.read_text(encoding="utf-8"))
    export = parse_json((WIKI / "reports/integer-encoding-npy.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("vocab_size") != 35328
            or identity(manifest_path) != {key: export["source_manifest"][key] for key in ("bytes", "sha256")}):
        raise ValueError("The complete preserved K=35,328 Wiki corpus and its verified NumPy export are required")
    targets = manifest["statistics"]["train_target_windows"]
    if targets != 124232535:
        raise ValueError("Expected the verified 124,232,535 training target windows")
    env = dict(os.environ, **THREAD_ENVIRONMENT)
    cases = [dict(case) for case in CASES]
    report = {
        "kind": "REAL_WIKI_MAC_CPU_OPTIMIZATION_EXPERIMENTS", "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "scope": "Bounded real-Wiki CPU compute timings on one Mac; no old Python training, full epoch, GPU, quality or machine-selection benchmark.",
        "configuration": {**FIXED_CONFIG, "requested_threads": 4, "seconds_per_trial": args.seconds,
                          "repeats": args.repeats, "optimizer": "AdamW", "gradient_max_norm": 1.0},
        "source_before": frozen, "runner": {"path": str(Path(__file__).resolve()), **identity(Path(__file__))},
        "corpus": {"path": str(corpus), "manifest": identity(manifest_path), "full_train_target_windows": targets,
                   "numpy_export_report": identity(WIKI / "reports/integer-encoding-npy.json"), "arrays": export["arrays"]},
        "host": {"platform": platform.platform(), "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
                 "load_average_before": list(os.getloadavg()), "process_nice": os.getpriority(os.PRIO_PROCESS, 0)},
        "environment": THREAD_ENVIRONMENT, "openmp_prefix": str(args.openmp_prefix),
        "parallelism": "Every numerical checker and timed child runs serially under exclusive cpu-benchmark/.timing.lock; no outer OpenMP region wraps BLAS.",
        "thread_scope": "Requested 4 threads; actual OpenMP probe team is recorded. Apple Accelerate's actual internal worker count remains unmeasured.",
        "timing_scope": "Monotonic compute duration includes Wiki window gather, forward/backward, norm-1 clipping and AdamW; excludes RAM file load/setup, two warmups, full state/sampler reset and final-loss evaluation.",
        "duration_scope": "Fixed duration and variable completed steps. Each child ends after a complete update, so one-step overshoot is recorded.",
        "compiler_scope": "Options affect these experiment C translation units only; linked Apple Accelerate is unchanged and is not rebuilt.",
        "precision_scope": "native retains ordinary FP semantics; relaxed_simd explicitly adds -ffast-math. Approximate exp is separately labeled and independently checked.",
        "quality_scope": "Final losses after different completed update counts are recorded for finite sanity only and are not compared as model quality.",
        "combination_rule": "Only individually checked cases with >=2% median gain and gain in both paired repeats qualify; choose at most one tile backend. Combined throughput is measured separately, not added or assumed.",
        "module_test_reports": [], "builds": [], "correctness": [], "case_status": {},
        "trial_order": [], "trials": [], "summary": [], "cases": cases,
    }

    def save():
        atomic_json(output, report)

    def unchanged():
        for name, expected in frozen.items():
            if identity(Path(name)) != expected:
                raise RuntimeError(f"Frozen source/checker changed during experiment: {name}")
        if identity(manifest_path) != report["corpus"]["manifest"]:
            raise RuntimeError("Corpus manifest changed during experiment")

    def run_logged(command, label, timeout=90):
        result = capture(command, timeout=timeout, env=env)
        stdout_path, stderr_path = logs / f"{label}.stdout.txt", logs / f"{label}.stderr.txt"
        stdout_path.write_text(result["stdout"], encoding="utf-8")
        stderr_path.write_text(result["stderr"], encoding="utf-8")
        result["stdout_log"] = {"path": str(stdout_path), **identity(stdout_path)}
        result["stderr_log"] = {"path": str(stderr_path), **identity(stderr_path)}
        return result

    def build(name):
        optimizations = COMPILERS[name]
        flags = [*COMMON_FLAGS, *optimizations]
        if name == "relaxed_simd":
            flags += ["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize"]
        binary = BASE / f"cpu_cumulative_{name}"
        include, library = args.openmp_prefix / "include", args.openmp_prefix / "lib"
        command = [args.compiler, "-DACCELERATE_NEW_LAPACK", "-DBENCH_BUILD_FLAGS=" + json.dumps(" ".join(flags)),
                   *flags, "-Xpreprocessor", "-fopenmp", "-isystem", str(include),
                   str(BASE / "cumulative_benchmark.c"), str(BASE / "gemm_kernels.c"), str(BASE / "vector_kernels.c"),
                   "-o", str(binary), "-L", str(library), "-Wl,-rpath," + str(library),
                   "-lomp", "-framework", "Accelerate", "-lm"]
        result = run_logged(command, f"build-{name}")
        row = {"compiler_variant": name, "flags": flags, "command": command,
               "exit_code": result["exit_code"], "timed_out": result["timed_out"],
               "supported": result["exit_code"] == 0 and not result["timed_out"],
               "stdout_log": result["stdout_log"], "stderr_log": result["stderr_log"]}
        if row["supported"]:
            row["binary"] = {"path": str(binary), **identity(binary)}
            linked = run_logged(["otool", "-L", str(binary)], f"linked-{name}")
            row["linked_libraries"] = linked["stdout"]
        if name == "relaxed_simd":
            vector_lines = [line for line in result["stderr"].splitlines()
                            if "vectorized loop" in line or "loop not vectorized" in line]
            row["vectorization_evidence"] = {"vectorized_remarks": sum("vectorized loop" in line for line in vector_lines),
                                             "missed_remarks": sum("loop not vectorized" in line for line in vector_lines),
                                             "remarks": vector_lines, "full_log": result["stderr_log"]}
        report["builds"].append(row)
        save()
        return row

    def check(case):
        unchanged()
        build_record = next(row for row in report["builds"] if row["compiler_variant"] == case["compiler"])
        if not build_record["supported"]:
            report["case_status"][case["id"]] = {"status": "unsupported_build", "passed": False}
            save()
            return False
        binary = Path(build_record["binary"]["path"])
        if identity(binary) != {key: build_record["binary"][key] for key in ("bytes", "sha256")}:
            raise RuntimeError("Benchmark binary changed before its checker")
        checked_path = logs / f"check-{case['id']}.json"
        command = [str(python), str(checker), "--executable", str(binary), "--report", str(checked_path),
                   "--variant", case["checker_variant"], "--threads", "1", "4", "--clip-norms", "1", ".05",
                   "--lock-held", "--quiet", "--build-label", case["id"]]
        command += ["--extra-args", json.dumps(case["args"])]
        result = run_logged(command, f"check-{case['id']}")
        try:
            checked = parse_json(checked_path.read_text(encoding="utf-8")) if checked_path.exists() else parse_json(result["stdout"])
        except (OSError, ValueError) as error:
            checked = {"passed": False, "parse_error": str(error)}
        passed = (result["exit_code"] == 0 and not result["timed_out"] and checked.get("passed") is True
                  and checked.get("planned_cases") == 8 and checked.get("passed_cases") == 8
                  and checked.get("tolerances_predeclared", {}).get("default_baseline_policy") is True)
        report["correctness"].append({"case": case["id"], "passed": passed, "exit_code": result["exit_code"],
                                      "timed_out": result["timed_out"], "report": checked,
                                      "stdout_log": result["stdout_log"], "stderr_log": result["stderr_log"]})
        report["case_status"][case["id"]] = {"status": "checked" if passed else "correctness_failed", "passed": False,
                                             "correctness_passed": passed}
        save()
        return passed

    reference_prefix = None

    def execute(case, repeat):
        nonlocal reference_prefix
        unchanged()
        build_record = next(row for row in report["builds"] if row["compiler_variant"] == case["compiler"])
        binary = Path(build_record["binary"]["path"])
        if identity(binary) != {key: build_record["binary"][key] for key in ("bytes", "sha256")}:
            raise RuntimeError("Benchmark binary changed before timing")
        command = [str(binary), "--corpus", str(corpus), "--threads", "4", "--seed", "42",
                   "--vocab-size", "35328", "--context-size", "8", "--embedding-dim", "64", "--hidden-dim", "128",
                   "--batch-size", "128", "--warmup-steps", "2", "--seconds", str(args.seconds), *case["args"]]
        load_before = list(os.getloadavg())
        result = run_logged(["/usr/bin/time", "-l", *command], f"trial-{repeat:02d}-{case['id']}")
        row = {"case": case["id"], "repeat": repeat, "passed": False, "command": command,
               "process_wall_seconds": result["process_wall_seconds"], "exit_code": result["exit_code"],
               "timed_out": result["timed_out"], "stdout_log": result["stdout_log"], "stderr_log": result["stderr_log"],
               "load_average_before": load_before, "load_average_after": list(os.getloadavg())}
        try:
            if result["exit_code"] or result["timed_out"]:
                raise ValueError("Timing child failed or exceeded its 90-second process-group cap")
            measured = parse_json(result["stdout"])
            for key, expected in FIXED_CONFIG.items():
                if measured["config"].get(key) != expected:
                    raise ValueError(f"Unexpected model/warmup configuration: {key}")
            if (measured.get("kind") != "REAL_WIKI_C_MICROBENCHMARK" or measured.get("finite_flag") is not True
                    or measured.get("warmup_reset") is not True or measured.get("gradient_max_norm") != 1.0
                    or measured.get("steps", 0) <= 0 or measured["target_prediction_events"] != measured["steps"] * 128
                    or measured["corpus"]["target_windows"] != targets
                    or measured["elapsed_seconds"] + 1e-6 < args.seconds
                    or measured["elapsed_seconds"] > 30 or measured.get("requested_seconds") != args.seconds
                    or measured.get("completed_steps") != measured["steps"]
                    or measured["requested_threads"] != 4):
                raise ValueError("Invalid compute duration, completed work, reset, clipping or real-Wiki scope")
            options = expected_kernel_options(case["args"])
            if any(measured["kernel_options"].get(key) != value for key, value in options.items()):
                raise ValueError("The requested one-factor kernel options were not applied")
            omp = measured["openmp"]
            if (omp.get("compiled") is not True or omp.get("probe_team_size") != 4
                    or omp.get("nested_disabled") is not True or omp.get("wait_policy") != "PASSIVE"
                    or omp.get("kmp_blocktime") != 0):
                raise ValueError("OpenMP support/team/wait policy differs from the common experiment contract")
            if measured.get("fast_math_build") != (case["compiler"] == "relaxed_simd"):
                raise ValueError("Actual precision/compiler semantics differ from the labeled case")
            if not math.isclose(measured["targets_per_second"], measured["target_prediction_events"] / measured["elapsed_seconds"], rel_tol=1e-10):
                raise ValueError("Reported throughput disagrees with actual completed targets / compute seconds")
            prefix = (measured["initial_parameter_checksum"], measured["input_checksum"], measured["corpus"]["first_windows"])
            if reference_prefix is None:
                reference_prefix = prefix
            elif prefix != reference_prefix:
                raise ValueError("Initial parameter bits, first batch checksum or sampled Wiki windows differ across cases")
            rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", result["stderr"])
            if rss is None:
                raise ValueError("macOS time -l did not record peak resident set size")
            row.update(measured)
            row.update(passed=True, peak_rss_bytes=int(rss.group(1)),
                       peak_rss_scope="macOS per-process maximum RSS including corpus RAM/setup/warmup",
                       requested_compute_seconds=args.seconds,
                       compute_duration_overshoot_seconds=measured["elapsed_seconds"] - args.seconds,
                       precision_relaxed=case["compiler"] == "relaxed_simd",
                       approximate_exponential="--fast-exp" in case["args"])
        except (ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
        report["trials"].append(row)
        if not row["passed"]:
            report["case_status"][case["id"]] = {"status": "timing_failed", "passed": False, "error": row["error"]}
        save()
        return row

    save()
    try:
        with (BENCHMARK / ".timing.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["host"]["cpu"] = capture(["sysctl", "-n", "machdep.cpu.brand_string"])["stdout"].strip()
            report["host"]["physical_memory_bytes"] = int(capture(["sysctl", "-n", "hw.memsize"])["stdout"])
            report["compiler"] = run_logged([args.compiler, "--version"], "compiler-version")
            if not (args.openmp_prefix / "include/omp.h").exists() or not (args.openmp_prefix / "lib/libomp.dylib").exists():
                raise FileNotFoundError("A real libomp installation is required; no serial fallback will be timed")
            for path in args.module_test_report:
                checked = parse_json(path.read_text(encoding="utf-8"))
                if checked.get("passed") is not True:
                    raise ValueError(f"Prerequisite module test report is not passed: {path}")
                report["module_test_reports"].append({"path": str(path.absolute()), **identity(path), "report": checked})
            for name in COMPILERS:
                print(f"Build {name}; no numerical jobs overlap.", flush=True)
                unchanged()
                build(name)
            if not report["builds"][0]["supported"]:
                raise RuntimeError("The O3 reference did not compile")
            ready = []
            for case in cases:
                print(f"Independent check: {case['id']}", flush=True)
                if check(case):
                    ready.append(case)
                elif case["id"] == "baseline":
                    raise RuntimeError("The reference numerical checker failed; no performance comparison is valid")
            for repeat in range(1, args.repeats + 1):
                order = ready if repeat % 2 else list(reversed(ready))
                report["trial_order"].append({"repeat": repeat, "cases": [case["id"] for case in order]})
                for case in order:
                    if report["case_status"][case["id"]]["status"] == "timing_failed":
                        continue
                    print(f"Trial {repeat}/{args.repeats}: {case['id']}, {args.seconds}s compute", flush=True)
                    row = execute(case, repeat)
                    if row["passed"]:
                        print(f"  {row['steps']} steps / {row['elapsed_seconds']:.3f}s / {row['targets_per_second']:.2f} targets/s", flush=True)
                    else:
                        print(f"  NOT PASSED: {row['error']}", flush=True)
            report["summary"] = summarize(report["trials"], cases, args.repeats, targets)
            combination = None if args.no_combined else choose_combination(report["summary"])
            if combination is None:
                report["combined_case"] = {"status": "not_run", "passed": False,
                                           "reason": "disabled or no individually verified beneficial setting"}
            else:
                report["cases"].append(combination)
                report["combined_case"] = {"status": "pending_check", **combination}
                print(f"Check measured-beneficial combination: {combination['components']}", flush=True)
                if check(combination):
                    for repeat in range(1, args.repeats + 1):
                        print(f"Combined trial {repeat}/{args.repeats}, {args.seconds}s compute", flush=True)
                        row = execute(combination, repeat)
                        if not row["passed"]:
                            break
                    report["summary"] = summarize(report["trials"], report["cases"], args.repeats, targets)
                    combined_summary = next(row for row in report["summary"] if row["case"] == combination["id"])
                    report["combined_case"].update(status=combined_summary["status"], passed=combined_summary["passed"])
                else:
                    report["combined_case"].update(status="correctness_failed", passed=False)
                    report["summary"] = summarize(report["trials"], report["cases"], args.repeats, targets)
            for entry in report["summary"]:
                if entry["passed"]:
                    report["case_status"][entry["case"]] = {"status": "passed", "passed": True, "correctness_passed": True}
            unchanged()
            report["source_after"] = {str(path): identity(path) for path in source_paths}
            passed_trials = [row for row in report["trials"] if row["passed"]]
            report["same_initialization_validation"] = {"passed_trials_checked": len(passed_trials),
                                                        "identical_initial_parameter_bits_for_passed_trials": bool(passed_trials),
                                                        "identical_first_batch_checksum_and_four_windows_for_passed_trials": bool(passed_trials),
                                                        "not_passed_trials": len(report["trials"]) - len(passed_trials),
                                                        "workload_duration_fixed_steps_variable": True,
                                                        "no_cross_step_count_loss_equivalence_claim": True}
            report["status"] = "complete" if all(row["passed"] for row in report["summary"]) else "complete_with_not_passed_cases"
    except BaseException as error:
        report["status"] = "canceled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        for case in report["cases"]:
            if report["case_status"].get(case["id"], {}).get("passed") is not True:
                report["case_status"][case["id"]] = {"status": "not_passed", "passed": False, "reason": report["error"]}
        raise
    finally:
        report["host"]["load_average_after"] = list(os.getloadavg())
        report["completed_at"] = datetime.now().astimezone().isoformat()
        save()


if __name__ == "__main__":
    main()
