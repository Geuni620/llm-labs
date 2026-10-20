#!/usr/bin/env python3
"""Bind the unchanged numerical verifier to four accepted-optimizer candidates.

The v1 verifier, its CPU oracle/tolerances, and its rejected native-fused report
are preserved. Only the engine import and expected option table are adapted in
memory, in a fresh child. No numerical comparison function is replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback


BASE = Path(__file__).resolve().parent
GPU = BASE.parent
ROOT = GPU.parent
sys.path.insert(0, str(GPU))
from optimization import verify as frozen
from optimization_round2 import engine as refined


CASES = tuple(refined.CASES)


def sources():
    from corpus_fixture import helper_library_path
    paths = [Path(__file__), GPU / "optimization/verify.py", BASE / "engine.py",
             GPU / "optimization/engine_refined.py", GPU / "optimization/engine.py",
             GPU / "optimization/compiled_path.py", GPU / "optimization/metal_clip.py", GPU / "benchmark_mps.py",
             GPU / "verify_mps.py", GPU / "corpus_fixture.py", GPU / "init_reference.c", helper_library_path()]
    return {str(path.resolve()): frozen.oracle.identity(path) for path in paths}


def check_refined(case, corpus):
    """Use frozen.check_case unchanged, with an explicit refined engine binding."""
    frozen.CASES = CASES
    frozen.OPTIONS = {name: dict(options) for name, options in refined.CASE_OPTIONS.items()}
    frozen.require(all(options["foreach"] is True and options["fused"] is False
                       for options in frozen.OPTIONS.values()), "Refinement must retain the accepted foreach optimizer")
    original_alias = sys.modules.get("engine")
    sys.modules["engine"] = refined
    try:
        result = frozen.check_case(case, corpus)
    finally:
        if original_alias is None:
            sys.modules.pop("engine", None)
        else:
            sys.modules["engine"] = original_alias
    result["refined_engine_binding"] = {
        "module": "optimization_round2.engine", "engine": frozen.oracle.identity(BASE / "engine.py"),
        "numerical_verifier": frozen.oracle.identity(GPU / "optimization/verify.py"),
        "expected_options": dict(refined.CASE_OPTIONS[case]),
        "cpu_oracle_and_comparison_functions_replaced": False,
        "tolerances_changed": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training-corpora/wiki-k35328")
    parser.add_argument("--report", type=Path, default=BASE / "verification-refined.json")
    parser.add_argument("--child-timeout", type=float, default=900)
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--child-case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_case:
        with frozen.oracle.numerical_lock(externally_held=True):
            try:
                result = check_refined(args.child_case, args.corpus)
            except Exception as error:
                result = {"case": args.child_case, "status": "unsupported" if frozen.unsupported(error) else "failed",
                          "passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["passed"] else 2 if result["status"] == "unsupported" else 1
    if args.report.exists():
        raise FileExistsError(f"Preserving existing refined report: {args.report}")
    names = list(dict.fromkeys(args.case or CASES))
    report = {"kind": "MPS_REFINED_OPTIMIZATION_INDEPENDENT_CPU_ORACLE_GATES", "status": "running", "passed": False,
              "started_at": datetime.now().astimezone().isoformat(), "cases": {}, "expected_cases": names,
              "expected_gates_per_case": 6, "expected_total_gates": len(names) * 6,
              "tolerances_predeclared": frozen.oracle.TOLERANCES,
              "scope": "Four composed foreach=True/fused=False candidates; tiny/irregular/full Wiki shapes, clips 1/.05, four consecutive updates after two warmups and exact reset.",
              "adapter_policy": "Reuse frozen verify.check_case and every numerical helper unchanged; explicitly bind refined engine and expected options in each fresh child.",
              "prior_failure_policy": "The native-fused v1 failure remains preserved. This experiment changes the optimizer path, never the tolerance.",
              "cache_environment": {key: os.environ.get(key) for key in frozen.CACHE_VARIABLES}}
    try:
        frozen.require(args.child_timeout > 0 and args.child_timeout < float("inf"), "Child timeout must be positive and finite")
        report["sources_before"] = sources()
        prior = ROOT / "reports/mps-optimization-verification-20261018-v1.json"
        if prior.exists():
            report["preserved_prior_failure_report"] = frozen.oracle.identity(prior)
        frozen.oracle.atomic_json(args.report, report)
        with frozen.oracle.numerical_lock(args.lock_held):
            for case in names:
                command = [sys.executable, str(Path(__file__).resolve()), "--child-case", case,
                           "--corpus", str(args.corpus.resolve())]
                process = subprocess.Popen(command, cwd=BASE, env=frozen.oracle.environment("preload_defer_foreach"),
                                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    stdout, stderr = process.communicate(timeout=args.child_timeout)
                    result = frozen.oracle.parse_json(stdout)
                    frozen.require(result.get("case") == case, "Refined child case mismatch")
                    result.update(command=command, exit_code=process.returncode, stderr=stderr,
                                  child_json_sha256=hashlib.sha256(stdout.encode()).hexdigest())
                    frozen.require(not result.get("passed") or process.returncode == 0, "Passing child exited nonzero")
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    result = {"case": case, "status": "failed", "passed": False, "timed_out": True,
                              "error": "Refined numerical child exceeded root-configured timeout",
                              "stderr": stderr, "partial_stdout": stdout, "command": command}
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
                frozen.oracle.atomic_json(args.report, report)
            report["sources_after"] = sources()
            frozen.require(report["sources_before"] == report["sources_after"], "Refined/frozen source bindings changed during gates")
            if "preserved_prior_failure_report" in report:
                frozen.require(report["preserved_prior_failure_report"] == frozen.oracle.identity(prior), "Prior rejected result was modified")
        checks = [row for case in report["cases"].values() for row in case.get("checks", [])]
        report.update(status="complete", passed=bool(report["cases"]) and all(row["passed"] for row in report["cases"].values()),
                      unsupported_cases=[name for name, row in report["cases"].items() if row["status"] == "unsupported"],
                      failed_cases=[name for name, row in report["cases"].items() if row["status"] == "failed"],
                      attempted_gates=len(checks), completed_gates=sum(row.get("status") == "complete" for row in checks),
                      passed_gates=sum(row["passed"] for row in checks),
                      comparison_count=sum(row.get("comparison_count", 0) for row in report["cases"].values()),
                      failed_comparisons=sum(row.get("failed_comparisons", 0) for row in report["cases"].values()))
    except Exception as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["completed_at"] = datetime.now().astimezone().isoformat()
    frozen.oracle.atomic_json(args.report, report)
    if args.quiet:
        result = {key: report[key] for key in ("kind", "status", "passed", "unsupported_cases", "failed_cases",
                                              "attempted_gates", "passed_gates", "comparison_count", "failed_comparisons", "error") if key in report}
        result["cases"] = {name: {key: row.get(key) for key in ("status", "passed", "small_gate_passed", "full_size_gate_passed", "comparison_count", "failed_comparisons")}
                           for name, row in report["cases"].items()}
        result["report"] = str(args.report.resolve())
    else:
        result = report
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
