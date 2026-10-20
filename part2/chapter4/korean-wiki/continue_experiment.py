"""Continue the authorized Wiki pipeline once existing writers finish.

One bounded sequential driver: independent verification, complete coverage and
plots, idle hardware profiling, then the approved 14-hour training comparison.
Never launches another tokenizer or frequency watcher.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from training_corpus import atomic_json, identity


ROOT = Path(__file__).resolve().parent
READY_REPORTS = ["step5-state.json", "token-frequency-state.json", "step6-numeric-state.json",
                 "step5-summary.json", "token-frequency-summary.json", "step6-numeric-summary.json",
                 "numeric-token-frequency-summary.json"]


def writers_complete(root):
    for name in READY_REPORTS:
        try:
            report = json.loads((root / "reports" / name).read_text())
        except (OSError, ValueError):
            return False
        if report.get("status") != "complete":
            return False
    return True


def reusable_tokenization_verification(root):
    """Reuse a full hash scan only while its exact files and manifests are stable.

    Matches the frequency verifier's digest-bound source policy: previous full
    SHA/CRC measurements plus unchanged byte size and nanosecond mtime. Small
    reference inputs are hashed again; no second 292M-token scan is needed.
    """
    root = Path(root)
    reports = root / "reports"
    report = json.loads((reports / "step5-verification.json").read_text())
    summary = json.loads((reports / "step5-summary.json").read_text())
    state = json.loads((reports / "step5-state.json").read_text())
    snapshot = json.loads((reports / "step5-input-snapshot.json").read_text())
    names = {f"part_{i:02d}.txt" for i in range(32)}
    verified = {row["file"]: row for row in report.get("shards", [])}
    if (report.get("passed") is not True or report.get("scope") != "full"
            or report.get("verified_shards") != 32 or report.get("failures") != 0
            or set(verified) != names or set(state.get("completed", {})) != names
            or set(state.get("inputs", {})) != names
            or state.get("status") != "complete" or summary.get("status") != "complete"
            or state.get("config") != summary.get("config")):
        raise ValueError("Cannot reuse incomplete or inconsistent tokenization verification")
    source_check = report.get("input_identity", {})
    if (source_check.get("checked_files") != len(snapshot) or source_check.get("mismatches")
            or source_check.get("sha256_unverified") or source_check.get("sha256_missing_from_snapshot")):
        raise ValueError("Cannot reuse a verification with unverified or changed source identities")
    for name in names:
        for directory, side, expected in [("sentences_dedup", "input", state["inputs"][name]),
                                           ("tokens", "output", state["completed"][name]["output"])]:
            measured = verified[name][side]
            current = (root / directory / name).stat()
            if ((current.st_size, current.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"])
                    or any(measured.get(key) != expected[key] for key in ("bytes", "sha256", "crc32"))):
                raise ValueError(f"Previously verified file changed: {directory}/{name}")
    for relative, expected in snapshot.items():
        current = (root / relative).stat()
        metadata = {"bytes": current.st_size, "mtime_ns": current.st_mtime_ns}
        # Reference entries in the original snapshot intentionally contain SHA
        # only; source shards additionally bind their byte size and mtime.
        if any(metadata[key] != expected[key] for key in metadata if key in expected):
            raise ValueError(f"Previously verified input changed: {relative}")
        if relative.startswith("sentences_dedup/"):
            if expected["sha256"] != state["inputs"][Path(relative).name]["sha256"]:
                raise ValueError(f"Source snapshot differs: {relative}")
        elif identity(root / relative)["sha256"] != expected["sha256"]:
            raise ValueError(f"Previously verified reference changed: {relative}")
    for field, summary_field in [("token_sentences", "sentences"), ("token_paragraphs", "paragraphs"),
                                  ("tokens", "tokens"), ("space_tokens", "space_tokens")]:
        if report["counts"][field] != summary[summary_field]:
            raise ValueError(f"Verification/summary totals differ: {field}")
    return {"policy": "reuse full SHA/CRC verification with unchanged bound manifests and file size/mtime",
            "verified_shards": 32, "report": identity(reports / "step5-verification.json"),
            "summary": identity(reports / "step5-summary.json"), "state": identity(reports / "step5-state.json")}


def choose_configuration(coverage, benchmark):
    if coverage.get("status") != "complete" or coverage["scope"]["shard_count"] != 32:
        raise ValueError("Only the complete 32-shard coverage can select a candidate")
    variant = next(row for row in coverage["variants"] if row["id"] == "numeric")
    k = variant["candidate_knee"]["k"]
    if not isinstance(k, int) or not 16384 <= k <= 65536:
        raise ValueError("Invalid bounded vocabulary candidate")
    if benchmark.get("status") not in {"complete", "completed_with_failures"}:
        raise ValueError("Hardware profiling is incomplete")
    candidates = []
    required = {"vocab_size": k, "context_size": 8, "embedding_dim": 64,
                "hidden_dim": 128, "batch_size": 128, "cpu_threads": 4}
    for row in benchmark["results"]:
        if row.get("status") != "complete" or any(row["config"].get(key) != value for key, value in required.items()):
            continue
        rate = row["timing"]["median_target_predictions_per_second"]
        if math.isfinite(rate) and rate > 0:
            candidates.append((rate, row["config"]["device"]))
    if not candidates:
        raise ValueError("No successful matching CPU/MPS profile for the selected K")
    rate, device = max(candidates)
    return {"vocab_size": k, "device": device, "profile_target_predictions_per_second": rate,
            "criterion": "Numeric vocabulary retention knee in K=16,384..65,536; fastest measured matching CPU/MPS configuration",
            "quality_optimum_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-step5-verification", action="store_true",
                        help="Reuse the passed full Step 5 check only if its digest-bound files are unchanged")
    args = parser.parse_args()
    reports = ROOT / "reports"
    state_path = reports / "experiment-continuation.json"
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    state = {"status": "running", "phase": "waiting_for_existing_writers", "child_pid": None,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "driver": identity(Path(__file__)), "completed_phases": []}
    def update(**values):
        state.update(values)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(state_path, state)
    def stopped():
        return stop["requested"] or (reports / "STOP-EXPERIMENT").exists()
    def run(phase, command, timeout):
        update(phase=phase)
        logfile = reports / f"{phase}.log"
        with logfile.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen([str(x) for x in command], cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=child.pid, log=str(logfile))
            started = time.monotonic()
            while child.poll() is None:
                if stopped() or time.monotonic() - started > timeout:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    if stopped():
                        raise InterruptedError("Requested experiment stop")
                    raise TimeoutError(f"Phase {phase} exceeded {timeout:g}s; see {logfile}")
                time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"Phase {phase} exited {child.returncode}; see {logfile}")
        state["completed_phases"].append(phase)
        update(child_pid=None)
    with (reports / ".experiment-continuation.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            update()
            wait_started = time.monotonic()
            while not writers_complete(ROOT):
                if stopped():
                    raise InterruptedError("Requested experiment stop")
                if time.monotonic() - wait_started > 3 * 3600:
                    raise TimeoutError("Existing preprocessing writers did not finish within the 3-hour wait limit")
                update()
                time.sleep(15)
            python = ROOT / ".venv-kiwi/bin/python"
            for phase, script, arguments, timeout in [
                ("step5-full-verification", "verify_tokenization.py", ["--workers", 4], 1800),
                ("token-frequency-verification", "verify_token_frequencies.py", [], 1800),
                ("step6-numeric-verification", "verify_numeric_tokens.py", ["--workers", 4], 1800),
                ("vocabulary-coverage-full", "analyze_vocabulary_coverage.py", ["--workers", 4], 3600),
                ("vocabulary-coverage-plot-full", "build_vocabulary_coverage_plot.py", [], 300),
                ("token-frequency-plot-full", "build_token_frequency_plot.py", [], 300),
            ]:
                if phase == "step5-full-verification" and args.reuse_step5_verification:
                    evidence = reusable_tokenization_verification(ROOT)
                    state["completed_phases"].append(phase)
                    update(phase=phase, reused_step5_verification=evidence)
                    print("Reusing passed full Step 5 verification: all 32 bound inputs/outputs unchanged.", flush=True)
                    continue
                run(phase, [python, "-u", ROOT / script, *arguments], timeout)
            coverage = json.loads((reports / "vocabulary-coverage.json").read_text())
            numeric = next(row for row in coverage["variants"] if row["id"] == "numeric")
            vocabs = sorted({16384, 32768, 49152, 65000, 65536, numeric["candidate_knee"]["k"]})
            benchmark_path = reports / "nplm-benchmark.json"
            if not benchmark_path.exists():
                run("nplm-benchmark", [sys.executable, "-u", ROOT / "benchmark_nplm.py", "--devices", "cpu", "mps",
                                       "--cpu-threads", 4, "--vocab-sizes", *vocabs], 3600)
            benchmark = json.loads(benchmark_path.read_text())
            decision = choose_configuration(coverage, benchmark)
            update(decision=decision)
            command = [sys.executable, "-u", ROOT / "run_training_comparison.py", "--vocab-size", decision["vocab_size"],
                       "--device", decision["device"]]
            queue_path = reports / "training-queue.json"
            if queue_path.exists():
                queue = json.loads(queue_path.read_text())
                if queue["status"] == "complete":
                    update(status="complete", phase="complete")
                    return
                command.append("--resume")
            run("training-comparison-queue", command, 16 * 3600)
            queue = json.loads(queue_path.read_text())
            update(status=queue["status"], phase="complete" if queue["status"] == "complete" else queue["phase"])
        except InterruptedError:
            update(status="stopped", child_pid=None)
        except BaseException as error:
            update(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None)
            raise


if __name__ == "__main__":
    main()
