"""Sequential 14-hour comparison: 10%/3h, 25%/3h, 100%/8h with a 3h checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark_nplm import assert_pipeline_idle
from build_training_comparison import collect
from training_corpus import atomic_json, identity


ROOT = Path(__file__).resolve().parent


def execute(vocab_size, device, resume=False):
    assert_pipeline_idle(ROOT)
    benchmark = json.loads((ROOT / "reports/nplm-benchmark.json").read_text())
    if benchmark["status"] not in {"complete", "completed_with_failures"}:
        raise ValueError("Wait until bounded hardware profiling has finished")
    matches = [row for row in benchmark["results"] if row.get("status") == "complete"
               and row["config"]["vocab_size"] == vocab_size and row["config"]["device"] == device]
    if len(matches) != 1:
        raise ValueError("The selected vocabulary/device must have one successful measured profile")
    root = ROOT / "training-runs" / f"wiki-k{vocab_size}"
    corpus = ROOT / "training-corpora" / f"wiki-k{vocab_size}"
    queue_path = ROOT / "reports/training-queue.json"
    if queue_path.exists() and not resume:
        raise FileExistsError("Preserving existing comparison queue; explicitly resume it")
    root.mkdir(parents=True, exist_ok=True)
    plan = {"vocab_size": vocab_size, "device": device, "root": str(root), "corpus": str(corpus),
            "nominal_total_training_hours": 14, "benchmark_identity": identity(ROOT / "reports/nplm-benchmark.json"),
            "scripts": {name: identity(ROOT / name) for name in ("run_training_comparison.py", "train_nplm.py",
                                                               "prepare_training_experiment.py", "training_corpus.py")},
            "runs": [{"name": "data-10pct", "fraction": .1, "hours": 3},
                     {"name": "data-25pct", "fraction": .25, "hours": 3},
                     {"name": "data-100pct", "fraction": 1.0, "hours": 8}]}
    if resume:
        old = json.loads(queue_path.read_text())
        if old["plan"] != plan:
            raise ValueError("Queue plan/scripts differ from the original run")
        if old["status"] == "complete":
            raise ValueError("Comparison queue is already complete")
    state = {"status": "running", "phase": "prepare", "plan": plan, "completed_runs": [],
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(queue_path, state)
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def launch(command, logfile):
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
        with logfile.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen([sys.executable, "-u", *map(str, command)], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT, env=environment)
            state["child_pid"] = child.pid
            atomic_json(queue_path, state)
            while child.poll() is None:
                if stop["requested"] or (root / "STOP").exists():
                    stop["requested"] = True
                    child.terminate()
                    # Training handles SIGTERM and preserves a checkpoint.
                    try:
                        child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                collect(root, corpus, ROOT / "reports/training-comparison")
                time.sleep(30)
        state.pop("child_pid", None)
        if stop["requested"]:
            return False
        if child.returncode:
            raise RuntimeError(f"Child failed ({child.returncode}); see {logfile}")
        return True
    try:
        if not (corpus / "manifest.json").exists():
            if not launch([ROOT / "prepare_training_experiment.py", "--output", corpus, "--vocab-size", vocab_size], root / "prepare.log"):
                state["status"] = "stopped"
                return
        state["phase"] = "pilot"
        atomic_json(queue_path, state)
        pilot = root / "pilot-64steps"
        pilot_summary = json.loads((pilot / "summary.json").read_text()) if (pilot / "summary.json").exists() else None
        if pilot_summary is None or pilot_summary["status"] not in {"complete", "step-limited-pilot-complete"}:
            command = [ROOT / "train_nplm.py", "--corpus", corpus, "--output", pilot, "--device", device,
                       "--hours", .025, "--checkpoint-hours", .01, "--evaluation-windows", 256, "--max-steps", 64]
            if (pilot / "latest.pt").exists():
                command.append("--resume")
            if not launch(command, root / "pilot.log"):
                state["status"] = "stopped"
                return
        for row in plan["runs"]:
            directory = root / row["name"]
            if (directory / "summary.json").exists():
                summary = json.loads((directory / "summary.json").read_text())
                if summary["status"] == "complete":
                    state["completed_runs"].append(row["name"])
                    continue
            state["phase"] = row["name"]
            atomic_json(queue_path, state)
            command = [ROOT / "train_nplm.py", "--corpus", corpus, "--output", directory,
                       "--fraction", row["fraction"], "--device", device, "--hours", row["hours"],
                       "--checkpoint-hours", 1, 3]
            if row["hours"] == 8:
                command.append(8)
            if (directory / "latest.pt").exists():
                command.append("--resume")
            if not launch(command, root / f"{row['name']}.log"):
                state["status"] = "stopped"
                return
            summary = json.loads((directory / "summary.json").read_text())
            if summary["status"] != "complete":
                state["status"] = "stopped"
                return
            state["completed_runs"].append(row["name"])
        collect(root, corpus, ROOT / "reports/training-comparison")
        state.update(status="complete", phase="complete", completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(queue_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with (ROOT / "reports/.training-comparison.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.vocab_size, args.device, args.resume)
