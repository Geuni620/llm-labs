"""Observe an existing training queue after the T3 parent process disappears.

Never starts or restarts data/model jobs. Keeps the existing queue awake and
returns its terminal status so the new T3 session receives a completion event.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from training_corpus import atomic_json


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main():
    with (REPORTS / ".training-queue-observer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path, queue_path = REPORTS / "experiment-continuation.json", REPORTS / "training-queue.json"
        state, queue = json.loads(state_path.read_text()), json.loads(queue_path.read_text())
        queue_pid = state.get("child_pid")
        if type(queue_pid) is not int or queue.get("status") != "running":
            raise RuntimeError("No running existing queue to adopt")
        process = subprocess.run(["ps", "-p", str(queue_pid), "-o", "stat=,command="],
                                 capture_output=True, text=True, check=True).stdout.strip()
        if process.startswith("Z") or str(ROOT / "run_training_comparison.py") not in process:
            raise RuntimeError("Recorded PID is not this workspace's live training queue")
        state["supervision"] = {"mode": "adopt_existing_queue", "observer_pid": os.getpid(),
                                "queue_pid": queue_pid, "previous_parent_cancelled_by_server_restart": True,
                                "training_restarted": False, "queue_restarted": False,
                                "adopted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        state["updated_at"] = state["supervision"]["adopted_at"]
        atomic_json(state_path, state)
        atomic_json(REPORTS / "server-restart-recovery.json", state["supervision"])
        print(f"Adopted existing queue PID {queue_pid}; model/data processes unchanged.", flush=True)
        # -w observes the existing PID; it does not execute a new utility/model.
        subprocess.run(["/usr/bin/caffeinate", "-i", "-w", str(queue_pid)], check=True)
        queue = json.loads(queue_path.read_text())
        state = json.loads(state_path.read_text())
        state.update(status=queue["status"], child_pid=None,
                     phase="complete" if queue["status"] == "complete" else queue["phase"],
                     updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if queue["status"] == "complete":
            state["completed_phases"] = list(dict.fromkeys(state["completed_phases"] + ["training-comparison-queue"]))
        atomic_json(state_path, state)
        if queue["status"] not in {"complete", "stopped"}:
            raise RuntimeError(f"Observed queue exited without completion: {queue['status']}")
        print(f"Existing queue finished: {queue['status']}; result delivery is checked separately.", flush=True)


if __name__ == "__main__":
    main()
