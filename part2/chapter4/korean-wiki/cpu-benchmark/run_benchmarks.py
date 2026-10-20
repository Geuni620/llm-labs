"""Run bounded, fresh-process C CPU timing trials; never run a full epoch."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=base / "cpu_benchmark")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[4])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not 1 <= args.steps <= 512 or not 1 <= args.repeats <= 5
            or not 0 <= args.warmup_steps <= 16
            or any(not 1 <= n <= 8 for n in args.threads)):
        parser.error("Require bounded timing: steps1..512, repeats1..5, warmup0..16, threads1..8")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite timing evidence: {args.output}")
    binary = args.binary.resolve()
    corpus = args.corpus.resolve() if args.corpus else None
    report = {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "purpose": "Fixed-work CPU timing before selecting a computer; no full-epoch or model-quality measurement.",
        "host": {"system": platform.system(), "release": platform.release(),
                 "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                 "load_average_before": list(os.getloadavg()), "process_nice": os.getpriority(os.PRIO_PROCESS, 0)},
        "binary": {"path": str(binary), **identity(binary)},
        "source": identity(base / "cpu_benchmark.c"),
        "trials": [], "summary": [],
    }
    if corpus:
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        report["corpus"] = {"path": str(corpus), "manifest": identity(manifest_path),
                            "full_train_target_windows": manifest["statistics"]["train_target_windows"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, args.output)

    save()
    try:
        for threads in dict.fromkeys(args.threads):
            for repeat in range(args.repeats):
                command = [str(binary), "--threads", str(threads), "--steps", str(args.steps),
                           "--warmup-steps", str(args.warmup_steps)]
                if corpus:
                    command += ["--corpus", str(corpus)]
                start = time.monotonic()
                result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=180)
                row = json.loads(result.stdout)
                if not row.get("finite_flag") or row["steps"] != args.steps:
                    raise ValueError("Invalid or incomplete C timing result")
                row.update(repeat=repeat + 1, process_wall_seconds=time.monotonic() - start,
                           command=command, stderr=result.stderr)
                report["trials"].append(row)
                save()
                print(f"threads={threads} repeat={repeat + 1}: {row['elapsed_seconds']:.3f}s, "
                      f"{row['targets_per_second']:.2f} targets/s", flush=True)
            rows = [row for row in report["trials"] if row["requested_threads"] == threads]
            elapsed = [row["elapsed_seconds"] for row in rows]
            rates = [row["targets_per_second"] for row in rows]
            summary = {"requested_threads": threads, "measured_steps_per_trial": args.steps,
                       "repeats": len(rows), "median_elapsed_seconds": statistics.median(elapsed),
                       "min_elapsed_seconds": min(elapsed), "max_elapsed_seconds": max(elapsed),
                       "median_targets_per_second": statistics.median(rates)}
            if corpus:
                targets = report["corpus"]["full_train_target_windows"]
                summary["extrapolation_only"] = {
                    "epoch1_hours": targets / statistics.median(rates) / 3600,
                    "epoch5_hours": 5 * targets / statistics.median(rates) / 3600,
                    "caveat": "Short RAM-resident C timing extrapolated by target count; not a measured full epoch, long-run speed, quality result or production trainer completion.",
                }
            report["summary"].append(summary)
        report.update(status="complete", completed_at=datetime.now().astimezone().isoformat())
    except Exception as error:
        report.update(status="failed", error=str(error), completed_at=datetime.now().astimezone().isoformat())
        raise
    finally:
        report["host"]["load_average_after"] = list(os.getloadavg())
        save()


if __name__ == "__main__":
    main()
