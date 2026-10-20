"""Compare bounded, identical real-Wiki C work across compiler optimizations.

This changes compilation of our C loops only.  The linked Apple Accelerate
library is unchanged and is not rebuilt by these compiler options.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
COMMON_FLAGS = ["-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                "-ffp-contract=off", "-fno-math-errno"]
VARIANTS = [("O0", ["-O0"]), ("O3", ["-O3"]),
            ("native", ["-O3", "-mcpu=native"])]


def identity(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def capture(command, *, timeout=60, env=None):
    process = subprocess.Popen(command, cwd=BASE, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # The time(1) wrapper has a benchmark child. Bound the entire process
        # group so an expired wrapper cannot leave model updates running.
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return {"command": command, "exit_code": process.returncode,
            "stdout": stdout, "stderr": stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--corpus", type=Path,
                        default=BASE.parent / "training-corpora/wiki-k35328")
    parser.add_argument("--python", type=Path,
                        default=BASE.parent / ".venv-training/bin/python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This comparison explicitly targets Mac/Apple Accelerate")
    if args.warmup_steps not in {1, 2}:
        parser.error("Use the same bounded 1 or 2 warm-up steps for every trial")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing comparison report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the virtualenv executable path: resolving its symlink invokes the
    # base interpreter and drops the virtualenv's installed torch reference.
    corpus, python = args.corpus.resolve(), args.python.absolute()
    source, makefile = BASE / "cpu_benchmark.c", BASE / "Makefile"
    source_identity, makefile_identity = identity(source), identity(makefile)
    checker_identity = identity(BASE / "check_correctness.py")
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["vocab_size"] != 35328:
        raise ValueError("Require the complete preserved K=35,328 Wiki corpus")
    export = json.loads((BASE.parent / "reports/integer-encoding-npy.json").read_text())
    if identity(manifest_path)["sha256"] != export["source_manifest"]["sha256"]:
        raise ValueError("Corpus manifest differs from its verified NumPy export")
    report = {
        "status": "running", "kind": "REAL_WIKI_C_COMPILER_FLAG_MICROBENCHMARK",
        "started_at": datetime.now().astimezone().isoformat(),
        "scope": "Same short RAM-resident Wiki batches and C algorithm; no full epoch, saved model, quality evaluation, GPU or host selection.",
        "source_before": {"path": str(source), **source_identity},
        "makefile_before": makefile_identity,
        "checker": {"path": str(BASE / "check_correctness.py"), **checker_identity},
        "compiler": capture([args.compiler, "--version"]),
        "host": {"os": platform.platform(), "architecture": platform.machine(),
                 "logical_cpus": os.cpu_count(),
                 "cpu": capture(["sysctl", "-n", "machdep.cpu.brand_string"])["stdout"].strip(),
                 "physical_memory_bytes": int(capture(["sysctl", "-n", "hw.memsize"])["stdout"]),
                 "load_average_before": list(os.getloadavg()),
                 "process_nice": os.getpriority(os.PRIO_PROCESS, 0)},
        "corpus": {"path": str(corpus), "manifest": identity(manifest_path),
                   "full_train_target_windows": manifest["statistics"]["train_target_windows"],
                   "arrays": [{"split": row["split"], "shape": row["shape"],
                               "output": row["output"]} for row in export["arrays"]]},
        "fixed_configuration": {"vocab_size": 35328, "context_size": 8,
                                "embedding_dim": 64, "hidden_dim": 128,
                                "batch_size": 128, "seed": 42,
                                "requested_threads": 4, "actual_accelerate_workers": None,
                                "warmup_steps": args.warmup_steps, "repeats": 3,
                                "dtype": "float32", "optimizer": "AdamW",
                                "gradient_max_norm": 1.0},
        "compiler_scope": "Only this C translation unit changes; already optimized Apple Accelerate/CBLAS is linked unchanged and is not rebuilt by -O0/-O3/native.",
        "thread_scope": "Requested environment budget 4; actual Accelerate worker count is not measured.",
        "timing_scope": "Timed C window gather, forward/backward, clipping and AdamW; excludes owned-RAM file load/validation, model setup, warm-up, reset and final loss check. Process elapsed is separately recorded. OS cache and usual GUI load are uncontrolled.",
        "parallelism": "Compiler builds may overlap; all numerical checks, pilots and measured model processes run serially under an exclusive .timing.lock.",
        "builds": [], "correctness": [], "pilots": [], "trial_order": [],
        "trials": [], "summary": [],
    }

    def save():
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n")
        os.replace(temporary, output)

    def unchanged():
        if identity(source) != source_identity or identity(makefile) != makefile_identity:
            raise RuntimeError("C source/Makefile changed during the comparison")
        if identity(BASE / "check_correctness.py") != checker_identity:
            raise RuntimeError("Independent checker changed during the comparison")

    def build(item):
        name, optimizations = item
        flags = optimizations + COMMON_FLAGS
        binary = BASE / f"cpu_benchmark_{name}"
        command = [args.compiler, "-DACCELERATE_NEW_LAPACK",
                   "-DBENCH_BUILD_FLAGS=" + json.dumps(" ".join(flags)),
                   *flags, str(source), "-o", str(binary),
                   "-framework", "Accelerate", "-lm"]
        result = capture(command)
        result.update(variant=name, flags=flags,
                      supported=result["exit_code"] == 0)
        if result["supported"]:
            result["binary"] = {"path": str(binary), **identity(binary)}
            result["linked_libraries"] = capture(["otool", "-L", str(binary)])["stdout"]
        return result

    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="4", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def execute(variant, steps):
        unchanged()
        binary = BASE / f"cpu_benchmark_{variant}"
        command = [str(binary), "--corpus", str(corpus), "--threads", "4",
                   "--seed", "42", "--batch-size", "128",
                   "--warmup-steps", str(args.warmup_steps), "--steps", str(steps)]
        before_load = list(os.getloadavg())
        start = time.monotonic()
        timed_command = ["/usr/bin/time", "-l", *command]
        result = capture(timed_command, timeout=180, env=env)
        duration = time.monotonic() - start
        if result["exit_code"]:
            raise RuntimeError(f"{variant} timing process failed: {result['stderr']}")
        row = json.loads(result["stdout"])
        if (row.get("kind") != "REAL_WIKI_C_MICROBENCHMARK"
                or row.get("finite_flag") is not True
                or row["steps"] != steps or row["target_prediction_events"] != steps * 128
                or row["corpus"]["target_windows"] != report["corpus"]["full_train_target_windows"]
                or row["elapsed_seconds"] <= 0):
            raise ValueError(f"Invalid {variant} measurement")
        rss = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", result["stderr"])
        if rss is None:
            raise ValueError("macOS /usr/bin/time did not report per-process maximum RSS")
        row.update(variant=variant, command=command, timing_wrapper_command=timed_command,
                   process_wall_seconds=duration, peak_rss_bytes=int(rss.group(1)),
                   peak_rss_scope="macOS /usr/bin/time -l per-process maximum resident set size, including RAM corpus/model/setup/warm-up",
                   load_average_before=before_load, load_average_after=list(os.getloadavg()),
                   stderr=result["stderr"])
        return row

    save()
    with (BASE / ".timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print("Building O0/O3/native variants; numerical runs remain serialized.", flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                report["builds"] = list(pool.map(build, VARIANTS))
            unchanged()
            variants = [row["variant"] for row in report["builds"] if row["supported"]]
            if not {"O0", "O3"}.issubset(variants):
                raise RuntimeError("Both baseline O0 and O3 must compile")
            save()
            for variant in variants:
                print(f"Independent tiny numerical check: {variant}", flush=True)
                result = capture([str(python), str(BASE / "check_correctness.py"),
                                  "--executable", str(BASE / f"cpu_benchmark_{variant}")],
                                 timeout=90, env=env)
                checked = json.loads(result["stdout"])
                report["correctness"].append({"variant": variant,
                    "exit_code": result["exit_code"], "report": checked,
                    "stderr": result["stderr"]})
                save()
                if result["exit_code"] or checked.get("passed") is not True:
                    raise ValueError(f"Independent tiny check failed: {variant}")
            for variant in ["O0", "O3"]:
                print(f"Excluded 2-step pilot: {variant}", flush=True)
                report["pilots"].append(execute(variant, 2))
                save()
            slow_step = max(row["elapsed_seconds"] / row["steps"] for row in report["pilots"])
            choices = [steps for steps in [8, 16, 32] if slow_step * steps <= 15]
            steps = max(choices) if choices else 8
            report["pilot_selection"] = {
                "measured_steps": steps, "slow_pilot_seconds_per_step": slow_step,
                "predicted_slow_trial_seconds": slow_step * steps,
                "rule": "Largest common 8/16/32 step count predicted <=15s for the slower pilot; pilots and warm-up excluded from final medians.",
            }
            report["fixed_configuration"]["measured_steps"] = steps
            save()
            orders = [variants, list(reversed(variants)), variants[1:] + variants[:1]]
            reference = None
            for repeat, order in enumerate(orders, 1):
                report["trial_order"].append({"repeat": repeat, "variants": order})
                for variant in order:
                    print(f"Measured repeat {repeat}/3, {variant}, {steps} steps", flush=True)
                    row = execute(variant, steps)
                    row["repeat"] = repeat
                    common = (row["initial_parameter_checksum"], row["input_checksum"],
                              row["corpus"]["first_windows"])
                    if reference is None:
                        reference = common
                    elif common != reference:
                        raise ValueError("Initial parameters/real-data batch stream differ across trials")
                    report["trials"].append(row)
                    save()
                    print(f"  {row['elapsed_seconds']:.6f}s; {row['targets_per_second']:.2f} targets/s", flush=True)
            baseline = statistics.median(row["elapsed_seconds"] for row in report["trials"]
                                         if row["variant"] == "O0")
            for variant in variants:
                rows = [row for row in report["trials"] if row["variant"] == variant]
                elapsed = [row["elapsed_seconds"] for row in rows]
                rates = [row["targets_per_second"] for row in rows]
                median_elapsed = statistics.median(elapsed)
                rate = statistics.median(rates)
                report["summary"].append({
                    "variant": variant, "measured_steps_per_trial": steps,
                    "target_prediction_events_per_trial": steps * 128, "repeats": len(rows),
                    "median_elapsed_seconds": median_elapsed,
                    "min_elapsed_seconds": min(elapsed), "max_elapsed_seconds": max(elapsed),
                    "median_targets_per_second": rate,
                    "median_speedup_relative_to_O0": baseline / median_elapsed,
                    "median_corpus_load_seconds": statistics.median(row["corpus"]["corpus_load_seconds"] for row in rows),
                    "median_process_wall_seconds": statistics.median(row["process_wall_seconds"] for row in rows),
                    "maximum_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
                })
            # A checksum binds only identical initial inputs; allow numerical BLAS
            # reductions to differ after updates, and compare finite losses explicitly.
            losses = [row["final_loss"] for row in report["trials"]]
            if not all(math.isclose(loss, losses[0], rel_tol=2e-5, abs_tol=2e-5) for loss in losses):
                raise ValueError("Final losses exceed the cross-variant sanity tolerance")
            report["same_work_validation"] = {
                "same_initial_parameter_checksum": True, "same_first_batch_checksum": True,
                "same_first_four_real_windows": True,
                "same_seed_sampler_measured_steps": True,
                "finite_final_losses_close": True,
                "final_loss_min": min(losses), "final_loss_max": max(losses),
                "final_loss_atol": 2e-5, "final_loss_rtol": 2e-5,
            }
            unchanged()
            report.update(status="complete", completed_at=datetime.now().astimezone().isoformat())
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}",
                          completed_at=datetime.now().astimezone().isoformat())
            raise
        finally:
            report["source_after"] = identity(source)
            report["makefile_after"] = identity(makefile)
            report["source_unchanged"] = report["source_after"] == source_identity
            report["makefile_unchanged"] = report["makefile_after"] == makefile_identity
            report["host"]["load_average_after"] = list(os.getloadavg())
            save()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
