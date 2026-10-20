"""Build new ILP artifacts without rebuilding or replacing old executables.

Only run this between model observations while holding the shared timing lock.
The previous build manifest supplies the compiler/math/backend flags unchanged.
"""

from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

WIKI = Path(__file__).resolve().parent.parent
EXP = WIKI / "cpu-benchmark/experiments"
NEW = EXP / "round2"
MANIFEST = WIKI / "reports/cpu-ilp-round2-build-20261018-v1.json"


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    if MANIFEST.exists():
        raise FileExistsError("Preserve prior build result")
    old_manifest = EXP / "fixedwork-build-manifest.v2.json"
    previous = json.loads(old_manifest.read_text())
    command = previous["builds"]["combined"]["command"] if "builds" in previous else previous["variants"]["combined"]["command"]
    binary = NEW / "cpu_fixedwork_ilp"
    unit = NEW / "test_softmax_ilp"
    for path in (binary, unit):
        if path.exists():
            raise FileExistsError(f"Preserve {path}")
    sources = [Path(__file__), old_manifest, NEW / "softmax_ilp.c", NEW / "test_softmax_ilp.c",
               EXP / "cpu_fixedwork_combined", *EXP.glob("*.c"), *EXP.glob("*.h")]
    before = {str(p): identity(p) for p in sources}
    report = {"kind": "CPU_SOFTMAX_ILP_BUILD", "status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "sources_before": before, "builds": [], "policy": "Only linked softmax source and output path differ from the frozen combined build; no global fast-math or contraction change."}
    with (WIKI / "cpu-benchmark/.timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            candidate = [str(NEW / "softmax_ilp.c") if value == str(EXP / "fixedwork_softmax.c") else
                         str(binary) if value == str(EXP / "cpu_fixedwork_combined") else value for value in command]
            candidate.insert(1, "-I" + str(EXP))
            diagnostic = [command[0], "-O3", "-std=c11", "-ffp-contract=off", "-fno-math-errno", "-mcpu=native",
                          "-DEXPERIMENT_VFORCE_EXP=1", "-Xpreprocessor", "-fopenmp", "-isystem",
                          "/opt/homebrew/opt/libomp/include", "-I" + str(EXP), "-Rpass-missed=loop-vectorize",
                          "-c", str(EXP / "fixedwork_softmax.c"), "-o", str(NEW / "baseline_softmax.o")]
            baseline_object = list(diagnostic)
            baseline_object.insert(1, "-Dfw_softmax=baseline_fw_softmax")
            objects = [NEW / "baseline_softmax.o", NEW / "candidate_softmax.o", NEW / "vector_vforce.o"]
            common = [command[0], "-O3", "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-ffp-contract=off",
                      "-fno-math-errno", "-mcpu=native", "-DEXPERIMENT_VFORCE_EXP=1", "-Xpreprocessor", "-fopenmp",
                      "-isystem", "/opt/homebrew/opt/libomp/include", "-I" + str(EXP)]
            commands = [candidate, baseline_object,
                        [*common, "-c", str(NEW / "softmax_ilp.c"), "-o", str(objects[1])],
                        [*common, "-c", str(EXP / "vector_vforce.c"), "-o", str(objects[2])],
                        [*common, str(NEW / "test_softmax_ilp.c"), *map(str, objects), "-framework", "Accelerate",
                         "-L/opt/homebrew/opt/libomp/lib", "-Wl,-rpath,/opt/homebrew/opt/libomp/lib", "-lomp", "-lm", "-o", str(unit)]]
            for cmd in commands:
                result = subprocess.run(cmd, cwd=NEW, text=True, capture_output=True, timeout=120)
                report["builds"].append({"command": cmd, "exit_code": result.returncode,
                                         "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode:
                    raise RuntimeError("Compilation failed; see preserved build record")
            report.update(status="complete", binary={"path": str(binary), **identity(binary)},
                          unit_binary={"path": str(unit), **identity(unit)})
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["sources_after"] = {str(p): identity(p) for p in sources}
        if report["sources_after"] != before:
            report.update(status="failed", error="A source or old binary changed during build")
        report["completed_at"] = datetime.now().astimezone().isoformat()
        MANIFEST.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("status", "error", "binary", "unit_binary")}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
