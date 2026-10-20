"""Prepare a visual from the completed, same-work CPU ABBA benchmark.

This reads measurements only. It never launches an engine, benchmark, browser,
model, compiler, or training job. Run it after all timing trials have ended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "reports/fixedwork-cpu-comparison.template.html"
PLACEHOLDER = "__DATA__"
CANDIDATES = ("fixed_norm", "softmax_inv_batch", "minimal_zero", "combined")
EPOCH5_TARGETS = 621_162_675
EXPECTED_CONFIG = {"V": 35328, "C": 8, "D": 64, "H": 128, "B": 128, "threads": 4}
CONFIG_DISPLAY_KEYS = {"V": "vocab_size", "C": "context_size", "D": "embedding_dim", "H": "hidden_dim", "B": "batch_size"}


def _number(value: object, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)


def _integer(value: object, label: str, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(f"Invalid integer {label}: {value!r}")
    return value


def _close(actual: object, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-5):
        raise ValueError(f"Inconsistent {label}: {actual!r} vs {expected!r}")


def validated_payload(report: dict) -> dict:
    if report.get("kind") != "REAL_WIKI_MAC_FIXED_WORK_ABBA" or report.get("status") != "complete":
        raise ValueError("A completed real-Wiki same-work ABBA measurement is required")
    source_config = report.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("The measured configuration is missing")
    # Keep the runner's original long-name fields and normalize a separate
    # display copy; the source report is never mutated.
    config = {**source_config, **{short: source_config.get(long) for short, long in CONFIG_DISPLAY_KEYS.items()}}
    if any(config.get(key) != value for key, value in EXPECTED_CONFIG.items()):
        raise ValueError("The measured V/C/D/H/B/thread configuration does not match this comparison")
    _integer(config.get("seed"), "config.seed", positive=False)
    steps = _integer(report.get("main_steps"), "main_steps")
    events = steps * config["B"]
    if report.get("events_per_trial") != events:
        raise ValueError("All trials must contain exactly N * B target-prediction events")
    trials = report.get("trials")
    comparisons = report.get("comparisons")
    if not isinstance(trials, list) or len(trials) != 16 or not isinstance(comparisons, list) or len(comparisons) != 4:
        raise ValueError("Four complete ABBA blocks (16 main trials) are required; pilots do not count")
    by_id = {}
    trimmed_trials = []
    for trial in trials:
        identifier = trial.get("trial_id")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("Main trial IDs must be unique nonempty strings")
        if trial.get("block") not in CANDIDATES or trial.get("position") not in (1, 2, 3, 4):
            raise ValueError(f"Invalid ABBA block or position in {identifier}")
        if trial.get("passed") is not True or trial.get("inputs_equal") is not True or trial.get("phase_accounting_passed") is not True:
            raise ValueError(f"A measurement/input/accounting gate did not pass: {identifier}")
        measurement = trial.get("measurement")
        if not isinstance(measurement, dict) or measurement.get("steps") != steps or measurement.get("target_prediction_events") != events:
            raise ValueError(f"Trial {identifier} did not perform the same measured work")
        elapsed = _number(measurement.get("elapsed_seconds"), f"{identifier}.elapsed_seconds", positive=True)
        _number(measurement.get("process_cpu_seconds"), f"{identifier}.process_cpu_seconds")
        rate = _number(measurement.get("targets_per_second"), f"{identifier}.targets_per_second", positive=True)
        _close(rate, events / elapsed, f"{identifier}.targets_per_second")
        by_id[identifier] = trial
        trimmed_trials.append({"trial_id": identifier, "block": trial["block"], "position": trial["position"],
                               "variant": trial.get("variant"), "passed": True, "inputs_equal": True, "phase_accounting_passed": True,
                               "measurement": {key: measurement.get(key) for key in ("elapsed_seconds", "process_cpu_seconds", "thread_cpu_seconds", "thread_clock_available", "targets_per_second", "steps", "target_prediction_events")},
                               "process_wall_seconds": trial.get("process_wall_seconds"), "peak_rss_bytes": trial.get("peak_rss_bytes")})
    if {item.get("candidate") for item in comparisons} != set(CANDIDATES):
        raise ValueError("All three independent candidates and their combination are required")
    trimmed_comparisons = []
    for comparison in comparisons:
        candidate = comparison["candidate"]
        block = sorted((trial for trial in trials if trial["block"] == candidate), key=lambda trial: trial["position"])
        if len(block) != 4 or [trial["position"] for trial in block] != [1, 2, 3, 4] or [trial.get("variant") for trial in block] != ["baseline", candidate, candidate, "baseline"]:
            raise ValueError(f"Block {candidate} is not baseline/candidate/candidate/baseline")
        control_ids = [block[0]["trial_id"], block[3]["trial_id"]]
        candidate_ids = [block[1]["trial_id"], block[2]["trial_id"]]
        if comparison.get("control_trial_ids") != control_ids or comparison.get("candidate_trial_ids") != candidate_ids:
            raise ValueError(f"Comparison {candidate} does not use its own neighboring controls")
        control_times = [by_id[identifier]["measurement"]["elapsed_seconds"] for identifier in control_ids]
        candidate_times = [by_id[identifier]["measurement"]["elapsed_seconds"] for identifier in candidate_ids]
        candidate_rates = [by_id[identifier]["measurement"]["targets_per_second"] for identifier in candidate_ids]
        control_mean = statistics.mean(control_times)
        _close(comparison.get("control_elapsed_seconds"), control_mean, f"{candidate}.local_control_mean")
        _close(comparison.get("control_effective_targets_per_second"), events / control_mean, f"{candidate}.local_control_effective_rate")
        supplied_times = comparison.get("candidate_elapsed_seconds")
        if not isinstance(supplied_times, list) or len(supplied_times) != 2:
            raise ValueError(f"Two candidate elapsed times are required: {candidate}")
        for actual, expected in zip(supplied_times, candidate_times):
            _close(actual, expected, f"{candidate}.candidate_elapsed_seconds")
        savings = [100 * (1 - elapsed / control_mean) for elapsed in candidate_times]
        per_trial = comparison.get("per_trial")
        if not isinstance(per_trial, list) or [item.get("trial_id") for item in per_trial] != candidate_ids:
            raise ValueError(f"Two ordered per-trial savings records are required: {candidate}")
        for item, saving in zip(per_trial, savings):
            _close(item.get("wall_saved_percent"), saving, f"{candidate}.per_trial_wall_saved_percent")
        _close(comparison.get("wall_saved_percent_min"), min(savings), f"{candidate}.saved_percent_min")
        _close(comparison.get("wall_saved_percent_max"), max(savings), f"{candidate}.saved_percent_max")
        first_rate, last_rate = (by_id[identifier]["measurement"]["targets_per_second"] for identifier in control_ids)
        drift = max(abs(last_rate - first_rate) / first_rate * 100,
                    abs(control_times[1] - control_times[0]) / control_times[0] * 100)
        if comparison.get("high_control_drift") is not (drift > 20):
            raise ValueError(f"The 20% local-control drift flag is inconsistent: {candidate}")
        extrapolation = comparison.get("extrapolation_only")
        if not isinstance(extrapolation, dict) or extrapolation.get("epoch5_targets") != EPOCH5_TARGETS:
            raise ValueError(f"The five-epoch extrapolation scope is missing: {candidate}")
        hours = extrapolation.get("candidate_epoch5_hours_per_trial")
        if not isinstance(hours, list) or len(hours) != 2:
            raise ValueError(f"Two candidate short-run extrapolations are required: {candidate}")
        for actual, rate in zip(hours, candidate_rates):
            _close(actual, EPOCH5_TARGETS / rate / 3600, f"{candidate}.candidate_extrapolated_hours")
        _close(extrapolation.get("control_epoch5_hours"), EPOCH5_TARGETS / (events / control_mean) / 3600, f"{candidate}.control_extrapolated_hours")
        trimmed = {key: comparison.get(key) for key in ("candidate", "status", "control_trial_ids", "candidate_trial_ids", "control_elapsed_seconds", "control_effective_targets_per_second", "control_endpoint_rate_arithmetic_mean", "control_drift_percent", "high_control_drift", "candidate_elapsed_seconds", "candidate_process_cpu_seconds", "candidate_targets_per_second", "wall_saved_percent_min", "wall_saved_percent_max", "wall_saved_percent_median", "process_cpu_saved_percent_median", "per_trial", "extrapolation_only")}
        # This is the independently checked larger drift across time and rate.
        # Keep the runner's original drift field as well for review.
        trimmed["checked_local_control_drift_percent"] = drift
        trimmed_comparisons.append(trimmed)
    return {"kind": report["kind"], "status": "complete", "main_steps": steps, "events_per_trial": events,
            "config": config, "trials": trimmed_trials, "comparisons": trimmed_comparisons,
            "visual_validation": {"same_work_checked": True, "local_control_savings_checked": True,
                                  "wall_rate_extrapolation_checked": True, "cpu_seconds_not_used_for_extrapolation": True}}


def build_visual(input_path: str | Path, output_path: str | Path, template: str | Path = TEMPLATE) -> dict:
    report = json.loads(Path(input_path).read_text(encoding="utf-8"))
    payload = validated_payload(report)
    markup = Path(template).read_text(encoding="utf-8")
    if markup.count(PLACEHOLDER) != 1:
        raise ValueError("The fixed-work template must contain one __DATA__ placeholder")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    markup = markup.replace(PLACEHOLDER, encoded)
    if len(markup.encode("utf-8")) > 1_000_000:
        raise ValueError("The fixed-work visual exceeds the 1 MB limit")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{output_path.name}-", suffix=".partial", dir=output_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(markup)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "complete", "main_trials": len(payload["trials"]), "steps_per_trial": payload["main_steps"],
            "events_per_trial": payload["events_per_trial"], "output": str(output_path), "html_bytes": output_path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cpu-benchmark/results/cpu-fixedwork-mac-20261018-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/fixedwork-cpu-comparison.html")
    args = parser.parse_args()
    print(json.dumps(build_visual(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
