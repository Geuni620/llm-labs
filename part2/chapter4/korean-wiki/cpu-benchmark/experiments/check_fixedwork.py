#!/usr/bin/env python3
"""Independent one-step and four-update PyTorch checks for fixed-work kernels.

Default scope per binary: eight ordinary one-step checks plus eight four-update
checks with gradient poisoning. Five feature builds therefore produce 80 cases
in one Python process. C child calculations and the CPU PyTorch reference are
serialized under cpu-benchmark/.timing.lock. This program never times training,
compiles a binary, reads a Wiki corpus, or runs a full training experiment.

Moment tolerances are deliberately tighter than ordinary model-value limits.
They are declared before execution; all-zero moments and unwritten poisoned
gradient buffers cannot be accepted merely because their values are small.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import check_variants as common


FEATURE_KEYS = ("fixed_norm", "softmax_inv_batch", "minimal_zero")
FEATURE_NAMES = ("baseline", *FEATURE_KEYS, "combined")
OMP_ARGUMENTS = ("--softmax-parallel", "--reduction-parallel", "--adam-parallel")
SCENARIOS = {
    "one_step": {"check_updates": 1, "poison_gradients": False},
    "four_update_poison": {"check_updates": 4, "poison_gradients": True},
}
MOMENT1_ATOL, MOMENT1_RTOL = 2e-9, 2e-4
MOMENT2_ATOL, MOMENT2_RTOL = 2e-11, 5e-4
TOLERANCES = {
    "rule": "abs(actual-reference) <= atol + rtol*abs(reference)",
    "ordinary_loss_norm_scale_gradients_parameters": {"atol": common.ATOL, "rtol": common.RTOL},
    "moment1": {"atol": MOMENT1_ATOL, "rtol": MOMENT1_RTOL},
    "moment2": {"atol": MOMENT2_ATOL, "rtol": MOMENT2_RTOL},
    "declared_before_execution": True,
    "moment_rationale": (
        "An ordinary 2e-6 absolute allowance can hide incorrect zero moments. "
        "Separate smaller absolute limits retain float32 rounding room. C may form "
        "1-.999f rather than Python's 1-.999; the approximately 1.29e-5 relative "
        "coefficient difference is covered by the declared moment2 rtol, not by "
        "loosening limits after results are seen. Explicit zero/nonzero checks "
        "supplement elementwise comparisons."
    ),
}


def require(condition, message):
    common.require(condition, message)


def feature_values(name):
    require(name in FEATURE_NAMES, f"Unknown feature build: {name}")
    return {key: name == key or name == "combined" for key in FEATURE_KEYS}


def finite_list(value, count, label):
    return common.numeric_list(value, count, label)


def validate_ids(contexts, targets, shape, label):
    vocab, context, batch = (shape[name] for name in ("vocab_size", "context_size", "batch_size"))
    require(isinstance(contexts, list) and len(contexts) == batch
            and all(isinstance(row, list) and len(row) == context for row in contexts), f"{label}: context shape")
    ids = [token for row in contexts for token in row]
    require(all(type(token) is int and 0 <= token < vocab for token in ids), f"{label}: context ID range")
    require(isinstance(targets, list) and len(targets) == batch
            and all(type(token) is int and 0 <= token < vocab for token in targets), f"{label}: target ID range")
    return ids


def validate_dump(data, fixture, feature, threads, clip_norm, updates, poison):
    expected_kind = "SYNTHETIC_C_NPLM_CHECK" if updates == 1 else "FIXED_WORK_MULTI_UPDATE_CHECK"
    require(data.get("kind") == expected_kind, f"Incorrect dump kind for {updates} update(s)")
    shape = common.FIXTURES[fixture]
    config = data["config"]
    for name in ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size"):
        require(type(config.get(name)) is int and config[name] == shape[name], f"Incorrect fixture {name}")
    for name, value in {"seed": 42, "warmup_steps": 0, "steps": updates,
                        "threads": threads, "check_updates": updates}.items():
        require(type(config.get(name)) is int and config[name] == value, f"Incorrect config.{name}")
    require(type(config.get("poison_gradients")) is bool and config["poison_gradients"] == poison,
            "Incorrect config.poison_gradients")
    for name in ("steps", "check_updates"):
        require(type(data.get(name)) is int and data[name] == updates, f"Incorrect top-level {name}")
    require(type(data.get("poison_gradients")) is bool and data["poison_gradients"] == poison,
            "Gradient poison argument was not applied")
    require(type(data.get("target_prediction_events")) is int
            and data["target_prediction_events"] == updates * shape["batch_size"], "Incorrect prediction-event counter")
    actual_clip = common.finite_number(config["clip_norm"], "config.clip_norm")
    require(math.isclose(actual_clip, clip_norm, rel_tol=2e-7, abs_tol=1e-9), "Clip norm was not applied")
    features = data["features"]
    require(isinstance(features, dict) and set(features) == set(FEATURE_KEYS), "Incorrect feature schema")
    for key, expected in feature_values(feature).items():
        require(type(features.get(key)) is bool and features[key] == expected, f"Build macro was not applied: {key}")
    kernel = data["kernel_options"]
    for key, expected in common.expected_options(OMP_ARGUMENTS).items():
        actual = kernel.get(key)
        if isinstance(expected, bool):
            require(type(actual) in {bool, int} and actual in (0, 1) and bool(actual) == expected,
                    f"Incorrect runtime kernel option: {key}")
        else:
            require(type(actual) is int and actual == expected, f"Incorrect runtime kernel option: {key}")
    require(type(kernel.get("threads")) is int and kernel["threads"] == threads, "Incorrect kernel thread budget")
    vocab, context, dim, hidden = (shape[name] for name in
                                   ("vocab_size", "context_size", "embedding_dim", "hidden_dim"))
    expected_layout = [("embedding", [vocab, dim]), ("hidden_weight", [hidden, context * dim]),
                       ("hidden_bias", [hidden]), ("output_weight", [vocab, hidden]), ("output_bias", [vocab])]
    layout = data["parameter_layout"]
    require(isinstance(layout, list) and len(layout) == len(expected_layout), "Incorrect parameter layout")
    offset = 0
    for entry, (name, dims) in zip(layout, expected_layout):
        length = math.prod(dims)
        require(entry.get("name") == name and entry.get("shape") == dims
                and entry.get("offset") == offset and entry.get("length") == length,
                f"Incorrect layout for {name}")
        offset += length
    for field in ("initial_params", "grads", "optimizer_grads", "updated_params", "moment1", "moment2"):
        finite_list(data[field], offset, field)
    if "parameter_count" in data:
        require(type(data["parameter_count"]) is int and data["parameter_count"] == offset,
                "Parameter counter disagrees with tensor layout")
    require(all(value >= 0 for value in data["moment2"]), "Second moments cannot be negative")
    for key, expected in common.OPTIMIZER.items():
        actual = data["optimizer"].get(key)
        if isinstance(expected, str):
            require(actual == expected, f"Incorrect optimizer {key}")
        else:
            common.finite_number(actual, f"optimizer.{key}")
            require(math.isclose(actual, expected, rel_tol=2e-7, abs_tol=1e-15), f"Incorrect optimizer {key}")
    history = data.get("history")
    require(isinstance(history, list) and len(history) == updates, "History does not cover every update")
    for step, row in enumerate(history, 1):
        require(type(row.get("step")) is int and row["step"] == step, "History update counter mismatch")
        require(type(row.get("batch_index")) is int and row["batch_index"] == (step - 1) % 4,
                "History did not follow the four fixed batches")
        validate_ids(row["contexts"], row["targets"], shape, f"history[{step}]")
        for field in ("loss", "gradient_norm", "clipping_scale"):
            common.finite_number(row[field], f"history[{step}].{field}")
        require(row["loss"] >= 0 and row["gradient_norm"] >= 0 and 0 < row["clipping_scale"] <= 1,
                "Invalid history loss/norm/scale")
        require(type(row.get("clip_active")) is bool and row["clip_active"] == (row["clipping_scale"] < 1),
                "History clipping-active flag disagrees with scale")
    validate_ids(data["contexts"], data["targets"], shape, "final batch")
    require(data["contexts"] == history[-1]["contexts"] and data["targets"] == history[-1]["targets"],
            "Top-level batch is not the final history batch")
    for field in ("loss", "gradient_norm", "clipping_scale"):
        common.finite_number(data[field], field)
        require(data[field] == history[-1][field], f"Top-level {field} disagrees with final update")
    return offset, history


def moment_diagnostics(torch, name, actual, expected, atol, rtol):
    actual = torch.as_tensor(actual, dtype=torch.float64)
    expected = expected.detach().to(dtype=torch.float64)
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), f"Non-finite {name}")
    expected_nonzero = expected != 0
    actual_nonzero = actual != 0
    expected_strong = expected.abs() > atol + rtol * expected.abs()
    zeroed_strong = int(((actual == 0) & expected_strong).sum())
    all_zero_while_expected_nonzero = bool(expected_nonzero.any()) and not bool(actual_nonzero.any())
    require(not all_zero_while_expected_nonzero, f"{name}: entire moment vector was incorrectly zeroed")
    require(zeroed_strong == 0, f"{name}: nonzero reference components above tolerance were erased")
    return {"name": name, "expected_nonzero_elements": int(expected_nonzero.sum()),
            "actual_nonzero_elements": int(actual_nonzero.sum()), "expected_strong_nonzero_elements": int(expected_strong.sum()),
            "zeroed_strong_nonzero_elements": zeroed_strong,
            "zeroed_reference_nonzero_elements_including_below_absolute_allowance": int(((actual == 0) & expected_nonzero).sum()),
            "reference_l2_norm": float(torch.linalg.vector_norm(expected)),
            "actual_l2_norm": float(torch.linalg.vector_norm(actual)),
            "all_zero_while_reference_nonzero": all_zero_while_expected_nonzero,
            "passed": True}


def check_data(data, fixture, feature, threads, clip_norm, updates, poison):
    count, history = validate_dump(data, fixture, feature, threads, clip_norm, updates, poison)
    import torch
    from torch import nn
    from torch.nn import functional as functional
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    shape = common.FIXTURES[fixture]
    vocab, context, dim, hidden, batch = (shape[name] for name in
                                         ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size"))

    class ReferenceNplm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab, dim)
            self.hidden = nn.Linear(context * dim, hidden)
            self.output = nn.Linear(hidden, vocab)

        def forward(self, ids):
            flat = self.embedding(ids).reshape(batch, context * dim)
            return self.output(torch.tanh(self.hidden(flat)))

    model = ReferenceNplm().to(dtype=torch.float32, device="cpu")
    parameters = [model.embedding.weight, model.hidden.weight, model.hidden.bias,
                  model.output.weight, model.output.bias]
    initial = torch.tensor(data["initial_params"], dtype=torch.float32)
    require(bool(torch.isfinite(initial).all()), "Initial values overflow float32")
    with torch.no_grad():
        for entry, parameter in zip(data["parameter_layout"], parameters):
            parameter.copy_(initial[entry["offset"]:entry["offset"] + entry["length"]].reshape(entry["shape"]))
    opt = common.OPTIMIZER
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=(opt["beta1"], opt["beta2"]),
                                 eps=opt["eps"], weight_decay=opt["weight_decay"], foreach=False, fused=False)
    comparisons, step_checks, all_input_ids, sequence_bytes = [], [], [], bytearray()
    for record in history:
        contexts = torch.tensor(record["contexts"], dtype=torch.int64)
        targets = torch.tensor(record["targets"], dtype=torch.int64)
        ids = [token for row in record["contexts"] for token in row]
        all_input_ids.extend(ids)
        sequence_bytes.extend(b"".join(struct.pack("<I", token) for token in ids + record["targets"]))
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(model(contexts), targets, reduction="mean")
        loss.backward()
        raw_gradients = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
        require(bool(torch.isfinite(loss)) and bool(torch.isfinite(raw_gradients).all()), "Non-finite PyTorch forward/backward")
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True, foreach=False)
        clipped_gradients = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
        scale = min(1.0, clip_norm / (float(norm) + 1e-6))
        active = scale < 1
        require(record["clip_active"] == active, "C and PyTorch disagree on clipping activation")
        if math.isclose(clip_norm, 0.05, rel_tol=0, abs_tol=1e-9):
            require(active, "The .05 fixture did not exercise active clipping")
        step = record["step"]
        for label, actual, expected in (("loss", record["loss"], float(loss.detach())),
                                         ("gradient_norm", record["gradient_norm"], float(norm)),
                                         ("clipping_scale", record["clipping_scale"], scale)):
            comparisons.append(common.compare(torch, f"update_{step}.{label}", [actual], [expected], common.ATOL, common.RTOL))
        optimizer.step()
        state_steps = [float(optimizer.state[parameter]["step"]) for parameter in parameters]
        require(all(math.isfinite(value) and value == step for value in state_steps), "Reference optimizer-step counter mismatch")
        step_checks.append({"step": step, "batch_index": record["batch_index"], "clip_active": active,
                            "reference_loss": float(loss.detach()), "reference_gradient_norm": float(norm),
                            "reference_clipping_scale": scale, "optimizer_step_counters": state_steps})
    updated = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    moment1 = torch.cat([optimizer.state[parameter]["exp_avg"].detach().reshape(-1) for parameter in parameters])
    moment2 = torch.cat([optimizer.state[parameter]["exp_avg_sq"].detach().reshape(-1) for parameter in parameters])
    require(all(bool(torch.isfinite(tensor).all()) for tensor in (clipped_gradients, updated, moment1, moment2)),
            "Non-finite PyTorch optimizer state")
    ordinary = (("raw_gradients", data["grads"], raw_gradients),
                ("clipped_gradients", data["optimizer_grads"], clipped_gradients),
                ("updated_parameters", data["updated_params"], updated))
    for label, actual, expected in ordinary:
        comparisons.append(common.compare(torch, "final.all_" + label, actual, expected, common.ATOL, common.RTOL))
    comparisons.append(common.compare(torch, "final.all_moment1", data["moment1"], moment1, MOMENT1_ATOL, MOMENT1_RTOL))
    comparisons.append(common.compare(torch, "final.all_moment2", data["moment2"], moment2, MOMENT2_ATOL, MOMENT2_RTOL))
    for entry in data["parameter_layout"]:
        start, stop = entry["offset"], entry["offset"] + entry["length"]
        for label, actual, expected in ordinary:
            comparisons.append(common.compare(torch, f"final.{entry['name']}.{label}",
                                               actual[start:stop], expected[start:stop], common.ATOL, common.RTOL))
        for label, actual, expected, atol, rtol in (("moment1", data["moment1"], moment1, MOMENT1_ATOL, MOMENT1_RTOL),
                                                     ("moment2", data["moment2"], moment2, MOMENT2_ATOL, MOMENT2_RTOL)):
            comparisons.append(common.compare(torch, f"final.{entry['name']}.{label}",
                                               actual[start:stop], expected[start:stop], atol, rtol))
    diagnostics = [moment_diagnostics(torch, "moment1", data["moment1"], moment1, MOMENT1_ATOL, MOMENT1_RTOL),
                   moment_diagnostics(torch, "moment2", data["moment2"], moment2, MOMENT2_ATOL, MOMENT2_RTOL)]
    final_ids = [token for row in history[-1]["contexts"] for token in row]
    final_frequency = Counter(final_ids)
    repeated = sorted(token for token, frequency in final_frequency.items() if frequency > 1)
    last_unused = sorted(set(range(vocab)) - set(final_ids))
    never_used = sorted(set(range(vocab)) - set(all_input_ids))
    momentum_only = sorted(set(last_unused) & set(all_input_ids))
    require(bool(repeated), "Final batch does not exercise repeated embedding scatter-add")
    last_unused_indices = [token * dim + dimension for token in last_unused for dimension in range(dim)]
    never_used_indices = [token * dim + dimension for token in never_used for dimension in range(dim)]
    momentum_only_indices = [token * dim + dimension for token in momentum_only for dimension in range(dim)]
    require(all(data[field][index] == 0 for field in ("grads", "optimizer_grads") for index in last_unused_indices),
            "Final unused embedding rows have nonzero gradients")
    require(all(data[field][index] == 0 for field in ("moment1", "moment2") for index in never_used_indices),
            "Never-used embedding rows have nonzero moments")
    if last_unused_indices:
        require(bool(torch.all(raw_gradients[last_unused_indices] == 0))
                and bool(torch.all(clipped_gradients[last_unused_indices] == 0)), "Reference unused embedding gradient is nonzero")
    if momentum_only_indices:
        for label, actual, expected, atol, rtol in (("moment1", data["moment1"], moment1, MOMENT1_ATOL, MOMENT1_RTOL),
                                                     ("moment2", data["moment2"], moment2, MOMENT2_ATOL, MOMENT2_RTOL)):
            comparisons.append(common.compare(torch, "embedding_prior_batch_momentum." + label,
                                               [actual[index] for index in momentum_only_indices],
                                               expected[momentum_only_indices], atol, rtol))
    if updates == 1:
        scalar_moment1 = [(1 - opt["beta1"]) * gradient for gradient in data["optimizer_grads"]]
        scalar_moment2 = [(1 - opt["beta2"]) * gradient * gradient for gradient in data["optimizer_grads"]]
        comparisons.append(common.compare(torch, "first_step_scalar_moment1", data["moment1"], scalar_moment1,
                                           MOMENT1_ATOL, MOMENT1_RTOL))
        comparisons.append(common.compare(torch, "first_step_scalar_moment2", data["moment2"], scalar_moment2,
                                           MOMENT2_ATOL, MOMENT2_RTOL))
    return {"passed": all(row["passed"] for row in comparisons), "fixture": fixture, "feature": feature,
            "features": data["features"], "kernel_options": data["kernel_options"], "config": data["config"],
            "check_updates": updates, "poison_gradients": poison, "parameter_count": count,
            "target_prediction_events": updates * batch, "counter_checks_passed": True,
            "history_record_count": len(history), "step_checks": step_checks,
            "python_finite_checked_every_value": True, "c_reported_finite_flag": data.get("finite_flag"),
            "c_finite_flag_is_authoritative": False, "torch_version": torch.__version__,
            "initial_parameter_float32_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            "input_sequence_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
            "final_repeated_embedding_ids": repeated, "final_unused_embedding_ids": last_unused,
            "never_used_embedding_ids": never_used, "previously_used_but_final_unused_embedding_ids": momentum_only,
            "never_used_moments_exactly_zero": True, "final_unused_gradients_exactly_zero": True,
            "moment_diagnostics": diagnostics, "comparisons": comparisons}


def run_dump(executable, fixture, threads, clip_norm, updates, poison, *, legacy=False):
    command = [str(Path(executable).resolve()), common.FIXTURES[fixture]["dump_argument"],
               "--threads", str(threads), "--clip-norm", str(clip_norm), *OMP_ARGUMENTS]
    if not legacy:
        command += ["--check-updates", str(updates)]
        if poison:
            command.append("--poison-gradients")
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE",
                       KMP_BLOCKTIME="0", VECLIB_MAXIMUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads), OMP_MAX_ACTIVE_LEVELS="1", OMP_THREAD_LIMIT=str(threads))
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=90,
                            env=environment, cwd=Path(executable).resolve().parent)
    return common.parse_json(result.stdout), {"command": command, "exit_code": result.returncode,
                                             "stderr": result.stderr,
                                             "dump_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def legacy_crosscheck(path, baseline_data, threads):
    result = {"requested": path is not None, "required": False, "status": "not_run", "passed": None}
    if path is None:
        result["reason"] = "No legacy executable was requested; no original source is rebuilt."
        return result
    if not Path(path).is_file():
        result["reason"] = "Requested legacy executable is absent; no original source is rebuilt."
        return result
    require(baseline_data is not None, "Legacy comparison requires a baseline clean tiny one-step case at clip 1")
    import torch
    before = common.identity(path)
    data, process = run_dump(path, "tiny", threads, 1.0, 1, False, legacy=True)
    common.validate_schema(data, "tiny", 1.0, threads, common.expected_options(OMP_ARGUMENTS))
    require(data["contexts"] == baseline_data["contexts"] and data["targets"] == baseline_data["targets"],
            "New and legacy engine input IDs differ")
    require(data["parameter_layout"] == baseline_data["parameter_layout"], "New and legacy layouts differ")
    require(torch.tensor(data["initial_params"], dtype=torch.float32).numpy().tobytes()
            == torch.tensor(baseline_data["initial_params"], dtype=torch.float32).numpy().tobytes(),
            "New and legacy initial parameter bits differ")
    comparisons = []
    for field in ("loss", "gradient_norm", "clipping_scale", "grads", "optimizer_grads", "updated_params"):
        old, new = data[field], baseline_data[field]
        comparisons.append(common.compare(torch, "legacy_vs_fixedwork." + field,
                                           new if isinstance(new, list) else [new],
                                           old if isinstance(old, list) else [old], common.ATOL, common.RTOL))
    after = common.identity(path)
    require(before == after, "Legacy binary changed during comparison")
    return {**result, **process, "status": "complete", "passed": all(row["passed"] for row in comparisons),
            "fixture": "tiny", "threads": threads, "clip_norm": 1.0, "executable_before": before,
            "executable_after": after, "identical_initial_parameter_bits": True, "identical_input_ids": True,
            "comparisons": comparisons,
            "scope": "Existing legacy and new baseline one-step common values; legacy moments are not available. No source rebuilt."}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_markdown(path, report):
    lines = ["# Fixed-work model verification", "", f"Status: {'PASS' if report.get('passed') else report.get('status', 'pending')}", "",
             "CPU synthetic correctness only. No Wiki training, timings, convergence or quality comparison.", "",
             "Ordinary values: abs error <= 2e-6 + 2e-4 * abs(reference).",
             "Moment1: abs error <= 2e-9 + 2e-4 * abs(reference).",
             "Moment2: abs error <= 2e-11 + 5e-4 * abs(reference).", "",
             TOLERANCES["moment_rationale"], "",
             "Each feature normally checks tiny/irregular shapes, 1/4 threads, clip 1/.05,",
             "one ordinary update and four consecutive updates with NaN gradient poisoning.", "",
             "| Feature | Cases | Passed | One step | Four updates | Poison |", "|---|---:|---:|---|---|---|"]
    for name, variant in report.get("variants", {}).items():
        lines.append(f"| {name} | {len(variant.get('case_indices', []))} | {variant.get('passed_cases', 0)} | "
                     f"{variant.get('one_step_passed', False)} | {variant.get('multi_update_passed', False)} | "
                     f"{variant.get('multiupdate_poison_passed', False)} |")
    lines += ["", f"Total case count: {len(report.get('cases', []))}.",
              f"Failed comparisons: {report.get('failed_comparisons', 'pending')}.",
              "Checksums bind inputs and executables; numerical comparisons determine correctness.",
              "Unused rows in the final batch can still update from moments accumulated in earlier batches.",
              "Only rows unused across every update must have exactly zero moments.", "",
              f"Legacy comparison: {report.get('legacy_crosscheck', {}).get('status', 'not_run')}."]
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def binary_specs(args):
    if args.binary:
        result = {}
        for item in args.binary:
            name, separator, value = item.partition("=")
            require(separator and name in FEATURE_NAMES and value, "Use --binary FEATURE=PATH")
            require(name not in result, f"Duplicate feature binary: {name}")
            result[name] = Path(value).resolve()
        return result
    return {args.feature: args.executable.resolve()}


def finalize_variant(report, name, expected_count):
    variant = report["variants"][name]
    cases = [report["cases"][index] for index in variant["case_indices"]]
    def passed_subset(predicate):
        subset = [row for row in cases if predicate(row)]
        return bool(subset) and all(row["passed"] for row in subset)
    variant.update({"passed_cases": sum(row["passed"] for row in cases),
                    "passed": len(cases) == expected_count and all(row["passed"] for row in cases),
                    "one_step_passed": passed_subset(lambda row: row["check_updates"] == 1),
                    "multi_update_passed": passed_subset(lambda row: row["check_updates"] > 1),
                    "poison_passed": passed_subset(lambda row: row["poison_gradients"]),
                    "multiupdate_poison_passed": passed_subset(lambda row: row["check_updates"] > 1 and row["poison_gradients"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--binary", action="append", help="FEATURE=PATH; repeat to check all five feature builds in one process")
    source.add_argument("--executable", type=Path, default=BASE / "cpu_fixedwork_baseline")
    parser.add_argument("--feature", choices=FEATURE_NAMES, default="baseline")
    parser.add_argument("--fixture", action="append", choices=list(common.FIXTURES))
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--clip-norms", nargs="+", type=float, default=[1.0, 0.05])
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS))
    parser.add_argument("--legacy-executable", type=Path)
    parser.add_argument("--report", type=Path, default=BASE / "fixedwork-model-verification.json")
    parser.add_argument("--markdown", type=Path, default=BASE / "fixedwork-model-verification.md")
    parser.add_argument("--lock-held", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.report.exists() or args.markdown.exists():
        raise FileExistsError("Preserving existing fixed-work verification reports")
    report = {"kind": "FIXEDWORK_MODEL_INDEPENDENT_TORCH_VERIFICATION", "status": "running", "passed": False,
              "started_at": datetime.now().astimezone().isoformat(), "checker": common.identity(Path(__file__)),
              "tolerances_predeclared": TOLERANCES, "variants": {}, "cases": [],
              "scope": "CPU float32 synthetic one/four-update forward/backward/clipping/AdamW and optimizer moments; no timing or learned-quality claim.",
              "finite_policy": "Every dumped scalar/parameter/gradient/moment is checked in Python; C finite flags and final checksums do not replace the oracle.",
              "moment_zero_policy": "Exact zero on never-used embedding rows, finite and nonnegative second moments, no erased moment vector or strong expected components.",
              "poison_policy": "C fills the full gradient array with NaNs before each backward; the production zeroing/overwrite paths must produce finite complete gradients.",
              "reference": {"implementation": "PyTorch CPU float32 F.cross_entropy + clip_grad_norm_ + AdamW(foreach=False,fused=False)",
                            "oracle_threads": 1, "optimizer": common.OPTIMIZER},
              "legacy_crosscheck": {"status": "pending"}}
    try:
        binaries = binary_specs(args)
        require(all(type(value) is int and 1 <= value <= 64 for value in args.threads), "Invalid thread budget")
        require(all(math.isfinite(value) and value > 0 for value in args.clip_norms), "Clip norms must be positive and finite")
        fixtures = list(dict.fromkeys(args.fixture or common.FIXTURES))
        scenarios = list(dict.fromkeys(args.scenario or SCENARIOS))
        threads_list, clips = list(dict.fromkeys(args.threads)), list(dict.fromkeys(args.clip_norms))
        expected_per_variant = len(fixtures) * len(scenarios) * len(threads_list) * len(clips)
        report["validation_spec"] = {"fixtures": fixtures, "threads": threads_list, "clip_norms": clips,
                                     "scenarios": {name: SCENARIOS[name] for name in scenarios},
                                     "expected_cases_per_variant": expected_per_variant,
                                     "expected_total_cases": expected_per_variant * len(binaries),
                                     "runtime_kernel_arguments": list(OMP_ARGUMENTS)}
        paths = sorted({*BASE.glob("*.c"), *BASE.glob("*.h"), Path(common.__file__).resolve()})
        report["sources_before"] = [common.identity(path) for path in paths]
        for name, binary in binaries.items():
            report["variants"][name] = {"features": feature_values(name), "executable_before": common.identity(binary),
                                        "case_indices": [], "passed": False}
        atomic_json(args.report, report)
        legacy_data, legacy_threads = None, None
        with common.numerical_lock(args.lock_held):
            for feature, executable in binaries.items():
                for fixture in fixtures:
                    for threads in threads_list:
                        for clip in clips:
                            for scenario in scenarios:
                                parameters = SCENARIOS[scenario]
                                updates, poison = parameters["check_updates"], parameters["poison_gradients"]
                                row = {"feature": feature, "fixture": fixture, "scenario": scenario, "threads": threads,
                                       "clip_norm": clip, "check_updates": updates, "poison_gradients": poison, "passed": False}
                                try:
                                    data, process = run_dump(executable, fixture, threads, clip, updates, poison)
                                    row.update(process)
                                    row.update(check_data(data, fixture, feature, threads, clip, updates, poison))
                                    if (feature == "baseline" and fixture == "tiny" and updates == 1
                                            and not poison and clip == 1.0 and (legacy_data is None or threads == 4)):
                                        legacy_data, legacy_threads = data, threads
                                except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError,
                                        subprocess.SubprocessError, ImportError) as error:
                                    row["error"] = f"{type(error).__name__}: {error}"
                                    if isinstance(error, subprocess.CalledProcessError):
                                        row["stderr"] = error.stderr
                                        row["exit_code"] = error.returncode
                                index = len(report["cases"])
                                report["variants"][feature]["case_indices"].append(index)
                                report["cases"].append(row)
                                atomic_json(args.report, report)
                report["variants"][feature]["executable_after"] = common.identity(executable)
                require(report["variants"][feature]["executable_before"] == report["variants"][feature]["executable_after"],
                        f"Feature binary changed during checks: {feature}")
                finalize_variant(report, feature, expected_per_variant)
                atomic_json(args.report, report)
            report["legacy_crosscheck"] = legacy_crosscheck(args.legacy_executable, legacy_data, legacy_threads)
            report["sources_after"] = [common.identity(path) for path in paths]
            require(report["sources_before"] == report["sources_after"], "Fixed-work kernel sources changed during checks")
            require(report["checker"] == common.identity(Path(__file__)), "Fixed-work checker changed during checks")
            initial_ok = sequence_ok = True
            for fixture in fixtures:
                rows = [row for row in report["cases"] if row["fixture"] == fixture and "initial_parameter_float32_sha256" in row]
                initial_ok &= len({row["initial_parameter_float32_sha256"] for row in rows}) <= 1
                for scenario in scenarios:
                    selected = [row for row in rows if row["scenario"] == scenario]
                    sequence_ok &= len({row["input_sequence_sha256"] for row in selected}) <= 1
            report["same_work_validation"] = {"identical_initial_parameter_bits": initial_ok,
                                               "identical_input_sequences_for_matching_fixture_and_scenario": sequence_ok,
                                               "checksums_are_not_the_correctness_oracle": True}
            require(initial_ok and sequence_ok, "Feature builds did not use identical initial data/parameters")
        comparisons = [value for row in report["cases"] for value in row.get("comparisons", [])]
        report.update({"status": "complete", "passed_cases": sum(row["passed"] for row in report["cases"]),
                       "failed_cases": sum(not row["passed"] for row in report["cases"]),
                       "comparison_count": len(comparisons), "failed_comparisons": sum(not row["passed"] for row in comparisons),
                       "maximum_absolute_error_by_family": {
                           family: max((row["max_absolute_error"] for row in comparisons
                                        if (("moment1" in row["name"]) if family == "moment1" else
                                            ("moment2" in row["name"]) if family == "moment2" else
                                            ("moment1" not in row["name"] and "moment2" not in row["name"]))), default=None)
                           for family in ("ordinary", "moment1", "moment2")}})
        report["passed"] = (len(report["cases"]) == report["validation_spec"]["expected_total_cases"]
                            and all(variant["passed"] for variant in report["variants"].values())
                            and report["legacy_crosscheck"].get("passed") is not False)
        report["multiupdate_poison_passed"] = all(variant["multiupdate_poison_passed"] for variant in report["variants"].values())
        if len(binaries) == 1:
            only = next(iter(report["variants"].values()))
            for key in ("executable_before", "executable_after", "features", "one_step_passed", "multi_update_passed", "poison_passed"):
                report[key] = only[key]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError,
            subprocess.SubprocessError, ImportError) as error:
        report["status"], report["passed"] = "failed", False
        report["error"] = f"{type(error).__name__}: {error}"
    report["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_json(args.report, report)
    write_markdown(args.markdown, report)
    if args.quiet:
        rendered = {key: report[key] for key in ("kind", "status", "passed", "passed_cases", "failed_cases",
                                                "comparison_count", "failed_comparisons", "maximum_absolute_error_by_family",
                                                "multiupdate_poison_passed", "error") if key in report}
        rendered["variants"] = {name: {key: variant.get(key) for key in ("passed", "passed_cases", "one_step_passed",
                                                                       "multi_update_passed", "multiupdate_poison_passed")}
                                for name, variant in report["variants"].items()}
        rendered["report"] = str(args.report.resolve())
    else:
        rendered = report
    print(json.dumps(rendered, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
