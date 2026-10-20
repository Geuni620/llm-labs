#!/usr/bin/env python3
"""Independently validate experimental C kernels against CPU PyTorch.

Every invocation checks one-step synthetic forward/backward, global clipping,
and AdamW. Tiny and irregular fixtures exercise repeated embedding scatter-add,
unused rows, tile tails, and inner dimensions crossing 32/64-element blocks.
No Wiki arrays, model-quality evaluation, or performance timing is performed.

Default tolerances are declared before any executable is run. Python checks
every dumped scalar/array value for finiteness; a C finite_flag is never used
as evidence, particularly for compiler builds with -ffast-math.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile


BASE = Path(__file__).resolve().parent
LOCK_PATH = BASE.parent / ".timing.lock"
ATOL = 2e-6
RTOL = 2e-4
FIXTURES = {
    "tiny": {"vocab_size": 11, "context_size": 2, "embedding_dim": 3,
             "hidden_dim": 5, "batch_size": 4, "dump_argument": "--dump-check"},
    "irregular": {"vocab_size": 73, "context_size": 3, "embedding_dim": 7,
                  "hidden_dim": 67, "batch_size": 71, "dump_argument": "--dump-check-large"},
}
OPTIMIZER = {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01,
             "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
OPTION_DEFAULTS = {"softmax_parallel": False, "reduction_parallel": False,
                   "adam_parallel": False, "tiled_gemm": 0, "fast_exp": False}
VARIANTS = {
    "baseline": (),
    "softmax": ("--softmax-parallel",),
    "reduction": ("--reduction-parallel",),
    "adam": ("--adam-parallel",),
    "tile32": ("--tiled-gemm", "32"),
    "tile64": ("--tiled-gemm", "64"),
    "fast-exp": ("--fast-exp",),
    "combined": ("--softmax-parallel", "--reduction-parallel", "--adam-parallel", "--tiled-gemm", "32"),
    "combined-fast": ("--softmax-parallel", "--reduction-parallel", "--adam-parallel", "--tiled-gemm", "32", "--fast-exp"),
    "combined-fast64": ("--softmax-parallel", "--reduction-parallel", "--adam-parallel", "--tiled-gemm", "64", "--fast-exp"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(payload):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)


def finite_number(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), f"Non-finite/non-numeric {label}: {value!r}")
    return value


def numeric_list(value, length, label):
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain exactly {length} numbers")
    for index, number in enumerate(value):
        finite_number(number, f"{label}[{index}]")
    return value


def expected_options(arguments):
    result = dict(OPTION_DEFAULTS)
    toggles = {"--softmax-parallel": "softmax_parallel", "--reduction-parallel": "reduction_parallel",
               "--adam-parallel": "adam_parallel", "--fast-exp": "fast_exp"}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in toggles:
            result[toggles[argument]] = True
        elif argument in {"--tiled-gemm32", "--tiled-gemm64"}:
            result["tiled_gemm"] = int(argument[-2:])
        elif argument == "--tiled-gemm":
            index += 1
            require(index < len(arguments) and arguments[index] in {"32", "64"}, "Tile must be 32 or 64")
            result["tiled_gemm"] = int(arguments[index])
        else:
            raise ValueError(f"Unsupported kernel argument in correctness check: {argument}")
        index += 1
    return result


def identity(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def validate_schema(data, fixture, clip_norm, threads, options):
    shape = FIXTURES[fixture]
    require(data.get("kind") == "SYNTHETIC_C_NPLM_CHECK", "Wrong dump kind")
    config = data["config"]
    for name in ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size"):
        require(type(config.get(name)) is int and config[name] == shape[name], f"Incorrect {fixture} {name}")
    for name, value in {"seed": 42, "warmup_steps": 0, "steps": 1, "threads": threads}.items():
        require(type(config.get(name)) is int and config[name] == value, f"Incorrect check config {name}")
    actual_clip = finite_number(config["clip_norm"], "config.clip_norm")
    require(math.isclose(actual_clip, clip_norm, rel_tol=2e-7, abs_tol=1e-9), "Clip norm argument was not applied")
    kernel = data["kernel_options"]
    for name, value in options.items():
        actual = kernel.get(name)
        if isinstance(value, bool):
            require(type(actual) in {bool, int} and actual in {0, 1} and bool(actual) == value,
                    f"Kernel option was not applied: {name}: {actual!r} != {value!r}")
        else:
            require(type(actual) is int and actual == value, f"Kernel option was not applied: {name}")
    vocab, context, dim, hidden, batch = (shape[name] for name in
                                         ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size"))
    layouts = [("embedding", [vocab, dim]), ("hidden_weight", [hidden, context * dim]),
               ("hidden_bias", [hidden]), ("output_weight", [vocab, hidden]), ("output_bias", [vocab])]
    require(isinstance(data["parameter_layout"], list) and len(data["parameter_layout"]) == 5, "Wrong parameter layout")
    offset = 0
    for entry, (name, tensor_shape) in zip(data["parameter_layout"], layouts):
        length = math.prod(tensor_shape)
        require(entry.get("name") == name and entry.get("shape") == tensor_shape
                and entry.get("offset") == offset and entry.get("length") == length,
                f"Incorrect parameter layout for {name}")
        offset += length
    for field in ("initial_params", "grads", "optimizer_grads", "updated_params"):
        numeric_list(data[field], offset, field)
    for field in ("loss", "gradient_norm", "clipping_scale"):
        finite_number(data[field], field)
    require(data["loss"] >= 0 and data["gradient_norm"] >= 0 and 0 < data["clipping_scale"] <= 1,
            "Invalid loss/norm/clipping scale")
    contexts, targets = data["contexts"], data["targets"]
    require(isinstance(contexts, list) and len(contexts) == batch
            and all(isinstance(row, list) and len(row) == context for row in contexts), "Incorrect contexts shape")
    ids = [token for row in contexts for token in row]
    require(all(type(token) is int and 0 <= token < vocab for token in ids), "Invalid context token ID")
    require(isinstance(targets, list) and len(targets) == batch
            and all(type(token) is int and 0 <= token < vocab for token in targets), "Invalid target token ID")
    frequency = Counter(ids)
    repeated = sorted(token for token, count in frequency.items() if count > 1)
    unused = sorted(set(range(vocab)) - set(ids))
    require(bool(repeated), "Fixture must exercise repeated embedding scatter-add")
    if fixture == "tiny":
        require(bool(unused), "Tiny fixture must exercise unused embedding rows")
    for name, expected in OPTIMIZER.items():
        actual = data["optimizer"].get(name)
        if isinstance(expected, str):
            require(actual == expected, f"Incorrect optimizer {name}")
        else:
            finite_number(actual, f"optimizer.{name}")
            require(math.isclose(actual, expected, rel_tol=2e-7, abs_tol=1e-15), f"Incorrect optimizer {name}")
    return repeated, unused


def compare(torch, label, actual, expected, atol, rtol):
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    expected = torch.as_tensor(expected, dtype=torch.float64).reshape(-1)
    require(actual.shape == expected.shape and actual.numel() > 0, f"Comparison shape mismatch: {label}")
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
            f"Non-finite comparison tensor: {label}")
    absolute = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    failed = absolute > allowed
    worst = int(torch.argmax(absolute))
    return {"name": label, "passed": not bool(failed.any()), "elements": actual.numel(),
            "max_absolute_error": float(absolute.max()),
            "max_relative_error": float((absolute / expected.abs().clamp_min(1e-12)).max()),
            "failed_elements": int(failed.sum()), "worst_absolute_error_index": worst,
            "actual_at_worst": float(actual[worst]), "expected_at_worst": float(expected[worst]),
            "atol": atol, "rtol": rtol}


def check_data(data, fixture, clip_norm, threads, options, atol=ATOL, rtol=RTOL):
    repeated, unused = validate_schema(data, fixture, clip_norm, threads, options)
    import torch
    from torch import nn
    from torch.nn import functional as functional
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    shape = FIXTURES[fixture]
    vocab, context, dim, hidden, batch = (shape[name] for name in
                                         ("vocab_size", "context_size", "embedding_dim", "hidden_dim", "batch_size"))

    class ReferenceNplm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab, dim)
            self.hidden = nn.Linear(context * dim, hidden)
            self.output = nn.Linear(hidden, vocab)

        def forward(self, ids):
            embedded = self.embedding(ids).reshape(batch, context * dim)
            return self.output(torch.tanh(self.hidden(embedded)))

    model = ReferenceNplm().to(device="cpu", dtype=torch.float32)
    parameters = [model.embedding.weight, model.hidden.weight, model.hidden.bias,
                  model.output.weight, model.output.bias]
    initial = torch.tensor(data["initial_params"], dtype=torch.float32)
    require(bool(torch.isfinite(initial).all()), "Initial parameters overflow float32")
    with torch.no_grad():
        for entry, parameter in zip(data["parameter_layout"], parameters):
            parameter.copy_(initial[entry["offset"]:entry["offset"] + entry["length"]].reshape(entry["shape"]))
    optimizer = torch.optim.AdamW(parameters, lr=OPTIMIZER["lr"], weight_decay=OPTIMIZER["weight_decay"],
                                 betas=(OPTIMIZER["beta1"], OPTIMIZER["beta2"]), eps=OPTIMIZER["eps"],
                                 foreach=False, fused=False)
    contexts = torch.tensor(data["contexts"], dtype=torch.int64)
    targets = torch.tensor(data["targets"], dtype=torch.int64)
    loss = functional.cross_entropy(model(contexts), targets, reduction="mean")
    loss.backward()
    raw_gradients = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(raw_gradients).all()), "Non-finite PyTorch reference")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True, foreach=False)
    clipped_gradients = torch.cat([parameter.grad.detach().reshape(-1).clone() for parameter in parameters])
    optimizer.step()
    updated = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    require(bool(torch.isfinite(updated).all()), "Non-finite PyTorch optimizer update")
    active = float(norm) > clip_norm
    if math.isclose(clip_norm, 0.05, rel_tol=0, abs_tol=1e-9):
        require(active, "The .05 fixture did not exercise active gradient clipping")
    comparisons = [
        compare(torch, "mean_cross_entropy", [data["loss"]], [float(loss.detach())], atol, rtol),
        compare(torch, "global_gradient_norm", [data["gradient_norm"]], [float(norm)], atol, rtol),
        compare(torch, "gradient_clipping_scale", [data["clipping_scale"]],
                [min(1.0, clip_norm / (float(norm) + 1e-6))], atol, rtol),
        compare(torch, "all_raw_gradients", data["grads"], raw_gradients, atol, rtol),
        compare(torch, "all_clipped_gradients", data["optimizer_grads"], clipped_gradients, atol, rtol),
        compare(torch, "all_updated_parameters", data["updated_params"], updated, atol, rtol),
    ]
    for entry in data["parameter_layout"]:
        start, stop = entry["offset"], entry["offset"] + entry["length"]
        for label, actual, expected in (("raw_gradient", data["grads"], raw_gradients),
                                         ("clipped_gradient", data["optimizer_grads"], clipped_gradients),
                                         ("updated_parameter", data["updated_params"], updated)):
            comparisons.append(compare(torch, entry["name"] + "." + label,
                                       actual[start:stop], expected[start:stop], atol, rtol))
    scalar_updated = [parameter * (1 - OPTIMIZER["lr"] * OPTIMIZER["weight_decay"])
                      - OPTIMIZER["lr"] * gradient / (abs(gradient) + OPTIMIZER["eps"])
                      for parameter, gradient in zip(data["initial_params"], data["optimizer_grads"])]
    comparisons.append(compare(torch, "independent_first_step_AdamW_scalar", data["updated_params"], scalar_updated, atol, rtol))
    unused_indices = [token * dim + dimension for token in unused for dimension in range(dim)]
    if unused_indices:
        require(all(data[field][index] == 0 for field in ("grads", "optimizer_grads") for index in unused_indices),
                "C unused embedding rows do not have exactly zero gradients")
        require(bool(torch.all(raw_gradients[unused_indices] == 0)), "PyTorch unused embedding gradient is nonzero")
        comparisons.append(compare(torch, "unused_embedding_weight_decay",
                                   [data["updated_params"][index] for index in unused_indices],
                                   [scalar_updated[index] for index in unused_indices],
                                   0, 4 * torch.finfo(torch.float32).eps))
    return {"passed": all(row["passed"] for row in comparisons), "fixture": fixture,
            "config": data["config"], "kernel_options": data["kernel_options"],
            "parameter_count": initial.numel(), "torch_version": torch.__version__,
            "initial_parameter_float32_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            "context_target_ids_sha256": hashlib.sha256(
                b"".join(struct.pack("<I", token) for row in data["contexts"] for token in row)
                + b"".join(struct.pack("<I", token) for token in data["targets"])).hexdigest(),
            "python_finite_checked_every_value": True, "c_reported_finite_flag": data.get("finite_flag"),
            "c_finite_flag_is_authoritative": False,
            "gradient_clipping": {"max_norm": clip_norm, "active": active,
                                  "reference_norm": float(norm), "c_norm": data["gradient_norm"]},
            "repeated_embedding_ids": repeated, "unused_embedding_ids": unused,
            "comparisons": comparisons}


@contextmanager
def numerical_lock(externally_held=False):
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        if externally_held:
            if acquired:
                fcntl.flock(lock, fcntl.LOCK_UN)
                raise RuntimeError("--lock-held requires an external owner of cpu-benchmark/.timing.lock")
            yield
        else:
            if not acquired:
                raise RuntimeError("Another model calculation owns cpu-benchmark/.timing.lock")
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def save_report(path, report):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--executable", type=Path, default=BASE / "benchmark")
    source.add_argument("--dump-json", type=Path, help="Validate one pre-existing dump without executing C")
    parser.add_argument("--fixture", action="append", choices=list(FIXTURES))
    parser.add_argument("--variant", action="append", choices=list(VARIANTS))
    parser.add_argument("--extra-args", default="[]", help="JSON array of additional kernel-only arguments")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--clip-norms", nargs="+", type=float, default=[1.0, 0.05])
    parser.add_argument("--build-label", default="unspecified")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--lock-held", action="store_true", help="Coordinator already owns the shared numerical lock")
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--quiet", action="store_true", help="Print a compact result; preserve full details in --report")
    args = parser.parse_args()
    report = {"kind": "EXPERIMENTAL_C_NPLM_INDEPENDENT_TORCH_VALIDATION", "passed": False,
              "status": "running", "build_label": args.build_label,
              "started_at": datetime.now().astimezone().isoformat(),
              "scope": "Synthetic one-step CPU correctness only; no Wiki training, throughput or learned-quality claims.",
              "finite_policy": "Every C scalar/array number checked with Python math.isfinite and converted reference tensors checked again. C finite_flag is not authoritative.",
              "tolerances_predeclared": {"atol": args.atol, "rtol": args.rtol,
                                          "rule": "abs(actual-reference) <= atol + rtol*abs(reference)",
                                          "default_baseline_policy": args.atol == ATOL and args.rtol == RTOL,
                                          "unused_embedding_decay_rtol": 4 * 2 ** -23,
                                          "unused_embedding_decay_atol": 0},
              "host": {"python": platform.python_version(), "platform": platform.platform()},
              "checker": identity(Path(__file__)), "cases": []}
    if args.report and args.report.exists():
        raise FileExistsError(f"Preserving prior correctness report: {args.report}")
    try:
        require(math.isfinite(args.atol) and math.isfinite(args.rtol) and args.atol >= 0 and args.rtol >= 0,
                "Tolerances must be finite and nonnegative")
        require(all(type(value) is int and 1 <= value <= 64 for value in args.threads), "Invalid thread budget")
        require(all(math.isfinite(value) and value > 0 for value in args.clip_norms), "Clip norms must be positive and finite")
        extra = parse_json(args.extra_args)
        require(isinstance(extra, list) and all(isinstance(value, str) for value in extra), "extra-args must be a JSON string array")
        expected_options(extra)
        variants = list(dict.fromkeys(args.variant or VARIANTS))
        fixtures = list(dict.fromkeys(args.fixture or FIXTURES))
        if args.dump_json:
            require(len(variants) == 1 and len(fixtures) == 1 and len(args.threads) == 1 and len(args.clip_norms) == 1,
                    "Offline dump validation requires one explicit variant/fixture/thread/clip norm")
        else:
            report["executable_before"] = identity(args.executable)
        source_paths = [path for pattern in ("*.c", "*.h") for path in BASE.glob(pattern)]
        report["sources_before"] = [identity(path) for path in sorted(source_paths)]
        report["planned_cases"] = len(variants) * len(fixtures) * len(args.threads) * len(args.clip_norms)
        save_report(args.report, report)
        with numerical_lock(args.lock_held):
            for variant in variants:
                arguments = list(VARIANTS[variant]) + extra
                options = expected_options(arguments)
                for fixture in fixtures:
                    for threads in dict.fromkeys(args.threads):
                        for clip in dict.fromkeys(args.clip_norms):
                            row = {"variant": variant, "fixture": fixture, "threads": threads,
                                   "clip_norm": clip, "kernel_arguments": arguments, "passed": False,
                                   "approximate_exp": bool(options["fast_exp"])}
                            try:
                                if args.dump_json:
                                    payload = args.dump_json.read_text(encoding="utf-8")
                                    row["dump"] = identity(args.dump_json)
                                else:
                                    executable = args.executable.resolve()
                                    command = [str(executable), FIXTURES[fixture]["dump_argument"],
                                               "--threads", str(threads), "--clip-norm", str(clip), *arguments]
                                    environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                                                       VECLIB_MAXIMUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                                                       MKL_NUM_THREADS=str(threads))
                                    completed = subprocess.run(command, capture_output=True, text=True, check=True,
                                                               env=environment, timeout=60, cwd=executable.parent)
                                    payload = completed.stdout
                                    row["command"], row["stderr"] = command, completed.stderr
                                row["dump_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                                data = parse_json(payload)
                                row.update(check_data(data, fixture, clip, threads, options, args.atol, args.rtol))
                            except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                                    subprocess.SubprocessError, ImportError) as error:
                                row["error"] = f"{type(error).__name__}: {error}"
                            report["cases"].append(row)
                            save_report(args.report, report)
            report["sources_after"] = [identity(path) for path in sorted(source_paths)]
            require(report["sources_before"] == report["sources_after"], "Kernel sources changed during independent checks")
            require(report["checker"] == identity(Path(__file__)), "Checker changed during independent checks")
            if not args.dump_json:
                report["executable_after"] = identity(args.executable)
                require(report["executable_before"] == report["executable_after"], "Executable changed during independent checks")
        report["passed"] = bool(report["cases"]) and all(row["passed"] for row in report["cases"])
        for fixture in fixtures:
            matching = [row for row in report["cases"] if row["fixture"] == fixture and "initial_parameter_float32_sha256" in row]
            require(len({row["initial_parameter_float32_sha256"] for row in matching}) <= 1,
                    f"Initial parameters differ across runtime options: {fixture}")
            require(len({row["context_target_ids_sha256"] for row in matching}) <= 1,
                    f"Input IDs differ across runtime options: {fixture}")
        report["same_initial_parameters_and_inputs_across_runtime_options"] = True
        report["status"] = "complete"
        report["passed_cases"] = sum(row["passed"] for row in report["cases"])
        report["failed_cases"] = len(report["cases"]) - report["passed_cases"]
        comparisons = [comparison for row in report["cases"] for comparison in row.get("comparisons", [])]
        report["comparison_count"] = len(comparisons)
        report["failed_comparisons"] = sum(not comparison["passed"] for comparison in comparisons)
        report["maximum_absolute_error"] = max((comparison["max_absolute_error"] for comparison in comparisons), default=None)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError, ImportError) as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["status"] = "failed"
    report["completed_at"] = datetime.now().astimezone().isoformat()
    save_report(args.report, report)
    if args.quiet:
        keys = ("kind", "status", "passed", "build_label", "planned_cases", "passed_cases", "failed_cases",
                "comparison_count", "failed_comparisons", "maximum_absolute_error", "error")
        rendered = {key: report[key] for key in keys if key in report}
        rendered["report"] = str(args.report.resolve()) if args.report else None
    else:
        rendered = report
    print(json.dumps(rendered, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
