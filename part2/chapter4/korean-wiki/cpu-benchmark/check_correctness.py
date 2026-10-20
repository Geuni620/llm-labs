#!/usr/bin/env python3
"""Validate the tiny C NPLM step against independent PyTorch math.

This checker runs only V=11/C=2/D=3/H=5/B=4, one optimizer step.  PyTorch is
an independent numerical reference here, not the C timing implementation.
The existing Wiki trainer, corpora and checkpoints are never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys


TINY_CONFIG = {
    "vocab_size": 11, "context_size": 2, "embedding_dim": 3,
    "hidden_dim": 5, "batch_size": 4, "seed": 42,
    "warmup_steps": 0, "steps": 1,
}
OPTIMIZER = {
    "name": "AdamW", "lr": 0.001, "weight_decay": 0.01,
    "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
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


def numeric_list(value, length, label):
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain exactly {length} numbers")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) for x in value), f"Nonfinite/non-numeric {label}")
    return value


def validate_schema(data):
    require(data.get("kind") == "SYNTHETIC_C_NPLM_CHECK", "Wrong dump kind")
    config = data["config"]
    for name, expected in TINY_CONFIG.items():
        actual = config.get(name)
        require(type(actual) is int and actual == expected,
                f"Tiny check requires {name}={expected}, got {actual!r}")
    require(type(config.get("threads")) is int and config["threads"] >= 1,
            "threads must be a positive integer")
    shapes = [
        ("embedding", [11, 3]), ("hidden_weight", [5, 6]),
        ("hidden_bias", [5]), ("output_weight", [11, 5]),
        ("output_bias", [11]),
    ]
    require(len(data["parameter_layout"]) == len(shapes), "Wrong parameter layout")
    offset = 0
    for entry, (name, shape) in zip(data["parameter_layout"], shapes):
        length = math.prod(shape)
        require(entry.get("name") == name and entry.get("shape") == shape
                and entry.get("offset") == offset and entry.get("length") == length,
                f"Incorrect parameter layout for {name}")
        offset += length
    for field in ("initial_params", "grads", "optimizer_grads", "updated_params"):
        numeric_list(data[field], offset, field)
    for field in ("loss", "gradient_norm", "clipping_scale"):
        numeric_list([data[field]], 1, field)
    require(data["gradient_norm"] >= 0 and 0 < data["clipping_scale"] <= 1,
            "Invalid gradient norm/clipping scale")
    contexts = data["contexts"]
    require(isinstance(contexts, list) and len(contexts) == 4,
            "contexts must have four rows")
    require(all(isinstance(row, list) and len(row) == 2 for row in contexts),
            "contexts must have two token IDs per row")
    ids = [token for row in contexts for token in row]
    require(all(type(token) is int and 0 <= token < 11 for token in ids),
            "Context ID outside the tiny vocabulary")
    require(isinstance(data["targets"], list) and len(data["targets"]) == 4
            and all(type(token) is int and 0 <= token < 11 for token in data["targets"]),
            "targets must contain four vocabulary IDs")
    counts = Counter(ids)
    repeated = sorted(token for token, count in counts.items() if count > 1)
    unused = sorted(set(range(11)) - set(ids))
    require(repeated, "Tiny batch must repeat an embedding ID to exercise scatter-add")
    require(unused, "Tiny batch must leave unused embedding rows to exercise zero gradients")
    for name, expected in OPTIMIZER.items():
        actual = data["optimizer"].get(name)
        if isinstance(expected, str):
            require(actual == expected, f"Unexpected optimizer {name}")
        else:
            require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    and math.isfinite(actual)
                    and math.isclose(actual, expected, rel_tol=2e-7, abs_tol=1e-15),
                    f"Unexpected optimizer {name}: {actual!r}")
    return repeated, unused


def compare(torch, label, actual, expected, atol, rtol):
    actual = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    expected = torch.as_tensor(expected, dtype=torch.float64).reshape(-1)
    require(actual.shape == expected.shape, f"Shape mismatch: {label}")
    absolute = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    failed = (~torch.isfinite(actual)) | (~torch.isfinite(expected)) | (absolute > allowed)
    worst = int(torch.argmax(absolute).item())
    return {
        "name": label, "passed": not bool(failed.any()), "elements": actual.numel(),
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float((absolute / expected.abs().clamp_min(1e-12)).max()),
        "failed_elements": int(failed.sum()), "worst_absolute_error_index": worst,
        "actual_at_worst": float(actual[worst]), "expected_at_worst": float(expected[worst]),
        "atol": atol, "rtol": rtol,
    }


def check(data, atol, rtol):
    repeated, unused = validate_schema(data)
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    class TinyNplm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(11, 3)
            self.hidden = nn.Linear(6, 5)
            self.output = nn.Linear(5, 11)

        def forward(self, contexts):
            embedded = self.embedding(contexts).reshape(4, 6)
            return self.output(torch.tanh(self.hidden(embedded)))

    model = TinyNplm().to(dtype=torch.float32, device="cpu")
    parameters = [model.embedding.weight, model.hidden.weight, model.hidden.bias,
                  model.output.weight, model.output.bias]
    initial = torch.tensor(data["initial_params"], dtype=torch.float32)
    require(bool(torch.isfinite(initial).all()), "Initial parameters do not fit finite float32")
    with torch.no_grad():
        for entry, parameter in zip(data["parameter_layout"], parameters):
            parameter.copy_(initial[entry["offset"]:entry["offset"] + entry["length"]]
                            .reshape(entry["shape"]))
    optimizer = torch.optim.AdamW(parameters, lr=OPTIMIZER["lr"],
                                 betas=(OPTIMIZER["beta1"], OPTIMIZER["beta2"]),
                                 eps=OPTIMIZER["eps"], weight_decay=OPTIMIZER["weight_decay"],
                                 foreach=False, fused=False)
    contexts = torch.tensor(data["contexts"], dtype=torch.int64)
    targets = torch.tensor(data["targets"], dtype=torch.int64)
    loss = F.cross_entropy(model(contexts), targets, reduction="mean")
    loss.backward()
    gradients = torch.cat([p.grad.detach().reshape(-1) for p in parameters])
    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(gradients).all()),
            "The independent float32 loss/gradients are not finite")
    require(torch.count_nonzero(model.embedding.weight.grad[unused]).item() == 0,
            "Independent reference has unexpected unused-embedding gradients")
    require(any(bool(torch.any(model.embedding.weight.grad[token] != 0)) for token in repeated),
            "Repeated-token fixture has no nonzero embedding gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True,
                                                 foreach=False)
    optimizer_gradients = torch.cat([p.grad.detach().reshape(-1) for p in parameters])
    optimizer.step()
    updated = torch.cat([p.detach().reshape(-1) for p in parameters])
    comparisons = [compare(torch, "mean_cross_entropy", [data["loss"]], [float(loss.detach())],
                           atol, rtol)]
    comparisons.append(compare(torch, "global_gradient_norm", [data["gradient_norm"]],
                               [float(gradient_norm)], atol, rtol))
    comparisons.append(compare(torch, "gradient_clipping_scale", [data["clipping_scale"]],
                               [min(1.0, 1.0 / (float(gradient_norm) + 1e-6))], atol, rtol))
    comparisons.append(compare(torch, "optimizer_gradients_after_norm1_clipping",
                               data["optimizer_grads"], optimizer_gradients, atol, rtol))
    for entry in data["parameter_layout"]:
        start, stop = entry["offset"], entry["offset"] + entry["length"]
        comparisons.append(compare(torch, entry["name"] + ".gradient",
                                   data["grads"][start:stop], gradients[start:stop], atol, rtol))
        comparisons.append(compare(torch, entry["name"] + ".AdamW_updated",
                                   data["updated_params"][start:stop], updated[start:stop], atol, rtol))

    # Independent scalar t=1 AdamW identity: m_hat=g, v_hat=g*g.  This also
    # checks weight decay on untouched (zero-gradient) embedding parameters.
    scalar_updated = [p * (1 - OPTIMIZER["lr"] * OPTIMIZER["weight_decay"])
                      - OPTIMIZER["lr"] * g / (abs(g) + OPTIMIZER["eps"])
                      for p, g in zip(data["initial_params"], data["optimizer_grads"])]
    comparisons.append(compare(torch, "bias_corrected_first_step_AdamW_scalar",
                               data["updated_params"], scalar_updated, atol, rtol))
    unused_indices = [token * 3 + dimension for token in unused for dimension in range(3)]
    require(all(data[field][index] == 0 for field in ("grads", "optimizer_grads")
                for index in unused_indices),
            "C unused embedding rows must have exactly zero gradient")
    comparisons.append(compare(torch, "zero_gradient_embedding_AdamW_decay",
                               [data["updated_params"][i] for i in unused_indices],
                               [scalar_updated[i] for i in unused_indices],
                               0, 4 * torch.finfo(torch.float32).eps))
    return {
        "kind": "SYNTHETIC_C_NPLM_CORRECTNESS", "passed": all(row["passed"] for row in comparisons),
        "scope": "one tiny synthetic CPU forward/backward/AdamW step, not Wiki training or timing",
        "config": data["config"], "torch_version": torch.__version__,
        "gradient_clipping": {"max_norm": 1.0, "norm_type": 2.0,
                              "active_in_tiny_fixture": float(gradient_norm) > 1.0},
        "parameter_count": len(data["initial_params"]), "repeated_embedding_ids": repeated,
        "unused_embedding_ids": unused, "zero_gradient_embedding_parameters": len(unused_indices),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--executable", type=Path,
                        default=Path(__file__).resolve().parent / "cpu_benchmark")
    source.add_argument("--dump-json", type=Path, help="Check an existing --dump-check JSON file")
    parser.add_argument("--report", type=Path, help="Optionally save the checker result as JSON")
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    report = {"kind": "SYNTHETIC_C_NPLM_CORRECTNESS", "passed": False}
    try:
        require(math.isfinite(args.atol) and math.isfinite(args.rtol)
                and args.atol >= 0 and args.rtol >= 0, "Tolerances must be finite and nonnegative")
        if args.dump_json:
            payload = args.dump_json.read_text(encoding="utf-8")
            source_description = str(args.dump_json.resolve())
        else:
            executable = args.executable.resolve()
            environment = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1")
            completed = subprocess.run([str(executable), "--dump-check", "--threads", "1"],
                                       check=True, capture_output=True, text=True,
                                       env=environment, timeout=60, cwd=executable.parent)
            payload = completed.stdout
            source_description = str(executable)
        data = json.loads(payload, object_pairs_hook=unique_object)
        report = check(data, args.atol, args.rtol)
        report["source"] = source_description
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, ImportError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
