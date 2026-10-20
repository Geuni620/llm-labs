"""Strict, lazy model-only Inductor candidate for the isolated MPS benchmark.

Importing this module does not import Torch, compile a graph, or run a model.
The benchmark owns first-forward/backward support checks and timing. Errors
are intentionally allowed to escape; this module never substitutes eager or
CPU execution after an unsupported compiled path.
"""

from __future__ import annotations

import os
import time


COMPILE_OPTIONS = {
    "backend": "inductor",
    "fullgraph": True,
    "dynamic": False,
    "mode": "default",
}


def configure_model(bundle: dict) -> dict:
    """Wrap the existing model; preserve optimizer and every Parameter object.

    This configures a lazy wrapper. Returning successfully does not establish
    that this workload's forward or backward can compile, or that it is fast.
    Dynamo settings below deliberately last for this fresh child process so
    they still apply when compilation happens on the first actual execution.
    """
    started = time.perf_counter()
    expected_environment = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "PYTORCH_MPS_FAST_MATH": "0",
        "PYTORCH_MPS_PREFER_METAL": None,
    }
    if any(os.environ.get(name) != value for name, value in expected_environment.items()):
        raise RuntimeError("Configure fallback=0, FAST_MATH=0 and PREFER_METAL unset before importing Torch")
    if bundle["backend"] != "mps" or str(bundle["device"]) != "mps":
        raise ValueError("The compiled candidate requires the existing MPS bundle")

    import torch
    from torch._dynamo import config as dynamo_config

    if dynamo_config.disable or dynamo_config.debug_backend_override:
        raise RuntimeError("Dynamo disable/backend overrides are incompatible with this strict Inductor case")
    dynamo_config.suppress_errors = False
    dynamo_config.fail_on_recompile_limit_hit = True

    original_model = bundle["model"]
    parameters = bundle["parameters"]
    optimizer = bundle["optimizer"]
    expected_ids = tuple(id(parameter) for parameter in parameters)
    if tuple(id(parameter) for parameter in original_model.parameters()) != expected_ids:
        raise ValueError("Model and bundle parameter identity/order differ before compilation")
    if tuple(id(parameter) for group in optimizer.param_groups for parameter in group["params"]) != expected_ids:
        raise ValueError("Optimizer and bundle parameter identity/order differ before compilation")
    if any(parameter.device.type != "mps" or parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("The compiled comparison requires unchanged float32 MPS parameters")

    compiled = torch.compile(original_model, **COMPILE_OPTIONS)
    if tuple(id(parameter) for parameter in compiled.parameters()) != expected_ids:
        raise RuntimeError("The compiler wrapper replaced or reordered model parameters")

    bundle["model"] = compiled
    bundle.setdefault("optimization_metadata", {})["compile"] = {
        "enabled": True,
        "status": "wrapper_configured_first_forward_and_backward_pending",
        "options": dict(COMPILE_OPTIONS),
        "scope": "model forward and its autograd backward; loss, clipping and optimizer remain outside",
        "dynamo_suppress_errors": False,
        "fail_on_recompile_limit_hit": True,
        "dynamo_config_scope": "fresh benchmark child process lifetime",
        "environment": dict(expected_environment),
        "parameter_identity_and_order_preserved": True,
        "optimizer_object_preserved": bundle["optimizer"] is optimizer,
        "parameter_values_written_by_factory": False,
        "initial_bits_validation": "independent benchmark verifier; no value checksum is performed here",
        "wrapper_setup_seconds": time.perf_counter() - started,
        "first_execution_compile_seconds": None,
        "first_execution_timing_owner": "benchmark warmup; report separately from wrapper setup",
        "graph_support_verified": False,
        "performance_verified": False,
    }
    return bundle
