"""Explicit MPS optimizer/clip candidates beside the frozen FP32 benchmark core.

Importing this module does not import Torch, build models, compile, or execute
operators. Only the caller may run checked cases under the shared timing lock.
Unsupported fused/compiled operations propagate; there is no eager/CPU fallback.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time


GPU_BASE = Path(__file__).resolve().parents[1]
if str(GPU_BASE) not in sys.path:
    sys.path.insert(0, str(GPU_BASE))
import benchmark_mps as frozen


CASES = ("control", "native_fused", "native_fused_zero_none", "metal_clip_fused", "compiled_model_fused")
CASE_OPTIONS = {
    "control": {"foreach": True, "fused": False, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": False},
    "native_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": False},
    "native_fused_zero_none": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": False, "metal_clip": False},
    "metal_clip_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": False, "metal_clip": True},
    "compiled_model_fused": {"foreach": False, "fused": True, "zero_grad_set_to_none": True, "compiled_model": True, "metal_clip": False},
}
BASE_VARIANT = "preload_defer_foreach"
UnsupportedCase = frozen.UnsupportedCase


def _case_options(case: str) -> dict:
    if case not in CASE_OPTIONS:
        raise ValueError(f"Unknown MPS optimization case: {case}")
    return dict(CASE_OPTIONS[case])


def configure_environment(case: str) -> None:
    _case_options(case)
    # All cases use the same ordinary Metal preference. A value
    # of '0' still counts as presence in installed Torch, so it remains unset.
    frozen.configure_environment(BASE_VARIANT)


def _parameter_identity(bundle: dict) -> tuple[int, ...]:
    expected = tuple(id(parameter) for parameter in bundle["parameters"])
    actual = tuple(id(parameter) for parameter in bundle["model"].parameters())
    grouped = tuple(id(parameter) for group in bundle["optimizer"].param_groups for parameter in group["params"])
    if actual != expected or grouped != expected:
        raise ValueError("Model wrapper or optimizer changed parameter objects/order")
    return expected


def build_model_optimizer(config: dict, case: str) -> dict:
    """Retain C initialization/layout, changing only the requested optimizer path."""
    options = _case_options(case)
    configure_environment(case)
    begun = time.perf_counter()
    bundle = frozen.build_model_optimizer(config, "mps", BASE_VARIANT)
    if options["fused"]:
        import torch
        # The original optimizer has not run or allocated moments. Replacement
        # is setup-only; the post-warmup reset never recreates this optimizer.
        if bundle["optimizer"].state:
            raise ValueError("Cannot replace an optimizer after state allocation")
        bundle["optimizer"] = torch.optim.AdamW(
            bundle["parameters"], **frozen.OPTIMIZER,
            foreach=False, fused=True, capturable=False,
        )
    bundle.update(case=case, base_variant=BASE_VARIANT, variant=case,
                  foreach=options["foreach"], fused=options["fused"],
                  zero_grad_set_to_none=options["zero_grad_set_to_none"], metal_clip=options["metal_clip"])
    bundle["optimization_metadata"] = {
        "case": case, "requested_options": options,
        "optimizer": {"name": "AdamW", **frozen.OPTIMIZER,
                      "foreach": options["foreach"], "fused": options["fused"], "capturable": False},
        "fp32_only": True, "clip_foreach": False, "clip_error_if_nonfinite": False,
        "gradient_metadata_guard": "Every case checks all five dense float32 MPS gradients and parameter-matching shapes after backward; no gradient values are read to CPU.",
        "gradient_clip_implementation": "native Metal blockwise norm and in-place scaling" if options["metal_clip"]
        else "unchanged torch.nn.utils.clip_grad_norm_",
        "metal_step_stats_policy": "new two-float buffer per update; norm/scale history must not alias later steps"
        if options["metal_clip"] else "not applicable",
        "unsupported_policy": "Propagate the original exception; no eager/CPU/foreach fallback.",
        "parameter_references_preserved": True,
    }
    before = _parameter_identity(bundle)
    optimizer = bundle["optimizer"]
    for group in optimizer.param_groups:
        if group.get("foreach") is not options["foreach"] or group.get("fused") is not options["fused"]:
            raise ValueError("Actual optimizer group flags differ from the requested case")
        if group.get("capturable") is not False:
            raise ValueError("Capturable AdamW is outside this comparison")
    if options["compiled_model"]:
        from optimization.compiled_path import configure_model
        configured = configure_model(bundle)
        if configured is not bundle:
            raise ValueError("configure_model must return the same bundle")
        if bundle["optimizer"] is not optimizer or _parameter_identity(bundle) != before:
            raise ValueError("Model compilation replaced optimizer/parameter references")
    bundle["optimization_metadata"].update(
        host_setup_wall_seconds=time.perf_counter() - begun,
        setup_timing_scope="Host-observed construction/wrapping only; first-forward compilation and warmup execution are separate and excluded from training timing.",
        compiled_wrapper_parameter_references_preserved=options["compiled_model"] and _parameter_identity(bundle) == before,
    )
    return bundle


def train_update(bundle: dict, contexts, targets, clip_norm: float = 1.0, capture: bool = False) -> dict:
    """Preserve update order; only requested clearing/clip implementations differ.

    Capture uses the same eight device-resident fields as the original engine.
    Numerical/finite checks and readback timing remain the caller's policy.
    """
    import torch
    from torch.nn import functional as functional
    optimizer, parameters = bundle["optimizer"], bundle["parameters"]
    optimizer.zero_grad(set_to_none=bundle["zero_grad_set_to_none"])
    loss = functional.cross_entropy(bundle["model"](contexts), targets, reduction="mean")
    loss.backward()
    if len(parameters) != 5:
        raise ValueError("The fixed NPLM must have exactly five parameter tensors")
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if (gradient is None or gradient.is_sparse or gradient.layout != torch.strided
                or gradient.dtype != torch.float32 or gradient.device.type != "mps"
                or gradient.shape != parameter.shape):
            raise ValueError(f"Invalid dense float32 MPS gradient metadata for parameter {index}")
    raw = torch.cat([parameter.grad.detach().reshape(-1) for parameter in parameters]) if capture else None
    if bundle["metal_clip"]:
        from optimization.metal_clip import clip_gradients
        norm, scale = clip_gradients(bundle, clip_norm)
    else:
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, foreach=False, error_if_nonfinite=False)
        scale = torch.clamp(clip_norm / (norm.detach() + 1e-6), max=1.0)
    clipped = torch.cat([parameter.grad.detach().reshape(-1) for parameter in parameters]) if capture else None
    optimizer.step()
    result = {"loss": loss.detach(), "gradient_norm": norm.detach(), "clipping_scale": scale}
    if capture:
        result.update(raw_gradients=raw, clipped_gradients=clipped,
                      updated_params=torch.cat([parameter.detach().reshape(-1) for parameter in parameters]),
                      moment1=torch.cat([optimizer.state[parameter]["exp_avg"].detach().reshape(-1) for parameter in parameters]),
                      moment2=torch.cat([optimizer.state[parameter]["exp_avg_sq"].detach().reshape(-1) for parameter in parameters]))
    return result


def reset_after_warmup(bundle: dict) -> None:
    """Keep allocated optimizer state; fused step tensors may reside on MPS."""
    optimizer = bundle["optimizer"]
    before = _parameter_identity(bundle)
    frozen.reset_after_warmup(bundle)
    if bundle["zero_grad_set_to_none"]:
        optimizer.zero_grad(set_to_none=True)
        frozen._sync(bundle["device"])
    if bundle["optimizer"] is not optimizer or _parameter_identity(bundle) != before:
        raise ValueError("Reset replaced optimizer/parameter references")
    bundle["optimization_metadata"]["reset"] = {
        "optimizer_recreated": False, "parameter_references_preserved": True,
        "weights": "copied from unchanged C-initialized tensor in place",
        "adam_moments_and_step": "existing state tensors zeroed in place on their actual device",
        "gradient_clear_policy": "None" if bundle["zero_grad_set_to_none"] else "existing gradient tensors zeroed in place",
        "device_synchronized": True,
    }


def actual_devices(bundle: dict, contexts=None) -> dict:
    return frozen.actual_devices(bundle, contexts)


def runtime_metadata(case: str) -> dict:
    options = _case_options(case)
    return {**frozen.runtime_metadata(), "case": case, "requested_options": options,
            "native_fused_step_device_policy": "CPU or MPS scalar state is recorded explicitly; model/gradients/moments must remain MPS.",
            "optimizer_scope": "Only the model may be compiled; AdamW remains outside torch.compile."}
