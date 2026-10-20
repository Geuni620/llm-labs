"""Independent MPS candidates retaining the numerically accepted optimizer.

The original optimization engine and its rejected native-fused results remain
unchanged. Each refined case changes at most one control setting. Importing
this module neither imports Torch nor runs/compiles a model.
"""

from __future__ import annotations

import time

from . import engine as original


CASES = ("control", "zero_none", "metal_clip", "compiled_model")
CASE_OPTIONS = {
    "control": {"foreach": True, "fused": False, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": False},
    "zero_none": {"foreach": True, "fused": False, "zero_grad_set_to_none": True, "compiled_model": False, "metal_clip": False},
    "metal_clip": {"foreach": True, "fused": False, "zero_grad_set_to_none": False, "compiled_model": False, "metal_clip": True},
    "compiled_model": {"foreach": True, "fused": False, "zero_grad_set_to_none": False, "compiled_model": True, "metal_clip": False},
}
UnsupportedCase = original.UnsupportedCase
train_update = original.train_update
reset_after_warmup = original.reset_after_warmup
actual_devices = original.actual_devices


def _case_options(case: str) -> dict:
    if case not in CASE_OPTIONS:
        raise ValueError(f"Unknown refined MPS case: {case}")
    return dict(CASE_OPTIONS[case])


def configure_environment(case: str) -> None:
    _case_options(case)
    original.configure_environment("control")


def build_model_optimizer(config: dict, case: str) -> dict:
    options = _case_options(case)
    begun = time.perf_counter()
    bundle = original.build_model_optimizer(config, "control")
    optimizer = bundle["optimizer"]
    parameter_ids = original._parameter_identity(bundle)
    bundle.update(case=case, variant=case, foreach=True, fused=False,
                  zero_grad_set_to_none=options["zero_grad_set_to_none"], metal_clip=options["metal_clip"])
    metadata = bundle["optimization_metadata"]
    metadata.update(
        case=case, requested_options=options,
        refinement="Retain the accepted foreach=True/fused=False AdamW; test one remaining change independently.",
        control_factory="unchanged optimization.engine.build_model_optimizer(config, 'control')",
        gradient_clip_implementation="native Metal blockwise norm and in-place scaling" if options["metal_clip"]
        else "unchanged torch.nn.utils.clip_grad_norm_",
        metal_step_stats_policy="new two-float buffer per update; norm/scale history must not alias later steps"
        if options["metal_clip"] else "not applicable",
    )
    for group in optimizer.param_groups:
        if group.get("foreach") is not True or group.get("fused") is not False or group.get("capturable") is not False:
            raise ValueError("Refined candidates must preserve the accepted foreach AdamW flags")
    if options["compiled_model"]:
        from .compiled_path import configure_model
        if configure_model(bundle) is not bundle:
            raise ValueError("Compiled wrapper must return the same bundle")
    if bundle["optimizer"] is not optimizer or original._parameter_identity(bundle) != parameter_ids:
        raise ValueError("Refined factory replaced the accepted optimizer or parameter objects/order")
    metadata.update(
        parameter_references_preserved=True,
        compiled_wrapper_parameter_references_preserved=options["compiled_model"],
        refined_factory_host_setup_seconds=time.perf_counter() - begun,
        setup_timing_scope="Host-observed setup only; first-forward compilation and warmup execution remain outside training timing.",
    )
    return bundle


def runtime_metadata(case: str) -> dict:
    return {**original.frozen.runtime_metadata(), "case": case, "requested_options": _case_options(case),
            "native_fused_requested": False,
            "optimizer_scope": "Unchanged foreach AdamW; only the compiled-model case wraps model forward/backward.",
            "refinement_policy": "Prior native-fused failure and numerical tolerances are preserved; no automatic fallback or tolerance relaxation."}
