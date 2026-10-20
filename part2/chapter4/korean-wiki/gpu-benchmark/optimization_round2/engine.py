"""Compose previously accepted MPS options without changing the optimizer.

Old standalone engines and failed native-fused results remain unchanged.
Every combination must independently pass the frozen CPU-oracle tolerances.
"""

from optimization import engine as original
from optimization import engine_refined as standalone

CASE_OPTIONS = {
    "metal_clip": dict(foreach=True, fused=False, zero_grad_set_to_none=False,
                      compiled_model=False, metal_clip=True),
    "metal_zero": dict(foreach=True, fused=False, zero_grad_set_to_none=True,
                      compiled_model=False, metal_clip=True),
    "metal_compiled": dict(foreach=True, fused=False, zero_grad_set_to_none=False,
                          compiled_model=True, metal_clip=True),
    "metal_compiled_zero": dict(foreach=True, fused=False, zero_grad_set_to_none=True,
                               compiled_model=True, metal_clip=True),
}
CASES = tuple(CASE_OPTIONS)
UnsupportedCase = original.UnsupportedCase
train_update = original.train_update
reset_after_warmup = original.reset_after_warmup
actual_devices = original.actual_devices


def configure_environment(case):
    if case not in CASE_OPTIONS:
        raise ValueError(f"Unknown round2 MPS case: {case}")
    standalone.configure_environment("metal_clip")


def build_model_optimizer(config, case):
    configure_environment(case)
    options = dict(CASE_OPTIONS[case])
    bundle = standalone.build_model_optimizer(config, "metal_clip")
    optimizer = bundle["optimizer"]
    parameter_ids = original._parameter_identity(bundle)
    bundle.update(case=case, variant=case,
                  zero_grad_set_to_none=options["zero_grad_set_to_none"])
    metadata = bundle["optimization_metadata"]
    metadata.update(case=case, requested_options=options,
                    refinement="Compose accepted Metal clipping, None gradients and model-only compilation; unchanged foreach AdamW.",
                    control_factory="unchanged optimization.engine_refined metal_clip")
    if options["compiled_model"]:
        from optimization.compiled_path import configure_model
        if configure_model(bundle) is not bundle:
            raise ValueError("Compiled wrapper replaced bundle")
    if bundle["optimizer"] is not optimizer or original._parameter_identity(bundle) != parameter_ids:
        raise ValueError("Combination changed optimizer or parameter references")
    for group in optimizer.param_groups:
        if group["foreach"] is not True or group["fused"] is not False or group["capturable"] is not False:
            raise ValueError("Combination changed accepted optimizer flags")
    metadata["compiled_wrapper_parameter_references_preserved"] = options["compiled_model"]
    return bundle


def runtime_metadata(case):
    configure_environment(case)
    return {**standalone.runtime_metadata("metal_clip"), "case": case,
            "requested_options": dict(CASE_OPTIONS[case]),
            "refinement_policy": "Independent gate for every combination; prior native-fused failure and numerical tolerances unchanged."}
