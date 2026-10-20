#!/usr/bin/env python3
"""One real-Wiki B512 update versus an independent CPU-PyTorch oracle.

Large arrays remain in memory. The JSON contains numerical summaries only.
This is a separate batch-size experiment, not evidence of equal training
quality at four times fewer Adam updates. Root must already own the shared
CPU/MPS numerical lock. No compilation, throughput timing or training occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import benchmark_mps as engine
import verify_mps as verifier
from corpus_fixture import Corpus, FULL_CONFIG, checksum_batch, checksum_f32, helper_library_path, parameter_layout


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def check(variant, corpus_path):
    engine.configure_environment(variant)
    torch = verifier.configure_reference_threads()
    config = {**FULL_CONFIG, "batch_size": 512, "threads": 4, "warmup_steps": 2, "steps": 1}
    corpus = Corpus(corpus_path, config["context_size"], config["vocab_size"])
    verifier.require(corpus.windows == 124232535, "The retained Wiki target pool changed")
    sampler = corpus.sampler(42)
    sampler_initial = sampler.snapshot()
    bundle = engine.build_model_optimizer(config, "mps", variant)
    initial, layout = bundle["initial_flat"], parameter_layout(config)
    verifier.require(bundle["layout"] == layout and initial.size == 6883968, "Model shape/parameter order changed")
    initial_checksum = checksum_f32(initial)
    verifier.require(initial_checksum == "b5b7b03ca89a72af", "Initializer bits differ from the preserved C model")
    actual_initial = torch.cat([parameter.detach().reshape(-1) for parameter in bundle["parameters"]]).cpu().numpy()
    verifier.require(actual_initial.tobytes() == initial.tobytes(), "Actual MPS initial values differ from the C initializer")
    del actual_initial
    warmup_evidence = []
    for _ in range(2):
        warm_contexts, warm_targets = sampler.next_batch(512)
        live = engine.train_update(bundle, torch.from_numpy(warm_contexts).to(bundle["device"]),
                                   torch.from_numpy(warm_targets).to(bundle["device"]), 1.0, capture=False)
        scalars = {key: float(value.item()) for key, value in live.items()}
        verifier.require(all(math.isfinite(value) for value in scalars.values()), "Non-finite warmup loss/norm/scale")
        warmup_evidence.append(scalars)
    torch.mps.synchronize()
    steps_before_reset = [float(bundle["optimizer"].state[parameter]["step"].item())
                          for parameter in bundle["parameters"]]
    verifier.require(all(value == 2 for value in steps_before_reset), "Two real warmup updates did not complete")
    engine.reset_after_warmup(bundle)
    restored = torch.cat([parameter.detach().reshape(-1) for parameter in bundle["parameters"]]).cpu().numpy()
    verifier.require(restored.tobytes() == initial.tobytes(), "Warmup reset did not restore exact initializer bits")
    del restored
    for parameter in bundle["parameters"]:
        state = bundle["optimizer"].state[parameter]
        verifier.require(float(state["step"].item()) == 0, "Adam step counter was not reset")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key].detach().cpu()
            verifier.require(bool(torch.isfinite(value).all()) and bool(torch.all(value == 0)),
                             f"Warmup reset did not clear {key}")
        verifier.require(parameter.grad is not None and bool(torch.all(parameter.grad.detach().cpu() == 0)),
                         "Warmup reset did not clear parameter gradients")
    sampler.reset(42)
    verifier.require(sampler.snapshot() == sampler_initial, "Warmup sampler reset did not restore its initial state")
    contexts, targets = sampler.next_batch(512)
    first128 = checksum_batch(contexts[:128], targets[:128])
    verifier.require(first128 == "1a55aed564d5b308", "First 128 integer contexts/targets differ from the preserved C batch")
    reference = verifier.build_reference(torch, config, initial, layout)
    device_contexts = torch.from_numpy(contexts).to(bundle["device"])
    device_targets = torch.from_numpy(targets).to(bundle["device"])
    captured = engine.train_update(bundle, device_contexts, device_targets, 1.0, capture=True)
    torch.mps.synchronize()
    for key, tensor in captured.items():
        verifier.require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                         and str(tensor.device).split(":")[0] == "mps", f"Actual {key} is not MPS float32")
    actual = {key: tensor.detach().cpu() for key, tensor in captured.items()}
    verifier.require(all(bool(torch.isfinite(value).all()) for value in actual.values()), "Non-finite device result")
    expected = verifier.reference_update(torch, reference, contexts, targets, 1.0)
    comparisons = [verifier.compare(torch, "batch512.update1." + key, actual[key], expected[key],
                                    key if key in ("moment1", "moment2") else "ordinary") for key in expected]
    runtime, devices = engine.runtime_metadata(), engine.actual_devices(bundle, device_contexts)
    verifier.validate_runtime(runtime, variant)
    verifier.validate_devices(devices, "mps")
    for parameter in bundle["parameters"]:
        verifier.require(float(bundle["optimizer"].state[parameter]["step"].item()) == 1,
                         "Optimizer did not complete exactly one update")
    return {"status": "complete", "passed": all(row["passed"] for row in comparisons),
            "backend": "mps", "variant": variant, "dtype": "float32", "config": config,
            "parameter_count": int(initial.size), "updates": 1, "clip_norm": 1.0,
            "target_prediction_events": 512, "runtime": runtime, "actual_devices": devices,
            "warmup_updates": 2, "total_optimizer_updates_executed": 3,
            "warmup_scalar_evidence": warmup_evidence,
            "warmup_reset_verified": {"initial_parameter_bits_restored": True, "adam_step_zero": True,
                                       "moment1_zero": True, "moment2_zero": True, "gradients_zero": True,
                                       "sampler_restored": True, "optimizer_steps_before_reset": steps_before_reset},
            "all_arrays_finite_checked_on_CPU": True, "second_moments_nonnegative_checked": True,
            "initial_parameter_bits_match_C": True, "initial_parameter_checksum": initial_checksum,
            "input_checksum": checksum_batch(contexts, targets), "first128_input_checksum": first128,
            "sampler_initial": sampler_initial, "sampler_after_one_batch": sampler.snapshot(),
            "first_windows": corpus.first_samples(), "comparison_count": len(comparisons),
            "failed_comparisons": sum(not row["passed"] for row in comparisons), "comparisons": comparisons,
            "all_large_arrays_kept_in_memory": True, "quality_comparison_valid": False,
            "batch_size_scope": "B512 changes optimizer updates per target versus B128; this one-step correctness check does not establish equal quality or convergence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=engine.VARIANTS, default="preload_defer_foreach")
    parser.add_argument("--corpus", type=Path, default=ROOT / "training-corpora/wiki-k35328")
    parser.add_argument("--report", type=Path, default=BASE / "batch512-verification.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Preserving existing numerical report: {args.report}")
    report = {"kind": "MPS_REAL_WIKI_BATCH512_ONE_UPDATE_GATE", "status": "running", "passed": False,
              "started_at": datetime.now().astimezone().isoformat(), "variant": args.variant,
              "tolerances_predeclared": verifier.TOLERANCES,
              "scope": "One actual Wiki batch, FP32, independent CPU oracle; no throughput measurement, complete epoch or quality comparison."}
    try:
        paths = [Path(__file__), BASE / "benchmark_mps.py", BASE / "verify_mps.py",
                 BASE / "corpus_fixture.py", BASE / "init_reference.c", helper_library_path()]
        report["sources_before"] = {str(path.resolve()): verifier.identity(path) for path in paths}
        with verifier.numerical_lock(externally_held=True):
            report.update(check(args.variant, args.corpus))
            report["sources_after"] = {str(path.resolve()): verifier.identity(path) for path in paths}
            verifier.require(report["sources_before"] == report["sources_after"], "Frozen sources changed during the gate")
    except Exception as error:
        unsupported = verifier.is_unsupported(error) or type(error).__name__ == "UnsupportedCase"
        report.update(status="unsupported" if unsupported else "failed", passed=False,
                      error=f"{type(error).__name__}: {error}", cpu_fallback_used=False)
    report["completed_at"] = datetime.now().astimezone().isoformat()
    verifier.atomic_json(args.report, report)
    if args.quiet:
        rendered = {key: report[key] for key in ("kind", "status", "passed", "variant", "comparison_count", "failed_comparisons", "error") if key in report}
        rendered["report"] = str(args.report.resolve())
    else:
        rendered = report
    print(json.dumps(rendered, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 2 if report["status"] == "unsupported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
