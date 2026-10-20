"""One fresh-process FP32 CPU/MPS NPLM check or bounded fixed-work timing.

Torch is imported lazily, after the parent/CLI configures MPS environment.
No CUDA, remote compute, autocast, model checkpoint or complete epoch is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback


BASE = Path(__file__).resolve().parent
WIKI = BASE.parent
VARIANTS = ("default", "preload_defer", "preload_defer_foreach", "preload_defer_foreach_metal")
OPTIMIZER = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01}
FIXTURES = {
    "tiny": {"vocab_size": 11, "context_size": 2, "embedding_dim": 3, "hidden_dim": 5, "batch_size": 4},
    "irregular": {"vocab_size": 73, "context_size": 3, "embedding_dim": 7, "hidden_dim": 67, "batch_size": 71},
}


class UnsupportedCase(RuntimeError):
    pass


def configure_environment(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError("Unknown benchmark variant")
    wanted = "1" if variant.endswith("_metal") else None
    if "torch" in sys.modules:
        if (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
                or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
                or os.environ.get("PYTORCH_MPS_PREFER_METAL") != wanted):
            raise RuntimeError("Configure the requested environment before importing Torch in a fresh child")
        return
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "0"
    if wanted is None:
        # Installed Torch tests variable presence; setting '0' is not baseline.
        os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    else:
        os.environ["PYTORCH_MPS_PREFER_METAL"] = wanted


def _sync(device) -> None:
    if str(device).startswith("mps"):
        import torch
        torch.mps.synchronize()


def build_model_optimizer(config: dict, backend: str, variant: str = "default") -> dict:
    """Public verifier API. Initial bits/layout come from the independent helper."""
    import torch
    from torch import nn
    from corpus_fixture import create_initial
    if backend == "mps" and not torch.backends.mps.is_available():
        raise UnsupportedCase("PyTorch reports MPS unavailable; CPU fallback is disabled")
    if backend not in {"cpu", "mps"} or variant not in VARIANTS:
        raise ValueError("Unsupported backend or benchmark policy")
    device = torch.device(backend)

    class Nplm(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
            self.hidden = nn.Linear(config["context_size"] * config["embedding_dim"], config["hidden_dim"])
            self.output = nn.Linear(config["hidden_dim"], config["vocab_size"])

        def forward(self, contexts):
            embedded = self.embedding(contexts).reshape(contexts.shape[0], -1)
            return self.output(torch.tanh(self.hidden(embedded)))

    initial_flat, layout = create_initial(config)
    initial_tensor = torch.from_numpy(initial_flat.copy()).to(device=device, dtype=torch.float32)
    model = Nplm().to(device=device, dtype=torch.float32)
    parameters = list(model.parameters())
    if len(parameters) != len(layout):
        raise ValueError("C initializer and model parameter layouts differ")
    with torch.no_grad():
        for parameter, entry in zip(parameters, layout):
            if list(parameter.shape) != list(entry["shape"]):
                raise ValueError("C initializer parameter shape differs")
            parameter.copy_(initial_tensor[entry["offset"]:entry["offset"] + entry["length"]].reshape(parameter.shape))
    foreach = "foreach" in variant
    optimizer = torch.optim.AdamW(parameters, **OPTIMIZER, foreach=foreach, fused=False, capturable=False)
    return {"model": model, "optimizer": optimizer, "parameters": parameters, "params": parameters, "layout": layout,
            "initial_flat": initial_flat, "initial_tensor": initial_tensor, "device": device,
            "backend": backend, "variant": variant, "config": dict(config), "foreach": foreach}


def train_update(bundle: dict, contexts, targets, clip_norm: float = 1.0, capture: bool = False) -> dict:
    """Public verifier API; capture tensors remain on the actual parameter device."""
    import torch
    from torch.nn import functional as functional
    optimizer, parameters = bundle["optimizer"], bundle["parameters"]
    optimizer.zero_grad(set_to_none=False)
    loss = functional.cross_entropy(bundle["model"](contexts), targets, reduction="mean")
    loss.backward()
    raw = torch.cat([parameter.grad.detach().reshape(-1) for parameter in parameters]) if capture else None
    # foreach clipping rejects MPS; error_if_nonfinite=True adds an implicit
    # scalar readback. We explicitly inspect every loss/norm at the chosen point.
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
    import torch
    with torch.no_grad():
        for parameter, entry in zip(bundle["parameters"], bundle["layout"]):
            parameter.copy_(bundle["initial_tensor"][entry["offset"]:entry["offset"] + entry["length"]].reshape(parameter.shape))
            if parameter.grad is not None:
                parameter.grad.zero_()
            state = bundle["optimizer"].state[parameter]
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    state[key].zero_()
            if "step" in state:
                if hasattr(state["step"], "zero_"):
                    state["step"].zero_()
                else:
                    state["step"] = 0
    _sync(bundle["device"])


def actual_devices(bundle: dict, contexts=None) -> dict:
    def unique(values):
        return sorted({str(value.device) for value in values})
    parameters, optimizer = bundle["parameters"], bundle["optimizer"]
    return {"parameters": unique(parameters), "inputs": [] if contexts is None else [str(contexts.device)],
            "gradients": unique(parameter.grad for parameter in parameters if parameter.grad is not None),
            "moments": unique(optimizer.state[parameter][key] for parameter in parameters
                              for key in ("exp_avg", "exp_avg_sq") if key in optimizer.state[parameter]),
            "optimizer_step_state": unique(optimizer.state[parameter]["step"] for parameter in parameters
                                           if hasattr(optimizer.state[parameter].get("step"), "device"))}


def runtime_metadata() -> dict:
    import torch
    return {"torch_version": torch.__version__, "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_fast_math_env": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "prefer_metal_env": os.environ.get("PYTORCH_MPS_PREFER_METAL"),
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads()}


def _memory(backend: str) -> dict:
    if backend != "mps":
        return {"scope": "outside-timer snapshots; not a peak measurement", "mps_current_allocated_bytes": None}
    import torch
    return {"scope": "outside-timer snapshots; not a peak measurement",
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "mps_recommended_max_bytes": torch.mps.recommended_max_memory() if hasattr(torch.mps, "recommended_max_memory") else None}


def _check_final_finite(bundle: dict) -> None:
    import torch
    for parameter in bundle["parameters"]:
        values = [parameter, parameter.grad]
        values += [bundle["optimizer"].state[parameter].get(key) for key in ("exp_avg", "exp_avg_sq")]
        for value in values:
            if value is not None and not bool(torch.isfinite(value).all().item()):
                raise ValueError("Nonfinite final parameter, gradient or Adam moment")


def run_check(args) -> dict:
    import torch
    from corpus_fixture import synthetic_batches, checksum_f32
    config = {**FIXTURES[args.fixture], "seed": args.seed, "threads": args.threads,
              "warmup_steps": 0, "steps": args.check_updates}
    bundle = build_model_optimizer(config, args.backend, args.variant)
    batches = synthetic_batches(config)
    history = []
    for step in range(args.check_updates):
        host_contexts, host_targets = batches[step % len(batches)]
        contexts = torch.from_numpy(host_contexts).to(bundle["device"])
        targets = torch.from_numpy(host_targets).to(bundle["device"])
        live = train_update(bundle, contexts, targets, args.clip_norm, capture=True)
        row = {key: value.detach().cpu().reshape(-1).tolist() if value.numel() != 1 else float(value.item())
               for key, value in live.items()}
        row.update(step=step + 1, contexts=host_contexts.tolist(), targets=host_targets.tolist(),
                   clip_active=row["clipping_scale"] < 1.0)
        history.append(row)
    _check_final_finite(bundle)
    return {"kind": "MPS_NPLM_NUMERICAL_CHECK", "status": "complete", "config": config,
            "backend": args.backend, "variant": args.variant, "fixture": args.fixture,
            "clip_norm": args.clip_norm, "dtype": "float32", "optimizer": {**OPTIMIZER, "foreach": bundle["foreach"], "fused": False, "capturable": False},
            "parameter_layout": bundle["layout"], "initial_params": bundle["initial_flat"].tolist(),
            "initial_parameter_checksum": checksum_f32(bundle["initial_flat"]),
            "history": history, "runtime": runtime_metadata(), "actual_devices": actual_devices(bundle, contexts),
            "actual_parameter_device": str(bundle["device"]), "finite_checks_passed": True,
            "grads": history[-1]["raw_gradients"], "optimizer_grads": history[-1]["clipped_gradients"],
            "updated_params": history[-1]["updated_params"], "moment1": history[-1]["moment1"], "moment2": history[-1]["moment2"]}


def run_benchmark(args) -> dict:
    import numpy as np
    import torch
    from corpus_fixture import Corpus, checksum_batch, checksum_f32
    config = {"vocab_size": 35328, "context_size": 8, "embedding_dim": 64, "hidden_dim": 128,
              "batch_size": args.batch_size, "seed": args.seed, "threads": args.threads,
              "warmup_steps": args.warmup_steps, "steps": args.steps}
    corpus = Corpus(args.corpus, config["context_size"], config["vocab_size"])
    sampler = corpus.sampler(args.seed)
    first_samples = sampler.first_samples(4)
    bundle = build_model_optimizer(config, args.backend, args.variant)
    for _ in range(args.warmup_steps):
        host_contexts, host_targets = sampler.next_batch(args.batch_size)
        train_update(bundle, torch.from_numpy(host_contexts).to(bundle["device"]),
                     torch.from_numpy(host_targets).to(bundle["device"]), args.clip_norm)
    reset_after_warmup(bundle)
    sampler.reset(args.seed)
    initial_sampler = sampler.snapshot()
    preload = args.variant != "default"
    losses, norms, scales = [], [], []
    first_batch = last_batch = None
    memory_before = _memory(args.backend)
    _sync(bundle["device"])
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_before, thread_before, begun = time.process_time(), time.thread_time(), time.perf_counter()
    preparation = 0.0
    if preload:
        before = time.perf_counter()
        batches = [sampler.next_batch(args.batch_size) for _ in range(args.steps)]
        first_batch, last_batch = batches[0], batches[-1]
        contexts_all = torch.from_numpy(np.stack([batch[0] for batch in batches])).to(bundle["device"])
        targets_all = torch.from_numpy(np.stack([batch[1] for batch in batches])).to(bundle["device"])
        preparation += time.perf_counter() - before
    for step in range(args.steps):
        if preload:
            contexts, targets = contexts_all[step], targets_all[step]
        else:
            before = time.perf_counter()
            batch = sampler.next_batch(args.batch_size)
            if first_batch is None:
                first_batch = batch
            last_batch = batch
            contexts = torch.from_numpy(batch[0]).to(bundle["device"])
            targets = torch.from_numpy(batch[1]).to(bundle["device"])
            preparation += time.perf_counter() - before
        live = train_update(bundle, contexts, targets, args.clip_norm)
        scales.append(live["clipping_scale"])
        if preload:
            losses.append(live["loss"])
            norms.append(live["gradient_norm"])
        else:
            loss, norm = float(live["loss"].item()), float(live["gradient_norm"].item())
            if not math.isfinite(loss) or not math.isfinite(norm):
                raise ValueError("Nonfinite per-update loss or gradient norm")
            losses.append(loss)
            norms.append(norm)
    if preload:
        loss_values = torch.stack(losses).detach().cpu().tolist()
        norm_values = torch.stack(norms).detach().cpu().tolist()
        if not all(math.isfinite(value) for value in [*loss_values, *norm_values]):
            raise ValueError("Nonfinite deferred loss/norm in at least one completed update")
    else:
        loss_values, norm_values = losses, norms
    _sync(bundle["device"])
    elapsed = time.perf_counter() - begun
    cpu_elapsed, thread_elapsed = time.process_time() - cpu_before, time.thread_time() - thread_before
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    memory_after = _memory(args.backend)
    audit_begun = time.perf_counter()
    _check_final_finite(bundle)
    scale_values = torch.stack(scales).detach().cpu().tolist()
    if not all(math.isfinite(value) and 0 < value <= 1 for value in scale_values):
        raise ValueError("Nonfinite or invalid actual clipping scale")
    _sync(bundle["device"])
    finite_audit_seconds = time.perf_counter() - audit_begun
    final_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in bundle["parameters"]]).cpu().numpy()
    devices = actual_devices(bundle, contexts)
    expected_device = args.backend
    if any(not value.startswith(expected_device) for key in ("parameters", "inputs", "gradients", "moments") for value in devices[key]):
        raise ValueError("Actual model/input/gradient/moment device differs; CPU fallback is not accepted")
    return {"kind": "REAL_WIKI_TORCH_CPU_MPS_FIXED_WORK", "status": "complete", "config": config,
            "backend": args.backend, "variant": args.variant, "dtype": "float32", "actual_parameter_device": str(bundle["device"]),
            "actual_devices": devices, "runtime": runtime_metadata(), "optimizer": {**OPTIMIZER, "foreach": bundle["foreach"], "fused": False, "capturable": False},
            "steps": args.steps, "target_prediction_events": args.steps * args.batch_size,
            "elapsed_seconds": elapsed, "targets_per_second": args.steps * args.batch_size / elapsed,
            "process_cpu_seconds": cpu_elapsed, "thread_cpu_seconds": thread_elapsed,
            "rusage": {"user_seconds": usage_after.ru_utime - usage_before.ru_utime,
                       "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
                       "voluntary_context_switches": usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                       "involuntary_context_switches": usage_after.ru_nivcsw - usage_before.ru_nivcsw},
            "corpus": corpus.metadata, "first_windows": first_samples,
            "initial_parameter_checksum": checksum_f32(bundle["initial_flat"]),
            "input_checksum": checksum_batch(*first_batch), "final_input_checksum": checksum_batch(*last_batch),
            "final_parameter_checksum": checksum_f32(final_parameters),
            "sampler_initial": initial_sampler, "sampler_final": sampler.snapshot(),
            "losses": loss_values, "gradient_norms": norm_values, "clipping_scales": scale_values, "finite_checks_passed": True,
            "final_finite_audit_seconds": finite_audit_seconds,
            "clip_active_updates": sum(value < 1 for value in scale_values),
            "warmup_reset": True, "optimizer_state_reset": "in-place weights, gradients, Adam step/m/v after two warmups; optimizer not recreated",
            "finite_policy": {"every_step_loss_and_norm": "all deferred tensors checked at end" if preload else "per-step host scalar readback",
                              "final_parameters_gradients_moments": True, "clip_foreach": False, "clip_error_if_nonfinite": False},
            "host_preparation_and_transfer_observed_seconds": preparation,
            "timing_scope": "Synchronized end-to-end wall includes all timed host batch gathering/preloading, transfers, updates and every-step loss/norm checks; excludes file RAM load, model/optimizer setup, warmups/reset and final full-state finite/checksum audit (reported separately).",
            "enqueue_scope": "Host preparation/transfer observations are not GPU kernel execution times; only synchronized end-to-end duration is used for throughput.",
            "memory_before": memory_before, "memory_after": memory_after}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--variant", choices=VARIANTS, default="default")
    parser.add_argument("--corpus", type=Path, default=WIKI / "training-corpora/wiki-k35328")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, choices=(128, 512), default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--clip-norm", type=float, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixture", choices=FIXTURES, default="tiny")
    parser.add_argument("--check-updates", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"status": "failed", "backend": args.backend, "variant": args.variant}
    try:
        if not (1 <= args.steps <= 64 and args.threads == 4 and args.seed == 42 and args.warmup_steps == 2
                and args.check_updates == 4 and math.isfinite(args.clip_norm) and args.clip_norm > 0):
            raise ValueError("Require bounded work, four CPU threads, seed42, two warmups and four-update checks")
        configure_environment(args.variant)
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        result = run_check(args) if args.check else run_benchmark(args)
    except UnsupportedCase as error:
        result.update(status="unsupported", error=f"{type(error).__name__}: {error}", cpu_fallback_used=False)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), cpu_fallback_used=False)
        if isinstance(error, NotImplementedError) or (isinstance(error, RuntimeError)
                and any(text in str(error).lower() for text in ("not implemented for", "not supported on mps", "does not support mps"))):
            result["status"] = "unsupported"
    rendered = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Preserving prior child result: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return 0 if result["status"] == "complete" else 2 if result["status"] == "unsupported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
