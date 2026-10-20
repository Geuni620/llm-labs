"""Fresh-child confirmation wrapper around the unchanged FP32 timing core.

This file only extends the accepted fixed-work step bound. The caller owns
the shared timing lock, process-group deadline and serial trial scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import traceback

import benchmark_mps as engine


ENGINE_SHA256 = "3757b3a5c5da8ecb9cd69bf9ba898d29dfe441617786ceaa9d36c2ab5a504935"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--variant", choices=engine.VARIANTS, default="default")
    parser.add_argument("--corpus", type=Path, default=engine.WIKI / "training-corpora/wiki-k35328")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, choices=(128, 512), default=128)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--threads", type=int, choices=(4,), default=4)
    parser.add_argument("--warmup-steps", type=int, choices=(2,), default=2)
    parser.add_argument("--clip-norm", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 8 <= args.steps <= 1024:
        parser.error("Use one caller-approved fixed step count between 8 and 1024")
    args.corpus = args.corpus.expanduser().resolve(strict=True)
    if not args.corpus.is_dir():
        parser.error("--corpus must resolve to an existing absolute directory")
    if args.output:
        args.output = args.output.expanduser().absolute()
        if args.output.exists():
            raise FileExistsError(f"Preserving existing confirmation result: {args.output}")

    def canceled(signum, frame):
        raise KeyboardInterrupt(f"Confirmation canceled by signal {signum}")
    signal.signal(signal.SIGTERM, canceled)
    result = {"status": "failed", "backend": args.backend, "variant": args.variant,
              "requested_steps": args.steps, "batch_size": args.batch_size}
    code = 1
    try:
        engine_path = Path(engine.__file__).resolve()
        source_hash = hashlib.sha256(engine_path.read_bytes()).hexdigest()
        if source_hash != ENGINE_SHA256:
            raise ValueError("The frozen benchmark core changed; no measurement was started")
        engine.configure_environment(args.variant)
        # No Torch import occurs before configure_environment in this child.
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        result = engine.run_benchmark(args)
        if hashlib.sha256(engine_path.read_bytes()).hexdigest() != source_hash:
            raise ValueError("The frozen benchmark core changed during the observation")
        result["confirmation_wrapper"] = {
            "engine_path": str(engine_path), "engine_sha256": source_hash,
            "core": "unchanged benchmark_mps.run_benchmark(args)",
            "lock_and_deadline": "owned by external coordinator; this wrapper does not launch another process",
            "batch_size_changes_training_conditions": args.batch_size == 512,
            "optimizer_updates": args.steps, "prediction_events": args.steps * args.batch_size,
            "quality_equality_unverified": True,
        }
        code = 0
    except engine.UnsupportedCase as error:
        result.update(status="unsupported", error=f"{type(error).__name__}: {error}", cpu_fallback_used=False)
        code = 2
    except KeyboardInterrupt as error:
        result.update(status="canceled", error=str(error), cpu_fallback_used=False)
        code = 130
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), cpu_fallback_used=False)
        if isinstance(error, NotImplementedError) or (isinstance(error, RuntimeError) and any(
                message in str(error).lower() for message in
                ("not implemented for", "not supported on mps", "does not support mps"))):
            result["status"], code = "unsupported", 2
    rendered = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
