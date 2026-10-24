"""Reproduce the initial reference mismatch without loading the C runtime."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as functional

from evaluate_checkpoint import (ARTIFACT, TOLERANCE, identity, now, output_path,
                                 read, write)


def main():
    raw = read(output_path(".raw.json"))
    selected = raw["selection"]
    if identity(selected["checkpoint"]["path"]) != selected["checkpoint"]["identity"]:
        raise ValueError("Selected weights changed")
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with np.load(ARTIFACT, allow_pickle=False) as saved:
        x = torch.from_numpy(saved["x"].astype(np.int64))
        y = torch.from_numpy(saved["y"].astype(np.int64))
        c_logits = torch.from_numpy(saved["logits"])
        c_stats = saved["stats"]
    with np.load(selected["checkpoint"]["path"], allow_pickle=False) as saved:
        parameters = saved["parameters"]
    cfg = selected["config"]
    vocab, context, dim, hidden = [cfg[key] for key in ("vocab_size", "context_size", "embedding_dim", "hidden_dim")]
    arrays, offset = [], 0
    for shape in ((vocab, dim), (hidden, context * dim), (hidden,), (vocab, hidden), (vocab,)):
        size = math.prod(shape)
        arrays.append(torch.from_numpy(parameters[offset:offset + size].reshape(shape)))
        offset += size
    embedding, weight, bias, out_weight, out_bias = arrays
    with torch.no_grad():
        encoded = embedding[x].reshape(len(x), context * dim)
        fused = functional.linear(torch.tanh(functional.linear(encoded, weight, bias)), out_weight, out_bias)
        separate = functional.linear(torch.tanh(functional.linear(encoded, weight) + bias), out_weight) + out_bias
        cases = []
        for name, logits in (("initial_fused_bias_reference", fused), ("aligned_separate_bias_reference", separate)):
            error = np.abs(logits.numpy().astype(np.float64) - c_logits.numpy().astype(np.float64))
            allowed = TOLERANCE["logits_atol"] + TOLERANCE["logits_rtol"] * np.abs(logits.numpy().astype(np.float64))
            nll = functional.cross_entropy(logits.double(), y, reduction="sum").item()
            cases.append({"name": name, "max_logit_error": float(error.max()),
                          "logits_outside_original_tolerance": int((error > allowed).sum()),
                          "fp64_nll_sum": nll, "mean_nll_error": abs(nll - c_stats[1]) / len(y)})
        same_logits_losses = []
        for name, logits in (("same_c_logits_torch_fp32_cross_entropy", c_logits),
                             ("same_c_logits_torch_fp64_cross_entropy", c_logits.double())):
            value = functional.cross_entropy(logits, y, reduction="none").double().sum().item()
            same_logits_losses.append({"name": name, "nll_sum": value,
                                       "mean_error_against_c_double_accumulation": abs(value - c_stats[1]) / len(y)})
    assert cases[1]["logits_outside_original_tolerance"] == 0
    assert cases[1]["mean_nll_error"] <= TOLERANCE["mean_nll_absolute_error"]
    assert same_logits_losses[1]["mean_error_against_c_double_accumulation"] <= TOLERANCE["mean_nll_absolute_error"]
    result = {"kind": "FINAL_CHECKPOINT_REFERENCE_DIAGNOSIS", "at": now(), "targets": len(y),
              "cases": cases, "same_c_logits_loss_diagnosis": same_logits_losses,
              "original_tolerances": TOLERANCE, "tolerances_relaxed": False,
              "checkpoint_modified": False, "trainer_modified": False,
              "initial_failure_preserved": str(output_path(".oracle.json")),
              "aligned_comparison": str(output_path(".oracle-aligned.json")),
              "conclusion": "The initial reference used fused bias additions and FP32 cross_entropy. Match C's separate FP32 GEMM/bias order and independently evaluate NLL in FP64. The original tolerances then pass; no model or training arithmetic was changed.",
              "documentation": "https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html",
              "scope": "The FP32 cross_entropy discrepancy is an observation of this installed CPU runtime and fixture, not a claim about every PyTorch environment."}
    write(".reference-diagnosis.json", result)
    print(json.dumps({"diagnosis_saved": True, "aligned_case": cases[1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
