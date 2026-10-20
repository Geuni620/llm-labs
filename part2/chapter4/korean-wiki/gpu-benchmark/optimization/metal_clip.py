"""Experimental FP32 MPS global-norm clip; no Torch work during import.

Only the caller under the benchmark lock may compile/run this candidate.
The three kernels share the current MPS stream. Shape arguments are host
metadata; no GPU scalar is read by Python. This changes reduction grouping,
so the independent numerical gate remains mandatory.
"""

from __future__ import annotations

import hashlib
import os
import time


THREADS = 256
ELEMENTS_PER_THREAD = 4
BLOCK_ELEMENTS = THREADS * ELEMENTS_PER_THREAD

SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float read_grad(uint i, device const float* g0, device const float* g1,
                       device const float* g2, device const float* g3,
                       device const float* g4, constant long* ends) {
    if (i < ends[0]) return g0[i];
    if (i < ends[1]) return g1[i - ends[0]];
    if (i < ends[2]) return g2[i - ends[1]];
    if (i < ends[3]) return g3[i - ends[2]];
    return g4[i - ends[3]];
}

kernel void clip_partials(device const float* g0, device const float* g1,
                          device const float* g2, device const float* g3,
                          device const float* g4, device float* partials,
                          constant long* ends,
                          uint lane [[thread_index_in_threadgroup]],
                          uint block [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    float value = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        const uint i = block * 1024 + lane + j * 256;
        if (i < ends[4]) {
            const float x = read_grad(i, g0, g1, g2, g3, g4, ends);
            value += x * x;
        }
    }
    sums[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partials[block] = sums[0];
}

kernel void clip_norm_scale(device const float* partials, device float* stats,
                            constant long& block_count, constant float& max_norm,
                            uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    float value = 0.0f;
    for (long block = lane; block < block_count; block += 256) value += partials[block];
    sums[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float norm = sqrt(sums[0]);
        const float ratio = max_norm / (norm + 1e-6f);
        stats[0] = norm;
        // Metal min may select a numeric operand for NaN; Tensor.clamp
        // propagates NaN. Preserve that behavior explicitly.
        stats[1] = isnan(ratio) ? ratio : min(1.0f, ratio);
    }
}

kernel void clip_write(device float* g0, device float* g1, device float* g2,
                       device float* g3, device float* g4,
                       device const float* stats, constant long* ends,
                       uint lane [[thread_index_in_threadgroup]],
                       uint block [[threadgroup_position_in_grid]]) {
    const float scale = stats[1];
    for (uint j = 0; j < 4; ++j) {
        const uint i = block * 1024 + lane + j * 256;
        if (i < ends[4]) {
            if (i < ends[0]) g0[i] *= scale;
            else if (i < ends[1]) g1[i - ends[0]] *= scale;
            else if (i < ends[2]) g2[i - ends[1]] *= scale;
            else if (i < ends[3]) g3[i - ends[2]] *= scale;
            else g4[i - ends[3]] *= scale;
        }
    }
}
"""

SHADER_SHA256 = hashlib.sha256(SHADER_SOURCE.encode()).hexdigest()


def clip_gradients(bundle: dict, max_norm: float):
    """Return fresh 0-D norm/scale views after clipping the current five grads.

    Cache only shader handles, partial scratch and static shape metadata.
    Gradient storage can change after zero_grad(set_to_none=True), so current
    parameter.grad tensors are rebound on every call. The fresh two-float
    result buffer keeps deferred history independent between updates.
    """
    if (os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or "PYTORCH_MPS_PREFER_METAL" in os.environ):
        raise RuntimeError("The Metal clip case requires FAST_MATH=0, fallback=0 and PREFER_METAL unset")
    if not isinstance(max_norm, (int, float)):
        raise TypeError("max_norm must be a host Python number; GPU scalar readback is forbidden")
    max_norm = float(max_norm)

    import torch

    parameters = bundle["parameters"]
    if len(parameters) != 5:
        raise ValueError("Metal clip supports exactly five dense NPLM gradients")
    gradients = tuple(parameter.grad for parameter in parameters)
    device = parameters[0].device
    if device.type != "mps" or any(
        gradient is None or gradient.device != device or gradient.dtype != torch.float32
        or gradient.layout != torch.strided or not gradient.is_contiguous()
        for gradient in gradients
    ):
        raise ValueError("Metal clip requires five present contiguous float32 MPS gradients")
    lengths = tuple(gradient.numel() for gradient in gradients)
    total = sum(lengths)
    if total > 0xFFFFFFFF:
        raise ValueError("Metal clip logical indices must fit uint32")
    blocks = max(1, (total + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS)
    state = bundle.get("_metal_clip_state")
    if state is None:
        started = time.perf_counter()
        library = torch.mps.compile_shader(SHADER_SOURCE)
        kernels = (library.clip_partials, library.clip_norm_scale, library.clip_write)
        if any(kernel.max_threads_per_threadgroup < THREADS for kernel in kernels):
            raise RuntimeError("Metal clip kernels require 256 threads per threadgroup")
        ends, end = [], 0
        for length in lengths:
            end += length
            ends.append(end)
        state = {"library": library, "kernels": kernels, "lengths": lengths,
                 "device": device, "ends": tuple(ends), "blocks": blocks,
                 "partials": torch.empty(blocks, dtype=torch.float32, device=device)}
        bundle["_metal_clip_state"] = state
        bundle.setdefault("optimization_metadata", {})["metal_clip"] = {
            "implementation": "three native Metal kernels; float32 block sum of squares",
            "shader_sha256": SHADER_SHA256,
            "threads_per_group": THREADS,
            "elements_per_thread": ELEMENTS_PER_THREAD,
            "logical_block_elements": BLOCK_ELEMENTS,
            "gradient_numels": list(lengths), "cumulative_ends": list(ends),
            "total_gradient_elements": total, "partial_count": blocks,
            "partial_bytes": blocks * 4,
            "dispatch_threads": {"partials": blocks * THREADS, "norm_scale": THREADS,
                                 "write": blocks * THREADS},
            "shape_binding": "host tensor metadata; Python integer tuple binds constant int64 ends",
            "partial_write_contract": "lane 0 of every full group overwrites its partial, including an empty logical block",
            "partial_overwrite_runtime_verified": False,
            "scratch_reused_without_zeroing": True,
            "gradient_storage_rebound_every_call": True,
            "fresh_two_float_stats_buffer_each_call": True,
            "returned_scalars": "separate 0-D views of that call's fresh result buffer",
            "all_threads_participate_in_all_reduction_barriers": True,
            "nan_scale_propagation": "explicit isnan guard before min",
            "epsilon": 1e-6, "fast_math_env": "0", "host_readback": False,
            "setup_host_seconds": time.perf_counter() - started,
            "numerical_support_verified": False, "performance_verified": False,
        }
    elif state["lengths"] != lengths or state["device"] != device:
        raise ValueError("Changing Metal clip gradient shapes/devices after setup is unsupported")

    # Every dispatch uses full 256-thread groups; no barrier participant exits
    # early. Current-stream ordering protects scratch reuse between updates.
    with torch.no_grad():
        stats = torch.empty(2, dtype=torch.float32, device=device)
        partials_kernel, norm_kernel, write_kernel = state["kernels"]
        partials_kernel(*gradients, state["partials"], state["ends"],
                        threads=blocks * THREADS, group_size=THREADS)
        norm_kernel(state["partials"], stats, blocks, max_norm,
                    threads=THREADS, group_size=THREADS)
        write_kernel(*gradients, stats, state["ends"],
                     threads=blocks * THREADS, group_size=THREADS)
        return stats[0], stats[1]
