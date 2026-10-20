/* Isolated batch-normalization experiment, built beside vector_vforce.c.
 * The only arithmetic change is /batch -> *exact_batch_reciprocal for
 * positive power-of-two batches. /sum, row ownership, sum order and loss
 * computation retain the original operations. All other batches delegate
 * directly to the original implementation.
 */
#include "fixedwork_api.h"
#include "fast_exp.h"

#include <math.h>
#if !defined(__APPLE__)
#error "This experiment requires Apple Accelerate CPU vForce."
#endif
#if !EXPERIMENT_VFORCE_EXP
#error "Build both baseline and candidate with EXPERIMENT_VFORCE_EXP=1."
#endif
#ifdef __FAST_MATH__
#error "Global fast-math is outside the fixed-work softmax experiment."
#endif
#include <Accelerate/Accelerate.h>

static float fw_softmax_row(float *row, const float *bias, uint32_t target,
                            int vocab, int fast_exp, float inverse_batch) {
    float maximum = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        row[v] += bias[v];
        if (row[v] > maximum) maximum = row[v];
    }
    const float target_logit = row[target];
    float sum = 0.0f;
    if (fast_exp) {
        for (int v = 0; v < vocab; ++v) {
            row[v] = fast_exp_approx(row[v] - maximum);
            sum += row[v];
        }
    } else {
        for (int v = 0; v < vocab; ++v) row[v] -= maximum;
        vvexpf(row, row, &vocab);
        for (int v = 0; v < vocab; ++v) sum += row[v];
    }
    const float loss = logf(sum) + maximum - target_logit;
    for (int v = 0; v < vocab; ++v)
        row[v] = (row[v] / sum) * inverse_batch;
    row[target] -= inverse_batch;
    return loss;
}

float fw_softmax(float *logits, const float *bias, const uint32_t *targets,
                 int batch, int vocab, const ExperimentKernelOptions *options) {
    if (batch < 1 || vocab < 1 ||
        ((unsigned int)batch & ((unsigned int)batch - 1U)) != 0U) {
        return experiment_softmax(logits, bias, targets, batch, vocab, options);
    }
    const float inverse_batch = 1.0f / (float)batch;
    const int threads = options->threads > 0 ? options->threads : 1;
    float loss_sum = 0.0f;
    if (options->softmax_parallel && threads > 1) {
        float losses[batch];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int b = 0; b < batch; ++b)
            losses[b] = fw_softmax_row(logits + (size_t)b * vocab, bias,
                                       targets[b], vocab, options->fast_exp,
                                       inverse_batch);
        for (int b = 0; b < batch; ++b) loss_sum += losses[b];
    } else {
        for (int b = 0; b < batch; ++b)
            loss_sum += fw_softmax_row(logits + (size_t)b * vocab, bias,
                                       targets[b], vocab, options->fast_exp,
                                       inverse_batch);
    }
    /* Keep the original mean-loss division rather than broadening scope. */
    return loss_sum / (float)batch;
}
