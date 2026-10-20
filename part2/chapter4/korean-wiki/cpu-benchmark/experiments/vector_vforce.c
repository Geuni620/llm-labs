/* Independent OpenMP loop experiments for the RAM-resident Wiki benchmark.
 *
 * Parallel softmax workers own whole rows.  Bias reductions own output
 * columns; norm reduction owns a private double accumulator per worker.
 * AdamW/clipping workers own parameter elements.  None of these functions
 * calls BLAS or starts a nested parallel region.
 *
 * Serial branches deliberately retain the original float32 arithmetic and
 * loop order.  Parallel norm accumulation can round differently because
 * OpenMP combines partial double sums in a different order.  Softmax losses
 * are combined in their original row order even when rows run in parallel.
 */
#include "experiment_kernels.h"
#include "fast_exp.h"

#include <math.h>
#include <string.h>
#if EXPERIMENT_VFORCE_EXP
#if !defined(__APPLE__)
#error "The vForce experiment requires Apple Accelerate on CPU"
#endif
#include <Accelerate/Accelerate.h>
#endif

static int worker_count(const ExperimentKernelOptions *options) {
    return options->threads > 0 ? options->threads : 1;
}

static float softmax_row(float *row, const float *bias, uint32_t target,
                         int batch, int vocab, int fast_exp) {
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
#if EXPERIMENT_VFORCE_EXP
        /* Public CPU vector API. vForce explicitly permits exact in-place
         * operands. Each OpenMP worker owns its row; scalar sum order stays
         * unchanged. This path does not enable global fast-math. */
        for (int v = 0; v < vocab; ++v) row[v] -= maximum;
        vvexpf(row, row, &vocab);
        for (int v = 0; v < vocab; ++v) sum += row[v];
#else
        for (int v = 0; v < vocab; ++v) {
            row[v] = expf(row[v] - maximum);
            sum += row[v];
        }
#endif
    }
    const float loss = logf(sum) + maximum - target_logit;
    for (int v = 0; v < vocab; ++v)
        row[v] = (row[v] / sum) / (float)batch;
    row[target] -= 1.0f / (float)batch;
    return loss;
}

float experiment_softmax(float *logits, const float *bias,
                         const uint32_t *targets, int batch, int vocab,
                         const ExperimentKernelOptions *options) {
    if (batch < 1 || vocab < 1) return NAN;
    const int threads = worker_count(options);
    float loss_sum = 0.0f;
    if (options->softmax_parallel && threads > 1) {
        /* Engine bounds B to at most 1024: this scratch is at most 4 KiB.
         * Each worker writes only losses[b] and logits[b,:]. */
        float losses[batch];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int b = 0; b < batch; ++b)
            losses[b] = softmax_row(logits + (size_t)b * vocab, bias, targets[b],
                                    batch, vocab, options->fast_exp);
        for (int b = 0; b < batch; ++b) loss_sum += losses[b];
    } else {
        for (int b = 0; b < batch; ++b)
            loss_sum += softmax_row(logits + (size_t)b * vocab, bias, targets[b],
                                    batch, vocab, options->fast_exp);
    }
    return loss_sum / (float)batch;
}

void experiment_bias_reduce(const float *values, int rows, int cols, float *out,
                             const ExperimentKernelOptions *options) {
    if (cols < 1) return;
    const int threads = worker_count(options);
    memset(out, 0, (size_t)cols * sizeof(*out));
    if (options->reduction_parallel && threads > 1) {
        /* Each 64-column block has one owner.  Keeping columns contiguous
         * permits vectorization and cache-line reuse, while each column's
         * row summation order remains the same as the serial baseline. */
        const int blocks = (cols - 1) / 64 + 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int block = 0; block < blocks; ++block) {
            const int start = block * 64;
            const int end = start + 64 < cols ? start + 64 : cols;
            for (int row = 0; row < rows; ++row)
                for (int column = start; column < end; ++column)
                    out[column] += values[(size_t)row * cols + column];
        }
    } else {
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < cols; ++column)
                out[column] += values[(size_t)row * cols + column];
    }
}

float experiment_clip_gradients(float *gradients, size_t count,
                                float *clipping_scale,
                                const ExperimentKernelOptions *options) {
    const int threads = worker_count(options);
    double squares = 0.0;
    if (options->reduction_parallel && threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:squares) schedule(static) num_threads(threads)
#endif
        for (size_t i = 0; i < count; ++i)
            squares += (double)gradients[i] * gradients[i];
    } else {
        for (size_t i = 0; i < count; ++i)
            squares += (double)gradients[i] * gradients[i];
    }
    const float norm = (float)sqrt(squares);
    const float scale = fminf(1.0f, options->clip_norm / (norm + 1e-6f));
    *clipping_scale = scale;
    if (scale < 1.0f) {
        if (options->reduction_parallel && threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (size_t i = 0; i < count; ++i) gradients[i] *= scale;
        } else {
            for (size_t i = 0; i < count; ++i) gradients[i] *= scale;
        }
    }
    return norm;
}

void experiment_adamw(float *parameters, float *moment1, float *moment2,
                      const float *gradients, size_t count, int step,
                      const ExperimentKernelOptions *options) {
    const int threads = worker_count(options);
    const float correction2_root = (float)sqrt(1.0 - pow(0.999, (double)step));
    const float step_size = (float)(0.001 / (1.0 - pow(0.9, (double)step)));
    const float decay = (float)(1.0 - 0.001 * 0.01);
    if (options->adam_parallel && threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (size_t i = 0; i < count; ++i) {
            const float gradient = gradients[i];
            parameters[i] *= decay;
            moment1[i] += 0.1f * (gradient - moment1[i]);
            moment2[i] = 0.999f * moment2[i] + 0.001f * (gradient * gradient);
            const float denominator = sqrtf(moment2[i]) / correction2_root + 1e-8f;
            parameters[i] -= step_size * moment1[i] / denominator;
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            const float gradient = gradients[i];
            parameters[i] *= decay;
            moment1[i] += 0.1f * (gradient - moment1[i]);
            moment2[i] = 0.999f * moment2[i] + 0.001f * (gradient * gradient);
            const float denominator = sqrtf(moment2[i]) / correction2_root + 1e-8f;
            parameters[i] -= step_size * moment1[i] / denominator;
        }
    }
}
