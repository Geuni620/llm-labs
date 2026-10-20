/* Tiny correctness checks only: no training corpus or timing workload. */
#include "experiment_kernels.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { BATCH = 17, VOCAB = 257, PARAMS = 1003 };

static ExperimentKernelOptions options(int threads, int enabled) {
    ExperimentKernelOptions result = {0};
    result.threads = threads;
    result.softmax_parallel = result.reduction_parallel = result.adam_parallel = enabled;
    result.clip_norm = 1.0f;
    return result;
}

static void softmax_checks(void) {
    ExperimentKernelOptions serial = options(1, 0);
    float input[BATCH * VOCAB], actual[BATCH * VOCAB], expected[BATCH * VOCAB];
    float bias[VOCAB];
    uint32_t targets[BATCH];
    for (int v = 0; v < VOCAB; ++v) bias[v] = (float)(v % 7 - 3) * 0.01f;
    for (int b = 0; b < BATCH; ++b) {
        targets[b] = (uint32_t)((b * 17) % VOCAB);
        for (int v = 0; v < VOCAB; ++v)
            input[b * VOCAB + v] = (float)((b * 37 + v * 11) % 113 - 56) * 0.125f;
    }
    /* Include saturation and a large negative target logit. */
    input[0] = 1000.0f;
    input[1] = -1000.0f;
    targets[0] = 1;
    memcpy(expected, input, sizeof(input));
    const float loss = experiment_softmax(expected, bias, targets, BATCH, VOCAB, &serial);
    assert(isfinite(loss) && loss > 0.0f);
    for (int b = 0; b < BATCH; ++b) {
        double sum = 0.0;
        for (int v = 0; v < VOCAB; ++v) {
            assert(isfinite(expected[b * VOCAB + v]));
            sum += expected[b * VOCAB + v];
        }
        assert(fabs(sum) < 1e-7); /* Every mean-CE gradient row sums to zero. */
    }
    for (int threads = 1; threads <= 8; threads *= 2) {
        ExperimentKernelOptions parallel = options(threads, 1);
        memcpy(actual, input, sizeof(input));
        const float other = experiment_softmax(actual, bias, targets, BATCH, VOCAB, &parallel);
        assert(!memcmp(&other, &loss, sizeof(loss)));
        assert(!memcmp(actual, expected, sizeof(expected)));
    }
    float equal_logits[3] = {0}, equal_bias[3] = {0};
    uint32_t target = 1;
    const float uniform_loss = experiment_softmax(equal_logits, equal_bias, &target, 1, 3, &serial);
    assert(fabsf(uniform_loss - logf(3.0f)) < 1e-6f);
    assert(fabsf(equal_logits[0] - 1.0f / 3.0f) < 1e-7f);
    assert(fabsf(equal_logits[1] + 2.0f / 3.0f) < 1e-7f);
}

static void reduction_checks(void) {
    ExperimentKernelOptions serial = options(1, 0);
    float values[BATCH * VOCAB], expected[VOCAB], actual[VOCAB];
    for (int b = 0; b < BATCH; ++b)
        for (int v = 0; v < VOCAB; ++v)
            values[b * VOCAB + v] = (float)((b * 11 + v * 5) % 29 - 14) * 0.03f;
    experiment_bias_reduce(values, BATCH, VOCAB, expected, &serial);
    for (int threads = 1; threads <= 8; threads *= 2) {
        ExperimentKernelOptions parallel = options(threads, 1);
        experiment_bias_reduce(values, BATCH, VOCAB, actual, &parallel);
        assert(!memcmp(actual, expected, sizeof(expected)));
    }
    /* Summation must retain row order even across cancelling magnitudes. */
    float cancellation[3] = {1e20f, 1.0f, -1e20f};
    experiment_bias_reduce(cancellation, 3, 1, actual, &serial);
    assert(actual[0] == 0.0f);
    for (int threads = 1; threads <= 8; threads *= 2) {
        ExperimentKernelOptions parallel = options(threads, 1);
        float gradient[3] = {3.0f, 4.0f, 0.0f}, scale;
        const float norm = experiment_clip_gradients(gradient, 3, &scale, &parallel);
        assert(norm == 5.0f);
        assert(scale == 1.0f / (5.0f + 1e-6f));
        assert(fabsf(gradient[0] - 3.0f * scale) < 1e-7f);
        assert(fabsf(gradient[1] - 4.0f * scale) < 1e-7f);
        parallel.clip_norm = 0.05f;
        float strong[3] = {3.0f, 4.0f, 0.0f};
        experiment_clip_gradients(strong, 3, &scale, &parallel);
        assert(scale == 0.05f / (5.0f + 1e-6f));
        assert(hypot((double)strong[0], strong[1]) <= 0.05000001);
        float zeros[3] = {0};
        assert(experiment_clip_gradients(zeros, 3, &scale, &parallel) == 0.0f);
        assert(scale == 1.0f);
    }
    float original[PARAMS], expected_grad[PARAMS];
    for (int i = 0; i < PARAMS; ++i) original[i] = (float)(i % 101 - 50) * 0.03125f;
    memcpy(expected_grad, original, sizeof(original));
    float serial_scale;
    const float serial_norm = experiment_clip_gradients(expected_grad, PARAMS, &serial_scale, &serial);
    for (int threads = 2; threads <= 8; threads *= 2) {
        ExperimentKernelOptions parallel = options(threads, 1);
        float actual_grad[PARAMS], scale;
        memcpy(actual_grad, original, sizeof(original));
        const float norm = experiment_clip_gradients(actual_grad, PARAMS, &scale, &parallel);
        assert(fabsf(norm - serial_norm) <= 2e-6f * serial_norm);
        assert(fabsf(scale - serial_scale) <= 2e-6f * serial_scale);
        for (int i = 0; i < PARAMS; ++i)
            assert(fabsf(actual_grad[i] - expected_grad[i]) <= 2e-7f);
    }
}

static void adam_checks(void) {
    ExperimentKernelOptions serial = options(1, 0);
    float initial[PARAMS], gradient[PARAMS], expected[PARAMS], first[PARAMS] = {0}, second[PARAMS] = {0};
    for (int i = 0; i < PARAMS; ++i) {
        initial[i] = (float)(i % 67 - 33) * 0.125f;
        gradient[i] = i % 5 ? (float)(i % 31 - 15) * 0.002f : 0.0f;
    }
    initial[0] = 1.0f; /* An unused embedding still receives decoupled decay. */
    memcpy(expected, initial, sizeof(initial));
    experiment_adamw(expected, first, second, gradient, PARAMS, 1, &serial);
    assert(expected[0] == (float)(1.0 - 0.001 * 0.01));
    assert(first[0] == 0.0f && second[0] == 0.0f);
    for (int step = 2; step <= 5; ++step)
        experiment_adamw(expected, first, second, gradient, PARAMS, step, &serial);
    for (int threads = 1; threads <= 8; threads *= 2) {
        ExperimentKernelOptions parallel = options(threads, 1);
        float actual[PARAMS], m1[PARAMS] = {0}, m2[PARAMS] = {0};
        memcpy(actual, initial, sizeof(initial));
        for (int step = 1; step <= 5; ++step)
            experiment_adamw(actual, m1, m2, gradient, PARAMS, step, &parallel);
        assert(!memcmp(actual, expected, sizeof(expected)));
        assert(!memcmp(m1, first, sizeof(first)));
        assert(!memcmp(m2, second, sizeof(second)));
    }
}

int main(void) {
    softmax_checks();
    reduction_checks();
    adam_checks();
#ifdef _OPENMP
    puts("vector kernel checks passed (OpenMP 1/2/4/8 workers)");
#else
    puts("vector kernel checks passed (serial fallback; OpenMP not enabled)");
#endif
    return 0;
}
