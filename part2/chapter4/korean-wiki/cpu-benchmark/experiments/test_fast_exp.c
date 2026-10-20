#include "fast_exp.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

enum { DENSE_INTERVALS = 2000000 };
static uint32_t rng_state = UINT32_C(0x726e3532);

static uint32_t next_random(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}

static uint32_t float_bits(float x) {
    uint32_t value;
    memcpy(&value, &x, sizeof(value));
    return value;
}

typedef struct {
    double max_relative_error;
    double max_absolute_error;
    float relative_error_input;
    uint32_t max_ulp_from_expf;
    uint64_t count;
} ErrorSummary;

static int check_one(float x, ErrorSummary *summary) {
    const float actual = fast_exp_approx(x);
    const float library = expf(x);
    const double expected = exp((double)x);
    const double absolute_error = fabs((double)actual - expected);
    const double relative_error = absolute_error / expected;
    const uint32_t a = float_bits(actual), b = float_bits(library);
    const uint32_t ulps = a > b ? a - b : b - a;
    if (relative_error > summary->max_relative_error) {
        summary->max_relative_error = relative_error;
        summary->relative_error_input = x;
    }
    if (absolute_error > summary->max_absolute_error) {
        summary->max_absolute_error = absolute_error;
    }
    if (ulps > summary->max_ulp_from_expf) {
        summary->max_ulp_from_expf = ulps;
    }
    ++summary->count;
    return isfinite(actual) && actual > 0.0f && relative_error <= 1e-4;
}

int main(void) {
    ErrorSummary error = {0};
    int passed = 1;
    for (int i = 0; i <= DENSE_INTERVALS; ++i) {
        const float x = (float)(-80.0 + 80.0 * (double)i / DENSE_INTERVALS);
        passed &= check_one(x, &error);
    }

    /* Check both sides and neighbors of range-reduction boundaries. */
    uint64_t boundary_count = 0;
    for (int n = -115; n < 0; ++n) {
        const float center = (float)(((double)n + 0.5) * log(2.0));
        for (int direction = -1; direction <= 1; direction += 2) {
            float x = center;
            for (int neighbor = 0; neighbor <= 4; ++neighbor) {
                if (x >= -80.0f && x <= 0.0f) {
                    passed &= check_one(x, &error);
                    ++boundary_count;
                }
                x = nextafterf(x, direction < 0 ? -INFINITY : INFINITY);
            }
        }
    }
    const float tiny_edges[] = {-0.0f, 0.0f, -FLT_TRUE_MIN, -FLT_MIN,
                                -80.0f, nextafterf(-80.0f, 0.0f)};
    for (size_t i = 0; i < sizeof(tiny_edges) / sizeof(tiny_edges[0]); ++i) {
        passed &= check_one(tiny_edges[i], &error);
    }

    /* Out-of-domain values must preserve the actual libm return value. */
    const float fallback[] = {nextafterf(-80.0f, -INFINITY), -81.0f,
        -87.33655f, -90.0f, -100.0f, -104.0f, -FLT_MAX, -INFINITY,
        nextafterf(0.0f, INFINITY), 0.5f, 1.0f, 80.0f, 100.0f,
        FLT_MAX, INFINITY, NAN};
    int fallback_failures = 0;
    for (size_t i = 0; i < sizeof(fallback) / sizeof(fallback[0]); ++i) {
        const float a = fast_exp_approx(fallback[i]);
        const float b = expf(fallback[i]);
        if (!(isnan(a) && isnan(b)) && float_bits(a) != float_bits(b)) {
            ++fallback_failures;
        }
    }
    passed &= fallback_failures == 0;

    const int vocab_sizes[] = {1, 7, 37, 257, 1025, 35328};
    double max_probability_error = 0.0, max_kl = 0.0;
    double max_float32_probability_error = 0.0, max_ce_error = 0.0;
    uint64_t softmax_rows = 0, softmax_values = 0;
    float *baseline = malloc(35328 * sizeof(float));
    float *approximate = malloc(35328 * sizeof(float));
    if (!baseline || !approximate) {
        free(baseline);
        free(approximate);
        fputs("allocation failure\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t size = 0; size < sizeof(vocab_sizes) / sizeof(vocab_sizes[0]); ++size) {
        const int vocab = vocab_sizes[size];
        for (int row = 0; row < 20; ++row) {
            /* Includes narrow/tied rows, normal rows, and expf fallback tails.
             * index 0 is the maximum so every row is stably shifted. */
            const float width = row % 4 == 0 ? 0.0001f
                : row % 4 == 1 ? 8.0f : row % 4 == 2 ? 80.0f : 150.0f;
            const int target = row % 2 == 0 ? 0 : vocab - 1;
            float target_logit = 0.0f;
            double sum0 = 0.0, sum1 = 0.0;
            float float_sum0 = 0.0f, float_sum1 = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                const float x = v == 0 ? 0.0f
                    : -width * (float)((next_random() >> 8) * (1.0 / 16777216.0));
                if (v == target) target_logit = x;
                baseline[v] = expf(x);
                approximate[v] = fast_exp_approx(x);
                sum0 += baseline[v];
                sum1 += approximate[v];
                float_sum0 += baseline[v];
                float_sum1 += approximate[v];
            }
            double kl = 0.0;
            for (int v = 0; v < vocab; ++v) {
                const double p = baseline[v] / sum0;
                const double q = approximate[v] / sum1;
                const double delta = fabs(p - q);
                const double float_delta = fabs((double)(baseline[v] / float_sum0)
                    - (double)(approximate[v] / float_sum1));
                if (delta > max_probability_error) max_probability_error = delta;
                if (float_delta > max_float32_probability_error) {
                    max_float32_probability_error = float_delta;
                }
                if (p > 0.0 && q > 0.0) kl += p * log(p / q);
            }
            if (kl > max_kl) max_kl = kl;
            /* CE via stable log-sum-exp for targets at both peak and tail. */
            const float ce0 = logf(float_sum0) - target_logit;
            const float ce1 = logf(float_sum1) - target_logit;
            const double ce_delta = fabs((double)ce0 - ce1);
            if (ce_delta > max_ce_error) max_ce_error = ce_delta;
            ++softmax_rows;
            softmax_values += (uint64_t)vocab;
        }
    }
    free(baseline);
    free(approximate);
    passed &= max_probability_error < 1e-6 && max_float32_probability_error < 1e-6
        && max_ce_error < 1e-4 && max_kl < 1e-8;

    printf("{\n"
        "  \"passed\": %s,\n"
        "  \"compiler\": \"%s\",\n"
        "  \"fast_domain\": [-80, 0],\n"
        "  \"relative_error_limit\": 0.0001,\n"
        "  \"dense_samples\": %d,\n"
        "  \"boundary_samples\": %" PRIu64 ",\n"
        "  \"total_in_domain_samples\": %" PRIu64 ",\n"
        "  \"max_relative_error_vs_double_exp\": %.17g,\n"
        "  \"max_relative_error_input\": %.9g,\n"
        "  \"max_absolute_error_vs_double_exp\": %.17g,\n"
        "  \"max_ulp_difference_vs_expf\": %" PRIu32 ",\n"
        "  \"fallback_samples\": %zu,\n"
        "  \"fallback_failures\": %d,\n"
        "  \"softmax_rows\": %" PRIu64 ",\n"
        "  \"softmax_values\": %" PRIu64 ",\n"
        "  \"max_probability_absolute_error_double_normalization\": %.17g,\n"
        "  \"max_probability_absolute_error_float32_normalization\": %.17g,\n"
        "  \"max_cross_entropy_absolute_error_float32_normalization\": %.17g,\n"
        "  \"max_kl_divergence_double_normalization\": %.17g\n"
        "}\n",
        passed ? "true" : "false", __VERSION__, DENSE_INTERVALS + 1,
        boundary_count, error.count, error.max_relative_error,
        error.relative_error_input, error.max_absolute_error, error.max_ulp_from_expf,
        sizeof(fallback) / sizeof(fallback[0]), fallback_failures,
        softmax_rows, softmax_values, max_probability_error,
        max_float32_probability_error, max_ce_error, max_kl);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
