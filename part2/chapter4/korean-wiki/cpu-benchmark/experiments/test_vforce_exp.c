/* Wide numerical equivalence check, not a timing or training-quality test.
 *
 * Compile vector_kernels.c and vector_vforce.c as separate translation units.
 * The latter must have EXPERIMENT_VFORCE_EXP=1 plus these export renames:
 *   -Dexperiment_softmax=vforce_experiment_softmax
 *   -Dexperiment_bias_reduce=vforce_experiment_bias_reduce
 *   -Dexperiment_clip_gradients=vforce_experiment_clip_gradients
 *   -Dexperiment_adamw=vforce_experiment_adamw
 * Link both objects with this file, libomp and Apple Accelerate.  Use strict
 * -O3 -ffp-contract=off; never global fast-math.  Each vForce call overwrites
 * its own exact input/output row, exercising the intended in-place API.
 *
 * Acceptance limits below are fixed before executing this experiment:
 *   abs(actual-reference) <= atol + rtol*abs(reference).
 * They allow float32 elementary-function rounding, without promising
 * bit-identical results or equivalent convergence after long training.
 */
#include "experiment_kernels.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__APPLE__)
#error "This verification requires the Apple CPU vForce implementation"
#endif
#if !defined(_OPENMP)
#error "Compile with OpenMP so the four-worker cases exercise parallel rows"
#endif
#if defined(__FAST_MATH__)
#error "Global fast-math is outside the numerical verification contract"
#endif

_Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24,
               "Verification requires binary32 floats");

float vforce_experiment_softmax(float *logits, const float *bias,
                                const uint32_t *targets, int batch, int vocab,
                                const ExperimentKernelOptions *options);

/* Do not loosen these limits after observing results. */
static const double LOSS_ATOL = 1e-5;
static const double LOSS_RTOL = 2e-6;
static const double GRADIENT_ATOL = 2e-7;
static const double GRADIENT_RTOL = 2e-6;

enum { GUARD = 8, MAX_FAILURE_EXAMPLES = 8 };
static const float GUARD_VALUE = -1234567.0f;
static const int VOCABS[] = {
    1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 128, 129, 255, 256, 257, 1023, 1024, 1025,
    16384, 32768, 35328, 65536
};
static const int BATCHES[] = {1, 3};
static const int THREADS[] = {1, 4};
static const char *PATTERNS[] = {
    "uniform_zero", "near_zero", "moderate_random", "wide_minus80_to0",
    "wide_plusminus1000", "positive_10000_offset", "near_equal_maxima",
    "one_dominant_class", "mixed_exp_underflow_tail", "log2_boundaries",
    "finite_plusminus_1e30", "row_translation_with_bias", "subnormal_inputs",
    "normal_subnormal_exp_boundaries"
};

typedef struct {
    int pattern, batch, vocab, threads;
    size_t index;
} Location;

typedef struct {
    double max_absolute_error, max_relative_error_nonzero_reference;
    double max_tolerance_fraction;
    uint64_t compared, outside_tolerance, nonfinite_pairs, bitwise_differences;
    uint64_t zero_reference_differences;
    Location worst_absolute, worst_tolerance;
} ErrorStats;

typedef struct {
    const char *kind;
    Location where;
    double reference, actual, allowed;
} Failure;

typedef struct {
    float *allocation, *values;
    size_t count;
} Guarded;

static Failure failures[MAX_FAILURE_EXAMPLES];
static size_t failure_examples;

static void *checked_calloc(size_t count, size_t bytes) {
    if (bytes && count > SIZE_MAX / bytes) {
        fputs("test_vforce_exp: allocation overflow\n", stderr);
        exit(2);
    }
    void *result = calloc(count, bytes);
    if (!result) {
        fputs("test_vforce_exp: allocation failed\n", stderr);
        exit(2);
    }
    return result;
}

static Guarded guarded_create(size_t count) {
    Guarded result;
    result.allocation = checked_calloc(count + 2 * GUARD, sizeof(float));
    result.values = result.allocation + GUARD;
    result.count = count;
    for (int i = 0; i < GUARD; ++i) {
        result.allocation[i] = GUARD_VALUE;
        result.values[count + (size_t)i] = GUARD_VALUE;
    }
    return result;
}

static int guarded_ok(const Guarded *array) {
    for (int i = 0; i < GUARD; ++i)
        if (array->allocation[i] != GUARD_VALUE
            || array->values[array->count + (size_t)i] != GUARD_VALUE) return 0;
    return 1;
}

static uint32_t random32(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return *state = value;
}

static float uniform(uint32_t *state) {
    return (float)(random32(state) >> 8) * (1.0f / 16777216.0f);
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t count) {
    const unsigned char *bytes = data;
    for (size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void fill_inputs(float *input, float *bias, uint32_t *targets,
                         int batch, int vocab, int pattern) {
    /* Threads intentionally do not enter the seed: both thread budgets
     * receive exactly the same inputs, including all finite stress cases. */
    uint32_t state = UINT32_C(0x6d2b79f5) ^ (uint32_t)vocab
                     ^ ((uint32_t)batch << 20) ^ ((uint32_t)pattern << 24);
    for (int v = 0; v < vocab; ++v) {
        const int add_bias = pattern == 2 || pattern == 3 || pattern == 4
                              || pattern == 5 || pattern == 6 || pattern == 11;
        bias[v] = add_bias ? (uniform(&state) * 2.0f - 1.0f) * 0.25f : 0.0f;
    }
    for (int b = 0; b < batch; ++b) {
        targets[b] = (uint32_t)((b * 17 + pattern * 13) % vocab);
        for (int v = 0; v < vocab; ++v) {
            const float u = uniform(&state);
            float value;
            switch (pattern) {
                case 0: value = 0.0f; break;
                case 1: value = (u * 2.0f - 1.0f) * 1e-6f; break;
                case 2: value = (u * 2.0f - 1.0f) * 8.0f; break;
                case 3: value = -80.0f * u; break;
                case 4: value = (u * 2.0f - 1.0f) * 1000.0f; break;
                case 5: value = 10000.0f + (u * 2.0f - 1.0f) * 8.0f; break;
                case 6: value = (float)(v % 3) * -1e-5f; break;
                case 7: value = v == vocab / 2 ? 1000.0f : -1000.0f; break;
                case 8: value = v == 0 ? 0.0f : -70.0f - 60.0f * u; break;
                case 9:
                    value = -(float)(v % 151) * 0.6931471805599453f;
                    value = v % 2 ? nextafterf(value, -INFINITY)
                                  : nextafterf(value, INFINITY);
                    break;
                case 10: value = v % 2 ? 1e30f : -1e30f; break;
                case 11:
                    value = (u * 2.0f - 1.0f) * 16.0f + (float)b * 1000.0f;
                    break;
                case 12: value = (float)(v % 31 - 15) * 0x1p-149f; break;
                case 13: {
                    const float boundaries[] = {
                        0.0f, -80.0f, -87.33654475f, -87.33655f,
                        -103.2789299f, -103.9720840f, -104.0f, -1000.0f
                    };
                    value = boundaries[v % 8];
                    break;
                }
                default: abort();
            }
            input[(size_t)b * vocab + v] = value;
        }
        if (pattern == 7)
            targets[b] = (uint32_t)(b % 2 ? 0 : vocab / 2);
    }
}

static void compare_number(ErrorStats *stats, double reference, double actual,
                            double atol, double rtol, Location where,
                            const char *kind) {
    ++stats->compared;
    const double allowed = atol + rtol * fabs(reference);
    if (!isfinite(reference) || !isfinite(actual)) {
        ++stats->nonfinite_pairs;
        if (failure_examples < MAX_FAILURE_EXAMPLES)
            failures[failure_examples++] = (Failure){kind, where, reference, actual, allowed};
        return;
    }
    const double absolute = fabs(actual - reference);
    const double fraction = absolute / allowed;
    if (stats->compared == 1 || absolute > stats->max_absolute_error) {
        stats->max_absolute_error = absolute;
        stats->worst_absolute = where;
    }
    if (stats->compared == 1 || fraction > stats->max_tolerance_fraction) {
        stats->max_tolerance_fraction = fraction;
        stats->worst_tolerance = where;
    }
    if (reference != 0.0) {
        const double relative = absolute / fabs(reference);
        if (relative > stats->max_relative_error_nonzero_reference)
            stats->max_relative_error_nonzero_reference = relative;
    } else if (absolute != 0.0) {
        ++stats->zero_reference_differences;
    }
    if (absolute > allowed) {
        ++stats->outside_tolerance;
        if (failure_examples < MAX_FAILURE_EXAMPLES)
            failures[failure_examples++] = (Failure){kind, where, reference, actual, allowed};
    }
    const float reference_float = (float)reference, actual_float = (float)actual;
    if (memcmp(&reference_float, &actual_float, sizeof(float)))
        ++stats->bitwise_differences;
}

static void print_location(Location where) {
    printf("{\"pattern\":\"%s\",\"batch\":%d,\"vocab\":%d,\"threads\":%d,\"index\":%zu}",
           PATTERNS[where.pattern], where.batch, where.vocab, where.threads, where.index);
}

static void print_number(double value) {
    if (isfinite(value)) printf("%.17g", value);
    else fputs("null", stdout);
}

static void print_stats(const ErrorStats *stats) {
    printf("{\"compared\":%" PRIu64 ",\"outside_tolerance\":%" PRIu64
           ",\"nonfinite_pairs\":%" PRIu64 ",\"bitwise_differences\":%" PRIu64
           ",\"zero_reference_differences\":%" PRIu64
           ",\"max_absolute_error\":%.17g,\"max_relative_error_nonzero_reference\":%.17g"
           ",\"max_tolerance_fraction\":%.17g,\"worst_absolute\":",
           stats->compared, stats->outside_tolerance, stats->nonfinite_pairs,
           stats->bitwise_differences, stats->zero_reference_differences,
           stats->max_absolute_error, stats->max_relative_error_nonzero_reference,
           stats->max_tolerance_fraction);
    print_location(stats->worst_absolute);
    fputs(",\"worst_tolerance\":", stdout);
    print_location(stats->worst_tolerance);
    putchar('}');
}

int main(void) {
    ErrorStats losses = {0}, gradients = {0};
    uint64_t cases = 0, guard_failures = 0, immutable_input_failures = 0;
    uint64_t invalid_input_values = 0, unique_input_values = 0, unique_input_rows = 0;
    uint64_t checksum = UINT64_C(14695981039346656037);
    double input_min = INFINITY, input_max = -INFINITY;
    double biased_min = INFINITY, biased_max = -INFINITY;
    const size_t vocab_count = sizeof(VOCABS) / sizeof(VOCABS[0]);
    const size_t batch_count = sizeof(BATCHES) / sizeof(BATCHES[0]);
    const size_t thread_count = sizeof(THREADS) / sizeof(THREADS[0]);
    const size_t pattern_count = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

    for (size_t iv = 0; iv < vocab_count; ++iv)
        for (size_t ib = 0; ib < batch_count; ++ib) {
            const int vocab = VOCABS[iv], batch = BATCHES[ib];
            const size_t count = (size_t)vocab * batch;
            float *input = checked_calloc(count, sizeof(float));
            float *bias = checked_calloc((size_t)vocab, sizeof(float));
            float *bias_copy = checked_calloc((size_t)vocab, sizeof(float));
            uint32_t *targets = checked_calloc((size_t)batch, sizeof(uint32_t));
            uint32_t *targets_copy = checked_calloc((size_t)batch, sizeof(uint32_t));
            Guarded scalar = guarded_create(count), vector = guarded_create(count);
            for (size_t ip = 0; ip < pattern_count; ++ip) {
                fill_inputs(input, bias, targets, batch, vocab, (int)ip);
                memcpy(bias_copy, bias, (size_t)vocab * sizeof(float));
                memcpy(targets_copy, targets, (size_t)batch * sizeof(uint32_t));
                unique_input_values += count;
                unique_input_rows += (uint64_t)batch;
                checksum = hash_bytes(checksum, input, count * sizeof(float));
                checksum = hash_bytes(checksum, bias, (size_t)vocab * sizeof(float));
                checksum = hash_bytes(checksum, targets, (size_t)batch * sizeof(uint32_t));
                for (size_t i = 0; i < count; ++i) {
                    const float biased = input[i] + bias[i % (size_t)vocab];
                    if (!isfinite(input[i]) || !isfinite(biased)) ++invalid_input_values;
                    if (input[i] < input_min) input_min = input[i];
                    if (input[i] > input_max) input_max = input[i];
                    if (biased < biased_min) biased_min = biased;
                    if (biased > biased_max) biased_max = biased;
                }
                for (size_t it = 0; it < thread_count; ++it) {
                    ExperimentKernelOptions options = {0};
                    options.threads = THREADS[it];
                    options.softmax_parallel = THREADS[it] > 1;
                    options.clip_norm = 1.0f;
                    memcpy(scalar.values, input, count * sizeof(float));
                    memcpy(vector.values, input, count * sizeof(float));
                    const float reference_loss = experiment_softmax(
                        scalar.values, bias, targets, batch, vocab, &options);
                    const float vector_loss = vforce_experiment_softmax(
                        vector.values, bias, targets, batch, vocab, &options);
                    Location where = {(int)ip, batch, vocab, THREADS[it], 0};
                    compare_number(&losses, reference_loss, vector_loss,
                                   LOSS_ATOL, LOSS_RTOL, where, "loss");
                    for (size_t i = 0; i < count; ++i) {
                        where.index = i;
                        compare_number(&gradients, scalar.values[i], vector.values[i],
                                       GRADIENT_ATOL, GRADIENT_RTOL, where, "gradient");
                    }
                    if (!guarded_ok(&scalar) || !guarded_ok(&vector)) ++guard_failures;
                    if (memcmp(bias, bias_copy, (size_t)vocab * sizeof(float))
                        || memcmp(targets, targets_copy, (size_t)batch * sizeof(uint32_t)))
                        ++immutable_input_failures;
                    ++cases;
                }
            }
            free(input); free(bias); free(bias_copy); free(targets); free(targets_copy);
            free(scalar.allocation); free(vector.allocation);
        }

    const int passed = !losses.outside_tolerance && !losses.nonfinite_pairs
                        && !gradients.outside_tolerance && !gradients.nonfinite_pairs
                        && !guard_failures && !immutable_input_failures && !invalid_input_values;
    printf("{\"schema_version\":1,\"passed\":%s,\"status\":\"%s\","
           "\"comparison\":\"scalar expf vs Apple CPU vvexpf stable mean CE and all gradients\","
           "\"global_fast_math\":false,\"openmp_version\":%d,\"cases\":%" PRIu64 ","
           "\"acceptance\":{\"rule\":\"abs(actual-reference)<=atol+rtol*abs(reference)\","
           "\"declared_before_execution\":true,\"loss\":{\"atol\":%.17g,\"rtol\":%.17g},"
           "\"gradient\":{\"atol\":%.17g,\"rtol\":%.17g}},\"scope\":{\"vocab_sizes\":[",
           passed ? "true" : "false", passed ? "PASS" : "FAIL", _OPENMP, cases,
           LOSS_ATOL, LOSS_RTOL, GRADIENT_ATOL, GRADIENT_RTOL);
    for (size_t i = 0; i < vocab_count; ++i) printf("%s%d", i ? "," : "", VOCABS[i]);
    fputs("],\"batches\":[1,3],\"threads\":[1,4],\"patterns\":[", stdout);
    for (size_t i = 0; i < pattern_count; ++i) printf("%s\"%s\"", i ? "," : "", PATTERNS[i]);
    printf("],\"input_min\":%.17g,\"input_max\":%.17g,"
           "\"biased_input_min\":%.17g,\"biased_input_max\":%.17g,"
           "\"unique_input_values\":%" PRIu64 ",\"unique_input_rows\":%" PRIu64 ","
           "\"input_checksum_fnv1a64\":\"%016" PRIx64 "\",\"target_seed_base\":\"0x6d2b79f5\"},"
           "\"guard_failures\":%" PRIu64 ",\"immutable_input_failures\":%" PRIu64
           ",\"invalid_input_values\":%" PRIu64 ",\"loss\":",
           input_min, input_max, biased_min, biased_max, unique_input_values,
           unique_input_rows, checksum,
           guard_failures, immutable_input_failures, invalid_input_values);
    print_stats(&losses);
    fputs(",\"gradient\":", stdout);
    print_stats(&gradients);
    fputs(",\"failure_examples\":[", stdout);
    for (size_t i = 0; i < failure_examples; ++i) {
        const Failure *failure = failures + i;
        printf("%s{\"kind\":\"%s\",\"location\":", i ? "," : "", failure->kind);
        print_location(failure->where);
        fputs(",\"reference\":", stdout); print_number(failure->reference);
        fputs(",\"actual\":", stdout); print_number(failure->actual);
        fputs(",\"allowed_absolute_error\":", stdout); print_number(failure->allowed);
        putchar('}');
    }
    fputs("],\"limitations\":[\"Finite-input standalone softmax numerical comparison only\","
          "\"No full model parameter updates or end-to-end optimizer equivalence in this test\","
          "\"No convergence or final training-quality validation\","
          "\"No throughput timing or five-epoch execution\"]}\n", stdout);
    return passed ? 0 : 1;
}
