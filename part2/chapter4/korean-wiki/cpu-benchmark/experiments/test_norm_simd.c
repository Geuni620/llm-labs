/* Standalone numerical check, not a timing or model-quality test.
 * Link the original vector_vforce.c as its own object with
 * EXPERIMENT_VFORCE_EXP=1. Compile vector_norm_simd.c separately with the
 * same flags plus EXPERIMENT_NORM_SIMD=1 and rename all four exported
 * experiment_* functions to normsimd_experiment_* to avoid collisions.
 * Strict build: -O3 -mcpu=native -ffp-contract=off, no global fast-math.
 */
#include "experiment_kernels.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _OPENMP
#error "OpenMP is required to verify actual scalar/parallel SIMD reductions"
#endif
#ifdef __FAST_MATH__
#error "Global fast-math is outside this verification contract"
#endif
_Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24,
               "Require binary32 gradients");

float normsimd_experiment_clip_gradients(float *gradients, size_t count,
                                         float *scale,
                                         const ExperimentKernelOptions *options);

/* Fixed before running. For each finite value:
 * abs(actual-reference) <= atol + rtol*abs(reference). */
static const double NORM_ATOL = 1e-12, NORM_RTOL = 2e-7;
static const double SCALE_ATOL = 1e-12, SCALE_RTOL = 2e-7;
static const double GRAD_ATOL = 1e-15, GRAD_RTOL = 4e-7;

enum { GUARD = 8, FAILURE_LIMIT = 8 };
static const float GUARD_VALUE = -1234567.0f;
static const size_t COUNTS[] = {
    0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 255, 256, 257, 1003, 1023, 1024, 1025, 6883968
};
static const char *PATTERNS[] = {"zero", "uniform_random_plusminus1", "mixed_magnitudes"};
static const int THREADS[] = {1, 4};
static const float CLIP_NORMS[] = {1.0f, 0.05f};

typedef struct { size_t count, index; int pattern, threads; float clip_norm; } Location;
typedef struct {
    uint64_t compared, nonfinite_pairs, outside_tolerance, bitwise_differences;
    double max_absolute_error, max_relative_error_nonzero_reference, max_tolerance_fraction;
    Location worst_absolute, worst_relative, worst_tolerance;
} Stats;
typedef struct { const char *kind; Location where; double reference, actual, allowed; } Failure;
typedef struct { float *allocation, *values; size_t count; } Guarded;
static Failure failures[FAILURE_LIMIT];
static size_t failure_count;

static Guarded create_array(size_t count) {
    Guarded result = {0};
    result.allocation = calloc(count + 2 * GUARD, sizeof(float));
    if (!result.allocation) { fputs("test_norm_simd: allocation failed\n", stderr); exit(2); }
    result.values = result.allocation + GUARD;
    result.count = count;
    for (int i = 0; i < GUARD; ++i) {
        result.allocation[i] = GUARD_VALUE;
        result.values[count + (size_t)i] = GUARD_VALUE;
    }
    return result;
}

static int guards_ok(const Guarded *array) {
    for (int i = 0; i < GUARD; ++i)
        if (array->allocation[i] != GUARD_VALUE
            || array->values[array->count + (size_t)i] != GUARD_VALUE) return 0;
    return 1;
}

static uint32_t random32(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13; value ^= value >> 17; value ^= value << 5;
    return *state = value;
}

static void fill(float *values, size_t count, int pattern) {
    uint32_t state = UINT32_C(0x9e3779b9) ^ (uint32_t)count ^ ((uint32_t)pattern << 24);
    const float mixed[] = {1e12f, -1e12f, 1e-12f, -1e-12f, 1e6f, -1e6f, 0.0f, 1.0f, -1.0f};
    for (size_t i = 0; i < count; ++i) {
        if (pattern == 0) values[i] = 0.0f;
        else if (pattern == 1)
            values[i] = (float)(random32(&state) >> 8) * (2.0f / 16777216.0f) - 1.0f;
        else values[i] = mixed[i % (sizeof(mixed) / sizeof(mixed[0]))];
    }
}

/* Independent compensated double summation. This is not OpenMP-reassociated
 * and does not reuse either implementation's reduction or clipping loops. */
static float oracle_norm(const float *values, size_t count) {
    double sum = 0.0, correction = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double term = (double)values[i] * (double)values[i];
        const double corrected = term - correction;
        const double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    return (float)sqrt(sum);
}

static void compare(Stats *stats, float reference, float actual,
                     double atol, double rtol, Location where, const char *kind) {
    ++stats->compared;
    const double allowed = atol + rtol * fabs((double)reference);
    if (!isfinite(reference) || !isfinite(actual)) {
        ++stats->nonfinite_pairs;
        if (failure_count < FAILURE_LIMIT)
            failures[failure_count++] = (Failure){kind, where, reference, actual, allowed};
        return;
    }
    const double absolute = fabs((double)actual - reference);
    const double relative = reference != 0.0f ? absolute / fabs((double)reference) : 0.0;
    const double fraction = absolute / allowed;
    if (stats->compared == 1 || absolute > stats->max_absolute_error) {
        stats->max_absolute_error = absolute; stats->worst_absolute = where;
    }
    if (stats->compared == 1 || relative > stats->max_relative_error_nonzero_reference) {
        stats->max_relative_error_nonzero_reference = relative; stats->worst_relative = where;
    }
    if (stats->compared == 1 || fraction > stats->max_tolerance_fraction) {
        stats->max_tolerance_fraction = fraction; stats->worst_tolerance = where;
    }
    if (memcmp(&reference, &actual, sizeof(float))) ++stats->bitwise_differences;
    if (absolute > allowed) {
        ++stats->outside_tolerance;
        if (failure_count < FAILURE_LIMIT)
            failures[failure_count++] = (Failure){kind, where, reference, actual, allowed};
    }
}

static int stats_ok(const Stats *stats) {
    return !stats->nonfinite_pairs && !stats->outside_tolerance;
}

static void location_json(Location where) {
    printf("{\"count\":%zu,\"pattern\":\"%s\",\"threads\":%d,\"clip_norm\":%.9g,\"index\":%zu}",
           where.count, PATTERNS[where.pattern], where.threads, where.clip_norm, where.index);
}

static void number_json(double number) {
    if (isfinite(number)) printf("%.17g", number); else fputs("null", stdout);
}

static void stats_json(const Stats *stats) {
    printf("{\"compared\":%" PRIu64 ",\"nonfinite_pairs\":%" PRIu64
           ",\"outside_tolerance\":%" PRIu64 ",\"bitwise_differences\":%" PRIu64
           ",\"max_absolute_error\":%.17g,\"max_relative_error_nonzero_reference\":%.17g,"
           "\"max_tolerance_fraction\":%.17g,\"worst_absolute\":",
           stats->compared, stats->nonfinite_pairs, stats->outside_tolerance,
           stats->bitwise_differences, stats->max_absolute_error,
           stats->max_relative_error_nonzero_reference, stats->max_tolerance_fraction);
    location_json(stats->worst_absolute);
    fputs(",\"worst_relative\":", stdout); location_json(stats->worst_relative);
    fputs(",\"worst_tolerance\":", stdout); location_json(stats->worst_tolerance);
    putchar('}');
}

int main(void) {
    Stats norm = {0}, scale = {0}, gradients = {0};
    Stats oracle_norms = {0}, oracle_scales = {0}, oracle_gradients = {0};
    uint64_t cases = 0, guard_failures = 0, active_clipping_cases = 0;
    uint64_t scalar_input_positions = 0;
    double input_min = INFINITY, input_max = -INFINITY;
    const size_t count_length = sizeof(COUNTS) / sizeof(COUNTS[0]);
    for (size_t n = 0; n < count_length; ++n) {
        const size_t count = COUNTS[n];
        Guarded input = create_array(count), scalar = create_array(count), simd = create_array(count);
        for (int pattern = 0; pattern < 3; ++pattern) {
            fill(input.values, count, pattern);
            scalar_input_positions += count;
            for (size_t i = 0; i < count; ++i) {
                if (input.values[i] < input_min) input_min = input.values[i];
                if (input.values[i] > input_max) input_max = input.values[i];
            }
            const float compensated_norm = oracle_norm(input.values, count);
            for (int t = 0; t < 2; ++t)
                for (int c = 0; c < 2; ++c) {
                    ExperimentKernelOptions options = {0};
                    options.threads = THREADS[t]; options.reduction_parallel = 1;
                    options.clip_norm = CLIP_NORMS[c];
                    memcpy(scalar.values, input.values, count * sizeof(float));
                    memcpy(simd.values, input.values, count * sizeof(float));
                    float scalar_scale, simd_scale;
                    const float scalar_norm = experiment_clip_gradients(
                        scalar.values, count, &scalar_scale, &options);
                    const float simd_norm = normsimd_experiment_clip_gradients(
                        simd.values, count, &simd_scale, &options);
                    const float compensated_scale = fminf(
                        1.0f, options.clip_norm / (compensated_norm + 1e-6f));
                    Location where = {count, 0, pattern, THREADS[t], CLIP_NORMS[c]};
                    compare(&norm, scalar_norm, simd_norm, NORM_ATOL, NORM_RTOL, where, "norm_vs_original");
                    compare(&scale, scalar_scale, simd_scale, SCALE_ATOL, SCALE_RTOL, where, "scale_vs_original");
                    compare(&oracle_norms, compensated_norm, simd_norm, NORM_ATOL, NORM_RTOL, where, "norm_vs_compensated_oracle");
                    compare(&oracle_scales, compensated_scale, simd_scale, SCALE_ATOL, SCALE_RTOL, where, "scale_vs_compensated_oracle");
                    for (size_t i = 0; i < count; ++i) {
                        where.index = i;
                        const float oracle_gradient = input.values[i] * compensated_scale;
                        compare(&gradients, scalar.values[i], simd.values[i], GRAD_ATOL, GRAD_RTOL, where, "gradient_vs_original");
                        compare(&oracle_gradients, oracle_gradient, simd.values[i], GRAD_ATOL, GRAD_RTOL, where, "gradient_vs_compensated_oracle");
                    }
                    if (simd_scale < 1.0f) ++active_clipping_cases;
                    if (!guards_ok(&input) || !guards_ok(&scalar) || !guards_ok(&simd)) ++guard_failures;
                    ++cases;
                }
        }
        free(input.allocation); free(scalar.allocation); free(simd.allocation);
    }
    const int passed = stats_ok(&norm) && stats_ok(&scale) && stats_ok(&gradients)
        && stats_ok(&oracle_norms) && stats_ok(&oracle_scales) && stats_ok(&oracle_gradients)
        && !guard_failures;
    printf("{\"schema_version\":1,\"passed\":%s,\"status\":\"%s\",\"cases\":%" PRIu64
           ",\"comparison\":\"original double gradient norm vs OpenMP SIMD double reduction\","
           "\"global_fast_math\":false,\"openmp_version\":%d,\"acceptance\":{"
           "\"declared_before_execution\":true,\"rule\":\"abs(actual-reference)<=atol+rtol*abs(reference)\","
           "\"norm\":{\"atol\":%.17g,\"rtol\":%.17g},"
           "\"scale\":{\"atol\":%.17g,\"rtol\":%.17g},"
           "\"gradient\":{\"atol\":%.17g,\"rtol\":%.17g}},\"scope\":{\"counts\":[",
           passed ? "true" : "false", passed ? "PASS" : "FAIL", cases, _OPENMP,
           NORM_ATOL, NORM_RTOL, SCALE_ATOL, SCALE_RTOL, GRAD_ATOL, GRAD_RTOL);
    for (size_t i = 0; i < count_length; ++i) printf("%s%zu", i ? "," : "", COUNTS[i]);
    printf("],\"threads\":[1,4],\"clip_norms\":[1,0.05],"
           "\"patterns\":[\"zero\",\"uniform_random_plusminus1\",\"mixed_magnitudes\"],"
           "\"scalar_input_positions_before_thread_and_clip_repetition\":%" PRIu64
           ",\"input_min\":%.17g,\"input_max\":%.17g},"
           "\"active_clipping_cases\":%" PRIu64 ",\"guard_failures\":%" PRIu64 ",\"norm\":",
           scalar_input_positions, input_min, input_max, active_clipping_cases, guard_failures);
    stats_json(&norm); fputs(",\"scale\":", stdout); stats_json(&scale);
    fputs(",\"gradient\":", stdout); stats_json(&gradients);
    fputs(",\"compensated_oracle\":{\"norm\":", stdout); stats_json(&oracle_norms);
    fputs(",\"scale\":", stdout); stats_json(&oracle_scales);
    fputs(",\"gradient\":", stdout); stats_json(&oracle_gradients);
    fputs("},\"failure_examples\":[", stdout);
    for (size_t i = 0; i < failure_count; ++i) {
        const Failure *failure = failures + i;
        printf("%s{\"kind\":\"%s\",\"location\":", i ? "," : "", failure->kind);
        location_json(failure->where);
        fputs(",\"reference\":", stdout); number_json(failure->reference);
        fputs(",\"actual\":", stdout); number_json(failure->actual);
        fputs(",\"allowed_absolute_error\":", stdout); number_json(failure->allowed);
        putchar('}');
    }
    fputs("],\"limitations\":[\"Standalone gradient norm and clipping numerical comparison only\","
          "\"SIMD reduction reassociates double sums; bitwise equivalence is not promised\","
          "\"No optimizer trajectory, convergence, generation quality or timing validation\"]}\n", stdout);
    return passed ? 0 : 1;
}
