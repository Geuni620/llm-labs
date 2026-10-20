/* Standalone fixed-block numerical check, not a timing or model-quality test.
 * Link the original vector_vforce.c as its own object with
 * EXPERIMENT_VFORCE_EXP=1, together with fixedwork_norm.c.
 * Strict build: -O3 -mcpu=native -ffp-contract=off, no global fast-math.
 */
#include "fixedwork_api.h"

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

/* Fixed before running. For each finite value:
 * abs(actual-reference) <= atol + rtol*abs(reference). */
static const double NORM_ATOL = 1e-12, NORM_RTOL = 2e-7;
static const double SCALE_ATOL = 1e-12, SCALE_RTOL = 2e-7;
static const double GRAD_ATOL = 1e-15, GRAD_RTOL = 4e-7;

enum { GUARD = 8, FAILURE_LIMIT = 8 };
static const float GUARD_VALUE = -1234567.0f;
static const size_t COUNTS[] = {
    0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 255, 256, 257, 1003, 1023, 1024, 1025,
    4095, 4096, 4097, 8191, 8192, 8193, 6883968
};
static const char *PATTERNS[] = {
    "zero", "uniform_random_plusminus1", "mixed_magnitudes",
    "threshold_sparse", "threshold_dense"
};
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
typedef struct { double *allocation, *values; size_t count; } Scratch;
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

static Scratch create_scratch(size_t count) {
    Scratch result = {0};
    result.allocation = calloc(count + 2 * GUARD, sizeof(double));
    if (!result.allocation) { fputs("test_fixedwork_norm: scratch allocation failed\n", stderr); exit(2); }
    result.values = result.allocation + GUARD;
    result.count = count;
    for (int i = 0; i < GUARD; ++i) {
        result.allocation[i] = (double)GUARD_VALUE;
        result.values[count + (size_t)i] = (double)GUARD_VALUE;
    }
    return result;
}

static int scratch_ok(const Scratch *array) {
    for (int i = 0; i < GUARD; ++i)
        if (array->allocation[i] != (double)GUARD_VALUE
            || array->values[array->count + (size_t)i] != (double)GUARD_VALUE) return 0;
    for (size_t i = 0; i < array->count; ++i)
        if (!isfinite(array->values[i]) || array->values[i] < 0.0) return 0;
    return 1;
}

static int scratch_contract_checks(void) {
    ExperimentKernelOptions options = {0};
    options.threads = 4; options.reduction_parallel = 1; options.clip_norm = 1.0f;
    float scale = -1.0f;
    if (fw_norm_scratch_count(0) != 0 || fw_norm_scratch_count(1) != 1
        || fw_norm_scratch_count(4096) != 1 || fw_norm_scratch_count(4097) != 2
        || fw_norm_scratch_count(6883968) != 1681
        || fw_norm_scratch_count(SIZE_MAX) != SIZE_MAX / 4096 + 1) return 0;
    if (fw_clip_gradients(NULL, 0, &scale, &options, NULL, 0) != 0.0f || scale != 1.0f)
        return 0;
    float values[] = {3.0f, 4.0f, -2.0f}, original[3];
    double scratch = 987654.0;
    memcpy(original, values, sizeof(values));
    if (!isnan(fw_clip_gradients(values, 3, &scale, &options, &scratch, 0))
        || !isnan(scale) || scratch != 987654.0 || memcmp(values, original, sizeof(values))) return 0;
    if (!isnan(fw_clip_gradients(values, 3, &scale, &options, NULL, 1))
        || !isnan(scale) || memcmp(values, original, sizeof(values))) return 0;
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

typedef struct {
    Location where;
    int target_group, target_offset;
    float target_norm, original_norm, fixed_norm, compensated_norm;
    float original_scale, fixed_scale, compensated_scale;
} BoundaryRecord;
typedef struct {
    Stats norm, scale, gradient, oracle_norm, oracle_scale, oracle_gradient;
    uint64_t cases, guard_failures, active_cases, pairs, input_positions;
    uint64_t thread_norm_differences, thread_scale_differences;
    uint64_t thread_gradient_differences, thread_partial_differences;
    uint64_t target_placement_failures, original_target_misses, fixed_target_misses;
    uint64_t original_decision_differences, oracle_decision_differences;
    BoundaryRecord records[96];
    size_t record_count;
} BoundaryChecks;

static void merge_stats(Stats *destination, const Stats *source) {
    if (!source->compared) return;
    if (!destination->compared || source->max_absolute_error > destination->max_absolute_error) {
        destination->max_absolute_error = source->max_absolute_error;
        destination->worst_absolute = source->worst_absolute;
    }
    if (!destination->compared || source->max_relative_error_nonzero_reference
                                  > destination->max_relative_error_nonzero_reference) {
        destination->max_relative_error_nonzero_reference = source->max_relative_error_nonzero_reference;
        destination->worst_relative = source->worst_relative;
    }
    if (!destination->compared || source->max_tolerance_fraction > destination->max_tolerance_fraction) {
        destination->max_tolerance_fraction = source->max_tolerance_fraction;
        destination->worst_tolerance = source->worst_tolerance;
    }
    destination->compared += source->compared;
    destination->nonfinite_pairs += source->nonfinite_pairs;
    destination->outside_tolerance += source->outside_tolerance;
    destination->bitwise_differences += source->bitwise_differences;
}

static float scale_for_norm(float norm, float clip) {
    return fminf(1.0f, clip / (norm + 1e-6f));
}

static int boundary_targets(float clip, float *targets) {
    targets[0] = nextafterf(clip, -INFINITY);
    targets[1] = clip;
    targets[2] = nextafterf(clip, INFINITY);
    /* Because the original code adds 1e-6 before division, norm==clip is
     * already clipped. Also deliberately straddle the actual scale<1
     * transition, rather than mistaking clip itself for that boundary. */
    float first_active = clip - 1e-6f;
    int moves = 0;
    while (scale_for_norm(first_active, clip) < 1.0f && moves++ < 32)
        first_active = nextafterf(first_active, -INFINITY);
    while (scale_for_norm(first_active, clip) == 1.0f && moves++ < 64)
        first_active = nextafterf(first_active, INFINITY);
    if (moves >= 64 || scale_for_norm(first_active, clip) >= 1.0f) return 0;
    targets[3] = nextafterf(first_active, -INFINITY);
    targets[4] = first_active;
    targets[5] = nextafterf(first_active, INFINITY);
    return scale_for_norm(targets[3], clip) == 1.0f;
}

static void fill_boundary(float *values, size_t count, float target, int dense) {
    if (!dense) {
        memset(values, 0, count * sizeof(float));
        values[0] = target;
        return;
    }
    /* Dense background contributes about one quarter of the squared norm;
     * tune the first coefficient to place the returned binary32 norm at
     * the requested float. This avoids relying on random near-boundaries. */
    const float background = (float)((double)target / (2.0 * sqrt((double)count)));
    const double remainder = (double)(count - 1) * ((double)background * background);
    float leading = (float)sqrt((double)target * target - remainder);
    for (int attempt = 0; attempt < 32; ++attempt) {
        const float estimated = (float)sqrt(remainder + (double)leading * leading);
        if (estimated == target) break;
        leading = nextafterf(leading, estimated < target ? INFINITY : -INFINITY);
    }
    for (size_t i = 0; i < count; ++i) values[i] = background;
    values[0] = leading;
}

static BoundaryChecks threshold_checks(void) {
    BoundaryChecks result = {0};
    const size_t lengths[] = {4097, 6883968};
    for (int length = 0; length < 2; ++length) {
        const size_t count = lengths[length], blocks = fw_norm_scratch_count(count);
        Guarded input = create_array(count), scalar = create_array(count);
        Guarded fixed = create_array(count), single_worker = create_array(count);
        Scratch scratch = create_scratch(blocks), single_scratch = create_scratch(blocks);
        for (int dense = 0; dense < 2; ++dense)
            for (int c = 0; c < 2; ++c) {
                const float clip = CLIP_NORMS[c];
                float targets[6];
                if (!boundary_targets(clip, targets)) {
                    ++result.target_placement_failures;
                    continue;
                }
                for (int target_index = 0; target_index < 6; ++target_index) {
                    const float target = targets[target_index];
                    fill_boundary(input.values, count, target, dense);
                    result.input_positions += count;
                    const float compensated_norm = oracle_norm(input.values, count);
                    const float compensated_scale = scale_for_norm(compensated_norm, clip);
                    if (compensated_norm != target) ++result.target_placement_failures;
                    float single_norm = 0.0f, single_scale = 0.0f;
                    for (int t = 0; t < 2; ++t) {
                        ExperimentKernelOptions options = {0};
                        options.threads = THREADS[t]; options.reduction_parallel = 1;
                        options.clip_norm = clip;
                        memcpy(scalar.values, input.values, count * sizeof(float));
                        memcpy(fixed.values, input.values, count * sizeof(float));
                        for (size_t i = 0; i < blocks; ++i) scratch.values[i] = NAN;
                        float original_scale, fixed_scale;
                        const float original_norm = experiment_clip_gradients(
                            scalar.values, count, &original_scale, &options);
                        const float fixed_norm = fw_clip_gradients(
                            fixed.values, count, &fixed_scale, &options, scratch.values, blocks);
                        const Location where = {count, 0, dense + 3, THREADS[t], clip};
                        compare(&result.norm, original_norm, fixed_norm, NORM_ATOL, NORM_RTOL, where, "threshold_norm_vs_original");
                        compare(&result.scale, original_scale, fixed_scale, SCALE_ATOL, SCALE_RTOL, where, "threshold_scale_vs_original");
                        compare(&result.oracle_norm, compensated_norm, fixed_norm, NORM_ATOL, NORM_RTOL, where, "threshold_norm_vs_oracle");
                        compare(&result.oracle_scale, compensated_scale, fixed_scale, SCALE_ATOL, SCALE_RTOL, where, "threshold_scale_vs_oracle");
                        for (size_t i = 0; i < count; ++i) {
                            Location gradient_location = where;
                            gradient_location.index = i;
                            compare(&result.gradient, scalar.values[i], fixed.values[i], GRAD_ATOL, GRAD_RTOL, gradient_location, "threshold_gradient_vs_original");
                            const float expected = input.values[i] * compensated_scale;
                            compare(&result.oracle_gradient, expected, fixed.values[i], GRAD_ATOL, GRAD_RTOL, gradient_location, "threshold_gradient_vs_oracle");
                        }
                        if (original_norm != target) ++result.original_target_misses;
                        if (fixed_norm != target) ++result.fixed_target_misses;
                        if ((original_scale < 1.0f) != (fixed_scale < 1.0f))
                            ++result.original_decision_differences;
                        if ((compensated_scale < 1.0f) != (fixed_scale < 1.0f))
                            ++result.oracle_decision_differences;
                        if (t == 0) {
                            single_norm = fixed_norm; single_scale = fixed_scale;
                            memcpy(single_worker.values, fixed.values, count * sizeof(float));
                            memcpy(single_scratch.values, scratch.values, blocks * sizeof(double));
                        } else {
                            ++result.pairs;
                            if (memcmp(&single_norm, &fixed_norm, sizeof(float))) ++result.thread_norm_differences;
                            if (memcmp(&single_scale, &fixed_scale, sizeof(float))) ++result.thread_scale_differences;
                            if (memcmp(single_worker.values, fixed.values, count * sizeof(float))) ++result.thread_gradient_differences;
                            if (memcmp(single_scratch.values, scratch.values, blocks * sizeof(double))) ++result.thread_partial_differences;
                        }
                        if (!guards_ok(&input) || !guards_ok(&scalar) || !guards_ok(&fixed)
                            || !guards_ok(&single_worker) || !scratch_ok(&scratch)
                            || !scratch_ok(&single_scratch)) ++result.guard_failures;
                        if (fixed_scale < 1.0f) ++result.active_cases;
                        result.records[result.record_count++] = (BoundaryRecord){
                            where, target_index / 3, target_index % 3 - 1, target,
                            original_norm, fixed_norm, compensated_norm, original_scale,
                            fixed_scale, compensated_scale};
                        ++result.cases;
                    }
                }
            }
        free(input.allocation); free(scalar.allocation); free(fixed.allocation);
        free(single_worker.allocation); free(scratch.allocation); free(single_scratch.allocation);
    }
    return result;
}

static void threshold_json(const BoundaryChecks *checks) {
    printf("{\"cases\":%" PRIu64 ",\"target_placement_failures\":%" PRIu64
           ",\"original_target_misses\":%" PRIu64 ",\"fixed_target_misses\":%" PRIu64
           ",\"original_clip_decision_differences\":%" PRIu64
           ",\"oracle_clip_decision_differences\":%" PRIu64 ",\"records\":[",
           checks->cases, checks->target_placement_failures, checks->original_target_misses,
           checks->fixed_target_misses, checks->original_decision_differences, checks->oracle_decision_differences);
    for (size_t i = 0; i < checks->record_count; ++i) {
        const BoundaryRecord *record = checks->records + i;
        printf("%s{\"location\":", i ? "," : ""); location_json(record->where);
        printf(",\"target_group\":\"%s\",\"target_offset\":%d,\"target_norm\":%.9g,"
               "\"original_norm\":%.9g,\"fixed_norm\":%.9g,\"compensated_norm\":%.9g,"
               "\"original_scale\":%.9g,\"fixed_scale\":%.9g,\"compensated_scale\":%.9g,"
               "\"original_clipped\":%s,\"fixed_clipped\":%s,\"compensated_clipped\":%s}",
               record->target_group ? "epsilon_adjusted_clip_transition" : "clip_norm_adjacent",
               record->target_offset, record->target_norm, record->original_norm, record->fixed_norm,
               record->compensated_norm, record->original_scale, record->fixed_scale, record->compensated_scale,
               record->original_scale < 1.0f ? "true" : "false", record->fixed_scale < 1.0f ? "true" : "false",
               record->compensated_scale < 1.0f ? "true" : "false");
    }
    fputs("]}", stdout);
}

int main(void) {
    Stats norm = {0}, scale = {0}, gradients = {0};
    Stats oracle_norms = {0}, oracle_scales = {0}, oracle_gradients = {0};
    uint64_t cases = 0, guard_failures = 0, active_clipping_cases = 0;
    uint64_t thread_pairs = 0, thread_norm_differences = 0, thread_scale_differences = 0;
    uint64_t thread_gradient_differences = 0, thread_partial_differences = 0;
    const int scratch_contract_passed = scratch_contract_checks();
    uint64_t scalar_input_positions = 0;
    double input_min = INFINITY, input_max = -INFINITY;
    const size_t count_length = sizeof(COUNTS) / sizeof(COUNTS[0]);
    for (size_t n = 0; n < count_length; ++n) {
        const size_t count = COUNTS[n];
        Guarded input = create_array(count), scalar = create_array(count), simd = create_array(count);
        Guarded single_worker = create_array(count);
        const size_t scratch_count = fw_norm_scratch_count(count);
        Scratch scratch = create_scratch(scratch_count), single_scratch = create_scratch(scratch_count);
        for (int pattern = 0; pattern < 3; ++pattern) {
            fill(input.values, count, pattern);
            scalar_input_positions += count;
            for (size_t i = 0; i < count; ++i) {
                if (input.values[i] < input_min) input_min = input.values[i];
                if (input.values[i] > input_max) input_max = input.values[i];
            }
            const float compensated_norm = oracle_norm(input.values, count);
            for (int c = 0; c < 2; ++c) {
                float single_norm = 0.0f, single_scale = 0.0f;
                for (int t = 0; t < 2; ++t) {
                    ExperimentKernelOptions options = {0};
                    options.threads = THREADS[t]; options.reduction_parallel = 1;
                    options.clip_norm = CLIP_NORMS[c];
                    memcpy(scalar.values, input.values, count * sizeof(float));
                    memcpy(simd.values, input.values, count * sizeof(float));
                    float scalar_scale, simd_scale;
                    const float scalar_norm = experiment_clip_gradients(
                        scalar.values, count, &scalar_scale, &options);
                    for (size_t i = 0; i < scratch_count; ++i) scratch.values[i] = NAN;
                    const float simd_norm = fw_clip_gradients(
                        simd.values, count, &simd_scale, &options, scratch.values, scratch_count);
                    if (t == 0) {
                        single_norm = simd_norm; single_scale = simd_scale;
                        memcpy(single_worker.values, simd.values, count * sizeof(float));
                        memcpy(single_scratch.values, scratch.values, scratch_count * sizeof(double));
                    } else {
                        ++thread_pairs;
                        if (memcmp(&single_norm, &simd_norm, sizeof(float))) ++thread_norm_differences;
                        if (memcmp(&single_scale, &simd_scale, sizeof(float))) ++thread_scale_differences;
                        if (memcmp(single_worker.values, simd.values, count * sizeof(float))) ++thread_gradient_differences;
                        if (memcmp(single_scratch.values, scratch.values, scratch_count * sizeof(double))) ++thread_partial_differences;
                    }
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
                    if (!guards_ok(&input) || !guards_ok(&scalar) || !guards_ok(&simd)
                        || !guards_ok(&single_worker) || !scratch_ok(&scratch)
                        || !scratch_ok(&single_scratch)) ++guard_failures;
                    ++cases;
                }
            }
        }
        free(input.allocation); free(scalar.allocation); free(simd.allocation);
        free(single_worker.allocation); free(scratch.allocation); free(single_scratch.allocation);
    }
    const BoundaryChecks boundaries = threshold_checks();
    merge_stats(&norm, &boundaries.norm); merge_stats(&scale, &boundaries.scale);
    merge_stats(&gradients, &boundaries.gradient); merge_stats(&oracle_norms, &boundaries.oracle_norm);
    merge_stats(&oracle_scales, &boundaries.oracle_scale); merge_stats(&oracle_gradients, &boundaries.oracle_gradient);
    cases += boundaries.cases; guard_failures += boundaries.guard_failures;
    active_clipping_cases += boundaries.active_cases; thread_pairs += boundaries.pairs;
    scalar_input_positions += boundaries.input_positions;
    thread_norm_differences += boundaries.thread_norm_differences;
    thread_scale_differences += boundaries.thread_scale_differences;
    thread_gradient_differences += boundaries.thread_gradient_differences;
    thread_partial_differences += boundaries.thread_partial_differences;
    const int passed = stats_ok(&norm) && stats_ok(&scale) && stats_ok(&gradients)
        && stats_ok(&oracle_norms) && stats_ok(&oracle_scales) && stats_ok(&oracle_gradients)
        && !guard_failures && scratch_contract_passed && !thread_norm_differences
        && !thread_scale_differences && !thread_gradient_differences && !thread_partial_differences
        && !boundaries.target_placement_failures && !boundaries.original_decision_differences
        && !boundaries.oracle_decision_differences;
    printf("{\"schema_version\":1,\"passed\":%s,\"status\":\"%s\",\"cases\":%" PRIu64
           ",\"comparison\":\"original norm vs fixed 4096-element double SIMD block sums and ordered merge\","
           "\"global_fast_math\":false,\"openmp_version\":%d,\"acceptance\":{"
           "\"declared_before_execution\":true,\"rule\":\"abs(actual-reference)<=atol+rtol*abs(reference)\","
           "\"norm\":{\"atol\":%.17g,\"rtol\":%.17g},"
           "\"scale\":{\"atol\":%.17g,\"rtol\":%.17g},"
           "\"gradient\":{\"atol\":%.17g,\"rtol\":%.17g}},\"scope\":{\"counts\":[",
           passed ? "true" : "false", passed ? "PASS" : "FAIL", cases, _OPENMP,
           NORM_ATOL, NORM_RTOL, SCALE_ATOL, SCALE_RTOL, GRAD_ATOL, GRAD_RTOL);
    for (size_t i = 0; i < count_length; ++i) printf("%s%zu", i ? "," : "", COUNTS[i]);
    printf("],\"threads\":[1,4],\"clip_norms\":[1,0.05],"
           "\"patterns\":[\"zero\",\"uniform_random_plusminus1\",\"mixed_magnitudes\",\"threshold_sparse\",\"threshold_dense\"],"
           "\"scalar_input_positions_before_thread_and_clip_repetition\":%" PRIu64
           ",\"input_min\":%.17g,\"input_max\":%.17g},"
           "\"active_clipping_cases\":%" PRIu64 ",\"guard_failures\":%" PRIu64 ",\"norm\":",
           scalar_input_positions, input_min, input_max, active_clipping_cases, guard_failures);
    stats_json(&norm); fputs(",\"scale\":", stdout); stats_json(&scale);
    fputs(",\"gradient\":", stdout); stats_json(&gradients);
    fputs(",\"compensated_oracle\":{\"norm\":", stdout); stats_json(&oracle_norms);
    fputs(",\"scale\":", stdout); stats_json(&oracle_scales);
    fputs(",\"gradient\":", stdout); stats_json(&oracle_gradients);
    fputs("},\"threshold_checks\":", stdout); threshold_json(&boundaries);
    printf(",\"scratch\":{\"logical_block_size\":4096,\"actual_model_partials\":1681,"
           "\"actual_model_bytes\":13448,\"allocated_outside_each_call\":true,"
           "\"contract_checks_passed\":%s},\"thread_stability_same_binary\":{"
           "\"pairs\":%" PRIu64 ",\"norm_different_pairs\":%" PRIu64
           ",\"scale_different_pairs\":%" PRIu64 ",\"gradient_different_pairs\":%" PRIu64
           ",\"partial_sum_different_pairs\":%" PRIu64 "},\"failure_examples\":[",
           scratch_contract_passed ? "true" : "false", thread_pairs, thread_norm_differences,
           thread_scale_differences, thread_gradient_differences, thread_partial_differences);
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
          "\"Thread stability applies only to the tested binary and floating-point environment\","
          "\"Block reassociation need not match original bits or different platforms\","
          "\"No optimizer trajectory, convergence, generation quality or timing validation\"]}\n", stdout);
    return passed ? 0 : 1;
}
