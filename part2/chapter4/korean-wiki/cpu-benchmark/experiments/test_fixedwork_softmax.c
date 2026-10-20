#include "fixedwork_api.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "This numerical comparison must not use global fast-math."
#endif

enum { GUARD = 8, PATTERNS = 14 };
static const uint32_t float_canary = UINT32_C(0x4a123456);
static const uint32_t int_canary = UINT32_C(0xbad0beef);
static uint32_t random_state = UINT32_C(0x4a63b019);
static uint64_t case_count, pow2_cases, fallback_cases, fast_exp_cases;
static uint64_t gradient_count, gradient_bit_differences, loss_bit_differences;
static uint64_t nonfinite_pairs, guard_failures, immutable_input_failures;
static uint64_t gradient_tolerance_failures, loss_tolerance_failures;
static double max_gradient_absolute_error, max_loss_absolute_error;
static uint64_t source_positions;
static float input_min = INFINITY, input_max = -INFINITY;

static uint32_t bits(float x) {
    uint32_t out;
    memcpy(&out, &x, sizeof(out));
    return out;
}

static uint32_t random_u32(void) {
    uint32_t x = random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return random_state = x;
}

static float unit_random(void) {
    return (float)((random_u32() >> 8) * (1.0 / 16777216.0));
}

static void set_float_guards(float *values, size_t count) {
    for (size_t i = 0; i < GUARD; ++i) {
        memcpy(values + i, &float_canary, sizeof(float_canary));
        memcpy(values + GUARD + count + i, &float_canary, sizeof(float_canary));
    }
}

static int float_guards_ok(const float *values, size_t count) {
    for (size_t i = 0; i < GUARD; ++i) {
        if (bits(values[i]) != float_canary ||
            bits(values[GUARD + count + i]) != float_canary) return 0;
    }
    return 1;
}

static int int_guards_ok(const uint32_t *values, size_t count) {
    for (size_t i = 0; i < GUARD; ++i) {
        if (values[i] != int_canary ||
            values[GUARD + count + i] != int_canary) return 0;
    }
    return 1;
}

static float input_value(int pattern, int row, int column, int vocab) {
    const float u = unit_random();
    switch (pattern) {
        case 0: return 0.0f;
        case 1: return ((float)(column % 7) - 3.0f) * 1e-7f;
        case 2: return (u * 2.0f - 1.0f) * 8.0f;
        case 3: return -80.0f * u;
        case 4: return (u * 2.0f - 1.0f) * 1000.0f;
        case 5: return 10000.0f + (u * 2.0f - 1.0f) * 4.0f;
        case 6: return column % 3 == 0 ? 1.0f : nextafterf(1.0f, 0.0f);
        case 7: return column == row % vocab ? 0.0f : -80.0f;
        case 8: {
            static const float tail[] = {0.0f, -80.0f, -87.3f, -90.0f,
                                          -100.0f, -103.0f, -104.0f, -150.0f};
            return tail[column % 8];
        }
        case 9: return (float)(-((double)(column % 115) + 0.5) * log(2.0));
        case 10: return column % 2 == 0 ? 1e30f : -1e30f;
        case 11: return (float)row * 128.0f + (u * 2.0f - 1.0f) * 4.0f;
        case 12: return ((float)(column % 7) - 3.0f) * FLT_TRUE_MIN;
        default: {
            static const float boundary[] = {0.0f, -87.33654f, -87.33655f,
                -87.33656f, -103.27892f, -103.27893f, -103.97208f, -104.0f};
            return boundary[column % 8];
        }
    }
}

static void compare_values(float expected, float actual, int is_loss) {
    const double difference = fabs((double)actual - expected);
    const double tolerance = is_loss ? 1e-5 + 2e-6 * fabs((double)expected)
                                    : 2e-7 + 2e-6 * fabs((double)expected);
    if (!isfinite(expected) || !isfinite(actual)) ++nonfinite_pairs;
    if (is_loss) {
        if (difference > max_loss_absolute_error) max_loss_absolute_error = difference;
        if (bits(actual) != bits(expected)) ++loss_bit_differences;
        if (!(difference <= tolerance)) ++loss_tolerance_failures;
    } else {
        if (difference > max_gradient_absolute_error) max_gradient_absolute_error = difference;
        if (bits(actual) != bits(expected)) ++gradient_bit_differences;
        if (!(difference <= tolerance)) ++gradient_tolerance_failures;
        ++gradient_count;
    }
}

static void run_case(int batch, int vocab, int pattern, int threads, int fast_exp,
                     int parallel_enabled) {
    const size_t count = (size_t)batch * vocab;
    float *reference_guarded = malloc((count + 2 * GUARD) * sizeof(float));
    float *candidate_guarded = malloc((count + 2 * GUARD) * sizeof(float));
    float *bias_guarded = malloc(((size_t)vocab + 2 * GUARD) * sizeof(float));
    float *bias_original = malloc((size_t)vocab * sizeof(float));
    uint32_t *target_guarded = malloc(((size_t)batch + 2 * GUARD) * sizeof(uint32_t));
    uint32_t *target_original = malloc((size_t)batch * sizeof(uint32_t));
    if (!reference_guarded || !candidate_guarded || !bias_guarded ||
        !bias_original || !target_guarded || !target_original) {
        fputs("allocation failure\n", stderr);
        exit(EXIT_FAILURE);
    }
    set_float_guards(reference_guarded, count);
    set_float_guards(candidate_guarded, count);
    set_float_guards(bias_guarded, (size_t)vocab);
    for (size_t i = 0; i < GUARD; ++i) {
        target_guarded[i] = int_canary;
        target_guarded[GUARD + batch + i] = int_canary;
    }
    float *reference = reference_guarded + GUARD;
    float *candidate = candidate_guarded + GUARD;
    float *bias = bias_guarded + GUARD;
    uint32_t *target = target_guarded + GUARD;
    for (int v = 0; v < vocab; ++v) {
        bias[v] = pattern == 11 ? ((float)(v % 11) - 5.0f) * 0.125f
            : pattern == 2 ? (unit_random() - 0.5f) * 0.1f : 0.0f;
    }
    memcpy(bias_original, bias, (size_t)vocab * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        target[b] = (uint32_t)((b * 71 + pattern * 17) % vocab);
        for (int v = 0; v < vocab; ++v) {
            const float value = input_value(pattern, b, v, vocab);
            reference[(size_t)b * vocab + v] = value;
            candidate[(size_t)b * vocab + v] = value;
            if (value < input_min) input_min = value;
            if (value > input_max) input_max = value;
        }
    }
    memcpy(target_original, target, (size_t)batch * sizeof(uint32_t));
    ExperimentKernelOptions options = {0};
    options.threads = threads;
    options.softmax_parallel = parallel_enabled;
    options.fast_exp = fast_exp;
    const float reference_loss = experiment_softmax(reference, bias, target,
                                                    batch, vocab, &options);
    if (memcmp(bias_original, bias, (size_t)vocab * sizeof(float)) != 0 ||
        memcmp(target_original, target, (size_t)batch * sizeof(uint32_t)) != 0) {
        ++immutable_input_failures;
    }
    const float candidate_loss = fw_softmax(candidate, bias, target,
                                           batch, vocab, &options);
    if (memcmp(bias_original, bias, (size_t)vocab * sizeof(float)) != 0 ||
        memcmp(target_original, target, (size_t)batch * sizeof(uint32_t)) != 0) {
        ++immutable_input_failures;
    }
    compare_values(reference_loss, candidate_loss, 1);
    for (size_t i = 0; i < count; ++i)
        compare_values(reference[i], candidate[i], 0);
    if (!float_guards_ok(reference_guarded, count) ||
        !float_guards_ok(candidate_guarded, count) ||
        !float_guards_ok(bias_guarded, (size_t)vocab) ||
        !int_guards_ok(target_guarded, (size_t)batch)) ++guard_failures;
    ++case_count;
    if ((batch & (batch - 1)) == 0) ++pow2_cases;
    else ++fallback_cases;
    if (fast_exp) ++fast_exp_cases;
    source_positions += count;
    free(reference_guarded);
    free(candidate_guarded);
    free(bias_guarded);
    free(bias_original);
    free(target_guarded);
    free(target_original);
}

int main(void) {
    const int vocab_sizes[] = {1, 2, 3, 7, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025,
        16384, 32768, 35328, 65536};
    const int batches[] = {1, 3, 4};
    for (size_t vi = 0; vi < sizeof(vocab_sizes) / sizeof(vocab_sizes[0]); ++vi)
        for (size_t bi = 0; bi < sizeof(batches) / sizeof(batches[0]); ++bi)
            for (int pattern = 0; pattern < PATTERNS; ++pattern)
                for (int threads = 1; threads <= 4; threads += 3)
                    run_case(batches[bi], vocab_sizes[vi], pattern, threads, 0, 1);
    /* Actual B128/V35328 plus non-power-of-two irregular fixture batch71. */
    const int extra_vocab[] = {7, 1025, 35328};
    const int extra_patterns[] = {0, 2, 8};
    for (size_t vi = 0; vi < sizeof(extra_vocab) / sizeof(extra_vocab[0]); ++vi)
        for (size_t pi = 0; pi < sizeof(extra_patterns) / sizeof(extra_patterns[0]); ++pi)
            for (int threads = 1; threads <= 4; threads += 3)
                run_case(128, extra_vocab[vi], extra_patterns[pi], threads, 0, 1);
    for (int threads = 1; threads <= 4; threads += 3) {
        for (int pattern = 0; pattern < PATTERNS; ++pattern) {
            run_case(71, 71, pattern, threads, 0, 1);
            run_case(4, 37, pattern, threads, 1, 1);
            run_case(71, 37, pattern, threads, 1, 1);
        }
        run_case(128, 1025, 8, threads, 0, 0);
    }

    ExperimentKernelOptions options = {0};
    const int invalid_guard_passed = isnan(fw_softmax(NULL, NULL, NULL, 0, 0, &options))
        && isnan(fw_softmax(NULL, NULL, NULL, -1, 3, &options))
        && isnan(fw_softmax(NULL, NULL, NULL, 4, 0, &options));
    /* Exact returned float32 bits are the strict acceptance condition. */
    const int passed = invalid_guard_passed && nonfinite_pairs == 0
        && guard_failures == 0 && immutable_input_failures == 0
        && loss_tolerance_failures == 0 && gradient_tolerance_failures == 0
        && loss_bit_differences == 0 && gradient_bit_differences == 0;
    printf("{\n"
        "  \"schema_version\": 1,\n"
        "  \"passed\": %s,\n"
        "  \"cases\": %" PRIu64 ",\n"
        "  \"power_of_two_batch_cases\": %" PRIu64 ",\n"
        "  \"non_power_of_two_fallback_cases\": %" PRIu64 ",\n"
        "  \"fast_exp_compatibility_cases\": %" PRIu64 ",\n"
        "  \"gradient_elements_compared\": %" PRIu64 ",\n"
        "  \"source_logit_positions_including_thread_repeats\": %" PRIu64 ",\n"
        "  \"input_min\": %.9g,\n"
        "  \"input_max\": %.9g,\n"
        "  \"loss_bit_differences\": %" PRIu64 ",\n"
        "  \"gradient_bit_differences\": %" PRIu64 ",\n"
        "  \"max_loss_absolute_error\": %.17g,\n"
        "  \"max_gradient_absolute_error\": %.17g,\n"
        "  \"loss_tolerance_failures\": %" PRIu64 ",\n"
        "  \"gradient_tolerance_failures\": %" PRIu64 ",\n"
        "  \"nonfinite_pairs\": %" PRIu64 ",\n"
        "  \"guard_failures\": %" PRIu64 ",\n"
        "  \"immutable_bias_target_failures\": %" PRIu64 ",\n"
        "  \"invalid_dimension_guard_passed\": %s\n"
        "}\n",
        passed ? "true" : "false", case_count, pow2_cases, fallback_cases,
        fast_exp_cases, gradient_count, source_positions, input_min, input_max,
        loss_bit_differences, gradient_bit_differences, max_loss_absolute_error,
        max_gradient_absolute_error, loss_tolerance_failures,
        gradient_tolerance_failures, nonfinite_pairs, guard_failures,
        immutable_input_failures, invalid_guard_passed ? "true" : "false");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
