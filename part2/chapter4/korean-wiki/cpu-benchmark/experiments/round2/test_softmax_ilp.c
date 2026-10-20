/* Bitwise equivalence test: frozen fixedwork_softmax.c vs round2/softmax_ilp.c.
 *
 * Link contract (see NOTES.md):
 *   baseline object = ../fixedwork_softmax.c compiled with
 *                     -Dfw_softmax=baseline_fw_softmax
 *   candidate object = softmax_ilp.c (exports fw_softmax and
 *                      fw_softmax_ilp_rows)
 *   shared object = ../vector_vforce.c (experiment_softmax fallback)
 *
 * Acceptance (finite gate, exit 0): for every finite-input case, the whole
 * guarded logits buffer (canaries included) is memcmp-identical between the
 * baseline and the candidate. The returned loss bits match, bias/targets are
 * unchanged, every canary is intact, and baseline outputs are finite (the
 * same finite-output rule as test_fixedwork_softmax.c). No tolerance is
 * used.
 *
 * Non-finite probe (reported separately, exit 3 if only this differs):
 * NaN/Inf inputs are compared with memcmp too, but outside the finite gate.
 * The training gate defines finite inputs only, and IEEE 754 does not fix
 * which NaN payload propagates when two different NaNs meet. To keep
 * memcmp meaningful the probe uses only the single canonical NAN payload.
 * Mixed NaN payloads are deliberately not tested, and the JSON says so.
 */
#include "../fixedwork_api.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "This numerical comparison must not use global fast-math."
#endif

/* Frozen original, renamed at compile time by -Dfw_softmax=baseline_fw_softmax. */
float baseline_fw_softmax(float *logits, const float *bias,
                          const uint32_t *targets, int batch, int vocab,
                          const ExperimentKernelOptions *options);
/* Candidate row engine; same prototype as in softmax_ilp.c. */
void fw_softmax_ilp_rows(float *logits, const float *bias,
                         const uint32_t *targets, int rows, int vocab,
                         float inverse_batch, int fast_exp, int threads,
                         int parallel, float *losses);

/* 16 floats = 64 bytes: data starts 64-byte aligned in both buffers, so any
 * alignment-dependent vvexpf code path is the same for both sides. */
enum { GUARD = 16, FINITE_PATTERNS = 19, NONFINITE_PATTERNS = 7 };
static const uint32_t float_canary = UINT32_C(0x4a123456);
static const uint32_t int_canary = UINT32_C(0xbad0beef);
static uint32_t random_state = 1;

typedef struct { int threads; int parallel; } Config;

/* Finite gate counters. */
static uint64_t finite_cases, grouped_vforce_cases, tail_only_vforce_cases;
static uint64_t fast_exp_row_cases, delegated_non_pow2_cases;
static uint64_t effective_parallel_cases, actual_vocab_cases;
static uint64_t compared_floats, gradient_bit_differences, loss_bit_differences;
static uint64_t mismatching_cases, nonfinite_outputs_in_finite_cases;
static uint64_t guard_failures, immutable_input_failures;
static int first_failure_recorded;
static char first_failure[256];
/* Row-engine (groups + tail) counters. */
static uint64_t engine_cases, engine_gradient_bit_differences;
static uint64_t engine_loss_bit_differences, engine_guard_failures;
/* Non-finite probe counters. */
static uint64_t nonfinite_cases, nonfinite_mismatching_cases;
static uint64_t nonfinite_gradient_bit_differences, nonfinite_loss_bit_differences;
static uint64_t nonfinite_guard_failures, nonfinite_class_differences;
static uint64_t nonfinite_isolation_failures;

static uint32_t bits(float x) {
    uint32_t out;
    memcpy(&out, &x, sizeof(out));
    return out;
}

static void seed_random(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t x = UINT32_C(0x4a63b019) ^ (a * UINT32_C(0x9e3779b1))
               ^ (b * UINT32_C(0x85ebca6b)) ^ (c * UINT32_C(0xc2b2ae35));
    random_state = x ? x : UINT32_C(0x1234567);
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

static int is_pow2(int n) {
    return n > 0 && (((unsigned int)n & ((unsigned int)n - 1U)) == 0U);
}

static void note_failure(const char *what, int batch, int vocab, int pattern,
                         Config cfg, int fast_exp) {
    if (first_failure_recorded) return;
    first_failure_recorded = 1;
    snprintf(first_failure, sizeof(first_failure),
             "%s B=%d V=%d pattern=%d threads=%d parallel=%d fast_exp=%d",
             what, batch, vocab, pattern, cfg.threads, cfg.parallel, fast_exp);
}

/* ---- guarded, 64-byte-aligned float buffers ---------------------------- */

typedef struct { void *base; float *guarded; float *data; size_t count; } Buffer;

static Buffer buffer_new(size_t count) {
    Buffer out;
    out.count = count;
    out.base = malloc((count + 2 * GUARD) * sizeof(float) + 64);
    if (!out.base) {
        fputs("allocation failure\n", stderr);
        exit(EXIT_FAILURE);
    }
    uintptr_t at = ((uintptr_t)out.base + 63U) & ~(uintptr_t)63U;
    out.guarded = (float *)at;
    out.data = out.guarded + GUARD;
    for (size_t i = 0; i < GUARD; ++i) {
        memcpy(out.guarded + i, &float_canary, sizeof(float_canary));
        memcpy(out.data + count + i, &float_canary, sizeof(float_canary));
    }
    return out;
}

static int buffer_guards_ok(const Buffer *b) {
    for (size_t i = 0; i < GUARD; ++i)
        if (bits(b->guarded[i]) != float_canary ||
            bits(b->data[b->count + i]) != float_canary) return 0;
    return 1;
}

static void buffer_free(Buffer *b) {
    free(b->base);
    b->base = NULL;
}

typedef struct { uint32_t *guarded; uint32_t *data; size_t count; } Targets;

static Targets targets_new(size_t count) {
    Targets out;
    out.count = count;
    out.guarded = malloc((count + 2 * GUARD) * sizeof(uint32_t));
    if (!out.guarded) {
        fputs("allocation failure\n", stderr);
        exit(EXIT_FAILURE);
    }
    out.data = out.guarded + GUARD;
    for (size_t i = 0; i < GUARD; ++i) {
        out.guarded[i] = int_canary;
        out.data[count + i] = int_canary;
    }
    return out;
}

static int targets_guards_ok(const Targets *t) {
    for (size_t i = 0; i < GUARD; ++i)
        if (t->guarded[i] != int_canary || t->data[t->count + i] != int_canary)
            return 0;
    return 1;
}

/* Counts differing float bit patterns over the whole guarded region. */
static uint64_t count_bit_differences(const Buffer *a, const Buffer *b) {
    const size_t total = a->count + 2 * GUARD;
    if (memcmp(a->guarded, b->guarded, total * sizeof(float)) == 0) return 0;
    uint64_t differences = 0;
    for (size_t i = 0; i < total; ++i)
        if (bits(a->guarded[i]) != bits(b->guarded[i])) ++differences;
    return differences;
}

/* ---- deterministic finite inputs --------------------------------------- */

static float finite_value(int pattern, int row, int column, int vocab) {
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
        case 13: {
            static const float boundary[] = {0.0f, -87.33654f, -87.33655f,
                -87.33656f, -103.27892f, -103.27893f, -103.97208f, -104.0f};
            return boundary[column % 8];
        }
        /* Signed-zero ties: odd rows start with -0.0, so their max is -0.0. */
        case 14: return ((column + row) % 2) ? -0.0f : 0.0f;
        /* Duplicate max at first and last column. */
        case 15: return (column == 0 || column == vocab - 1)
                     ? 7.0f : (u * 2.0f - 1.0f) * 4.0f;
        /* Trained-logit-like: roughly bell-shaped, sd about 3, row offset. */
        case 16: {
            const float u2 = unit_random(), u3 = unit_random(), u4 = unit_random();
            return ((u + u2 + u3 + u4) - 2.0f) * 3.0f + 0.25f * (float)(row % 7);
        }
        /* Max first on even rows, last on odd rows: lanes in a group differ. */
        case 17: return row % 2 == 0 ? -(float)column * 0.01f
                                     : (float)column * 0.01f - 50.0f;
        /* Strongly different scale per row within one four-row group. */
        default: return (u * 2.0f - 1.0f) * (float)(1 + row % 5) * 20.0f;
    }
}

static float finite_bias(int pattern, int column) {
    switch (pattern) {
        case 2: return (unit_random() - 0.5f) * 0.1f;
        case 11: return ((float)(column % 11) - 5.0f) * 0.125f;
        case 14: return -0.0f;
        case 16: return (unit_random() - 0.5f) * 0.5f;
        default: return 0.0f;
    }
}

static uint32_t target_for(int row, int vocab, int pattern) {
    switch (row % 4) {
        case 0: return 0U;
        case 1: return (uint32_t)(vocab - 1);
        case 2: return (uint32_t)((row * 71 + pattern * 17) % vocab);
        default: return (uint32_t)((row * 7919 + 13) % vocab);
    }
}

/* Non-finite probe inputs. Only canonical NAN; finite values elsewhere. */
static float nonfinite_value(int pattern, int row, int column, int batch,
                             int vocab) {
    const float u = (unit_random() * 2.0f - 1.0f) * 4.0f;
    const int nan_row = batch > 1 ? 1 : 0;
    switch (pattern) {
        case 0: return (row == nan_row && column == 3 % vocab) ? NAN : u;
        case 1: return (row == 0 && column == 5 % vocab) ? INFINITY : u;
        case 2: return row == 0 ? -INFINITY : u;
        case 3: return column % 2 == 0 ? -INFINITY : u;
        case 4: return column == 0 ? INFINITY
                       : column == vocab - 1 ? -INFINITY : u;
        case 5: return u; /* NaN lives in the bias for this pattern. */
        default: return row == (batch > 2 ? 2 : 0) ? INFINITY : u;
    }
}

/* ---- one public fw_softmax comparison ---------------------------------- */

static void run_case(int batch, int vocab, int pattern, Config cfg,
                     int fast_exp, int nonfinite) {
    const size_t count = (size_t)batch * vocab;
    Buffer reference = buffer_new(count), candidate = buffer_new(count);
    Buffer bias = buffer_new((size_t)vocab);
    Targets targets = targets_new((size_t)batch);
    float *bias_original = malloc((size_t)vocab * sizeof(float));
    uint32_t *targets_original = malloc((size_t)batch * sizeof(uint32_t));
    if (!bias_original || !targets_original) {
        fputs("allocation failure\n", stderr);
        exit(EXIT_FAILURE);
    }
    seed_random((uint32_t)batch, (uint32_t)vocab,
                (uint32_t)(pattern + (nonfinite ? 1000 : 0)));
    for (int v = 0; v < vocab; ++v)
        bias.data[v] = nonfinite ? (pattern == 5 && v == 2 % vocab ? NAN : 0.0f)
                                 : finite_bias(pattern, v);
    for (int b = 0; b < batch; ++b) {
        targets.data[b] = target_for(b, vocab, pattern);
        for (int v = 0; v < vocab; ++v)
            reference.data[(size_t)b * vocab + v] = nonfinite
                ? nonfinite_value(pattern, b, v, batch, vocab)
                : finite_value(pattern, b, v, vocab);
    }
    memcpy(candidate.data, reference.data, count * sizeof(float));
    memcpy(bias_original, bias.data, (size_t)vocab * sizeof(float));
    memcpy(targets_original, targets.data, (size_t)batch * sizeof(uint32_t));

    ExperimentKernelOptions options = {0};
    options.threads = cfg.threads;
    options.softmax_parallel = cfg.parallel;
    options.fast_exp = fast_exp;
    options.clip_norm = 1.0f;

    int immutable_ok = 1;
    const float reference_loss = baseline_fw_softmax(reference.data, bias.data,
        targets.data, batch, vocab, &options);
    if (memcmp(bias_original, bias.data, (size_t)vocab * sizeof(float)) ||
        memcmp(targets_original, targets.data, (size_t)batch * sizeof(uint32_t)))
        immutable_ok = 0;
    const float candidate_loss = fw_softmax(candidate.data, bias.data,
        targets.data, batch, vocab, &options);
    if (memcmp(bias_original, bias.data, (size_t)vocab * sizeof(float)) ||
        memcmp(targets_original, targets.data, (size_t)batch * sizeof(uint32_t)))
        immutable_ok = 0;

    const uint64_t differences = count_bit_differences(&reference, &candidate);
    const int loss_differs = bits(reference_loss) != bits(candidate_loss);
    const int guards_ok = buffer_guards_ok(&reference) &&
        buffer_guards_ok(&candidate) && buffer_guards_ok(&bias) &&
        targets_guards_ok(&targets);

    if (nonfinite) {
        ++nonfinite_cases;
        nonfinite_gradient_bit_differences += differences;
        nonfinite_loss_bit_differences += (uint64_t)loss_differs;
        if (differences || loss_differs) ++nonfinite_mismatching_cases;
        if (!guards_ok || !immutable_ok) ++nonfinite_guard_failures;
        /* Weaker diagnostic: same NaN/Inf/finite class per element. */
        for (size_t i = 0; i < count; ++i) {
            const float x = reference.data[i], y = candidate.data[i];
            if (isnan(x) != isnan(y) || isinf(x) != isinf(y)) {
                ++nonfinite_class_differences;
                break;
            }
        }
        /* A NaN in row 1 (pattern 0) or Inf in row 0/2 must not leak into
         * the other rows of the same four-row group: rows without injected
         * non-finite input must stay finite. */
        if (pattern == 0 || pattern == 1 || pattern == 6) {
            const int dirty = pattern == 0 ? (batch > 1 ? 1 : 0)
                            : pattern == 1 ? 0 : (batch > 2 ? 2 : 0);
            for (int b = 0; b < batch; ++b) {
                if (b == dirty) continue;
                for (int v = 0; v < vocab; ++v)
                    if (!isfinite(candidate.data[(size_t)b * vocab + v])) {
                        ++nonfinite_isolation_failures;
                        b = batch;
                        break;
                    }
            }
        }
    } else {
        ++finite_cases;
        const int threads = cfg.threads > 0 ? cfg.threads : 1;
        if (!is_pow2(batch)) ++delegated_non_pow2_cases;
        else if (fast_exp) ++fast_exp_row_cases;
        else if (batch >= 4) ++grouped_vforce_cases;
        else ++tail_only_vforce_cases;
        if (cfg.parallel && threads > 1) ++effective_parallel_cases;
        if (vocab == 35328) ++actual_vocab_cases;
        compared_floats += count + 2 * GUARD;
        gradient_bit_differences += differences;
        loss_bit_differences += (uint64_t)loss_differs;
        if (differences || loss_differs) {
            ++mismatching_cases;
            note_failure("bit mismatch", batch, vocab, pattern, cfg, fast_exp);
        }
        if (!guards_ok) {
            ++guard_failures;
            note_failure("guard", batch, vocab, pattern, cfg, fast_exp);
        }
        if (!immutable_ok) {
            ++immutable_input_failures;
            note_failure("bias/target mutated", batch, vocab, pattern, cfg,
                         fast_exp);
        }
        int finite = isfinite(reference_loss);
        for (size_t i = 0; finite && i < count; ++i)
            finite = isfinite(reference.data[i]);
        if (!finite) {
            ++nonfinite_outputs_in_finite_cases;
            note_failure("nonfinite output", batch, vocab, pattern, cfg,
                         fast_exp);
        }
    }
    buffer_free(&reference);
    buffer_free(&candidate);
    buffer_free(&bias);
    free(targets.guarded);
    free(bias_original);
    free(targets_original);
}

/* ---- row engine: four-row groups followed by 1..3 leftover rows -------- */

/* baseline_fw_softmax(full_batch rows) vs fw_softmax_ilp_rows(rows <
 * full_batch, inverse 1/full_batch). Rows are independent, so the first
 * `rows` gradient rows must match bitwise; the canary sits right after
 * row `rows` to catch overruns. Per-row losses of the partial call must
 * also equal those of a full-batch engine call (tail path vs group path for
 * the same row). The full call's ordered loss sum / batch must match the
 * baseline loss bits. */
static void run_engine_case(int rows, int full_batch, int vocab, int pattern,
                            Config cfg, int fast_exp) {
    const size_t full_count = (size_t)full_batch * vocab;
    const size_t count = (size_t)rows * vocab;
    Buffer reference = buffer_new(full_count), partial = buffer_new(count);
    Buffer full = buffer_new(full_count), bias = buffer_new((size_t)vocab);
    Targets targets = targets_new((size_t)full_batch);
    float *partial_losses = malloc((size_t)rows * sizeof(float));
    float *full_losses = malloc((size_t)full_batch * sizeof(float));
    if (!partial_losses || !full_losses) {
        fputs("allocation failure\n", stderr);
        exit(EXIT_FAILURE);
    }
    seed_random((uint32_t)rows * 131U + (uint32_t)full_batch, (uint32_t)vocab,
                (uint32_t)(pattern + 2000));
    for (int v = 0; v < vocab; ++v) bias.data[v] = finite_bias(pattern, v);
    for (int b = 0; b < full_batch; ++b) {
        targets.data[b] = target_for(b, vocab, pattern);
        for (int v = 0; v < vocab; ++v)
            reference.data[(size_t)b * vocab + v] = finite_value(pattern, b, v, vocab);
    }
    memcpy(partial.data, reference.data, count * sizeof(float));
    memcpy(full.data, reference.data, full_count * sizeof(float));

    ExperimentKernelOptions options = {0};
    options.threads = cfg.threads;
    options.softmax_parallel = cfg.parallel;
    options.fast_exp = fast_exp;
    const int threads = cfg.threads > 0 ? cfg.threads : 1;
    const float inverse_batch = 1.0f / (float)full_batch;
    const float reference_loss = baseline_fw_softmax(reference.data, bias.data,
        targets.data, full_batch, vocab, &options);
    fw_softmax_ilp_rows(partial.data, bias.data, targets.data, rows, vocab,
                        inverse_batch, fast_exp, threads, cfg.parallel,
                        partial_losses);
    fw_softmax_ilp_rows(full.data, bias.data, targets.data, full_batch, vocab,
                        inverse_batch, fast_exp, threads, cfg.parallel,
                        full_losses);

    uint64_t differences = 0;
    if (memcmp(reference.data, partial.data, count * sizeof(float)))
        for (size_t i = 0; i < count; ++i)
            differences += bits(reference.data[i]) != bits(partial.data[i]);
    differences += count_bit_differences(&reference, &full);
    uint64_t loss_differences = 0;
    for (int b = 0; b < rows; ++b)
        loss_differences += bits(partial_losses[b]) != bits(full_losses[b]);
    float loss_sum = 0.0f;
    for (int b = 0; b < full_batch; ++b) loss_sum += full_losses[b];
    loss_differences += bits(loss_sum / (float)full_batch) != bits(reference_loss);

    ++engine_cases;
    engine_gradient_bit_differences += differences;
    engine_loss_bit_differences += loss_differences;
    if (differences || loss_differences)
        note_failure("engine bit mismatch", rows, vocab, pattern, cfg, fast_exp);
    if (!buffer_guards_ok(&reference) || !buffer_guards_ok(&partial) ||
        !buffer_guards_ok(&full) || !buffer_guards_ok(&bias) ||
        !targets_guards_ok(&targets)) {
        ++engine_guard_failures;
        note_failure("engine guard", rows, vocab, pattern, cfg, fast_exp);
    }
    buffer_free(&reference);
    buffer_free(&partial);
    buffer_free(&full);
    buffer_free(&bias);
    free(targets.guarded);
    free(partial_losses);
    free(full_losses);
}

/* ---- negative controls: the comparison must detect a 1-ulp change ------ */

static int negative_controls(void) {
    Buffer a = buffer_new(64), b = buffer_new(64);
    for (int i = 0; i < 64; ++i) a.data[i] = b.data[i] = (float)i * 0.5f;
    int ok = count_bit_differences(&a, &b) == 0;
    b.data[17] = nextafterf(b.data[17], INFINITY);
    ok = ok && count_bit_differences(&a, &b) == 1;
    b.data[17] = a.data[17];
    b.data[0] = -0.0f; /* +0.0 vs -0.0 must count as a difference. */
    ok = ok && count_bit_differences(&a, &b) == 1;
    b.data[0] = a.data[0];
    b.data[64] = 0.0f; /* trailing canary overwritten */
    ok = ok && !buffer_guards_ok(&b) && count_bit_differences(&a, &b) == 1;
    buffer_free(&a);
    buffer_free(&b);
    return ok;
}

int main(void) {
    const int negative_controls_passed = negative_controls();

    /* A. Small matrix: every batch x boundary vocab x pattern x config. */
    static const int batches[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 71, 128, 256, 512};
    static const int small_vocab[] = {1, 2, 3, 4, 5, 7, 8, 16, 17, 64, 257, 1025};
    static const Config configs[] = {
        {1, 0}, {1, 1}, {4, 0}, {4, 1}, {8, 1}, {0, 1}};
    static const Config fast_configs[] = {{1, 0}, {4, 1}};
    const size_t nb = sizeof(batches) / sizeof(batches[0]);
    const size_t nv = sizeof(small_vocab) / sizeof(small_vocab[0]);
    const size_t nc = sizeof(configs) / sizeof(configs[0]);
    for (size_t bi = 0; bi < nb; ++bi)
        for (size_t vi = 0; vi < nv; ++vi)
            for (int pattern = 0; pattern < FINITE_PATTERNS; ++pattern) {
                for (size_t ci = 0; ci < nc; ++ci)
                    run_case(batches[bi], small_vocab[vi], pattern, configs[ci], 0, 0);
                for (size_t ci = 0; ci < 2; ++ci)
                    run_case(batches[bi], small_vocab[vi], pattern, fast_configs[ci], 1, 0);
            }

    /* B. Actual vocabulary V35328: a few batches and patterns, not the full
     * matrix. Peak memory is the B512 case (~2 x 72 MB logits). */
    static const int actual_batches[] = {1, 2, 3, 4, 8, 16, 71, 128};
    static const int actual_patterns[] = {0, 2, 8, 13, 16, 18};
    static const Config actual_configs[] = {{1, 0}, {4, 1}, {8, 1}};
    for (size_t bi = 0; bi < sizeof(actual_batches) / sizeof(actual_batches[0]); ++bi)
        for (size_t pi = 0; pi < sizeof(actual_patterns) / sizeof(actual_patterns[0]); ++pi)
            for (size_t ci = 0; ci < 3; ++ci)
                run_case(actual_batches[bi], 35328, actual_patterns[pi],
                         actual_configs[ci], 0, 0);
    run_case(128, 35328, 16, (Config){1, 0}, 1, 0);
    run_case(128, 35328, 16, (Config){4, 1}, 1, 0);
    run_case(256, 35328, 16, (Config){4, 1}, 0, 0);
    run_case(256, 35328, 16, (Config){8, 1}, 0, 0);
    run_case(512, 35328, 16, (Config){4, 1}, 0, 0);

    /* C. Row engine with groups + 1..3 leftover rows (unreachable through
     * the public power-of-two path except B1/B2). */
    static const Config engine_configs[] = {{1, 0}, {4, 1}, {8, 1}};
    static const int engine_vocab[] = {1, 7, 257};
    for (size_t ci = 0; ci < 3; ++ci)
        for (size_t vi = 0; vi < 3; ++vi)
            for (int pattern = 0; pattern < FINITE_PATTERNS; pattern += 3) {
                for (int rows = 1; rows < 8; ++rows)
                    run_engine_case(rows, 8, engine_vocab[vi], pattern,
                                    engine_configs[ci], 0);
                for (int rows = 9; rows < 16; ++rows)
                    run_engine_case(rows, 16, engine_vocab[vi], pattern,
                                    engine_configs[ci], 0);
                run_engine_case(7, 8, engine_vocab[vi], pattern,
                                engine_configs[ci], 1);
            }
    for (size_t ci = 0; ci < 3; ++ci)
        run_engine_case(7, 8, 35328, 16, engine_configs[ci], 0);

    /* D. Non-finite probe (separate verdict). */
    static const int nf_batches[] = {1, 4, 8, 16};
    static const int nf_vocab[] = {1, 7, 257};
    static const Config nf_configs[] = {{1, 0}, {4, 1}, {8, 1}};
    for (size_t bi = 0; bi < 4; ++bi)
        for (size_t vi = 0; vi < 3; ++vi)
            for (int pattern = 0; pattern < NONFINITE_PATTERNS; ++pattern)
                for (size_t ci = 0; ci < 3; ++ci)
                    for (int fast_exp = 0; fast_exp <= 1; ++fast_exp)
                        run_case(nf_batches[bi], nf_vocab[vi], pattern,
                                 nf_configs[ci], fast_exp, 1);

    /* E. Invalid dimensions: both must delegate and return the same bits. */
    ExperimentKernelOptions options = {0};
    const int invalid_dims[][2] = {{0, 0}, {-1, 3}, {4, 0}, {0, 5}};
    int invalid_guard_passed = 1;
    for (size_t i = 0; i < 4; ++i) {
        const float x = baseline_fw_softmax(NULL, NULL, NULL, invalid_dims[i][0],
                                            invalid_dims[i][1], &options);
        const float y = fw_softmax(NULL, NULL, NULL, invalid_dims[i][0],
                                   invalid_dims[i][1], &options);
        if (!isnan(y) || bits(x) != bits(y)) invalid_guard_passed = 0;
    }

    const int finite_gate_passed = negative_controls_passed
        && invalid_guard_passed && mismatching_cases == 0
        && gradient_bit_differences == 0 && loss_bit_differences == 0
        && nonfinite_outputs_in_finite_cases == 0 && guard_failures == 0
        && immutable_input_failures == 0 && engine_gradient_bit_differences == 0
        && engine_loss_bit_differences == 0 && engine_guard_failures == 0
        && grouped_vforce_cases > 0 && actual_vocab_cases > 0;
    const int nonfinite_probe_identical = nonfinite_mismatching_cases == 0
        && nonfinite_guard_failures == 0 && nonfinite_class_differences == 0
        && nonfinite_isolation_failures == 0;

    printf("{\n"
        "  \"schema_version\": 1,\n"
        "  \"comparison\": \"baseline_fw_softmax (fixedwork_softmax.c) vs fw_softmax (round2/softmax_ilp.c)\",\n"
        "  \"finite_gate_passed\": %s,\n"
        "  \"finite_gate_rule\": \"memcmp of whole guarded logits + loss bits; finite outputs required; no tolerance\",\n"
        "  \"negative_controls_passed\": %s,\n"
        "  \"finite_cases\": %" PRIu64 ",\n"
        "  \"grouped_vforce_cases\": %" PRIu64 ",\n"
        "  \"tail_only_vforce_cases_b1_b2\": %" PRIu64 ",\n"
        "  \"fast_exp_row_cases\": %" PRIu64 ",\n"
        "  \"delegated_non_power_of_two_cases\": %" PRIu64 ",\n"
        "  \"effective_parallel_cases\": %" PRIu64 ",\n"
        "  \"actual_vocab_35328_cases\": %" PRIu64 ",\n"
        "  \"compared_floats_including_canaries\": %" PRIu64 ",\n"
        "  \"mismatching_cases\": %" PRIu64 ",\n"
        "  \"gradient_bit_differences\": %" PRIu64 ",\n"
        "  \"loss_bit_differences\": %" PRIu64 ",\n"
        "  \"nonfinite_outputs_in_finite_cases\": %" PRIu64 ",\n"
        "  \"guard_failures\": %" PRIu64 ",\n"
        "  \"immutable_bias_target_failures\": %" PRIu64 ",\n"
        "  \"engine_group_plus_tail_cases\": %" PRIu64 ",\n"
        "  \"engine_gradient_bit_differences\": %" PRIu64 ",\n"
        "  \"engine_loss_bit_differences\": %" PRIu64 ",\n"
        "  \"engine_guard_failures\": %" PRIu64 ",\n"
        "  \"invalid_dimension_guard_passed\": %s,\n"
        "  \"nonfinite_probe\": {\n"
        "    \"part_of_finite_gate\": false,\n"
        "    \"nan_payloads_tested\": \"canonical NAN only; mixed payloads not tested\",\n"
        "    \"bitwise_identical\": %s,\n"
        "    \"cases\": %" PRIu64 ",\n"
        "    \"mismatching_cases\": %" PRIu64 ",\n"
        "    \"gradient_bit_differences\": %" PRIu64 ",\n"
        "    \"loss_bit_differences\": %" PRIu64 ",\n"
        "    \"class_differences\": %" PRIu64 ",\n"
        "    \"row_isolation_failures\": %" PRIu64 ",\n"
        "    \"guard_or_immutability_failures\": %" PRIu64 "\n"
        "  },\n"
        "  \"first_failure\": \"%s\"\n"
        "}\n",
        finite_gate_passed ? "true" : "false",
        negative_controls_passed ? "true" : "false",
        finite_cases, grouped_vforce_cases, tail_only_vforce_cases,
        fast_exp_row_cases, delegated_non_pow2_cases, effective_parallel_cases,
        actual_vocab_cases, compared_floats, mismatching_cases,
        gradient_bit_differences, loss_bit_differences,
        nonfinite_outputs_in_finite_cases, guard_failures,
        immutable_input_failures, engine_cases, engine_gradient_bit_differences,
        engine_loss_bit_differences, engine_guard_failures,
        invalid_guard_passed ? "true" : "false",
        nonfinite_probe_identical ? "true" : "false", nonfinite_cases,
        nonfinite_mismatching_cases, nonfinite_gradient_bit_differences,
        nonfinite_loss_bit_differences, nonfinite_class_differences,
        nonfinite_isolation_failures, nonfinite_guard_failures,
        first_failure_recorded ? first_failure : "");
    if (!finite_gate_passed) return EXIT_FAILURE;
    return nonfinite_probe_identical ? EXIT_SUCCESS : 3;
}
