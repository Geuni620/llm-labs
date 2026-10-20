/* Round-2 softmax ILP candidate: drop-in replacement for fixedwork_softmax.c.
 *
 * Same API and the same per-row float32 operation sequence as the frozen
 * fw_softmax. The only change is scheduling: four rows run together so their
 * otherwise serial max and sum chains (vocab compare/select or add steps per
 * row) become four independent chains. Within each row:
 *   - bias add, the `>` max update and v order are the same as the original;
 *   - target_logit is still read after the bias add;
 *   - `row[v] -= maximum`, then one in-place vvexpf call per row with that
 *     row's own vocab length (rows are never merged into one vvexpf call);
 *   - the exp sum adds v = 0..vocab-1 into a float32 accumulator starting
 *     at 0.0f;
 *   - loss = logf(sum) + maximum - target_logit;
 *   - gradient = (row[v] / sum) * inverse_batch, then the target subtraction.
 * The loss is summed in the original row order, then divided by (float)batch.
 *
 * Unchanged from the original: non-power-of-two or invalid batches delegate
 * to experiment_softmax, and the fast_exp option uses a verbatim copy of the
 * original single-row function. Leftover rows after the four-row groups
 * (only batch 1 or 2 among power-of-two batches) use that same single-row
 * function with vvexpf. No explicit NEON intrinsics are used. A
 * NEON max (vmaxnm) was deliberately left out because of NaN/signed-zero
 * semantics.
 *
 * Precondition (also what the benchmark does): logits must not overlap bias
 * or targets. The original processes rows one after another and this
 * candidate interleaves four, so overlapping buffers could differ. The
 * engine's real B128/V35328 call passes separate allocations.
 */
#pragma STDC FP_CONTRACT OFF

#include "../fixedwork_api.h"
#include "../fast_exp.h"

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

/* Row engine, exported only so test_softmax_ilp.c can exercise four-row
 * groups plus leftover rows (for example 7 rows), which the public
 * power-of-two path never reaches. It writes one loss per row into
 * losses[0..rows-1] and touches only logits[0..rows*vocab-1]. */
void fw_softmax_ilp_rows(float *logits, const float *bias,
                         const uint32_t *targets, int rows, int vocab,
                         float inverse_batch, int fast_exp, int threads,
                         int parallel, float *losses);

/* Verbatim copy of fixedwork_softmax.c:fw_softmax_row. */
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

/* The original's shift loop and its single per-row vvexpf call. */
static void fw_shift_exp_row(float *row, float maximum, int vocab) {
    for (int v = 0; v < vocab; ++v) row[v] -= maximum;
    vvexpf(row, row, &vocab);
}

/* The original's normalization loop and target correction. */
static void fw_normalize_row(float *row, float sum, uint32_t target,
                             int vocab, float inverse_batch) {
    for (int v = 0; v < vocab; ++v)
        row[v] = (row[v] / sum) * inverse_batch;
    row[target] -= inverse_batch;
}

/* Rows first..first+3 with vForce exp. Each row has its own named max and
 * sum, so per-row results match the original bit for bit. */
static void fw_softmax_group4(float *logits, const float *bias,
                              const uint32_t *targets, int first, int vocab,
                              float inverse_batch, float *losses) {
    float *const r0 = logits + (size_t)first * vocab;
    float *const r1 = r0 + vocab;
    float *const r2 = r1 + vocab;
    float *const r3 = r2 + vocab;
    const uint32_t t0 = targets[first];
    const uint32_t t1 = targets[first + 1];
    const uint32_t t2 = targets[first + 2];
    const uint32_t t3 = targets[first + 3];

    /* Pass 1: row[v] += bias[v] (row value first), then the strict `>`
     * update, as in the original. bias[v] is loaded once for all four rows;
     * that relies on the no-overlap precondition above. */
    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        const float b = bias[v];
        const float x0 = r0[v] + b;
        const float x1 = r1[v] + b;
        const float x2 = r2[v] + b;
        const float x3 = r3[v] + b;
        r0[v] = x0;
        r1[v] = x1;
        r2[v] = x2;
        r3[v] = x3;
        if (x0 > m0) m0 = x0;
        if (x1 > m1) m1 = x1;
        if (x2 > m2) m2 = x2;
        if (x3 > m3) m3 = x3;
    }
    const float target_logit0 = r0[t0];
    const float target_logit1 = r1[t1];
    const float target_logit2 = r2[t2];
    const float target_logit3 = r3[t3];

    /* Pass 2: per-row shift + vvexpf with each row's own vocab length. */
    fw_shift_exp_row(r0, m0, vocab);
    fw_shift_exp_row(r1, m1, vocab);
    fw_shift_exp_row(r2, m2, vocab);
    fw_shift_exp_row(r3, m3, vocab);

    /* Pass 3: four independent float32 sums, each in ascending v order. */
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int v = 0; v < vocab; ++v) {
        s0 += r0[v];
        s1 += r1[v];
        s2 += r2[v];
        s3 += r3[v];
    }

    losses[first] = logf(s0) + m0 - target_logit0;
    losses[first + 1] = logf(s1) + m1 - target_logit1;
    losses[first + 2] = logf(s2) + m2 - target_logit2;
    losses[first + 3] = logf(s3) + m3 - target_logit3;

    /* Pass 4: unchanged element-wise normalization per row. */
    fw_normalize_row(r0, s0, t0, vocab, inverse_batch);
    fw_normalize_row(r1, s1, t1, vocab, inverse_batch);
    fw_normalize_row(r2, s2, t2, vocab, inverse_batch);
    fw_normalize_row(r3, s3, t3, vocab, inverse_batch);
}

void fw_softmax_ilp_rows(float *logits, const float *bias,
                         const uint32_t *targets, int rows, int vocab,
                         float inverse_batch, int fast_exp, int threads,
                         int parallel, float *losses) {
    if (rows < 1 || vocab < 1) return;
    const int use_parallel = parallel && threads > 1;
    if (fast_exp) {
        /* Keep the original row-wise fast_exp path and row distribution. */
        if (use_parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int b = 0; b < rows; ++b)
                losses[b] = fw_softmax_row(logits + (size_t)b * vocab, bias,
                                           targets[b], vocab, 1,
                                           inverse_batch);
        } else {
            for (int b = 0; b < rows; ++b)
                losses[b] = fw_softmax_row(logits + (size_t)b * vocab, bias,
                                           targets[b], vocab, 1,
                                           inverse_batch);
        }
        return;
    }
    /* Each OpenMP worker owns whole four-row groups: rows 4g..4g+3 and their
     * losses slots. With threads == 1 or parallel off, the same group code
     * runs serially. */
    const int groups = rows / 4;
    if (use_parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int g = 0; g < groups; ++g)
            fw_softmax_group4(logits, bias, targets, 4 * g, vocab,
                              inverse_batch, losses);
    } else {
        for (int g = 0; g < groups; ++g)
            fw_softmax_group4(logits, bias, targets, 4 * g, vocab,
                              inverse_batch, losses);
    }
    /* 0..3 leftover rows: the original single-row function, run serially.
     * Per-row results do not depend on which thread runs them. */
    for (int b = 4 * groups; b < rows; ++b)
        losses[b] = fw_softmax_row(logits + (size_t)b * vocab, bias,
                                   targets[b], vocab, 0, inverse_batch);
}

float fw_softmax(float *logits, const float *bias, const uint32_t *targets,
                 int batch, int vocab, const ExperimentKernelOptions *options) {
    if (batch < 1 || vocab < 1 ||
        ((unsigned int)batch & ((unsigned int)batch - 1U)) != 0U) {
        return experiment_softmax(logits, bias, targets, batch, vocab, options);
    }
    const float inverse_batch = 1.0f / (float)batch;
    const int threads = options->threads > 0 ? options->threads : 1;
    /* The engine bounds batch to 1024 (4 KiB), the same bound the original
     * relies on for its parallel losses array. */
    float losses[batch];
    fw_softmax_ilp_rows(logits, bias, targets, batch, vocab, inverse_batch,
                        options->fast_exp, threads, options->softmax_parallel,
                        losses);
    /* ((0 + l0) + l1) + ...: the same float sequence as both original
     * branches. */
    float loss_sum = 0.0f;
    for (int b = 0; b < batch; ++b) loss_sum += losses[b];
    return loss_sum / (float)batch;
}
