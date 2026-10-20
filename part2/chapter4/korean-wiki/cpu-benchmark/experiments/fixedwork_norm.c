/* Fixed logical-block gradient norm, independent of OpenMP scheduling.
 * Only norm summation order changes: gradients remain float32 and every
 * product, block sum and final ordered merge uses double.  The caller owns
 * scratch storage; this module never allocates memory.
 */
#include "fixedwork_api.h"

#include <math.h>

#ifdef __FAST_MATH__
#error "The fixed-block norm comparison forbids global fast-math"
#endif

enum { FW_NORM_BLOCK = 4096 };

size_t fw_norm_scratch_count(size_t count) {
    /* Avoid overflow from the usual (count + block - 1) expression. */
    return count / FW_NORM_BLOCK + (count % FW_NORM_BLOCK != 0);
}

/* Both one-worker and multi-worker calls enter this same compiled helper.
 * Keeping the helper out of the parallel/serial caller prevents separate
 * inlining contexts from choosing different reduction implementations. */
#if defined(__clang__) || defined(__GNUC__)
__attribute__((noinline))
#endif
static double block_squared_norm(const float *gradients, size_t count) {
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
    for (size_t i = 0; i < count; ++i)
        sum += (double)gradients[i] * gradients[i];
    return sum;
}

float fw_clip_gradients(float *gradients, size_t count, float *clipping_scale,
                        const ExperimentKernelOptions *options,
                        double *partial_sums, size_t partial_capacity) {
    const size_t blocks = fw_norm_scratch_count(count);
    /* A zero-length input accepts NULL gradients and NULL/zero scratch.
     * Reject insufficient workspace before touching gradients or scratch. */
    if (!clipping_scale || !options || (count && !gradients)
        || blocks > partial_capacity || (blocks && !partial_sums)) {
        if (clipping_scale) *clipping_scale = NAN;
        return NAN;
    }
    const int threads = options->threads > 0 ? options->threads : 1;
    const int workers = options->reduction_parallel && threads > 1 ? threads : 1;
#ifndef _OPENMP
    (void)workers;
#endif
    if (blocks) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1)
#endif
        for (size_t block = 0; block < blocks; ++block) {
            const size_t start = block * FW_NORM_BLOCK;
            const size_t remaining = count - start;
            const size_t length = remaining < FW_NORM_BLOCK ? remaining : FW_NORM_BLOCK;
            partial_sums[block] = block_squared_norm(gradients + start, length);
        }
    }
    /* Intentionally ordered by logical block index, independent of which
     * worker computed each partial. No OpenMP/SIMD reduction on this merge. */
    double squares = 0.0;
    for (size_t block = 0; block < blocks; ++block) squares += partial_sums[block];
    const float norm = (float)sqrt(squares);
    const float scale = fminf(1.0f, options->clip_norm / (norm + 1e-6f));
    *clipping_scale = scale;

    /* Preserve the original conditional clipping write and dense traversal. */
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
