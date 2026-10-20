/* Cache-tiled row-major SGEMM for the CPU timing experiments.
 *
 * This is deliberately independent of BLAS.  Each output tile is owned by one
 * worker, so its reduction over K is serial and needs neither atomics nor an
 * extra cross-worker reduction.  A, B and C are packed into bounded stack
 * scratch space; the innermost loop reads and writes contiguous elements.
 *
 * Inputs must satisfy the usual SGEMM contract: nonnegative dimensions,
 * adequate leading dimensions and nonoverlapping input/output matrices.
 * trans_a/trans_b are 0 for no transpose and nonzero for transpose.  tile is
 * 32 or 64; a different value uses 32.  Threads are a requested upper budget
 * and only take effect in a build with OpenMP enabled.
 */
#include "experiment_kernels.h"

#include <stddef.h>
#include <string.h>

enum { GEMM_MAX_TILE = 64 };

static int smaller(int left, int right) {
    return left < right ? left : right;
}

static void pack_a(float *restrict packed, const float *restrict source,
                   int transposed, int row, int inner, int rows, int depth,
                   int leading, int stride) {
    if (!transposed) {
        for (int i = 0; i < rows; ++i)
            memcpy(packed + (size_t)i * stride,
                   source + (size_t)(row + i) * leading + inner,
                   (size_t)depth * sizeof(float));
    } else {
        /* Small transposition blocks keep both source and packed cache lines
         * resident, including at irregular matrix and tile boundaries. */
        for (int p0 = 0; p0 < depth; p0 += 8)
            for (int i0 = 0; i0 < rows; i0 += 8)
                for (int p = p0; p < smaller(p0 + 8, depth); ++p)
                    for (int i = i0; i < smaller(i0 + 8, rows); ++i)
                        packed[(size_t)i * stride + p] =
                            source[(size_t)(inner + p) * leading + row + i];
    }
}

static void pack_b(float *restrict packed, const float *restrict source,
                   int transposed, int inner, int column, int depth, int columns,
                   int leading, int stride) {
    if (!transposed) {
        for (int p = 0; p < depth; ++p)
            memcpy(packed + (size_t)p * stride,
                   source + (size_t)(inner + p) * leading + column,
                   (size_t)columns * sizeof(float));
    } else {
        for (int j0 = 0; j0 < columns; j0 += 8)
            for (int p0 = 0; p0 < depth; p0 += 8)
                for (int j = j0; j < smaller(j0 + 8, columns); ++j)
                    for (int p = p0; p < smaller(p0 + 8, depth); ++p)
                        packed[(size_t)p * stride + j] =
                            source[(size_t)(column + j) * leading + inner + p];
    }
}

static void output_tile(int trans_a, int trans_b, int m, int n, int k,
                        float alpha, const float *restrict a, int lda,
                        const float *restrict b, int ldb, float beta,
                        float *restrict c, int ldc, int tile, int row, int column) {
    _Alignas(64) float packed_a[GEMM_MAX_TILE * GEMM_MAX_TILE];
    _Alignas(64) float packed_b[GEMM_MAX_TILE * GEMM_MAX_TILE];
    _Alignas(64) float packed_c[GEMM_MAX_TILE * GEMM_MAX_TILE];
    const int rows = smaller(tile, m - row);
    const int columns = smaller(tile, n - column);

    for (int i = 0; i < rows; ++i) {
        float *restrict out = packed_c + (size_t)i * tile;
        for (int j = 0; j < columns; ++j) out[j] = 0.0f;
    }

    if (alpha != 0.0f) {
        for (int inner = 0; inner < k; inner += tile) {
            const int depth = smaller(tile, k - inner);
            pack_a(packed_a, a, trans_a, row, inner, rows, depth, lda, tile);
            pack_b(packed_b, b, trans_b, inner, column, depth, columns, ldb, tile);
            for (int i = 0; i < rows; ++i) {
                float *restrict out = packed_c + (size_t)i * tile;
                const float *restrict left = packed_a + (size_t)i * tile;
                for (int p = 0; p < depth; ++p) {
                    const float value = left[p];
                    const float *restrict right = packed_b + (size_t)p * tile;
#ifdef _OPENMP
#pragma omp simd
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
                    for (int j = 0; j < columns; ++j)
                        out[j] += value * right[j];
                }
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        const float *restrict sum = packed_c + (size_t)i * tile;
        float *restrict destination = c + (size_t)(row + i) * ldc + column;
        if (alpha == 0.0f || k == 0) {
            if (beta == 0.0f) {
                for (int j = 0; j < columns; ++j) destination[j] = 0.0f;
            } else {
                for (int j = 0; j < columns; ++j) destination[j] *= beta;
            }
        } else if (beta == 0.0f) {
            /* SGEMM beta=0 must not read the previous C (which may be NaN). */
            for (int j = 0; j < columns; ++j) destination[j] = alpha * sum[j];
        } else {
            for (int j = 0; j < columns; ++j)
                destination[j] = alpha * sum[j] + beta * destination[j];
        }
    }
}

void experiment_sgemm(int trans_a, int trans_b, int m, int n, int k,
                      float alpha, const float *a, int lda,
                      const float *b, int ldb, float beta, float *c, int ldc,
                      int tile, int threads) {
    if (m <= 0 || n <= 0) return;
    tile = tile == 64 ? 64 : 32;
    if (threads < 1) threads = 1;
#ifndef _OPENMP
    (void)threads;
#endif
    const int row_tiles = (m - 1) / tile + 1;
    const int column_tiles = (n - 1) / tile + 1;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads) if(threads > 1)
#endif
    for (int i = 0; i < row_tiles; ++i)
        for (int j = 0; j < column_tiles; ++j)
            output_tile(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
                        beta, c, ldc, tile, i * tile, j * tile);
}
