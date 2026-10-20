#ifndef KOWIKI_EXPERIMENT_KERNELS_H
#define KOWIKI_EXPERIMENT_KERNELS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int threads;
    int softmax_parallel;
    int reduction_parallel;
    int adam_parallel;
    int tiled_gemm; /* 0 = platform CBLAS; otherwise 32 or 64. */
    int fast_exp;
    float clip_norm;
} ExperimentKernelOptions;

/* Row-major matrices; trans_a/trans_b are 0 (normal) or 1 (transpose).
 * Output tiles are independent. Implementations may parallelize tiles once;
 * no outer OpenMP region surrounds BLAS or calls these kernels. */
void experiment_sgemm(int trans_a, int trans_b, int m, int n, int k,
                      float alpha, const float *a, int lda,
                      const float *b, int ldb, float beta,
                      float *c, int ldc, int tile, int threads);

/* Bias-add + stable mean cross-entropy. logits[B,V] is overwritten with
 * (softmax-onehot)/B; targets[B] are ordinary integer class IDs.
 * Untoggled serial mode preserves the original float32 loop order. */
float experiment_softmax(float *logits, const float *bias,
                         const uint32_t *targets, int batch, int vocab,
                         const ExperimentKernelOptions *options);

/* out[cols] becomes the sum of row-major values[rows,cols] over rows.
 * Untoggled serial mode uses the baseline row-then-column addition order. */
void experiment_bias_reduce(const float *values, int rows, int cols, float *out,
                             const ExperimentKernelOptions *options);

/* Returns raw L2 norm and writes clipped gradients in place plus scale.
 * Baseline uses double norm accumulation and min(1,max_norm/(norm+1e-6)). */
float experiment_clip_gradients(float *gradients, size_t count,
                                float *clipping_scale,
                                const ExperimentKernelOptions *options);

/* Dense float32 AdamW: lr=.001, decay=.01, betas=(.9,.999), eps=1e-8.
 * Bias corrections use double literals; step is one-based. */
void experiment_adamw(float *parameters, float *moment1, float *moment2,
                      const float *gradients, size_t count, int step,
                      const ExperimentKernelOptions *options);

/* Optional exp implementation is the static-inline fast_exp_approx(float)
 * from fast_exp.h. Only the fast_exp softmax branch includes/uses it. */

#endif
