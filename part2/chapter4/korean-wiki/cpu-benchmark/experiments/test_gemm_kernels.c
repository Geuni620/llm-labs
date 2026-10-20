/* Independent small SGEMM correctness checks; no performance measurements. */
#include "experiment_kernels.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

static uint32_t rng = 42;
static float random_value(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

static int test_case(int m, int n, int k, int trans_a, int trans_b,
                     float alpha, float beta, int tile, int threads,
                     double *max_abs, double *max_scaled) {
    const int a_rows = trans_a ? k : m;
    const int b_rows = trans_b ? n : k;
    const int lda = (trans_a ? m : k) + 3;
    const int ldb = (trans_b ? k : n) + 5;
    const int ldc = n + 7;
    const size_t a_items = (size_t)(a_rows > 0 ? a_rows : 1) * lda;
    const size_t b_items = (size_t)(b_rows > 0 ? b_rows : 1) * ldb;
    const size_t c_items = (size_t)m * ldc;
    float *a = malloc(a_items * sizeof(float));
    float *b = malloc(b_items * sizeof(float));
    float *actual = malloc(c_items * sizeof(float));
    float *expected = malloc(c_items * sizeof(float));
    if (!a || !b || !actual || !expected) return 1;
    for (size_t i = 0; i < a_items; ++i) a[i] = random_value();
    for (size_t i = 0; i < b_items; ++i) b[i] = random_value();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j)
            actual[(size_t)i * ldc + j] = j < n ? random_value() : 9182.5f;
    if (beta == 0.0f)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                actual[(size_t)i * ldc + j] = NAN;
    memcpy(expected, actual, c_items * sizeof(float));
    cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                a, lda, b, ldb, beta, expected, ldc);
    experiment_sgemm(trans_a, trans_b, m, n, k, alpha,
                      a, lda, b, ldb, beta, actual, ldc, tile, threads);
    int failed = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < ldc; ++j) {
            const size_t at = (size_t)i * ldc + j;
            if (j >= n) {
                if (actual[at] != 9182.5f) failed = 1;
                continue;
            }
            const double error = fabs((double)actual[at] - expected[at]);
            /* Absolute near zero, relative for large reference values. */
            const double scale = fmax(1.0, fabs((double)expected[at]));
            if (error > *max_abs) *max_abs = error;
            if (error / scale > *max_scaled) *max_scaled = error / scale;
            if (!isfinite(actual[at]) || error > 3e-5 + 3e-5 * fabs(expected[at]))
                failed = 1;
        }
    if (failed)
        fprintf(stderr, "SGEMM mismatch: M%d N%d K%d trans%d%d alpha%.2g beta%.2g tile%d threads%d\n",
                m, n, k, trans_a, trans_b, alpha, beta, tile, threads);
    free(a); free(b); free(actual); free(expected);
    return failed;
}

int main(int argc, char **argv) {
    int quick = argc == 2 && !strcmp(argv[1], "--quick");
    if (argc > 1 && !quick) return EXIT_FAILURE;
    const int shapes[][3] = {
        {1, 1, 1}, {3, 7, 5}, {31, 33, 65}, {33, 31, 63},
        {63, 65, 33}, {65, 63, 67}, {67, 73, 71}, {71, 67, 73},
        {128, 73, 127}, {73, 128, 65}, {3, 5, 0}
    };
    const float coefficients[][2] = {{1, 0}, {0.75f, -0.25f}, {-1.25f, 0.5f}, {0, 0}, {0, 1.5f}};
    const int thread_options[] = {1, 4};
    double max_abs = 0, max_scaled = 0;
    int failures = 0, cases = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); ++s) {
        if (quick && s != 6) continue;
        for (int ta = 0; ta <= 1; ++ta)
            for (int tb = 0; tb <= 1; ++tb)
                for (int tile = 32; tile <= 64; tile += 32)
                    for (size_t t = 0; t < sizeof(thread_options) / sizeof(*thread_options); ++t)
                        for (size_t c = 0; c < sizeof(coefficients) / sizeof(*coefficients); ++c) {
                            failures += test_case(shapes[s][0], shapes[s][1], shapes[s][2], ta, tb,
                                                  coefficients[c][0], coefficients[c][1], tile,
                                                  thread_options[t], &max_abs, &max_scaled);
                            ++cases;
                        }
    }
    /* alpha=0 must not read A/B, and empty output must not dereference C. */
    float guarded[4] = {NAN, NAN, NAN, NAN};
    experiment_sgemm(0, 0, 2, 2, 3, 0, NULL, 3, NULL, 2, 0, guarded, 2, 32, 4);
    for (int i = 0; i < 4; ++i) if (guarded[i] != 0.0f) ++failures;
    experiment_sgemm(0, 0, 0, 4, 3, 1, NULL, 3, NULL, 4, 0, NULL, 4, 64, 4);
    experiment_sgemm(0, 0, 4, 0, 3, 1, NULL, 3, NULL, 1, 0, NULL, 1, 64, 4);
    printf("{\"kind\":\"SMALL_SGEMM_CORRECTNESS\",\"cases\":%d,\"special_cases\":3,"
           "\"failures\":%d,\"max_absolute_error\":%.17g,\"max_scaled_error\":%.17g,"
           "\"openmp_enabled\":%s}\n", cases, failures, max_abs, max_scaled,
#ifdef _OPENMP
           "true"
#else
           "false"
#endif
    );
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
