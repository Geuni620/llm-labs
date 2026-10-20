#ifndef KOWIKI_FAST_EXP_H
#define KOWIKI_FAST_EXP_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Experimental expf replacement ONLY for stably shifted softmax logits.
 *
 * Fast domain: binary32 inputs in [-80, 0]. Outside that domain, including
 * NaN/infinities, use the platform expf unchanged. The fast domain's outputs
 * are normal floats: subnormal/underflow handling stays with libm.
 *
 * x = n*log(2) + r, |r| <= log(2)/2 (up to floating-point rounding).
 * Split log(2) limits cancellation when x is near -80. A sixth-degree Taylor
 * polynomial approximates exp(r); memcpy constructs 2^n without aliasing UB.
 * Do not combine this experiment with -ffast-math: comparison/domain and
 * non-finite behavior are part of the test contract. Rounding flags/errno
 * inside the fast domain are not guaranteed to reproduce libm behavior.
 */
static inline float fast_exp_approx(float x) {
#if FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && defined(UINT32_MAX)
    if (sizeof(float) != sizeof(uint32_t) || !(x >= -80.0f && x <= 0.0f)) {
        return expf(x);
    }

    /* All accepted x are non-positive; conversion truncates toward zero. */
    const int n = (int)(x * 1.4426950408889634074f - 0.5f);
    const float nf = (float)n;
    const float r = (x - nf * 0.693145751953125f)
                    - nf * 0.000001428606765330187045f;
    const float polynomial = 1.0f + r * (1.0f + r * (0.5f + r *
        (0.16666666666666666667f + r * (0.04166666666666666667f + r *
        (0.00833333333333333333f + r * 0.00138888888888888889f)))));
    const uint32_t scale_bits = (uint32_t)(n + 127) << 23;
    float scale;
    memcpy(&scale, &scale_bits, sizeof(scale_bits));
    return polynomial * scale;
#else
    /* Portable C11 fallback for non-IEEE-binary32 float implementations. */
    return expf(x);
#endif
}

#endif
