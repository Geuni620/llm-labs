# fast-expf numerical verification

- API: `static inline float fast_exp_approx(float x)` in `fast_exp.h`.
- Compiler: Apple LLVM 17.0.0 (clang-1700.4.4.1); architecture: arm64.
- Build: `cc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror cpu-benchmark/experiments/test_fast_exp.c -lm -o /tmp/kowiki-test-fast-exp`. No global `-ffast-math`.
- Fast domain: stable softmax shifts **[-80, 0]**. All other values return platform `expf`, including NaN, infinities and underflow/subnormal tails. The accepted-domain scale exponent is -115 through 0, so its scaling factor is a normal binary32 float.
- Method: round to the nearest multiple of ln(2), subtract split high/low ln(2), evaluate sixth-degree Taylor polynomial on approximately [-ln(2)/2, ln(2)/2], and construct 2^n with `memcpy`. There is no pointer aliasing or signed-shift reinterpretation.

## Observed error

The **2,001,157** in-domain tests comprise 2,000,001 dense samples, 1,150 range-reduction-boundary neighbors, and six zero/end-point/tiny-input cases. This is a dense numerical check, not exhaustive enumeration of every float32 value.

- Maximum relative error against double `exp`: **2.274072026e-07** at x=-16.2887592; required limit 1e-4.
- Maximum absolute error: **1.42696657e-07**.
- Maximum difference from platform `expf`: **3 ULP**.
- Out-of-domain fallback: **16** cases, **0** mismatches. Non-NaN returned bits must match platform `expf`; NaNs must remain NaN.

Softmax tests cover **120 rows / 733,100 values**, vocab sizes 1/7/37/257/1,025/35,328, and shifted input widths 0.0001/8/80/150. Both functions use identical sums and normalization for each comparison.

- Maximum probability error with float32 sums: **5.960464478e-08**.
- Maximum stable cross-entropy error with float32 sums: **4.768371582e-07**.
- Maximum probability error with double sums: **4.009557986e-08**.
- Maximum KL divergence after double normalization: **3.294836963e-15**.

## Scope

The real-arithmetic Taylor relative remainder bound on exactly [-ln(2)/2, ln(2)/2] is 2.383239657e-07; this excludes range-reduction and floating-point rounding error. The measured bound above includes those effects on this compiler/platform.

This test does not measure model throughput or establish learned quality. The shared runner will compare actual end-to-end training steps and gradients/updates separately. Approximation may change the training trajectory even when these local numerical errors are small. `errno` and floating-point exception flags inside the fast domain are not specified to match libm.

Full machine-readable evidence: `fast-exp-verification.json`.
