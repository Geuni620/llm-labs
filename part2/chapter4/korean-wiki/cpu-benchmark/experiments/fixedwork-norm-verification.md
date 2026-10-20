# Fixed logical-block gradient norm verification, revision 2

Status: **PASS**. This revision extends numerical verification only. The norm kernel SHA-256 remains `30ee80ed77bbe3548eb2825604c029f6c958d54dc1c2505fddd6dd1cb6d8a865`; the updated test SHA-256 is `87fd2145dd742ce5c705ed28c22326d35b56229d756ab09cfbcd294b15223a1b`. Full revision-1 evidence is preserved inside `previous_verification_v1` in the JSON.

The kernel partitions gradients into fixed 4,096-element logical blocks, computes each with the same non-inlined double SIMD helper and merges partials in increasing block-index order. Scheduling does not select the logical grouping or merge order.

## Scratch contract

`fw_norm_scratch_count(n)` returns `n/4096 + (n%4096 != 0)` without overflow. Allocate this many doubles before timing. Actual model `n=6,883,968` requires **1,681 doubles = 13,448 bytes**, with a 2,688-element final block. Each partial is overwritten per call; no per-update allocation or zero initialization is needed. Zero-length inputs accept NULL gradient/scratch pointers. Insufficient workspace returns NaN norm/scale before touching gradients or scratch. Contract checks passed: `True`.

## Numerical scope

- 444 cases: the original 348 plus **96 deliberate threshold cases**.
- Lengths include 0/1/2, SIMD/block tails and actual model size. Zero/random/mixed inputs remain covered.
- 413,740,932 clipped gradients compared with the original, and the same count with independent compensated-double summation.
- Tolerances are unchanged from before the first run: norm/scale atol `1e-12`, rtol `2e-7`; gradients atol `1e-15`, rtol `4e-7`. All use `abs(actual-reference) <= atol + rtol*abs(reference)`.

- norm: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.
- scale: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.
- gradient: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.

## Deliberate threshold placement

Sparse and dense finite vectors at lengths 4,097 and 6,883,968 place the returned float32 norm at clip 1 or 0.05 and its immediate predecessor/successor. Additional targets straddle the actual `scale<1` transition. Because the implementation uses `clip/(norm+1e-6)`, `norm==clip` is already clipped; the effective transition lies slightly below clip.

Placement failures `0`; original returned-norm target misses `0`; fixed-block returned-norm target misses `0`. Clip-decision disagreements with the original `0`, with the compensated oracle `0`. Every boundary case records target norm, returned norms/scales and all three clip decisions in the JSON.

## Thread stability and compiler

Same-binary worker-1/worker-4 pairs: **222**. Different norm pairs `0`, scales `0`, gradients `0`, partial sums `0`. Guard failures `0`.

Strict `-O3 -mcpu=native -ffp-contract=off`, global fast-math disabled, float32 gradients and double products/sums. Rpass/LLVM evidence confirms the common non-inlined SIMD block helper; the final ordered merge has chained ordered reductions. Clipping writes and AdamW remain unchanged.

Verification execution took `8.384895` seconds. This is not a throughput benchmark or model-quality test. Bitwise stability applies only to the tested binary and floating-point environment. Different platforms/compilers/groupings may round differently. No actual training or five-epoch experiment ran.
