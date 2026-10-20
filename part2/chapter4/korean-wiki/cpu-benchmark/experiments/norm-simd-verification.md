# CPU double gradient-norm SIMD verification

Status: **PASS**. The experiment changes only gradient squared-norm summation in a preserved copy of `vector_vforce.c`. Float32 gradients, the double accumulator, clipping threshold and later scaling/AdamW loops remain unchanged. Both builds fix `EXPERIMENT_VFORCE_EXP=1`; candidate additionally uses `EXPERIMENT_NORM_SIMD=1`.

## Numerical scope

- 276 cases across 23 lengths, including 0, 1, 2, irregular tails and the actual 6,883,968 parameters.
- Worker budgets 1 and 4; clip thresholds 1 and 0.05; zero, random and mixed-magnitude inputs.
- Every clipped gradient compared: 82,671,444 against the original, plus 82,671,444 against independent compensated-double summation.
- Input range `-9.99999996e+11` to `9.99999996e+11`. Guard failures `0`; active-clipping cases `170`.
- Tolerances fixed before execution: norm and scale atol `1e-12`, rtol `2e-7`; gradients atol `1e-15`, rtol `4e-7`. Acceptance is `abs(actual-reference) <= atol + rtol*abs(reference)`.

- norm: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.
- scale: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.
- gradient: max absolute error `0`, max relative error `0`, tolerance violations `0`, nonfinite pairs `0`.

Default macro 0 produces the same preprocessed tokens as the original (whitespace normalized): `True`. Source SHA-256 hashes stayed unchanged: `True`.

## Actual compiler evidence

Strict `-O3 -mcpu=native -ffp-contract=off`, OpenMP enabled, global fast-math disabled. The original norm loop and the candidate are reported separately; other vectorized float loops are not counted as norm SIMD evidence.

**Important distinction:** the original also reports width 4/interleave 4. Its products are vectorized, while successive `llvm.vector.reduce.fadd` calls carry the preceding scalar accumulator in order. The candidate instead keeps independent `fadd <4 x double>` lane accumulators and reduces them at the end; it also has a two-double remainder path. This experiment changes ordered accumulation into a SIMD reduction; it does not turn an entirely non-vectorized loop into SIMD. IR vectorization width is not the number of double lanes in one physical M1 instruction.

Original norm:

```text
/Users/geuni/Desktop/hong-labs/part2/chapter4/korean-wiki/cpu-benchmark/experiments/vector_vforce.c:129:9: remark: vectorized loop (vectorization width: 4, interleaved count: 4) [-Rpass=loop-vectorize]
  129 |         for (size_t i = 0; i < count; ++i)
      |         ^
/Users/geuni/Desktop/hong-labs/part2/chapter4/korean-wiki/cpu-benchmark/experiments/vector_vforce.c:124:1: remark: vectorized loop (vectorization width: 4, interleaved count: 4) [-Rpass=loop-vectorize]
  124 | #pragma omp parallel for reduction(+:squares) schedule(static) num_threads(threads)
      | ^
```

Candidate norm:

```text
/Users/geuni/Desktop/hong-labs/part2/chapter4/korean-wiki/cpu-benchmark/experiments/vector_norm_simd.c:145:1: remark: vectorized loop (vectorization width: 4, interleaved count: 4) [-Rpass=loop-vectorize]
  145 | #pragma omp simd reduction(+:squares)
      | ^
/Users/geuni/Desktop/hong-labs/part2/chapter4/korean-wiki/cpu-benchmark/experiments/vector_norm_simd.c:136:1: remark: vectorized loop (vectorization width: 4, interleaved count: 4) [-Rpass=loop-vectorize]
  136 | #pragma omp parallel for simd reduction(+:squares) schedule(static) num_threads(threads)
      | ^
```

Candidate LLVM IR contains vector operations on **double**, including the horizontal reduction. Full instruction examples, source line ranges and compile commands are stored in `norm-simd-verification.json`.

## Limits

The numerical check took `2.547785` seconds; this is not a throughput benchmark. OpenMP SIMD reorders double additions, so bitwise equality is not guaranteed even if these inputs happen to match. Long-run optimizer trajectories, convergence, generation quality and five-epoch timing are unverified.
