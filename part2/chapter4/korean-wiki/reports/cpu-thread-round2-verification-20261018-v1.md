# Fixed-work model verification

Status: PASS

CPU synthetic correctness only. No Wiki training, timings, convergence or quality comparison.

Ordinary values: abs error <= 2e-6 + 2e-4 * abs(reference).
Moment1: abs error <= 2e-9 + 2e-4 * abs(reference).
Moment2: abs error <= 2e-11 + 5e-4 * abs(reference).

An ordinary 2e-6 absolute allowance can hide incorrect zero moments. Separate smaller absolute limits retain float32 rounding room. C may form 1-.999f rather than Python's 1-.999; the approximately 1.29e-5 relative coefficient difference is covered by the declared moment2 rtol, not by loosening limits after results are seen. Explicit zero/nonzero checks supplement elementwise comparisons.

Each feature normally checks tiny/irregular shapes, 1/4 threads, clip 1/.05,
one ordinary update and four consecutive updates with NaN gradient poisoning.

| Feature | Cases | Passed | One step | Four updates | Poison |
|---|---:|---:|---|---|---|
| combined | 24 | 24 | True | True | True |

Total case count: 24.
Failed comparisons: 0.
Checksums bind inputs and executables; numerical comparisons determine correctness.
Unused rows in the final batch can still update from moments accumulated in earlier batches.
Only rows unused across every update must have exactly zero moments.

Legacy comparison: not_run.
