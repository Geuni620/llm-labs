# Apple CPU vForce exp numerical verification

Status: **PASS**. Compared scalar `expf` with in-place CPU `vvexpf` inside stable mean cross-entropy softmax, using identical biased logits, targets and float32 summation order.

- 1,456 shape/pattern/thread cases; vocabulary lengths 1–65,536, including the actual 35,328.
- Batch sizes 1 and 3; OpenMP worker budgets 1 and 4; 14 finite-input patterns.
- 8,656,984 scalar input positions across shape/pattern cases, counted before repeating thread budgets (not distinct numeric values); 17,313,968 gradient comparisons including both thread budgets.
- Input range: -1.00000002e+30 to 1.00000002e+30.
- Strict `-O3 -mcpu=native -ffp-contract=off`; global fast-math disabled.
- Tolerances declared in the test source before the run: loss atol `1e-5`, rtol `2e-6`; gradient atol `2e-7`, rtol `2e-6`. Every value must satisfy `abs(actual-reference) <= atol + rtol*abs(reference)`.

## Observed differences

- Loss: maximum absolute difference `2.38418579e-07`; maximum tolerance fraction `0.0171625161`; out-of-tolerance `0`; nonfinite pairs `0`.
- Gradient: maximum absolute difference `5.96046448e-08`; maximum tolerance fraction `0.0570404308`; out-of-tolerance `0`; nonfinite pairs `0`.
- Guard failures `0`; modified bias/target inputs `0`; invalid input values `0`.
- All source SHA-256 hashes stayed unchanged during the check: `True`.

## Scope limits

This is a standalone softmax numerical check. It does not validate end-to-end optimizer trajectories, long-run convergence, generation quality, throughput or an actual five-epoch run. Bitwise differences are permitted only within the predeclared numerical tolerances. Full counts, worst-case locations, build commands and source hashes are in `vforce-exp-verification.json`.
