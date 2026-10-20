# CPU optimization experiments

This directory is a separate experimental copy of the frozen C benchmark. The
original `../cpu_benchmark.c`, original trainer, Wiki arrays, vocabulary and
checkpoints are unchanged. It measures short Mac CPU computations on the actual
Wiki train arrays loaded completely into owned RAM. It does not run complete
training, save a model, or establish generation quality.

The common model is V=35,328 / C=8 / D=64 / H=128 / batch=128, float32, mean
full-softmax cross-entropy, embedding scatter-add, dense AdamW (lr=.001,
decay=.01, betas=.9/.999, epsilon=1e-8), and gradient norm limit 1. Untoggled
strict mode retains the baseline operation/order and platform CBLAS. Initial
weights use volatile intermediate stores to preserve identical bits even in
the separately labeled relaxed compiler build.

Build on this Mac after installing Homebrew `libomp`:

```sh
make
./benchmark --corpus ../../training-corpora/wiki-k35328 \
  --threads 4 --warmup-steps 2 --seconds 12
```

Apple Clang uses `-Xpreprocessor -fopenmp` and links `libomp` from
`/opt/homebrew/opt/libomp` (override `LIBOMP_PREFIX` if needed), alongside
Accelerate. The Makefile's Linux support links OpenBLAS/OpenMP, but remote
experiments are outside the current Mac-only scope. `CFLAGS` and `OUTPUT` may be
overridden for distinct strict `-O3`, native-target, or explicitly relaxed
fast-math builds. Only this experimental copy allows fast-math. Finite checks
use float bit patterns so they remain meaningful under relaxed compilation.
Compiler vectorization remarks can be collected with `-Rpass=loop-vectorize`;
they establish compiler behavior, not speed or numerical correctness.

Independent runtime switches:

| Switch | Experiment |
|---|---|
| `--softmax-parallel` | OpenMP over independent softmax rows |
| `--reduction-parallel` | Column-owned bias reductions and parallel gradient-norm reduction |
| `--adam-parallel` | OpenMP over independent dense AdamW parameter updates |
| `--tiled-gemm 32` / `64` | Custom packed cache-tiled SGEMM with independent output tiles |
| `--fast-exp` | Approximate exponential used only inside softmax |

`--softmax-omp`, `--reduction-omp`, `--adam-omp`, and `--tile` are aliases. Tiled
SGEMM replaces the six matrix multiplications, while untoggled cases use
Accelerate. `backend` identifies the linked BLAS; `matmul_backend` and
`kernel_options.tiled_gemm` identify the path actually selected. These custom
kernels are experiments, with no assumed improvement over the platform library.

The executable sets requested BLAS/OpenMP budgets before library computation,
disables dynamic/nested OpenMP teams, and uses passive waiting with
`KMP_BLOCKTIME=0`. No outer OpenMP region surrounds BLAS or another kernel.
`openmp.probe_team_size` is an actual team probe outside timing; it does not
measure Accelerate's internal workers. Do not equate the requested budget with
an observed BLAS worker count.

Elapsed mode accepts 10–30 seconds. A duration ends after the current update
finishes, so elapsed time can overshoot. JSON `steps` / `completed_steps` and
`target_prediction_events=steps*batch` record actual work; `requested_seconds`
records the budget. `config.steps=0` indicates elapsed mode, not zero completed
updates. `--internal-short --seconds 1` is reserved for bounded internal pilots.
Fixed `--steps N` remains available for correctness comparisons.

Two warm-up updates are the default. Parameters, Adam moments, step counter,
batch sequence and affine sampler reset afterward. Timing includes fresh RAM
window gathering, forward/backward, norm clipping and AdamW. It excludes array
loading/validation, initialization/checksums, warm-up/reset and the final loss
evaluation. `corpus.corpus_load_seconds` includes RAM allocation and validation;
OS file-cache state is uncontrolled. Owned buffer sizes are not process peak RSS.
The own C affine sampler visits each target once per pass, but its ordering
differs from the Python hash sampler. A short measured epoch fraction or an
extrapolated five-pass duration is not a measured full training run.

Correctness entry points use the same kernels and update path:

```sh
./benchmark --dump-check --threads 4
./benchmark --dump-check-large --threads 4 --clip-norm .05 \
  --softmax-parallel --reduction-parallel --adam-parallel --tiled-gemm 32 --fast-exp
./benchmark --corpus ../../training-corpora/wiki-k35328 --dump-corpus-check
```

The first fixture forces V11/C2/D3/H5/B4; the irregular fixture forces
V73/C3/D7/H67/B71, including matrix/tile tails and inner dimensions above a tile.
Both use seed42, one update and no warm-up, while retaining kernel switches and
the requested clip norm. Dump JSON contains the parameter layout, initial
parameters, inputs, raw/clipped gradients, norm/scale and updated parameters.
`--clip-norm .05` exercises active clipping. Independent checks must compare
these against a mathematical reference and check finite values outside the
fast-math executable before timed results are accepted.

The timing coordinator runs cases serially with a shared lock, checks frozen
source/data/config fingerprints, repeats each case, and keeps numerical errors
alongside speed. No quality inference follows from a faster approximate kernel.

## Recorded Mac run

`../results/cpu-optimizations-mac-real-wiki-20261018-12s.json` contains the nine
individual settings, two forward/reverse repeats each. A requested 12 seconds
resulted in 12.004832–16.662562 measured compute seconds, finishing the current
update. All 72 numerical fixtures and 1,656 comparisons passed. The final
independent review also checked the actual train arrays against their preserved
hashes after timing; this is separate from the runner's per-trial sample checks.

The host was under variable background load. Baseline throughput itself changed
by 4.34x, so these observations do not establish a precise optimization ranking.
No setting passed the two-repeat improvement screen, and a combined configuration
was not timed. Default O3 plus Accelerate remains the comparison reference.
`native` is CPU targeting, and `relaxed_simd` also relaxes floating-point rules;
these are not SIMD off/on comparisons, and neither rebuilds Accelerate.

Generate the compact Korean summary and comparison figure from the recorded data:

```sh
../../.venv-training/bin/python build_comparison_report.py
```

This command only reads reports and writes `../../reports/cpu-optimization-comparison.*`.
It does not run a model or benchmark. The text report includes reproduction scope,
numerical tolerances, reduction/AdamW explanations and verification limitations.

## Cumulative and CPU vector follow-up

The learning objective is to investigate CPU-only optimization one change at a
time; seven hours is a lecture reference, not a required deadline. Model, corpus,
batch and optimizer settings remain fixed in these additional experiments.

`cumulative_benchmark.c` preserves the original calculation and adds disjoint
caller-wall-time counters for RAM gathering, SGEMM, softmax, bias reduction,
gradient clipping and AdamW. Warm-up counters reset, and final evaluation is
excluded. Counters include scheduling waits and timer overhead. Normalize by
actual updates or target predictions when comparing phases.

`run_cumulative.py` tests five explicit configurations. Strict native+OpenMP,
strict native+OpenMP+approximate-exp, relaxed native+OpenMP and relaxed
native+OpenMP+tile64 are separate branches. Approximate-exp and global fast-math
are never combined in the declared cases.

`vector_vforce.c` and `vforce_benchmark.c` use the public CPU `vvexpf` API for
the additional vector-exponential variant. The scalar and vector controls use
the same native compiler options, with no global fast-math. There are no Metal,
MPS or GPU calls. Apple documents exact in-place operands for vForce, but does
not promise all libm return bits, denormal handling or floating-point flags.

`test_vforce_exp.c` independently checks scalar/vector softmax mean loss and
logits gradients over 1,456 cases, including V=35,328. This is separate from
the model-level forward/backward/AdamW fixture gates and from learned quality.

To reproduce short experiments into new, non-overwriting result files:

```sh
../../.venv-training/bin/python run_cumulative.py --seconds 20 --repeats 2 \
  --no-combined --output ../results/cumulative-retry.json
../../.venv-training/bin/python run_vector_exp.py --seconds 20 --repeats 2 \
  --no-combined --module-test-report vforce-exp-verification.json \
  --output ../results/vector-exp-retry.json
```

These coordinators share the exclusive timing lock and use unique executable
names. Each completed update uses a fresh real-Wiki batch from RAM. They perform
short training computations, not complete epochs or saved-model training.
`combined_case` in their raw output refers only to the disabled automatic
extra-combination screen; the explicit cumulative configurations were measured.

The latest matched native+OpenMP control reached 4,941.32 target predictions/s,
and adding CPU vector exponential reached 5,488.66/s (+11.08% in two short
observations). Their five-epoch arithmetic estimates are 34.92 and 31.44 hours.
These are neither measured full-epoch durations nor quality-optimal settings.
The original and follow-up generic/native baselines belong to different cohorts;
do not combine them as if all results were one controlled run.

Official CPU API references:

- https://developer.apple.com/accelerate/
- https://developer.apple.com/documentation/accelerate/vforce-library

## Norm SIMD follow-up

`vector_norm_simd.c` preserves the CPU vForce implementation and changes only
the double sum-of-squares reduction when `EXPERIMENT_NORM_SIMD=1`. The original
already uses vector products with ordered scalar accumulation; the candidate
uses vector partial accumulators followed by a final horizontal sum. This is
not an experiment with all SIMD disabled versus enabled. Compiler vector widths
describe generated loops and must not be equated with physical instruction lanes.

The module's 276 numerical cases include the full 6,883,968 gradient length.
Both whole-model builds also passed their eight independent fixture gates.
No global fast-math, model, batch, corpus or optimizer settings changed.

The three 20-second repeats did not reproduce a whole-update throughput gain.
Preserve the current vForce/OpenMP reference and retain norm SIMD as an
experimental candidate. New noisy epoch extrapolations do not replace the
previous estimate. `reports/norm-simd-review.txt` records all observations;
its norm/clipping phase includes scaling when active, not only the sum itself.

```sh
../../.venv-training/bin/python run_norm_simd.py --seconds 20 --repeats 3 \
  --no-combined --module-test-report norm-simd-verification.json \
  --output ../results/norm-simd-retry.json
```
