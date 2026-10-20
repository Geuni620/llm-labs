# C CPU compute comparison

This standalone C11 program measures short NPLM CPU computations. With
`--corpus`, it loads the existing Wiki training arrays into owned RAM and measures
fresh real-data updates. Without that flag it uses synthetic batches. It does
not run a full epoch, save a trained model, or measure generation quality. The
executable uses no Python runtime or GPU.

The model is embedding → concatenated context → linear → tanh → linear → full
stable softmax/cross-entropy, with float32 backward propagation (including
embedding scatter-add) and dense AdamW. Defaults are V=35,328, C=8, D=64, H=128,
batch=128, seed=42, four warm-up updates, and 64 measured updates. AdamW uses
lr=0.001, weight decay=0.01, betas=(0.9, 0.999), and epsilon=1e-8.
Global gradient L2 clipping uses max norm 1.0 and coefficient
`min(1, 1/(norm+1e-6))`, matching the retained Wiki trainer's clipping policy.
The retained Python trainer, corpus, and checkpoints remain unchanged.

Build on macOS with the Xcode command-line tools:

```sh
make
./cpu_benchmark --dump-check --threads 1
```

On Ubuntu, install a C compiler, `make`, `pkg-config`, and OpenBLAS development
headers/libraries (`build-essential pkg-config libopenblas-dev`), then run the
same commands. The Makefile links Apple Accelerate on macOS and OpenBLAS on
Linux. It defaults to `-O3` without fast-math; builds with fast-math are rejected.
No binary or machine result is committed by the build.

The current scope is a bounded Mac CPU measurement with the full train arrays
in RAM. No Ubuntu comparison or full training is started by these instructions:

```sh
./cpu_benchmark --corpus ../training-corpora/wiki-k35328 \
  --threads 4 --warmup-steps 4 --steps 64
```

The loader accepts bounded NumPy v1.0/v2.0 headers, little-endian uint16 ID arrays,
and the exact packed 28-byte paragraph metadata dtype. It reads complete payloads
with `fread` into owned memory and validates vocabulary bounds, contiguous
paragraph boundaries, and BOS/EOS context size before timing. Data-header offsets
are parsed; 128/192-byte offsets are not assumed. An in-RAM cumulative target
index supports gathering each context and target without crossing paragraphs.
The reader opens source files read-only. It validates shape/dtype/boundaries, not
the cryptographic corpus manifest; retain the independent artifact verification.

`corpus.corpus_load_seconds` includes file reading, RAM allocation, and corpus
validation. It is separate from timed updates. The OS file-cache state is
uncontrolled, so this is not a cold-SSD I/O benchmark. `corpus.owned_ram_bytes`
counts the ID payload, metadata payload, and cumulative index; model buffers are
reported separately in `workspace_bytes` and neither figure is process peak RSS.

Real-data mode uses its own deterministic affine permutation over target windows,
with a coprime multiplier and offset chosen from the C PRNG for each pass. Every
target is visited once per pass; this is not a uniform shuffle and its order
differs from the retained Python hash sampler. The counter and permutation stream
reset after warm-up. Each timed update gathers a fresh mini-batch from RAM and
includes gather, forward, backward, gradient clipping, and AdamW. It is a short
sample of work, not an observed full-epoch or five-epoch duration. Any estimate of
that duration from throughput must be labeled as an extrapolation.

For loader/window inspection without any BLAS computation or weight updates:

```sh
./cpu_benchmark --corpus ../training-corpora/wiki-k35328 \
  --dump-corpus-check --threads 1
```

This reports up to four actual window indices, paragraph indices, contexts, and
targets for independent comparison, as well as a checksum of the first complete
mini-batch. The JSON kind is `REAL_WIKI_C_CORPUS_CHECK`.

Synthetic compute mode remains available for mathematical checks and future
explicitly requested device comparisons:

```sh
./cpu_benchmark --threads 4 --warmup-steps 4 --steps 64
```

Thread-limit environment variables are set before the first BLAS call. Linux
also applies the OpenBLAS thread setter and reports its configured thread count.
For Accelerate, `requested_threads` is the requested budget; the program does not
claim to have measured the number of workers used by macOS. Setting
`VECLIB_MAXIMUM_THREADS` before launching the executable is also supported.

Initialization and the four synthetic mini-batches use deterministic xorshift32.
Weights are uniform in [-0.01, 0.01); seed zero maps to 0x9e3779b9. Warm-up is
followed by a full reset of parameters, Adam moments, step counter, and batch
sequence before timing. Check matching initial parameter/input checksums when
comparing machines. Checksums use FNV-1a over explicitly little-endian 32-bit
float/integer representations. BLAS reduction orders may still cause small
cross-platform floating-point differences after updates.

One JSON object is written to stdout. It reports compiler/build flags/backend,
configuration, checksums, update/target counts, monotonic elapsed time, throughput,
median update time, and finite losses. Allocation, initialization, warm-up, reset,
corpus loading, and the final post-update loss evaluation are excluded from the timed interval.
`last_timed_loss` is measured before the last update; `final_loss` is evaluated
after the updates on that same mini-batch. Synthetic batches are reused and do
not represent dataset epochs. JSON kinds distinguish `REAL_WIKI_C_MICROBENCHMARK`
from `SYNTHETIC_C_NPLM_BENCHMARK`. Neither establishes trained model quality.

CLI options: `--vocab-size`, `--context-size`, `--embedding-dim`, `--hidden-dim`,
`--batch-size`, `--threads`, `--seed`, `--warmup-steps`, `--steps`, `--corpus`,
`--dump-corpus-check`, `--dump-check`, and `--help`. Supported bounds are
V=2..65,536, C=1..32, D=1..512, H=1..1,024,
batch=1..1,024, threads=1..64, seed=uint32, warm-up=0..256, steps=1..4,096.
Model working buffers must fit 512 MiB and logits must fit 256 MiB. Individual
NumPy payloads are bounded at 2 GiB, and real-data target counts must be below
2^31 to keep the affine product within uint64.

`--dump-check` forces V=11/C=2/D=3/H=5/batch=4/seed=42/warm-up=0/steps=1 and
uses the same forward/backward/AdamW code. Its JSON contains `initial_params`,
`contexts`, `targets`, pre-update `loss`, raw `grads`, clipped `optimizer_grads`,
`gradient_norm`, `clipping_scale`, and `updated_params`. Flattened
parameters have the explicit `parameter_layout`: embedding [V,D], hidden weight
[H,C*D], hidden bias [H], output weight [V,H], output bias [V]. The first two
context positions repeat a token, while some embedding rows remain unused in
the tiny batch, exercising scatter-add and zero-gradient weight decay. A separate
independent Python/Torch checker can consume this tiny JSON; the benchmark
executable itself has no Python dependency.

Run the independent checker from this directory with:

```sh
../.venv-training/bin/python check_correctness.py --executable ./cpu_benchmark \
  --report ../reports/cpu-benchmark-correctness.json
```

This checks only the tiny one-update fixture against PyTorch and a separate
first-step AdamW calculation. It checks mean loss, all 134 parameter gradients
and updates, repeated-token scatter-add, and decay for untouched embedding rows.
A numerical pass does not establish training quality or which device is faster.
