#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include "experiment_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define BACKEND "Apple Accelerate / CBLAS"
#else
#include <cblas.h>
extern void openblas_set_num_threads(int);
extern int openblas_get_num_threads(void);
#define BACKEND "OpenBLAS / CBLAS"
#endif

#ifndef BENCH_BUILD_FLAGS
#define BENCH_BUILD_FLAGS "not recorded"
#endif
_Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24, "Require IEEE binary32 floats");

enum { BATCHES = 4 };
typedef struct {
    int vocab, context, dim, hidden, batch, threads, warmup, steps, dump;
    uint32_t seed;
    const char *corpus_path;
    int corpus_check;
    ExperimentKernelOptions kernels;
    double seconds;
    int internal_short;
} Config;
typedef struct {
    size_t embedding, hidden_weight, hidden_bias, output_weight, output_bias, count;
} Layout;
typedef struct {
    Config config;
    Layout layout;
    float *parameters, *initial, *gradients, *moment1, *moment2;
    float *input, *hidden, *logits, *dhidden, *dinput;
    uint32_t *contexts, *targets;
    uint64_t initial_checksum, input_checksum;
    size_t workspace_bytes;
    float *check_raw_gradients;
    float gradient_norm, clipping_scale;
} Model;
typedef struct {
    unsigned char *data;
    size_t count, bytes, header_bytes;
} Npy;
typedef struct {
    uint64_t window, paragraph;
    uint32_t contexts[32], target;
} Sample;
typedef struct {
    Npy ids, paragraphs;
    uint64_t *cumulative, windows;
    double load_seconds;
    Sample samples[4];
    int sample_count;
} Corpus;
typedef struct {
    uint32_t random;
    uint64_t position, multiplier, shift, consumed;
} Sampler;
static int omp_probe_team_size = 1;
enum { PH_GATHER, PH_GEMM, PH_SOFTMAX, PH_BIAS, PH_CLIP, PH_ADAM, PH_COUNT };
static double phase_seconds[PH_COUNT];
static uint64_t phase_calls[PH_COUNT];
static const char *phase_names[PH_COUNT] = {
    "ram_gather", "matrix_multiply", "softmax", "bias_reduce", "gradient_clip", "adamw"
};

/* Bit tests remain valid even in the explicitly experimental fast-math build,
 * where compilers are otherwise allowed to assume finite floating values. */
static int finite_float(float value) {
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static void fail(const char *message) {
    fprintf(stderr, "cpu_benchmark: %s\n", message);
    exit(EXIT_FAILURE);
}
static void *allocate(size_t count, size_t size) {
    void *value = calloc(count, size);
    if (!value) fail("allocation failed");
    return value;
}
static uint32_t random32(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return *state = value;
}
static uint64_t hash32(uint64_t hash, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash ^= (value >> shift) & 255u;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
static uint64_t parameter_checksum(const float *values, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof(bits));
        hash = hash32(hash, bits);
    }
    return hash;
}
static double seconds_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) fail("monotonic clock failed");
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}
/* Only the calling main thread updates these counters. Each measured kernel
 * completes its internal workers before returning; no nested accounting. */
static void record_phase(int phase, double begun) {
    phase_seconds[phase] += seconds_now() - begun;
    ++phase_calls[phase];
}
static uint64_t number(const char *value) {
    if (!*value || *value == '-') fail("arguments must be nonnegative integers");
    errno = 0;
    char *end;
    unsigned long long result = strtoull(value, &end, 10);
    if (errno || *end) fail("invalid integer argument");
    return (uint64_t)result;
}
static void usage(void) {
    puts("Standalone synthetic C/CBLAS CPU NPLM compute benchmark (no Wiki corpus/training).\n"
         "Options: --vocab-size N --context-size N --embedding-dim N --hidden-dim N\n"
         "         --batch-size N --threads N --seed N --warmup-steps N --steps N\n"
         "         --corpus PATH (RAM-resident real Wiki arrays)\n"
         "         --dump-corpus-check (load and sample only, requires --corpus)\n"
         "         --seconds 10..30 (elapsed mode) --clip-norm FLOAT\n"
         "         --softmax-parallel --reduction-parallel --adam-parallel\n"
         "         --tiled-gemm 32|64 --fast-exp\n"
         "         --dump-check (tiny) --dump-check-large (irregular); one update");
}
static Config arguments(int argc, char **argv) {
    Config c = {.vocab=35328,.context=8,.dim=64,.hidden=128,.batch=128,
                .threads=4,.warmup=2,.steps=64,.seed=42,
                .kernels={.threads=4,.clip_norm=1.0f}};
    for (int i = 1; i < argc; ++i) {
        const char *name = argv[i];
        if (!strcmp(name, "--help")) { usage(); exit(EXIT_SUCCESS); }
        if (!strcmp(name, "--dump-check")) { c.dump = 1; continue; }
        if (!strcmp(name, "--dump-check-large") || !strcmp(name,"--dump-irregular-check")) { c.dump=2; continue; }
        if (!strcmp(name, "--dump-corpus-check")) { c.corpus_check = 1; continue; }
        if (!strcmp(name,"--softmax-parallel") || !strcmp(name,"--softmax-omp")) { c.kernels.softmax_parallel=1; continue; }
        if (!strcmp(name,"--reduction-parallel") || !strcmp(name,"--reduction-omp")) { c.kernels.reduction_parallel=1; continue; }
        if (!strcmp(name,"--adam-parallel") || !strcmp(name,"--adam-omp")) { c.kernels.adam_parallel=1; continue; }
        if (!strcmp(name,"--fast-exp")) { c.kernels.fast_exp=1; continue; }
        if (!strcmp(name,"--tiled-gemm32") || !strcmp(name,"--tile32")) { c.kernels.tiled_gemm=32; continue; }
        if (!strcmp(name,"--tiled-gemm64") || !strcmp(name,"--tile64")) { c.kernels.tiled_gemm=64; continue; }
        if (!strcmp(name,"--internal-short")) { c.internal_short=1; continue; }
        if (i + 1 == argc) fail("option is missing its value");
        if (!strcmp(name, "--corpus")) { c.corpus_path = argv[++i]; continue; }
        if (!strcmp(name,"--seconds") || !strcmp(name,"--clip-norm")) {
            char *end; errno=0;
            double value=strtod(argv[++i],&end);
            if (errno || end==argv[i] || *end || !finite_float((float)value) || value<=0)
                fail("invalid duration or clipping norm");
            if (!strcmp(name,"--seconds")) c.seconds=value; else c.kernels.clip_norm=(float)value;
            continue;
        }
        uint64_t value = number(argv[++i]);
        if (!strcmp(name, "--seed")) {
            if (value > UINT32_MAX) fail("seed exceeds uint32 range");
            c.seed = (uint32_t)value;
            continue;
        }
        if (value > INT_MAX) fail("integer argument exceeds supported range");
        int n = (int)value;
        if (!strcmp(name, "--vocab-size")) c.vocab = n;
        else if (!strcmp(name, "--context-size")) c.context = n;
        else if (!strcmp(name, "--embedding-dim")) c.dim = n;
        else if (!strcmp(name, "--hidden-dim")) c.hidden = n;
        else if (!strcmp(name, "--batch-size")) c.batch = n;
        else if (!strcmp(name, "--threads")) c.threads = n;
        else if (!strcmp(name, "--warmup-steps")) c.warmup = n;
        else if (!strcmp(name, "--steps")) c.steps = n;
        else if (!strcmp(name,"--tiled-gemm") || !strcmp(name,"--tile")) c.kernels.tiled_gemm=n;
        else fail("unknown option; use --help");
    }
    if (c.dump && (c.corpus_path || c.corpus_check)) fail("tiny check and corpus mode are separate");
    if (c.corpus_check && !c.corpus_path) fail("--dump-corpus-check requires --corpus PATH");
    if (c.dump) {
        c.vocab = c.dump==1 ? 11 : 73; c.context = c.dump==1 ? 2 : 3;
        c.dim = c.dump==1 ? 3 : 7; c.hidden = c.dump==1 ? 5 : 67;
        c.batch = c.dump==1 ? 4 : 71; c.seed=42; c.warmup=0; c.steps=1; c.seconds=0;
    }
    if (c.seconds && (c.seconds>(double)30 || c.seconds<(c.internal_short ? 1.0 : 10.0)))
        fail("duration must be 10..30 seconds (internal pilot may use 1..30)");
    if (c.kernels.clip_norm<1e-6f || c.kernels.clip_norm>100.0f
        || (c.kernels.tiled_gemm!=0 && c.kernels.tiled_gemm!=32 && c.kernels.tiled_gemm!=64))
        fail("invalid clipping norm or tile size");
    if (c.vocab < 2 || c.vocab > 65536 || c.context < 1 || c.context > 32
        || c.dim < 1 || c.dim > 512 || c.hidden < 1 || c.hidden > 1024
        || c.batch < 1 || c.batch > 1024 || c.threads < 1 || c.threads > 64
        || c.warmup < 0 || c.warmup > 256 || c.steps < 1 || c.steps > 4096)
        fail("configuration exceeds bounded supported ranges; see README");
    c.kernels.threads=c.threads;
    return c;
}
static void thread_budget(int threads) {
    char value[32];
    snprintf(value, sizeof(value), "%d", threads);
    const char *names[] = {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "MKL_NUM_THREADS"};
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); ++i)
        if (setenv(names[i], value, 1)) fail("could not set thread budget environment");
    if (setenv("OMP_WAIT_POLICY","PASSIVE",1) || setenv("KMP_BLOCKTIME","0",1)
        || setenv("OMP_DYNAMIC","FALSE",1) || setenv("OMP_MAX_ACTIVE_LEVELS","1",1))
        fail("could not set non-spinning OpenMP policy");
#ifndef __APPLE__
    openblas_set_num_threads(threads);
#endif
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_max_active_levels(1);
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        omp_probe_team_size=omp_get_num_threads();
    }
#endif
}
static Model create_model(Config c) {
    Model m = {0};
    m.config = c;
    Layout *l = &m.layout;
    size_t cd = (size_t)c.context * c.dim, bh = (size_t)c.batch * c.hidden;
    l->embedding = 0;
    l->hidden_weight = (size_t)c.vocab * c.dim;
    l->hidden_bias = l->hidden_weight + (size_t)c.hidden * cd;
    l->output_weight = l->hidden_bias + c.hidden;
    l->output_bias = l->output_weight + (size_t)c.vocab * c.hidden;
    l->count = l->output_bias + c.vocab;
    size_t activations = 2 * (size_t)c.batch * cd + 2 * bh + (size_t)c.batch * c.vocab;
    size_t input_items = (size_t)BATCHES * c.batch * (c.context + 1);
    m.workspace_bytes = (5 * l->count + activations) * sizeof(float) + input_items * sizeof(uint32_t);
    if (m.workspace_bytes > (size_t)512 * 1024 * 1024
        || (size_t)c.batch * c.vocab * sizeof(float) > (size_t)256 * 1024 * 1024)
        fail("workspace exceeds 512 MiB or logits exceed 256 MiB");
    m.parameters = allocate(l->count, sizeof(float));
    m.initial = allocate(l->count, sizeof(float));
    m.gradients = allocate(l->count, sizeof(float));
    if (c.dump) m.check_raw_gradients = allocate(l->count, sizeof(float));
    m.moment1 = allocate(l->count, sizeof(float));
    m.moment2 = allocate(l->count, sizeof(float));
    m.input = allocate((size_t)c.batch * cd, sizeof(float));
    m.hidden = allocate(bh, sizeof(float));
    m.logits = allocate((size_t)c.batch * c.vocab, sizeof(float));
    m.dhidden = allocate(bh, sizeof(float));
    m.dinput = allocate((size_t)c.batch * cd, sizeof(float));
    m.contexts = allocate((size_t)BATCHES * c.batch * c.context, sizeof(uint32_t));
    m.targets = allocate((size_t)BATCHES * c.batch, sizeof(uint32_t));
    uint32_t state = c.seed ? c.seed : UINT32_C(0x9e3779b9);
    for (size_t i = 0; i < l->count; ++i) {
        float uniform = (float)(random32(&state) >> 8) * (1.0f / 16777216.0f);
        /* Preserve initial bits across strict and explicitly relaxed builds. */
        volatile float scaled=uniform*2.0f;
        volatile float centered=scaled-1.0f;
        m.initial[i] = centered * 0.01f;
    }
    memcpy(m.parameters, m.initial, l->count * sizeof(float));
    for (int batch = 0; batch < BATCHES; ++batch) {
        size_t offset = (size_t)batch * c.batch * c.context;
        for (size_t i = 0; i < (size_t)c.batch * c.context; ++i)
            m.contexts[offset + i] = random32(&state) % (uint32_t)c.vocab;
        if ((size_t)c.batch * c.context >= 2)
            m.contexts[offset] = m.contexts[offset + 1] = 1;
        for (int i = 0; i < c.batch; ++i)
            m.targets[(size_t)batch * c.batch + i] = random32(&state) % (uint32_t)c.vocab;
    }
    m.initial_checksum = parameter_checksum(m.initial, l->count);
    m.input_checksum = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < (size_t)BATCHES * c.batch * c.context; ++i)
        m.input_checksum = hash32(m.input_checksum, m.contexts[i]);
    for (size_t i = 0; i < (size_t)BATCHES * c.batch; ++i)
        m.input_checksum = hash32(m.input_checksum, m.targets[i]);
    return m;
}

static uint32_t little32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t little64(const unsigned char *p) {
    return (uint64_t)little32(p) | (uint64_t)little32(p + 4) << 32;
}
static const char *spaces(const char *p) {
    while (isspace((unsigned char)*p)) ++p;
    return p;
}
static const char *header_field(const char *header, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "'%s'", key);
    const char *p = strstr(header, pattern);
    if (!p) { snprintf(pattern, sizeof(pattern), "\"%s\"", key); p = strstr(header, pattern); }
    if (!p) fail("missing NumPy header field");
    p = spaces(p + strlen(pattern));
    if (*p != ':') fail("invalid NumPy header field");
    return spaces(p + 1);
}
static Npy load_npy(const char *directory, const char *name, size_t itemsize, int metadata) {
    size_t path_length = strlen(directory) + strlen(name) + 2;
    char *path = allocate(path_length, 1);
    snprintf(path, path_length, "%s/%s", directory, name);
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "cpu_benchmark: cannot open %s\n", path); exit(EXIT_FAILURE); }
    free(path);
    unsigned char magic[8], length_bytes[4] = {0};
    if (fread(magic, 1, 8, file) != 8 || memcmp(magic, "\x93NUMPY", 6)
        || magic[7] != 0 || (magic[6] != 1 && magic[6] != 2))
        fail("require NumPy version 1.0 or 2.0");
    size_t length_size = magic[6] == 1 ? 2 : 4;
    if (fread(length_bytes, 1, length_size, file) != length_size) fail("truncated NumPy header");
    size_t header_size = little32(length_bytes);
    if (!header_size || header_size > 1048576) fail("NumPy header exceeds bound");
    char *header = allocate(header_size + 1, 1);
    if (fread(header, 1, header_size, file) != header_size) fail("truncated NumPy header");
    if (strncmp(header_field(header, "fortran_order"), "False", 5)) fail("require C-order NumPy array");
    const char *descr = header_field(header, "descr");
    if (!metadata) {
        if (!((*descr == '\'' || *descr == '"') && !strncmp(descr + 1, "<u2", 3) && descr[4] == *descr))
            fail("integer array must have little-endian uint16 dtype");
    } else {
        if (*descr != '[') fail("paragraph metadata must be a structured dtype");
        char normalized[512]; size_t used = 0;
        const char *p = descr;
        for (; *p && *p != ']'; ++p) {
            if (!isspace((unsigned char)*p) && *p != '\'' && *p != '"') {
                if (used + 2 >= sizeof(normalized)) fail("paragraph dtype exceeds bound");
                normalized[used++] = *p;
            }
        }
        if (*p != ']') fail("unterminated paragraph dtype");
        normalized[used++] = ']'; normalized[used] = '\0';
        if (strcmp(normalized, "[(start,<u8),(length,<u4),(subset_key,<u4),(sentences,<u4),(doc_key,<u8)]"))
            fail("paragraph metadata must match the packed 28-byte corpus dtype");
    }
    const char *shape = header_field(header, "shape");
    if (*shape++ != '(') fail("invalid NumPy shape");
    shape = spaces(shape);
    if (!isdigit((unsigned char)*shape)) fail("invalid NumPy array length");
    errno = 0; char *end;
    unsigned long long count = strtoull(shape, &end, 10);
    if (errno || !count || count > SIZE_MAX / itemsize) fail("NumPy array length exceeds bound");
    const char *tail = spaces(end);
    if (*tail == ',') tail = spaces(tail + 1);
    if (*tail != ')') fail("require one-dimensional NumPy arrays");
    Npy array = {NULL, (size_t)count, (size_t)count * itemsize, 8 + length_size + header_size};
    if (array.bytes > (size_t)2 * 1024 * 1024 * 1024) fail("NumPy payload exceeds 2 GiB bound");
    struct stat info;
    if (fstat(fileno(file), &info) || (uint64_t)info.st_size != (uint64_t)array.header_bytes + array.bytes)
        fail("NumPy shape does not match file length");
    array.data = allocate(array.bytes, 1);
    if (fread(array.data, 1, array.bytes, file) != array.bytes || fgetc(file) != EOF || ferror(file))
        fail("truncated or changing NumPy payload");
    if (fclose(file)) fail("closing NumPy source failed");
    free(header);
    return array;
}
static Corpus load_corpus(Config c) {
    double begun = seconds_now();
    Corpus corpus = {0};
    corpus.ids = load_npy(c.corpus_path, "train.ids.npy", 2, 0);
    corpus.paragraphs = load_npy(c.corpus_path, "train.paragraphs.npy", 28, 1);
    corpus.cumulative = allocate(corpus.paragraphs.count, sizeof(uint64_t));
    const uint16_t endian = 1;
    if (!*(const unsigned char *)&endian) {
        for (size_t i = 0; i < corpus.ids.count; ++i) {
            unsigned char *p = corpus.ids.data + i * 2;
            unsigned char first = p[0]; p[0] = p[1]; p[1] = first;
        }
    }
    const uint16_t *ids = (const uint16_t *)corpus.ids.data;
    uint64_t bos = 0, eos = 0, end_previous = 0;
    for (size_t i = 0; i < corpus.ids.count; ++i) {
        if (ids[i] < 2 || ids[i] >= c.vocab) fail("corpus token ID outside retained vocabulary");
        bos += ids[i] == 2; eos += ids[i] == 3;
    }
    for (size_t i = 0; i < corpus.paragraphs.count; ++i) {
        const unsigned char *row = corpus.paragraphs.data + i * 28;
        uint64_t start = little64(row), length = little32(row + 8);
        if (start != end_previous || length <= (uint64_t)c.context
            || start > corpus.ids.count || length > corpus.ids.count - start)
            fail("invalid or noncontiguous paragraph boundaries");
        for (int j = 0; j < c.context; ++j)
            if (ids[start + j] != 2) fail("paragraph BOS context mismatch");
        if (ids[start + length - 1] != 3) fail("paragraph EOS mismatch");
        end_previous = start + length;
        corpus.windows += length - (uint64_t)c.context;
        corpus.cumulative[i] = corpus.windows;
    }
    if (end_previous != corpus.ids.count || bos != corpus.paragraphs.count * (uint64_t)c.context
        || eos != corpus.paragraphs.count || corpus.windows >= UINT64_C(2147483648))
        fail("corpus boundaries/context or supported target-count bound mismatch");
    corpus.load_seconds = seconds_now() - begun;
    return corpus;
}
static uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) { uint64_t next = a % b; a = b; b = next; }
    return a;
}
static void next_pass(Sampler *sampler, uint64_t windows) {
    sampler->position = 0;
    sampler->multiplier = random32(&sampler->random) % windows;
    if (!sampler->multiplier) sampler->multiplier = 1;
    while (gcd64(sampler->multiplier, windows) != 1) {
        sampler->multiplier = (sampler->multiplier + 1) % windows;
        if (!sampler->multiplier) sampler->multiplier = 1;
    }
    sampler->shift = random32(&sampler->random) % windows;
}
static Sampler sampler_start(Config c, uint64_t windows) {
    Sampler sampler = {0};
    sampler.random = c.seed ^ UINT32_C(0x85ebca6b);
    if (!sampler.random) sampler.random = UINT32_C(0x9e3779b9);
    next_pass(&sampler, windows);
    return sampler;
}
static void gather_real(Model *m, Corpus *corpus, Sampler *sampler, int capture) {
    double phase_begun = seconds_now();
    Config c = m->config;
    const uint16_t *ids = (const uint16_t *)corpus->ids.data;
    for (int b = 0; b < c.batch; ++b) {
        if (sampler->position == corpus->windows) next_pass(sampler, corpus->windows);
        uint64_t window = (sampler->multiplier * sampler->position + sampler->shift) % corpus->windows;
        ++sampler->position; ++sampler->consumed;
        size_t low = 0, high = corpus->paragraphs.count;
        while (low < high) {
            size_t middle = low + (high - low) / 2;
            if (corpus->cumulative[middle] <= window) low = middle + 1; else high = middle;
        }
        uint64_t before = low ? corpus->cumulative[low - 1] : 0;
        uint64_t start = little64(corpus->paragraphs.data + low * 28) + window - before;
        for (int j = 0; j < c.context; ++j) m->contexts[(size_t)b*c.context+j] = ids[start+j];
        m->targets[b] = ids[start + c.context];
        if (capture && b < 4) {
            Sample *sample = &corpus->samples[b];
            sample->window = window; sample->paragraph = low; sample->target = m->targets[b];
            for (int j = 0; j < c.context; ++j) sample->contexts[j] = m->contexts[(size_t)b*c.context+j];
            corpus->sample_count = b + 1;
        }
    }
    if (capture) {
        m->input_checksum = UINT64_C(14695981039346656037);
        for (int i = 0; i < c.batch*c.context; ++i) m->input_checksum = hash32(m->input_checksum, m->contexts[i]);
        for (int i = 0; i < c.batch; ++i) m->input_checksum = hash32(m->input_checksum, m->targets[i]);
    }
    record_phase(PH_GATHER, phase_begun);
}
static void multiply(Model *m,int trans_a,int trans_b,int rows,int cols,int inner,
                     float alpha,const float *a,int lda,const float *b,int ldb,
                     float beta,float *out,int ldc) {
    double phase_begun = seconds_now();
    if (m->config.kernels.tiled_gemm) {
        experiment_sgemm(trans_a,trans_b,rows,cols,inner,alpha,a,lda,b,ldb,beta,out,ldc,
                         m->config.kernels.tiled_gemm,m->config.threads);
    } else {
        cblas_sgemm(CblasRowMajor,trans_a ? CblasTrans : CblasNoTrans,
                    trans_b ? CblasTrans : CblasNoTrans,rows,cols,inner,
                    alpha,a,lda,b,ldb,beta,out,ldc);
    }
    record_phase(PH_GEMM, phase_begun);
}
/* logits is reused for the mean cross-entropy gradient after forward(). */
static float forward(Model *m, int batch_index) {
    Config c = m->config;
    Layout l = m->layout;
    int cd = c.context * c.dim;
    const uint32_t *contexts = m->contexts + (size_t)batch_index * c.batch * c.context;
    const uint32_t *targets = m->targets + (size_t)batch_index * c.batch;
    for (int i = 0; i < c.batch * c.context; ++i)
        memcpy(m->input + (size_t)i * c.dim,
               m->parameters + (size_t)contexts[i] * c.dim, (size_t)c.dim * sizeof(float));
    multiply(m, 0, 1, c.batch, c.hidden, cd,
                1.0f, m->input, cd, m->parameters + l.hidden_weight, cd,
                0.0f, m->hidden, c.hidden);
    for (int b = 0; b < c.batch; ++b)
        for (int h = 0; h < c.hidden; ++h) {
            size_t at = (size_t)b * c.hidden + h;
            m->hidden[at] = tanhf(m->hidden[at] + m->parameters[l.hidden_bias + h]);
        }
    multiply(m, 0, 1, c.batch, c.vocab, c.hidden,
                1.0f, m->hidden, c.hidden, m->parameters + l.output_weight, c.hidden,
                0.0f, m->logits, c.vocab);
    double phase_begun = seconds_now();
    float loss = experiment_softmax(m->logits,m->parameters+l.output_bias,targets,c.batch,c.vocab,&c.kernels);
    record_phase(PH_SOFTMAX, phase_begun);
    return loss;
}
static void backward(Model *m, int batch_index) {
    Config c = m->config;
    Layout l = m->layout;
    int cd = c.context * c.dim;
    float *g = m->gradients;
    memset(g, 0, l.count * sizeof(float));
    multiply(m, 1, 0, c.vocab, c.hidden, c.batch,
                1.0f, m->logits, c.vocab, m->hidden, c.hidden,
                0.0f, g + l.output_weight, c.hidden);
    double phase_begun = seconds_now();
    experiment_bias_reduce(m->logits,c.batch,c.vocab,g+l.output_bias,&c.kernels);
    record_phase(PH_BIAS, phase_begun);
    multiply(m, 0, 0, c.batch, c.hidden, c.vocab,
                1.0f, m->logits, c.vocab, m->parameters + l.output_weight, c.hidden,
                0.0f, m->dhidden, c.hidden);
    for (int b = 0; b < c.batch; ++b)
        for (int h = 0; h < c.hidden; ++h) {
            size_t at = (size_t)b * c.hidden + h;
            m->dhidden[at] *= 1.0f - m->hidden[at] * m->hidden[at];
        }
    phase_begun = seconds_now();
    experiment_bias_reduce(m->dhidden,c.batch,c.hidden,g+l.hidden_bias,&c.kernels);
    record_phase(PH_BIAS, phase_begun);
    multiply(m, 1, 0, c.hidden, cd, c.batch,
                1.0f, m->dhidden, c.hidden, m->input, cd,
                0.0f, g + l.hidden_weight, cd);
    multiply(m, 0, 0, c.batch, cd, c.hidden,
                1.0f, m->dhidden, c.hidden, m->parameters + l.hidden_weight, cd,
                0.0f, m->dinput, cd);
    const uint32_t *contexts = m->contexts + (size_t)batch_index * c.batch * c.context;
    for (int i = 0; i < c.batch * c.context; ++i)
        for (int d = 0; d < c.dim; ++d)
            g[(size_t)contexts[i] * c.dim + d] += m->dinput[(size_t)i * c.dim + d];
}
static void adamw(Model *m, int step) {
    experiment_adamw(m->parameters,m->moment1,m->moment2,m->gradients,m->layout.count,step,&m->config.kernels);
}
static float update(Model *m, int batch, int step) {
    float loss = forward(m, batch);
    if (!finite_float(loss)) fail("non-finite loss");
    backward(m, batch);
    if (m->check_raw_gradients)
        memcpy(m->check_raw_gradients, m->gradients, m->layout.count * sizeof(float));
    double phase_begun = seconds_now();
    m->gradient_norm=experiment_clip_gradients(m->gradients,m->layout.count,&m->clipping_scale,&m->config.kernels);
    record_phase(PH_CLIP, phase_begun);
    if (!finite_float(m->gradient_norm) || !finite_float(m->clipping_scale)) fail("non-finite gradient norm");
    phase_begun = seconds_now();
    adamw(m, step);
    record_phase(PH_ADAM, phase_begun);
    return loss;
}
static int finite_model(const Model *m) {
    for (size_t i = 0; i < m->layout.count; ++i)
        if (!finite_float(m->parameters[i]) || !finite_float(m->gradients[i])
            || !finite_float(m->moment1[i]) || !finite_float(m->moment2[i])) return 0;
    return 1;
}
static void json_string(const char *value) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}
static void print_config(Config c) {
    printf("{\"vocab_size\":%d,\"context_size\":%d,\"embedding_dim\":%d,\"hidden_dim\":%d,"
           "\"batch_size\":%d,\"threads\":%d,\"seed\":%" PRIu32 ",\"warmup_steps\":%d,\"steps\":%d,"
           "\"seconds\":%.17g,\"clip_norm\":%.9g}",
           c.vocab,c.context,c.dim,c.hidden,c.batch,c.threads,c.seed,c.warmup,c.seconds>0 ? 0 : c.steps,
           c.seconds,c.kernels.clip_norm);
}
static void print_kernel_options(Config c) {
    printf("{\"threads\":%d,\"softmax_parallel\":%s,\"reduction_parallel\":%s,"
           "\"adam_parallel\":%s,\"tiled_gemm\":%d,\"fast_exp\":%s,\"clip_norm\":%.9g}",
           c.threads,c.kernels.softmax_parallel ? "true" : "false",
           c.kernels.reduction_parallel ? "true" : "false",c.kernels.adam_parallel ? "true" : "false",
           c.kernels.tiled_gemm,c.kernels.fast_exp ? "true" : "false",c.kernels.clip_norm);
}
static void print_openmp(void) {
    printf("{\"compiled\":%s,\"probe_team_size\":%d,\"nested_disabled\":true,"
           "\"wait_policy\":\"PASSIVE\",\"kmp_blocktime\":0,"
           "\"scope\":\"available OpenMP team probe outside timer; Accelerate actual worker count is unmeasured\"}",
#ifdef _OPENMP
           "true",
#else
           "false",
#endif
           omp_probe_team_size);
}
static void print_optimizer(void) {
    printf("{\"name\":\"AdamW\",\"lr\":0.001,\"weight_decay\":0.01,"
           "\"beta1\":0.9,\"beta2\":0.999,\"eps\":1e-8}");
}
static void print_array(const float *values, size_t count) {
    putchar('[');
    for (size_t i = 0; i < count; ++i) printf("%s%.9g", i ? "," : "", values[i]);
    putchar(']');
}
static void print_corpus(const Corpus *corpus, Config c) {
    printf("{\"path\":"); json_string(c.corpus_path);
    printf(",\"split\":\"train\",\"stored_ids\":%zu,\"paragraphs\":%zu,\"target_windows\":%" PRIu64 ","
           "\"id_data_offset\":%zu,\"paragraph_data_offset\":%zu,\"paragraph_record_bytes\":28,"
           "\"owned_ram_bytes\":%zu,\"corpus_load_seconds\":%.17g,"
           "\"load_scope\":\"file_read_RAM_load including validation; OS file-cache state uncontrolled, not cold SSD I/O\","
           "\"sampling_policy\":\"deterministic C affine permutation, coprime multiplier, unique targets per pass; differs from legacy hash sampler\","
           "\"sampling_seed\":%" PRIu32 ",\"first_windows\":[",
           corpus->ids.count,corpus->paragraphs.count,corpus->windows,
           corpus->ids.header_bytes,corpus->paragraphs.header_bytes,
           corpus->ids.bytes+corpus->paragraphs.bytes+corpus->paragraphs.count*sizeof(uint64_t),
           corpus->load_seconds,c.seed);
    for (int i = 0; i < corpus->sample_count; ++i) {
        const Sample *sample = &corpus->samples[i];
        printf("%s{\"window_index\":%" PRIu64 ",\"paragraph_index\":%" PRIu64 ",\"contexts\":[",
               i ? "," : "", sample->window,sample->paragraph);
        for (int j = 0; j < c.context; ++j) printf("%s%" PRIu32, j ? "," : "", sample->contexts[j]);
        printf("],\"target\":%" PRIu32 "}",sample->target);
    }
    printf("]}");
}
static void print_check(Model *m) {
    Config c = m->config;
    Layout l = m->layout;
    float loss = update(m, 0, 1);
    if (!finite_model(m)) fail("non-finite tiny correctness result");
    printf("{\"kind\":\"SYNTHETIC_C_NPLM_CHECK\",\"config\":"); print_config(c);
    printf(",\"kernel_options\":"); print_kernel_options(c);
    printf(",\"openmp\":"); print_openmp();
    printf(",\"check_fixture\":\"%s\"",c.dump==2 ? "large_irregular" : "tiny");
    printf(",\"optimizer\":"); print_optimizer();
    printf(",\"parameter_layout\":["
           "{\"name\":\"embedding\",\"shape\":[%d,%d],\"offset\":%zu,\"length\":%zu},"
           "{\"name\":\"hidden_weight\",\"shape\":[%d,%d],\"offset\":%zu,\"length\":%zu},"
           "{\"name\":\"hidden_bias\",\"shape\":[%d],\"offset\":%zu,\"length\":%zu},"
           "{\"name\":\"output_weight\",\"shape\":[%d,%d],\"offset\":%zu,\"length\":%zu},"
           "{\"name\":\"output_bias\",\"shape\":[%d],\"offset\":%zu,\"length\":%zu}],\"initial_params\":",
           c.vocab,c.dim,l.embedding,l.hidden_weight-l.embedding,
           c.hidden,c.context*c.dim,l.hidden_weight,l.hidden_bias-l.hidden_weight,
           c.hidden,l.hidden_bias,l.output_weight-l.hidden_bias,
           c.vocab,c.hidden,l.output_weight,l.output_bias-l.output_weight,
           c.vocab,l.output_bias,l.count-l.output_bias);
    print_array(m->initial, l.count);
    printf(",\"contexts\":[");
    for (int b = 0; b < c.batch; ++b) {
        printf("%s[", b ? "," : "");
        for (int j = 0; j < c.context; ++j)
            printf("%s%" PRIu32, j ? "," : "", m->contexts[(size_t)b*c.context+j]);
        putchar(']');
    }
    printf("],\"targets\":[");
    for (int b = 0; b < c.batch; ++b) printf("%s%" PRIu32, b ? "," : "", m->targets[b]);
    printf("],\"loss\":%.9g,\"grads\":", loss); print_array(m->check_raw_gradients, l.count);
    printf(",\"optimizer_grads\":"); print_array(m->gradients, l.count);
    printf(",\"gradient_norm\":%.9g,\"clipping_scale\":%.9g", m->gradient_norm,m->clipping_scale);
    printf(",\"updated_params\":"); print_array(m->parameters, l.count);
    printf(",\"finite_flag\":true}\n");
}
static int compare_double(const void *a, const void *b) {
    double left = *(const double *)a, right = *(const double *)b;
    return (left > right) - (left < right);
}
static void print_benchmark(Model *m, Corpus *corpus) {
    Config c = m->config;
    Sampler sampler = {0};
    if (corpus) sampler = sampler_start(c, corpus->windows);
    for (int step = 0; step < c.warmup; ++step) {
        if (corpus) gather_real(m, corpus, &sampler, 0);
        update(m, corpus ? 0 : step % BATCHES, step + 1);
    }
    memcpy(m->parameters, m->initial, m->layout.count * sizeof(float));
    memset(m->moment1, 0, m->layout.count * sizeof(float));
    memset(m->moment2, 0, m->layout.count * sizeof(float));
    if (corpus) sampler = sampler_start(c, corpus->windows);
    size_t capacity=c.seconds>0 ? 4096 : (size_t)c.steps;
    double *times=allocate(capacity,sizeof(double));
    float last_loss = 0.0f;
    int completed=0;
    memset(phase_seconds,0,sizeof(phase_seconds));
    memset(phase_calls,0,sizeof(phase_calls));
    double begun = seconds_now();
    while (c.seconds>0 ? seconds_now()-begun<c.seconds : completed<c.steps) {
        if ((size_t)completed==capacity) {
            if (capacity>=(size_t)1048576) fail("too many updates for this bounded elapsed benchmark");
            capacity*=2;
            double *larger=realloc(times,capacity*sizeof(double));
            if (!larger) fail("timing sample allocation failed");
            times=larger;
        }
        double before = seconds_now();
        if (corpus) gather_real(m, corpus, &sampler, 0);
        last_loss=update(m,corpus ? 0 : completed%BATCHES,completed+1);
        times[completed++]=seconds_now()-before;
    }
    double elapsed = seconds_now() - begun;
    double measured_phases[PH_COUNT];
    uint64_t measured_calls[PH_COUNT];
    memcpy(measured_phases,phase_seconds,sizeof(measured_phases));
    memcpy(measured_calls,phase_calls,sizeof(measured_calls));
    float final_loss=forward(m,corpus ? 0 : (completed-1)%BATCHES);
    int finite=finite_float(final_loss) && finite_model(m);
    if (!completed || !finite || elapsed<=0) fail("non-finite benchmark result or invalid timer");
    qsort(times,(size_t)completed,sizeof(double),compare_double);
    double median=completed%2 ? times[completed/2] : (times[completed/2-1]+times[completed/2])*0.5;
    struct utsname platform;
    if (uname(&platform)) fail("platform query failed");
    printf("{\"kind\":\"%s\",\"config\":",corpus ? "REAL_WIKI_C_MICROBENCHMARK" : "SYNTHETIC_C_NPLM_BENCHMARK");
    print_config(c);
    printf(",\"kernel_options\":"); print_kernel_options(c);
    printf(",\"openmp\":"); print_openmp();
    printf(",\"optimizer\":"); print_optimizer();
    printf(",\"compiler\":"); json_string(__VERSION__);
    printf(",\"build_flags\":"); json_string(BENCH_BUILD_FLAGS);
    printf(",\"backend\":"); json_string(BACKEND);
    printf(",\"matmul_backend\":");
    json_string(c.kernels.tiled_gemm ? "custom cache-tiled SGEMM / independent OpenMP output tiles" : BACKEND);
    printf(",\"fast_math_build\":%s",
#ifdef __FAST_MATH__
           "true");
#else
           "false");
#endif
    printf(",\"platform\":{\"os\":"); json_string(platform.sysname);
    printf(",\"architecture\":"); json_string(platform.machine); printf("}");
    printf(",\"requested_threads\":%d,\"backend_reported_threads\":", c.threads);
#ifdef __APPLE__
    printf("null");
#else
    printf("%d", openblas_get_num_threads());
#endif
    printf(",\"thread_policy\":\"Environment budget set before BLAS; Accelerate worker count is not measured.\","
           "\"dtype\":\"float32\",\"parameter_count\":%zu,\"workspace_bytes\":%zu,"
           "\"checksum_algorithm\":\"fnv1a64 over little-endian 32-bit values\","
           "\"initial_parameter_checksum\":\"%016" PRIx64 "\",\"input_checksum\":\"%016" PRIx64 "\","
           "\"steps\":%d,\"completed_steps\":%d,\"requested_seconds\":%.17g,\"target_prediction_events\":%" PRIu64 ",\"elapsed_seconds\":%.17g,"
           "\"targets_per_second\":%.17g,\"median_step_seconds\":%.17g,"
           "\"last_timed_loss\":%.9g,\"final_loss\":%.9g,\"finite_flag\":true,"
           "\"warmup_reset\":true,\"gradient_clipping\":true,\"gradient_max_norm\":%.9g,"
           "\"initialization\":\"xorshift32 uniform[-0.01,0.01); zero seed maps to 0x9e3779b9\","
           "\"final_loss_phase\":\"after timed updates on last mini-batch; evaluation excluded from timer\"",
           m->layout.count,m->workspace_bytes,m->initial_checksum,m->input_checksum,completed,completed,c.seconds,
           (uint64_t)completed*(uint64_t)c.batch,elapsed,
           (double)completed*c.batch/elapsed,median,last_loss,final_loss,c.kernels.clip_norm);
    printf(",\"synthetic_batches\":%d,\"scope\":",corpus ? 0 : BATCHES);
    json_string(corpus ? "Short real Wiki RAM-resident CPU timing; includes window gather and updates; no complete epoch or quality comparison"
                       : "Synthetic CPU compute only: four reused mini-batches, no Wiki files, no quality or corpus training-time measurement");
    if (corpus) {
        printf(",\"measured_epoch_equivalent\":%.17g,\"corpus\":",
               (double)sampler.consumed/(double)corpus->windows);
        print_corpus(corpus,c);
    }
    double accounted = 0.0;
    printf(",\"phase_wall_seconds\":{");
    for (int phase=0; phase<PH_COUNT; ++phase) {
        printf("%s\"%s\":%.17g",phase ? "," : "",phase_names[phase],measured_phases[phase]);
        accounted += measured_phases[phase];
    }
    if (accounted > elapsed + 1e-6) fail("phase accounting exceeds measured wall time");
    printf(",\"other\":%.17g},\"phase_calls\":{",fmax(0.0,elapsed-accounted));
    for (int phase=0; phase<PH_COUNT; ++phase)
        printf("%s\"%s\":%" PRIu64,phase ? "," : "",phase_names[phase],measured_calls[phase]);
    printf("},\"phase_scope\":\"non-overlapping caller wall time; includes scheduling waits and timer overhead; excludes warmup and final evaluation\"");
    printf("}\n");
    free(times);
}
static void destroy_model(Model *m) {
    free(m->parameters); free(m->initial); free(m->gradients); free(m->moment1); free(m->moment2);
    free(m->check_raw_gradients);
    free(m->input); free(m->hidden); free(m->logits); free(m->dhidden); free(m->dinput);
    free(m->contexts); free(m->targets);
}
int main(int argc, char **argv) {
    Config c = arguments(argc, argv);
    thread_budget(c.threads);
    Corpus corpus = {0};
    if (c.corpus_path) corpus = load_corpus(c);
    Model model = create_model(c);
    if (c.corpus_path) {
        Sampler first = sampler_start(c,corpus.windows);
        gather_real(&model,&corpus,&first,1);
    }
    if (c.dump) print_check(&model);
    else if (c.corpus_check) {
        printf("{\"kind\":\"REAL_WIKI_C_CORPUS_CHECK\",\"config\":"); print_config(c);
        printf(",\"kernel_options\":"); print_kernel_options(c);
        printf(",\"openmp\":"); print_openmp();
        printf(",\"input_checksum\":\"%016" PRIx64 "\",\"corpus\":",model.input_checksum);
        print_corpus(&corpus,c); printf("}\n");
    } else print_benchmark(&model,c.corpus_path ? &corpus : NULL);
    destroy_model(&model);
    free(corpus.ids.data); free(corpus.paragraphs.data); free(corpus.cumulative);
    return EXIT_SUCCESS;
}
