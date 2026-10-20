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

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define BACKEND "Apple Accelerate / CBLAS"
#else
#include <cblas.h>
extern void openblas_set_num_threads(int);
extern int openblas_get_num_threads(void);
#define BACKEND "OpenBLAS / CBLAS"
#endif

#ifdef __FAST_MATH__
#error "This comparison requires builds without fast-math"
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
         "         --dump-check (fixed tiny V11/C2/D3/H5/B4, seed42, one update)");
}
static Config arguments(int argc, char **argv) {
    Config c = {35328, 8, 64, 128, 128, 4, 4, 64, 0, 42, NULL, 0};
    for (int i = 1; i < argc; ++i) {
        const char *name = argv[i];
        if (!strcmp(name, "--help")) { usage(); exit(EXIT_SUCCESS); }
        if (!strcmp(name, "--dump-check")) { c.dump = 1; continue; }
        if (!strcmp(name, "--dump-corpus-check")) { c.corpus_check = 1; continue; }
        if (i + 1 == argc) fail("option is missing its value");
        if (!strcmp(name, "--corpus")) { c.corpus_path = argv[++i]; continue; }
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
        else fail("unknown option; use --help");
    }
    if (c.dump && (c.corpus_path || c.corpus_check)) fail("tiny check and corpus mode are separate");
    if (c.corpus_check && !c.corpus_path) fail("--dump-corpus-check requires --corpus PATH");
    if (c.dump) {
        c.vocab = 11; c.context = 2; c.dim = 3; c.hidden = 5;
        c.batch = 4; c.seed = 42; c.warmup = 0; c.steps = 1;
    }
    if (c.vocab < 2 || c.vocab > 65536 || c.context < 1 || c.context > 32
        || c.dim < 1 || c.dim > 512 || c.hidden < 1 || c.hidden > 1024
        || c.batch < 1 || c.batch > 1024 || c.threads < 1 || c.threads > 64
        || c.warmup < 0 || c.warmup > 256 || c.steps < 1 || c.steps > 4096)
        fail("configuration exceeds bounded supported ranges; see README");
    return c;
}
static void thread_budget(int threads) {
    char value[32];
    snprintf(value, sizeof(value), "%d", threads);
    const char *names[] = {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "MKL_NUM_THREADS"};
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); ++i)
        if (setenv(names[i], value, 1)) fail("could not set thread budget environment");
#ifndef __APPLE__
    openblas_set_num_threads(threads);
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
        m.initial[i] = (uniform * 2.0f - 1.0f) * 0.01f;
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c.batch, c.hidden, cd,
                1.0f, m->input, cd, m->parameters + l.hidden_weight, cd,
                0.0f, m->hidden, c.hidden);
    for (int b = 0; b < c.batch; ++b)
        for (int h = 0; h < c.hidden; ++h) {
            size_t at = (size_t)b * c.hidden + h;
            m->hidden[at] = tanhf(m->hidden[at] + m->parameters[l.hidden_bias + h]);
        }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c.batch, c.vocab, c.hidden,
                1.0f, m->hidden, c.hidden, m->parameters + l.output_weight, c.hidden,
                0.0f, m->logits, c.vocab);
    float loss_sum = 0.0f;
    for (int b = 0; b < c.batch; ++b) {
        float *row = m->logits + (size_t)b * c.vocab;
        float maximum = -INFINITY;
        for (int v = 0; v < c.vocab; ++v) {
            row[v] += m->parameters[l.output_bias + v];
            if (row[v] > maximum) maximum = row[v];
        }
        float target_logit = row[targets[b]], sum = 0.0f;
        for (int v = 0; v < c.vocab; ++v) { row[v] = expf(row[v] - maximum); sum += row[v]; }
        loss_sum += logf(sum) + maximum - target_logit;
        for (int v = 0; v < c.vocab; ++v) row[v] = (row[v] / sum) / (float)c.batch;
        row[targets[b]] -= 1.0f / (float)c.batch;
    }
    return loss_sum / (float)c.batch;
}
static void backward(Model *m, int batch_index) {
    Config c = m->config;
    Layout l = m->layout;
    int cd = c.context * c.dim;
    float *g = m->gradients;
    memset(g, 0, l.count * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, c.vocab, c.hidden, c.batch,
                1.0f, m->logits, c.vocab, m->hidden, c.hidden,
                0.0f, g + l.output_weight, c.hidden);
    for (int b = 0; b < c.batch; ++b)
        for (int v = 0; v < c.vocab; ++v)
            g[l.output_bias + v] += m->logits[(size_t)b * c.vocab + v];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c.batch, c.hidden, c.vocab,
                1.0f, m->logits, c.vocab, m->parameters + l.output_weight, c.hidden,
                0.0f, m->dhidden, c.hidden);
    for (int b = 0; b < c.batch; ++b)
        for (int h = 0; h < c.hidden; ++h) {
            size_t at = (size_t)b * c.hidden + h;
            m->dhidden[at] *= 1.0f - m->hidden[at] * m->hidden[at];
            g[l.hidden_bias + h] += m->dhidden[at];
        }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, c.hidden, cd, c.batch,
                1.0f, m->dhidden, c.hidden, m->input, cd,
                0.0f, g + l.hidden_weight, cd);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c.batch, cd, c.hidden,
                1.0f, m->dhidden, c.hidden, m->parameters + l.hidden_weight, cd,
                0.0f, m->dinput, cd);
    const uint32_t *contexts = m->contexts + (size_t)batch_index * c.batch * c.context;
    for (int i = 0; i < c.batch * c.context; ++i)
        for (int d = 0; d < c.dim; ++d)
            g[(size_t)contexts[i] * c.dim + d] += m->dinput[(size_t)i * c.dim + d];
}
static void adamw(Model *m, int step) {
    float correction2_root = (float)sqrt(1.0 - pow(0.999, (double)step));
    float step_size = (float)(0.001 / (1.0 - pow(0.9, (double)step)));
    float decay = (float)(1.0 - 0.001 * 0.01);
    for (size_t i = 0; i < m->layout.count; ++i) {
        float g = m->gradients[i];
        m->parameters[i] *= decay;
        m->moment1[i] += 0.1f * (g - m->moment1[i]);
        m->moment2[i] = 0.999f * m->moment2[i] + 0.001f * (g * g);
        float denominator = sqrtf(m->moment2[i]) / correction2_root + 1e-8f;
        m->parameters[i] -= step_size * m->moment1[i] / denominator;
    }
}
static float update(Model *m, int batch, int step) {
    float loss = forward(m, batch);
    if (!isfinite(loss)) fail("non-finite loss");
    backward(m, batch);
    if (m->check_raw_gradients)
        memcpy(m->check_raw_gradients, m->gradients, m->layout.count * sizeof(float));
    double squares = 0.0;
    for (size_t i = 0; i < m->layout.count; ++i)
        squares += (double)m->gradients[i] * m->gradients[i];
    m->gradient_norm = (float)sqrt(squares);
    if (!isfinite(m->gradient_norm)) fail("non-finite gradient norm");
    m->clipping_scale = fminf(1.0f, 1.0f / (m->gradient_norm + 1e-6f));
    if (m->clipping_scale < 1.0f)
        for (size_t i = 0; i < m->layout.count; ++i) m->gradients[i] *= m->clipping_scale;
    adamw(m, step);
    return loss;
}
static int finite_model(const Model *m) {
    for (size_t i = 0; i < m->layout.count; ++i)
        if (!isfinite(m->parameters[i]) || !isfinite(m->gradients[i])
            || !isfinite(m->moment1[i]) || !isfinite(m->moment2[i])) return 0;
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
           "\"batch_size\":%d,\"threads\":%d,\"seed\":%" PRIu32 ",\"warmup_steps\":%d,\"steps\":%d}",
           c.vocab, c.context, c.dim, c.hidden, c.batch, c.threads, c.seed, c.warmup, c.steps);
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
    double *times = allocate((size_t)c.steps, sizeof(double));
    float last_loss = 0.0f;
    double begun = seconds_now();
    for (int step = 0; step < c.steps; ++step) {
        double before = seconds_now();
        if (corpus) gather_real(m, corpus, &sampler, 0);
        last_loss = update(m, corpus ? 0 : step % BATCHES, step + 1);
        times[step] = seconds_now() - before;
    }
    double elapsed = seconds_now() - begun;
    float final_loss = forward(m, corpus ? 0 : (c.steps - 1) % BATCHES);
    int finite = isfinite(final_loss) && finite_model(m);
    if (!finite || elapsed <= 0) fail("non-finite benchmark result or invalid timer");
    qsort(times, (size_t)c.steps, sizeof(double), compare_double);
    double median = c.steps % 2 ? times[c.steps/2] : (times[c.steps/2-1]+times[c.steps/2])*0.5;
    struct utsname platform;
    if (uname(&platform)) fail("platform query failed");
    printf("{\"kind\":\"%s\",\"config\":",corpus ? "REAL_WIKI_C_MICROBENCHMARK" : "SYNTHETIC_C_NPLM_BENCHMARK");
    print_config(c);
    printf(",\"optimizer\":"); print_optimizer();
    printf(",\"compiler\":"); json_string(__VERSION__);
    printf(",\"build_flags\":"); json_string(BENCH_BUILD_FLAGS);
    printf(",\"backend\":"); json_string(BACKEND);
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
           "\"steps\":%d,\"target_prediction_events\":%" PRIu64 ",\"elapsed_seconds\":%.17g,"
           "\"targets_per_second\":%.17g,\"median_step_seconds\":%.17g,"
           "\"last_timed_loss\":%.9g,\"final_loss\":%.9g,\"finite_flag\":true,"
           "\"warmup_reset\":true,\"gradient_clipping\":true,\"gradient_max_norm\":1.0,"
           "\"initialization\":\"xorshift32 uniform[-0.01,0.01); zero seed maps to 0x9e3779b9\","
           "\"final_loss_phase\":\"after timed updates on last mini-batch; evaluation excluded from timer\"",
           m->layout.count,m->workspace_bytes,m->initial_checksum,m->input_checksum,c.steps,
           (uint64_t)c.steps*(uint64_t)c.batch,elapsed,
           (double)c.steps*c.batch/elapsed,median,last_loss,final_loss);
    printf(",\"synthetic_batches\":%d,\"scope\":",corpus ? 0 : BATCHES);
    json_string(corpus ? "Short real Wiki RAM-resident CPU timing; includes window gather and updates; no complete epoch or quality comparison"
                       : "Synthetic CPU compute only: four reused mini-batches, no Wiki files, no quality or corpus training-time measurement");
    if (corpus) {
        printf(",\"measured_epoch_equivalent\":%.17g,\"corpus\":",
               (double)sampler.consumed/(double)corpus->windows);
        print_corpus(corpus,c);
    }
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
        printf(",\"input_checksum\":\"%016" PRIx64 "\",\"corpus\":",model.input_checksum);
        print_corpus(&corpus,c); printf("}\n");
    } else print_benchmark(&model,c.corpus_path ? &corpus : NULL);
    destroy_model(&model);
    free(corpus.ids.data); free(corpus.paragraphs.data); free(corpus.cumulative);
    return EXIT_SUCCESS;
}
