/* Exact, BLAS-free initialization and FNV helpers copied from the frozen C
 * reference's documented arithmetic. Root compiles this file separately.
 * Example: cc -O3 -std=c11 -ffp-contract=off -dynamiclib init_reference.c
 *              -o init_reference.dylib
 */
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "The initialization reference must not be built with global fast-math"
#endif
_Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24,
               "The reference requires IEEE binary32 floats");

uint32_t init_reference_abi_version(void) { return 1; }

static uint32_t next_random(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return *state = value;
}

uint32_t fill_init(float *out, size_t count, uint32_t seed) {
    uint32_t state = seed ? seed : UINT32_C(0x9e3779b9);
    for (size_t index = 0; index < count; ++index) {
        float uniform = (float)(next_random(&state) >> 8) * (1.0f / 16777216.0f);
        volatile float scaled = uniform * 2.0f;
        volatile float centered = scaled - 1.0f;
        out[index] = centered * 0.01f;
    }
    return state;
}

uint64_t fnv32(const uint32_t *values, size_t count, uint64_t initial_hash) {
    uint64_t hash = initial_hash;
    for (size_t index = 0; index < count; ++index) {
        uint32_t value = values[index];
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 255u;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

uint64_t fnv32_floats(const float *values, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        hash = fnv32(&bits, 1, hash);
    }
    return hash;
}

/* The frozen engine generates four synthetic batches after all initializer
 * draws, and forces the first two context IDs in each batch to 1. */
uint32_t fill_synthetic(uint32_t *contexts, uint32_t *targets, size_t parameter_count,
                        size_t batches, size_t batch_size, size_t context_size,
                        uint32_t vocab_size, uint32_t seed) {
    uint32_t state = seed ? seed : UINT32_C(0x9e3779b9);
    for (size_t index = 0; index < parameter_count; ++index) (void)next_random(&state);
    for (size_t batch = 0; batch < batches; ++batch) {
        size_t offset = batch * batch_size * context_size;
        for (size_t index = 0; index < batch_size * context_size; ++index)
            contexts[offset + index] = next_random(&state) % vocab_size;
        if (batch_size * context_size >= 2)
            contexts[offset] = contexts[offset + 1] = 1;
        for (size_t index = 0; index < batch_size; ++index)
            targets[batch * batch_size + index] = next_random(&state) % vocab_size;
    }
    return state;
}
