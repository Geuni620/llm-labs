#ifndef KOWIKI_FIXEDWORK_API_H
#define KOWIKI_FIXEDWORK_API_H

#include "experiment_kernels.h"

#ifndef FW_FIXED_NORM
#define FW_FIXED_NORM 0
#endif
#ifndef FW_SOFTMAX_INV_BATCH
#define FW_SOFTMAX_INV_BATCH 0
#endif
#ifndef FW_MINIMAL_ZERO
#define FW_MINIMAL_ZERO 0
#endif

/* Workspace is allocated by the engine before timing for both control and
 * candidate. Logical blocks and final merge order do not depend on threads. */
size_t fw_norm_scratch_count(size_t gradient_count);
float fw_clip_gradients(float *gradients, size_t count, float *clipping_scale,
                        const ExperimentKernelOptions *options,
                        double *partial_sums, size_t partial_capacity);

/* CPU vForce softmax with only the power-of-two batch normalization division
 * replaced by multiplication by its exact reciprocal. Other batch sizes use
 * the baseline division. Sum/max/exp order and precision remain unchanged. */
float fw_softmax(float *logits, const float *bias, const uint32_t *targets,
                 int batch, int vocab, const ExperimentKernelOptions *options);

#endif
