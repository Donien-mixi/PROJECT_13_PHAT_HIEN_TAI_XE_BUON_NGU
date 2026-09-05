#ifndef AI_INFERENCE_H_
#define AI_INFERENCE_H_

#include <stdint.h>
#include <stdbool.h>
#include "pnp_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes TensorFlow Lite Micro interpreter and allocates
 * the Tensor Arena in Octal PSRAM (1.5 MB).
 * @return true on success.
 */
bool ai_inference_init(void);

/**
 * Returns a direct pointer to the 96x96 INT8 input tensor buffer.
 */
int8_t* ai_inference_get_input_buffer(void);

/**
 * Executes neural network inference on Core 1 using esp-nn SIMD acceleration.
 *
 * @param out_landmarks Array of 22 normalized keypoints [0.0, 1.0].
 * @param out_latency_us Pointer to store execution latency in microseconds.
 * @return true on success.
 */
bool ai_inference_run(point2d_t out_landmarks[22], int64_t* out_latency_us);

#ifdef __cplusplus
}
#endif

#endif // AI_INFERENCE_H_
