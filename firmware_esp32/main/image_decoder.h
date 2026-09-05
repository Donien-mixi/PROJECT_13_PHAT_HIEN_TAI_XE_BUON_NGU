#ifndef IMAGE_DECODER_H_
#define IMAGE_DECODER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes image decoding buffers in Octal PSRAM.
 * @return true on success.
 */
bool image_decoder_init(void);

/**
 * Decodes a 1:1 Square JPEG buffer from PSRAM and applies
 * Isomorphic Downsampling (scale_x == scale_y) directly into
 * the 96x96 INT8 input tensor buffer of TinyDriverNet.
 *
 * @param jpeg_data Pointer to JPEG byte array in PSRAM.
 * @param jpeg_len Length of JPEG byte array.
 * @param out_int8_tensor Pointer to 96x96 INT8 tensor buffer (size 9216 bytes).
 * @param input_scale Quantization scale parameter of model input.
 * @param input_zero_point Quantization zero-point parameter of model input.
 * @return true on successful decode & transform, false on corrupt frame.
 */
bool image_decoder_process_jpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                                int8_t* out_int8_tensor,
                                float input_scale, int32_t input_zero_point);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_DECODER_H_
