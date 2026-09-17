#ifndef IMAGE_DECODER_H_
#define IMAGE_DECODER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** [v2.9.0] Vùng crop vuông trong khung đã giải mã (pixel). */
typedef struct {
    int x0;      // góc trên-trái
    int y0;
    int size;    // cạnh vuông
    int src_w;   // kích thước khung gốc (đầu ra)
    int src_h;
} image_crop_roi_t;

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

/**
 * [v2.9.0 - FACE ROI] Như image_decoder_process_jpeg nhưng crop theo ROI chỉ định
 * (bám khuôn mặt). Nếu roi_in == NULL hoặc không hợp lệ -> tự crop giữa khung.
 * @param roi_in   ROI mong muốn (có thể NULL).
 * @param roi_used ROI thực tế đã dùng (để map landmark ngược ra khung gốc).
 */
bool image_decoder_process_jpeg_roi(const uint8_t* jpeg_data, size_t jpeg_len,
                                    int8_t* out_int8_tensor,
                                    float input_scale, int32_t input_zero_point,
                                    const image_crop_roi_t* roi_in,
                                    image_crop_roi_t* roi_used);

/**
 * [v2.9.4 - DIAG] Lấy buffer xám TOÀN KHUNG của lần giải mã gần nhất (để gửi preview).
 * @param out_w,out_h  Kích thước khung (ra).
 * @return con trỏ buffer xám (không được free), hoặc NULL.
 */
const uint8_t* image_decoder_get_last_gray(int* out_w, int* out_h);

/**
 * [v2.9.5 - M1] Cấu hình CROP CỐ ĐỊNH (deterministic, không vòng phản hồi).
 * @param enabled  1 = dùng crop cố định theo %; 0 = dùng ROI truyền vào (auto)
 * @param cx_pct,cy_pct  tâm crop theo % (0..100)
 * @param size_pct  cạnh crop theo % cạnh ngắn (30..100)
 */
void image_decoder_set_fixed_crop(int enabled, int cx_pct, int cy_pct, int size_pct);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_DECODER_H_
