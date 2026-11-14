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
 * [P4 - DIAG] Chỉ giải mã JPEG -> buffer xám toàn khung (KHÔNG crop/downsample).
 * Dùng cho chế độ chẩn đoán tuần tự: decode 1 lần, detect trên cùng frame,
 * rồi mới crop+quantize từ buffer xám này (tránh decode 2 lần).
 */
bool image_decoder_decode_jpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                               int* out_w, int* out_h);

/**
 * [P4 - DIAG] Crop + downsampling 96x96 từ buffer xám của lần decode gần nhất.
 * Không giải mã lại JPEG. roi_in == NULL -> dùng crop cố định/giữa khung.
 */
bool image_decoder_crop_from_last_gray(const image_crop_roi_t* roi_in,
                                       int8_t* out_int8_tensor,
                                       float input_scale, int32_t input_zero_point,
                                       image_crop_roi_t* roi_used);

/**
 * [v2.9.5 - M1] Cấu hình CROP CỐ ĐỊNH (deterministic, không vòng phản hồi).
 * @param enabled  1 = dùng crop cố định theo %; 0 = dùng ROI truyền vào (auto)
 * @param cx_pct,cy_pct  tâm crop theo % (0..100)
 * @param size_pct  cạnh crop theo % cạnh ngắn (30..100)
 */
void image_decoder_set_fixed_crop(int enabled, int cx_pct, int cy_pct, int size_pct);


/**
 * [14/11 - SPEED] Load the Y plane of a YUV422 (YUYV) frame straight into the grey frame buffer.
 * Replaces the JPEG decode when the OV5640 is configured with PIXFORMAT_GRAYSCALE/YUV422.
 * Y is byte 0 of every pixel pair, so this is a memcpy-with-stride (~5-20 ms for QVGA).
 * @param raw   frame bytes from the camera (stride 2: Y0 U0 Y1 V0 ...)
 * @param len   raw length (bytes); must be >= w*h*2 for the simple stride-2 layout
 * @param out_w,out_h  frame size (out)
 * @return true on success
 */
bool image_decoder_load_gray_raw(const uint8_t* raw, size_t len, int* out_w, int* out_h);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_DECODER_H_
