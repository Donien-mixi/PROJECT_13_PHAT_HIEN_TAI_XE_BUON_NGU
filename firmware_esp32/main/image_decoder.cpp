#include "image_decoder.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// =====================================================================
// [SYNC 2025] Giải mã JPEG THẬT bằng espressif/esp_new_jpeg.
// Trước đây hàm này chỉ là stub "map byte sau marker SOS" -> ảnh nạp vào
// tensor là nhiễu rác trên mạch thật, mô hình AI không thể hoạt động dù
// đã được train chuẩn. esp_new_jpeg tận dụng SIMD ESP32-S3, chỉ hỗ trợ
// baseline JPEG (OpenCV imencode mặc định là baseline -> tương thích).
// Thêm dependency trong main/idf_component.yml:
//   dependencies:
//     espressif/esp_new_jpeg: "^1.0.2"
// =====================================================================
#if __has_include("esp_jpeg_dec.h")
#include "esp_jpeg_dec.h"
#define HAS_ESP_NEW_JPEG 1
#else
#define HAS_ESP_NEW_JPEG 0
#endif

static const char* TAG = "IMAGE_DECODER";

// Rolling phase timers (logged every 60 frames) to target the real decode bottleneck.
static int64_t s_acc_jpeg_us = 0;
static int64_t s_acc_ds_us = 0;
static int s_dec_frames = 0;

#define TARGET_WIDTH  96
#define TARGET_HEIGHT 96
#define MAX_IMG_W 320
#define MAX_IMG_H 320

// Intermediate buffers in Octal PSRAM
static uint8_t* s_gray_buffer = NULL;   // grayscale đã decode (MAX_IMG_W*MAX_IMG_H)
#if HAS_ESP_NEW_JPEG
static uint8_t* s_rgb_buffer = NULL;    // RGB888 output của decoder (x3)
#endif

// [v2.9.4] Kích thước khung gần nhất (cho preview)
static int s_last_img_w = 0;
static int s_last_img_h = 0;

// [v2.9.5 - M1] CROP CỐ ĐỊNH (deterministic, không vòng phản hồi)
static int s_crop_fixed    = 1;   // 1 = dùng crop cố định; 0 = dùng ROI (nếu truyền vào)
static int s_crop_cx_pct   = 50;  // tâm crop theo % chiều rộng
static int s_crop_cy_pct   = 52;  // tâm crop theo % chiều cao
static int s_crop_size_pct = 75;  // cạnh crop theo % cạnh ngắn

bool image_decoder_init(void) {
    ESP_LOGI(TAG, "Cấp phát buffer giải mã ảnh trong Octal PSRAM (gray %d KB%s)...",
             (MAX_IMG_W * MAX_IMG_H) / 1024,
             HAS_ESP_NEW_JPEG ? " + rgb" : ", STUB-MODE không có esp_new_jpeg!");
    s_gray_buffer = (uint8_t*) heap_caps_malloc(MAX_IMG_W * MAX_IMG_H, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_gray_buffer) {
        ESP_LOGE(TAG, "LỖI: Không thể cấp phát gray_buffer trong PSRAM!");
        return false;
    }
#if HAS_ESP_NEW_JPEG
    s_rgb_buffer = (uint8_t*) heap_caps_aligned_alloc(16, MAX_IMG_W * MAX_IMG_H * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_rgb_buffer) {
        ESP_LOGE(TAG, "LỖI: Không thể cấp phát rgb_buffer trong PSRAM!");
        return false;
    }
    ESP_LOGI(TAG, "✅ esp_new_jpeg đã sẵn sàng (RGB888 output, baseline JPEG).");
#else
    ESP_LOGW(TAG, "⚠️ esp_new_jpeg KHÔNG có trong build! Thêm 'espressif/esp_new_jpeg: ^1.0.2' "
                  "vào main/idf_component.yml, nếu không ảnh nạp vào AI sẽ là rác.");
#endif
    return true;
}

#if !HAS_ESP_NEW_JPEG
// Fast embedded JPEG dimension parser (SOF0/SOF1 marker: 0xFF 0xC0/0xC1)
static bool parse_jpeg_dimensions(const uint8_t* data, size_t len, int* out_w, int* out_h) {
    if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false; // Not a valid JPEG SOI
    }

    size_t i = 2;
    while (i < len - 8) {
        if (data[i] == 0xFF) {
            uint8_t marker = data[i + 1];
                return true;
            }
            if (marker != 0xD8 && marker != 0xD9 && marker != 0x00 && marker != 0xFF) {
                uint16_t segment_len = (data[i + 2] << 8) | data[i + 3];
                i += 2 + segment_len;
                continue;
            }
        }
        i++;
    }
    return false;
}
#endif

// [v2.9.0] Wrapper tương thích: crop giữa khung (không ROI).
bool image_decoder_process_jpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                                int8_t* out_int8_tensor,
                                float input_scale, int32_t input_zero_point) {
    return image_decoder_process_jpeg_roi(jpeg_data, jpeg_len, out_int8_tensor,
                                          input_scale, input_zero_point, NULL, NULL);
}

// [v2.9.4] Trả buffer xám toàn khung gần nhất + kích thước (cho preview)
const uint8_t* image_decoder_get_last_gray(int* out_w, int* out_h) {
    if (out_w) *out_w = s_last_img_w;
    if (out_h) *out_h = s_last_img_h;
    if (s_last_img_w <= 0 || s_last_img_h <= 0) return NULL;
    return s_gray_buffer;
}

// [v2.9.5 - M1] Bật/tắt + cấu hình crop cố định (deterministic)
void image_decoder_set_fixed_crop(int enabled, int cx_pct, int cy_pct, int size_pct) {
    s_crop_fixed = enabled ? 1 : 0;
    if (cx_pct >= 0 && cx_pct <= 100)   s_crop_cx_pct = cx_pct;
    if (cy_pct >= 0 && cy_pct <= 100)   s_crop_cy_pct = cy_pct;
    if (size_pct >= 30 && size_pct <= 100) s_crop_size_pct = size_pct;
    ESP_LOGI(TAG, "Crop mode: %s (cx=%d%% cy=%d%% size=%d%%)",
             s_crop_fixed ? "FIXED" : "ROI/AUTO", s_crop_cx_pct, s_crop_cy_pct, s_crop_size_pct);
}

// Bilinear Downsampling & INT8 Quantization Engine
// Guarantees strict 1:1 Aspect Ratio Preservation (scale_x == scale_y)
// [v2.9.0] Nhận vùng crop vuông (crop_x0, crop_y0, crop_size) thay vì cố định giữa khung.
static void downsample_and_quantize_crop(const uint8_t* src_gray, int src_w, int src_h,
                                         int crop_x0, int crop_y0, int crop_size,
                                         int8_t* out_tensor,
                                         float input_scale, int32_t input_zp) {
    if (crop_size < 1) {
        crop_size = (src_w < src_h) ? src_w : src_h;
    }
    float step = (float)crop_size / (float)TARGET_WIDTH; // Identical step in X and Y

    for (int dst_y = 0; dst_y < TARGET_HEIGHT; dst_y++) {
        float src_y_f = (float)crop_y0 + dst_y * step;
        int sy0 = (int)floorf(src_y_f);
        int sy1 = sy0 + 1;
        float dy = src_y_f - (float)sy0;
        if (sy0 < 0) sy0 = 0;
        if (sy0 > src_h - 1) sy0 = src_h - 1;
        if (sy1 < 0) sy1 = 0;
        if (sy1 > src_h - 1) sy1 = src_h - 1;

        for (int dst_x = 0; dst_x < TARGET_WIDTH; dst_x++) {
            float src_x_f = (float)crop_x0 + dst_x * step;
            int sx0 = (int)floorf(src_x_f);
            int sx1 = sx0 + 1;
            float dx = src_x_f - (float)sx0;
            if (sx0 < 0) sx0 = 0;
            if (sx0 > src_w - 1) sx0 = src_w - 1;
            if (sx1 < 0) sx1 = 0;
            if (sx1 > src_w - 1) sx1 = src_w - 1;

            // 4 neighbor pixels
            float p00 = src_gray[sy0 * src_w + sx0];
            float p01 = src_gray[sy0 * src_w + sx1];
            float p10 = src_gray[sy1 * src_w + sx0];
            float p11 = src_gray[sy1 * src_w + sx1];

            // Bilinear interpolation
            float top = p00 * (1.0f - dx) + p01 * dx;
            float bottom = p10 * (1.0f - dx) + p11 * dx;
            float gray_val = top * (1.0f - dy) + bottom * dy;

            // Normalize [-1.0, 1.0] matching training pipeline and quantize to INT8
            float norm_val = (gray_val - 128.0f) / 128.0f;
            int32_t q = (int32_t)roundf(norm_val / input_scale) + input_zp;

            // Clamp to [-128, 127]
            if (q < -128) q = -128;
            if (q > 127)  q = 127;

            out_tensor[dst_y * TARGET_WIDTH + dst_x] = (int8_t)q;
        }
    }
}

#if HAS_ESP_NEW_JPEG
// Decode baseline JPEG -> RGB888 (esp_new_jpeg) -> luminance grayscale
static bool decode_jpeg_to_gray(const uint8_t* jpeg_data, size_t jpeg_len,
                                int* out_w, int* out_h) {
    jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
    config.output_type = JPEG_PIXEL_FORMAT_RGB888;
    config.rotate = JPEG_ROTATE_0D;

    jpeg_dec_handle_t jpeg_dec = NULL;
    jpeg_error_t err = jpeg_dec_open(&config, &jpeg_dec);
    if (err != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg_dec_open lỗi: %d", (int)err);
        return false;
    }

    jpeg_dec_io_t jpeg_io = {};
    jpeg_io.inbuf = (unsigned char*)jpeg_data;
    jpeg_io.inbuf_len = jpeg_len;

    jpeg_dec_header_info_t jpeg_info = {};
    err = jpeg_dec_parse_header(jpeg_dec, &jpeg_io, &jpeg_info);
    if (err != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg_dec_parse_header lỗi: %d (JPEG progressive không hỗ trợ!)", (int)err);
        jpeg_dec_close(jpeg_dec);
        return false;
    }

    int img_w = (int)jpeg_info.width;
    int img_h = (int)jpeg_info.height;
    if (img_w <= 0 || img_h <= 0 || img_w > MAX_IMG_W || img_h > MAX_IMG_H) {
        ESP_LOGW(TAG, "Kích thước JPEG ngoài giới hạn: %dx%d (max %dx%d)", img_w, img_h, MAX_IMG_W, MAX_IMG_H);
        jpeg_dec_close(jpeg_dec);
        return false;
    }

    // Reuse the pre-allocated, 16-byte aligned RGB buffer (no per-frame calloc/zeroing)
    size_t rgb_size = (size_t)img_w * img_h * 3;
    if (rgb_size > (size_t)MAX_IMG_W * MAX_IMG_H * 3) {
        ESP_LOGE(TAG, "RGB buffer quá nhỏ: cần %u bytes", (unsigned)rgb_size);
        jpeg_dec_close(jpeg_dec);
        return false;
    }
    jpeg_io.outbuf = (unsigned char*)s_rgb_buffer;

    err = jpeg_dec_process(jpeg_dec, &jpeg_io);
    if (err != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg_dec_process lỗi: %d", (int)err);
        jpeg_dec_close(jpeg_dec);
        return false;
    }

    // RGB888 -> Luminance (ITU-R BT.601) vào gray buffer
    const uint8_t* rgb = jpeg_io.outbuf;
    for (int p = 0; p < img_w * img_h; p++) {
        uint32_t r = rgb[p * 3 + 0];
        uint32_t g = rgb[p * 3 + 1];
        uint32_t b = rgb[p * 3 + 2];
        s_gray_buffer[p] = (uint8_t)((r * 299 + g * 587 + b * 114) / 1000);
    }

    jpeg_dec_close(jpeg_dec);

    *out_w = img_w;
    *out_h = img_h;
    return true;
}
#endif

bool image_decoder_process_jpeg_roi(const uint8_t* jpeg_data, size_t jpeg_len,
                                    int8_t* out_int8_tensor,
                                    float input_scale, int32_t input_zero_point,
                                    const image_crop_roi_t* roi_in,
                                    image_crop_roi_t* roi_used) {
    if (!jpeg_data || jpeg_len == 0 || !out_int8_tensor) {
        return false;
    }

    // Quick SOI validation
    if (jpeg_len < 4 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8) {
        ESP_LOGW(TAG, "Frame không phải JPEG hợp lệ (thiếu SOI marker)");
        return false;
    }

    int img_w = 0, img_h = 0;
    int64_t t_dec_start = esp_timer_get_time();

#if HAS_ESP_NEW_JPEG
    if (!decode_jpeg_to_gray(jpeg_data, jpeg_len, &img_w, &img_h)) {
        return false;
    }
#else
    // Fallback stub (chỉ dùng khi chưa thêm esp_new_jpeg - KÊU CẢNH BÁO RÕ)
    if (!parse_jpeg_dimensions(jpeg_data, jpeg_len, &img_w, &img_h)) {
        img_w = 240;
        img_h = 240;
    }
    if (img_w <= 0 || img_h <= 0 || img_w > MAX_IMG_W || img_h > MAX_IMG_H) {
        ESP_LOGW(TAG, "Kích thước ảnh JPEG bất thường: %dx%d", img_w, img_h);
        return false;
    }
    size_t gray_size = (size_t)img_w * img_h;
    size_t scan_idx = 0;
    for (size_t i = 0; i < jpeg_len - 1; i++) {
        if (jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA) { // SOS
            scan_idx = i + 2;
            break;
        }
    }
    if (scan_idx > 0 && scan_idx < jpeg_len) {
        size_t available_bytes = jpeg_len - scan_idx;
        for (size_t p = 0; p < gray_size; p++) {
            s_gray_buffer[p] = jpeg_data[scan_idx + (p % available_bytes)];
        }
    } else {
        memset(s_gray_buffer, 128, gray_size);
    }
#endif

    int64_t t_dec_end = esp_timer_get_time();
    s_last_img_w = img_w;
    s_last_img_h = img_h;

    // [v2.9.5 - M1] Xác định vùng crop theo thứ tự ưu tiên:
    //   1) ROI tuyệt đối nếu được truyền (detector/tracker)
    //   2) CROP CỐ ĐỊNH theo % (mặc định, deterministic)
    //   3) Crop giữa khung
    int crop_size = (img_w < img_h) ? img_w : img_h;
    int crop_x0 = (img_w - crop_size) / 2;
    int crop_y0 = (img_h - crop_size) / 2;
    if (roi_in && roi_in->size > 0) {
        crop_size = roi_in->size;
        crop_x0 = roi_in->x0;
        crop_y0 = roi_in->y0;
    } else if (s_crop_fixed) {
        int minDim = (img_w < img_h) ? img_w : img_h;
        crop_size = minDim * s_crop_size_pct / 100;
        if (crop_size < 32) crop_size = 32;
        if (crop_size > minDim) crop_size = minDim;
        int cx = img_w * s_crop_cx_pct / 100;
        int cy = img_h * s_crop_cy_pct / 100;
        crop_x0 = cx - crop_size / 2;
        crop_y0 = cy - crop_size / 2;
    }
    // Kẹp trong khung
    if (crop_size > img_w) crop_size = img_w;
    if (crop_size > img_h) crop_size = img_h;
    if (crop_x0 < 0) crop_x0 = 0;
    if (crop_y0 < 0) crop_y0 = 0;
    if (crop_x0 + crop_size > img_w) crop_x0 = img_w - crop_size;
    if (crop_y0 + crop_size > img_h) crop_y0 = img_h - crop_size;
    if (crop_x0 < 0) crop_x0 = 0;
    if (crop_y0 < 0) crop_y0 = 0;
    if (roi_used) {
        roi_used->x0 = crop_x0;
        roi_used->y0 = crop_y0;
        roi_used->size = crop_size;
        roi_used->src_w = img_w;
        roi_used->src_h = img_h;
    }

    // Isomorphic Downsampling to 96x96 INT8 (scale_x == scale_y tuyệt đối)
    downsample_and_quantize_crop(s_gray_buffer, img_w, img_h, crop_x0, crop_y0, crop_size,
                                 out_int8_tensor,
                                 input_scale, input_zero_point);
    int64_t t_dec_all = esp_timer_get_time();

    s_acc_jpeg_us += (t_dec_end - t_dec_start);
    s_acc_ds_us += (t_dec_all - t_dec_end);
    s_dec_frames++;
    if (s_dec_frames >= 60) {
        ESP_LOGI(TAG, "[Decode avg] JPEG: %.1f ms | Gray+Downsample: %.1f ms",
                 (double)s_acc_jpeg_us / 60.0 / 1000.0, (double)s_acc_ds_us / 60.0 / 1000.0);
        s_acc_jpeg_us = 0;
        s_acc_ds_us = 0;
        s_dec_frames = 0;
    }

    return true;
}
