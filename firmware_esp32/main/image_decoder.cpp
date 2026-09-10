#include "image_decoder.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

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

#define TARGET_WIDTH  96
#define TARGET_HEIGHT 96
#define MAX_IMG_W 320
#define MAX_IMG_H 320

// Intermediate buffers in Octal PSRAM
static uint8_t* s_gray_buffer = NULL;   // grayscale đã decode (MAX_IMG_W*MAX_IMG_H)
#if HAS_ESP_NEW_JPEG
static uint8_t* s_rgb_buffer = NULL;    // RGB888 output của decoder (x3)
#endif

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
    s_rgb_buffer = (uint8_t*) heap_caps_malloc(MAX_IMG_W * MAX_IMG_H * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
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

// Fast embedded JPEG dimension parser (SOF0/SOF1 marker: 0xFF 0xC0/0xC1)
static bool parse_jpeg_dimensions(const uint8_t* data, size_t len, int* out_w, int* out_h) {
    if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false; // Not a valid JPEG SOI
    }

    size_t i = 2;
    while (i < len - 8) {
        if (data[i] == 0xFF) {
            uint8_t marker = data[i + 1];
            if (marker == 0xC0 || marker == 0xC1) {
                *out_h = (data[i + 5] << 8) | data[i + 6];
                *out_w = (data[i + 7] << 8) | data[i + 8];
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

// Bilinear Downsampling & INT8 Quantization Engine
// Guarantees strict 1:1 Aspect Ratio Preservation (scale_x == scale_y)
static void downsample_and_quantize(const uint8_t* src_gray, int src_w, int src_h,
                                    int8_t* out_tensor,
                                    float input_scale, int32_t input_zp) {
    // Determine square crop bounds
    int S = (src_w < src_h) ? src_w : src_h;
    int x0 = (src_w - S) / 2;
    int y0 = (src_h - S) / 2;

    float step = (float)S / (float)TARGET_WIDTH; // Identical step in X and Y (scale_x == scale_y)

    for (int dst_y = 0; dst_y < TARGET_HEIGHT; dst_y++) {
        float src_y_f = y0 + dst_y * step;
        int sy0 = (int)floorf(src_y_f);
        int sy1 = (sy0 + 1 < src_h) ? sy0 + 1 : sy0;
        float dy = src_y_f - (float)sy0;

        for (int dst_x = 0; dst_x < TARGET_WIDTH; dst_x++) {
            float src_x_f = x0 + dst_x * step;
            int sx0 = (int)floorf(src_x_f);
            int sx1 = (sx0 + 1 < src_w) ? sx0 + 1 : sx0;
            float dx = src_x_f - (float)sx0;

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
    config.rotate = JPEG_ROTATE_0;

    jpeg_dec_handle_t jpeg_dec = NULL;
    jpeg_error_t err = jpeg_dec_open(&config, &jpeg_dec);
    if (err != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg_dec_open lỗi: %d", (int)err);
        return false;
    }

    jpeg_dec_io_t jpeg_io = {0};
    jpeg_io.inbuf = (unsigned char*)jpeg_data;
    jpeg_io.inbuf_len = jpeg_len;

    jpeg_dec_header_info_t jpeg_info = {0};
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

    // Outbuf bắt buộc align 16-byte (FAQ esp_new_jpeg: tránh ảnh bị lệch cột)
    size_t rgb_size = (size_t)img_w * img_h * 3;
    jpeg_io.outbuf = (unsigned char*)jpeg_calloc_align(rgb_size, 16);
    if (jpeg_io.outbuf == NULL) {
        ESP_LOGE(TAG, "Không cấp phát được outbuf %u bytes", (unsigned)rgb_size);
        jpeg_dec_close(jpeg_dec);
        return false;
    }

    err = jpeg_dec_process(jpeg_dec, &jpeg_io);
    if (err != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg_dec_process lỗi: %d", (int)err);
        jpeg_free_align(jpeg_io.outbuf);
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

    jpeg_free_align(jpeg_io.outbuf);
    jpeg_dec_close(jpeg_dec);

    *out_w = img_w;
    *out_h = img_h;
    return true;
}
#endif

bool image_decoder_process_jpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                                int8_t* out_int8_tensor,
                                float input_scale, int32_t input_zero_point) {
    if (!jpeg_data || jpeg_len == 0 || !out_int8_tensor) {
        return false;
    }

    // Quick SOI validation
    if (jpeg_len < 4 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8) {
        ESP_LOGW(TAG, "Frame không phải JPEG hợp lệ (thiếu SOI marker)");
        return false;
    }

    int img_w = 0, img_h = 0;

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

    // Isomorphic Downsampling to 96x96 INT8 (scale_x == scale_y tuyệt đối)
    downsample_and_quantize(s_gray_buffer, img_w, img_h,
                            out_int8_tensor,
                            input_scale, input_zero_point);

    return true;
}
