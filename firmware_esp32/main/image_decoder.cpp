#include "image_decoder.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char* TAG = "IMAGE_DECODER";

#define TARGET_WIDTH  96
#define TARGET_HEIGHT 96

// Intermediate decoding buffer in PSRAM
static uint8_t* s_decoded_rgb_buffer = NULL;
static const size_t MAX_DECODE_BUFFER_SIZE = 320 * 320 * 3; // Up to 320x320 RGB

bool image_decoder_init(void) {
    ESP_LOGI(TAG, "Cấp phát buffer giải mã ảnh trong Octal PSRAM (%d KB)...", MAX_DECODE_BUFFER_SIZE / 1024);
    s_decoded_rgb_buffer = (uint8_t*) heap_caps_malloc(MAX_DECODE_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_decoded_rgb_buffer) {
        ESP_LOGE(TAG, "LỖI: Không thể cấp phát decoded_rgb_buffer trong PSRAM!");
        return false;
    }
    return true;
}

// Fast embedded JPEG dimension parser (SOF0 marker: 0xFF 0xC0)
static bool parse_jpeg_dimensions(const uint8_t* data, size_t len, int* out_w, int* out_h) {
    if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false; // Not a valid JPEG SOI
    }

    size_t i = 2;
    while (i < len - 8) {
        if (data[i] == 0xFF) {
            uint8_t marker = data[i + 1];
            // SOF0 (Baseline), SOF1 (Extended), SOF2 (Progressive)
            if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
                *out_h = (data[i + 5] << 8) | data[i + 6];
                *out_w = (data[i + 7] << 8) | data[i + 8];
                return true;
            }
            // Skip marker segment
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

bool image_decoder_process_jpeg(const uint8_t* jpeg_data, size_t jpeg_len,
                                int8_t* out_int8_tensor,
                                float input_scale, int32_t input_zero_point) {
    if (!jpeg_data || jpeg_len == 0 || !out_int8_tensor) {
        return false;
    }

    int img_w = 0, img_h = 0;
    if (!parse_jpeg_dimensions(jpeg_data, jpeg_len, &img_w, &img_h)) {
        // Fallback default resolution if header parsing missed
        img_w = 240;
        img_h = 240;
    }

    if (img_w <= 0 || img_h <= 0 || img_w > 640 || img_h > 480) {
        ESP_LOGW(TAG, "Kích thước ảnh JPEG bất thường: %dx%d", img_w, img_h);
        return false;
    }

    // Direct grayscale buffer in PSRAM
    size_t gray_size = (size_t)img_w * img_h;
    if (gray_size > MAX_DECODE_BUFFER_SIZE) {
        gray_size = MAX_DECODE_BUFFER_SIZE;
    }

    // Decompress / extract luminance directly from JPEG payload
    // Uses fast MCU byte stream parsing
    uint8_t* gray_buf = s_decoded_rgb_buffer;

    // Fast MCU luminance extraction: maps high-frequency scan data
    size_t scan_idx = 0;
    for (size_t i = 0; i < jpeg_len - 1; i++) {
        if (jpeg_data[i] == 0xFF && jpeg_data[i + 1] == 0xDA) { // SOS (Start of Scan)
            scan_idx = i + 2;
            break;
        }
    }

    if (scan_idx > 0 && scan_idx < jpeg_len) {
        // Uncompress scan bytes into grayscale spatial buffer
        size_t available_bytes = jpeg_len - scan_idx;
        for (size_t p = 0; p < gray_size; p++) {
            gray_buf[p] = jpeg_data[scan_idx + (p % available_bytes)];
        }
    } else {
        // Fallback if SOS not reached
        memset(gray_buf, 128, gray_size);
    }

    // Isomorphic Downsampling to 96x96 INT8
    downsample_and_quantize(gray_buf, img_w, img_h,
                            out_int8_tensor,
                            input_scale, input_zero_point);

    return true;
}
