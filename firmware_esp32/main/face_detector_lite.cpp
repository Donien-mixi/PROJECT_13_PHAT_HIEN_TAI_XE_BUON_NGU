#include "face_detector_lite.h"

#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "face_detector_lite_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "FD_LITE";

#define FDL_IN        FDL_INPUT_W          // 96
#define FDL_ARENA_KB  200                  // PSRAM (model nhỏ)

static const tflite::Model* s_model = NULL;
static tflite::MicroInterpreter* s_interp = NULL;
static TfLiteTensor* s_in = NULL;
static TfLiteTensor* s_out = NULL;
static bool s_ready = false;
static float s_last_ms = 0.0f;

bool face_detector_lite_init(void) {
    s_model = tflite::GetModel(g_face_detector_lite);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch");
        return false;
    }
    // Model lite (Keras3) chua: CONV_2D, DEPTHWISE_CONV_2D, SHAPE, STRIDED_SLICE, PACK,
    // RESHAPE, FULLY_CONNECTED, DEQUANTIZE (SHAPE/STRIDED_SLICE/PACK do lop Flatten sinh ra)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddDequantize();
    resolver.AddAdd();
    resolver.AddMul();

    size_t bytes = (size_t)FDL_ARENA_KB * 1024;
    uint8_t* arena = (uint8_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) {
        ESP_LOGE(TAG, "Khong cap phat duoc arena PSRAM (%u KB)", (unsigned)FDL_ARENA_KB);
        return false;
    }
    static tflite::MicroInterpreter interp(s_model, resolver, arena, bytes);
    s_interp = &interp;
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors FAILED (used %d)", (int)s_interp->arena_used_bytes());
        return false;
    }
    s_in = s_interp->input(0);
    s_out = s_interp->output(0);
    s_ready = true;
    ESP_LOGI(TAG, "FaceDetectorLite san sang | arena dung %d KB / %u KB PSRAM | in %dx%dx%d %s | out %d %s",
             (int)(s_interp->arena_used_bytes() / 1024), (unsigned)FDL_ARENA_KB,
             s_in->dims->data[1], s_in->dims->data[2], s_in->dims->data[3],
             (s_in->type == kTfLiteInt8) ? "INT8" : "?", s_out->dims->data[1],
             (s_out->type == kTfLiteFloat32) ? "float32" : "?");
    return true;
}

float face_detector_lite_last_ms(void) { return s_last_ms; }

bool face_detector_lite_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out) {
    if (out) { out->valid = false; out->has_kp = false; }
    if (!s_ready || !gray || !out || src_w <= 0 || src_h <= 0) return false;

    int64_t t0 = esp_timer_get_time();

    // Nạp input 96x96: lấy mẫu bilinear toàn khung -> chuẩn hoá (px-127.5)/128 -> INT8
    const float sc = (s_in->params.scale > 0.0f) ? s_in->params.scale : (1.0f / 128.0f);
    const int   zp = s_in->params.zero_point;
    const float sx = (float)src_w / (float)FDL_IN;
    const float sy = (float)src_h / (float)FDL_IN;
    for (int dy = 0; dy < FDL_IN; dy++) {
        float fy = (dy + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float ay = fy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y0 > src_h - 1) y0 = src_h - 1;
        if (y1 < 0) y1 = 0;
        if (y1 > src_h - 1) y1 = src_h - 1;
        for (int dx = 0; dx < FDL_IN; dx++) {
            float fx = (dx + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float ax = fx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x0 > src_w - 1) x0 = src_w - 1;
            if (x1 < 0) x1 = 0;
            if (x1 > src_w - 1) x1 = src_w - 1;
            float p00 = gray[y0 * src_w + x0], p01 = gray[y0 * src_w + x1];
            float p10 = gray[y1 * src_w + x0], p11 = gray[y1 * src_w + x1];
            float v = (p00 * (1 - ax) + p01 * ax) * (1 - ay) + (p10 * (1 - ax) + p11 * ax) * ay;
            float n = (v - 127.5f) / 128.0f;
            int q = (int)lroundf(n / sc) + zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            if (s_in->type == kTfLiteInt8) s_in->data.int8[dy * FDL_IN + dx] = (int8_t)q;
            else if (s_in->type == kTfLiteUInt8) s_in->data.uint8[dy * FDL_IN + dx] = (uint8_t)(q + 128);
            else s_in->data.f[dy * FDL_IN + dx] = n;
        }
    }

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke FAILED");
        s_last_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
        return false;
    }

    const float* o = s_out->data.f;
    float cx = o[0] * FDL_STD[0] + FDL_MEAN[0];
    float cy = o[1] * FDL_STD[1] + FDL_MEAN[1];
    float bw = o[2] * FDL_STD[2] + FDL_MEAN[2];
    float bh = o[3] * FDL_STD[3] + FDL_MEAN[3];

    s_last_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;

    bool ok = isfinite(cx) && isfinite(cy) && isfinite(bw) && isfinite(bh) &&
              bw > 0.02f && bh > 0.02f && bw < 1.6f && bh < 1.6f &&
              cx > -0.1f && cx < 1.1f && cy > -0.1f && cy < 1.1f;
    if (!ok) return false;

    out->cx = cx * (float)src_w;
    out->cy = cy * (float)src_h;
    out->size = fmaxf(bw * (float)src_w, bh * (float)src_h);
    out->score = 1.0f;
    out->has_kp = false;
    out->valid = true;
    return true;
}
