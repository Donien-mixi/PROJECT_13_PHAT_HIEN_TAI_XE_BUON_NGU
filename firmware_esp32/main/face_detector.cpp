#include "face_detector.h"

#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "blazeface_int8_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "FACE_DETECTOR";

#define DET_IN        128
#define DET_ANCHORS   896
#define DET_ARENA_KB  700     // PSRAM (project 5 đo ~602KB)

static const tflite::Model* s_model = NULL;
static tflite::MicroInterpreter* s_interp = NULL;
static TfLiteTensor* s_in = NULL;
static TfLiteTensor* s_scores = NULL;
static TfLiteTensor* s_boxes = NULL;

static float s_anchors[DET_ANCHORS][2];
static float s_conf_thr = 0.75f;
static bool  s_ready = false;

static inline float sigmoid_f(float x) {
    if (x < -80.0f) x = -80.0f;
    if (x > 80.0f) x = 80.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float tensor_val(const TfLiteTensor* t, int i) {
    if (t->type == kTfLiteFloat32) {
        return t->data.f[i];
    } else if (t->type == kTfLiteInt8) {
        return ((float)t->data.int8[i] - (float)t->params.zero_point) * t->params.scale;
    } else if (t->type == kTfLiteUInt8) {
        return ((float)t->data.uint8[i] - (float)t->params.zero_point) * t->params.scale;
    }
    return 0.0f;
}

bool face_detector_init(void) {
    s_model = tflite::GetModel(g_blazeface_model);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch");
        return false;
    }

    static tflite::MicroMutableOpResolver<8> resolver;
    // AddConv2D()/AddDepthwiseConv2D() chuẩn -> component esp-tflite-micro đã tự
    // thay bằng kernel esp-nn SIMD (-DESP_NN). Không cần glue riêng.
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddMaxPool2D();
    resolver.AddPad();
    resolver.AddReshape();

    size_t arena_bytes = (size_t)DET_ARENA_KB * 1024;
    uint8_t* arena = (uint8_t*)heap_caps_malloc(arena_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) {
        ESP_LOGE(TAG, "Khong cap phat duoc arena PSRAM (%u KB)", (unsigned)DET_ARENA_KB);
        return false;
    }

    static tflite::MicroInterpreter interp(s_model, resolver, arena, arena_bytes);
    s_interp = &interp;
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors FAILED (arena %d / %u)", (int)s_interp->arena_used_bytes(), (unsigned)arena_bytes);
        return false;
    }

    s_in = s_interp->input(0);
    TfLiteTensor* o0 = s_interp->output(0);
    TfLiteTensor* o1 = s_interp->output(1);
    if (o0->dims->data[o0->dims->size - 1] == 1) {
        s_scores = o0; s_boxes = o1;
    } else {
        s_scores = o1; s_boxes = o0;
    }

    // 896 anchors = 16x16 x2 + 8x8 x6
    int k = 0;
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            float cx = (x + 0.5f) / 16.0f, cy = (y + 0.5f) / 16.0f;
            s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++;
            s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++;
        }
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            float cx = (x + 0.5f) / 8.0f, cy = (y + 0.5f) / 8.0f;
            for (int i = 0; i < 6; i++) { s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++; }
        }

    s_ready = true;
    ESP_LOGI(TAG, "BlazeFace INT8 san sang | arena dung %d KB / %u KB PSRAM | in %dx%dx%d %s",
             (int)(s_interp->arena_used_bytes() / 1024), (unsigned)DET_ARENA_KB,
             s_in->dims->data[1], s_in->dims->data[2], s_in->dims->data[3],
             (s_in->type == kTfLiteInt8) ? "INT8" : "other");
    return true;
}

void face_detector_set_threshold(float thr) {
    if (thr > 0.0f && thr < 1.0f) s_conf_thr = thr;
}

bool face_detector_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out) {
    if (out) { out->valid = false; out->score = 0.0f; }
    if (!s_ready || !gray || !out || src_w <= 0 || src_h <= 0) return false;

    // Vùng detector nhìn = crop vuông GIỮA khung (giống lúc đánh giá D1)
    int S = (src_w < src_h) ? src_w : src_h;
    int x0 = (src_w - S) / 2;
    int y0 = (src_h - S) / 2;

    // Nạp input 128x128x3 (xám -> RGB lặp) với normalize (px-127.5)/128
    const float sc = (s_in->params.scale > 0.0f) ? s_in->params.scale : (1.0f / 128.0f);
    const int   zp = s_in->params.zero_point;
    for (int dy = 0; dy < DET_IN; dy++) {
        int sy = y0 + (dy * S) / DET_IN;
        if (sy > src_h - 1) sy = src_h - 1;
        for (int dx = 0; dx < DET_IN; dx++) {
            int sx = x0 + (dx * S) / DET_IN;
            if (sx > src_w - 1) sx = src_w - 1;
            float v = (float)gray[sy * src_w + sx];
            float n = (v - 127.5f) / 128.0f;
            int q = (int)lroundf(n / sc) + zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            int idx = (dy * DET_IN + dx) * 3;
            if (s_in->type == kTfLiteInt8) {
                s_in->data.int8[idx + 0] = (int8_t)q;
                s_in->data.int8[idx + 1] = (int8_t)q;
                s_in->data.int8[idx + 2] = (int8_t)q;
            } else if (s_in->type == kTfLiteUInt8) {
                s_in->data.uint8[idx + 0] = (uint8_t)q;
                s_in->data.uint8[idx + 1] = (uint8_t)q;
                s_in->data.uint8[idx + 2] = (uint8_t)q;
            } else {
                s_in->data.f[idx + 0] = n;
                s_in->data.f[idx + 1] = n;
                s_in->data.f[idx + 2] = n;
            }
        }
    }

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke FAILED");
        return false;
    }

    int best = -1;
    float best_score = -1.0f;
    for (int i = 0; i < DET_ANCHORS; i++) {
        float s = sigmoid_f(tensor_val(s_scores, i));
        if (s > best_score) { best_score = s; best = i; }
    }
    if (best < 0 || best_score < s_conf_thr) {
        return false;
    }

    const int stride = s_boxes->dims->data[s_boxes->dims->size - 1];
    if (stride < 4) return false;
    float dx = tensor_val(s_boxes, best * stride + 0);
    float dy = tensor_val(s_boxes, best * stride + 1);
    float dw = tensor_val(s_boxes, best * stride + 2);
    float dh = tensor_val(s_boxes, best * stride + 3);

    float cx128 = dx + s_anchors[best][0] * (float)DET_IN;
    float cy128 = dy + s_anchors[best][1] * (float)DET_IN;
    if (!isfinite(cx128) || !isfinite(cy128) || dw <= 0.0f || dh <= 0.0f) return false;

    // Map 128 -> khung gốc
    float k = (float)S / (float)DET_IN;
    out->cx = (float)x0 + cx128 * k;
    out->cy = (float)y0 + cy128 * k;
    out->size = fmaxf(dw, dh) * k;
    out->score = best_score;
    out->valid = (out->size > 1.0f);

    // [D3] Giải mã 6 keypoints (offsets pixel so với anchor) -> khung gốc
    out->has_kp = false;
    if (stride >= 16) {
        bool ok = true;
        for (int i = 0; i < 6; i++) {
            float kx = tensor_val(s_boxes, best * stride + 4 + i * 2) + s_anchors[best][0] * (float)DET_IN;
            float ky = tensor_val(s_boxes, best * stride + 5 + i * 2) + s_anchors[best][1] * (float)DET_IN;
            if (!isfinite(kx) || !isfinite(ky)) { ok = false; break; }
            out->kp[i][0] = (float)x0 + kx * k;
            out->kp[i][1] = (float)y0 + ky * k;
        }
        if (ok) {
            float d_eyes = hypotf(out->kp[1][0] - out->kp[0][0], out->kp[1][1] - out->kp[0][1]);
            out->has_kp = (d_eyes > 4.0f);   // keypoints đủ xa nhau mới tin
        }
    }
    return out->valid;
}
