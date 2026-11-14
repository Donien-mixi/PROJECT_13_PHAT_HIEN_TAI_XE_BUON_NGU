// BlazeFace short-range (MediaPipe) INT8 detector + ROI tracker.
//
// Model: output/port_models_20261110/blaze_iq/blazeface_nchw_full_integer_quant.tflite
//        (onnx2tf -oiqt from blazeface_nchw.onnx, calibrated on real QVGA frames)
//        -> firmware_esp32/main/blaze_iq_model_data.h (187,416 B rodata)
// Runtime: TFLM (esp-tflite-micro). Input/output scale + zero-point are read from
// the tensors, never hard-coded. Preprocessing matches the offline parity script:
// letterbox to 128x128 (grey, pad 0), x -> (pixel/127.5 - 1).
//
// Decode: 896 anchors (16x16x2 + 8x8x6), cx = raw + anchor*128, w/h = raw pixels,
// sigmoid on the classifier logits, NMS IoU 0.3 -> top-1.
//
// Tracker (roi_tracker-derived): EMA on cx/cy/size + velocity feed-forward, size
// gate to reject jumps, hold-last-good for up to 8 frames.

#include "blaze_tracker.h"

#include <math.h>
#include <string.h>

#include <new>
#include "blaze_iq_model_data.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

static const char* TAG = "BLAZE";

#define DET_IN 128
#define DET_ANCHORS 896
#define MAX_CAND 32

// Candidate collection is deliberately looser than the acquisition gate so the
// serial log shows how many anchors fire (evidence for calibration/decode).
#define CAND_SCORE 0.25f
// Empty-scene runs (13/11) showed the model answering exactly 0.500 (logit 0 = the
// classifier zero-point) with a box covering the whole frame. Real faces measured
// 0.855-0.972, so acquire at 0.60 and reject implausible box sizes.
#define ACQ_SCORE 0.60f
#define KEEP_SCORE 0.30f
#define TRK_MAX_SIZE_DELTA 0.40f
#define TRK_HOLD_FRAMES 8
#define BOX_MIN_PX 8.0f      // in 128-input space
#define BOX_MAX_PX 100.0f    // >78 % of the input: not a face at 60-70 cm

// TFLM arena in PSRAM. Measured on hardware: AllocateTensors asks for 574,688 B and the
// graph reports arena_used = 602,648 B with a 640 KB arena. TFLM documents the requirement
// as arena_used_bytes() + 16, so try that exact size first and keep 640 KB as fallback.
static const size_t kArenaTry[] = {602664, 640 * 1024, 768 * 1024};

static uint8_t* s_arena = nullptr;
static size_t s_arena_bytes = 0;
static tflite::MicroInterpreter* s_interp = nullptr;
static const TfLiteTensor* s_in = nullptr;
static const TfLiteTensor* s_out_cls = nullptr;
static const TfLiteTensor* s_out_reg = nullptr;
static float s_anchors[DET_ANCHORS][2];
static bool s_ready = false;

static float s_last_ms = 0.0f;
static int s_last_cand = 0;
static float s_last_score = 0.0f;
static int s_arena_used = 0;

// Tracker state
static bool s_have = false;
static float s_cx, s_cy, s_size, s_vx, s_vy;
static int s_miss = 0;
static float s_last_dt = 1.0f;

static void build_anchors(void) {
    int k = 0;
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            float cx = (x + 0.5f) / 16.0f, cy = (y + 0.5f) / 16.0f;
            s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++;
            s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++;
        }
    }
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            float cx = (x + 0.5f) / 8.0f, cy = (y + 0.5f) / 8.0f;
            for (int i = 0; i < 6; i++) { s_anchors[k][0] = cx; s_anchors[k][1] = cy; k++; }
        }
    }
}

static inline float sigmoid_f(float x) {
    if (x < -60.0f) x = -60.0f;
    if (x > 60.0f) x = 60.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float iou(const float a[4], const float b[4]) {
    float x0 = fmaxf(a[0], b[0]), y0 = fmaxf(a[1], b[1]);
    float x1 = fminf(a[2], b[2]), y1 = fminf(a[3], b[3]);
    float iw = fmaxf(0.0f, x1 - x0), ih = fmaxf(0.0f, y1 - y0);
    float inter = iw * ih;
    float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 1e-6f ? inter / ua : 0.0f;
}

bool blaze_init(void) {
    if (s_ready && s_interp) return true;   // idempotent

    build_anchors();

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddPad();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddConcatenation();

    const tflite::Model* model = tflite::GetModel(g_blaze_iq_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "schema mismatch: model=%d runtime=%d", (int)model->version(),
                 (int)TFLITE_SCHEMA_VERSION);
        return false;
    }

    for (size_t i = 0; i < sizeof(kArenaTry) / sizeof(kArenaTry[0]); i++) {
        size_t bytes = kArenaTry[i];
        uint8_t* arena = (uint8_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!arena) {
            ESP_LOGW(TAG, "arena alloc %u B failed (psram free %u KB)",
                     (unsigned)bytes, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10));
            continue;
        }
        auto* interp = new (std::nothrow) tflite::MicroInterpreter(
            model, resolver, arena, bytes, nullptr, nullptr);
        if (!interp) {
            heap_caps_free(arena);
            continue;
        }
        if (interp->AllocateTensors() != kTfLiteOk) {
            ESP_LOGW(TAG, "AllocateTensors failed with arena %u B", (unsigned)bytes);
            delete interp;
            heap_caps_free(arena);
            continue;
        }
        s_arena = arena;
        s_arena_bytes = bytes;
        s_interp = interp;
        s_arena_used = (int)interp->arena_used_bytes();
        break;
    }
    if (!s_interp) {
        ESP_LOGE(TAG, "BlazeFace int8 init FAILED (no arena fits)");
        return false;
    }

    if (s_interp->inputs().size() != 1) {
        ESP_LOGE(TAG, "unexpected input count %d", (int)s_interp->inputs().size());
        return false;
    }
    s_in = s_interp->input(0);
    if (s_in->type != kTfLiteInt8 || s_in->dims->size != 4 ||
        s_in->dims->data[1] != DET_IN || s_in->dims->data[2] != DET_IN ||
        s_in->dims->data[3] != 3) {
        ESP_LOGE(TAG, "unexpected input tensor");
        return false;
    }
    for (int i = 0; i < (int)s_interp->outputs().size(); i++) {
        const TfLiteTensor* t = s_interp->output(i);
        if (t->type != kTfLiteInt8 || t->dims->size != 3) continue;
        int last = t->dims->data[2];
        if (last == 1) s_out_cls = t;
        else if (last == 16) s_out_reg = t;
    }
    if (!s_out_cls || !s_out_reg) {
        ESP_LOGE(TAG, "outputs missing (cls=%p reg=%p)", (const void*)s_out_cls,
                 (const void*)s_out_reg);
        return false;
    }

    ESP_LOGI(TAG, "BlazeFace int8 ready: in %dx%dx%d int8 s=%.6f zp=%d | cls s=%.6f zp=%d | "
                  "reg s=%.6f zp=%d | arena %d/%u B | psram %u KB",
             (int)s_in->dims->data[1], (int)s_in->dims->data[2], (int)s_in->dims->data[3],
             (double)s_in->params.scale, (int)s_in->params.zero_point,
             (double)s_out_cls->params.scale, (int)s_out_cls->params.zero_point,
             (double)s_out_reg->params.scale, (int)s_out_reg->params.zero_point,
             s_arena_used, (unsigned)s_arena_bytes,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10));
    s_ready = true;
    return true;
}

// Letterbox + quantize a grey frame into the model input tensor.
// Returns the letterbox scale so the caller can map boxes back to pixels.
static float prep_input(const uint8_t* gray, int w, int h, int* padx_out, int* pady_out) {
    float lb = (float)DET_IN / (float)(w > h ? w : h);
    int nw = (int)(w * lb + 0.5f), nh = (int)(h * lb + 0.5f);
    int padx = (DET_IN - nw) / 2, pady = (DET_IN - nh) / 2;
    if (padx_out) *padx_out = padx;
    if (pady_out) *pady_out = pady;
    const float qs = s_in->params.scale;
    const int qz = s_in->params.zero_point;
    int8_t* dst = s_in->data.int8;
    // Pad pixels are pixel value 0 -> x = -1.0 -> q = round(-1/qs) + zp (clamped).
    int q_pad = (int)lrintf(-1.0f / qs) + qz;
    if (q_pad < -128) q_pad = -128;
    if (q_pad > 127) q_pad = 127;
    // Bilinear resize with the half-pixel-centre convention used by the offline
    // parity script (PIL resize), so on-board numbers are comparable to it.
    const float sx_scale = (float)w / (float)(nw > 0 ? nw : 1);
    const float sy_scale = (float)h / (float)(nh > 0 ? nh : 1);
    for (int dy = 0; dy < DET_IN; dy++) {
        bool row_ok = (dy >= pady) && (dy < pady + nh);
        float fy = (dy - pady + 0.5f) * sy_scale - 0.5f;
        if (fy < 0.0f) fy = 0.0f;
        if (fy > (float)(h - 1)) fy = (float)(h - 1);
        int y0 = (int)fy;
        int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
        float ay = fy - (float)y0;
        for (int dx = 0; dx < DET_IN; dx++) {
            int q = q_pad;
            if (row_ok && dx >= padx && dx < padx + nw) {
                float fx = (dx - padx + 0.5f) * sx_scale - 0.5f;
                if (fx < 0.0f) fx = 0.0f;
                if (fx > (float)(w - 1)) fx = (float)(w - 1);
                int x0 = (int)fx;
                int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
                float ax = fx - (float)x0;
                float v = (1.0f - ax) * (1.0f - ay) * (float)gray[y0 * w + x0] +
                          ax * (1.0f - ay) * (float)gray[y0 * w + x1] +
                          (1.0f - ax) * ay * (float)gray[y1 * w + x0] +
                          ax * ay * (float)gray[y1 * w + x1];
                float x = v / 127.5f - 1.0f;   // [-1,1]
                q = (int)lrintf(x / qs) + qz;
                if (q < -128) q = -128;
                if (q > 127) q = 127;
            }
            int b = (dy * DET_IN + dx) * 3;
            dst[b] = (int8_t)q; dst[b + 1] = (int8_t)q; dst[b + 2] = (int8_t)q;
        }
    }
    return lb;
}

bool blaze_detect(const uint8_t* gray, int w, int h, float box[4], float* score) {
    if (!s_ready || !s_interp || !gray || !box || w < 8 || h < 8) return false;
    int64_t t0 = esp_timer_get_time();

    int padx = 0, pady = 0;
    float lb = prep_input(gray, w, h, &padx, &pady);

    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return false;
    }

    const int8_t* cd = s_out_cls->data.int8;
    const int8_t* rd = s_out_reg->data.int8;
    const float s_cls = s_out_cls->params.scale;
    const int z_cls = s_out_cls->params.zero_point;
    const float s_reg = s_out_reg->params.scale;
    const int z_reg = s_out_reg->params.zero_point;

    float cs[MAX_CAND][5];
    int n = 0;
    float max_logit = -1e9f;
    for (int i = 0; i < DET_ANCHORS && n < MAX_CAND; i++) {
        float logit = ((float)cd[i] - (float)z_cls) * s_cls;
        if (logit > max_logit) max_logit = logit;
        float sc = sigmoid_f(logit);
        if (sc < CAND_SCORE) continue;
        float dx = ((float)rd[i * 16 + 0] - (float)z_reg) * s_reg;
        float dy = ((float)rd[i * 16 + 1] - (float)z_reg) * s_reg;
        float dw = ((float)rd[i * 16 + 2] - (float)z_reg) * s_reg;
        float dh = ((float)rd[i * 16 + 3] - (float)z_reg) * s_reg;
        if (dw <= 0.5f || dh <= 0.5f) continue;
        if (dw < BOX_MIN_PX || dh < BOX_MIN_PX || dw > BOX_MAX_PX || dh > BOX_MAX_PX) continue;
        float cx = dx + s_anchors[i][0] * (float)DET_IN;
        float cy = dy + s_anchors[i][1] * (float)DET_IN;
        cs[n][0] = cx - dw * 0.5f; cs[n][1] = cy - dh * 0.5f;
        cs[n][2] = cx + dw * 0.5f; cs[n][3] = cy + dh * 0.5f;
        cs[n][4] = sc;
        n++;
    }
    s_last_cand = n;
    if (n == 0) {
        s_last_score = 0.0f;
        s_last_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
        ESP_LOGI(TAG, "cand=0 (no anchor above %.2f, max_logit=%.2f) in %.0f ms",
                 (double)CAND_SCORE, (double)max_logit, (double)s_last_ms);
        return false;
    }

    // ---- sort candidates by score (desc), then greedy NMS IoU 0.3 -> top-1 ----
    // Candidates are collected in anchor order, so sorting first is mandatory:
    // a weak anchor must never suppress a stronger overlapping one.
    for (int i = 1; i < n; i++) {
        float key[5];
        memcpy(key, cs[i], sizeof(key));
        int j = i - 1;
        while (j >= 0 && cs[j][4] < key[4]) { memcpy(cs[j + 1], cs[j], sizeof(key)); j--; }
        memcpy(cs[j + 1], key, sizeof(key));
    }
    bool used[MAX_CAND] = {false};
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (used[i]) continue;
        if (best < 0) best = i;
        for (int j = i + 1; j < n; j++)
            if (!used[j] && iou(cs[i], cs[j]) > 0.3f) used[j] = true;
    }
    if (best < 0) return false;
    float best_s = cs[best][4];

    // ---- map 128x128 letterbox coords back to source pixels ----
    float inv = 1.0f / lb;
    box[0] = (cs[best][0] - (float)padx) * inv;
    box[1] = (cs[best][1] - (float)pady) * inv;
    box[2] = (cs[best][2] - (float)padx) * inv;
    box[3] = (cs[best][3] - (float)pady) * inv;
    if (score) *score = best_s;
    s_last_score = best_s;
    s_last_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
    ESP_LOGI(TAG, "cand=%d score=%.3f box=%.1f,%.1f,%.1f,%.1f ms=%.0f", n, (double)best_s,
             (double)box[0], (double)box[1], (double)box[2], (double)box[3],
             (double)s_last_ms);
    return best_s >= ACQ_SCORE;
}

float blaze_last_ms(void) { return s_last_ms; }
float blaze_box_iou(const float a[4], const float b[4]) { return iou(a, b); }

// ---- fusion state ----
static float s_lm_box[4] = {0, 0, 0, 0};
static bool s_lm_valid = false;
static float s_fusion_iou = -1.0f;
static bool s_fusion_used_lm = false;

void blaze_fusion_set_landmarks(const float box[4], bool valid) {
    if (!valid || !box) { s_lm_valid = false; return; }
    memcpy(s_lm_box, box, sizeof(s_lm_box));
    s_lm_valid = true;
}

bool blaze_fusion_get(float out_det[4], float out_lm[4], float* out_iou, bool* out_used_lm) {
    if (out_lm) memcpy(out_lm, s_lm_box, sizeof(s_lm_box));
    if (out_iou) *out_iou = s_fusion_iou;
    if (out_used_lm) *out_used_lm = s_fusion_used_lm;
    (void)out_det;
    return s_lm_valid;
}int blaze_last_cand(void) { return s_last_cand; }
float blaze_last_score(void) { return s_last_score; }
int blaze_arena_used(void) { return s_arena_used; }

// Diagnostic: dump the quantized input fingerprint and the top classifier logits
// so they can be compared against the host reference run (tools/blaze_parity.py).
void blaze_debug_dump(const uint8_t* gray, int w, int h) {
    if (!s_ready || !s_interp || !gray) return;
    int padx = 0, pady = 0;
    prep_input(gray, w, h, &padx, &pady);
    const int8_t* q = s_in->data.int8;
    uint32_t fnv = 2166136261u;
    for (int i = 0; i < DET_IN * DET_IN * 3; i++) {
        fnv ^= (uint8_t)q[i];
        fnv *= 16777619u;
    }
    ESP_LOGI(TAG, "DUMP in fnv=%08x q0..7=%d,%d,%d,%d,%d,%d,%d,%d",
             (unsigned)fnv, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
    if (s_interp->Invoke() != kTfLiteOk) return;
    const int8_t* cd = s_out_cls->data.int8;
    bool taken[DET_ANCHORS] = {false};
    for (int rank = 0; rank < 5; rank++) {
        int best = -1;
        for (int i = 0; i < DET_ANCHORS; i++) {
            if (taken[i]) continue;
            if (best < 0 || cd[i] > cd[best]) best = i;
        }
        if (best < 0) break;
        taken[best] = true;
        float logit = ((float)cd[best] - (float)s_out_cls->params.zero_point) * s_out_cls->params.scale;
        ESP_LOGI(TAG, "DUMP top%d anchor=%d q=%d logit=%.3f score=%.4f", rank, best,
                 (int)cd[best], (double)logit, (double)sigmoid_f(logit));
    }
    for (int a : {241, 273, 275, 305, 728}) {
        const int8_t* rd = s_out_reg->data.int8;
        ESP_LOGI(TAG, "DUMP anchor=%d qcls=%d reg=%d,%d,%d,%d", a, (int)s_out_cls->data.int8[a],
                 (int)rd[a * 16 + 0], (int)rd[a * 16 + 1], (int)rd[a * 16 + 2], (int)rd[a * 16 + 3]);
    }
    const int8_t* rd = s_out_reg->data.int8;
    int npos = 0;
    for (int i = 0; i < DET_ANCHORS; i++) if (cd[i] > 123) npos++;
    ESP_LOGI(TAG, "DUMP anchors_above_zp=%d", npos);
    (void)rd;
}

void blaze_reset(void) { s_have = false; s_miss = 0; s_vx = s_vy = 0; }

bool blaze_track(float box[4], float score, float dt_s, float out[4]) {
    float cx = 0.5f * (box[0] + box[2]);
    float cy = 0.5f * (box[1] + box[3]);
    float sz = fmaxf(box[2] - box[0], box[3] - box[1]);
    if (dt_s > 0.0f) s_last_dt = dt_s;

    if (score >= KEEP_SCORE) {
        if (s_have) {
            float dsz = fabsf(sz - s_size) / (s_size > 1.0f ? s_size : 1.0f);
            if (dsz > TRK_MAX_SIZE_DELTA) return false;   // reject a jump in scale
        }
        if (!s_have) {
            s_cx = cx; s_cy = cy; s_size = sz; s_have = true;
        } else {
            const float a = 0.55f;
            float nvx = (cx - s_cx) / s_last_dt, nvy = (cy - s_cy) / s_last_dt;
            s_vx = 0.7f * s_vx + 0.3f * nvx;
            s_vy = 0.7f * s_vy + 0.3f * nvy;
            s_cx += a * (cx - s_cx) + s_vx * s_last_dt * 0.3f;
            s_cy += a * (cy - s_cy) + s_vy * s_last_dt * 0.3f;
            s_size += a * (sz - s_size);
        }
        s_miss = 0;
    } else {
        if (!s_have) return false;
        if (++s_miss > TRK_HOLD_FRAMES) return false;
        s_cx += s_vx * s_last_dt;
        s_cy += s_vy * s_last_dt;
    }
    out[0] = s_cx - s_size * 0.5f;
    out[1] = s_cy - s_size * 0.5f;
    out[2] = s_cx + s_size * 0.5f;
    out[3] = s_cy + s_size * 0.5f;
    return true;
}
