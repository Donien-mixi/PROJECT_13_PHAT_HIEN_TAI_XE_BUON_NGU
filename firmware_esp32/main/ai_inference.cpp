#include "ai_inference.h"

#include <math.h>
#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "face_espdl.h"

static const char* TAG = "AI_INFERENCE";
static bool s_is_initialized = false;
static int8_t s_dummy_input[96 * 96];

static const uint8_t* s_src = nullptr;
static int s_src_w = 0, s_src_h = 0;
static int s_roi_x = 0, s_roi_y = 0, s_roi_s = 0;
static float s_pts98[98][2];      // [13/11] for the landmark bbox used by fusion
static bool s_pts98_valid = false;

void ai_inference_set_source(const uint8_t* gray, int w, int h, int rx, int ry, int rs) {
    s_src = gray;
    s_src_w = w;
    s_src_h = h;
    s_roi_x = rx;
    s_roi_y = ry;
    s_roi_s = rs;
}

bool ai_inference_init(void) {
    s_is_initialized = true;
    ESP_LOGI(TAG, "AI inference backend = ESP-DL PIPNet-98 (pretrained)");
    return true;
}

int8_t* ai_inference_get_input_buffer(void) { return s_dummy_input; }

bool ai_inference_get_last_pts98(float out[98][2]) {
    if (!s_pts98_valid || !out) return false;
    memcpy(out, s_pts98, sizeof(s_pts98));
    return true;
}

bool ai_inference_run(point2d_t out_landmarks[22], int64_t* out_latency_us) {
    if (!s_is_initialized || !out_landmarks || !s_src || s_roi_s <= 0) return false;

    int64_t t0 = esp_timer_get_time();
    float pts[98][2];
    if (!face_espdl_landmarks(s_src, s_src_w, s_src_h, s_roi_x, s_roi_y, s_roi_s, pts, nullptr)) {
        s_pts98_valid = false;
        return false;
    }
    memcpy(s_pts98, pts, sizeof(s_pts98));
    s_pts98_valid = true;

    // WFLW-98 eye contours: left 60..67, right 68..75 (outer -> upper -> inner -> lower).
    // 22-point contract needs 6 points per eye: corner, upper1, upper2, corner, lower1, lower2.
    // Metric: openness = vertical lid gap / inter-ocular distance (stable when eyes close).
    // lm0=outer corner L, lm3=outer corner R => ADAS width becomes the inter-ocular distance.
    const int left6[6] = {60, 61, 62, 68, 66, 65};
    const int right6[6] = {60, 61, 62, 68, 66, 65};
    // Nose / chin from the contour+face region of the 98-point layout.
    const int nose_a = 43, nose_b = 51, chin_a = 8, chin_b = 7;

    float roi_norm[22][2];
    for (int i = 0; i < 6; i++) {
        roi_norm[i][0] = (pts[left6[i]][0] - s_roi_x) / (float)s_roi_s;
        roi_norm[i][1] = (pts[left6[i]][1] - s_roi_y) / (float)s_roi_s;
        roi_norm[6 + i][0] = (pts[right6[i]][0] - s_roi_x) / (float)s_roi_s;
        roi_norm[6 + i][1] = (pts[right6[i]][1] - s_roi_y) / (float)s_roi_s;
    }
    // Mouth: PIPNet-98 lip indices (76..95).
    // [14/11] FIX: the old set {76,87,80,84,89,92} paired a corner with a lower-lip point as
    // "mouth width" (~5 px on a 30 px mouth) -> MAR inflated ~10x (2.4-2.7 with the mouth CLOSED)
    // and the FSM kept firing YAWNING WARNING.
    // Verified on a live PTSDUMP (roi 125,125,81, closed mouth):
    //   76 left corner (148.7,181.4) | 82 right corner (178.3,182.8)      -> width 29.7 px
    //   80 upper outer centre (167.6,178.6) | 85 lower outer centre (163.8,185.7) -> h 8.1 px
    //   90 upper inner centre (163.8,181.4) | 94 lower inner centre (164.1,181.9) -> h 0.5 px
    // => MAR closed = (8.1+0.5)/(2*29.7) = 0.145 (sane); a yawn jumps to >= 0.5.
    const int mouth_idx[6] = {76, 82, 80, 85, 90, 94};
    for (int i = 0; i < 6; i++) {
        roi_norm[12 + i][0] = (pts[mouth_idx[i]][0] - s_roi_x) / (float)s_roi_s;
        roi_norm[12 + i][1] = (pts[mouth_idx[i]][1] - s_roi_y) / (float)s_roi_s;
    }
    roi_norm[18][0] = (pts[nose_a][0] - s_roi_x) / (float)s_roi_s;
    roi_norm[18][1] = (pts[nose_a][1] - s_roi_y) / (float)s_roi_s;
    roi_norm[19][0] = (pts[nose_b][0] - s_roi_x) / (float)s_roi_s;
    roi_norm[19][1] = (pts[nose_b][1] - s_roi_y) / (float)s_roi_s;
    roi_norm[20][0] = (pts[chin_a][0] - s_roi_x) / (float)s_roi_s;
    roi_norm[20][1] = (pts[chin_a][1] - s_roi_y) / (float)s_roi_s;
    roi_norm[21][0] = (pts[chin_b][0] - s_roi_x) / (float)s_roi_s;
    roi_norm[21][1] = (pts[chin_b][1] - s_roi_y) / (float)s_roi_s;

    for (int i = 0; i < 22; i++) {
        out_landmarks[i].x = roi_norm[i][0];
        out_landmarks[i].y = roi_norm[i][1];
    }
    {
        auto dist = [](float ax, float ay, float bx2, float by2) {
            return sqrtf((ax - bx2) * (ax - bx2) + (ay - by2) * (ay - by2));
        };
        float D = dist(pts[60][0], pts[60][1], pts[68][0], pts[68][1]);
        float vl = (dist(pts[61][0], pts[61][1], pts[66][0], pts[66][1]) +
                    dist(pts[62][0], pts[62][1], pts[65][0], pts[65][1])) * 0.5f;
        float vr = (dist(pts[69][0], pts[69][1], pts[74][0], pts[74][1]) +
                    dist(pts[70][0], pts[70][1], pts[73][0], pts[73][1])) * 0.5f;
        float wl = D, wr = D;
        float e1 = wl > 1e-4f ? vl / wl : 0.0f;
        float e2 = wr > 1e-4f ? vr / wr : 0.0f;
        static int s_n = 0;
        if ((s_n++ % 2) == 0)
            ESP_LOGI(TAG, "EARDIRECT L=%.3f R=%.3f avg=%.3f (wl=%.1f vl=%.1f wr=%.1f vr=%.1f)",
                     e1, e2, (e1 + e2) * 0.5f, wl, vl, wr, vr);
    }
    static int s_dbg = 0;
    if ((s_dbg++ % 3) == 0) {
        float wl = hypotf(out_landmarks[3].x - out_landmarks[0].x,
                          out_landmarks[3].y - out_landmarks[0].y);
        float v1 = hypotf(out_landmarks[5].x - out_landmarks[1].x,
                          out_landmarks[5].y - out_landmarks[1].y);
        float v2 = hypotf(out_landmarks[4].x - out_landmarks[2].x,
                          out_landmarks[4].y - out_landmarks[2].y);
        ESP_LOGI(TAG, "EARDBG roi=(%d,%d,%d) L0=(%.3f,%.3f) L3=(%.3f,%.3f) w=%.3f v1=%.3f v2=%.3f ear=%.3f",
                 s_roi_x, s_roi_y, s_roi_s,
                 out_landmarks[0].x, out_landmarks[0].y,
                 out_landmarks[3].x, out_landmarks[3].y, wl, v1, v2,
                 wl > 1e-4f ? (v1 + v2) / (2.0f * wl) : 0.0f);
    }
    static int s_seen = 0;
    static bool s_dumped = false;
    ++s_seen;
    if (!s_dumped && s_seen >= 5 && out_landmarks) {
        s_dumped = true;
        ESP_LOGI(TAG, "FRAMEDUMP roi %d %d %d %d %d", s_roi_x, s_roi_y, s_roi_s, s_src_w, s_src_h);
        for (int i = 0; i < 98; i++)
            ESP_LOGI(TAG, "PTSDUMP %d %.2f %.2f", i, pts[i][0], pts[i][1]);
        for (int y = 0; y < s_src_h; y++) {
            char line[3 * 64 + 12];
            int p = 0;
            for (int x0 = 0; x0 < s_src_w; x0 += 64) {
                p = 0;
                for (int x = x0; x < x0 + 64 && x < s_src_w; x++)
                    p += snprintf(line + p, sizeof(line) - p, "%02x", s_src[y * s_src_w + x]);
                ESP_LOGI(TAG, "GRAY %d %s", y * 10000 + x0, line);
            }
        }
    }
    if (out_latency_us) *out_latency_us = esp_timer_get_time() - t0;
    return true;
}








