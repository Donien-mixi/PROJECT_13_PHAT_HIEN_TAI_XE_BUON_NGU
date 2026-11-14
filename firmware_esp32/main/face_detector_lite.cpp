#include "face_detector_lite.h"

#include <math.h>
#include <string.h>

#include "blaze_tracker.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "face_det_espdl.h"

static const char* TAG = "DETLITE";

#ifndef BLAZE_LIVE_DIAG
#define BLAZE_LIVE_DIAG 1   // log a coarse brightness grid with every live detect
#endif

// [13/11] Fusion threshold from measurement: detector-vs-landmark IoU was 0,873 / 0,831
// on clean stills, so 0,60 means "clearly disagree" rather than "small jitter".
#define FUSION_MIN_IOU 0.60f

static float s_last_ms = 0.0f;
static bool s_init_done = false;
static bool s_ready = false;
static int64_t s_last_us = 0;

bool face_detector_lite_init(void) {
    if (s_init_done) return s_ready;
    s_init_done = true;
    // BlazeFace short-range INT8 (TFLM). 187 KB rodata, arena in PSRAM.
    // The MSR/MNP ESP-DL detector needs far more RAM next to PIPNet, so BlazeFace
    // is the only external detector in this build.
    s_ready = blaze_init();
    if (s_ready) blaze_reset();
    ESP_LOGI(TAG, "blaze detector %s", s_ready ? "READY" : "UNAVAILABLE (centre ROI)");
    return s_ready;
}

bool face_detector_lite_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out) {
    if (!out) return false;
    out->valid = false;

    int64_t t0 = esp_timer_get_time();
    float box[4] = {0, 0, 0, 0};
    float score = 0.0f;
    bool ok = s_ready && blaze_detect(gray, src_w, src_h, box, &score);

#if BLAZE_LIVE_DIAG
    // Coarse view of what the camera actually delivers (16x12 block means) so a
    // "no face" result can be told apart from a dark / blank / misaimed frame.
    {
        int gw = 16, gh = 12;
        int bw = src_w / gw, bh = src_h / gh;
        long sum = 0; int mx = 0, mn = 255;
        char grid[16 * 12 * 4 + 1];
        int gi = 0;
        for (int by = 0; by < gh; by++) {
            for (int bx = 0; bx < gw; bx++) {
                long s2 = 0; int cnt = 0;
                for (int y = by * bh; y < (by + 1) * bh; y += 2)
                    for (int x = bx * bw; x < (bx + 1) * bw; x += 2) {
                        uint8_t v = gray[y * src_w + x];
                        s2 += v; cnt++;
                        if (v > mx) mx = v;
                        if (v < mn) mn = v;
                    }
                int m = cnt ? (int)(s2 / cnt) : 0;
                sum += m;
                grid[gi++] = "0123456789abcdef"[m >> 4];
                if (bx == gw - 1) grid[gi++] = '|';
            }
        }
        grid[gi] = 0;
        ESP_LOGI(TAG, "frame %dx%d mean=%ld min=%d max=%d grid=%s", src_w, src_h,
                 sum / (gw * gh), mn, mx, grid);
    }
#endif

    int64_t now = t0;
    float dt = (s_last_us > 0) ? (float)(now - s_last_us) / 1e6f : 0.0f;
    s_last_us = now;

    float tbox[4] = {0, 0, 0, 0};
    bool tracked = false;
    if (ok) {
        // [13/11] Fusion with the landmark bbox of the previous frame.
        float lm[4] = {0, 0, 0, 0};
        float fiou = -1.0f;
        bool used_lm = false;
        const bool have_lm = blaze_fusion_get(nullptr, lm, &fiou, &used_lm);
        if (have_lm) {
            float det[4] = {box[0], box[1], box[2], box[3]};
            const float iou_dl = blaze_box_iou(det, lm);
            // [13/11] Plausibility gate — ĐO THỰC TẾ buổi 18:10 (mặt ở ~40 cm, box 145 px):
            // bbox landmark rộng gấp **1,67×** (median, max 4,5×) và lệch tâm 0,30–0,45 cạnh box,
            // EAR vọt 0,6–2,7 ⇒ landmark rác. Nếu chỉ so IoU thì 37/41 khung chọn landmark rác.
            const float det_w = det[2] - det[0], det_h = det[3] - det[1];
            const float lm_w = lm[2] - lm[0], lm_h = lm[3] - lm[1];
            bool lm_plausible = false;
            if (det_w > 16.0f && det_h > 16.0f && lm_w > 16.0f && lm_h > 16.0f) {
                const float rw = lm_w / det_w, rh = lm_h / det_h;
                const float cx_off = fabsf((lm[0] + lm[2]) - (det[0] + det[2])) * 0.5f / det_w;
                const float cy_off = fabsf((lm[1] + lm[3]) - (det[1] + det[3])) * 0.5f / det_h;
                lm_plausible = rw >= 0.60f && rw <= 1.40f && rh >= 0.60f && rh <= 1.40f &&
                               cx_off <= 0.35f && cy_off <= 0.35f;
            }
            if (lm_plausible && iou_dl < FUSION_MIN_IOU) {
                // Landmark geometry disagrees with the detector: trust the landmarks
                // (they follow the face every inference) and keep the detector score.
                memcpy(box, lm, sizeof(float) * 4);
                used_lm = true;
            }
            ESP_LOGI(TAG, "fusion iou=%.3f use=%s lm_ok=%d det=%.1f,%.1f,%.1f,%.1f lm=%.1f,%.1f,%.1f,%.1f",
                     (double)iou_dl, used_lm ? "landmark" : "detector", (int)lm_plausible,
                     (double)det[0], (double)det[1], (double)det[2], (double)det[3],
                     (double)lm[0], (double)lm[1], (double)lm[2], (double)lm[3]);
        }
        tracked = blaze_track(box, score, dt, tbox);
    } else if (s_ready) {
        // Detector missed: keep the last good box for a few frames.
        tracked = blaze_track(box, 0.0f, dt, tbox);
    }

    s_last_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;

    if (tracked) {
        out->cx = (tbox[0] + tbox[2]) * 0.5f;
        out->cy = (tbox[1] + tbox[3]) * 0.5f;
        out->size = fmaxf(tbox[2] - tbox[0], tbox[3] - tbox[1]);
        out->score = score;
        out->valid = true;
        out->has_kp = false;
        ESP_LOGI(TAG, "blaze OK cand=%d score=%.3f box=%.1f,%.1f,%.1f,%.1f det=%.0fms",
                 blaze_last_cand(), (double)score, (double)tbox[0], (double)tbox[1],
                 (double)tbox[2], (double)tbox[3], (double)blaze_last_ms());
        return true;
    }

    // [13/11] Detector missed and no tracked box: use the landmark bbox ONLY if we have
    // a face history and the miss streak is short. Otherwise the landmark bbox is derived
    // from PIPNet running on the fallback centre ROI (garbage) and would drift the ROI.
    static bool s_had_face = false;
    static int s_miss_streak = 0;
    if (ok) { s_had_face = true; s_miss_streak = 0; } else { s_miss_streak++; }
    if (!ok && s_miss_streak > 12) {          // ≈100 s @0.12 FPS without any detection
        if (s_had_face) ESP_LOGW(TAG, "no face for %d frames -> reset tracker", s_miss_streak);
        s_had_face = false;
        blaze_reset();
    }
    {
        float lm[4] = {0, 0, 0, 0};
        if (s_had_face && blaze_fusion_get(nullptr, lm, nullptr, nullptr) &&
            lm[2] > lm[0] + 16.0f && lm[3] > lm[1] + 16.0f) {
            out->cx = (lm[0] + lm[2]) * 0.5f;
            out->cy = (lm[1] + lm[3]) * 0.5f;
            out->size = fmaxf(lm[2] - lm[0], lm[3] - lm[1]);
            out->score = 0.0f;
            out->valid = true;
            out->has_kp = false;
            ESP_LOGI(TAG, "blaze MISS cand=%d -> landmark ROI (%.0f,%.0f,%.0f)",
                     blaze_last_cand(), (double)out->cx, (double)out->cy, (double)out->size);
            return true;
        }
    }

    // Fallback: centre ROI. The landmark tracker refines it every frame.
    float side = 0.75f * (float)(src_w < src_h ? src_w : src_h);
    out->cx = src_w * 0.5f;
    out->cy = src_h * 0.5f;
    out->size = side;
    out->score = 0.0f;
    out->valid = true;
    out->has_kp = false;
    ESP_LOGI(TAG, "blaze MISS cand=%d -> centre ROI (%.0f,%.0f,%.0f)",
             blaze_last_cand(), (double)out->cx, (double)out->cy, (double)side);
    return true;
}

float face_detector_lite_last_ms(void) { return s_last_ms; }
