#include "roi_tracker.h"
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"

static const char* TAG = "ROI_TRACKER";

// ============================================================================
// [D7] PORT 1:1 FaceTracker của laptop (host_laptop/local_model_tester.py):
//   - Landmark-Driven Tracking: tâm/cạnh suy ra từ 22 landmark frame TRƯỚC
//     bằng công thức canonical anatomical anchor lúc TRAIN.
//   - Lọc One-Euro (per-axis): đứng yên -> triệt rung; xoay nhanh -> zero-lag.
//   - Deadband: dịch < 1.5px / scale < 2% -> giữ nguyên (chống micro-jitter).
//   - Detector (lite/BlazeFace) đóng vai trò mỏ neo chống trôi (như Haar/MediaPipe).
// ============================================================================

#define ROI_HYST_POS     1.5f    // px  (laptop: hysteresis_pos)
#define ROI_HYST_SCALE   0.020f  // 2%  (laptop: hysteresis_scale)
#define ROI_MIN_S_RATIO  0.33f   // laptop: min 160px tren khung 480 -> 33%
#define ROI_BOOTSTRAP_FRAMES 2
#define ROI_MISS_RESET       8   // mất mặt liên tục -> reset để bám lại

// ------------------------- One-Euro Filter -------------------------
typedef struct {
    float min_cutoff, beta, d_cutoff;
    float x_prev, dx_prev, t_prev;
} one_euro_t;

static float oe_alpha(float cutoff, float dt) {
    float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
    return 1.0f / (1.0f + tau / dt);
}

static float oe_filter(one_euro_t* f, float x, float t) {
    if (f->t_prev <= 0.0f) {
        f->t_prev = t; f->x_prev = x; f->dx_prev = 0.0f;
        return x;
    }
    float dt = t - f->t_prev;
    if (dt <= 1e-5f) return f->x_prev;
    float dx = (x - f->x_prev) / dt;
    float ad = oe_alpha(f->d_cutoff, dt);
    float dx_hat = ad * dx + (1.0f - ad) * f->dx_prev;
    float cutoff = f->min_cutoff + f->beta * fabsf(dx_hat);
    float a = oe_alpha(cutoff, dt);
    float x_hat = a * x + (1.0f - a) * f->x_prev;
    f->x_prev = x_hat; f->dx_prev = dx_hat; f->t_prev = t;
    return x_hat;
}

// ------------------------- Trạng thái -------------------------
static bool  s_valid = false;
static float s_cx = 0.0f, s_cy = 0.0f, s_size = 0.0f;
static int   s_src_w = 0, s_src_h = 0;
static int   s_hits = 0;
static int   s_miss = 0;
static bool  s_smooth_ready = false;

static one_euro_t s_fx = { 0.70f, 0.080f, 1.0f, 0, 0, 0 };
static one_euro_t s_fy = { 0.70f, 0.080f, 1.0f, 0, 0, 0 };
static one_euro_t s_fS = { 0.50f, 0.040f, 1.0f, 0, 0, 0 };

static float now_s(void) { return (float)esp_timer_get_time() / 1e6f; }

void roi_tracker_reset(void) {
    s_valid = false;
    s_cx = s_cy = s_size = 0.0f;
    s_src_w = s_src_h = 0;
    s_hits = 0;
    s_miss = 0;
    s_smooth_ready = false;
    s_fx.t_prev = s_fy.t_prev = s_fS.t_prev = 0.0f;
}

bool roi_tracker_is_valid(void) { return s_valid; }

static void clamp_state(int src_w, int src_h) {
    int minDim = (src_w < src_h) ? src_w : src_h;
    float minS = ROI_MIN_S_RATIO * (float)minDim;
    if (s_size < minS) s_size = minS;
    if (s_size > (float)minDim) s_size = (float)minDim;
    float half = s_size * 0.5f;
    if (s_cx < half) s_cx = half;
    if (s_cx > (float)src_w - half) s_cx = (float)src_w - half;
    if (s_cy < half) s_cy = half;
    if (s_cy > (float)src_h - half) s_cy = (float)src_h - half;
    if (s_cx < 0) s_cx = 0;
    if (s_cy < 0) s_cy = 0;
}

face_roi_t roi_tracker_get(void) {
    face_roi_t r = { 0, 0, 0 };
    if (s_src_w <= 0 || s_src_h <= 0 || !s_valid) return r;
    clamp_state(s_src_w, s_src_h);
    r.x0 = (int)lroundf(s_cx - s_size * 0.5f);
    r.y0 = (int)lroundf(s_cy - s_size * 0.5f);
    r.size = (int)lroundf(s_size);
    if (r.size < 16) r.size = 16;
    return r;
}

// Áp mục tiêu (cx,cy,S) qua deadband + One-Euro rồi lưu trạng thái
static void rt_apply_target(float cx, float cy, float S, int src_w, int src_h, bool hard) {
    s_src_w = src_w; s_src_h = src_h;
    if (hard || !s_smooth_ready) {
        s_cx = cx; s_cy = cy; s_size = S;
        s_smooth_ready = true;
        s_fx.t_prev = s_fy.t_prev = s_fS.t_prev = 0.0f;   // reset bộ lọc
        oe_filter(&s_fx, cx, now_s());
        oe_filter(&s_fy, cy, now_s());
        oe_filter(&s_fS, S,  now_s());
        clamp_state(src_w, src_h);
        return;
    }

    float t = now_s();
    // Deadband: chống micro-jitter (giống laptop)
    float move = hypotf(cx - s_cx, cy - s_cy);
    if (move < ROI_HYST_POS) { cx = s_cx; cy = s_cy; }
    if (fabsf(S - s_size) / (s_size > 1.0f ? s_size : 1.0f) < ROI_HYST_SCALE) S = s_size;

    s_cx   = oe_filter(&s_fx, cx, t);
    s_cy   = oe_filter(&s_fy, cy, t);
    s_size = oe_filter(&s_fS, S,  t);
    clamp_state(src_w, src_h);
}

// Công thức canonical anchor (đồng bộ isomorphic_transform.compute_canonical_anchor)
static void compute_canonical_anchor(const float px[22], const float py[22],
                                     float* out_cx, float* out_cy, float* out_S) {
    float eyeLx = 0, eyeLy = 0, eyeRx = 0, eyeRy = 0;
    for (int i = 0; i < 6; i++) { eyeLx += px[i]; eyeLy += py[i]; }
    for (int i = 6; i < 12; i++) { eyeRx += px[i]; eyeRy += py[i]; }
    eyeLx /= 6.0f; eyeLy /= 6.0f; eyeRx /= 6.0f; eyeRy /= 6.0f;

    float eye_x = (eyeLx + eyeRx) * 0.5f;
    float eye_y = (eyeLy + eyeRy) * 0.5f;
    float nose_x = px[19];
    float nose_y = py[19];
    float chin_y = py[21];

    float d_eyes = hypotf(eyeRx - eyeLx, eyeRy - eyeLy);
    float d_eye_nose = fmaxf(fmaxf(nose_y - eye_y, 0.45f * d_eyes), 1.0f);
    float d_eye_chin = fmaxf(fmaxf(chin_y - eye_y, d_eye_nose), 1.0f);
    float h_skull = fmaxf(fmaxf(d_eye_nose * 2.10f, d_eyes * 1.40f), d_eye_chin / 1.20f);

    *out_S = h_skull * 2.05f;
    *out_cx = (eye_x + nose_x) * 0.5f;
    *out_cy = eye_y + 0.32f * h_skull;
}

void roi_tracker_update_from_landmarks(const point2d_t landmarks[22],
                                       int src_w, int src_h,
                                       int crop_x0, int crop_y0, int crop_size) {
    if (!landmarks || src_w <= 0 || src_h <= 0 || crop_size <= 0) return;

    float px[22], py[22];
    for (int i = 0; i < 22; i++) {
        px[i] = (float)crop_x0 + landmarks[i].x * (float)crop_size;
        py[i] = (float)crop_y0 + landmarks[i].y * (float)crop_size;
    }

    float eyeLx = 0, eyeLy = 0, eyeRx = 0, eyeRy = 0;
    for (int i = 0; i < 6; i++) { eyeLx += px[i]; eyeLy += py[i]; }
    for (int i = 6; i < 12; i++) { eyeRx += px[i]; eyeRy += py[i]; }
    eyeLx /= 6.0f; eyeLy /= 6.0f; eyeRx /= 6.0f; eyeRy /= 6.0f;
    float d_eyes = hypotf(eyeRx - eyeLx, eyeRy - eyeLy);
    float eye_y = (eyeLy + eyeRy) * 0.5f;
    float chin_y = py[21];

    // Kiểm tra hợp lệ tương đối theo kích thước mặt kỳ vọng (không theo crop,
    // vì crop có thể đang là crop giữa khung khi chưa bám được).
    if (!(d_eyes > 1.0f) || !(chin_y > eye_y)) {
        if (s_valid) {
            s_miss++;
            if (s_miss >= ROI_MISS_RESET) {
                ESP_LOGW(TAG, "Mất mặt %d frame -> reset ROI", s_miss);
                roi_tracker_reset();
            }
        }
        return;
    }
    s_miss = 0;

    float cx, cy, S;
    compute_canonical_anchor(px, py, &cx, &cy, &S);

    // [D8] CHỐNG TRÔI SCALE (quan trọng): vòng lặp landmark là feedback dương —
    // nếu để nó tự dẫn, sai số nhỏ của model (nhất là cằm P21) làm crop nhỏ dần
    // mỗi frame -> sụp về sàn (giống laptop nếu thiếu detector neo). Kẹp S trong
    // ±20% quanh giá trị hiện tại; chỉ detector/anchor mới được đổi scale lớn.
    if (s_valid && s_size > 1.0f) {
        float dS = (S - s_size) / s_size;
        if (dS > 0.20f) S = s_size * 1.20f;
        else if (dS < -0.20f) S = s_size * 0.80f;
    }

    if (!s_valid) {
        s_hits++;
        rt_apply_target(cx, cy, S, src_w, src_h, !s_smooth_ready);
        if (s_hits >= ROI_BOOTSTRAP_FRAMES) {
            s_valid = true;
            ESP_LOGI(TAG, "✅ ROI bám mặt: tâm=(%.0f,%.0f) cạnh=%.0f (khung %dx%d)",
                     cx, cy, S, src_w, src_h);
        }
        return;
    }
    rt_apply_target(cx, cy, S, src_w, src_h, false);
}

void roi_tracker_anchor(int cx, int cy, int size, int src_w, int src_h) {
    if (size <= 0 || src_w <= 0 || src_h <= 0) return;
    float S = (float)size;
    int minDim = (src_w < src_h) ? src_w : src_h;
    if (S > (float)minDim) S = (float)minDim;
    // Chưa bám -> đặt cứng; đang bám -> hiệu chỉnh nhẹ qua bộ lọc (chống trôi)
    rt_apply_target((float)cx, (float)cy, S, src_w, src_h, !s_valid);
    s_valid = true;
    s_smooth_ready = true;
}
