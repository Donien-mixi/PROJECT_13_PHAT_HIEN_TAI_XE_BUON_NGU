#include "adas_controller.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_log.h"
#else
#include <chrono>
// Host build (tools/adas_replay_test.cpp): the clock is driven by the test so the
// same code can be replayed at 4,3 FPS and at 0,12 FPS without waiting in real time.
static int64_t g_adas_host_now_us = 0;
void adas_host_set_now_us(int64_t t_us) { g_adas_host_now_us = t_us; }
static int64_t esp_timer_get_time() { return g_adas_host_now_us; }
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) printf("[%s][W] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) fprintf(stderr, "[ERROR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

static const char* TAG = "ADAS_CONTROLLER";

// Timing thresholds (microseconds)
// [13/11] Calibration: frame-count driven (xem khối calibration) — 5 s wall-clock chỉ chứa
// 1 khung ở 0,12 FPS nên không bao giờ đủ mẫu để tính bias pose / ngưỡng EAR.
#define CALIB_MIN_SAMPLES         4
#define CALIB_MAX_SAMPLES         10
#define CALIB_MIN_TIME_US         (4 * 1000 * 1000)
#define SLOW_BLINK_DURATION_US    (500 * 1000)        // 0.5s slow blink
#define EYE_MAX_FRAME_AGE_US       (600 * 1000)
#define EYE_MAX_SAMPLE_GAP_US      (500 * 1000)
#define MICROSLEEP_DURATION_US    (1500 * 1000)       // 1.5s microsleep
// [13/11] Cổng tin cậy mí mắt: khi đầu quay/cúi quá mức, điểm mí của PIPNet bị kéo
// (đo được EAR 0,143 khi quay trái so với 0,20 lúc nhìn thẳng) -> coi mẫu là không hợp lệ
// thay vì tính thành "nhắm mắt".
#define EYE_MAX_YAW_DEV_DEG       25.0f
#define EYE_MAX_PITCH_DEV_DEG     30.0f
#define YAWN_EVENT_DURATION_US    (1200 * 1000)       // [D14] 1.2s (bù việc lọc MAR 4-tap ăn ~0.35s mỗi đầu, yawn thật 1.5-3s vẫn dư biên)
#define DISTRACTION_DURATION_US   (3000 * 1000)       // 3.0s head turned away
#define YAWN_WINDOW_US            (180 * 1000 * 1000) // 3 minutes rolling window

#define MAX_YAWN_HISTORY 10

// Internal Controller State
static int64_t s_start_time_us = 0;
static bool s_is_calibrated = false;

// Calibration Accumulators — [v2.3.0] dùng MEDIAN để khớp 100% với laptop
// (mean bị lệch baseline nếu có 1 cú chớp mắt trong 5s hiệu chuẩn).
#define MAX_CALIB_SAMPLES 256
static float s_calib_ear_buf[MAX_CALIB_SAMPLES];
static float s_calib_mar_buf[MAX_CALIB_SAMPLES];
static float s_calib_pitch_buf[MAX_CALIB_SAMPLES];
static float s_calib_yaw_buf[MAX_CALIB_SAMPLES];   // [D9] baseline yaw (POSIT lệch hệ thống ~+15°)
static float s_pitch_bias = 0.0f;   // [v2.6.6] baseline pitch (bias POSIT) trừ đi khi báo mất tập trung
static float s_yaw_bias = 0.0f;     // [D9] baseline yaw — tránh BÁO ĐỘNG DISTRACTION GIẢ khi nhìn thẳng
static uint32_t s_calib_count = 0;

// Median (sắp xếp nổi bọt/chèn tại chỗ, n <= 256)
// [11/11] Robust baseline: percentile of the calibration EAR samples, so blinks or
// half-closed eyes during calibration do not drag the open-eye baseline down.
static float compute_percentile(float* buf, uint32_t n, float p) {
    if (n == 0 || !buf) return 0.0f;
    float tmp[MAX_CALIB_SAMPLES];
    uint32_t m = n < MAX_CALIB_SAMPLES ? n : MAX_CALIB_SAMPLES;
    for (uint32_t i = 0; i < m; i++) tmp[i] = buf[i];
    for (uint32_t i = 1; i < m; i++) {
        float v = tmp[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = v;
    }
    uint32_t idx = (uint32_t)(p * (float)(m - 1) + 0.5f);
    if (idx >= m) idx = m - 1;
    return tmp[idx];
}

static float compute_median(float* buf, uint32_t n) {
    if (n == 0) return 0.0f;
    for (uint32_t i = 1; i < n; i++) {
        float key = buf[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
        buf[j + 1] = key;
    }
    if (n & 1u) return buf[n / 2];
    return 0.5f * (buf[n / 2 - 1] + buf[n / 2]);
}

// Adaptive Dynamic Thresholds
static float s_ear_threshold = 0.21f;
static float s_ear_baseline = 0.0f;   // [14/11] p75 EAR luc hieu chuan (mo mat)
// [11/11] Student eye-state: P(closed) tu mo hinh 48x48, -1 khi khong co.
static float s_eye_prob = -1.0f;
static const float EYE_PROB_CLOSED_THRESH = 0.50f;
static float s_mar_threshold = 0.45f;
static const float s_yaw_threshold_deg = 30.0f;
static const float s_pitch_threshold_deg = 25.0f;

// Duration Tracking Timers
static int64_t s_eye_closed_start_us = 0;
static int64_t s_last_eye_capture_us = 0;
static bool s_eye_sample_lost = false;
// [14/11] "Kiem tra dinh ky": lich su trang thai mat de mot mau bi lo (detector miss)
// khong xoa mat ca dot nham mat. 1 = nham, 0 = mo, -1 = khong biet.
#define EYE_HIST_LEN 12
static int8_t s_eye_hist[EYE_HIST_LEN] = {0};
static int s_eye_hist_pos = 0, s_eye_hist_n = 0;
static int64_t s_mouth_open_start_us = 0;
static int64_t s_distraction_start_us = 0;

// [13/11] FPS-adaptive eye timing.
// Ở 4,3 FPS (v3/Lite) một cử chỉ 1,5 s ≈ 7 khung; ở 0,12 FPS (PIPNet) chỉ có 1 khung.
// Vì vậy "nhắm mắt" được đếm bằng SỐ MẪU liên tiếp, với ngưỡng suy ra từ chu kỳ khung
// đo được, để cùng một logic chạy đúng ở cả hai tốc độ.
static int64_t s_period_us = 0;          // EMA chu kỳ khung (capture_us)
static int64_t s_last_any_capture_us = 0; // mọi mẫu, kể cả không hợp lệ (đo chu kỳ)
static int s_eye_closed_samples = 0;     // số mẫu "nhắm" liên tiếp
static int s_k_slow = 1, s_k_micro = 2;  // ngưỡng mẫu cho slow-blink / microsleep
// [13/11] PIPNet trả landmark rác khi không có mặt trong khung (ROI dự phòng) -> EAR 0,03
// sẽ bị tính là "nhắm". Pipeline bật/tắt đường mắt theo kết quả detector.
static bool s_eye_enabled = true;

// Yawn Tracking History
static int64_t s_yawn_timestamps[MAX_YAWN_HISTORY] = {0};
static int s_yawn_history_count = 0;
static bool s_is_currently_yawning = false;

// Blink Counter
static uint32_t s_total_blinks = 0;
static uint32_t s_total_yawns = 0;
static bool s_was_eye_closed = false;

static inline float euclidean_dist(point2d_t a, point2d_t b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

// [D23] Bias pose (đo lúc hiệu chuẩn 5s) để hiển thị pose đã trừ bias
float adas_controller_get_pitch_bias(void) { return s_pitch_bias; }
float adas_controller_get_yaw_bias(void)   { return s_yaw_bias; }
float adas_controller_get_ear_baseline(void) { return s_ear_baseline; }

static void reset_eye_timer(void) {
    s_eye_closed_start_us = 0;
    s_eye_closed_samples = 0;
    s_was_eye_closed = false;
    // lich su KHONG bi xoa: chi cuon theo thoi gian de chiu duoc 1 mau bi lo
}

void adas_controller_invalidate_eye_sample(void) {
    reset_eye_timer();
    s_eye_sample_lost = true;
}

void adas_controller_set_eye_prob(float p_closed) { s_eye_prob = p_closed; }

void adas_controller_set_eye_enabled(bool enabled) { s_eye_enabled = enabled; }

void adas_controller_init(void) {
    s_eye_prob = -1.0f;
    s_start_time_us = esp_timer_get_time();
    s_is_calibrated = false;
    s_calib_count = 0;
    s_period_us = 0;                 // [13/11] đo lại chu kỳ khung sau khi khởi tạo lại
    s_last_any_capture_us = 0;
    s_eye_enabled = true;
    s_eye_closed_samples = 0;

    s_eye_closed_start_us = 0;
    s_last_eye_capture_us = 0;
    s_eye_sample_lost = false;
    s_mouth_open_start_us = 0;
    s_distraction_start_us = 0;

    s_total_blinks = 0;
    s_total_yawns = 0;
    s_yawn_history_count = 0;
    s_was_eye_closed = false;
    s_is_currently_yawning = false;

    ESP_LOGI(TAG, "Đã khởi tạo ADAS Controller State Machine. Bắt đầu giai đoạn tự hiệu chuẩn (5 giây)...");
}

void adas_controller_update(const point2d_t landmarks[22],
                            const head_pose_t* pose,
                            float fps,
                            adas_metrics_t* out_metrics) {
    adas_controller_update_capture(landmarks, pose, fps, esp_timer_get_time(), out_metrics);
}

void adas_controller_update_capture(const point2d_t landmarks[22],
                                    const head_pose_t* pose,
                                    float fps,
                                    int64_t capture_us,
                                    adas_metrics_t* out_metrics) {
    int64_t now = esp_timer_get_time();

    out_metrics->eye = {capture_us > 0 && capture_us <= now
        ? (float)(now - capture_us) / 1000.0f : -1.0f,
        0.0f, false, false, "calibrating"};

    // 1. Calculate Eye Aspect Ratio (EAR)
    // Left Eye: 0 (outer), 1, 2, 3 (inner), 4, 5
    float left_v1 = euclidean_dist(landmarks[1], landmarks[5]);
    float left_v2 = euclidean_dist(landmarks[2], landmarks[4]);
    float left_h  = euclidean_dist(landmarks[0], landmarks[3]);
    float ear_l   = (left_h > 1e-4f) ? (left_v1 + left_v2) / (2.0f * left_h) : 0.0f;

    // Right Eye: 6 (outer), 7, 8, 9 (inner), 10, 11
    float right_v1 = euclidean_dist(landmarks[7], landmarks[11]);
    float right_v2 = euclidean_dist(landmarks[8], landmarks[10]);
    float right_h  = euclidean_dist(landmarks[6], landmarks[9]);
    float ear_r    = (right_h > 1e-4f) ? (right_v1 + right_v2) / (2.0f * right_h) : 0.0f;

    float ear = (ear_l + ear_r) * 0.5f;

    // 2. Calculate Mouth Aspect Ratio (MAR)
    // [SYNC 2025] Công thức đồng bộ 100% với training_tinyml/wing_loss.py (compute_tensor_mar)
    // và host_laptop/local_model_tester.py (compute_mar):
    //     MAR = (h_outer + h_inner) / (2 * w_mouth)
    // Trước đây firmware dùng v_outer/h thuần -> MAR firmware khác MAR laptop/train
    // -> ngưỡng hiệu chuẩn lệch -> hành vi báo ngáp khác nhau giữa 2 nền tảng.
    float mouth_h_outer = euclidean_dist(landmarks[14], landmarks[15]);
    float mouth_h_inner = euclidean_dist(landmarks[16], landmarks[17]);
    float mouth_w       = euclidean_dist(landmarks[12], landmarks[13]);
    float mar_raw = (mouth_w > 1e-4f) ? ((mouth_h_outer + mouth_h_inner) / (2.0f * mouth_w)) : 0.0f;

    // [D10] LÀM MƯỢT MAR (trung bình trượt 4 frame).
    // ĐO THỰC TẾ: điểm miệng của model nhấp nháy (0.99 -> 0.22 trong <1s) vì miệng
    // là vùng model yếu nhất (lệch MediaPipe 5.68px) -> điều kiện "MAR>ngưỡng LIÊN TỤC
    // 1.5s" không bao giờ đạt -> KHÔNG BAO GIỜ BÁO NGÁP. Lọc trượt khử nhấp nháy.
    // [D22] KHÔNG lọc trượt MAR nữa.
    // ĐO THỰC TẾ (so log song song): CÙNG model TinyDriverNet, cùng người:
    //   - Trên laptop: MAR ngáp đạt 0.84 – 1.05
    //   - Trên ESP32  : chỉ 0.60  => bộ lọc trung bình trượt 4-tap (D10) ĂN MẤT ĐỈNH NGÁP.
    // => Dùng MAR THÔ (khớp đúng laptop), chống nhấp nháy bằng "grace 2 frame" ở
    //    state machine phía dưới (chỉ reset bộ đếm khi miệng đóng LIÊN TỤC 2 frame).
    float mar = mar_raw;

    // 3. Calibration Phase
    // [13/11] Calibration is now FRAME-COUNT driven, not wall-clock driven.
    // ĐO THỰC TẾ: ở chế độ PIPNet-98 (0,12 FPS = 8,2 s/khung) cửa sổ 5 s chỉ chứa 1 khung
    // -> s_calib_count = 1 -> khối "count > 10" không bao giờ chạy -> không có bias pose,
    // không có ngưỡng EAR => yaw thô +40..+65° khi nhìn thẳng => DISTRACTION ALARM vĩnh viễn
    // (bằng chứng: serial_live5.log 400 s, 8/9 khung báo DISTRACTION, 0 lần "Hiệu chuẩn thành công").
    if (!s_is_calibrated) {
        const int64_t cal_elapsed = now - s_start_time_us;
        // [13/11] Chỉ lấy mẫu hiệu chuẩn khi có mặt + EAR hợp lý. Trước đây gom mù nên
        // khi vắng người baseline EAR = 0,09 (rác) -> ngưỡng báo sai.
        const bool cal_plausible = s_eye_enabled && isfinite(ear) && ear > 0.05f && ear < 0.60f &&
                                   left_h > 1e-4f && right_h > 1e-4f;
        const bool cal_ready = (s_calib_count >= CALIB_MIN_SAMPLES) &&
                               (cal_elapsed >= CALIB_MIN_TIME_US || s_calib_count >= CALIB_MAX_SAMPLES);
        if (!cal_ready) {
            if (cal_plausible && s_calib_count < MAX_CALIB_SAMPLES) {
                s_calib_ear_buf[s_calib_count] = ear;
                s_calib_mar_buf[s_calib_count] = mar;
                s_calib_pitch_buf[s_calib_count] = pose->pitch;
                s_calib_yaw_buf[s_calib_count] = pose->yaw;
                s_calib_count++;
            }

            out_metrics->ear_left = ear_l;
            out_metrics->ear_right = ear_r;
            out_metrics->ear = ear;
            out_metrics->mar = mar;
            out_metrics->pose = *pose;
            out_metrics->state = ADAS_STATE_CALIBRATING;
            out_metrics->is_alarm_active = false;
            out_metrics->total_blinks = 0;
            out_metrics->total_yawns = 0;
            out_metrics->esp32_fps = fps;
            out_metrics->ear_threshold = s_ear_threshold;
            out_metrics->mar_threshold = s_mar_threshold;
            out_metrics->mouth_open_s = 0.0f;
            snprintf(out_metrics->status_str, sizeof(out_metrics->status_str), "CALIBRATING... %us/%u",
                     (unsigned)s_calib_count, (unsigned)CALIB_MIN_SAMPLES);
            // Không log mỗi khung (8,2 s/khung -> spam); chỉ log khi có mẫu mới hoặc mỗi ~60 s.
            static uint32_t s_last_logged_count = 99;
            static int64_t s_last_log_us = 0;
            if (s_calib_count != s_last_logged_count || now - s_last_log_us > 60 * 1000 * 1000) {
                s_last_logged_count = s_calib_count;
                s_last_log_us = now;
                ESP_LOGI(TAG, "calib sample %u/%u (elapsed %lld ms, %s)",
                         (unsigned)s_calib_count, (unsigned)CALIB_MIN_SAMPLES,
                         (long long)(cal_elapsed / 1000),
                         cal_plausible ? "face ok" : "waiting for a face");
            }
            return;
        } else {
            // Calibration Complete
            if (s_calib_count >= CALIB_MIN_SAMPLES) {
                float avg_ear = compute_percentile(s_calib_ear_buf, s_calib_count, 0.75f);
                float avg_mar = compute_median(s_calib_mar_buf, s_calib_count);
                // [D12] Baseline MAR quá cao = lúc hiệu chuẩn miệng đang mở/nói chuyện
                // -> ngưỡng bị đẩy lên TRẦN 0.65 -> KHÔNG BAO GIỜ BÁO NGÁP (đo thực tế:
                // baseline 0.55, MAR ngáp đỉnh 0.71 -> không đạt ngưỡng liên tục 1.5s).
                // Người nghỉ có MAR ≈ 0.20-0.35 (cả MediaPipe lẫn model đều vậy).
                if (avg_mar > 0.35f) {
                    ESP_LOGW(TAG, "⚠️ Baseline MAR=%.2f qua cao (mieng mo khi calib) -> kep ve 0.30", avg_mar);
                    avg_mar = 0.30f;
                }
                s_pitch_bias = compute_median(s_calib_pitch_buf, s_calib_count);
                s_yaw_bias   = compute_median(s_calib_yaw_buf, s_calib_count);
                // [14/11] 0,72 x baseline thay vi 0,70: tren 100 nguoi CEW, nguong 0,72*p75 tach
                // mo/nham tot hon (open p10 0,145 > 0,1375 > closed trung vi 0,116), va san 0,085
                // de nguoi mi mong (baseline ~0,13) khong bi san cu 0,15/0,10 chan mat do nhay.
                s_ear_baseline = avg_ear;
                // [14/11] 0,75 thay vi 0,72: do nghiem thu 20:29 thay mau NHAM = 0,107 va 0,130
                // trong khi nguong 0,72x0,181 = 0,130 (dung bien, bi coi la MO); mau MO thap nhat
                // do duoc la 0,147 -> nguong 0,75x0,181 = 0,136 tach duoc ca hai.
                s_ear_threshold = avg_ear * 0.75f;
                s_mar_threshold = avg_mar * 1.60f;
                if (s_ear_threshold < 0.085f) s_ear_threshold = 0.085f;
                if (s_ear_threshold > 0.25f) s_ear_threshold = 0.25f;
                if (s_mar_threshold < 0.40f) s_mar_threshold = 0.40f;
                if (s_mar_threshold > 0.58f) s_mar_threshold = 0.58f;   // [D12] trần hạ 0.65->0.58: đo thực tế MAR ngáp đỉnh 0.71-0.94
                ESP_LOGI(TAG, "✅ Hiệu chuẩn thành công! Baseline: EAR=%.2f, MAR=%.2f | Bias pose: pitch=%+.1f° yaw=%+.1f° | Ngưỡng báo: EAR<%.2f, MAR>%.2f",
                         avg_ear, avg_mar, s_pitch_bias, s_yaw_bias, s_ear_threshold, s_mar_threshold);
            }
            s_is_calibrated = true;
        }
    }

    // 4. Eye State & Blink / Microsleep Detection
    int64_t eye_closed_duration = 0;
    const char* invalid_reason = nullptr;
    // [13/11] Chu kỳ khung + ngưỡng mẫu suy ra từ chu kỳ (FPS-adaptive).
    // Chu kỳ phải đo từ MỌI mẫu (kể cả mẫu bị loại), nếu không sẽ tự khoá: mẫu bị
    // "stale" -> s_last_eye_capture_us không cập nhật -> period giữ 230 ms -> mãi stale.
    if (s_last_any_capture_us > 0 && capture_us > s_last_any_capture_us) {
        int64_t d = capture_us - s_last_any_capture_us;
        if (d > 50000 && d < 60000000) {
            s_period_us = s_period_us == 0 ? d : (s_period_us * 7 + d * 3) / 10;
        }
    }
    if (capture_us > 0) s_last_any_capture_us = capture_us;
    const int64_t period = s_period_us > 0 ? s_period_us : 230000;
    const int64_t age_limit_us = (period * 3 / 2 > EYE_MAX_FRAME_AGE_US) ? period * 3 / 2 : EYE_MAX_FRAME_AGE_US;
    const int64_t gap_limit_us = (period * 3 / 2 > EYE_MAX_SAMPLE_GAP_US) ? period * 3 / 2 : EYE_MAX_SAMPLE_GAP_US;
    int k_slow = (int)((SLOW_BLINK_DURATION_US + period - 1) / period);
    int k_micro = (int)((MICROSLEEP_DURATION_US + period - 1) / period);
    // [14/11] Chế độ "kiểm tra định kỳ" (0,12 FPS): một mẫu đơn lẻ KHÔNG đủ để kết luận
    // (không phân biệt được chớp mắt với nhắm mắt) -> mọi cảnh báo mắt cần >= 2 mẫu liên tiếp (~16 s).
    if (k_slow < 2) k_slow = 2;
    if (k_micro < k_slow) k_micro = k_slow;
    if (k_micro < 2) k_micro = 2;      // ít nhất 2 mẫu, kể cả khi khung rất thưa
    if (k_micro > 12) k_micro = 12;
    s_k_slow = k_slow;
    s_k_micro = k_micro;

    if (capture_us <= 0) invalid_reason = "missing_timestamp";
    else if (capture_us > now) invalid_reason = "future_timestamp";
    else if (capture_us <= s_last_eye_capture_us) invalid_reason = "non_increasing_timestamp";
    else if (!s_eye_enabled) invalid_reason = "no_face";
    else if (now - capture_us > age_limit_us) invalid_reason = "stale";
    else if (!(left_h > 1e-4f && right_h > 1e-4f && isfinite(ear))) invalid_reason = "eye_geometry";
    else if (pose->is_valid &&
             (fabsf(pose->yaw - s_yaw_bias) > EYE_MAX_YAW_DEV_DEG ||
              fabsf(pose->pitch - s_pitch_bias) > EYE_MAX_PITCH_DEV_DEG))
        invalid_reason = "head_turned";   // mí mắt không còn tin cậy khi đầu quay/cúi mạnh

    const bool eye_sample_valid = invalid_reason == nullptr;
    const bool gap = eye_sample_valid && s_last_eye_capture_us > 0 &&
        capture_us - s_last_eye_capture_us > gap_limit_us;
    if (!eye_sample_valid || gap) reset_eye_timer();
    out_metrics->eye.valid = eye_sample_valid;
    // [14/11] Luat tuong doi: nham khi EAR tut >= 25 % so voi baseline cua chinh tai xe
    // (chi ap dung khi baseline du tin cay >= 0,12).
    const bool ear_drop_25 = (s_ear_baseline >= 0.12f) && (ear < 0.75f * s_ear_baseline);
    out_metrics->eye.closed = eye_sample_valid &&
        (ear < s_ear_threshold || ear_drop_25 ||
         (s_eye_prob >= 0.0f && s_eye_prob >= EYE_PROB_CLOSED_THRESH));
    out_metrics->eye.reset_reason = invalid_reason ? invalid_reason :
        s_eye_sample_lost ? "processing_loss" : gap ? "capture_gap" : "none";
    s_eye_sample_lost = false;
    // day trang thai vao lich su (ca mau khong hop le: -1)
    {
        int8_t v = -1;
        if (eye_sample_valid) v = out_metrics->eye.closed ? 1 : 0;
        s_eye_hist[s_eye_hist_pos] = v;
        s_eye_hist_pos = (s_eye_hist_pos + 1) % EYE_HIST_LEN;
        if (s_eye_hist_n < EYE_HIST_LEN) s_eye_hist_n++;
    }

    if (eye_sample_valid) {
        s_last_eye_capture_us = capture_us;
        if (out_metrics->eye.closed) {
            if (s_eye_closed_start_us == 0) {
                s_eye_closed_start_us = capture_us;
            }
            s_eye_closed_samples++;                     // [13/11] đếm mẫu, không chỉ thời gian
            eye_closed_duration = capture_us - s_eye_closed_start_us;
            s_was_eye_closed = true;
        } else {
            if (s_was_eye_closed && s_eye_closed_start_us > 0) {
                int64_t blink_dur = capture_us - s_eye_closed_start_us;
                if (blink_dur > (100 * 1000) && blink_dur < SLOW_BLINK_DURATION_US) {
                    s_total_blinks++;
                }
            }
            if (s_was_eye_closed && !gap) out_metrics->eye.reset_reason = "opened";
            reset_eye_timer();
        }
    }

    out_metrics->eye.closed_ms = (float)eye_closed_duration / 1000.0f;

    // 5. Mouth State & Yawn Detection
    // [D22] GRACE: chỉ reset bộ đếm khi miệng ĐÓNG LIÊN TỤC >= 2 frame. Nhờ vậy khử được
    // hiện tượng model nhấp nháy (0.99 -> 0.22 trong 1 frame) mà KHÔNG làm mất đỉnh ngáp
    // như bộ lọc trượt trước đây.
    bool mouth_open = (mar > s_mar_threshold);
    int64_t mouth_open_duration = 0;
    static int s_mouth_closed_cnt = 0;

    if (mouth_open) {
        s_mouth_closed_cnt = 0;
        if (s_mouth_open_start_us == 0) {
            s_mouth_open_start_us = now;
        }
        mouth_open_duration = now - s_mouth_open_start_us;

        if (mouth_open_duration >= YAWN_EVENT_DURATION_US && !s_is_currently_yawning) {
            s_is_currently_yawning = true;
            s_total_yawns++;
            // Record yawn timestamp in rolling window
            if (s_yawn_history_count < MAX_YAWN_HISTORY) {
                s_yawn_timestamps[s_yawn_history_count++] = now;
            } else {
                // Shift left
                memmove(&s_yawn_timestamps[0], &s_yawn_timestamps[1], sizeof(int64_t) * (MAX_YAWN_HISTORY - 1));
                s_yawn_timestamps[MAX_YAWN_HISTORY - 1] = now;
            }
            ESP_LOGI(TAG, "🥱 Phát hiện sự kiện NGÁP (Lần thứ %u)!", (unsigned int)s_total_yawns);
        }
    } else {
        s_mouth_closed_cnt++;
        if (s_mouth_closed_cnt >= 2) {
            s_mouth_open_start_us = 0;
            s_is_currently_yawning = false;
        }
    }

    // Count recent yawns within 3-minute rolling window
    int recent_yawns = 0;
    for (int i = 0; i < s_yawn_history_count; i++) {
        if (now - s_yawn_timestamps[i] <= YAWN_WINDOW_US) {
            recent_yawns++;
        }
    }

    // 6. Head Pose Distraction Detection
    bool is_distracted_angle = false;
    if (pose->is_valid) {
        if (fabsf(pose->yaw - s_yaw_bias) > s_yaw_threshold_deg || fabsf(pose->pitch - s_pitch_bias) > s_pitch_threshold_deg) {
            is_distracted_angle = true;
        }
    }

    int64_t distraction_duration = 0;
    if (is_distracted_angle) {
        if (s_distraction_start_us == 0) {
            s_distraction_start_us = now;
        }
        distraction_duration = now - s_distraction_start_us;
    } else {
        s_distraction_start_us = 0;
    }

    // 7. Determine Final ADAS State & Alarm Priority
    adas_state_t current_state = ADAS_STATE_NORMAL;
    bool alarm = false;
    const char* status_text = "NORMAL";

    // [13/11] Quyết định theo SỐ MẪU liên tiếp (FPS-adaptive): 1,5 s @4,3 FPS = 7 mẫu,
    // 1,5 s @0,12 FPS = 2 mẫu (không thể phân biệt chớp mắt với nhắm ở tốc độ đó).
    // [14/11] Microsleep: >= k_micro mau NHAM trong cua so (k_micro+1 khi thoi gian lay mau thua)
    // -> o 0,12 FPS chi can 2/3 mau la bao, mot mau bi miss khong xoa dot.
    int eye_win = s_k_micro + (s_k_micro <= 3 ? 1 : 0);
    if (eye_win > s_eye_hist_n) eye_win = s_eye_hist_n;
    int eye_closed_in_window = 0;
    for (int i = 0; i < eye_win; i++) {
        int idx = (s_eye_hist_pos - 1 - i + EYE_HIST_LEN * 2) % EYE_HIST_LEN;
        if (s_eye_hist[idx] == 1) eye_closed_in_window++;
    }
    const bool eye_micro = (eye_win >= s_k_micro) && (eye_closed_in_window >= s_k_micro);
    const bool eye_slow = eye_sample_valid && (s_eye_closed_samples >= s_k_slow);

    if (eye_micro) {
        current_state = ADAS_STATE_MICROSLEEP_ALARM;
        alarm = true;
        status_text = "MICROSLEEP ALARM!";
    } else if (distraction_duration >= DISTRACTION_DURATION_US) {
        current_state = ADAS_STATE_DISTRACTION_ALARM;
        alarm = true;
        status_text = "DISTRACTION ALARM!";
    } else if (recent_yawns >= 3) {
        current_state = ADAS_STATE_FATIGUE_ALARM;
        alarm = true;
        status_text = "FATIGUE ALARM (>=3 YAWNS)!";
    } else if (eye_slow) {
        current_state = ADAS_STATE_SLOW_BLINK_WARNING;
        status_text = "SLOW BLINK WARNING";
    } else if (mouth_open_duration >= YAWN_EVENT_DURATION_US || s_is_currently_yawning) {
        current_state = ADAS_STATE_YAWN_WARNING;
        status_text = "YAWNING WARNING";
    }

    // Populate output metrics
    out_metrics->ear_left = ear_l;
    out_metrics->ear_right = ear_r;
    out_metrics->ear = ear;
    out_metrics->mar = mar;
    out_metrics->pose = *pose;
    out_metrics->state = current_state;
    out_metrics->is_alarm_active = alarm;
    out_metrics->total_blinks = s_total_blinks;
    out_metrics->total_yawns = s_total_yawns;
    out_metrics->esp32_fps = fps;
    out_metrics->ear_threshold = s_ear_threshold;
    out_metrics->mar_threshold = s_mar_threshold;
    out_metrics->mouth_open_s = (float)mouth_open_duration / 1e6f;
    out_metrics->eye_prob = s_eye_prob;
    strncpy(out_metrics->status_str, status_text, sizeof(out_metrics->status_str));
}

// [14/11] Calibration window finished? (exposed for /status)
bool adas_controller_is_calibrated(void) { return s_is_calibrated; }

// [14/11] Re-run the calibration window (pose bias + EAR/MAR baseline); alarm state is not touched.
void adas_controller_request_recalibration(void) {
    adas_controller_init();
    ESP_LOGI("adas_ctl", "recalibration requested from the web UI");
}



