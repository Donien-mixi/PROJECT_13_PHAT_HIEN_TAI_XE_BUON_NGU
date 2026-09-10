#include "adas_controller.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_log.h"
#else
#include <chrono>
static int64_t esp_timer_get_time() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) fprintf(stderr, "[ERROR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

static const char* TAG = "ADAS_CONTROLLER";

// Timing thresholds (microseconds)
#define CALIBRATION_DURATION_US   (5 * 1000 * 1000)   // 5 seconds calibration
#define SLOW_BLINK_DURATION_US    (500 * 1000)        // 0.5s slow blink
#define MICROSLEEP_DURATION_US    (1500 * 1000)       // 1.5s microsleep
#define YAWN_EVENT_DURATION_US    (1500 * 1000)       // 1.5s mouth wide open
#define DISTRACTION_DURATION_US   (3000 * 1000)       // 3.0s head turned away
#define YAWN_WINDOW_US            (180 * 1000 * 1000) // 3 minutes rolling window

#define MAX_YAWN_HISTORY 10

// Internal Controller State
static int64_t s_start_time_us = 0;
static bool s_is_calibrated = false;

// Calibration Accumulators
static float s_calib_ear_sum = 0.0f;
static float s_calib_mar_sum = 0.0f;
static uint32_t s_calib_samples = 0;

// Adaptive Dynamic Thresholds
static float s_ear_threshold = 0.21f;
static float s_mar_threshold = 0.45f;
static const float s_yaw_threshold_deg = 30.0f;
static const float s_pitch_threshold_deg = 25.0f;

// Duration Tracking Timers
static int64_t s_eye_closed_start_us = 0;
static int64_t s_mouth_open_start_us = 0;
static int64_t s_distraction_start_us = 0;

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

void adas_controller_init(void) {
    s_start_time_us = esp_timer_get_time();
    s_is_calibrated = false;
    s_calib_ear_sum = 0.0f;
    s_calib_mar_sum = 0.0f;
    s_calib_samples = 0;

    s_eye_closed_start_us = 0;
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
    int64_t now = esp_timer_get_time();

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
    float mar = (mouth_w > 1e-4f) ? ((mouth_h_outer + mouth_h_inner) / (2.0f * mouth_w)) : 0.0f;

    // 3. Calibration Phase (First 5 Seconds)
    if (!s_is_calibrated) {
        if (now - s_start_time_us < CALIBRATION_DURATION_US) {
            s_calib_ear_sum += ear;
            s_calib_mar_sum += mar;
            s_calib_samples++;

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
            snprintf(out_metrics->status_str, sizeof(out_metrics->status_str), "CALIBRATING... (%.1fs)",
                     (float)(CALIBRATION_DURATION_US - (now - s_start_time_us)) / 1e6f);
            return;
        } else {
            // Calibration Complete
            if (s_calib_samples > 10) {
                float avg_ear = s_calib_ear_sum / s_calib_samples;
                float avg_mar = s_calib_mar_sum / s_calib_samples;
                s_ear_threshold = avg_ear * 0.75f;
                s_mar_threshold = avg_mar * 1.60f;
                if (s_ear_threshold < 0.18f) s_ear_threshold = 0.18f;
                if (s_ear_threshold > 0.25f) s_ear_threshold = 0.25f;
                if (s_mar_threshold < 0.40f) s_mar_threshold = 0.40f;
                ESP_LOGI(TAG, "✅ Hiệu chuẩn thành công! Baseline: EAR=%.2f, MAR=%.2f | Ngưỡng báo: EAR<%.2f, MAR>%.2f",
                         avg_ear, avg_mar, s_ear_threshold, s_mar_threshold);
            }
            s_is_calibrated = true;
        }
    }

    // 4. Eye State & Blink / Microsleep Detection
    bool eye_closed = (ear < s_ear_threshold);
    int64_t eye_closed_duration = 0;

    if (eye_closed) {
        if (s_eye_closed_start_us == 0) {
            s_eye_closed_start_us = now;
        }
        eye_closed_duration = now - s_eye_closed_start_us;
        s_was_eye_closed = true;
    } else {
        if (s_was_eye_closed && s_eye_closed_start_us > 0) {
            int64_t blink_dur = now - s_eye_closed_start_us;
            if (blink_dur > (100 * 1000) && blink_dur < SLOW_BLINK_DURATION_US) {
                s_total_blinks++;
            }
        }
        s_eye_closed_start_us = 0;
        s_was_eye_closed = false;
    }

    // 5. Mouth State & Yawn Detection
    bool mouth_open = (mar > s_mar_threshold);
    int64_t mouth_open_duration = 0;

    if (mouth_open) {
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
        s_mouth_open_start_us = 0;
        s_is_currently_yawning = false;
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
        if (fabsf(pose->yaw) > s_yaw_threshold_deg || fabsf(pose->pitch) > s_pitch_threshold_deg) {
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

    if (eye_closed_duration >= MICROSLEEP_DURATION_US) {
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
    } else if (eye_closed_duration >= SLOW_BLINK_DURATION_US) {
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
    strncpy(out_metrics->status_str, status_text, sizeof(out_metrics->status_str));
}
