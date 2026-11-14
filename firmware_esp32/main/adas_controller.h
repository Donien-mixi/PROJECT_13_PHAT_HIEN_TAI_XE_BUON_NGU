#ifndef ADAS_CONTROLLER_H_
#define ADAS_CONTROLLER_H_

#include <stdint.h>
#include <stdbool.h>
#include "pnp_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ADAS_STATE_CALIBRATING = 0,
    ADAS_STATE_NORMAL,
    ADAS_STATE_SLOW_BLINK_WARNING,
    ADAS_STATE_MICROSLEEP_ALARM,
    ADAS_STATE_YAWN_WARNING,
    ADAS_STATE_FATIGUE_ALARM,
    ADAS_STATE_DISTRACTION_ALARM
} adas_state_t;

typedef struct {
    float age_ms;
    float closed_ms;
    bool valid;
    bool closed;
    const char* reset_reason;
} adas_eye_diag_t;

typedef struct {
    adas_eye_diag_t eye;
    float ear_left;
    float ear_right;
    float ear;
    float mar;
    head_pose_t pose;
    adas_state_t state;
    bool is_alarm_active;
    char status_str[32];
    uint32_t total_blinks;
    uint32_t total_yawns;
    float esp32_fps;
    // [v2.8.0 - DISPLAY] Ngưỡng hiện tại + thời lượng há miệng, để Laptop vẽ trực quan.
    float ear_threshold;
    float mar_threshold;
    float mouth_open_s;
    // [11/11] Student eye-state probability (P(closed)), -1 when unavailable.
    float eye_prob;
} adas_metrics_t;

/** Set the student eye-state probability for the current frame (-1 = unavailable). */
void adas_controller_set_eye_prob(float p_closed);

/**
 * [13/11] Enable/disable the eye path for the next sample. The pipeline disables it
 * when the face detector found no face: PIPNet then runs on the fallback centre ROI
 * and returns garbage eyelid points (EAR 0.03 -> would look like "eyes closed").
 */
void adas_controller_set_eye_enabled(bool enabled);

/**
 * Initializes the ADAS Controller state machine and timers.
 */
void adas_controller_init(void);

/**
 * Processes facial landmarks and head pose, updates the ADAS FSM,
 * and produces biometric decision metrics.
 *
 * @param landmarks_22 Array of 22 normalized landmarks [0.0, 1.0].
 * @param pose Calculated 3D head pose (Yaw, Pitch, Roll).
 * @param fps Current processing frame rate.
 * @param out_metrics Pointer to receive the updated ADAS metrics.
 */
void adas_controller_update(const point2d_t landmarks_22[22],
                            const head_pose_t* pose,
                            float fps,
                            adas_metrics_t* out_metrics);

/** Update with the capture timestamp in the esp_timer clock domain. */
void adas_controller_update_capture(const point2d_t landmarks_22[22],
                                    const head_pose_t* pose,
                                    float fps,
                                    int64_t capture_us,
                                    adas_metrics_t* out_metrics);

/** Break the eye episode when a sample is lost. */
void adas_controller_invalidate_eye_sample(void);

/** [D23] Lấy bias pose đo lúc hiệu chuẩn (độ) để HIỂN THỊ pose đã trừ bias
 *  -> trục X/Y/Z đúng khi mặt hướng thẳng (nếu không sẽ nghiêng theo bias). */
float adas_controller_get_pitch_bias(void);
float adas_controller_get_yaw_bias(void);
/** [14/11] EAR baseline (p75 lúc hiệu chuẩn) — dùng cho luật tụt tương đối. */
float adas_controller_get_ear_baseline(void);
/** [14/11] true khi cửa sổ hiệu chuẩn đã xong (dùng cho /status). */
bool adas_controller_is_calibrated(void);
/**
 * [14/11] Chạy lại cửa sổ hiệu chuẩn (bias pose + baseline EAR/MAR) mà không cần reboot.
 * Nối với nút "Hiệu chuẩn lại" trên trang web: ngồi thẳng, nhìn camera rồi bấm.
 */
void adas_controller_request_recalibration(void);

#ifdef __cplusplus
}
#endif

#endif // ADAS_CONTROLLER_H_
