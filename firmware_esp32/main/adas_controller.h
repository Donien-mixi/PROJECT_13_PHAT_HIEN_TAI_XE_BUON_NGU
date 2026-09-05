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
} adas_metrics_t;

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

#ifdef __cplusplus
}
#endif

#endif // ADAS_CONTROLLER_H_
