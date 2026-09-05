#ifndef TELEMETRY_SENDER_H_
#define TELEMETRY_SENDER_H_

#include <stdint.h>
#include <stdbool.h>
#include "adas_controller.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the UDP telemetry socket and hardware actuators (GPIO Buzzer & LED).
 * @return true on success.
 */
bool telemetry_sender_init(void);

/**
 * Encodes ADAS metrics into JSON and sends via UDP to the Laptop Host (Port 8889).
 * Also drives the physical hardware actuators (Buzzer and LED).
 *
 * @param metrics Pointer to current ADAS metrics and state.
 * @param landmarks_22 Array of 22 normalized landmarks to overlay on HUD.
 * @return true if packet sent successfully.
 */
bool telemetry_sender_dispatch(const adas_metrics_t* metrics, const point2d_t landmarks_22[22]);

/**
 * Actuates hardware buzzer and warning LED based on ADAS alarm state.
 */
void telemetry_sender_update_actuators(bool is_alarm_active);

#ifdef __cplusplus
}
#endif

#endif // TELEMETRY_SENDER_H_
