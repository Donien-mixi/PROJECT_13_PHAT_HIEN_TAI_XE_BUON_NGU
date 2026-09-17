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
 * [v2.8.0 - DISPLAY] Cập nhật thời gian xử lý từng khâu (ms) để Laptop hiển thị
 * trực quan độ trễ Dec/AI/Total. Thuần bổ sung - KHÔNG ảnh hưởng AI/ADAS/PnP.
 */
void telemetry_sender_set_timing(float dec_ms, float ai_ms, float total_ms);

/**
 * [v2.9.2 - DIAG] Cập nhật thông tin Face ROI đang dùng (để hiển thị/chẩn đoán).
 * active=1 nếu crop theo khuôn mặt, 0 nếu crop giữa khung.
 */
void telemetry_sender_set_roi(int active, int x0, int y0, int size);

/**
 * Actuates hardware buzzer and warning LED based on ADAS alarm state.
 */
void telemetry_sender_update_actuators(bool is_alarm_active);

/**
 * [v2.7.0 - DISPLAY ONLY] Gửi nguyên tensor xám 96x96 (đã lượng tử INT8) về Laptop
 * để HIỂN THỊ đúng những gì ESP32 "nhìn thấy". Gói nhị phân:
 *   magic AA56AA56 (4B) + w (uint16 LE) + h (uint16 LE) + pixel (w*h, uint8)
 * Thuần bổ sung - KHÔNG ảnh hưởng AI/ADAS/PnP.
 */
void telemetry_sender_image(const int8_t* tensor, float scale, int32_t zp, int w, int h);

/**
 * [v2.9.4 - DIAG] Gửi preview TOÀN KHUNG camera (đã downscale) để Laptop thấy
 * đúng những gì OV5640 chụp. Gói: magic AA56A001 + w + h + pixel.
 */
void telemetry_sender_preview(const uint8_t* gray, int src_w, int src_h, int out_w, int out_h);

#ifdef __cplusplus
}
#endif

#endif // TELEMETRY_SENDER_H_
