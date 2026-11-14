#ifndef OBS_SERVER_H_
#define OBS_SERVER_H_

#include <stdbool.h>
#include <stdint.h>

#include "image_decoder.h"   // image_crop_roi_t
#include "adas_controller.h" // adas_metrics_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * [14/11] Minimal observation server: Wi-Fi STA + HTTP, so the board can be watched from a
 * phone/PC in the same LAN *without* touching the AI pipeline.
 *
 *   GET /               tiny HTML page (auto-refreshing snapshot + live numbers)
 *   GET /status         JSON with the current frame metrics
 *   GET /snapshot.bmp   320x240 grey BMP with the ROI rectangle and the 98 landmarks drawn
 *
 * Credentials live in wifi_secret.h, which is gitignored on purpose.
 *
 * @return true when the HTTP server is listening (Wi-Fi may still be associating).
 */
bool obs_server_start(void);

/**
 * Publish the result of the frame that was just processed. Cheap: copies the grey frame into the
 * BMP buffer and draws ROI + landmarks (~1 ms). Safe to call every frame.
 */
void obs_server_publish(const uint8_t* gray, int w, int h,
                        const image_crop_roi_t* roi,
                        const float pts98[98][2],
                        const adas_metrics_t* metrics,
                        float det_score, float det_ms,
                        uint32_t frame_idx);

#ifdef __cplusplus
}
#endif

#endif // OBS_SERVER_H_
