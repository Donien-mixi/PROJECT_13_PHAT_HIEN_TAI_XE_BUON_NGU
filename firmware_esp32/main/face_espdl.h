#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool face_espdl_init(void);
bool face_espdl_ready(void);

/**
 * Run the 98-point PIPNet landmark model on a grayscale frame region.
 *
 * @param gray  full-resolution grayscale frame
 * @param w,h   frame size
 * @param bx,by,bs  face box (x, y, side) in pixels
 * @param pts98 output 98 points in the SAME pixel coordinates as the frame
 * @param out_us optional latency
 */
bool face_espdl_landmarks(const uint8_t* gray, int w, int h,
                          int bx, int by, int bs,
                          float pts98[98][2], int64_t* out_us);

#ifdef __cplusplus
}
#endif

