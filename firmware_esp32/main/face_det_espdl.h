#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Face detection with the pretrained ESP-DL MSR/MNP models (no training). */
bool face_det_espdl_init(void);
bool face_det_espdl_ready(void);

/**
 * Detect the largest/highest-score face in a grayscale frame.
 * @param gray   8-bit grayscale image (w*h)
 * @param w,h    frame size
 * @param box    output [x1,y1,x2,y2] in pixels
 * @param score  optional output score
 * @return true when a face was found
 */
bool face_det_espdl_run(const uint8_t* gray, int w, int h, float box[4], float* score);

#ifdef __cplusplus
}
#endif
