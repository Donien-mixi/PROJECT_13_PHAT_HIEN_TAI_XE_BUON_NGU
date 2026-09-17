#ifndef FACE_DETECTOR_LITE_H_
#define FACE_DETECTOR_LITE_H_

#include <stdbool.h>
#include <stdint.h>
#include "face_detector.h"   // face_box_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * [D5] Face detector NHẸ (96x96 grayscale) chạy MỖI FRAME.
 * Model = Mobile-Inverted-Bottleneck ~227k params (INT8 mixed-precision), output 4 float32
 *   -> box = out*FDL_STD + FDL_MEAN (norm), map về pixel khung gốc.
 * Nguồn độc lập với TinyDriverNet (chỉ để xác định vùng crop).
 */
bool face_detector_lite_init(void);
bool face_detector_lite_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out);

/** Thời gian suy luận (ms) của lần chạy gần nhất — để log. */
float face_detector_lite_last_ms(void);

#ifdef __cplusplus
}
#endif

#endif // FACE_DETECTOR_LITE_H_
