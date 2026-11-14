#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// BlazeFace short-range (MediaPipe) INT8, output of
//   onnx2tf -oiqt blazeface_nchw.onnx -> blazeface_nchw_full_integer_quant.tflite
// Executed with TFLM on the ESP32-S3; tensor scale/zero-point come from the model.
bool blaze_init(void);
bool blaze_detect(const uint8_t* gray, int w, int h, float box[4], float* score);
bool blaze_track(float box[4], float score, float dt_s, float out_box[4]);
void blaze_reset(void);

// Diagnostics for telemetry / serial measurement.
float blaze_last_ms(void);      // wall time of the last blaze_detect (ms)
int blaze_last_cand(void);      // candidates above the collection threshold
float blaze_last_score(void);   // top-1 score of the last detect
int blaze_arena_used(void);     // TFLM tensor arena actually used (bytes)
void blaze_debug_dump(const uint8_t* gray, int w, int h);  // parity diagnostics

// Fusion helper: IoU between the detector box and a landmark-derived face box
// (both x0,y0,x1,y1 in frame pixels). Used to decide which box drives the ROI.
float blaze_box_iou(const float a[4], const float b[4]);

// [13/11] Fusion state: the pipeline reports the bbox of the PIPNet 98 landmarks each
// frame (previous frame at detect time, because detect runs before inference).
// Measured IoU(detector, landmarks) on stills: 0,873 / 0,831 -> below FUSION_MIN_IOU
// the landmark box wins (and it is the only ROI source when the detector misses).
void blaze_fusion_set_landmarks(const float box[4], bool valid);
bool blaze_fusion_get(float out_det[4], float out_lm[4], float* out_iou, bool* out_used_lm);

#ifdef __cplusplus
}
#endif
