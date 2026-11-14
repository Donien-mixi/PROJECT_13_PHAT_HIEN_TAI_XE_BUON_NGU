#include "face_detector.h"

#include <math.h>

#include "face_det_espdl.h"

static float s_threshold = 0.5f;

bool face_detector_init(void) { return face_det_espdl_init(); }

void face_detector_set_threshold(float thr) { s_threshold = thr; }

bool face_detector_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out) {
    if (!out) return false;
    out->valid = false;
    float box[4] = {0, 0, 0, 0};
    float score = 0.0f;
    if (!face_det_espdl_run(gray, src_w, src_h, box, &score)) return false;
    if (score < s_threshold) return false;

    out->cx = (box[0] + box[2]) * 0.5f;
    out->cy = (box[1] + box[3]) * 0.5f;
    out->size = fmaxf(box[2] - box[0], box[3] - box[1]);
    out->score = score;
    out->valid = true;
    out->has_kp = false;
    return true;
}
