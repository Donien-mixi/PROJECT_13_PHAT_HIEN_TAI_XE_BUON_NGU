#include "face_det_espdl.h"

#include "dl_image_define.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "human_face_detect.hpp"

static const char *TAG = "FACE_DET_ESPDL";
static HumanFaceDetect *s_det = nullptr;
static uint8_t *s_rgb = nullptr;
static bool s_ready = false;

#define PIPNET_NO_DETECTOR 1   // landmark self-tracking (no external detector)

bool face_det_espdl_init(void) {
#if PIPNET_NO_DETECTOR
    // PIPNet mode: the MSR/MNP detector cannot be loaded alongside the 8 model parts.
    // ROI comes from the previous landmarks (tracker) or the frame centre on the first frame.
    ESP_LOGW(TAG, "MSR/MNP disabled: PIPNet mode uses landmark-tracked ROI");
    s_det = nullptr;
    s_ready = false;
    return false;
#endif
    ESP_LOGI(TAG, "detector init: psram=%uKB sram=%uKB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    s_det = new HumanFaceDetect();
    if (!s_det) {
        ESP_LOGE(TAG, "HumanFaceDetect alloc failed");
        return false;
    }
    s_det->set_score_thr(0.25f, 0);   // MSR stage
    s_det->set_score_thr(0.20f, 1);   // MNP stage (default 0.5 was dropping everything)
    s_det->set_nms_thr(0.3f, 0);
    s_det->set_nms_thr(0.3f, 1);
    if (!s_rgb) s_rgb = (uint8_t *)heap_caps_malloc(320 * 240 * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_rgb) {
        ESP_LOGE(TAG, "rgb buffer alloc failed");
        return false;
    }
    s_ready = true;
    // The detector is optional: PIPNet can run from a fallback ROI and then track
    // its own landmarks, so a failed detector must not take the pipeline down.
    ESP_LOGI(TAG, "MSR/MNP face detector loaded (%p)", (void *)s_det);
    return s_det != nullptr && s_ready;
}

bool face_det_espdl_ready(void) { return s_ready; }

bool face_det_espdl_run(const uint8_t* gray, int w, int h, float box[4], float* score) {
    if (!s_ready || !s_det || !gray || !box || w <= 0 || h <= 0) return false;
    // MSR/MNP expect RGB888: replicate the grayscale frame into RGB.
    int npix = w * h;
    for (int i = 0; i < npix; i++) {
        s_rgb[i * 3 + 0] = gray[i];
        s_rgb[i * 3 + 1] = gray[i];
        s_rgb[i * 3 + 2] = gray[i];
    }
    dl::image::img_t img = {};
    img.data = (void*)s_rgb;
    img.width = (uint16_t)w;
    img.height = (uint16_t)h;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_BGR888;   // experiment: MSR may expect BGR

    auto& results = s_det->run(img);
    if (results.empty()) return false;
    const auto& r = results.front();
    for (int i = 0; i < 4; i++) box[i] = r.box[i];
    if (score) *score = r.score;
    return true;
}











