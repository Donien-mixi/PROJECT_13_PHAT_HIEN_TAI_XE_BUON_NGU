#include "face_espdl.h"

#include <math.h>
#include <string.h>
#include <string>

#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "FACE_ESPDL";
static dl::Model *s_lm = nullptr;
static bool s_ready = false;

#define LM_SIZE 256
#define LM_NUM 98

bool face_espdl_init(void) {
    // Weights stay in flash (param_copy = false): only activations need PSRAM.
    s_lm = new dl::Model("pipnet", fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                         8 * 1024, dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    if (!s_lm || !s_lm->is_built()) {
        ESP_LOGE(TAG, "pipnet load FAILED (psram=%uKB sram=%uKB)",
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
        s_lm = nullptr;
        return false;
    }
    for (auto &kv : s_lm->get_outputs()) {
        auto *t2 = kv.second;
        char sh[64] = {0}; int q = 0;
        for (size_t i = 0; i < t2->shape.size(); i++)
            q += snprintf(sh + q, sizeof(sh) - q, "%d%s", t2->shape[i], i + 1 < t2->shape.size() ? "x" : "");
        ESP_LOGI(TAG, "OUTNAMES %s shape=%s exp=%d", kv.first.c_str(), sh, (int)t2->exponent);
    }
    ESP_LOGI(TAG, "PIPNet-98 ready (single model, weights in flash)");
    s_ready = true;
    return true;
}

bool face_espdl_ready(void) { return s_ready; }

bool face_espdl_landmarks(const uint8_t *gray, int w, int h,
                          int bx, int by, int bs,
                          float pts98[98][2], int64_t *out_us) {
    if (!s_ready || !s_lm || !gray || !pts98 || bs <= 0) return false;
    int64_t t0 = esp_timer_get_time();

    auto *in = s_lm->get_inputs().begin()->second;
    int8_t *dst = (int8_t *)in->data;
    float cx = (float)bx + bs * 0.5f, cy = (float)by + bs * 0.5f;
    float side = (float)bs * 1.5f;
    if (side > (float)w) side = (float)w;
    if (side > (float)h) side = (float)h;
    float x0 = cx - side * 0.5f, y0 = cy - side * 0.5f;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + side > (float)w) x0 = (float)w - side;
    if (y0 + side > (float)h) y0 = (float)h - side;
    float step = side / (float)LM_SIZE;

    for (int oy = 0; oy < LM_SIZE; oy++) {
        int py = (int)(y0 + (oy + 0.5f) * step + 0.5f);
        if (py < 0) py = 0;
        if (py >= h) py = h - 1;
        for (int ox = 0; ox < LM_SIZE; ox++) {
            int px = (int)(x0 + (ox + 0.5f) * step + 0.5f);
            if (px < 0) px = 0;
            if (px >= w) px = w - 1;
            int8_t q = (int8_t)((int)gray[py * w + px] - 128);
            int b = (oy * LM_SIZE + ox) * 3;
            dst[b] = q; dst[b + 1] = q; dst[b + 2] = q;
        }
    }

    int64_t t_pre = esp_timer_get_time();
    s_lm->run();
    int64_t t_run = esp_timer_get_time();

    // Find the three 98-channel heads; their storage layout is decided by ESP-DL,
    // so element access goes through get_element_index() instead of manual index math.
    dl::TensorBase *cls = nullptr, *xs = nullptr, *ys = nullptr;
    bool nhwc = true;
    int GH = 8, GW = 8;
    for (auto &kv : s_lm->get_outputs()) {
        auto *t = kv.second;
        if (t->shape.size() != 4) continue;
        int a = t->shape[1], b = t->shape[2], c = t->shape[3];
        bool hwc = (c == LM_NUM);
        int H = hwc ? a : b, W = hwc ? b : c;
        if (!hwc && a != LM_NUM) continue;
        nhwc = hwc;
        GH = H; GW = W;
        const std::string &nm = kv.first;
        if (nm == "cls") cls = t;
        else if (nm == "x") xs = t;
        else if (nm == "y") ys = t;
    }
    if (!cls || !xs || !ys) {
        ESP_LOGE(TAG, "heads not found (cls=%p x=%p y=%p)", (void *)cls, (void *)xs, (void *)ys);
        return false;
    }

    float s_cls = exp2f((float)(int)cls->exponent);
    float s_x = exp2f((float)(int)xs->exponent);
    float s_y = exp2f((float)(int)ys->exponent);
    const int8_t *cdata = (const int8_t *)cls->data;
    const int8_t *xdata = (const int8_t *)xs->data;
    const int8_t *ydata = (const int8_t *)ys->data;

    auto elem = [&](dl::TensorBase *t, const int8_t *d, int hh, int ww, int ch) -> float {
        std::vector<int> coord = nhwc ? std::vector<int>{0, hh, ww, ch}
                                      : std::vector<int>{0, ch, hh, ww};
        return (float)d[t->get_element_index(coord)];
    };
    (void)elem;

    for (int i = 0; i < LM_NUM; i++) {
        int best_h = 0, best_w = 0;
        float bv = -1e9f;
        for (int hh = 0; hh < GH; hh++) {
            for (int ww = 0; ww < GW; ww++) {
                std::vector<int> coord = nhwc ? std::vector<int>{0, hh, ww, i}
                                              : std::vector<int>{0, i, hh, ww};
                int idx = cls->get_element_index(coord);
                float v = (float)cdata[idx] * s_cls;
                if (v > bv) { bv = v; best_h = hh; best_w = ww; }
            }
        }
        std::vector<int> coord = nhwc ? std::vector<int>{0, best_h, best_w, i}
                                      : std::vector<int>{0, i, best_h, best_w};
        int xi = xs->get_element_index(coord);
        int yi = ys->get_element_index(coord);
        float px = ((float)best_w + (float)xdata[xi] * s_x) / (float)GW;
        float py = ((float)best_h + (float)ydata[yi] * s_y) / (float)GH;
        pts98[i][0] = x0 + px * side;
        pts98[i][1] = y0 + py * side;
    }
    static bool s_dumped = false;
    if (!s_dumped) {
        s_dumped = true;
        const int per = 32;
        for (int off = 0; off < 98 * 64; off += per) {
            char line[3 * 32 + 8]; int p = 0;
            for (int i = 0; i < per; i++) p += snprintf(line + p, sizeof(line) - p, "%02x", (uint8_t)cdata[off + i]);
            ESP_LOGI(TAG, "HEADCLS %d %s", off, line);
        }
        for (int off = 0; off < 98 * 64; off += per) {
            char line[3 * 32 + 8]; int p = 0;
            for (int i = 0; i < per; i++) p += snprintf(line + p, sizeof(line) - p, "%02x", (uint8_t)xdata[off + i]);
            ESP_LOGI(TAG, "HEADX %d %s", off, line);
        }
    }
    static int s_log = 0;
    if ((s_log++ % 3) == 0) {
        ESP_LOGI(TAG, "pip98 L=(%.0f,%.0f) R=(%.0f,%.0f) grid=%dx%d hwc=%d",
                 pts98[60][0], pts98[60][1], pts98[68][0], pts98[68][1], GH, GW, (int)nhwc);
    }
    if (out_us) *out_us = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "[LM split] preprocess %.1f ms | model %.1f ms | decode %.1f ms",
             (double)(t_pre - t0) / 1000.0, (double)(t_run - t_pre) / 1000.0,
             (double)(esp_timer_get_time() - t_run) / 1000.0);
    return true;
}





