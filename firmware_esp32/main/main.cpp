#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"

// Submodules
#include "wifi_stream_client.h"
#include "camera_capture.h"
#include "image_decoder.h"
#include "obs_server.h"
#include "roi_tracker.h"
#include "face_detector.h"
#include "face_detector_lite.h"
#include "blaze_tracker.h"   // [13/11] fusion landmark <-> detector
#include "ai_inference.h"
#include "face_espdl.h"
#include "face_det_espdl.h"
#include "pnp_solver.h"
#include "adas_controller.h"
#include "telemetry_sender.h"
#include "web_server.h"
#ifndef TD_OBS_ENABLE
#define TD_OBS_ENABLE 1   // [14/11] Wi-Fi + HTTP observation server (/ , /status , /snapshot.bmp)
#endif

#ifndef APP_ENABLE_WEB
#define APP_ENABLE_WEB 0   // PIPNet mode: inference + alarms only, no dashboard/stream
#endif
// Legacy TinyDriver model removed: the landmark model is now ESP-DL 2d106det.
// The 96x96 crop constants below only describe the decoder crop that still feeds
// the diagnostic tensor; inference itself uses the full frame + ROI.
#define TINYDRIVER_NUM_LANDMARKS 22
#define TINYDRIVER_INPUT_WIDTH 96
#define TINYDRIVER_INPUT_HEIGHT 96
#define TINYDRIVER_INPUT_SCALE 0.0078125f
#define TINYDRIVER_INPUT_ZERO_POINT (-128)
// [14/11] ROI scale: measured best on 13 real driver frames (k=1.20 -> 100 % plausible,
// k=1.08 -> 92 %, k=1.50 -> 77 %). tools/roi_scale_sweep.py
#ifndef TD_CROP_K
#define TD_CROP_K 1.080f
#endif

static const char* TAG = "MAIN_APP";

// [13/11] Frame capture for dataset / calibration collection. The board has no SD card and
// the web server is disabled in PIPNet mode, so the only way to get frames out is the serial
// console: each frame is dumped as hex rows tagged with its (y, x0) offset, prefixed by
// CAPTURE BEGIN <idx> <w> <h> and closed by CAPTURE END <idx>. Requires 921600 baud
// (CONFIG_ESP_CONSOLE_UART_BAUDRATE) to move ~144 KB per frame in ~1,6 s.
#ifndef BLAZE_CAPTURE_MODE
#define BLAZE_CAPTURE_MODE 0
#endif
#ifndef BLAZE_CAPTURE_COUNT
#define BLAZE_CAPTURE_COUNT 40
#endif
#ifndef BLAZE_SNAP_COUNT
#define BLAZE_SNAP_COUNT 20
#endif
#ifndef BLAZE_CAPTURE_RAW
#define BLAZE_CAPTURE_RAW 0   // 1 = also dump the raw pre-inference frame
#endif

#if BLAZE_CAPTURE_MODE
// [14/11] Visual snapshot for the serial viewer: the LAST processed frame with the ROI
// rectangle and the 98 landmarks drawn on it, sent over UART in the CAPTURE protocol so
// tools/blaze_frame_capture.py turns it into a PNG the user can open.
static int s_snap_idx = 0;
static void capture_dump_overlay(const uint8_t* g, int w, int h, const image_crop_roi_t* roi,
                                 const float pts[98][2], float ear, float thr, float score) {
    if (!g || w <= 0 || h <= 0 || s_snap_idx >= BLAZE_SNAP_COUNT) return;
    static uint8_t buf[320 * 240];
    if (w > 320 || h > 240) return;
    size_t n = (size_t)w * h;
    memcpy(buf, g, n);
    // ROI rectangle (value 255)
    if (roi && roi->size > 0) {
        int x0 = roi->x0, y0 = roi->y0, x1 = roi->x0 + roi->size - 1, y1 = roi->y0 + roi->size - 1;
        if (x1 >= w) x1 = w - 1;
        if (y1 >= h) y1 = h - 1;
        for (int x = x0; x <= x1; x++) { if (y0 >= 0) buf[y0 * w + x] = 255; if (y1 < h) buf[y1 * w + x] = 255; }
        for (int y = y0; y <= y1; y++) { if (x0 >= 0) buf[y * w + x0] = 255; if (x1 < w) buf[y * w + x1] = 255; }
    }
    // landmarks: 2x2 blocks, value 255 (pupils 96/97 use 200 to distinguish)
    if (pts) {
        for (int k = 0; k < 98; k++) {
            int px = (int)(pts[k][0] + 0.5f), py = (int)(pts[k][1] + 0.5f);
            uint8_t v = (k >= 96) ? 200 : 255;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int x = px + dx, y = py + dy;
                    if (x >= 0 && x < w && y >= 0 && y < h) buf[y * w + x] = v;
                }
            }
        }
    }
    ESP_LOGI("CAPTURE", "BEGIN %d %d %d EAR %.3f THR %.3f CLOSED %.0f", s_snap_idx, w, h,
             (double)ear, (double)thr, (double)score);
    char line[3 * 64 + 16];
    for (int y = 0; y < h; y++) {
        for (int x0 = 0; x0 < w; x0 += 64) {
            int p = 0;
            for (int x = x0; x < x0 + 64 && x < w; x++)
                p += snprintf(line + p, sizeof(line) - p, "%02x", buf[y * w + x]);
            ESP_LOGI("CAPTURE", "ROW %d %d %s", y, x0, line);
        }
    }
    ESP_LOGI("CAPTURE", "END %d", s_snap_idx);
    s_snap_idx++;
}
#endif

#if BLAZE_CAPTURE_MODE
static int s_cap_idx = 0;
static void capture_dump_gray(const uint8_t* g, int w, int h) {
    if (!g || w <= 0 || h <= 0 || s_cap_idx >= BLAZE_CAPTURE_COUNT) return;
    char line[3 * 64 + 16];
    ESP_LOGI("CAPTURE", "BEGIN %d %d %d", s_cap_idx, w, h);
    for (int y = 0; y < h; y++) {
        for (int x0 = 0; x0 < w; x0 += 64) {
            int p = 0;
            for (int x = x0; x < x0 + 64 && x < w; x++)
                p += snprintf(line + p, sizeof(line) - p, "%02x", g[y * w + x]);
            ESP_LOGI("CAPTURE", "ROW %d %d %s", y, x0, line);
        }
    }
    ESP_LOGI("CAPTURE", "END %d", s_cap_idx);
    s_cap_idx++;
}
#endif

static uint32_t s_device_frame_counter = 0;

// [D2] Crop do DETECTOR (BlazeFace) xác định (nguồn độc lập với TinyDriverNet)
static float s_pts98_frame[98][2] = {};   // last 98 landmarks in frame pixels
static bool s_face_present = false;   // detector found a face on this frame (gates the eye path)
static image_crop_roi_t s_det_crop = {0, 0, 0, 0, 0};
static bool s_det_crop_valid = false;
static int  s_detect_countdown = 0;   // (dự phòng) đếm frame tới lần detect lại
static int64_t s_det_crop_us = 0;     // [P1] thoi diem s_det_crop duoc cap nhat (do tuoi ROI)

// ============================================================================
// [P1 - TRUY VET DO TRE] Thong ke trung vi (p50) / p95 tren cua so truot.
// KHONG thay doi hanh vi pipeline - chi do luong va log.
// ============================================================================
#define TD_STAT_N 64
typedef struct {
    float v[TD_STAT_N];
    int   n;      // so mau hien co (toi da TD_STAT_N)
    int   head;   // vi tri ghi tiep theo
} td_stat_ring_t;

static void td_stat_add(td_stat_ring_t* r, float x) {
    r->v[r->head] = x;
    r->head = (r->head + 1) % TD_STAT_N;
    if (r->n < TD_STAT_N) r->n++;
}

// pct: 0..100. Tra ve gia tri phan vi pct (noi suy khong, lam tron len).
static float td_stat_pct(const td_stat_ring_t* r, float pct) {
    if (r->n <= 0) return -1.0f;
    float tmp[TD_STAT_N];
    memcpy(tmp, r->v, sizeof(float) * (size_t)r->n);
    // insertion sort (n <= 64)
    for (int i = 1; i < r->n; i++) {
        float key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = key;
    }
    int idx = (int)ceilf(pct / 100.0f * (float)(r->n - 1));
    if (idx < 0) idx = 0;
    if (idx > r->n - 1) idx = r->n - 1;
    return tmp[idx];
}

static td_stat_ring_t s_st_img_age  = {};   // capture -> publish (ms)
static td_stat_ring_t s_st_roi_age  = {};   // ROI duoc tao -> capture cua frame ap dung (ms)
static td_stat_ring_t s_st_dec      = {};   // decode (ms)
static td_stat_ring_t s_st_ai       = {};   // suy luan AI (ms)
static td_stat_ring_t s_st_core1    = {};   // nhan slot -> publish (ms)
static td_stat_ring_t s_st_total    = {};   // vong lap Core 1 (ms, gom cho buffer)

// [P1] Bo dem chan doan (khong thay doi hanh vi)
static uint32_t s_p1_stage_skip     = 0;   // JPEG vuot TD_STAGE_JPEG_MAX -> khong stage duoc
static uint32_t s_p1_stage_mismatch = 0;   // JPEG stage khac frame ma AI vua xu ly
static uint32_t s_p1_cam_fail       = 0;   // camera acquire that bai
static uint32_t s_p1_dec_fail       = 0;   // decode JPEG loi
static uint32_t s_p1_lite_miss      = 0;   // lite detector khong thay mat
static uint32_t s_p1_cam_gap        = 0;   // so frame camera bi nhay (seq gap)
static uint32_t s_p1_last_cam_seq   = 0;

// [D4b] Detector chạy SONG SONG trên Core 0 (không chặn vòng AI ở Core 1)
static portMUX_TYPE s_crop_mux = portMUX_INITIALIZER_UNLOCKED;
static uint8_t* s_pub_gray = NULL;          // PSRAM: bản sao ảnh xám cho detector task
static volatile bool s_pub_busy = false;    // true khi detector task đang dùng s_pub_gray
static int  s_pub_w = 0, s_pub_h = 0;
static TaskHandle_t s_det_task = NULL;
static volatile uint32_t s_redetect_count = 0;

// [D6] Detector chay tren Core 0? (BlazeFace dinh ky HOAC Lite moi frame)
#define TD_DET_C0_ENABLED (CONFIG_TD_DETECT_PERIODIC || \
                           (CONFIG_TD_USE_LITE_DETECTOR && CONFIG_TD_LITE_ON_CORE0))

// [D16] PIPELINE HAI NHAN: gop DECODE + PREVIEW vao chung task Core 0 voi Lite detector
//   Core 0: decode (36ms) + preview (10ms) + lite (80ms)  ~ 126ms
//   Core 1: AI 22 diem (124ms) + ADAS + telemetry          ~ 134ms
//   => thoi gian/frame = max(126,134) ~ 134ms  (truoc day 170ms vi decode+AI noi tiep)
#define TD_PIPELINE_DECODE_C0 (CONFIG_TD_USE_LITE_DETECTOR && CONFIG_TD_LITE_ON_CORE0)

// [P4] Che do chan doan tuan tu: tat pipeline Core 0 de detector chay tren
// CHINH frame vua decode (khong dung ROI cua frame truoc).
// [14/11] NGOAI LE co kiem soat: TD_ROI_REUSE_STABLE=1 van bat task detector Core 0 de co ROI
// cua khung TRUOC, nhung pipeline chi dung ROI do khi dau da dung yen (xem roi_reuse_ok()).
// Ly do: A/B 14/11 cho thay dung ROI khung truoc VO DIEU KIEN lam hong landmark khi quay dau
// (EAR 0,934 / score 0,5 / box 103 px), trong khi luc dau dung yen thi tiet kiem ~0,5 s/khung.
#ifndef TD_ROI_REUSE_STABLE
#define TD_ROI_REUSE_STABLE 0
#endif
// Gate for the hybrid: the head counts as steady only if the smoothed yaw/pitch barely changed
// between the last two frames, and the published ROI is younger than ~2 frames.
#ifndef TD_ROI_STABLE_DEG
#define TD_ROI_STABLE_DEG 8.0f
#endif
#ifndef TD_ROI_MAX_AGE_US
#define TD_ROI_MAX_AGE_US 20000000LL
#endif
static float s_pose_yaw_prev = 0.0f, s_pose_yaw_prev2 = 0.0f;
static float s_pose_pitch_prev = 0.0f, s_pose_pitch_prev2 = 0.0f;
static uint32_t s_roi_reused = 0;
#if CONFIG_TD_DIAG_SEQUENTIAL
#undef TD_DET_C0_ENABLED
#if TD_ROI_REUSE_STABLE
#define TD_DET_C0_ENABLED 1          // cho task Core 0 chay song song de co ROI khung truoc
#else
#define TD_DET_C0_ENABLED 0
#endif
#undef TD_PIPELINE_DECODE_C0
#define TD_PIPELINE_DECODE_C0 0

// [P4] Staging JPEG (PSRAM) de publish anh + telemetry cung frame o cuoi vong lap.
#define TD_DIAG_JPEG_MAX (40 * 1024)
static uint8_t* s_diag_jpeg = NULL;
static size_t   s_diag_len = 0;
#endif

// [D3] Suy ra vùng crop canonical từ hộp mặt (ưu tiên 6 keypoints -> khớp khung lúc train)
static void td_crop_from_box(const face_box_t* b, int gw, int gh) {
    int minDim = (gw < gh) ? gw : gh;

    float bcx = b->cx, bcy = b->cy, bsz = b->size;
    int64_t now_us = esp_timer_get_time();

#if CONFIG_TD_DIAG_SEQUENTIAL
    // [P4] CHAN DOAN: dung box THO cua detector tren chinh frame nay.
    // Tat du doan van toc + EMA de do dung baseline (khong che lap sai so di chuyen).
    float fS  = TD_CROP_K * bsz;
    float fcx = bcx + 0.040f * bsz;
    float fcy = bcy - 0.028f * bsz;
#else
    // [D17] BÙ VẬN TỐC + LỌC GẦN-ZERO-LAG (sửa "tracking không tới")
    //   ĐO THỰC TẾ: pipeline có trễ 1 frame (~143ms) + EMA cũ (α=0.22..0.5) trễ thêm
    //   2-4.5 frame => tổng 430-780ms => mặt đã đi 30-50px => khung + 22 điểm lệch.
    //   Giải pháp: ước lượng vận tốc hộp mặt rồi DỰ ĐOÁN vị trí tới (bù trễ),
    //   đồng thời nâng alpha gần như tức thời khi mặt đang di chuyển.
    static float s_prev_cx = 0.0f, s_prev_cy = 0.0f, s_prev_sz = 0.0f;
    static bool  s_prev_ok = false;
    static int64_t s_prev_us = 0;
    static float s_sm_cx = 0.0f, s_sm_cy = 0.0f, s_sm_sz = 0.0f;
    static bool  s_sm_ready = false;

    float dt = 0.0f;
    if (s_prev_ok && now_us > s_prev_us) {
        dt = (float)(now_us - s_prev_us) / 1e6f;
    }

    // Vận tốc (px/s) — LÀM MƯỢT (EMA) + DEADBAND.
    // ĐO THỰC TẾ: khi đứng yên, nhiễu hộp detector ~1.55px/frame -> nếu đem chia dt
    // để tính vận tốc thì "dự đoán" sẽ KHUẾCH ĐẠI nhiễu thành rung 22 điểm (2px/frame).
    // => làm mượt vận tốc + coi vận tốc < ngưỡng là ĐỨNG YÊN (không dự đoán).
    static float s_vcx = 0.0f, s_vcy = 0.0f, s_vsz = 0.0f;
    float vcx = 0.0f, vcy = 0.0f, vsz = 0.0f;
    if (s_prev_ok && dt > 1e-3f && dt < 1.0f) {
        const float VMAX = 600.0f;      // px/s
        float ncx = (bcx - s_prev_cx) / dt;
        float ncy = (bcy - s_prev_cy) / dt;
        float nsz = (bsz - s_prev_sz) / dt;
        if (ncx > VMAX) ncx = VMAX;
        if (ncx < -VMAX) ncx = -VMAX;
        if (ncy > VMAX) ncy = VMAX;
        if (ncy < -VMAX) ncy = -VMAX;
        if (nsz > VMAX) nsz = VMAX;
        if (nsz < -VMAX) nsz = -VMAX;
        s_vcx += 0.5f * (ncx - s_vcx);
        s_vcy += 0.5f * (ncy - s_vcy);
        s_vsz += 0.5f * (nsz - s_vsz);
        const float VDEAD = 40.0f;      // px/s: dưới ngưỡng = đứng yên -> KHÔNG dự đoán
        vcx = (fabsf(s_vcx) > VDEAD) ? s_vcx : 0.0f;
        vcy = (fabsf(s_vcy) > VDEAD) ? s_vcy : 0.0f;
        vsz = (fabsf(s_vsz) > VDEAD) ? s_vsz : 0.0f;
    }
    s_prev_cx = bcx; s_prev_cy = bcy; s_prev_sz = bsz; s_prev_ok = true; s_prev_us = now_us;

    // Dự đoán vị trí mặt TỚI (bù trễ pipeline ~0.15s)
    const float LEAD_S = 0.15f;
    float px = bcx + vcx * LEAD_S;
    float py = bcy + vcy * LEAD_S;
    float psz = bsz + vsz * LEAD_S;

    if (!s_sm_ready) {
        s_sm_cx = px; s_sm_cy = py; s_sm_sz = psz;
        s_sm_ready = true;
    } else {
        // Lọc nhẹ để khử nhiễu detector nhưng KHÔNG gây trễ khi mặt đang di chuyển.
        // Đứng yên (speed=0 sau deadband) -> alpha nhỏ (0.30) -> giảm rung 22 điểm.
        float speed = sqrtf(vcx * vcx + vcy * vcy);          // px/s
        float a = (speed > 120.0f) ? 0.95f : ((speed > 40.0f) ? 0.75f : 0.30f);
        s_sm_cx += a * (px  - s_sm_cx);
        s_sm_cy += a * (py  - s_sm_cy);
        s_sm_sz += a * (psz - s_sm_sz);
    }

    // [D9] Hiệu chuẩn LẠI box -> crop canonical bằng MEDIAPIPE ground-truth trên
    // 25 khung hình OV5640 thật (K=1.080 std .052 | OX=+0.040 | OY=-0.028).
    // Trước đây K=0.965 (crop chặt 11%) và OY=-0.090 (lệch cao ~7px -> CẮT MIỆNG/CẰM).
    float fS  = TD_CROP_K * s_sm_sz;
    float fcx = s_sm_cx + 0.040f * s_sm_sz;
    float fcy = s_sm_cy - 0.028f * s_sm_sz;
#endif  // CONFIG_TD_DIAG_SEQUENTIAL
    if (b->has_kp) {
        float ex = (b->kp[0][0] + b->kp[1][0]) * 0.5f;
        float ey = (b->kp[0][1] + b->kp[1][1]) * 0.5f;
        float nx = b->kp[2][0], ny = b->kp[2][1];
        float d_eyes = hypotf(b->kp[1][0] - b->kp[0][0], b->kp[1][1] - b->kp[0][1]);
        float d_eye_nose = fmaxf(fmaxf(ny - ey, 0.45f * d_eyes), 1.0f);
        float h_skull = fmaxf(2.10f * d_eye_nose, 1.40f * d_eyes);
        float S = 2.05f * h_skull;
        if (S > 40.0f && S < 2.0f * (float)minDim) {
            fS = S;
            fcx = (ex + nx) * 0.5f;
            fcy = ey + 0.32f * h_skull;
        }
    }
    int cs = (int)lroundf(fS);
    if (cs < 64) cs = 64;
    if (cs > minDim) cs = minDim;
    int x0 = (int)lroundf(fcx) - cs / 2;
    int y0 = (int)lroundf(fcy) - cs / 2;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + cs > gw) x0 = gw - cs;
    if (y0 + cs > gh) y0 = gh - cs;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    portENTER_CRITICAL(&s_crop_mux);
    s_det_crop.x0 = x0; s_det_crop.y0 = y0; s_det_crop.size = cs;
    s_det_crop.src_w = gw; s_det_crop.src_h = gh;
    s_det_crop_valid = true;
    s_det_crop_us = now_us;   // [P1] moc thoi gian tao ROI (de do ROI age khi ap dung)
    portEXIT_CRITICAL(&s_crop_mux);
}

// [D4b] Task Core 0: nhận ảnh xám đã publish -> detect -> cập nhật crop
#if CONFIG_TD_DETECT_PERIODIC
static void vTaskDetector(void* arg) {
    ESP_LOGI(TAG, "🎯 Detector task chạy song song trên Core 0");
    float prev_cx = 0, prev_cy = 0, prev_sz = 0;
    bool  prev_ok = false;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        int gap_ms = 20;
        if (!s_pub_gray || s_pub_w <= 0 || s_pub_h <= 0) {
            s_pub_busy = false;
            continue;
        }
        face_box_t b;
        int64_t t0 = esp_timer_get_time();
        if (face_detector_run(s_pub_gray, s_pub_w, s_pub_h, &b) && b.valid) {
            td_crop_from_box(&b, s_pub_w, s_pub_h);
            s_redetect_count++;
            // [D4c] GAP THÍCH ỨNG: mặt đang di chuyển -> bám sát; đứng yên -> nghỉ lâu
            float moved = 0.0f;
            if (prev_ok) {
                moved = fabsf(b.cx - prev_cx) + fabsf(b.cy - prev_cy) + fabsf(b.size - prev_sz);
            }
            prev_cx = b.cx; prev_cy = b.cy; prev_sz = b.size; prev_ok = true;
            gap_ms = (moved > 12.0f) ? 60 : CONFIG_TD_DETECT_GAP_IDLE_MS;
            ESP_LOGI(TAG, "REDETECT(C0): score=%.2f kp=%d -> crop=(%d,%d,%d) moved=%.0f gap=%dms [%lld ms]",
                     b.score, (int)b.has_kp, s_det_crop.x0, s_det_crop.y0, s_det_crop.size,
                     moved, gap_ms, (long long)((esp_timer_get_time() - t0) / 1000));
        } else {
            prev_ok = false;
            gap_ms = CONFIG_TD_DETECT_GAP_IDLE_MS;
        }
        s_pub_busy = false;
        vTaskDelay(pdMS_TO_TICKS(gap_ms));
    }
}
#endif  // CONFIG_TD_DETECT_PERIODIC

// [D6] Task Core 0: chạy LITE detector SONG SONG với mạng 22 điểm ở Core 1.
// Core 1 publish ảnh xám (không chặn) -> Core 0 detect (~66ms) -> cập nhật crop.
#if CONFIG_TD_USE_LITE_DETECTOR && CONFIG_TD_LITE_ON_CORE0
static void vTaskLiteDetectorC0(void* arg) {
    ESP_LOGI(TAG, "FD_LITE: task Core 0 san sang (song song voi AI o Core 1)");
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (!s_pub_gray || s_pub_w <= 0 || s_pub_h <= 0) {
            s_pub_busy = false;
            continue;
        }
        face_box_t lb;
        if (face_detector_lite_run(s_pub_gray, s_pub_w, s_pub_h, &lb) && lb.valid) {
            td_crop_from_box(&lb, s_pub_w, s_pub_h);
        }
        s_pub_busy = false;
    }
}
#endif

// ============================================================================
// [D16] Task Core 0: DECODE JPEG + PREVIEW + LITE DETECTOR (pipeline voi Core 1)
//   DOUBLE-BUFFER: Core 0 giai ma frame N+1 vao buffer B trong khi Core 1 chay AI
//   tren buffer A (frame N) => DECODE KHONG CON NAM TREN DUONG GANG.
//   Do luong: 1 buffer -> Total = Dec(34) + AI(130) = 169ms (FPS 5.9)
//             2 buffer -> frame = max(Dec+Preview+Lite ~121, AI+telemetry ~135)
//                       = ~135ms -> FPS ~7.4
// ============================================================================
#if TD_PIPELINE_DECODE_C0
// [14/11] Double buffer: the per-slot initialisers below carry TWO entries ({NULL, NULL}, {0, 0}...),
// so NBUF must be 2 — with 1 the build fails ("too many initializers"). This path had been disabled
// by CONFIG_TD_DIAG_SEQUENTIAL, which is why the mismatch was never caught.
#define TD_DEC_NBUF 2
static int8_t  s_dec_buf[TD_DEC_NBUF][TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT];
static image_crop_roi_t s_dec_roi[TD_DEC_NBUF];
static int64_t s_dec_us[TD_DEC_NBUF];
static uint32_t s_dec_frame_idx[TD_DEC_NBUF];
static volatile bool s_dec_ok[TD_DEC_NBUF];
static SemaphoreHandle_t s_buf_free[TD_DEC_NBUF]  = {NULL, NULL};   // Core 0 cho (buffer ranh)
static SemaphoreHandle_t s_buf_ready[TD_DEC_NBUF] = {NULL, NULL};   // Core 1 cho (buffer san sang)
static StaticSemaphore_t s_buf_free_storage[TD_DEC_NBUF];           // cap phat TINH -> khong the that bai
static StaticSemaphore_t s_buf_ready_storage[TD_DEC_NBUF];
// [D21] Staging JPEG theo tung slot: de Core 1 phat len web CUNG LUC voi du lieu AI.
// LY DO: web stream lay frame camera tho (~25-30fps) con overlay (22 diem + ROI + MAR)
// chi ~7fps va tre them ~140ms (pipeline AI) => lop phu LUON DI SAU video => nhin nhu
// "diem sai cho / mieng sai / quay dau sai / lag" khi nguoi dung di chuyen hay ngap.
// Phat frame cua CHINH frame ma AI vua xu ly => video va overlay KHOP NHAU.
#define TD_STAGE_JPEG_MAX (40 * 1024)
static uint8_t* s_stage_jpeg[TD_DEC_NBUF] = {NULL, NULL};
static size_t   s_stage_len[TD_DEC_NBUF] = {0, 0};
static volatile int s_dec_slot = 0;      // buffer Core 1 dang dung (chi Core 1 ghi/doc)
static TaskHandle_t s_decode_task = NULL;
static TaskHandle_t s_ai_task_h = NULL;

// [P1 - TRUY VET] Metadata theo tung slot: noi ket qua AI voi DUNG frame/JPEG/ROI
// ma no da xu ly (khong dung ROI/frame "moi nhat" toan cuc).
static uint32_t s_stage_cam_seq[TD_DEC_NBUF]    = {0, 0};       // cam_seq cua JPEG dang o staging
static int64_t  s_dec_cap_us[TD_DEC_NBUF]       = {0, 0};       // thoi diem DMA capture cua frame
static uint32_t s_dec_cam_seq[TD_DEC_NBUF]      = {0, 0};       // so thu tu frame camera
static int64_t  s_dec_roi_us[TD_DEC_NBUF]       = {0, 0};       // thoi diem ROI (s_det_crop) duoc tao
static bool     s_dec_roi_ts_valid[TD_DEC_NBUF] = {false, false};
static int      s_dec_roi_src[TD_DEC_NBUF]      = {WEB_ROI_SRC_CENTER, WEB_ROI_SRC_CENTER};

static void vTaskDecodeC0(void* arg) {
    ESP_LOGI(TAG, "D16: task DECODE+PREVIEW+LITE tren Core 0 (double-buffer) san sang");
    int cur = 0;
    for (;;) {
        // Cho buffer[cur] duoc tra ve (ban dau semaphore da duoc give tu app_main)
        if (xSemaphoreTake(s_buf_free[cur], portMAX_DELAY) != pdTRUE) continue;
        int64_t t0 = esp_timer_get_time();

#if CONFIG_TD_USE_ONBOARD_CAMERA
        camera_frame_t cam;
        if (!camera_capture_acquire(&cam)) {
            s_p1_cam_fail++;
            s_dec_ok[cur] = false;
            xSemaphoreGive(s_buf_ready[cur]);
            cur ^= 1;
            continue;
        }
        const uint8_t* jpeg = cam.data;
        size_t jlen = cam.length;
        s_dec_cap_us[cur]  = cam.timestamp_us;   // [P1]
        s_dec_cam_seq[cur] = cam.seq;            // [P1]
        // [D21] Chi COPY vao staging, KHONG phat len web o day (doi AI xu ly xong)
        if (s_stage_jpeg[cur] && jlen > 0 && jlen <= TD_STAGE_JPEG_MAX) {
            memcpy(s_stage_jpeg[cur], jpeg, jlen);
            s_stage_len[cur] = jlen;
            s_stage_cam_seq[cur] = cam.seq;      // [P1] JPEG nay thuoc frame nao
        } else {
            // [P2] KHONG giu length/anh cu: tranh publish JPEG cu voi landmark moi.
            s_stage_len[cur] = 0;
            s_p1_stage_skip++;
        }
#else
        frame_buffer_t frameb;
        if (!wifi_stream_acquire_latest_frame(&frameb, 100)) {
            s_dec_ok[cur] = false;
            xSemaphoreGive(s_buf_ready[cur]);
            cur ^= 1;
            continue;
        }
        const uint8_t* jpeg = frameb.buffer;
        size_t jlen = frameb.length;
        s_dec_cap_us[cur]  = esp_timer_get_time();   // [P1] TCP mode: khong co timestamp DMA
        s_dec_cam_seq[cur] = frameb.frame_index;     // [P1]
        if (s_stage_jpeg[cur] && jlen > 0 && jlen <= TD_STAGE_JPEG_MAX) {
            memcpy(s_stage_jpeg[cur], jpeg, jlen);
            s_stage_len[cur] = jlen;
            s_stage_cam_seq[cur] = frameb.frame_index;
        } else {
            s_stage_len[cur] = 0;   // [P2] khong tai su dung anh/do dai cu
            s_p1_stage_skip++;
        }
#endif

        // ROI: uu tien detector, fallback landmark tracker (giong laptop)
        image_crop_roi_t lroi = {0, 0, 0, 0, 0}, troi = {0, 0, 0, 0, 0};
        const image_crop_roi_t* p_roi = NULL;
        int64_t roi_ts = 0;
        portENTER_CRITICAL(&s_crop_mux);
        bool hv = s_det_crop_valid;
        if (hv) { lroi = s_det_crop; roi_ts = s_det_crop_us; }
        portEXIT_CRITICAL(&s_crop_mux);
        s_dec_roi_us[cur]       = roi_ts;   // [P1] ROI nay duoc tao tu luc nao
        s_dec_roi_ts_valid[cur] = hv;
        if (hv) {
            p_roi = &lroi;
        } else if (roi_tracker_is_valid()) {
            face_roi_t fr = roi_tracker_get();
            troi.x0 = fr.x0; troi.y0 = fr.y0; troi.size = fr.size;
            p_roi = &troi;
        }
        s_dec_roi_src[cur] = hv ? WEB_ROI_SRC_DETECTOR_PREV
                           : (p_roi ? WEB_ROI_SRC_TRACKER
                                    :
#if CONFIG_TD_CROP_FIXED
                                      WEB_ROI_SRC_FIXED
#else
                                      WEB_ROI_SRC_CENTER
#endif
                           );

        bool ok = image_decoder_process_jpeg_roi(
            jpeg, jlen, s_dec_buf[cur],
            TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT,
            p_roi, &s_dec_roi[cur]);

#if CONFIG_TD_USE_ONBOARD_CAMERA
        camera_capture_release();
#else
        wifi_stream_release_frame();
#endif
        s_dec_us[cur] = esp_timer_get_time() - t0;
        s_dec_frame_idx[cur] = ++s_device_frame_counter;
        s_dec_ok[cur] = ok;
        if (!ok) s_p1_dec_fail++;

        // Buffer cur da day du -> tra cho Core 1 (AI chay SONG SONG phan duoi day)
        xSemaphoreGive(s_buf_ready[cur]);

        if (ok) {
            int gw = 0, gh = 0;
            const uint8_t* g = image_decoder_get_last_gray(&gw, &gh);
            if (g && gw > 0 && gh > 0) {
                // Preview toan khung cho laptop (buffer pkt rieng, an toan da luong)
                #if APP_ENABLE_WEB
        telemetry_sender_preview(g, gw, gh, 128, 96);
#endif
                // Lite detector -> cap nhat crop canonical cho frame KE TIEP
                face_box_t lb;
                if (face_detector_lite_run(g, gw, gh, &lb) && lb.valid) {
                    td_crop_from_box(&lb, gw, gh);
                } else {
                    s_p1_lite_miss++;   // [P1] detector miss -> ROI cu se duoc dung lai
                }
            }
        }
        cur ^= 1;
    }
}
#endif  // TD_PIPELINE_DECODE_C0

// [D2] Detect 1 lần lúc khởi động -> suy ra vùng crop (thay cho crop cố định)
static void td_run_startup_detection(void) {
#if CONFIG_TD_USE_ONBOARD_CAMERA && CONFIG_TD_DIAG_SEQUENTIAL && CONFIG_TD_USE_LITE_DETECTOR
    ESP_LOGI(TAG, "P4: Lite handles same-frame ROI; skipping BlazeFace startup detection");
#elif CONFIG_TD_USE_ONBOARD_CAMERA
    if (!face_detector_init()) {
        ESP_LOGW(TAG, "Detector init that bai -> dung crop co dinh");
        return;
    }
    for (int tries = 0; tries < 10 && !s_det_crop_valid; tries++) {
        camera_frame_t cam;
        if (!camera_capture_acquire(&cam)) {
            vTaskDelay(pdMS_TO_TICKS(150));
            continue;
        }
        image_crop_roi_t used = {0, 0, 0, 0, 0};
        bool ok = image_decoder_process_jpeg_roi(
            cam.data, cam.length, ai_inference_get_input_buffer(),
            TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT, NULL, &used);
        camera_capture_release();
        if (!ok) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        int gw = 0, gh = 0;
        const uint8_t* g = image_decoder_get_last_gray(&gw, &gh);
        if (!g) continue;

        int64_t t0 = esp_timer_get_time();
        face_box_t b;
        bool found = face_detector_run(g, gw, gh, &b);
        s_face_present = found;
        int64_t dt_ms = (esp_timer_get_time() - t0) / 1000;

        if (found) {
            td_crop_from_box(&b, gw, gh);
            ESP_LOGI(TAG, "DETECT OK: score=%.2f kp=%d -> crop=(%d,%d,%d) [%lld ms]",
                     b.score, (int)b.has_kp, s_det_crop.x0, s_det_crop.y0, s_det_crop.size,
                     (long long)dt_ms);
        } else {
            ESP_LOGW(TAG, "DETECT: khong thay mat (lan %d, %lld ms)", tries + 1, (long long)dt_ms);
        }
    }
    if (!s_det_crop_valid) {
        ESP_LOGW(TAG, "Khong detect duoc -> dung crop co dinh");
    }
#endif
}

// Core 1 Task: Edge AI & ADAS State Machine Execution
static void vTaskEdgeAI_ADAS(void* pvParameters) {
    ESP_LOGI(TAG, "🚀 Khởi chạy Edge AI & ADAS Task trên Core 1 (Priority 6, 240MHz)");

    int8_t* input_tensor_ptr = ai_inference_get_input_buffer();
    point2d_t landmarks[TINYDRIVER_NUM_LANDMARKS];
    head_pose_t head_pose;
    adas_metrics_t adas_metrics;

    uint32_t processed_frames = 0;
    int64_t fps_timer_start = esp_timer_get_time();
    float current_fps = 0.0f;

#if TD_PIPELINE_DECODE_C0
    s_ai_task_h = xTaskGetCurrentTaskHandle();   // [D16] Core 0 bao khi buffer san sang
#endif

    while (1) {
        int64_t t0 = esp_timer_get_time();
        uint32_t frame_index = 0;
        image_crop_roi_t roi_used = {0, 0, 0, 0, 0};
        image_crop_roi_t local_roi = {0, 0, 0, 0, 0};
        const image_crop_roi_t* p_roi = NULL;
        int64_t t_decode = 0;
        bool has_roi = false;
        int dec_slot_used = 0;   // [D21] slot buffer AI vua xu ly (de phat web dung frame)
        // [P1] Metadata cua DUNG frame ma AI xu ly (khong phai "moi nhat" toan cuc)
        uint32_t cam_seq = 0;
        int64_t  cap_us = 0;
        int64_t  roi_ts_us = 0;
        bool     roi_ts_valid = false;
        int64_t  t_slot0 = 0;
        // [COMPARE] Metadata chan doan cua cung frame
        bool        diag_face_valid = false;
        face_box_t  diag_face = {};
        head_pose_t pose_raw = {};
        head_pose_t pose_smoothed = {};
        int         roi_source = WEB_ROI_SRC_CENTER;

#if TD_PIPELINE_DECODE_C0
        // [D16] Cho buffer DA GIAI MA SAN (double-buffer). Core 0 da giai ma frame nay
        // TRONG LUC ta dang chay AI cua frame truoc => decode khong con tren duong gang.
        {
            int slot = s_dec_slot;
            if (s_buf_ready[slot] == NULL || xSemaphoreTake(s_buf_ready[slot], portMAX_DELAY) != pdTRUE) {
                vTaskDelay(1);
                continue;
            }
            if (!s_dec_ok[slot]) {                 // frame loi -> tra buffer, thu frame ke
                adas_controller_invalidate_eye_sample();
                xSemaphoreGive(s_buf_free[slot]);
                s_dec_slot ^= 1;
                continue;
            }
            t_slot0     = esp_timer_get_time();    // [P1] moc bat dau xu ly frame nay
            frame_index = s_dec_frame_idx[slot];
            roi_used    = s_dec_roi[slot];
            t_decode    = s_dec_us[slot];
            cam_seq     = s_dec_cam_seq[slot];     // [P1]
            cap_us      = s_dec_cap_us[slot];      // [P1]
            roi_ts_us   = s_dec_roi_us[slot];      // [P1]
            roi_ts_valid = s_dec_roi_ts_valid[slot];
            roi_source  = s_dec_roi_src[slot];     // [COMPARE] nguon ROI da dung
            memcpy(input_tensor_ptr, s_dec_buf[slot],
                   (size_t)TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT);
            // [P2 - FIX RACE] KHONG tra slot o day: s_stage_jpeg[slot] con duoc doc
            // sau khi suy luan xong (luc publish web). Tra som -> Core 0 ghi de JPEG
            // cua frame khac -> dashboard hien anh moi + landmark cu (lech khi di chuyen).
            // Slot duoc tra sau khi da publish xong (xem cuoi vong lap).
            s_dec_slot ^= 1;
            dec_slot_used = slot;                  // [D21] nho slot de phat web sau khi AI xong
        }
        portENTER_CRITICAL(&s_crop_mux);
        has_roi = s_det_crop_valid;
        if (has_roi) local_roi = s_det_crop;
        portEXIT_CRITICAL(&s_crop_mux);
        if (has_roi) {
            p_roi = &local_roi;
        }
#else
        const uint8_t* jpeg_data = NULL;
        size_t jpeg_len = 0;

#if CONFIG_TD_USE_ONBOARD_CAMERA
        camera_frame_t cam_frame;
        if (!camera_capture_acquire(&cam_frame)) {
            adas_controller_invalidate_eye_sample();
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }
        jpeg_data = cam_frame.data;
        jpeg_len = cam_frame.length;
        frame_index = ++s_device_frame_counter;
        cam_seq = cam_frame.seq;              // [P1]
        cap_us  = cam_frame.timestamp_us;     // [P1]
        t_slot0 = esp_timer_get_time();
#if CONFIG_TD_DIAG_SEQUENTIAL
        // [P4] Chup JPEG vao staging rieng -> publish anh + telemetry cung frame.
        if (s_diag_jpeg && jpeg_len > 0 && jpeg_len <= TD_DIAG_JPEG_MAX) {
            memcpy(s_diag_jpeg, jpeg_data, jpeg_len);
            s_diag_len = jpeg_len;
        } else {
            s_diag_len = 0;
            s_p1_stage_skip++;
        }
#else
        web_server_update_frame(jpeg_data, jpeg_len);
#endif
#else
        frame_buffer_t frame;
        // 1. Acquire the latest 1:1 JPEG frame from PSRAM Double Buffer
        if (!wifi_stream_acquire_latest_frame(&frame, 100)) {
            adas_controller_invalidate_eye_sample();
            // Waiting for frame from laptop
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        jpeg_data = frame.buffer;
        jpeg_len = frame.length;
        frame_index = frame.frame_index;
        cam_seq = frame.frame_index;          // [P1]
        cap_us  = esp_timer_get_time();       // [P1] TCP mode: khong co timestamp DMA
        t_slot0 = esp_timer_get_time();
#if CONFIG_TD_DIAG_SEQUENTIAL
        if (s_diag_jpeg && jpeg_len > 0 && jpeg_len <= TD_DIAG_JPEG_MAX) {
            memcpy(s_diag_jpeg, jpeg_data, jpeg_len);
            s_diag_len = jpeg_len;
        } else {
            s_diag_len = 0;
            s_p1_stage_skip++;
        }
#else
        web_server_update_frame(jpeg_data, jpeg_len);
#endif
#endif

        // 2. Fast JPEG Decode & Isomorphic Downsample directly to 96x96 INT8
        //    [v2.9.0] Nếu ROI khuôn mặt đã sẵn sàng -> crop sát mặt (giống lúc train),
        //    ngược lại tự crop giữa khung (bootstrap).
        // [D8] Ưu tiên 1 (GIỐNG LAPTOP): DETECTOR là nguồn chính MỖI FRAME
        //   (laptop: MediaPipe/Haar). Đọc an toàn qua mutex.
#if CONFIG_TD_DIAG_SEQUENTIAL
        // [P4] TUAN TU CUNG FRAME: decode -> Lite detect tren chinh frame ->
        // crop canonical -> 96x96. Khong dung ROI frame truoc, khong du doan/EMA.
        bool dec_ok = false;
        {
            int gw = 0, gh = 0;
            // [14/11] RAW GRAY: the OV5640 can hand over the Y plane of YUV422 directly, so the
            // 0.60 s JPEG decode disappears (tools: benchmark boards 8,2 s -> ~7,6 s/frame).
            bool raw_ok = false;
#if TD_CAM_RAW_GRAY
            {
                raw_ok = image_decoder_load_gray_raw(jpeg_data, jpeg_len, &gw, &gh);
            }
#endif
            if (raw_ok || image_decoder_decode_jpeg(jpeg_data, jpeg_len, &gw, &gh)) {
                const uint8_t* gfull = image_decoder_get_last_gray(&gw, &gh);
                face_box_t db;
                bool found = false;
                bool roi_reused = false;
                image_crop_roi_t prev_roi;
                int64_t prev_roi_us = 0;
#if TD_ROI_REUSE_STABLE
                // [14/11] Hybrid: while the head is steady, reuse the ROI the Core-0 detector made
                // on the PREVIOUS frame (saves the 0,54 s inline detect). As soon as the head moves
                // (yaw/pitch) or the ROI is stale, fall back to the same-frame detector so a turn
                // can never feed PIPNet an out-of-date crop (the failure measured on build B).
                if (gfull && gw > 0 && gh > 0) {
                    bool have = false;
                    portENTER_CRITICAL(&s_crop_mux);
                    have = s_det_crop_valid;
                    prev_roi = s_det_crop;
                    prev_roi_us = s_det_crop_us;
                    portEXIT_CRITICAL(&s_crop_mux);
                    const float dyaw = fabsf(s_pose_yaw_prev - s_pose_yaw_prev2);
                    const float dpit = fabsf(s_pose_pitch_prev - s_pose_pitch_prev2);
                    const int64_t age = cap_us - prev_roi_us;
                    if (have && dyaw < TD_ROI_STABLE_DEG && dpit < TD_ROI_STABLE_DEG &&
                        age > 0 && age < TD_ROI_MAX_AGE_US) {
                        found = true;
                        roi_reused = true;
                        s_roi_reused++;
                    }
                }
                if (!roi_reused && gfull && gw > 0 && gh > 0) {
                    found = face_detector_lite_run(gfull, gw, gh, &db) && db.valid;
                    // face_detector_lite_run() always returns true with a fallback box whose
                    // score is 0, so "face present" must come from the detector score itself.
                }
                if (roi_reused) db.score = 1.0f, db.valid = true;
                s_face_present = roi_reused || (db.score >= 0.5f);
#else
                if (gfull && gw > 0 && gh > 0) {
                    found = face_detector_lite_run(gfull, gw, gh, &db) && db.valid;
                    // face_detector_lite_run() always returns true with a fallback box whose
                    // score is 0, so "face present" must come from the detector score itself.
                    s_face_present = (db.score >= 0.5f);
                }
#endif
                image_crop_roi_t roi_in;
                int64_t roi_us = 0;
                if (found && roi_reused) {
                    roi_in = prev_roi;
                    roi_us = prev_roi_us;
                    diag_face_valid = false;     // no same-frame detector reading this frame
                } else if (found) {
                    diag_face = db;              // [COMPARE] box tho cua frame nay
                    diag_face_valid = true;
                    td_crop_from_box(&db, gw, gh);
                    portENTER_CRITICAL(&s_crop_mux);
                    roi_in = s_det_crop;
                    roi_us = s_det_crop_us;
                    portEXIT_CRITICAL(&s_crop_mux);
                } else {
                    s_p1_lite_miss++;   // crop giua khung (khong co ROI)
                }
                dec_ok = image_decoder_crop_from_last_gray(found ? &roi_in : NULL,
                                                           input_tensor_ptr,
                                                           TINYDRIVER_INPUT_SCALE,
                                                           TINYDRIVER_INPUT_ZERO_POINT,
                                                           &roi_used);
                if (dec_ok) {
                    // [P4] ROI tao SAU capture -> roi_age AM (bang chung cung frame)
                    roi_ts_us = roi_us;
                    // Center fallback has neither a detector ROI nor its timestamp.
                    roi_ts_valid = found && roi_us > 0;
                    if (found && roi_used.size > 0) p_roi = &roi_used;
                }
            }
        }
#else
        portENTER_CRITICAL(&s_crop_mux);
        has_roi = s_det_crop_valid;
        if (has_roi) local_roi = s_det_crop;
        portEXIT_CRITICAL(&s_crop_mux);
        if (has_roi) {
            p_roi = &local_roi;
        }
        // [D8] Ưu tiên 2: LANDMARK TRACKER chỉ là FALLBACK ngắn hạn (khi detector miss)
        //   — đúng vai trò của nó trên laptop; KHÔNG để nó tự dẫn scale (gây trôi/sụp).
        else if (roi_tracker_is_valid()) {
            face_roi_t fr = roi_tracker_get();
            static image_crop_roi_t roi_in;   // chi dung trong nhanh khong-pipeline
            roi_in.x0 = fr.x0;
            roi_in.y0 = fr.y0;
            roi_in.size = fr.size;
            p_roi = &roi_in;
        }
        bool dec_ok = image_decoder_process_jpeg_roi(
            jpeg_data, jpeg_len,
            input_tensor_ptr,
            TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT,
            p_roi, &roi_used
        );
#endif  // CONFIG_TD_DIAG_SEQUENTIAL

#if CONFIG_TD_USE_ONBOARD_CAMERA
        camera_capture_release();
#else
        wifi_stream_release_frame();
#endif

        if (!dec_ok) {
            adas_controller_invalidate_eye_sample();
            ESP_LOGW(TAG, "Lỗi giải nén frame #%u", (unsigned int)frame_index);
            continue;
        }
        t_decode = esp_timer_get_time() - t0;

        // [v2.9.4] Gui preview TOAN KHUNG camera de chan doan huong/khung hinh tren Laptop
        {
            int gw = 0, gh = 0;
            const uint8_t* gray_full = image_decoder_get_last_gray(&gw, &gh);
            if (gray_full && gw > 0 && gh > 0) {
                #if APP_ENABLE_WEB
        telemetry_sender_preview(gray_full, gw, gh, 128, 96);
#endif
            }
        }

        // [D5] Lite detector chay MOI FRAME -> cap nhat crop canonical cho frame ke tiep
        // [D6] Neu da chay tren Core 0 (TD_LITE_ON_CORE0) thi KHONG chay o day nua.
#if CONFIG_TD_USE_LITE_DETECTOR && !CONFIG_TD_LITE_ON_CORE0
        {
            int gw = 0, gh = 0;
            const uint8_t* gl = image_decoder_get_last_gray(&gw, &gh);
            if (gl) {
                face_box_t lb;
                if (face_detector_lite_run(gl, gw, gh, &lb) && lb.valid) {
                    td_crop_from_box(&lb, gw, gh);
                }
            }
        }
#endif
#endif  // TD_PIPELINE_DECODE_C0

        // [COMPARE] Nguon ROI thuc te cua frame nay + face box fallback.
        // Chi coi la face khi ROI den tu detector/tracker (khong phai crop giua).
        if (!diag_face_valid && p_roi && roi_used.size > 0) {
            diag_face.cx    = (float)roi_used.x0 + roi_used.size * 0.5f;
            diag_face.cy    = (float)roi_used.y0 + roi_used.size * 0.5f;
            diag_face.size  = (float)roi_used.size;
            diag_face.score = 0.0f;
            diag_face.valid = true;
            diag_face_valid = true;
        }
#if CONFIG_TD_DIAG_SEQUENTIAL
        roi_source = diag_face_valid ? WEB_ROI_SRC_DETECTOR_SAME
#if CONFIG_TD_CROP_FIXED
                                     : WEB_ROI_SRC_FIXED;
#else
                                     : WEB_ROI_SRC_CENTER;
#endif
#elif !TD_PIPELINE_DECODE_C0
        if (p_roi == &local_roi)      roi_source = WEB_ROI_SRC_DETECTOR_PREV;
        else if (p_roi)               roi_source = WEB_ROI_SRC_TRACKER;
#if CONFIG_TD_CROP_FIXED
        else                          roi_source = WEB_ROI_SRC_FIXED;
#endif
#endif

        // [v2.9.6 - FIX] Snapshot ảnh input TRƯỚC Invoke():
        // TFLM tái sử dụng vùng arena nên input tensor BỊ GHI ĐÈ trong Invoke.
        // Ảnh debug gửi đi phải lấy bản snapshot này (không đọc sau Invoke).
        static int8_t *s_model_input_snapshot = nullptr;  // PSRAM, allocated on demand
        if (!s_model_input_snapshot) s_model_input_snapshot = (int8_t*)heap_caps_malloc(TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT, MALLOC_CAP_SPIRAM);
    if (s_model_input_snapshot) memcpy(s_model_input_snapshot, input_tensor_ptr,
               (size_t)TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT);

        // 3. Neural Network Inference on Core 1 (TinyDriver-LandmarkNet)
        int64_t t_ai = 0;
        int src_gw = 0, src_gh = 0;
        const uint8_t* src_gray = image_decoder_get_last_gray(&src_gw, &src_gh);
#if BLAZE_CAPTURE_MODE && BLAZE_CAPTURE_RAW
        capture_dump_gray(src_gray, src_gw, src_gh);
#endif
        ai_inference_set_source(src_gray, src_gw, src_gh, roi_used.x0, roi_used.y0, roi_used.size);
        if (!ai_inference_run(landmarks, &t_ai)) {
            adas_controller_invalidate_eye_sample();
            blaze_fusion_set_landmarks(nullptr, false);   // no landmarks -> no landmark ROI
            ESP_LOGE(TAG, "Lỗi suy luận mạng nơ-ron!");
#if TD_PIPELINE_DECODE_C0
            xSemaphoreGive(s_buf_free[dec_slot_used]);   // [P2] khong giu slot khi loi
#endif
            continue;
        }

        // [13/11] Landmark bbox (frame pixels) for the detector/landmark fusion.
        {
            // s_pts98_frame is file-scope (see above) so the snapshot can reuse it
            if (ai_inference_get_last_pts98(s_pts98_frame)) {
                float lx0 = 1e9f, ly0 = 1e9f, lx1 = -1e9f, ly1 = -1e9f;
                for (int i = 0; i < 96; i++) {         // 96,97 = pupils: exclude
                    if (s_pts98_frame[i][0] < lx0) lx0 = s_pts98_frame[i][0];
                    if (s_pts98_frame[i][0] > lx1) lx1 = s_pts98_frame[i][0];
                    if (s_pts98_frame[i][1] < ly0) ly0 = s_pts98_frame[i][1];
                    if (s_pts98_frame[i][1] > ly1) ly1 = s_pts98_frame[i][1];
                }
                float lm_box[4] = {lx0, ly0, lx1, ly1};
                blaze_fusion_set_landmarks(lm_box, (lx1 - lx0) > 16.0f && (ly1 - ly0) > 16.0f);
            } else {
                blaze_fusion_set_landmarks(nullptr, false);
            }
        }

#if !CONFIG_TD_DIAG_SEQUENTIAL
        // 3b. [D7] CẬP NHẬT ROI từ 22 landmark (landmark-driven tracking như laptop:
        //     tâm/cạnh canonical suy từ landmark frame này -> dùng cho crop frame sau).
        if (roi_used.size > 0) {
            roi_tracker_update_from_landmarks(landmarks, roi_used.src_w, roi_used.src_h,
                                              roi_used.x0, roi_used.y0, roi_used.size);
        }
        // [D7] NEO chống trôi từ detector (vai trò như Haar/MediaPipe của laptop):
        //     khi tracker chưa bám, hoặc định kỳ mỗi 6 frame (~1s).
        if (has_roi && (!roi_tracker_is_valid() || (frame_index % 6u) == 0u)) {
            roi_tracker_anchor(local_roi.x0 + local_roi.size / 2,
                               local_roi.y0 + local_roi.size / 2,
                               local_roi.size, local_roi.src_w, local_roi.src_h);
        }
#else
        // [P4] Chan doan: landmark KHONG duoc phep tu chinh ROI (crop da dung box
        // detector tren chinh frame). Khong anchor/de trôi.
#endif

        // 4. Solve 3D Head Pose (Yaw, Pitch, Roll) via Pure C++ POSIT / PnP
        // [v2.6.3] Khi đang ngáp (MAR>=0.5): GIỮ pose frame trước (tránh nhảy trục Y/Z do hàm/miệng biến dạng).
        static head_pose_t s_held_pose = {};
        static bool s_has_held_pose = false;
        float _wm = hypotf(landmarks[12].x - landmarks[13].x, landmarks[12].y - landmarks[13].y);
        if (_wm < 1e-4f) _wm = 1e-4f;
        float _mar_now = (hypotf(landmarks[14].x - landmarks[15].x, landmarks[14].y - landmarks[15].y) +
                          hypotf(landmarks[16].x - landmarks[17].x, landmarks[16].y - landmarks[17].y)) / (2.0f * _wm);
        int64_t t_pnp = 0;
        int64_t t_pnp_start = esp_timer_get_time();
        bool pnp_ok = pnp_solve_head_pose(landmarks, &head_pose) && head_pose.is_valid;
        t_pnp = esp_timer_get_time() - t_pnp_start;

        // [D11] POSE HÌNH HỌC — thay YAW/ROLL của POSIT.
        // ĐO THỰC TẾ: POSIT dùng model mặt gần phẳng (Z chỉ −20..−35 so với XY ±65)
        // nên SUY BIẾN: trả yaw ~0° khi đầu quay 54° và thất bại → fallback roll=0.
        // Hai góc dưới đây BẤT BIẾN TỈ LỆ, ổn định với mọi khung hình:
        //   ROLL = góc của đường nối 2 tâm mắt (P0..P5 vs P6..P11)
        //   YAW  = bất đối xứng ngang của chóp mũi P19 so với trung điểm 2 mắt
        //          (đúng công thức laptop: asin(dx_nose / (0.35*d_eyes)))
        {
            float eA_x = 0, eA_y = 0, eB_x = 0, eB_y = 0;
            for (int i = 0; i < 6; i++)  { eA_x += landmarks[i].x; eA_y += landmarks[i].y; }
            for (int i = 6; i < 12; i++) { eB_x += landmarks[i].x; eB_y += landmarks[i].y; }
            eA_x /= 6.0f; eA_y /= 6.0f; eB_x /= 6.0f; eB_y /= 6.0f;
            // [D20-FIX] P0..P5 nam o NUA TRAI anh (x~0.34), P6..P11 o NUA PHAI (x~0.66).
            // Truoc day tinh dxe = eA - eB => dxe AM (~-0.31) => atan2(dy,dxe) tra
            // ~+/-180 do va NHAY LOAN (do thuc te: std=171 do!) => roll rac tren HUD.
            // Dung vector mat-trai-anh -> mat-phai-anh (eB - eA) de dxe > 0.
            float dxe = eB_x - eA_x, dye = eB_y - eA_y;
            float d_eyes = sqrtf(dxe * dxe + dye * dye);
            if (d_eyes > 1e-4f) {
                float roll_geo = atan2f(dye, dxe) * 57.2957795f;
                float midx = (eA_x + eB_x) * 0.5f;
                float ratio = (landmarks[19].x - midx) / (0.35f * d_eyes);
                if (ratio > 1.0f)  ratio = 1.0f;
                if (ratio < -1.0f) ratio = -1.0f;
                float yaw_geo = asinf(ratio) * 57.2957795f;
                if (fabsf(yaw_geo) > 60.0f) yaw_geo = (yaw_geo > 0) ? 60.0f : -60.0f;
                head_pose.yaw  = yaw_geo;
                head_pose.roll = roll_geo;
            }
        }
        // PITCH: POSIT bất ổn định nên CHỈ cập nhật khi POSIT hợp lệ (trong dải hợp lý),
        // ngược lại GIỮ giá trị trước — tránh pitch rác gây báo động giả.
        {
            static float s_pitch_hold = 0.0f;
            if (pnp_ok && fabsf(head_pose.pitch) < 45.0f) s_pitch_hold = head_pose.pitch;
            head_pose.pitch = s_pitch_hold;
            head_pose.is_valid = true;      // pose hình học luôn dùng được
        }
        if (head_pose.is_valid) {
            // [v2.6.4] Khi ngáp: giữ pitch/roll (chống lệch do cằm tụt) nhưng CẬP NHẬT YAW
            // -> quay đầu lúc ngáp không bị trễ trục.
            if (_mar_now >= 0.50f && s_has_held_pose) {
                head_pose.pitch = s_held_pose.pitch;
                head_pose.roll  = s_held_pose.roll;
            }
            s_held_pose = head_pose;
            s_has_held_pose = true;
        } else if (s_has_held_pose) {
            head_pose = s_held_pose;
        }
        pose_raw = head_pose;   // [COMPARE] pose truoc EMA (cung frame)

        // [D8] LÀM MƯỢT POSE (EMA nhẹ) — PnP rất nhạy với nhiễu landmark nên yaw
        // dao động ±15-20° gây BÁO ĐỘNG DISTRACTION GIẢ. Lọc nhẹ (α=0.35) đủ êm
        // mà không trễ đáng kể (báo động cần giữ 3.0s).
        {
            static head_pose_t s_sm_pose = {};
            static bool s_sm_pose_ready = false;
            if (head_pose.is_valid) {
                if (!s_sm_pose_ready) {
                    s_sm_pose = head_pose;
                    s_sm_pose_ready = true;
                } else {
                    const float ap = 0.35f;
                    s_sm_pose.yaw   += ap * (head_pose.yaw   - s_sm_pose.yaw);
                    s_sm_pose.pitch += ap * (head_pose.pitch - s_sm_pose.pitch);
                    s_sm_pose.roll  += ap * (head_pose.roll  - s_sm_pose.roll);
                    head_pose.yaw   = s_sm_pose.yaw;
                    head_pose.pitch = s_sm_pose.pitch;
                    head_pose.roll  = s_sm_pose.roll;
                }
            }
        }

        pose_smoothed = head_pose;   // [COMPARE] pose sau EMA (input ADAS)


        // [13/11] Không có mặt trong khung -> PIPNet chạy trên ROI dự phòng và trả mí mắt rác
        // (EAR 0,03) -> nếu không khoá, FSM sẽ báo nhắm/microsleep giả.
        adas_controller_set_eye_enabled(s_face_present);
        // 5. Update ADAS Finite State Machine (EAR, MAR, Microsleep, Fatigue, Distraction)
        adas_controller_update_capture(landmarks, &head_pose, current_fps, cap_us, &adas_metrics);
        s_pose_yaw_prev2   = s_pose_yaw_prev;
        s_pose_pitch_prev2 = s_pose_pitch_prev;
        s_pose_yaw_prev    = head_pose.yaw;
        s_pose_pitch_prev  = head_pose.pitch;
#if BLAZE_CAPTURE_MODE
        capture_dump_overlay(src_gray, src_gw, src_gh, &roi_used, s_pts98_frame,
                             adas_metrics.ear, adas_metrics.ear_threshold, adas_metrics.eye.closed ? 1.0f : 0.0f);
#endif

        // [D23] HIỂN THỊ pose ĐÃ TRỪ BIAS: nếu không, khi mặt hướng thẳng thì
        // PITCH/YAW vẫn hiện ~+10..14° (bias POSIT) -> trục X/Y/Z nghiêng sai trên web.
        // (ADAS bên trong vẫn dùng bias riêng để báo mất tập trung — không đổi hành vi.)
        adas_metrics.pose.pitch -= adas_controller_get_pitch_bias();
        adas_metrics.pose.yaw   -= adas_controller_get_yaw_bias();
#if TD_OBS_ENABLE
        // [14/11] Publish AFTER the bias correction so /status shows the same yaw the demo sees.
        obs_server_publish(src_gray, src_gw, src_gh, &roi_used, s_pts98_frame, &adas_metrics,
                           diag_face.score, face_detector_lite_last_ms(), (uint32_t)frame_index);
#endif


        // 6. Dispatch Telemetry JSON to Laptop Host & Actuate Onboard Buzzer/LED
        telemetry_sender_set_timing((float)t_decode / 1000.0f, (float)t_ai / 1000.0f,
                                    (float)(esp_timer_get_time() - t0) / 1000.0f);
        // crop mode: 0 = giữa khung, 1 = cố định, 2 = ROI/bám mặt
        int crop_mode = 0;
        if (p_roi) {
            crop_mode = 2;
        }
#if CONFIG_TD_CROP_FIXED
        else {
            crop_mode = 1;
        }
#endif
        telemetry_sender_set_roi(crop_mode, roi_used.x0, roi_used.y0, roi_used.size);

        // [P1] Do tuoi anh/ROI tai thoi diem publish + thong ke p50/p95 (khong doi hanh vi)
        int64_t t_pub = esp_timer_get_time();
        float img_age_ms = (cap_us > 0) ? (float)(t_pub - cap_us) / 1000.0f : -1.0f;
        float roi_age_ms = roi_ts_valid ? (float)(cap_us - roi_ts_us) / 1000.0f : -1.0f;
        float core1_ms   = (t_slot0 > 0) ? (float)(t_pub - t_slot0) / 1000.0f : -1.0f;
        float loop_ms    = (float)(t_pub - t0) / 1000.0f;
        if (img_age_ms >= 0.0f) td_stat_add(&s_st_img_age, img_age_ms);
        // [P4] Ghi ca roi_age AM (ROI tao sau capture = cung frame) de thay ro
        // trong p50/p95; -1 chi khi khong co ROI hop le.
        if (roi_ts_valid) td_stat_add(&s_st_roi_age, roi_age_ms);
        td_stat_add(&s_st_dec, (float)t_decode / 1000.0f);
        td_stat_add(&s_st_ai, (float)t_ai / 1000.0f);
        if (core1_ms >= 0.0f) td_stat_add(&s_st_core1, core1_ms);
        td_stat_add(&s_st_total, loop_ms);
        if (s_p1_last_cam_seq != 0 && cam_seq > s_p1_last_cam_seq + 1) {
            s_p1_cam_gap += cam_seq - s_p1_last_cam_seq - 1;
        }
        if (cam_seq != 0) s_p1_last_cam_seq = cam_seq;

        telemetry_sender_set_diag(frame_index, img_age_ms, roi_age_ms, core1_ms);

        // [COMPARE] Gom metadata chan doan cua CHINH frame nay de publish atomic
        // cung JPEG/landmark. Tensor chi co o che do P4 (sequential same-frame).
        web_diag_meta_t diag = {};
        diag.tensor_w = TINYDRIVER_INPUT_WIDTH;
        diag.tensor_h = TINYDRIVER_INPUT_HEIGHT;
        diag.input_scale = TINYDRIVER_INPUT_SCALE;
        diag.input_zero_point = TINYDRIVER_INPUT_ZERO_POINT;
        diag.roi_source = roi_source;
        diag.face_valid = diag_face_valid;
        diag.face_score = diag_face.score;
        diag.face_cx    = diag_face.cx;
        diag.face_cy    = diag_face.cy;
        diag.face_size  = diag_face.size;
        diag.pose_raw      = pose_raw;
        diag.pose_smoothed = pose_smoothed;
        diag.pose_published = adas_metrics.pose;
        diag.bias_yaw   = adas_controller_get_yaw_bias();
        diag.bias_pitch = adas_controller_get_pitch_bias();
        diag.cam_seq    = cam_seq;
        diag.cap_us     = cap_us;
        diag.publish_us = t_pub;
#if CONFIG_TD_DIAG_SEQUENTIAL
        diag.tensor = s_model_input_snapshot;
#endif

        // [D21] Phat khung hinh len Web Dashboard CUNG LUC voi du lieu AI (22 diem/ROI/MAR)
        // => video va lop phu KHOP NHAU (het cam giac "diem sai cho / lag" khi di chuyen).
        const uint8_t* pub_jpeg = NULL;
        size_t pub_len = 0;
#if CONFIG_TD_DIAG_SEQUENTIAL
        // [P4] staging rieng, copy NGAY sau acquire -> luon la frame nay.
        (void)dec_slot_used;
        if (s_diag_jpeg && s_diag_len > 0) {
            pub_jpeg = s_diag_jpeg;
            pub_len  = s_diag_len;
        }
#else
        if (s_stage_jpeg[dec_slot_used] && s_stage_len[dec_slot_used] > 0) {
            // [P1] staging con dung frame AI vua xu ly? (voi P2 phai LUON khop)
            if (s_stage_cam_seq[dec_slot_used] != cam_seq) s_p1_stage_mismatch++;
            pub_jpeg = s_stage_jpeg[dec_slot_used];
            pub_len  = s_stage_len[dec_slot_used];
        }
#endif
        // [P3] Ảnh + telemetry + landmark của CÙNG frame trong một lần publish
        // (nguồn cho /debug/snapshot; /stream và /status vẫn đọc buffer này).
        web_server_publish_result(pub_jpeg, pub_len, &adas_metrics, landmarks,
                                  crop_mode, roi_used.x0, roi_used.y0, roi_used.size,
                                  roi_used.src_w, roi_used.src_h,
                                  (float)t_decode / 1000.0f,
                                  (float)t_ai / 1000.0f,
                                  (float)t_pnp / 1000.0f,
                                  loop_ms,
                                  frame_index, img_age_ms, roi_age_ms, core1_ms,
                                  &diag);
#if TD_PIPELINE_DECODE_C0
        // [P2] Da doc xong staging JPEG (web_server_publish_result copy ra buffer riêng)
        // -> tra slot cho Core 0 giai ma frame tiep theo.
        xSemaphoreGive(s_buf_free[dec_slot_used]);
#endif

        #if APP_ENABLE_WEB
        telemetry_sender_dispatch(&adas_metrics, landmarks);
#endif

        // Kiểm tra xem người dùng có bấm nút "Hiệu chuẩn lại" từ Web UI không
        if (web_server_is_recalibrate_requested()) {
            ESP_LOGI(TAG, "🔄 Kích hoạt lại quá trình tự hiệu chuẩn ADAS theo yêu cầu Web UI!");
            adas_controller_init();
        }

        // 6b. [v2.7.0] Gửi ảnh 96x96 (đúng cái ESP32 nhìn) để laptop HIỂN THỊ (display-only)
        #if APP_ENABLE_WEB
        telemetry_sender_image(s_model_input_snapshot,
                               TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT,
                               TINYDRIVER_INPUT_WIDTH, TINYDRIVER_INPUT_HEIGHT);
#endif

        processed_frames++;
        int64_t t_total = esp_timer_get_time() - t0;

        // [D4b] Publish ảnh xám cho detector task (Core 0) — chỉ khi task đang rảnh
        //   [D16] Voi pipeline decode-Core0 thi Lite da chay luon trong task decode -> khong can publish
#if TD_DET_C0_ENABLED && !TD_PIPELINE_DECODE_C0
        if (s_det_task && !s_pub_busy) {
            int gw = 0, gh = 0;
            const uint8_t* gdet = image_decoder_get_last_gray(&gw, &gh);
            if (gdet && gw > 0 && gh > 0 && (size_t)gw * gh <= (size_t)320 * 240) {
                memcpy(s_pub_gray, gdet, (size_t)gw * gh);
                s_pub_w = gw; s_pub_h = gh;
                s_pub_busy = true;
                xTaskNotifyGive(s_det_task);
            }
        }
#endif

        // Calculate rolling FPS every 5 frames (fast enough to observe realtime reaction)
        if (processed_frames % 5 == 0) {
#if TD_ROI_REUSE_STABLE
            ESP_LOGI("ROIHYB", "reuse %u/%u frame (%.0f%%) — chi dung ROI khung truoc khi dau dung yen",
                     (unsigned)s_roi_reused, (unsigned)processed_frames,
                     100.0f * (float)s_roi_reused / (float)(processed_frames ? processed_frames : 1));
#endif
            int64_t now = esp_timer_get_time();
            float elapsed_sec = (float)(now - fps_timer_start) / 1e6f;
            current_fps = (elapsed_sec > 0.0f) ? (5.0f / elapsed_sec) : 0.0f;
            fps_timer_start = now;

            ESP_LOGI(TAG, "[Frame #%4u] FPS: %4.1f | Dec: %3.1fms | AI: %3.1fms | PnP: %3.1fms | Lite: %4.1fms | Total: %3.1fms | Img: %4.0fms | ROIage: %4.0fms | cam#%u | ROI: %d,%d,%d | EAR: %.2f (thr %.2f base %.2f) | MAR: %.2f | Yaw: %+5.1f° (bias %+.1f) | eye: %s cl=%.0fms | Status: %s %s",
                     (unsigned int)processed_frames,
                     current_fps,
                     (float)t_decode / 1000.0f,
                     (float)t_ai / 1000.0f,
                     (float)t_pnp / 1000.0f,
                     (float)face_detector_lite_last_ms(),
                     (float)t_total / 1000.0f,
                     img_age_ms, roi_age_ms, (unsigned)cam_seq,
                     roi_used.x0, roi_used.y0, roi_used.size,
                     adas_metrics.ear,
                     adas_metrics.ear_threshold,
                     adas_controller_get_ear_baseline(),
                     adas_metrics.mar,
                     head_pose.is_valid ? head_pose.yaw : 0.0f,
                     adas_controller_get_yaw_bias(),
                     adas_metrics.eye.reset_reason ? adas_metrics.eye.reset_reason : "-",
                     adas_metrics.eye.closed_ms,
                     adas_metrics.status_str,
                     adas_metrics.is_alarm_active ? "🚨 [CÒI HÚ]" : "");

            // [P1] Trung vi/p95 (cua so 64 mau gan nhat) + bo dem chan doan
            ESP_LOGI(TAG, "[P1] img_age p50=%.0f p95=%.0f ms | roi_age p50=%.0f p95=%.0f ms | "
                          "dec p95=%.0f | ai p95=%.0f | core1 p50=%.0f p95=%.0f | loop p95=%.0f ms | "
                          "stage_mismatch=%u stage_skip=%u cam_gap=%u cam_fail=%u dec_fail=%u lite_miss=%u",
                     td_stat_pct(&s_st_img_age, 50.0f), td_stat_pct(&s_st_img_age, 95.0f),
                     td_stat_pct(&s_st_roi_age, 50.0f), td_stat_pct(&s_st_roi_age, 95.0f),
                     td_stat_pct(&s_st_dec, 95.0f), td_stat_pct(&s_st_ai, 95.0f),
                     td_stat_pct(&s_st_core1, 50.0f), td_stat_pct(&s_st_core1, 95.0f),
                     td_stat_pct(&s_st_total, 95.0f),
                     (unsigned)s_p1_stage_mismatch, (unsigned)s_p1_stage_skip,
                     (unsigned)s_p1_cam_gap, (unsigned)s_p1_cam_fail,
                     (unsigned)s_p1_dec_fail, (unsigned)s_p1_lite_miss);
        }
    }
}

#define PIPNET_LOADER_TEST 0
#define PIPNET_STILL_TEST 0
#define MSR_ONLY_TEST 0
#define BLAZE_ONLY_TEST 0   // 1 = measure the detector on embedded stills, 0 = live pipeline
#define BLAZE_E2E_TEST 0    // 1 = detector->ROI->PIPNet->EAR on the embedded stills
#define MULTI_ID_TEST 0     // 1 = run the chain on 8 other drivers (YawDD stills)
#define BLAZE_E2E_CLOSED_ONLY 0   // 0 = sweep both stills, 1 = only closed (2nd boot)
#define BLAZE_E2E_FUSION_ONLY 0   // 1 = only measure IoU(box detector, bbox 98 diem)

#include "still_pairs.h"
extern "C" void main_still_test(void) {
    // runs both still frames: open then closed

    vTaskDelay(pdMS_TO_TICKS(300));
    if (!face_espdl_init()) { ESP_LOGE("STILL", "model load failed"); return; }
    static float pts[98][2];
    int64_t us = 0;
    bool ok = face_espdl_landmarks(kStillOpen, 320, 240, 113, 69, 141, pts, &us);
    ESP_LOGI("STILL", "result %s in %lld ms", ok ? "OK" : "FAILED", (long long)(us / 1000));
    if (ok) {
        for (int i = 0; i < 98; i++) ESP_LOGI("STILLOPEN", "%d %.2f %.2f", i, pts[i][0], pts[i][1]);
        int64_t us2 = 0;
        if (face_espdl_landmarks(kStillClosed, 320, 240, 117, 77, 132, pts, &us2)) {
            for (int i = 0; i < 98; i++) ESP_LOGI("STILLCLOSED", "%d %.2f %.2f", i, pts[i][0], pts[i][1]);
        }
    }
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
#include "msr_test_image.h"
#include "face_det_espdl.h"
#include "blaze_tracker.h"

// Measure the INT8 BlazeFace detector on the embedded still frames. These are the
// same frames the offline parity script used, so the on-board box can be compared
// directly with the laptop's float/int8 result.
extern "C" void main_blaze_test(void) {
    vTaskDelay(pdMS_TO_TICKS(400));
    ESP_LOGI("BLAZETEST", "psram=%uKB sram=%uKB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    bool ok = blaze_init();
    ESP_LOGI("BLAZETEST", "init=%d arena=%d B psram=%uKB sram=%uKB", (int)ok,
             blaze_arena_used(),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    if (!ok) { while (true) vTaskDelay(pdMS_TO_TICKS(5000)); }

    // 3 passes over each frame: repeatability + ms/frame.
    const uint8_t* frames[2] = {kStillOpen, kStillClosed};
    const char* names[2] = {"still_open", "still_closed"};
    for (int rep = 0; rep < 3; rep++) {
        for (int f = 0; f < 2; f++) {
            float box[4] = {0, 0, 0, 0}, score = -1;
            bool det = blaze_detect(frames[f], 320, 240, box, &score);
            ESP_LOGI("BLAZETEST", "%s rep=%d det=%d cand=%d score=%.3f box=%.1f,%.1f,%.1f,%.1f ms=%.0f",
                     names[f], rep, (int)det, blaze_last_cand(), (double)score,
                     (double)box[0], (double)box[1], (double)box[2], (double)box[3],
                     (double)blaze_last_ms());
        }
    }
    // Negative control: flat mid-grey frame must not produce a face.
    static uint8_t s_grey[320 * 240];
    memset(s_grey, 128, sizeof(s_grey));
    float box[4] = {0, 0, 0, 0}, score = -1;
    bool det = blaze_detect(s_grey, 320, 240, box, &score);
    ESP_LOGI("BLAZETEST", "flat_grey det=%d cand=%d score=%.3f ms=%.0f", (int)det,
             blaze_last_cand(), (double)score, (double)blaze_last_ms());
    // Parity diagnostics against the host reference run.
    blaze_debug_dump(kStillOpen, 320, 240);
    ESP_LOGI("BLAZETEST", "done");
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}

// End-to-end on the embedded stills, using the SAME code path as the live camera:
// BlazeFace INT8 -> face_detector_lite_run() -> td_crop_from_box() -> PIPNet-98 -> EAR.
// No person needed: proves the ROI chain and the open/closed EAR split on the board.
static void vTaskE2E(void* arg) {
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(400));
    // PIPNet blocks the calling task for ~7.6 s per inference; in a test loop that
    // starves IDLE0 and the task watchdog resets the board. Production runs inference
    // in the AI task, which is unaffected. Test-only: drop the task WDT.
    esp_task_wdt_deinit();
    ESP_LOGI("E2E", "psram=%uKB sram=%uKB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    bool bok = face_detector_lite_init();   // sets the module state the live path uses
    bool aok = ai_inference_init();
    ESP_LOGI("E2E", "det_lite_init=%d ai_init=%d arena_used=%d B psram=%uKB", (int)bok, (int)aok,
             blaze_arena_used(), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10));
    bool fok = face_espdl_init();
    ESP_LOGI("E2E", "pipnet_init=%d psram=%uKB sram=%uKB", (int)fok,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    if (!bok || !fok || !aok) { while (true) vTaskDelay(pdMS_TO_TICKS(5000)); }

    const uint8_t* frames[2] = {kStillOpen, kStillClosed};
    const char* names[2] = {"open", "closed"};
    // ROI sweep: the crop scale changes the EAR scale, so pick it with numbers.
    // k = 1.08 is the current production policy; 1.5 is the "nới 1.5x" in the design.
    const float ks[3] = {1.08f, 1.25f, 1.50f};
    const int fixed_roi[2][3] = {{113, 69, 141}, {117, 77, 132}};   // still_rois.json (12/11)
#if BLAZE_E2E_CLOSED_ONLY
    // A second boot only covers the remaining closed-eye policies: changing the crop
    // size repeatedly in one boot exhausts the ESP-DL pool (board reset after ~6 runs).
    const int f_begin = 1, f_end = 2;
    const int k_begin = 2, k_end = 4;
#else
    const int f_begin = 0, f_end = 2;
    const int k_begin = 0, k_end = 4;
#endif
    if (BLAZE_E2E_FUSION_ONLY) {
        // Bo so ROI: chi kiem fusion IoU giua box detector va bbox 98 diem.
        for (int f = 0; f < 2; f++) {
            blaze_reset();
            face_box_t fb;
            memset(&fb, 0, sizeof(fb));
            if (!face_detector_lite_run(frames[f], 320, 240, &fb) || !fb.valid) continue;
            int cs = (int)lroundf(1.08f * fb.size);
            int rx = (int)lroundf(fb.cx + 0.040f * fb.size) - cs / 2;
            int ry = (int)lroundf(fb.cy - 0.028f * fb.size) - cs / 2;
            if (rx < 0) rx = 0;
            if (ry < 0) ry = 0;
            if (rx + cs > 320) rx = 320 - cs;
            if (ry + cs > 240) ry = 240 - cs;
            static float pts[98][2];
            int64_t us = 0;
            if (!face_espdl_landmarks(frames[f], 320, 240, rx, ry, cs, pts, &us)) continue;
            float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
            for (int i = 0; i < 96; i++) {   // 96,97 = pupils: exclude
                if (pts[i][0] < x0) x0 = pts[i][0];
                if (pts[i][0] > x1) x1 = pts[i][0];
                if (pts[i][1] < y0) y0 = pts[i][1];
                if (pts[i][1] > y1) y1 = pts[i][1];
            }
            float lmbox[4] = {x0, y0, x1, y1};
            float detbox[4] = {fb.cx - fb.size * 0.5f, fb.cy - fb.size * 0.5f,
                               fb.cx + fb.size * 0.5f, fb.cy + fb.size * 0.5f};
            float iou_dl = blaze_box_iou(detbox, lmbox);
            ESP_LOGI("E2E", "%s FUSION det=%.1f,%.1f,%.1f,%.1f lm=%.1f,%.1f,%.1f,%.1f iou=%.3f "
                            "det_w=%.1f lm_w=%.1f ratio=%.2f lm_from=%d,%d,%d ai=%lldms",
                     names[f], (double)detbox[0], (double)detbox[1], (double)detbox[2],
                     (double)detbox[3], (double)x0, (double)y0, (double)x1, (double)y1,
                     (double)iou_dl, (double)fb.size, (double)(x1 - x0),
                     (double)((x1 - x0) / fb.size), rx, ry, cs, (long long)(us / 1000));
            // [13/11] Validate the WIRED fusion path with the same still frame.
            blaze_fusion_set_landmarks(lmbox, true);
            face_box_t fb2;
            memset(&fb2, 0, sizeof(fb2));
            bool det2 = face_detector_lite_run(frames[f], 320, 240, &fb2) && fb2.valid;
            ESP_LOGI("E2E", "%s FUSION-WIRED det=%d score=%.3f side=%.1f c=%.1f,%.1f",
                     names[f], (int)det2, (double)fb2.score, (double)fb2.size,
                     (double)fb2.cx, (double)fb2.cy);
            vTaskDelay(pdMS_TO_TICKS(300));
        }
        ESP_LOGI("E2E", "fusion done");
        while (true) vTaskDelay(pdMS_TO_TICKS(5000));
    }
    for (int f = f_begin; f < f_end; f++) {
        blaze_reset();   // independent per frame: no tracker carry-over
        face_box_t lb;
        memset(&lb, 0, sizeof(lb));
        bool det = face_detector_lite_run(frames[f], 320, 240, &lb) && lb.valid;
        if (!det) { ESP_LOGE("E2E", "%s detector FAILED", names[f]); continue; }
        ESP_LOGI("E2E", "%s det score=%.3f box=%.1f,%.1f side=%.1f", names[f], (double)lb.score,
                 (double)(lb.cx - lb.size * 0.5f), (double)(lb.cy - lb.size * 0.5f),
                 (double)lb.size);
        for (int k = k_begin; k < k_end; k++) {
            int rx, ry, rs;
            const char* tag;
            if (k < 3) {
                int cs = (int)lroundf(ks[k] * lb.size);
                if (cs < 64) cs = 64;
                if (cs > 240) cs = 240;
                rx = (int)lroundf(lb.cx + 0.040f * lb.size) - cs / 2;
                ry = (int)lroundf(lb.cy - 0.028f * lb.size) - cs / 2;
                if (rx < 0) rx = 0;
                if (ry < 0) ry = 0;
                if (rx + cs > 320) rx = 320 - cs;
                if (ry + cs > 240) ry = 240 - cs;
                rs = cs;
                tag = (k == 0) ? "k1.08" : (k == 1) ? "k1.25" : "k1.50";
            } else {
                rx = fixed_roi[f][0]; ry = fixed_roi[f][1]; rs = fixed_roi[f][2];
                tag = "fixed";
            }
            static float pts[98][2];
            int64_t us = 0;
            bool ok = face_espdl_landmarks(frames[f], 320, 240, rx, ry, rs, pts, &us);
            float ear = 0.0f, el = 0.0f, er = 0.0f;
            if (ok) {
                auto d = [](float ax, float ay, float bx, float by) {
                    return sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
                };
                float D = d(pts[60][0], pts[60][1], pts[68][0], pts[68][1]);
                float vl = (d(pts[61][0], pts[61][1], pts[66][0], pts[66][1]) +
                            d(pts[62][0], pts[62][1], pts[65][0], pts[65][1])) * 0.5f;
                float vr = (d(pts[69][0], pts[69][1], pts[74][0], pts[74][1]) +
                            d(pts[70][0], pts[70][1], pts[73][0], pts[73][1])) * 0.5f;
                el = D > 1e-4f ? vl / D : 0.0f;
                er = D > 1e-4f ? vr / D : 0.0f;
                ear = (el + er) * 0.5f;
            }
            ESP_LOGI("E2E", "%s %s roi=%d,%d,%d ok=%d EAR=%.3f L=%.3f R=%.3f D=%.1f ai=%lldms psram=%uKB",
                     names[f], tag, rx, ry, rs, (int)ok, (double)ear, (double)el, (double)er,
                     (double)hypotf(pts[68][0] - pts[60][0], pts[68][1] - pts[60][1]),
                     (long long)(us / 1000),
                     (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10));
        }
    }
    // Cross-check: the production path (ai_inference_run -> EARDIRECT) must agree with
    // the formula above for the current policy.
    blaze_reset();
    face_box_t lb;
    memset(&lb, 0, sizeof(lb));
    if (face_detector_lite_run(kStillOpen, 320, 240, &lb) && lb.valid) {
        td_crop_from_box(&lb, 320, 240);
        image_crop_roi_t roi;
        memset(&roi, 0, sizeof(roi));
        portENTER_CRITICAL(&s_crop_mux);
        roi = s_det_crop;
        portEXIT_CRITICAL(&s_crop_mux);
        ai_inference_set_source(kStillOpen, 320, 240, roi.x0, roi.y0, roi.size);
        point2d_t lm[22];
        int64_t us = 0;
        bool ok = ai_inference_run(lm, &us);
        ESP_LOGI("E2E", "crosscheck path_ear roi=%d,%d,%d ok=%d ai=%lldms", roi.x0, roi.y0,
                 roi.size, (int)ok, (long long)(us / 1000));
    }
    ESP_LOGI("E2E", "done");
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}

// PIPNet needs the same 16 KB stack the production AI task has; the main task
// (3.5 KB default) overflows on the second inference.
extern "C" void main_blaze_e2e_test(void) {
    xTaskCreatePinnedToCore(vTaskE2E, "Task_E2E", 16384, NULL, 5, NULL, 1);
}

// [14/11] Multi-identity test: run the WHOLE chain (BlazeFace INT8 -> ROI -> PIPNet-98 -> EAR)
// on still frames of OTHER drivers (YawDD dash-cam, see tools/make_multiid_header.py), plus the
// board's own two stills as a control. Proves (or disproves) that the pipeline works for faces
// other than the owner's — no live subject needed.
#include "multiid_images.h"
static void vTaskMultiId(void* arg) {
    (void)arg;
    esp_task_wdt_deinit();
    vTaskDelay(pdMS_TO_TICKS(300));
    bool bok = face_detector_lite_init();
    bool aok = ai_inference_init();
    bool fok = face_espdl_init();
    ESP_LOGI("MULTIID", "init det=%d ai=%d pip=%d psram=%uKB sram=%uKB", (int)bok, (int)aok,
             (int)fok, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    if (!bok || !fok || !aok) { while (true) vTaskDelay(pdMS_TO_TICKS(5000)); }

    for (int i = 0; i < MULTIID_COUNT; i++) {
        blaze_reset();
        face_box_t b;
        memset(&b, 0, sizeof(b));
        bool det = face_detector_lite_run(kMultiId[i], 320, 240, &b) && b.valid;
        if (!det) {
            ESP_LOGI("MULTIID", "%d [%s] detect FAILED", i, kMultiIdName[i]);
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        td_crop_from_box(&b, 320, 240);
        image_crop_roi_t roi;
        memset(&roi, 0, sizeof(roi));
        portENTER_CRITICAL(&s_crop_mux);
        roi = s_det_crop;
        portEXIT_CRITICAL(&s_crop_mux);
        static float mpts[98][2];
        int64_t us = 0;
        bool ok2 = face_espdl_landmarks(kMultiId[i], 320, 240, roi.x0, roi.y0, roi.size, mpts, &us);
        float ear = 0.0f;
        if (ok2) {
            auto dm = [](float ax, float ay, float bx, float by) {
                return sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
            };
            float D = dm(mpts[60][0], mpts[60][1], mpts[68][0], mpts[68][1]);
            float vl = (dm(mpts[61][0], mpts[61][1], mpts[66][0], mpts[66][1]) +
                        dm(mpts[62][0], mpts[62][1], mpts[65][0], mpts[65][1])) * 0.5f;
            float vr = (dm(mpts[69][0], mpts[69][1], mpts[74][0], mpts[74][1]) +
                        dm(mpts[70][0], mpts[70][1], mpts[73][0], mpts[73][1])) * 0.5f;
            ear = D > 1e-4f ? ((vl / D) + (vr / D)) * 0.5f : 0.0f;
        }
        if (ok2) {
            for (int k = 0; k < 98; k++)
                ESP_LOGI("MULTIIDPTS", "%d %d %.2f %.2f", i, k, (double)mpts[k][0], (double)mpts[k][1]);
        }
#if BLAZE_CAPTURE_MODE
        capture_dump_overlay(kMultiId[i], 320, 240, &roi, mpts, ear, 0.15f,
                             b.score >= 0.5f ? 1.0f : 0.0f);
#endif
        ESP_LOGI("MULTIID", "%d [%s] score=%.3f side=%.1f roi=%d,%d,%d ear=%.3f ai=%lldms",
                 i, kMultiIdName[i], (double)b.score, (double)b.size, roi.x0, roi.y0, roi.size,
                 (double)ear, (long long)(us / 1000));
        vTaskDelay(pdMS_TO_TICKS(200));
    }
    ESP_LOGI("MULTIID", "done");
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}

extern "C" void main_multiid_test(void) {
    xTaskCreatePinnedToCore(vTaskMultiId, "Task_MultiID", 16384, NULL, 5, NULL, 1);
}

extern "C" void main_msr_test(void) {
    vTaskDelay(pdMS_TO_TICKS(400));
    ESP_LOGI("MSRTEST", "init psram=%uKB sram=%uKB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    bool ok_init = face_det_espdl_init();
    ESP_LOGI("MSRTEST", "init=%d psram=%uKB sram=%uKB", (int)ok_init,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
    float box[4] = {0, 0, 0, 0}, score = -1;
    bool ok = face_det_espdl_run(kMsrRgb, 320, 240, box, &score);
    ESP_LOGI("MSRTEST", "run=%d score=%.3f box=%.1f,%.1f,%.1f,%.1f", (int)ok, score,
             box[0], box[1], box[2], box[3]);
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}

extern "C" void main_still_test(void);

extern "C" void app_main(void) {
    if (MULTI_ID_TEST) {
        main_multiid_test();
        return;
    }
    if (BLAZE_E2E_TEST) {
        main_blaze_e2e_test();
        return;
    }
    if (BLAZE_ONLY_TEST) {
        main_blaze_test();
        return;
    }
    if (MSR_ONLY_TEST) {
        main_msr_test();
        return;
    }
    if (PIPNET_STILL_TEST) {
        main_still_test();
        return;
    }
    if (PIPNET_LOADER_TEST) {
        // Standalone loader test: no camera, no Wi-Fi, no web. Proves the 8-part chain.
        vTaskDelay(pdMS_TO_TICKS(300));
        uint8_t *gray = (uint8_t *)heap_caps_malloc(320 * 240, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!gray) { ESP_LOGE("LOADTEST", "gray alloc failed"); return; }
        for (int y = 0; y < 240; y++)
            for (int x = 0; x < 320; x++)
                gray[y * 320 + x] = (uint8_t)((x + y) & 0xFF);

        ESP_LOGI("LOADTEST", "free PSRAM %u KB / SRAM %u KB",
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
        if (!face_espdl_init()) { ESP_LOGE("LOADTEST", "init FAILED"); return; }
        float pts[98][2];
        int64_t us = 0;
        bool ok = face_espdl_landmarks(gray, 320, 240, 70, 40, 180, pts, &us);
        ESP_LOGI("LOADTEST", "chain %s in %lld ms | free PSRAM %u KB / SRAM %u KB",
                 ok ? "OK" : "FAILED", (long long)(us / 1000),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >> 10),
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
        vTaskDelay(pdMS_TO_TICKS(2000));
        ok = face_espdl_landmarks(gray, 320, 240, 70, 40, 180, pts, &us);
        ESP_LOGI("LOADTEST", "2nd frame %s in %lld ms", ok ? "OK" : "FAILED", (long long)(us / 1000));
        while (true) vTaskDelay(pdMS_TO_TICKS(5000));
    }
    ESP_LOGI(TAG, "=================================================================");
    ESP_LOGI(TAG, "🚘 PROJECT 13: DRIVER DROWSINESS & DISTRACTION DETECTION");
    ESP_LOGI(TAG, "⚡ 100%% EDGE AI ON ESP32-S3 N16R8 DEVKIT (16MB Flash, 8MB PSRAM)");
    ESP_LOGI(TAG, "=================================================================");

    // 1. Initialize NVS Flash
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Hardware Resource & PSRAM Verification
    size_t psram_size = esp_psram_get_size();
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_sram  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "Bộ nhớ Octal PSRAM : %d MB (Còn trống: %d KB)", (int)(psram_size / (1024 * 1024)), (int)(free_psram / 1024));
    ESP_LOGI(TAG, "Bộ nhớ Internal SRAM: %d KB", (int)(free_sram / 1024));

    if (psram_size == 0) {
        ESP_LOGE(TAG, "CẢNH BÁO NGUY HIỂM: Không phát hiện PSRAM! Dự án yêu cầu ESP32-S3 N16R8!");
    }

    // 3. Initialize Submodules
#if APP_ENABLE_WEB
    if (!wifi_stream_init()) {
        ESP_LOGE(TAG, "Khởi tạo Wi-Fi Stream thất bại!");
        return;
    }
#else
    ESP_LOGW(TAG, "PIPNet mode: Wi-Fi + web TẮT (chỉ suy luận + log)");
#endif

#if TD_OBS_ENABLE
    obs_server_start();
#endif

#if CONFIG_TD_USE_ONBOARD_CAMERA
    if (!camera_capture_init()) {
        ESP_LOGE(TAG, "Khởi tạo Camera OV5640 thất bại!");
        return;
    }
#endif

    if (!image_decoder_init()) {
        ESP_LOGE(TAG, "Khởi tạo Image Decoder thất bại!");
        return;
    }
    // [v2.9.5 - M1] Crop cố định (deterministic, không vòng phản hồi)
    image_decoder_set_fixed_crop(CONFIG_TD_CROP_FIXED, CONFIG_TD_CROP_CX_PCT,
                                 CONFIG_TD_CROP_CY_PCT, CONFIG_TD_CROP_SIZE_PCT);

    pnp_solver_init();
    adas_controller_init();
    roi_tracker_reset();

    if (!ai_inference_init()) {
        ESP_LOGE(TAG, "Khởi tạo AI Inference thất bại!");
        return;
    }

    // Load the tiny face detector FIRST (0.18 MB) while PSRAM is still unfragmented,
    // then PIPNet (activations only). This ordering is what makes both fit on N16R8.
    // 1) BlazeFace (184 KB, ESP-DL) first: tiny, keeps PSRAM un-fragmented.
    // 2) PIPNet-98 (12 MB, weights in flash) after.
    face_detector_lite_init();
    face_espdl_init();

#if APP_ENABLE_WEB
    if (!telemetry_sender_init()) {
        ESP_LOGE(TAG, "Khởi tạo Telemetry Sender thất bại!");
        return;
    }
#else
    ESP_LOGW(TAG, "PIPNet mode: telemetry TẮT (không mạng, chỉ log serial)");
#endif

    // Khởi tạo HTTP Web Server nhúng (Port 80) — tắt ở chế độ PIPNet để dành CPU/RAM
#if APP_ENABLE_WEB
    if (!web_server_init()) {
        ESP_LOGW(TAG, "Khởi tạo Web Server thất bại (kiểm tra tài nguyên mạng)!");
    }
#else
    ESP_LOGW(TAG, "PIPNet mode: dashboard/MJPEG TẮT (chỉ cảnh báo + log)");
#endif

    ESP_LOGI(TAG, "✅ Tất cả các module phần cứng và phần mềm đã sẵn sàng!");

    // [D2] Detect 1 lần để xác định vùng crop khuôn mặt (BlazeFace)
    td_run_startup_detection();
    // [D5] Khoi tao face detector NHE (chay moi frame)
#if CONFIG_TD_USE_LITE_DETECTOR
    face_detector_lite_init();
#endif

#if CONFIG_TD_DIAG_SEQUENTIAL
    // [P4] Staging JPEG cho che do chan doan tuan tu (publish anh + telemetry cung frame)
    s_diag_jpeg = (uint8_t*)heap_caps_malloc(TD_DIAG_JPEG_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_diag_jpeg) {
        ESP_LOGW(TAG, "P4: khong cap phat duoc staging JPEG -> publish khong co anh");
    }
    ESP_LOGW(TAG, "P4 DIAG: sequential same-frame mode ON (FPS se giam)");
#endif

    // Reset lại mốc hiệu chuẩn ADAS để 5s calib bắt đầu khi pipeline chạy
    adas_controller_init();

    // [D16] Task PIPELINE Core 0: DECODE (double-buffer) + PREVIEW + LITE
#if TD_PIPELINE_DECODE_C0
    // [D21] Cap phat staging JPEG (PSRAM) de phat web cung luc voi du lieu AI
    for (int i = 0; i < TD_DEC_NBUF; i++) {
        s_stage_jpeg[i] = (uint8_t*)heap_caps_malloc(TD_STAGE_JPEG_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_stage_jpeg[i]) {
            ESP_LOGW(TAG, "D21: khong cap phat duoc staging JPEG[%d] -> web stream se cham", i);
        }
    }
    for (int i = 0; i < TD_DEC_NBUF; i++) {
        s_buf_free[i]  = xSemaphoreCreateBinaryStatic(&s_buf_free_storage[i]);
        s_buf_ready[i] = xSemaphoreCreateBinaryStatic(&s_buf_ready_storage[i]);
        xSemaphoreGive(s_buf_free[i]);   // ban dau ca 2 buffer deu ranh
    }
    xTaskCreatePinnedToCore(vTaskDecodeC0, "Task_Decode_Core0", 8192, NULL, 4, &s_decode_task, 0);
#endif

    // [D4b] Task detector chạy SONG SONG trên Core 0 (tracking không chặn Core 1)
#if TD_DET_C0_ENABLED && !TD_PIPELINE_DECODE_C0
    s_pub_gray = (uint8_t*)heap_caps_malloc((size_t)320 * 240, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_pub_gray) {
#if CONFIG_TD_DETECT_PERIODIC
        xTaskCreatePinnedToCore(vTaskDetector, "Task_Detector_Core0", 8192, NULL, 3, &s_det_task, 0);
#else
        xTaskCreatePinnedToCore(vTaskLiteDetectorC0, "Task_LiteDet_Core0", 8192, NULL, 3, &s_det_task, 0);
#endif
    } else {
        ESP_LOGW(TAG, "Khong cap phat duoc buffer publish -> tracking cham (lite chay tai cho)");
    }
#endif

    // 4. Create Dual-Core FreeRTOS Tasks
#if CONFIG_TD_USE_ONBOARD_CAMERA
    ESP_LOGI(TAG, "📷 Chế độ ONBOARD CAMERA: ESP32 tự thu hình, không dùng TCP laptop.");
#else
#if APP_ENABLE_WEB
    // Task Core 0: Wi-Fi TCP Frame Receiver (Network Ingestion)
    xTaskCreatePinnedToCore(
        wifi_stream_receiver_task,
        "Task_Network_Core0",
        8192,
        NULL,
        5,  // Priority 5
        NULL,
        0   // Pinned to Core 0
    );
#endif
#endif

    // Task Core 1: TinyML Inference & ADAS FSM Decision (Real-Time Edge AI)
    xTaskCreatePinnedToCore(
        vTaskEdgeAI_ADAS,
        "Task_EdgeAI_Core1",
        16384,
        NULL,
        6,  // Priority 6 (Higher priority)
        NULL,
        1   // Pinned to Core 1
    );

    ESP_LOGI(TAG, "🎉 Hệ thống đa nhân FreeRTOS đã khởi chạy song song trên 2 nhân LX7!");
}

































