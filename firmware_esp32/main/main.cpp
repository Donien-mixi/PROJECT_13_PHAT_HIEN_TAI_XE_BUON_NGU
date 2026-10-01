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

// Submodules
#include "wifi_stream_client.h"
#include "camera_capture.h"
#include "image_decoder.h"
#include "roi_tracker.h"
#include "face_detector.h"
#include "face_detector_lite.h"
#include "ai_inference.h"
#include "pnp_solver.h"
#include "adas_controller.h"
#include "telemetry_sender.h"
#include "web_server.h"
#include "tinydriver_model_data.h"

static const char* TAG = "MAIN_APP";

static uint32_t s_device_frame_counter = 0;

// [D2] Crop do DETECTOR (BlazeFace) xác định (nguồn độc lập với TinyDriverNet)
static image_crop_roi_t s_det_crop = {0, 0, 0, 0, 0};
static bool s_det_crop_valid = false;
static int  s_detect_countdown = 0;   // (dự phòng) đếm frame tới lần detect lại

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

// [D3] Suy ra vùng crop canonical từ hộp mặt (ưu tiên 6 keypoints -> khớp khung lúc train)
static void td_crop_from_box(const face_box_t* b, int gw, int gh) {
    int minDim = (gw < gh) ? gw : gh;

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

    float bcx = b->cx, bcy = b->cy, bsz = b->size;

    int64_t now_us = esp_timer_get_time();
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
    float fS  = 1.080f * s_sm_sz;
    float fcx = s_sm_cx + 0.040f * s_sm_sz;
    float fcy = s_sm_cy - 0.028f * s_sm_sz;
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
            s_dec_ok[cur] = false;
            xSemaphoreGive(s_buf_ready[cur]);
            cur ^= 1;
            continue;
        }
        const uint8_t* jpeg = cam.data;
        size_t jlen = cam.length;
        // [D21] Chi COPY vao staging, KHONG phat len web o day (doi AI xu ly xong)
        if (s_stage_jpeg[cur] && jlen > 0 && jlen <= TD_STAGE_JPEG_MAX) {
            memcpy(s_stage_jpeg[cur], jpeg, jlen);
            s_stage_len[cur] = jlen;
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
        if (s_stage_jpeg[cur] && jlen > 0 && jlen <= TD_STAGE_JPEG_MAX) {
            memcpy(s_stage_jpeg[cur], jpeg, jlen);
            s_stage_len[cur] = jlen;
        }
#endif

        // ROI: uu tien detector, fallback landmark tracker (giong laptop)
        image_crop_roi_t lroi = {0, 0, 0, 0, 0}, troi = {0, 0, 0, 0, 0};
        const image_crop_roi_t* p_roi = NULL;
        portENTER_CRITICAL(&s_crop_mux);
        bool hv = s_det_crop_valid;
        if (hv) lroi = s_det_crop;
        portEXIT_CRITICAL(&s_crop_mux);
        if (hv) {
            p_roi = &lroi;
        } else if (roi_tracker_is_valid()) {
            face_roi_t fr = roi_tracker_get();
            troi.x0 = fr.x0; troi.y0 = fr.y0; troi.size = fr.size;
            p_roi = &troi;
        }

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

        // Buffer cur da day du -> tra cho Core 1 (AI chay SONG SONG phan duoi day)
        xSemaphoreGive(s_buf_ready[cur]);

        if (ok) {
            int gw = 0, gh = 0;
            const uint8_t* g = image_decoder_get_last_gray(&gw, &gh);
            if (g && gw > 0 && gh > 0) {
                // Preview toan khung cho laptop (buffer pkt rieng, an toan da luong)
                telemetry_sender_preview(g, gw, gh, 128, 96);
                // Lite detector -> cap nhat crop canonical cho frame KE TIEP
                face_box_t lb;
                if (face_detector_lite_run(g, gw, gh, &lb) && lb.valid) {
                    td_crop_from_box(&lb, gw, gh);
                }
            }
        }
        cur ^= 1;
    }
}
#endif  // TD_PIPELINE_DECODE_C0

// [D2] Detect 1 lần lúc khởi động -> suy ra vùng crop (thay cho crop cố định)
static void td_run_startup_detection(void) {
#if CONFIG_TD_USE_ONBOARD_CAMERA
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
                xSemaphoreGive(s_buf_free[slot]);
                s_dec_slot ^= 1;
                continue;
            }
            frame_index = s_dec_frame_idx[slot];
            roi_used    = s_dec_roi[slot];
            t_decode    = s_dec_us[slot];
            memcpy(input_tensor_ptr, s_dec_buf[slot],
                   (size_t)TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT);
            xSemaphoreGive(s_buf_free[slot]);      // tra buffer NGAY -> Core 0 giai ma tiep
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
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }
        jpeg_data = cam_frame.data;
        jpeg_len = cam_frame.length;
        frame_index = ++s_device_frame_counter;
        web_server_update_frame(jpeg_data, jpeg_len);
#else
        frame_buffer_t frame;
        // 1. Acquire the latest 1:1 JPEG frame from PSRAM Double Buffer
        if (!wifi_stream_acquire_latest_frame(&frame, 100)) {
            // Waiting for frame from laptop
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        jpeg_data = frame.buffer;
        jpeg_len = frame.length;
        frame_index = frame.frame_index;
        web_server_update_frame(jpeg_data, jpeg_len);
#endif

        // 2. Fast JPEG Decode & Isomorphic Downsample directly to 96x96 INT8
        //    [v2.9.0] Nếu ROI khuôn mặt đã sẵn sàng -> crop sát mặt (giống lúc train),
        //    ngược lại tự crop giữa khung (bootstrap).
        // [D8] Ưu tiên 1 (GIỐNG LAPTOP): DETECTOR là nguồn chính MỖI FRAME
        //   (laptop: MediaPipe/Haar). Đọc an toàn qua mutex.
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

#if CONFIG_TD_USE_ONBOARD_CAMERA
        camera_capture_release();
#else
        wifi_stream_release_frame();
#endif

        if (!dec_ok) {
            ESP_LOGW(TAG, "Lỗi giải nén frame #%u", (unsigned int)frame_index);
            continue;
        }
        t_decode = esp_timer_get_time() - t0;

        // [v2.9.4] Gui preview TOAN KHUNG camera de chan doan huong/khung hinh tren Laptop
        {
            int gw = 0, gh = 0;
            const uint8_t* gray_full = image_decoder_get_last_gray(&gw, &gh);
            if (gray_full && gw > 0 && gh > 0) {
                telemetry_sender_preview(gray_full, gw, gh, 128, 96);
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

        // [v2.9.6 - FIX] Snapshot ảnh input TRƯỚC Invoke():
        // TFLM tái sử dụng vùng arena nên input tensor BỊ GHI ĐÈ trong Invoke.
        // Ảnh debug gửi đi phải lấy bản snapshot này (không đọc sau Invoke).
        static int8_t s_model_input_snapshot[TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT];
        memcpy(s_model_input_snapshot, input_tensor_ptr,
               (size_t)TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT);

        // 3. Neural Network Inference on Core 1 (TinyDriver-LandmarkNet)
        int64_t t_ai = 0;
        if (!ai_inference_run(landmarks, &t_ai)) {
            ESP_LOGE(TAG, "Lỗi suy luận mạng nơ-ron!");
            continue;
        }

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

        // 5. Update ADAS Finite State Machine (EAR, MAR, Microsleep, Fatigue, Distraction)
        adas_controller_update(landmarks, &head_pose, current_fps, &adas_metrics);

        // [D23] HIỂN THỊ pose ĐÃ TRỪ BIAS: nếu không, khi mặt hướng thẳng thì
        // PITCH/YAW vẫn hiện ~+10..14° (bias POSIT) -> trục X/Y/Z nghiêng sai trên web.
        // (ADAS bên trong vẫn dùng bias riêng để báo mất tập trung — không đổi hành vi.)
        adas_metrics.pose.pitch -= adas_controller_get_pitch_bias();
        adas_metrics.pose.yaw   -= adas_controller_get_yaw_bias();

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
        telemetry_sender_dispatch(&adas_metrics, landmarks);

        // [D21] Phat khung hinh len Web Dashboard CUNG LUC voi du lieu AI (22 diem/ROI/MAR)
        // => video va lop phu KHOP NHAU (het cam giac "diem sai cho / lag" khi di chuyen).
        if (s_stage_jpeg[dec_slot_used] && s_stage_len[dec_slot_used] > 0) {
            web_server_update_frame(s_stage_jpeg[dec_slot_used], s_stage_len[dec_slot_used]);
        }

        // Cập nhật số liệu telemetry và 22 mốc lên Web Dashboard
        web_server_update_telemetry(&adas_metrics, landmarks,
                                   crop_mode, roi_used.x0, roi_used.y0, roi_used.size,
                                   (float)t_decode / 1000.0f,
                                   (float)t_ai / 1000.0f,
                                   (float)t_pnp / 1000.0f,
                                   (float)(esp_timer_get_time() - t0) / 1000.0f);

        // Kiểm tra xem người dùng có bấm nút "Hiệu chuẩn lại" từ Web UI không
        if (web_server_is_recalibrate_requested()) {
            ESP_LOGI(TAG, "🔄 Kích hoạt lại quá trình tự hiệu chuẩn ADAS theo yêu cầu Web UI!");
            adas_controller_init();
        }

        // 6b. [v2.7.0] Gửi ảnh 96x96 (đúng cái ESP32 nhìn) để laptop HIỂN THỊ (display-only)
        telemetry_sender_image(s_model_input_snapshot,
                               TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT,
                               TINYDRIVER_INPUT_WIDTH, TINYDRIVER_INPUT_HEIGHT);

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
            int64_t now = esp_timer_get_time();
            float elapsed_sec = (float)(now - fps_timer_start) / 1e6f;
            current_fps = (elapsed_sec > 0.0f) ? (5.0f / elapsed_sec) : 0.0f;
            fps_timer_start = now;

            ESP_LOGI(TAG, "[Frame #%4u] FPS: %4.1f | Dec: %3.1fms | AI: %3.1fms | PnP: %3.1fms | Lite: %4.1fms | Total: %3.1fms | ROI: %d,%d,%d | EAR: %.2f | MAR: %.2f | Yaw: %+5.1f° | Status: %s %s",
                     (unsigned int)processed_frames,
                     current_fps,
                     (float)t_decode / 1000.0f,
                     (float)t_ai / 1000.0f,
                     (float)t_pnp / 1000.0f,
                     (float)face_detector_lite_last_ms(),
                     (float)t_total / 1000.0f,
                     roi_used.x0, roi_used.y0, roi_used.size,
                     adas_metrics.ear,
                     adas_metrics.mar,
                     head_pose.is_valid ? head_pose.yaw : 0.0f,
                     adas_metrics.status_str,
                     adas_metrics.is_alarm_active ? "🚨 [CÒI HÚ]" : "");
        }
    }
}

extern "C" void app_main(void) {
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
    if (!wifi_stream_init()) {
        ESP_LOGE(TAG, "Khởi tạo Wi-Fi Stream thất bại!");
        return;
    }

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

    if (!telemetry_sender_init()) {
        ESP_LOGE(TAG, "Khởi tạo Telemetry Sender thất bại!");
        return;
    }

    // Khởi tạo HTTP Web Server nhúng (Port 80)
    if (!web_server_init()) {
        ESP_LOGW(TAG, "Khởi tạo Web Server thất bại (kiểm tra tài nguyên mạng)!");
    }

    ESP_LOGI(TAG, "✅ Tất cả các module phần cứng và phần mềm đã sẵn sàng!");

    // [D2] Detect 1 lần để xác định vùng crop khuôn mặt (BlazeFace)
    td_run_startup_detection();
    // [D5] Khoi tao face detector NHE (chay moi frame)
#if CONFIG_TD_USE_LITE_DETECTOR
    face_detector_lite_init();
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
