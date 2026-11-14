#include "camera_capture.h"

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

static const char* TAG = "CAMERA";

static camera_fb_t* s_fb = NULL;
static uint32_t s_cam_seq = 0;   // [P1] so thu tu frame camera (chi tang khi acquire OK)

bool camera_capture_init(void) {
    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    config.pin_pwdn     = CONFIG_TD_CAM_PIN_PWDN;
    config.pin_reset    = CONFIG_TD_CAM_PIN_RESET;
    config.pin_xclk     = CONFIG_TD_CAM_PIN_XCLK;
    config.pin_sccb_sda = CONFIG_TD_CAM_PIN_SIOD;
    config.pin_sccb_scl = CONFIG_TD_CAM_PIN_SIOC;

    config.pin_d0 = CONFIG_TD_CAM_PIN_D0;
    config.pin_d1 = CONFIG_TD_CAM_PIN_D1;
    config.pin_d2 = CONFIG_TD_CAM_PIN_D2;
    config.pin_d3 = CONFIG_TD_CAM_PIN_D3;
    config.pin_d4 = CONFIG_TD_CAM_PIN_D4;
    config.pin_d5 = CONFIG_TD_CAM_PIN_D5;
    config.pin_d6 = CONFIG_TD_CAM_PIN_D6;
    config.pin_d7 = CONFIG_TD_CAM_PIN_D7;

    config.pin_vsync = CONFIG_TD_CAM_PIN_VSYNC;
    config.pin_href  = CONFIG_TD_CAM_PIN_HREF;
    config.pin_pclk  = CONFIG_TD_CAM_PIN_PCLK;

    config.xclk_freq_hz = CONFIG_TD_CAM_XCLK_FREQ;
#if TD_CAM_RAW_GRAY
    // [14/11 - SPEED] YUV422 (the driver exposes it as GRAYSCALE): byte 0 of every pixel pair is Y,
    // so the pipeline can skip the 0.60 s JPEG decode (see image_decoder_load_gray_raw).
    config.pixel_format = PIXFORMAT_GRAYSCALE;
#else
    config.pixel_format = PIXFORMAT_JPEG;
#endif

#if CONFIG_TD_CAM_FS_VGA
    config.frame_size = FRAMESIZE_VGA;
#elif CONFIG_TD_CAM_FS_240X240
    config.frame_size = FRAMESIZE_240X240;
#else
    config.frame_size = FRAMESIZE_QVGA;
#endif

    config.jpeg_quality = CONFIG_TD_CAM_JPEG_QUALITY;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
#if CONFIG_TD_CAM_GRAB_LATEST
    config.grab_mode = CAMERA_GRAB_LATEST;
#else
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
#endif

    ESP_LOGI(TAG, "Khởi tạo OV5640 (DVP): XCLK=GPIO%d SIOD=GPIO%d SIOC=GPIO%d VSYNC=GPIO%d HREF=GPIO%d PCLK=GPIO%d",
             CONFIG_TD_CAM_PIN_XCLK, CONFIG_TD_CAM_PIN_SIOD, CONFIG_TD_CAM_PIN_SIOC,
             CONFIG_TD_CAM_PIN_VSYNC, CONFIG_TD_CAM_PIN_HREF, CONFIG_TD_CAM_PIN_PCLK);

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "❌ esp_camera_init thất bại: 0x%x (kiểm tra pinout/nguồn/cáp FPC)", err);
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        // [v2.9.4] Đồng bộ hướng ảnh với CameraWebServer đã kiểm chứng:
        // module gắn ngược -> bật V-Flip, giữ H-Mirror OFF.
        s->set_vflip(s, 1);
        s->set_hmirror(s, 0);
        s->set_quality(s, CONFIG_TD_CAM_JPEG_QUALITY);
        ESP_LOGI(TAG, "✅ Camera sẵn sàng: PID=0x%02X VER=0x%02X | %s | %s | V-Flip=1 H-Mirror=0 | Q=%d",
                 s->id.PID, s->id.VER,
#if CONFIG_TD_CAM_FS_VGA
                 "VGA 640x480"
#elif CONFIG_TD_CAM_FS_240X240
                 "240x240"
#else
                 "QVGA 320x240"
#endif
#if TD_CAM_RAW_GRAY
                 , "YUV422 (raw Y)"
#else
                 , "JPEG"
#endif
                 , CONFIG_TD_CAM_JPEG_QUALITY
        );
    } else {
        ESP_LOGW(TAG, "⚠️ Camera init OK nhưng không đọc được sensor_t.");
    }

#if CONFIG_CAMERA_AF_SUPPORT
    if (s && s->af_is_supported && s->af_is_supported(s)) {
        int af = s->af_init ? s->af_init(s, 3000) : -1;
        ESP_LOGI(TAG, "OV5640 autofocus init: %s", (af == 0) ? "OK" : "không khả dụng");
    }
#endif

    return true;
}

bool camera_capture_acquire(camera_frame_t* out_frame) {
    if (!out_frame) {
        return false;
    }

    if (s_fb) {
        esp_camera_fb_return(s_fb);
        s_fb = NULL;
    }

    s_fb = esp_camera_fb_get();
    if (!s_fb) {
        return false;
    }

    out_frame->data   = s_fb->buf;
    out_frame->length = s_fb->len;
    out_frame->width  = (int)s_fb->width;
    out_frame->height = (int)s_fb->height;

    // [P1 - TRUY VET] Timestamp DMA cua frame (cam_hal.c: set tai VSYNC dau frame).
    // Neu driver khong set (<=0) thi fallback ve thoi diem acquire.
    int64_t ts = (int64_t)s_fb->timestamp.tv_sec * 1000000LL + (int64_t)s_fb->timestamp.tv_usec;
    out_frame->timestamp_us = (ts > 0) ? ts : esp_timer_get_time();
    out_frame->seq = ++s_cam_seq;
    return true;
}

void camera_capture_release(void) {
    if (s_fb) {
        esp_camera_fb_return(s_fb);
        s_fb = NULL;
    }
}
