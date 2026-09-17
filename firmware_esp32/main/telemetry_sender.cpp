#include "telemetry_sender.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "esp_log.h"
#include "driver/gpio.h"

static const char* TAG = "TELEMETRY";

#ifndef CONFIG_LAPTOP_HOST_IP
#define CONFIG_LAPTOP_HOST_IP "192.168.1.100"
#endif

#ifndef CONFIG_UDP_TELEMETRY_PORT
#define CONFIG_UDP_TELEMETRY_PORT 8889
#endif

#ifndef CONFIG_BUZZER_GPIO_PIN
#define CONFIG_BUZZER_GPIO_PIN 4
#endif

#ifndef CONFIG_LED_STATUS_GPIO_PIN
#define CONFIG_LED_STATUS_GPIO_PIN 48
#endif

static int s_udp_sock = -1;
static struct sockaddr_in s_dest_addr;
static bool s_is_initialized = false;

static uint32_t s_tx_ok = 0;
static uint32_t s_tx_err = 0;

// [v2.8.0] Timing per frame (ms) for the laptop display
static float s_dec_ms = 0.0f;
static float s_ai_ms = 0.0f;
static float s_total_ms = 0.0f;

void telemetry_sender_set_timing(float dec_ms, float ai_ms, float total_ms) {
    s_dec_ms = dec_ms;
    s_ai_ms = ai_ms;
    s_total_ms = total_ms;
}

// [v2.9.2] Face ROI info for diagnostics
static int s_roi_active = 0;
static int s_roi_x0 = 0, s_roi_y0 = 0, s_roi_size = 0;

void telemetry_sender_set_roi(int active, int x0, int y0, int size) {
    s_roi_active = active;
    s_roi_x0 = x0;
    s_roi_y0 = y0;
    s_roi_size = size;
}

// JSON Serialization Buffer
static char s_json_buffer[2048];

bool telemetry_sender_init(void) {
    // 1. Configure Hardware Actuators (GPIO Buzzer & LED)
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << CONFIG_BUZZER_GPIO_PIN) | (1ULL << CONFIG_LED_STATUS_GPIO_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_ENABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);

    // Initial state: OFF
    gpio_set_level((gpio_num_t)CONFIG_BUZZER_GPIO_PIN, 0);
    gpio_set_level((gpio_num_t)CONFIG_LED_STATUS_GPIO_PIN, 0);

    // 2. Setup UDP Socket
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "LỖI: Không thể tạo UDP socket (errno=%d)", errno);
        return false;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(CONFIG_UDP_TELEMETRY_PORT);
    s_dest_addr.sin_addr.s_addr = inet_addr(CONFIG_LAPTOP_HOST_IP);

    ESP_LOGI(TAG, "✅ Đã khởi tạo Telemetry Sender -> Laptop UDP tại %s:%d",
             CONFIG_LAPTOP_HOST_IP, CONFIG_UDP_TELEMETRY_PORT);

    s_is_initialized = true;
    return true;
}

void telemetry_sender_update_actuators(bool is_alarm_active) {
    if (is_alarm_active) {
        gpio_set_level((gpio_num_t)CONFIG_BUZZER_GPIO_PIN, 1);
        gpio_set_level((gpio_num_t)CONFIG_LED_STATUS_GPIO_PIN, 1);
    } else {
        gpio_set_level((gpio_num_t)CONFIG_BUZZER_GPIO_PIN, 0);
        gpio_set_level((gpio_num_t)CONFIG_LED_STATUS_GPIO_PIN, 0);
    }
}

bool telemetry_sender_dispatch(const adas_metrics_t* metrics, const point2d_t landmarks_22[22]) {
    if (!s_is_initialized || s_udp_sock < 0 || !metrics) {
        return false;
    }

    // Update physical GPIO actuators
    telemetry_sender_update_actuators(metrics->is_alarm_active);

    // Construct compact JSON payload
    int offset = 0;
    offset += snprintf(s_json_buffer + offset, sizeof(s_json_buffer) - offset,
                       "{\"ear\":%.2f,\"mar\":%.2f,\"yaw\":%.1f,\"pitch\":%.1f,\"roll\":%.1f,"
                       "\"status\":\"%s\",\"alarm\":%s,\"fps\":%.1f,"
                       "\"dec\":%.1f,\"ai\":%.1f,\"total\":%.1f,"
                       "\"ear_thr\":%.3f,\"mar_thr\":%.3f,\"mouth_s\":%.2f,\"yawns\":%u,"
                       "\"roi\":%d,\"rx\":%d,\"ry\":%d,\"rs\":%d,\"landmarks\":[",
                       metrics->ear,
                       metrics->mar,
                       metrics->pose.is_valid ? metrics->pose.yaw : 0.0f,
                       metrics->pose.is_valid ? metrics->pose.pitch : 0.0f,
                       metrics->pose.is_valid ? metrics->pose.roll : 0.0f,
                       metrics->status_str,
                       metrics->is_alarm_active ? "true" : "false",
                       metrics->esp32_fps,
                       (double)s_dec_ms,
                       (double)s_ai_ms,
                       (double)s_total_ms,
                       (double)metrics->ear_threshold,
                       (double)metrics->mar_threshold,
                       (double)metrics->mouth_open_s,
                       (unsigned)metrics->total_yawns,
                       s_roi_active, s_roi_x0, s_roi_y0, s_roi_size);

    // Append 22 landmarks as 44 float array [x0, y0, x1, y1, ...]
    if (landmarks_22) {
        for (int i = 0; i < 22; i++) {
            offset += snprintf(s_json_buffer + offset, sizeof(s_json_buffer) - offset,
                               "%.3f,%.3f%s",
                               landmarks_22[i].x,
                               landmarks_22[i].y,
                               (i < 21) ? "," : "");
        }
    }

    offset += snprintf(s_json_buffer + offset, sizeof(s_json_buffer) - offset, "]}");

    // Send UDP Datagram to Laptop Host
    int sent = sendto(s_udp_sock, s_json_buffer, offset, 0,
                      (struct sockaddr*)&s_dest_addr, sizeof(s_dest_addr));

    if (sent > 0) {
        s_tx_ok++;
        if (s_tx_ok == 1 || (s_tx_ok % 100) == 0) {
            ESP_LOGI(TAG, "Đã gửi %u gói telemetry tới %s:%d",
                     (unsigned)s_tx_ok, CONFIG_LAPTOP_HOST_IP, CONFIG_UDP_TELEMETRY_PORT);
        }
    } else {
        s_tx_err++;
        if (s_tx_err <= 5) {
            ESP_LOGW(TAG, "sendto lỗi (sent=%d, errno=%d)", sent, errno);
        }
    }

    return (sent > 0);
}

void telemetry_sender_image(const int8_t* tensor, float scale, int32_t zp, int w, int h) {
    if (!s_is_initialized || s_udp_sock < 0 || !tensor || w <= 0 || h <= 0) {
        return;
    }
    if (w * h > 128 * 128) {
        return;
    }

    static uint8_t pkt[8 + 128 * 128];
    pkt[0] = 0xAA; pkt[1] = 0x56; pkt[2] = 0xAA; pkt[3] = 0x56;
    pkt[4] = (uint8_t)(w & 0xFF);      pkt[5] = (uint8_t)((w >> 8) & 0xFF);
    pkt[6] = (uint8_t)(h & 0xFF);      pkt[7] = (uint8_t)((h >> 8) & 0xFF);

    size_t n = 8;
    for (int i = 0; i < w * h; i++) {
        // Dequant: gray = (q - zp) * scale * 128 + 128  (do training normalize (px-128)/128)
        int g = (int)lroundf(((float)(tensor[i] - zp)) * scale * 128.0f + 128.0f);
        if (g < 0) g = 0; else if (g > 255) g = 255;
        pkt[n++] = (uint8_t)g;
    }

    sendto(s_udp_sock, pkt, n, 0, (struct sockaddr*)&s_dest_addr, sizeof(s_dest_addr));
}

// [v2.9.4] Preview toàn khung camera (nearest-neighbor downsample) - để chẩn đoán hướng/khung hình.
void telemetry_sender_preview(const uint8_t* gray, int src_w, int src_h, int out_w, int out_h) {
    if (!s_is_initialized || s_udp_sock < 0 || !gray || src_w <= 0 || src_h <= 0) {
        return;
    }
    if (out_w <= 0 || out_h <= 0 || out_w > 200 || out_h > 200) {
        return;
    }
    static uint8_t pkt[8 + 200 * 200];
    pkt[0] = 0xAA; pkt[1] = 0x56; pkt[2] = 0xA0; pkt[3] = 0x01;
    pkt[4] = (uint8_t)(out_w & 0xFF); pkt[5] = (uint8_t)((out_w >> 8) & 0xFF);
    pkt[6] = (uint8_t)(out_h & 0xFF); pkt[7] = (uint8_t)((out_h >> 8) & 0xFF);

    size_t n = 8;
    for (int y = 0; y < out_h; y++) {
        int sy = (int)((float)y * (float)src_h / (float)out_h);
        if (sy > src_h - 1) sy = src_h - 1;
        for (int x = 0; x < out_w; x++) {
            int sx = (int)((float)x * (float)src_w / (float)out_w);
            if (sx > src_w - 1) sx = src_w - 1;
            pkt[n++] = gray[sy * src_w + sx];
        }
    }
    sendto(s_udp_sock, pkt, n, 0, (struct sockaddr*)&s_dest_addr, sizeof(s_dest_addr));
}
