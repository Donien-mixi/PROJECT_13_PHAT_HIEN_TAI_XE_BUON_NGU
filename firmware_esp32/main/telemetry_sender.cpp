#include "telemetry_sender.h"
#include <string.h>
#include <stdio.h>
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
                       "\"status\":\"%s\",\"alarm\":%s,\"fps\":%.1f,\"landmarks\":[",
                       metrics->ear,
                       metrics->mar,
                       metrics->pose.is_valid ? metrics->pose.yaw : 0.0f,
                       metrics->pose.is_valid ? metrics->pose.pitch : 0.0f,
                       metrics->pose.is_valid ? metrics->pose.roll : 0.0f,
                       metrics->status_str,
                       metrics->is_alarm_active ? "true" : "false",
                       metrics->esp32_fps);

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

    return (sent > 0);
}
