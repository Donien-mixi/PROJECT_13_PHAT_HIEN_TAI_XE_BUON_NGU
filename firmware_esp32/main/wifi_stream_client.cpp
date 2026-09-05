#include "wifi_stream_client.h"
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"

static const char* TAG = "WIFI_STREAM";

// Configuration fallbacks if not defined in Kconfig
#ifndef CONFIG_ESP_WIFI_SSID
#define CONFIG_ESP_WIFI_SSID "AIoT_Driver_Net"
#endif

#ifndef CONFIG_ESP_WIFI_PASSWORD
#define CONFIG_ESP_WIFI_PASSWORD "12345678"
#endif

#ifndef CONFIG_LAPTOP_HOST_IP
#define CONFIG_LAPTOP_HOST_IP "192.168.1.100"
#endif

#ifndef CONFIG_TCP_STREAM_PORT
#define CONFIG_TCP_STREAM_PORT 8888
#endif

// Wi-Fi Connection State
static EventGroupHandle_t s_wifi_event_group;
static const int WIFI_CONNECTED_BIT = BIT0;

// Double Buffering in Octal PSRAM
static uint8_t* s_buffer_a = NULL;
static uint8_t* s_buffer_b = NULL;

static uint8_t* s_write_buffer = NULL; // Currently being filled by Network task
static uint8_t* s_ready_buffer = NULL; // Ready to be consumed by AI task

static size_t s_ready_length = 0;
static uint32_t s_frame_counter = 0;
static int64_t s_ready_timestamp = 0;

static SemaphoreHandle_t s_buffer_mutex = NULL;
static SemaphoreHandle_t s_new_frame_sem = NULL;

static bool s_is_connected = false;

// Wi-Fi Event Handler
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_is_connected = false;
        ESP_LOGW(TAG, "Mất kết nối Wi-Fi! Đang thử kết nối lại...");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Đã nhận IP từ Router: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

bool wifi_stream_init(void) {
    s_wifi_event_group = xEventGroupCreate();
    s_buffer_mutex = xSemaphoreCreateMutex();
    s_new_frame_sem = xSemaphoreCreateBinary();

    // Allocate Double Buffer in 8MB PSRAM
    ESP_LOGI(TAG, "Đang cấp phát Double Buffer trong Octal PSRAM (%d KB mỗi buffer)...", MAX_JPEG_BUFFER_SIZE / 1024);
    s_buffer_a = (uint8_t*) heap_caps_malloc(MAX_JPEG_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_buffer_b = (uint8_t*) heap_caps_malloc(MAX_JPEG_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!s_buffer_a || !s_buffer_b) {
        ESP_LOGE(TAG, "LỖI: Không đủ bộ nhớ PSRAM để cấp phát Double Buffer!");
        return false;
    }

    s_write_buffer = s_buffer_a;
    s_ready_buffer = s_buffer_b;
    s_ready_length = 0;

    // Initialize TCP/IP Adapter and Wi-Fi
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid, CONFIG_ESP_WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, CONFIG_ESP_WIFI_PASSWORD, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Đã khởi động Wi-Fi Station, đang kết nối tới SSID: %s...", CONFIG_ESP_WIFI_SSID);
    return true;
}

static bool recv_all(int sock, uint8_t* buffer, size_t length) {
    size_t total_received = 0;
    while (total_received < length) {
        int r = recv(sock, buffer + total_received, length - total_received, 0);
        if (r <= 0) {
            return false;
        }
        total_received += r;
    }
    return true;
}

void wifi_stream_receiver_task(void* pvParameters) {
    ESP_LOGI(TAG, "Khởi chạy Network Receiver Task trên Core 0.");

    while (1) {
        // 1. Wait for Wi-Fi connection
        xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

        // 2. Setup TCP Socket to Laptop Host
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        if (sock < 0) {
            ESP_LOGE(TAG, "Không thể tạo TCP Socket (errno=%d)", errno);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // Set TCP Timeout
        struct timeval tv;
        tv.tv_sec = 3;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct sockaddr_in dest_addr;
        dest_addr.sin_addr.s_addr = inet_addr(CONFIG_LAPTOP_HOST_IP);
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(CONFIG_TCP_STREAM_PORT);

        ESP_LOGI(TAG, "Đang kết nối tới Laptop Host tại %s:%d...", CONFIG_LAPTOP_HOST_IP, CONFIG_TCP_STREAM_PORT);
        int err = connect(sock, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
        if (err != 0) {
            ESP_LOGW(TAG, "Kết nối TCP thất bại (errno=%d). Thử lại sau 2 giây...", errno);
            close(sock);
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }

        ESP_LOGI(TAG, "✅ Đã kết nối thành công tới Laptop Camera Streamer!");
        s_is_connected = true;

        // 3. Receive Stream Loop
        while (1) {
            // Find 4-byte Magic: 0xAA55AA55
            uint32_t magic_window = 0;
            bool magic_found = false;

            while (!magic_found) {
                uint8_t byte = 0;
                int r = recv(sock, &byte, 1, 0);
                if (r <= 0) break;
                magic_window = (magic_window << 8) | byte;
                if (magic_window == TCP_STREAM_MAGIC) {
                    magic_found = true;
                }
            }

            if (!magic_found) {
                ESP_LOGW(TAG, "Mất đồng bộ Magic Header hoặc đứt kết nối.");
                break;
            }

            // Read 4-byte Payload Length (Big-Endian uint32)
            uint32_t payload_len_be = 0;
            if (!recv_all(sock, (uint8_t*)&payload_len_be, sizeof(payload_len_be))) {
                ESP_LOGW(TAG, "Lỗi đọc Payload Length.");
                break;
            }
            uint32_t payload_len = ntohl(payload_len_be);

            if (payload_len == 0 || payload_len > MAX_JPEG_BUFFER_SIZE) {
                ESP_LOGW(TAG, "Kích thước payload không hợp lệ: %u bytes", (unsigned int)payload_len);
                continue;
            }

            // Read JPEG Bytes directly into write buffer in PSRAM
            if (!recv_all(sock, s_write_buffer, payload_len)) {
                ESP_LOGW(TAG, "Lỗi đọc toàn bộ JPEG bytes.");
                break;
            }

            // Swap Double Buffers
            if (xSemaphoreTake(s_buffer_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                uint8_t* temp = s_ready_buffer;
                s_ready_buffer = s_write_buffer;
                s_write_buffer = temp;

                s_ready_length = payload_len;
                s_frame_counter++;
                s_ready_timestamp = esp_timer_get_time();

                xSemaphoreGive(s_buffer_mutex);
                xSemaphoreGive(s_new_frame_sem); // Notify AI task on Core 1
            }
        }

        s_is_connected = false;
        close(sock);
        ESP_LOGW(TAG, "Đã đóng kết nối TCP. Chờ 1 giây trước khi kết nối lại...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

bool wifi_stream_acquire_latest_frame(frame_buffer_t* out_frame, uint32_t timeout_ms) {
    if (xSemaphoreTake(s_new_frame_sem, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        return false;
    }

    if (xSemaphoreTake(s_buffer_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        out_frame->buffer = s_ready_buffer;
        out_frame->length = s_ready_length;
        out_frame->frame_index = s_frame_counter;
        out_frame->timestamp_us = s_ready_timestamp;
        xSemaphoreGive(s_buffer_mutex);
        return true;
    }
    return false;
}

void wifi_stream_release_frame(void) {
    // Double buffer swap handles release implicitly
}

bool wifi_stream_is_connected(void) {
    return s_is_connected;
}
