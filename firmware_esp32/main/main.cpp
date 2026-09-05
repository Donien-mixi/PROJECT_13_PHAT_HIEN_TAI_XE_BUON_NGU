#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "esp_timer.h"

// Submodules
#include "wifi_stream_client.h"
#include "image_decoder.h"
#include "ai_inference.h"
#include "pnp_solver.h"
#include "adas_controller.h"
#include "telemetry_sender.h"
#include "tinydriver_model_data.h"

static const char* TAG = "MAIN_APP";

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

    while (1) {
        frame_buffer_t frame;
        // 1. Acquire the latest 1:1 JPEG frame from PSRAM Double Buffer
        if (!wifi_stream_acquire_latest_frame(&frame, 100)) {
            // Waiting for frame from laptop
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        int64_t t0 = esp_timer_get_time();

        // 2. Fast JPEG Decode & Isomorphic Downsample directly to 96x96 INT8
        bool dec_ok = image_decoder_process_jpeg(
            frame.buffer, frame.length,
            input_tensor_ptr,
            TINYDRIVER_INPUT_SCALE, TINYDRIVER_INPUT_ZERO_POINT
        );
        wifi_stream_release_frame();

        if (!dec_ok) {
            ESP_LOGW(TAG, "Lỗi giải nén frame #%u", (unsigned int)frame.frame_index);
            continue;
        }
        int64_t t_decode = esp_timer_get_time() - t0;

        // 3. Neural Network Inference on Core 1 (TinyDriver-LandmarkNet)
        int64_t t_ai = 0;
        if (!ai_inference_run(landmarks, &t_ai)) {
            ESP_LOGE(TAG, "Lỗi suy luận mạng nơ-ron!");
            continue;
        }

        // 4. Solve 3D Head Pose (Yaw, Pitch, Roll) via Pure C++ POSIT / PnP
        int64_t t_pnp_start = esp_timer_get_time();
        pnp_solve_head_pose(landmarks, &head_pose);
        int64_t t_pnp = esp_timer_get_time() - t_pnp_start;

        // 5. Update ADAS Finite State Machine (EAR, MAR, Microsleep, Fatigue, Distraction)
        adas_controller_update(landmarks, &head_pose, current_fps, &adas_metrics);

        // 6. Dispatch Telemetry JSON to Laptop Host & Actuate Onboard Buzzer/LED
        telemetry_sender_dispatch(&adas_metrics, landmarks);

        processed_frames++;
        int64_t t_total = esp_timer_get_time() - t0;

        // Calculate rolling FPS every 15 frames
        if (processed_frames % 15 == 0) {
            int64_t now = esp_timer_get_time();
            float elapsed_sec = (float)(now - fps_timer_start) / 1e6f;
            current_fps = (elapsed_sec > 0.0f) ? (15.0f / elapsed_sec) : 0.0f;
            fps_timer_start = now;

            ESP_LOGI(TAG, "[Frame #%4u] FPS: %4.1f | Dec: %3.1fms | AI: %3.1fms | PnP: %3.1fms | Total: %3.1fms | EAR: %.2f | MAR: %.2f | Yaw: %+5.1f° | Status: %s %s",
                     (unsigned int)processed_frames,
                     current_fps,
                     (float)t_decode / 1000.0f,
                     (float)t_ai / 1000.0f,
                     (float)t_pnp / 1000.0f,
                     (float)t_total / 1000.0f,
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

    if (!image_decoder_init()) {
        ESP_LOGE(TAG, "Khởi tạo Image Decoder thất bại!");
        return;
    }

    pnp_solver_init();
    adas_controller_init();

    if (!ai_inference_init()) {
        ESP_LOGE(TAG, "Khởi tạo AI Inference thất bại!");
        return;
    }

    if (!telemetry_sender_init()) {
        ESP_LOGE(TAG, "Khởi tạo Telemetry Sender thất bại!");
        return;
    }

    ESP_LOGI(TAG, "✅ Tất cả các module phần cứng và phần mềm đã sẵn sàng!");

    // 4. Create Dual-Core FreeRTOS Tasks
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
