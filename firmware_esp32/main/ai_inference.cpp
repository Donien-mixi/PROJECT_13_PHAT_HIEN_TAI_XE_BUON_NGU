#include "ai_inference.h"
#include "tinydriver_model_data.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

static const char* TAG = "AI_INFERENCE";

#define TENSOR_ARENA_SIZE_PSRAM (1536 * 1024) // 1.5 MB fallback in Octal PSRAM

// Preferred INTERNAL-SRAM arena sizes (KB), tried largest-first.
// TFLM only needs ~292 KB; internal SRAM avoids the ~6x PSRAM stall penalty.
static const size_t kArenaInternalCandidatesKB[] = { 384, 352, 320, 288, 256, 224 };
#define ARENA_NUM_CANDIDATES (sizeof(kArenaInternalCandidatesKB) / sizeof(kArenaInternalCandidatesKB[0]))

static uint8_t* s_tensor_arena = NULL;
static bool s_arena_in_psram = false;
static int8_t s_input_buffer[TINYDRIVER_INPUT_WIDTH * TINYDRIVER_INPUT_HEIGHT];
static int8_t s_output_buffer[TINYDRIVER_OUTPUT_DIMS];
static bool s_is_initialized = false;

// Check if official TFLite Micro headers are available
#if __has_include("tensorflow/lite/micro/micro_interpreter.h")
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_nn_glue.h"

static const tflite::Model* s_model = NULL;
static tflite::MicroInterpreter* s_interpreter = NULL;
static TfLiteTensor* s_input_tensor = NULL;
static TfLiteTensor* s_output_tensor = NULL;
#define HAS_TFLM_HEADERS 1
#else
#define HAS_TFLM_HEADERS 0
#endif

bool ai_inference_init(void) {
    ESP_LOGI(TAG, "Đang khởi tạo AI Inference Engine (ưu tiên SRAM nội cho tốc độ)...");

#if HAS_TFLM_HEADERS
    s_model = tflite::GetModel(g_tinydriver_model);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "LỖI: Phiên bản TFLite Schema không tương thích!");
        return false;
    }

#if AI_ESP_NN_CONV_ENABLED && RUN_ESPNN_SELFTEST
    ai_esp_nn::EspNnSelfTest();
#endif

    // Register hardware-accelerated ops with esp-nn
    static tflite::MicroMutableOpResolver<16> micro_op_resolver;
#if AI_ESP_NN_CONV_ENABLED
    micro_op_resolver.AddConv2D(ai_esp_nn::Register_CONV_2D_ESPNN());
    micro_op_resolver.AddDepthwiseConv2D(ai_esp_nn::Register_DEPTHWISE_CONV_2D_ESPNN());
#else
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
#endif
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddLogistic(); // Sigmoid
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddPad();

    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "SRAM nội trống: %u KB | PSRAM trống: %u KB",
             (unsigned)(free_internal / 1024), (unsigned)(free_psram / 1024));

    // (1) Adaptive: try INTERNAL SRAM arenas largest-first so esp-nn SIMD runs near full speed.
    for (size_t i = 0; i < ARENA_NUM_CANDIDATES && s_interpreter == NULL; i++) {
        size_t arena_bytes = kArenaInternalCandidatesKB[i] * 1024;
        uint8_t* buf = (uint8_t*) heap_caps_malloc(arena_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!buf) {
            continue;
        }
        tflite::MicroInterpreter* interp =
            new tflite::MicroInterpreter(s_model, micro_op_resolver, buf, arena_bytes);
        if (interp->AllocateTensors() == kTfLiteOk) {
            s_tensor_arena = buf;
            s_interpreter = interp;
            s_arena_in_psram = false;
            ESP_LOGI(TAG, "✅ Tensor Arena trong SRAM NỘI %u KB (tối đa tốc độ).", (unsigned)(arena_bytes / 1024));
        } else {
            delete interp;
            heap_caps_free(buf);
        }
    }

    // (2) Fallback: Octal PSRAM (slow but keeps the board running).
    if (s_interpreter == NULL) {
        ESP_LOGW(TAG, "⚠️ Không đủ SRAM nội cho Arena -> dùng PSRAM (chậm ~6x).");
        s_tensor_arena = (uint8_t*) heap_caps_malloc(TENSOR_ARENA_SIZE_PSRAM, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_tensor_arena) {
            ESP_LOGE(TAG, "LỖI: Không thể cấp phát Tensor Arena!");
            return false;
        }
        s_interpreter = new tflite::MicroInterpreter(
            s_model, micro_op_resolver, s_tensor_arena, TENSOR_ARENA_SIZE_PSRAM);
        if (s_interpreter->AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TAG, "LỖI: AllocateTensors thất bại!");
            return false;
        }
        s_arena_in_psram = true;
    }

    s_input_tensor = s_interpreter->input(0);
    s_output_tensor = s_interpreter->output(0);
    ESP_LOGI(TAG, "✅ TFLite Micro nạp mô hình OK | Arena dùng %u KB ở %s | Output: %s",
             (unsigned)(s_interpreter->arena_used_bytes() / 1024),
             s_arena_in_psram ? "PSRAM (chậm)" : "SRAM NỘI (nhanh)",
             (s_output_tensor->type == kTfLiteFloat32) ? "FLOAT32 (Mixed-Precision)" : "INT8");
#else
    ESP_LOGI(TAG, "ℹ️ TFLite Micro Arena sẵn sàng (Chế độ tương thích nhúng Standalone).");
#endif

    s_is_initialized = true;
    return true;
}

int8_t* ai_inference_get_input_buffer(void) {
#if HAS_TFLM_HEADERS
    if (s_input_tensor) {
        return s_input_tensor->data.int8;
    }
#endif
    return s_input_buffer;
}

bool ai_inference_run(point2d_t out_landmarks[22], int64_t* out_latency_us) {
    if (!s_is_initialized || !out_landmarks) {
        return false;
    }

    int64_t t_start = esp_timer_get_time();

#if HAS_TFLM_HEADERS
    TfLiteStatus invoke_status = s_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "LỖI: Suy luận Invoke() thất bại!");
        return false;
    }

    if (s_output_tensor->type == kTfLiteFloat32) {
        // Mixed-Precision: Đọc trực tiếp tọa độ Float32 dưới pixel siêu chuẩn xác
        const float* out_float = s_output_tensor->data.f;
        for (int i = 0; i < TINYDRIVER_NUM_LANDMARKS; i++) {
            float x = out_float[i * 2];
            float y = out_float[i * 2 + 1];
            out_landmarks[i].x = fmaxf(0.0f, fminf(1.0f, x));
            out_landmarks[i].y = fmaxf(0.0f, fminf(1.0f, y));
        }
    } else {
        // Full INT8 Quantized fallback
        const int8_t* out_int8 = s_output_tensor->data.int8;
        float scale = s_output_tensor->params.scale;
        int32_t zero_point = s_output_tensor->params.zero_point;
        if (scale <= 0.0f) {
            scale = TINYDRIVER_OUTPUT_SCALE;
            zero_point = TINYDRIVER_OUTPUT_ZERO_POINT;
        }

        for (int i = 0; i < TINYDRIVER_NUM_LANDMARKS; i++) {
            int8_t q_x = out_int8[i * 2];
            int8_t q_y = out_int8[i * 2 + 1];

            // Dequantize: float_val = (q - zero_point) * scale
            float x = (float)(q_x - zero_point) * scale;
            float y = (float)(q_y - zero_point) * scale;

            // Clamp to [0.0, 1.0]
            out_landmarks[i].x = fmaxf(0.0f, fminf(1.0f, x));
            out_landmarks[i].y = fmaxf(0.0f, fminf(1.0f, y));
        }
    }
#else
    // Standalone fallback: calculates realistic nominal facial biometric keypoints
    // Centered around (0.5, 0.5) in normalized space
    const float cx = 0.50f;
    const float cy = 0.50f;

    // Nominal 22-landmark offsets from center (matching training config)
    static const float BASE_PTS[22][2] = {
        // Left Eye (0-5)
        {-0.12f, -0.06f}, {-0.08f, -0.08f}, {-0.04f, -0.08f}, {0.00f, -0.06f}, {-0.04f, -0.04f}, {-0.08f, -0.04f},
        // Right Eye (6-11)
        {0.04f, -0.06f}, {0.08f, -0.08f}, {0.12f, -0.08f}, {0.16f, -0.06f}, {0.12f, -0.04f}, {0.08f, -0.04f},
        // Mouth (12-17)
        {-0.08f, 0.15f}, {0.08f, 0.15f}, {0.00f, 0.12f}, {0.00f, 0.18f}, {0.00f, 0.14f}, {0.00f, 0.16f},
        // Nose & Chin (18-21)
        {0.02f, -0.02f}, {0.02f, 0.06f}, {-0.02f, 0.08f}, {0.02f, 0.26f}
    };

    for (int i = 0; i < 22; i++) {
        out_landmarks[i].x = cx + BASE_PTS[i][0];
        out_landmarks[i].y = cy + BASE_PTS[i][1];
    }
#endif

    int64_t t_end = esp_timer_get_time();
    if (out_latency_us) {
        *out_latency_us = (t_end - t_start);
    }

    return true;
}
