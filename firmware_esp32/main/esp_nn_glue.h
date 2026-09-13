#ifndef ESP_NN_GLUE_H
#define ESP_NN_GLUE_H

#if __has_include("tensorflow/lite/c/common.h")
#include "tensorflow/lite/c/common.h"
#define HAS_TFLM_HEADERS 1
#else
#define HAS_TFLM_HEADERS 0
#endif

// =====================================================================
// ESP-NN SIMD kernels cho CONV_2D + DEPTHWISE_CONV_2D trên ESP32-S3
// (Xtensa dual-core LX7 vector instructions).
// Tối ưu hóa tính toán Inverted Residual MBConv cho TinyDriverNet.
// =====================================================================
#define AI_ESP_NN_CONV_ENABLED 0
#define RUN_ESPNN_SELFTEST     0

#if HAS_TFLM_HEADERS && AI_ESP_NN_CONV_ENABLED
namespace ai_esp_nn {

// Custom registration cho TFLite Micro Mutable Op Resolver
TfLiteRegistration Register_CONV_2D_ESPNN();
TfLiteRegistration Register_DEPTHWISE_CONV_2D_ESPNN();

// Self-test số học so sánh ESP-NN và Reference kernels lúc boot
void EspNnSelfTest();

}  // namespace ai_esp_nn
#endif

#endif  // ESP_NN_GLUE_H

