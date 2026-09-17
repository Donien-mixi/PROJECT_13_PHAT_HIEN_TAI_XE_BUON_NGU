#ifndef ESP_NN_GLUE_H
#define ESP_NN_GLUE_H

#if __has_include("tensorflow/lite/c/common.h")
#include "tensorflow/lite/c/common.h"
#define HAS_TFLM_HEADERS 1
#else
#define HAS_TFLM_HEADERS 0
#endif

// =====================================================================
// LƯU Ý QUAN TRỌNG (v2.9.7):
// esp-nn ĐÃ được tích hợp sẵn qua managed component:
//   - "espressif__esp-nn" cung cấp kernel SIMD
//   - "espressif__esp-tflite-micro" TỰ thay conv.cc/depthwise_conv.cc... bằng
//     kernels/esp_nn/*.cc và định nghĩa -DESP_NN
// => Chỉ cần AddConv2D()/AddDepthwiseConv2D() chuẩn là ĐÃ chạy esp-nn SIMD.
// Glue vendored dưới đây là DI SẢN (Arduino/TFLM cũ) -> TẮT để tránh trùng symbol.
// =====================================================================
#define AI_ESP_NN_CONV_ENABLED 0
#define RUN_ESPNN_SELFTEST     0

#if HAS_TFLM_HEADERS && AI_ESP_NN_CONV_ENABLED
#include "tensorflow/lite/micro/micro_common.h"   // TFLMRegistration (API TFLM mới)

namespace ai_esp_nn {

// Custom registration cho TFLite Micro Mutable Op Resolver
TFLMRegistration Register_CONV_2D_ESPNN();
TFLMRegistration Register_DEPTHWISE_CONV_2D_ESPNN();

// Self-test số học so sánh ESP-NN và Reference kernels lúc boot
void EspNnSelfTest();

}  // namespace ai_esp_nn
#endif

#endif  // ESP_NN_GLUE_H

