// =====================================================================
// Glue nối ESP-NN SIMD kernels vào TFLite Micro cho ESP32-S3 (Xtensa LX7).
// Tối ưu hóa tính toán Inverted Residual MBConv cho TinyDriverNet.
// =====================================================================
#include "esp_nn_glue.h"

#if HAS_TFLM_HEADERS && AI_ESP_NN_CONV_ENABLED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "esp_nn/include/esp_nn.h"

static const char* TAG = "ESP_NN_GLUE";

namespace ai_esp_nn {
namespace {

// ---------------------------------------------------------------------
// CONV_2D
// ---------------------------------------------------------------------
struct ConvNodeData {
  tflite::OpDataConv op_data;  // tái dùng struct công khai của conv.h
  int buffer_idx;              // scratch buffer (aligned filter copy của esp-nn)
};

void* ConvInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(ConvNodeData));
}

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  ConvNodeData* data = static_cast<ConvNodeData*>(node->user_data);
  const auto& params = *(static_cast<const TfLiteConvParams*>(node->builtin_data));

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, tflite::kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, tflite::kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, tflite::kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];
  const int input_channels = input->dims->data[3];

  // Per-channel quant arrays — giống ConvPrepare trong conv_common.cpp
  const int num_channels = filter->dims->data[tflite::kConvQuantizedDimension];
  data->op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->op_data.per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[tflite::kConvQuantizedDimension]);
  }

  TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type,
      &data->op_data));

  // esp-nn scratch: bản sao filter đã align (đo từ esp_nn_get_conv_scratch_size)
  data_dims_t input_dims = {input_width, input_height, input_channels, 1};
  data_dims_t output_dims = {output_width, output_height,
                             output->dims->data[3], 1};
  data_dims_t filter_dims = {filter_width, filter_height,
                             filter->dims->data[3], 0};
  conv_params_t conv_params = {
      0, 0,
      {params.stride_width, params.stride_height},
      {data->op_data.padding.width, data->op_data.padding.height},
      {0, 0},
      {-128, 127}};
  int scratch_buf_size = esp_nn_get_conv_scratch_size(
      &input_dims, &filter_dims, &output_dims, &conv_params);
  if (scratch_buf_size > 0) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, scratch_buf_size, &data->buffer_idx));
  } else {
    data->buffer_idx = -1;
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  return kTfLiteOk;
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (node->inputs->size == 3)
          ? tflite::micro::GetEvalInput(context, node, tflite::kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const ConvNodeData& data = *(static_cast<const ConvNodeData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);

  if (params.dilation_width_factor == 1 && params.dilation_height_factor == 1) {
    // --- ESP-NN SIMD path (bắt chước EvalQuantizedPerChannel của espressif) ---
    tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    tflite::RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    tflite::RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    const int32_t input_offset = -data.op_data.input_zero_point;
    const int32_t output_offset = data.op_data.output_zero_point;
    const int pad_width = data.op_data.padding.width;
    const int pad_height = data.op_data.padding.height;

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = filter_shape.Dims(0);
    const int batch_size = input_shape.Dims(0);

    const int32_t activation_min = data.op_data.output_activation_min;
    const int32_t activation_max = data.op_data.output_activation_max;

    void* scratch_buf = nullptr;
    if (data.buffer_idx > -1) {
      scratch_buf = context->GetScratchBuffer(context, data.buffer_idx);
    }
    esp_nn_set_conv_scratch_buf(scratch_buf);

    const int input_size = input_width * input_height * input_depth;
    const int output_size = output_width * output_height * output_depth;

    data_dims_t in_dims = {input_width, input_height, input_depth, 1};
    data_dims_t out_dims = {output_width, output_height, output_depth, 1};
    data_dims_t filt_dims = {filter_width, filter_height,
                             filter->dims->data[3], 0};
    conv_params_t conv_params = {
        input_offset, output_offset,
        {params.stride_width, params.stride_height},
        {pad_width, pad_height},
        {0, 0},
        {activation_min, activation_max}};
    quant_data_t quant_data = {
        data.op_data.per_channel_output_shift,
        data.op_data.per_channel_output_multiplier};

    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
      esp_nn_conv_s8(&in_dims, input_data + i_batch * input_size,
                     &filt_dims, tflite::micro::GetTensorData<int8_t>(filter),
                     tflite::micro::GetOptionalTensorData<int32_t>(bias),
                     &out_dims, output_data + i_batch * output_size,
                     &conv_params, &quant_data);
    }
  } else {
    // dilation != 1: esp-nn chưa hỗ trợ -> reference integer ops
    tflite::reference_integer_ops::ConvPerChannel(
        tflite::ConvParamsQuantized(params, data.op_data),
        data.op_data.per_channel_output_multiplier,
        data.op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetOptionalTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

// ---------------------------------------------------------------------
// DEPTHWISE_CONV_2D
// ---------------------------------------------------------------------
// [CHỐT LỖI] Self-test chứng minh: esp-nn dw path "opt" (channels không bội 8)
// bit-exact (maxdiff=0), NHƯNG path s8pad (ch%16==0, 3x3, pad 1/1 hoặc 0/0)
// SAI (maxdiff=255) và crash StoreProhibited (heap poison 0xbaad5678) trên
// esp-nn v1.3 — đây chính là nguyên nhân embedding rác trước đây.
// => Chỉ bypass path s8pad; các path còn lại (opt, s16, compaction) giữ esp-nn.
static bool DwUseEspNn(int input_channels, int ch_mult, int fW, int fH,
                       int padW, int padH) {
  bool is_s8pad_broken = (ch_mult == 1) && (input_channels % 16 == 0) &&
                         (fW == 3) && (fH == 3) &&
                         ((padW == 1 && padH == 1) || (padW == 0 && padH == 0));
  return !is_s8pad_broken;
}

struct DwConvNodeData {
  tflite::OpDataConv op_data;
  int buffer_idx;
  bool use_espnn;  // cache quyết định từ Prepare (shape không đổi sau đó)
};

// DepthwiseConvParamsQuantized là static trong depthwise_conv.cpp của lib,
// nên tự dựng DepthwiseParams tại đây (copy đúng công thức của lib).
tflite::DepthwiseParams DwConvParamsQuantized(const TfLiteDepthwiseConvParams& params,
                                              const tflite::OpDataConv& data) {
  tflite::DepthwiseParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier = data.output_multiplier;
  op_params.output_shift = -data.output_shift;
  op_params.padding_type = tflite::micro::RuntimePaddingType(params.padding);
  op_params.padding_values.height = data.padding.height;
  op_params.padding_values.width = data.padding.width;
  op_params.stride_height = params.stride_height;
  op_params.stride_width = params.stride_width;
  op_params.dilation_height_factor = params.dilation_height_factor;
  op_params.dilation_width_factor = params.dilation_width_factor;
  op_params.depth_multiplier = params.depth_multiplier;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;
  return op_params;
}

void* DwConvInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(DwConvNodeData));
}

TfLiteStatus DwConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  DwConvNodeData* data = static_cast<DwConvNodeData*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data));

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, tflite::kDepthwiseConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, tflite::kDepthwiseConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, tflite::kDepthwiseConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];

  const int num_channels = filter->dims->data[tflite::kDepthwiseConvQuantizedDimension];
  data->op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->op_data.per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
  }

  TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataDepthwiseConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type,
      &data->op_data));

  // Quyết định esp-nn sau khi có padding (shape cố định của node)
  data->use_espnn = DwUseEspNn(input->dims->data[3], params.depth_multiplier,
                               filter_width, filter_height,
                               data->op_data.padding.width,
                               data->op_data.padding.height);

  // esp-nn scratch CHỈ khi dùng esp-nn (reference không cần scratch)
  data->buffer_idx = -1;
  if (data->use_espnn) {
    data_dims_t input_dims = {input_width, input_height, input->dims->data[3], 1};
    data_dims_t output_dims = {output_width, output_height,
                               output->dims->data[3], 1};
    data_dims_t filter_dims = {filter_width, filter_height,
                               filter->dims->data[3], 0};
    dw_conv_params_t conv_params = {
        0, 0,
        params.depth_multiplier,
        {params.stride_width, params.stride_height},
        {data->op_data.padding.width, data->op_data.padding.height},
        {0, 0},
        {-128, 127}};
    int scratch_buf_size = esp_nn_get_depthwise_conv_scratch_size(
        &input_dims, &filter_dims, &output_dims, &conv_params);
    if (scratch_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, scratch_buf_size, &data->buffer_idx));
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus DwConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (node->inputs->size == 3)
          ? tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kDepthwiseConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const DwConvNodeData& data = *(static_cast<const DwConvNodeData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);

  if (data.use_espnn && params.dilation_width_factor == 1 &&
      params.dilation_height_factor == 1) {
    tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    tflite::RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    tflite::RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = -data.op_data.input_zero_point;
    const int32_t output_offset = data.op_data.output_zero_point;
    const int pad_width = data.op_data.padding.width;
    const int pad_height = data.op_data.padding.height;

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = input_depth * depth_multiplier;
    const int batch_size = input_shape.Dims(0);

    const int32_t activation_min = data.op_data.output_activation_min;
    const int32_t activation_max = data.op_data.output_activation_max;

    void* scratch_buf = nullptr;
    if (data.buffer_idx > -1) {
      scratch_buf = context->GetScratchBuffer(context, data.buffer_idx);
    }
    esp_nn_set_depthwise_conv_scratch_buf(scratch_buf);

    const int input_size = input_width * input_height * input_depth;
    const int output_size = output_width * output_height * output_depth;

    data_dims_t in_dims = {input_width, input_height, input_depth, 1};
    data_dims_t out_dims = {output_width, output_height, output_depth, 1};
    data_dims_t filt_dims = {filter_width, filter_height,
                             filter->dims->data[3], 0};
    dw_conv_params_t conv_params = {
        input_offset, output_offset,
        depth_multiplier,
        {params.stride_width, params.stride_height},
        {pad_width, pad_height},
        {0, 0},
        {activation_min, activation_max}};
    quant_data_t quant_data = {
        data.op_data.per_channel_output_shift,
        data.op_data.per_channel_output_multiplier};

    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
      esp_nn_depthwise_conv_s8(&in_dims, input_data + i_batch * input_size,
                               &filt_dims, tflite::micro::GetTensorData<int8_t>(filter),
                               tflite::micro::GetOptionalTensorData<int32_t>(bias),
                               &out_dims, output_data + i_batch * output_size,
                               &conv_params, &quant_data);
    }
  } else {
    tflite::reference_integer_ops::DepthwiseConvPerChannel(
        DwConvParamsQuantized(params, data.op_data),
        data.op_data.per_channel_output_multiplier,
        data.op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetOptionalTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D_ESPNN() {
  return tflite::micro::RegisterOp(ConvInit, ConvPrepare, ConvEval);
}

TfLiteRegistration Register_DEPTHWISE_CONV_2D_ESPNN() {
  return tflite::micro::RegisterOp(DwConvInit, DwConvPrepare, DwConvEval);
}

// ---------------------------------------------------------------------
// [CHẨN ĐOÁN] Self-test số học: esp-nn vs reference integer ops.
// Dùng buffer ngẫu nhiên cố định (seed đơn giản), so sánh kết quả INT8.
// maxdiff = 0  -> kernel + glue bit-exact (đúng).
// maxdiff > 0  -> lỗi số học/kernel/glue — hệ thống sẽ nhận diện sai.
// ---------------------------------------------------------------------
static void FillRandomInt8(int8_t* buf, int n, int seed) {
  unsigned int s = (unsigned int)(seed * 2654435761u + 12345u);
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    buf[i] = (int8_t)((s >> 24) & 0xFF);
  }
}

void EspNnSelfTest() {
  Serial.println("[ESPNN-TEST] running self-test (esp-nn vs reference)...");

  // ======== HELPER: chạy 1 test case conv, trả về maxdiff esp-nn vs reference ========
  auto run_conv_case = [](const char* name, int inW, int inH, int inC,
                          int outC, int fW, int fH, int padW, int padH,
                          int sw, int sh, int seedBase,
                          int* out_scratch_size) -> int {
    const int kOutW = inW, kOutH = inH;  // SAME padding, stride 1 (giữ đơn giản)
    const int kInN = inW * inH * inC;
    const int kFN = outC * fW * fH * inC;
    const int kOutN = kOutW * kOutH * outC;

    // Buffer cấp phát PSRAM động (static array sẽ tràn DRAM 512KB)
    int8_t* in_buf = (int8_t*)heap_caps_malloc(kInN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* f_buf = (int8_t*)heap_caps_malloc(kFN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* out_e = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* out_r = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!in_buf || !f_buf || !out_e || !out_r) {
      Serial.printf("[ESPNN-TEST] %s PSRAM alloc FAIL\n", name);
      heap_caps_free(in_buf); heap_caps_free(f_buf); heap_caps_free(out_e); heap_caps_free(out_r);
      return -1;
    }
    static int32_t bias_buf[128], mult_buf[128], shift_buf[128];

    FillRandomInt8(in_buf, kInN, seedBase);
    FillRandomInt8(f_buf, kFN, seedBase + 1);
    FillRandomInt8((int8_t*)bias_buf, outC * 4, seedBase + 2);
    for (int i = 0; i < outC; ++i) {
static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
  }

  TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataDepthwiseConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type,
      &data->op_data));

  // Quyết định esp-nn sau khi có padding (shape cố định của node)
  data->use_espnn = DwUseEspNn(input->dims->data[3], params.depth_multiplier,
                               filter_width, filter_height,
                               data->op_data.padding.width,
                               data->op_data.padding.height);

  // esp-nn scratch CHỈ khi dùng esp-nn (reference không cần scratch)
  data->buffer_idx = -1;
  if (data->use_espnn) {
    data_dims_t input_dims = {input_width, input_height, input->dims->data[3], 1};
    data_dims_t output_dims = {output_width, output_height,
                               output->dims->data[3], 1};
    data_dims_t filter_dims = {filter_width, filter_height,
                               filter->dims->data[3], 0};
    dw_conv_params_t conv_params = {
        0, 0,
        params.depth_multiplier,
        {params.stride_width, params.stride_height},
        {data->op_data.padding.width, data->op_data.padding.height},
        {0, 0},
        {-128, 127}};
    int scratch_buf_size = esp_nn_get_depthwise_conv_scratch_size(
        &input_dims, &filter_dims, &output_dims, &conv_params);
    if (scratch_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, scratch_buf_size, &data->buffer_idx));
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus DwConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (node->inputs->size == 3)
          ? tflite::micro::GetEvalInput(context, node, tflite::kDepthwiseConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kDepthwiseConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const DwConvNodeData& data = *(static_cast<const DwConvNodeData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);

  if (data.use_espnn && params.dilation_width_factor == 1 &&
      params.dilation_height_factor == 1) {
    tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    tflite::RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    tflite::RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

    const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = -data.op_data.input_zero_point;
    const int32_t output_offset = data.op_data.output_zero_point;
    const int pad_width = data.op_data.padding.width;
    const int pad_height = data.op_data.padding.height;

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = input_depth * depth_multiplier;
    const int batch_size = input_shape.Dims(0);

    const int32_t activation_min = data.op_data.output_activation_min;
    const int32_t activation_max = data.op_data.output_activation_max;

    void* scratch_buf = nullptr;
    if (data.buffer_idx > -1) {
      scratch_buf = context->GetScratchBuffer(context, data.buffer_idx);
    }
    esp_nn_set_depthwise_conv_scratch_buf(scratch_buf);

    const int input_size = input_width * input_height * input_depth;
    const int output_size = output_width * output_height * output_depth;

    data_dims_t in_dims = {input_width, input_height, input_depth, 1};
    data_dims_t out_dims = {output_width, output_height, output_depth, 1};
    data_dims_t filt_dims = {filter_width, filter_height,
                             filter->dims->data[3], 0};
    dw_conv_params_t conv_params = {
        input_offset, output_offset,
        depth_multiplier,
        {params.stride_width, params.stride_height},
        {pad_width, pad_height},
        {0, 0},
        {activation_min, activation_max}};
    quant_data_t quant_data = {
        data.op_data.per_channel_output_shift,
        data.op_data.per_channel_output_multiplier};

    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
      esp_nn_depthwise_conv_s8(&in_dims, input_data + i_batch * input_size,
                               &filt_dims, tflite::micro::GetTensorData<int8_t>(filter),
                               tflite::micro::GetOptionalTensorData<int32_t>(bias),
                               &out_dims, output_data + i_batch * output_size,
                               &conv_params, &quant_data);
    }
  } else {
    tflite::reference_integer_ops::DepthwiseConvPerChannel(
        DwConvParamsQuantized(params, data.op_data),
        data.op_data.per_channel_output_multiplier,
        data.op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetOptionalTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

}  // namespace

static void FillRandomInt8(int8_t* buf, int n, int seed) {
  unsigned int s = (unsigned int)(seed * 2654435761u + 12345u);
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    buf[i] = (int8_t)((s >> 24) & 0xFF);
  }
}

void EspNnSelfTest() {
  auto run_conv_case = [](const char* name, int inW, int inH, int inC,
                          int outC, int fW, int fH, int padW, int padH,
                          int sw, int sh, int seedBase,
                          int* out_scratch_size) -> int {
    const int kOutW = (inW + sw - 1) / sw; const int kOutH = (inH + sh - 1) / sh;
    const int kInN = inW * inH * inC;
    const int kFN = outC * fW * fH * inC;
    const int kOutN = kOutW * kOutH * outC;
    int8_t* in_buf = (int8_t*)heap_caps_malloc(kInN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* f_buf = (int8_t*)heap_caps_malloc(kFN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* out_e = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* out_r = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!in_buf || !f_buf || !out_e || !out_r) {
      ESP_LOGE(TAG, "[ESPNN-TEST] %s PSRAM alloc FAIL", name);
      heap_caps_free(in_buf); heap_caps_free(f_buf); heap_caps_free(out_e); heap_caps_free(out_r);
      return -1;
    }
    static int32_t bias_buf[128], mult_buf[128], shift_buf[128];
    FillRandomInt8(in_buf, kInN, seedBase);
    FillRandomInt8(f_buf, kFN, seedBase + 1);
    FillRandomInt8((int8_t*)bias_buf, outC * 4, seedBase + 2);
    for (int i = 0; i < outC; ++i) { mult_buf[i] = 1073741824; shift_buf[i] = -8; }
    data_dims_t in_dims = {inW, inH, inC, 1};
    data_dims_t filt_dims = {fW, fH, inC, 0};
    data_dims_t out_dims = {kOutW, kOutH, outC, 1};
    conv_params_t cparams = {-3, 7, {sw, sh}, {padW, padH}, {0, 0}, {-128, 127}};
    quant_data_t qdata = {shift_buf, mult_buf};
    int scratch_size = esp_nn_get_conv_scratch_size(&in_dims, &filt_dims, &out_dims, &cparams);
    *out_scratch_size = scratch_size;
    void* scratch = nullptr;
    if (scratch_size > 0) {
      scratch = heap_caps_aligned_alloc(16, scratch_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
      if (!scratch) scratch = heap_caps_aligned_alloc(16, scratch_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      if (!scratch) { ESP_LOGE(TAG, "[ESPNN-TEST] %s malloc(%d) FAIL", name, scratch_size); return -1; }
    }
    esp_nn_set_conv_scratch_buf(scratch);
    memset(out_e, 0, kOutN);
    esp_nn_conv_s8(&in_dims, in_buf, &filt_dims, f_buf, bias_buf, &out_dims, out_e, &cparams, &qdata);
    esp_nn_set_conv_scratch_buf(nullptr);
    if (scratch) heap_caps_free(scratch);
    tflite::ConvParams ref_params;
    ref_params.input_offset = -3; ref_params.weights_offset = 0; ref_params.output_offset = 7;
    ref_params.output_multiplier = 1; ref_params.output_shift = 0; ref_params.padding_type = tflite::PaddingType::kSame;
    ref_params.padding_values.width = padW; ref_params.padding_values.height = padH;
    ref_params.stride_width = sw; ref_params.stride_height = sh; ref_params.dilation_width_factor = 1;
    ref_params.dilation_height_factor = 1; ref_params.quantized_activation_min = -128; ref_params.quantized_activation_max = 127;
    tflite::RuntimeShape in_shape(4, (const int32_t[]){1, inH, inW, inC});
    tflite::RuntimeShape f_shape(4, (const int32_t[]){outC, fH, fW, inC});
    tflite::RuntimeShape out_shape(4, (const int32_t[]){1, kOutH, kOutW, outC});
    tflite::RuntimeShape b_shape(1, (const int32_t[]){outC});
    memset(out_r, 0, kOutN);
    tflite::reference_integer_ops::ConvPerChannel(ref_params, mult_buf, shift_buf, in_shape, in_buf, f_shape, f_buf, b_shape, bias_buf, out_shape, out_r);
    int maxdiff = 0;
    for (int i = 0; i < kOutN; ++i) { int d = abs((int)out_e[i] - (int)out_r[i]); if (d > maxdiff) maxdiff = d; }
    ESP_LOGI(TAG, "[ESPNN-TEST] conv %s (in=%dx%dx%d f=%dx%d out=%d) maxdiff=%d scratch=%d", name, inW, inH, inC, fW, fH, outC, maxdiff, scratch_size);
    heap_caps_free(in_buf); heap_caps_free(f_buf); heap_caps_free(out_e); heap_caps_free(out_r);
    return maxdiff;
  };
  auto run_dw_case = [](const char* name, int inW, int inH, int C,
                        int fW, int fH, int padW, int padH, int seedBase,
                        int* out_scratch_size) -> int {
    const int kInN = inW * inH * C;
    const int kFN = C * fW * fH;
    const int kOutN = kInN;
    int8_t* dw_in = (int8_t*)heap_caps_malloc(kInN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* dw_f = (int8_t*)heap_caps_malloc(kFN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* dw_e = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* dw_r = (int8_t*)heap_caps_malloc(kOutN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!dw_in || !dw_f || !dw_e || !dw_r) {
      ESP_LOGE(TAG, "[ESPNN-TEST] %s PSRAM alloc FAIL", name);
      heap_caps_free(dw_in); heap_caps_free(dw_f); heap_caps_free(dw_e); heap_caps_free(dw_r);
      return -1;
    }
    static int32_t dw_bias[128], dw_mult[128], dw_shift[128];
    FillRandomInt8(dw_in, kInN, seedBase);
    FillRandomInt8(dw_f, kFN, seedBase + 1);
    FillRandomInt8((int8_t*)dw_bias, C * 4, seedBase + 2);
    for (int i = 0; i < C; ++i) { dw_mult[i] = 1073741824; dw_shift[i] = -8; }
    data_dims_t in_dims = {inW, inH, C, 1};
    data_dims_t f_dims = {fW, fH, C, 0};
    data_dims_t out_dims = {inW, inH, C, 1};
    dw_conv_params_t dparams = {-3, 7, 1, {1, 1}, {padW, padH}, {0, 0}, {-128, 127}};
    quant_data_t q = {dw_shift, dw_mult};
    int scratch = esp_nn_get_depthwise_conv_scratch_size(&in_dims, &f_dims, &out_dims, &dparams);
    *out_scratch_size = scratch;
    void* scr = nullptr;
    if (scratch > 0) {
      scr = heap_caps_aligned_alloc(16, scratch, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
      if (!scr) scr = heap_caps_aligned_alloc(16, scratch, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      if (!scr) { ESP_LOGE(TAG, "[ESPNN-TEST] %s malloc(%d) FAIL", name, scratch); return -1; }
    }
    esp_nn_set_depthwise_conv_scratch_buf(scr);
    memset(dw_e, 0, kOutN);
    esp_nn_depthwise_conv_s8(&in_dims, dw_in, &f_dims, dw_f, dw_bias, &out_dims, dw_e, &dparams, &q);
    esp_nn_set_depthwise_conv_scratch_buf(nullptr);
    if (scr) heap_caps_free(scr);
    tflite::DepthwiseParams rp;
    rp.input_offset = -3; rp.weights_offset = 0; rp.output_offset = 7; rp.output_multiplier = 1; rp.output_shift = 0;
    rp.padding_type = tflite::PaddingType::kSame; rp.padding_values.width = padW; rp.padding_values.height = padH;
    rp.stride_width = 1; rp.stride_height = 1; rp.dilation_width_factor = 1; rp.dilation_height_factor = 1;
    rp.depth_multiplier = 1; rp.quantized_activation_min = -128; rp.quantized_activation_max = 127;
    tflite::RuntimeShape in_shape(4, (const int32_t[]){1, inH, inW, C});
    tflite::RuntimeShape f_shape(4, (const int32_t[]){1, fH, fW, C});
    tflite::RuntimeShape out_shape(4, (const int32_t[]){1, inH, inW, C});
    tflite::RuntimeShape b_shape(1, (const int32_t[]){C});
    memset(dw_r, 0, kOutN);
    tflite::reference_integer_ops::DepthwiseConvPerChannel(rp, dw_mult, dw_shift, in_shape, dw_in, f_shape, dw_f, b_shape, dw_bias, out_shape, dw_r);
    int maxdiff = 0;
    for (int i = 0; i < kOutN; ++i) { int d = abs((int)dw_e[i] - (int)dw_r[i]); if (d > maxdiff) maxdiff = d; }
    ESP_LOGI(TAG, "[ESPNN-TEST] dw %s (in=%dx%dx%d f=%dx%d) maxdiff=%d scratch=%d", name, inW, inH, C, fW, fH, maxdiff, scratch);
    heap_caps_free(dw_in); heap_caps_free(dw_f); heap_caps_free(dw_e); heap_caps_free(dw_r);
    return maxdiff;
  };
  int cs1 = 0, cs2 = 0, cs3 = 0, cs4 = 0, ds1 = 0;
  int d1 = run_conv_case("im2col", 8, 8, 4, 8, 3, 3, 1, 1, 1, 1, 10, &cs1);
  int d2 = run_conv_case("padded", 8, 8, 8, 16, 3, 3, 1, 1, 1, 1, 20, &cs2);
  int d1x1 = run_conv_case("1x1_16", 16, 16, 16, 16, 1, 1, 0, 0, 1, 1, 15, &cs3);
  int d1x1s = run_conv_case("1x1_1ch", 16, 16, 1, 16, 1, 1, 0, 0, 1, 1, 16, &cs4);
  int d3 = run_dw_case("opt", 8, 8, 6, 3, 3, 1, 1, 30, &ds1);
  int conv_maxdiff = std::max({d1, d2, d1x1, d1x1s, 0});
  int dw_maxdiff = std::max(d3, 0);
  bool fail = (d1 != 0) || (d2 != 0) || (d1x1 != 0) || (d1x1s != 0) || (d3 != 0);
  ESP_LOGI(TAG, "[ESPNN-TEST] SUMMARY conv_maxdiff=%d dwconv_maxdiff=%d %s", conv_maxdiff, dw_maxdiff, fail ? "FAIL!!!" : "PASS");
}

}  // namespace ai_esp_nn
#endif  // HAS_TFLM_HEADERS && AI_ESP_NN_CONV_ENABLED
