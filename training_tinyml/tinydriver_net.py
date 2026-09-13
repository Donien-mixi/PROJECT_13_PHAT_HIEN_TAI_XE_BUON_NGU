"""
TinyDriver-LandmarkNet: PFLD-Edge-Final Architecture
====================================================
Bản CÂN BẰNG giữa độ chính xác (cần feature 48x48 cho khóe miệng/mí mắt) và
bộ nhớ (tensor arena đủ nhỏ để nằm trong SRAM nội ESP32-S3 -> esp-nn tốc độ cao).

BỐI CẢNH (đọc kỹ trước khi sửa):
  - Bản "gọn" trước đó bỏ hẳn tầng 48x48 -> NME Miệng BỊ KẸT TRẦN ~12.4% suốt 60 epoch
    (không thể cải thiện dù train lâu) -> trượt cổng <12%. Bản 266K cũ đạt 10.49%.
  - Bản này KHÔI PHỤC 2 khối 48x48 (expansion=1, 16 kênh - tốn ít bộ nhớ) + 2 block
    mỗi stage (24/12/6), đủ năng lực cho khóe miệng & mí mắt.

HỢP ĐỒNG DỮ LIỆU GIỮ NGUYÊN 100% (không đổi nếu chưa đồng bộ toàn hệ thống):
  - Input : (96, 96, 1) grayscale, (px-128)/128 -> INT8 scale 1/128, zp 0
  - Output: 44 giá trị = 22 landmarks (x, y) trong [0, 1], FLOAT32
  - Tên output: "landmarks_output" (+ "pose_output" cho nhánh phụ khi train)
  - Bộ op TFLM: Conv2D, DepthwiseConv2D, Add, ReLU6, AveragePooling2D,
    Concatenate, Reshape, Dense (+ Dequantize cho head float).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
try:
    from config import INPUT_SHAPE, OUTPUT_DIMS
except (ImportError, ValueError):
    from .config import INPUT_SHAPE, OUTPUT_DIMS


def _mbconv(inputs, expansion, stride, filters, block_id):
    """
    Inverted Residual (MobileNetV2), TFLite Micro / ESP-NN compatible.
    LUU Y: voi cac khoi downsample dung expansion=1 de khong sinh tensor
    expansion o do phan giai cao (48x48) lam phinh arena.
    """
    in_channels = inputs.shape[-1]
    x = inputs
    prefix = f"mb_{block_id}_"

    if expansion != 1:
        x = layers.Conv2D(in_channels * expansion, kernel_size=1, padding='same',
                          use_bias=False, name=prefix + "expand_conv")(x)
        x = layers.BatchNormalization(name=prefix + "expand_bn")(x)
        x = layers.ReLU(max_value=6.0, name=prefix + "expand_relu")(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same',
                               use_bias=False, name=prefix + "dw_conv")(x)
    x = layers.BatchNormalization(name=prefix + "dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=prefix + "dw_relu")(x)

    x = layers.Conv2D(filters, kernel_size=1, padding='same',
                      use_bias=False, name=prefix + "proj_conv")(x)
    x = layers.BatchNormalization(name=prefix + "proj_bn")(x)

    if stride == 1 and in_channels == filters:
        x = layers.Add(name=prefix + "add")([inputs, x])

    return x


def build_tinydriver_net(input_shape=INPUT_SHAPE, output_dims=OUTPUT_DIMS, include_pose_head=False):
    """
    Constructs TinyDriver PFLD-Edge-Final (accuracy/memory balanced).

    Args:
        input_shape: (96, 96, 1) Grayscale normalized to [-1.0, 1.0]
        output_dims: (44,) 22 keypoints (x, y) normalized to [0.0, 1.0]
        include_pose_head: If True -> multi-output [landmarks, pose] for training.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem: 96x96 -> 48x48, 16 kenh (giu do phan giai cao)
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)              # 48x48x16

    # Stage 1 (48x48): KHÔI PHỤC chi tiết cao tần cho khóe miệng P12/P13 + mí mắt.
    # expansion=1 giữ tensor 48x48 chỉ 16 kênh (~37 KB) -> arena vẫn nhỏ.
    x = _mbconv(x, expansion=1, stride=1, filters=16, block_id="s1_b1")
    x = _mbconv(x, expansion=1, stride=1, filters=16, block_id="s1_b2")

    # Stage 2 (48x48 -> 24x24)
    x = _mbconv(x, expansion=1, stride=2, filters=32, block_id="s2_b1")
    x = _mbconv(x, expansion=2, stride=1, filters=32, block_id="s2_b2")

    # Stage 3 (24x24 -> 12x12): đặc trưng khóe miệng / mí mắt mức trung
    x = _mbconv(x, expansion=1, stride=2, filters=48, block_id="s3_b1")
    s3 = _mbconv(x, expansion=2, stride=1, filters=48, block_id="s3_b2")

    # Stage 4 (12x12 -> 6x6): trường thu nhận toàn cục cho head pose
    x = _mbconv(s3, expansion=1, stride=2, filters=64, block_id="s4_b1")
    s4 = _mbconv(x, expansion=2, stride=1, filters=64, block_id="s4_b2")

    # PFLD Multi-Scale Feature Fusion
    s3_pooled = layers.AveragePooling2D(pool_size=2, strides=2, padding='same', name="s3_pool")(s3)
    merged = layers.Concatenate(name="multiscale_concat")([s3_pooled, s4])

    # Spatial Feature Preservation Head
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name="spatial_conv1x1")(merged)
    x = layers.BatchNormalization(name="spatial_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_relu")(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name="spatial_dw")(x)
    x = layers.BatchNormalization(name="spatial_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_dw_relu")(x)

    x = layers.Flatten(name="spatial_flatten")(x)                    # 6x6x32 = 1152
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.10, name="dropout")(x)

    landmarks_output = layers.Dense(output_dims, activation='linear', name="landmarks_output")(x)

    if include_pose_head:
        aux = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name="aux_conv")(s4)
        aux = layers.BatchNormalization(name="aux_bn")(aux)
        aux = layers.ReLU(max_value=6.0, name="aux_relu")(aux)
        aux = layers.GlobalAveragePooling2D(name="aux_gap")(aux)
        aux = layers.Dense(32, activation='relu', name="aux_fc")(aux)
        pose_output = layers.Dense(3, activation='linear', name="pose_output")(aux)

        model = models.Model(
            inputs=inputs,
            outputs={"landmarks_output": landmarks_output, "pose_output": pose_output},
            name="TinyDriver_PFLD_Final_TrainNet"
        )
    else:
        model = models.Model(
            inputs=inputs,
            outputs=landmarks_output,
            name="TinyDriver_PFLD_Final_LandmarkNet"
        )

    return model


def get_deploy_model(model):
    """Extracts landmark-only deployment model (single input/output for TFLite Micro)."""
    if isinstance(model.output, dict) and "landmarks_output" in model.output:
        return models.Model(inputs=model.input, outputs=model.output["landmarks_output"], name="TinyDriver_DeployNet")
    elif isinstance(model.output, (list, tuple)) and len(model.output) > 1:
        return models.Model(inputs=model.input, outputs=model.output[0], name="TinyDriver_DeployNet")
    return model


if __name__ == "__main__":
    deploy_model = build_tinydriver_net(include_pose_head=False)
    deploy_model.summary()
    print(f"\n[Deploy Model] Input: {deploy_model.input_shape}, Output: {deploy_model.output_shape}, Params: {deploy_model.count_params():,}")

    train_model = build_tinydriver_net(include_pose_head=True)
    print(f"[Train Model] Outputs: {train_model.output_names}, Params: {train_model.count_params():,}")
