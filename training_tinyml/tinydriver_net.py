"""
TinyDriver-LandmarkNet: Ultra-lightweight Neural Network Architecture
Optimized for ESP32-S3 N16R8 Edge AI ADAS (TFLite Micro + ESP-NN).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
try:
    from config import INPUT_SHAPE, OUTPUT_DIMS
except (ImportError, ValueError):
    from .config import INPUT_SHAPE, OUTPUT_DIMS

def _inverted_res_block(x, expansion, filters, stride, block_id):
    """
    MobileNetV3 Inverted Residual Block with Squeeze-and-Excitation (SE).
    Fully compatible with TFLite Micro INT8 kernels.
    """
    in_channels = x.shape[-1]
    prefix = f"inv_res_{block_id}_"

    shortcut = x

    # 1. Expansion (1x1 Conv)
    if expansion != 1:
        expanded_channels = in_channels * expansion
        x = layers.Conv2D(expanded_channels, kernel_size=1, padding='same',
                          use_bias=False, name=f"{prefix}expand")(x)
        x = layers.BatchNormalization(name=f"{prefix}expand_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"{prefix}expand_relu")(x)
    else:
        expanded_channels = in_channels

    # 2. Depthwise Convolution (3x3 Depthwise)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same',
                               use_bias=False, name=f"{prefix}depthwise")(x)
    x = layers.BatchNormalization(name=f"{prefix}depthwise_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{prefix}depthwise_relu")(x)

    # 3. Squeeze and Excitation (SE) block (Channel Attention)
    se_squeeze = layers.GlobalAveragePooling2D(name=f"{prefix}se_gap")(x)
    se_reduced = layers.Dense(max(1, expanded_channels // 4), activation='relu',
                              name=f"{prefix}se_dense1")(se_squeeze)
    se_excite = layers.Dense(expanded_channels, activation='hard_sigmoid',
                             name=f"{prefix}se_dense2")(se_reduced)
    se_excite = layers.Reshape((1, 1, expanded_channels), name=f"{prefix}se_reshape")(se_excite)
    x = layers.Multiply(name=f"{prefix}se_mult")([x, se_excite])

    # 4. Pointwise Projection (1x1 Conv)
    x = layers.Conv2D(filters, kernel_size=1, padding='same',
                      use_bias=False, name=f"{prefix}project")(x)
    x = layers.BatchNormalization(name=f"{prefix}project_bn")(x)

    # 5. Residual connection if stride == 1 and channel counts match
    if stride == 1 and in_channels == filters:
        x = layers.Add(name=f"{prefix}add")([shortcut, x])

    return x


def build_tinydriver_net(input_shape=INPUT_SHAPE, output_dims=OUTPUT_DIMS):
    """
    Constructs the TinyDriverNet Model.

    Input:  (96, 96, 1) Grayscale image normalized to [-1.0, 1.0]
    Output: (44,) 22 keypoints (x, y) normalized to [0.0, 1.0] with Sigmoid activation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem: 3x3 Conv, stride 2 -> (48, 48, 16)
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    # Stage 1: (48, 48, 16)
    x = _inverted_res_block(x, expansion=1, filters=16, stride=1, block_id=1)

    # Stage 2: (24, 24, 24)
    x = _inverted_res_block(x, expansion=2, filters=24, stride=2, block_id=2)
    x = _inverted_res_block(x, expansion=2, filters=24, stride=1, block_id=3)

    # Stage 3: (12, 12, 32)
    x = _inverted_res_block(x, expansion=2, filters=32, stride=2, block_id=4)
    x = _inverted_res_block(x, expansion=2, filters=32, stride=1, block_id=5)

    # Stage 4: (6, 6, 48)
    x = _inverted_res_block(x, expansion=3, filters=48, stride=2, block_id=6)
    x = _inverted_res_block(x, expansion=3, filters=48, stride=1, block_id=7)

    # Feature Aggregation
    x = layers.Conv2D(96, kernel_size=1, padding='same', use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(max_value=6.0, name="head_relu")(x)

    # Spatial Feature Preservation Head (Bảo toàn không gian 2D cho mắt và miệng khi ngáp)
    # Thay vì GlobalAveragePooling2D làm mất hoàn toàn tọa độ trục Y của miệng khi ngáp,
    # ta giảm số kênh qua 1x1 Conv và duỗi phẳng (6 x 6 x 32 = 1,152 đặc trưng không gian):
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name="spatial_reduce_conv")(x)
    x = layers.BatchNormalization(name="spatial_reduce_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_reduce_relu")(x)
    x = layers.Flatten(name="spatial_flatten")(x)

    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.15, name="dropout")(x)

    # Output: 44 values normalized strictly to [0.0, 1.0] via Sigmoid
    outputs = layers.Dense(output_dims, activation='sigmoid', name="landmarks_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="TinyDriver_LandmarkNet")
    return model


if __name__ == "__main__":
    # Test model summary and tensor shapes
    model = build_tinydriver_net()
    model.summary()
    print(f"\nModel successfully built! Input: {model.input_shape}, Output: {model.output_shape}")
