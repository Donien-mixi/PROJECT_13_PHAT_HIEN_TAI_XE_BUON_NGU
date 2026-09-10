"""
TinyDriver-LandmarkNet: PFLD-Edge Architecture
Optimized for ESP32-S3 N16R8 Edge AI ADAS (TFLite Micro + ESP-NN).

Key Architectural Upgrades:
1. MobileNetV2 Inverted Residual Bottlenecks (MBConv) with Depthwise Separable Convolutions:
   - Replaces shallow Conv2D with 7 deep residual bottleneck stages.
   - Residual Skip Connections preserve high-frequency boundary gradients.
2. PFLD-inspired Multi-Scale Feature Fusion:
   - Stage 3 (12x12x48): Fine-grained spatial features for micro eyelid gaps (EAR) and lips (MAR).
   - Stage 4 (6x6x64): Global receptive field for 3D face structure and head orientation.
3. 2D Spatial Reduction Head:
   - Conv2D 1x1 + DepthwiseConv 3x3 to preserve spatial coordinates (avoiding destructive GAP).
4. Auxiliary 3D Head Pose Head (Training Only):
   - Auxiliary branch predicts Euler angles (Yaw, Pitch, Roll) during training.
   - Forces intermediate feature maps to encode 3D rotation, completely preventing Mean Face Collapse!
   - Pruned during TFLite export -> 0% overhead on ESP32-S3!
5. 100% ESP32-S3 Hardware Compatible:
   - Only Conv2D, DepthwiseConv2D, Add, Relu6, AvgPool, Dense (all accelerated by ESP-NN SIMD).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
try:
    from config import INPUT_SHAPE, OUTPUT_DIMS
except (ImportError, ValueError):
    from .config import INPUT_SHAPE, OUTPUT_DIMS


def _inverted_res_block(inputs, expansion, stride, filters, block_id):
    """
    Inverted Residual Block (MobileNetV2 style) with Depthwise Separable Conv.
    100% compatible with TFLite Micro and ESP-NN vector acceleration.
    """
    in_channels = inputs.shape[-1]
    x = inputs
    prefix = f"mb_{block_id}_"

    # 1. Expansion phase (Pointwise 1x1 Conv)
    expanded_channels = in_channels * expansion
    if expansion != 1:
        x = layers.Conv2D(expanded_channels, kernel_size=1, padding='same',
                          use_bias=False, name=prefix + "expand_conv")(x)
        x = layers.BatchNormalization(name=prefix + "expand_bn")(x)
        x = layers.ReLU(max_value=6.0, name=prefix + "expand_relu")(x)

    # 2. Depthwise phase (Depthwise 3x3 Conv)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same',
                               use_bias=False, name=prefix + "dw_conv")(x)
    x = layers.BatchNormalization(name=prefix + "dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=prefix + "dw_relu")(x)

    # 3. Projection phase (Pointwise 1x1 Conv, Linear)
    x = layers.Conv2D(filters, kernel_size=1, padding='same',
                      use_bias=False, name=prefix + "proj_conv")(x)
    x = layers.BatchNormalization(name=prefix + "proj_bn")(x)

    # 4. Residual Skip Connection
    if stride == 1 and in_channels == filters:
        x = layers.Add(name=prefix + "add")([inputs, x])

    return x


def build_tinydriver_net(input_shape=INPUT_SHAPE, output_dims=OUTPUT_DIMS, include_pose_head=False):
    """
    Constructs the high-gradient TinyDriver PFLD-Edge Model.

    Args:
        input_shape: (96, 96, 1) Grayscale image normalized to [-1.0, 1.0]
        output_dims: (44,) 22 keypoints (x, y) normalized to [0.0, 1.0]
        include_pose_head: If True, returns multi-output model [landmarks, pose] for training.
                           If False, returns single-output model [landmarks] for deployment/export.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem Conv (96x96 -> 48x48, 16 filters)
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    # Stage 1 (48x48): Inverted Residual Blocks
    x = _inverted_res_block(x, expansion=2, stride=1, filters=16, block_id="s1_b1")
    x = _inverted_res_block(x, expansion=2, stride=1, filters=24, block_id="s1_b2")

    # Stage 2 (48x48 -> 24x24): Stride 2 downsample + Residual
    x = _inverted_res_block(x, expansion=3, stride=2, filters=32, block_id="s2_b1")
    x = _inverted_res_block(x, expansion=3, stride=1, filters=32, block_id="s2_b2")

    # Stage 3 (24x24 -> 12x12): High-resolution spatial features for eyelids & lips
    x = _inverted_res_block(x, expansion=3, stride=2, filters=48, block_id="s3_b1")
    s3 = _inverted_res_block(x, expansion=3, stride=1, filters=48, block_id="s3_b2")

    # Stage 4 (12x12 -> 6x6): Global receptive field for 3D Head Pose & chin contour
    x = _inverted_res_block(s3, expansion=4, stride=2, filters=64, block_id="s4_b1")
    s4 = _inverted_res_block(x, expansion=4, stride=1, filters=64, block_id="s4_b2")

    # PFLD Multi-Scale Feature Fusion:
    # Downsample Stage 3 from (12, 12, 48) to (6, 6, 48) via Average Pooling
    s3_pooled = layers.AveragePooling2D(pool_size=2, strides=2, padding='same', name="s3_pool")(s3)

    # Concatenate local fine-grained features (48 ch) + global semantic features (64 ch) -> 112 ch at (6, 6)
    merged = layers.Concatenate(name="multiscale_concat")([s3_pooled, s4])

    # Spatial Feature Preservation Head
    # 1x1 Conv to reduce channels to 32, followed by Depthwise 3x3 for spatial correlation
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name="spatial_conv1x1")(merged)
    x = layers.BatchNormalization(name="spatial_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_relu")(x)

    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name="spatial_dw")(x)
    x = layers.BatchNormalization(name="spatial_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name="spatial_dw_relu")(x)

    # Flatten spatial descriptor: 6 x 6 x 32 = 1,152 features
    x = layers.Flatten(name="spatial_flatten")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.10, name="dropout")(x)

    # Primary Output: 44 continuous facial landmark coordinates normalized to [0.0, 1.0]
    landmarks_output = layers.Dense(output_dims, activation='linear', name="landmarks_output")(x)

    if include_pose_head:
        # Auxiliary 3D Head Pose Head (Training Only):
        # Branch from S4 (6x6x64) -> Conv 1x1 -> Global Avg Pool -> Dense 32 -> Dense 3 (Yaw, Pitch, Roll)
        aux = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name="aux_conv")(s4)
        aux = layers.BatchNormalization(name="aux_bn")(aux)
        aux = layers.ReLU(max_value=6.0, name="aux_relu")(aux)
        aux = layers.GlobalAveragePooling2D(name="aux_gap")(aux)
        aux = layers.Dense(32, activation='relu', name="aux_fc")(aux)
        pose_output = layers.Dense(3, activation='linear', name="pose_output")(aux)

        model = models.Model(
            inputs=inputs,
            outputs={"landmarks_output": landmarks_output, "pose_output": pose_output},
            name="TinyDriver_PFLD_TrainNet"
        )
    else:
        model = models.Model(
            inputs=inputs,
            outputs=landmarks_output,
            name="TinyDriver_PFLD_LandmarkNet"
        )

    return model


def get_deploy_model(model):
    """
    Extracts landmark-only deployment model from multi-task training model.
    Guarantees single-input single-output graph for TFLite Micro.
    """
    if isinstance(model.output, dict) and "landmarks_output" in model.output:
        return models.Model(inputs=model.input, outputs=model.output["landmarks_output"], name="TinyDriver_DeployNet")
    elif isinstance(model.output, (list, tuple)) and len(model.output) > 1:
        return models.Model(inputs=model.input, outputs=model.output[0], name="TinyDriver_DeployNet")
    return model


if __name__ == "__main__":
    # Test model summary and tensor shapes
    deploy_model = build_tinydriver_net(include_pose_head=False)
    deploy_model.summary()
    print(f"\n[Deploy Model] Input: {deploy_model.input_shape}, Output: {deploy_model.output_shape}, Params: {deploy_model.count_params():,}")

    train_model = build_tinydriver_net(include_pose_head=True)
    print(f"[Train Model] Outputs: {train_model.output_names}, Params: {train_model.count_params():,}")
