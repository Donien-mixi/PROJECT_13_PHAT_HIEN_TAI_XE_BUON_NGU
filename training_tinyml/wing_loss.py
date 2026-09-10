"""
Biometric-Weighted Adaptive Wing Loss & Differentiable Geometric EAR/MAR Constraints.
Reference:
1. Feng et al., "Wing Loss for Robust Facial Landmark Localisation with CNNs", CVPR 2018.
2. Wang et al., "Adaptive Wing Loss for Robust Facial Landmark Detection", ICCV 2019.
3. Guo et al., "PFLD: A Practical Facial Landmark Detector", ICCV 2019.

Enhanced for Edge AI ADAS Driver Drowsiness Detection:
- High-Gradient Adaptive Wing Loss on 22 landmarks with Eyelid priority (x3.5) and Lip priority (x2.5).
- Differentiable Geometric Eye Aspect Ratio (EAR) Loss (forces eyelids to snap shut on blinking/microsleep).
- Differentiable Geometric Mouth Aspect Ratio (MAR) Loss (forces lips to open wide on yawning).
"""

import numpy as np
import tensorflow as tf

# Biometric weighting vector for 22 points (44 coordinates):
# Tối ưu hóa đặc thù cho Edge ADAS (phát hiện nhắm mắt ngủ gật & ngáp há miệng):
# - Mí mắt (P1, P2, P4, P5, P7, P8, P10, P11)  : 4.5 (nhạy cảm với vi chuyển động chớp mắt)
# - Khóe mắt (P0, P3, P6, P9)                  : 3.5 (neo giữ hốc mắt)
# - Khóe miệng (P12, P13) & Môi trên (P14, P16): 3.0 (neo giữ vòm miệng trên)
# - Môi dưới (P15, P17)                        : 5.0 (ép bám sát viền môi dưới khi hạ miệng)
# - Trục mũi (P18, P19, P20)                   : 2.0 (neo giữ trục đối xứng khuôn mặt)
# - Đáy cằm (P21)                              : 4.5 (buộc cằm phải di chuyển theo xương hàm dưới)
# Biometric weighting vector for 22 points (44 coordinates):
# Tối ưu hóa đặc thù cho Edge ADAS (phát hiện nhắm mắt ngủ gật & ngáp há miệng):
# - Mí mắt di động (P1, P2, P4, P5, P7, P8, P10, P11): 5.0 (ép cực mạnh vi chuyển động chớp mắt/nhắm mắt)
# - Khóe mắt (P0, P3, P6, P9)                         : 3.0 (neo giữ hốc mắt)
# - Khóe miệng (P12, P13) & Môi trên (P14, P16)       : 3.0 (neo giữ vòm miệng trên)
# - Môi dưới (P15, P17)                               : 5.0 (ép bám sát viền môi dưới khi hạ miệng)
# - Trục mũi (P18, P19, P20)                          : 1.5 (neo giữ trục đối xứng khuôn mặt)
# - Đáy cằm (P21)                                     : 3.5 (buộc cằm phải di chuyển theo xương hàm dưới)
POINT_WEIGHTS = [
    3.0, 5.0, 5.0, 3.0, 5.0, 5.0,  # P0..P5: Mắt trái (Mí trên P1,P2 và Mí dưới P4,P5 x5.0)
    3.0, 5.0, 5.0, 3.0, 5.0, 5.0,  # P6..P11: Mắt phải (Mí trên P7,P8 và Mí dưới P10,P11 x5.0)
    3.0, 3.0,                      # P12, P13: Khóe miệng trái & phải
    3.0, 5.0,                      # P14, P15: Môi trên ngoài, Môi dưới ngoài (P15 x5.0)
    3.0, 5.0,                      # P16, P17: Môi trên trong, Môi dưới trong (P17 x5.0)
    1.5, 1.5, 1.5,                 # P18, P19, P20: Nasion, Chóp mũi, Nhân trung
    3.5                            # P21: Đáy cằm (Gnathion bám theo hàm dưới khi ngáp)
]
BIOMETRIC_WEIGHTS_44 = np.array(
    [w for pt_w in POINT_WEIGHTS for w in (pt_w, pt_w)],
    dtype=np.float32
)


def _compute_point_distance(p1, p2):
    """Computes Euclidean distance between points p1 and p2 of shape (batch, 2)."""
    return tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=-1) + 1e-7)


def compute_tensor_ear(pts_44):
    """
    Computes Differentiable Eye Aspect Ratio (EAR) for both eyes from (batch, 44).
    Points:
      Left Eye: P0 (outer), P1 (top-out), P2 (top-in), P3 (inner), P4 (bot-in), P5 (bot-out)
      Right Eye: P6 (inner), P7 (top-in), P8 (top-out), P9 (outer), P10 (bot-out), P11 (bot-in)
    Returns:
      ear_left, ear_right: shape (batch,)
    """
    pts = tf.reshape(pts_44, [-1, 22, 2])

    # Left Eye
    p0 = pts[:, 0, :]
    p1 = pts[:, 1, :]
    p2 = pts[:, 2, :]
    p3 = pts[:, 3, :]
    p4 = pts[:, 4, :]
    p5 = pts[:, 5, :]

    w_l = _compute_point_distance(p0, p3)
    h1_l = _compute_point_distance(p1, p5)
    h2_l = _compute_point_distance(p2, p4)
    ear_l = (h1_l + h2_l) / (2.0 * w_l + 1e-5)

    # Right Eye
    p6 = pts[:, 6, :]
    p7 = pts[:, 7, :]
    p8 = pts[:, 8, :]
    p9 = pts[:, 9, :]
    p10 = pts[:, 10, :]
    p11 = pts[:, 11, :]

    w_r = _compute_point_distance(p6, p9)
    h1_r = _compute_point_distance(p7, p11)
    h2_r = _compute_point_distance(p8, p10)
    ear_r = (h1_r + h2_r) / (2.0 * w_r + 1e-5)

    return ear_l, ear_r


def compute_tensor_mar(pts_44):
    """
    Computes Differentiable Mouth Aspect Ratio (MAR) from (batch, 44).
    Points:
      P12 (left corner), P13 (right corner), P14 (top outer), P15 (bot outer),
      P16 (top inner), P17 (bot inner)
    Matches 100% with local_model_tester.py and ESP32 firmware calculation:
      MAR = (h_outer + h_inner) / (2.0 * w_m)
    Returns:
      mar: shape (batch,)
    """
    pts = tf.reshape(pts_44, [-1, 22, 2])

    p12 = pts[:, 12, :]
    p13 = pts[:, 13, :]
    p14 = pts[:, 14, :]
    p15 = pts[:, 15, :]
    p16 = pts[:, 16, :]
    p17 = pts[:, 17, :]

    w_m = _compute_point_distance(p12, p13)
    h_inner = _compute_point_distance(p16, p17)
    h_outer = _compute_point_distance(p14, p15)

    mar = (h_outer + h_inner) / (2.0 * w_m + 1e-5)
    return mar


class AdaptiveBiometricWingLoss(tf.keras.losses.Loss):
    """
    Combined PFLD-style Landmark Loss with Asymmetric Focal Penalties:
      L_total = L_Wing(landmarks)/44 + lambda_ear * L_Focal_EAR + lambda_mar * L_Focal_MAR
    """
    def __init__(self, w=10.0, epsilon=1.5, image_scale=96.0,
                 ear_weight=40.0, mar_weight=35.0,
                 name="adaptive_biometric_wing_loss", **kwargs):
        super(AdaptiveBiometricWingLoss, self).__init__(name=name, **kwargs)
        self.w = float(w)
        self.epsilon = float(epsilon)
        self.image_scale = float(image_scale)
        self.ear_weight = float(ear_weight)
        self.mar_weight = float(mar_weight)

        # Precompute Wing constant C = w - w * ln(1 + w / epsilon)
        self.c = self.w - self.w * tf.math.log(1.0 + self.w / self.epsilon)
        self.weights_tensor = tf.constant(
            np.expand_dims(BIOMETRIC_WEIGHTS_44, axis=0),
            dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        """
        y_true, y_pred: shape (batch, 44) normalized to [0.0, 1.0]
        """
        # 1. Base Wing Loss in pixel space (96x96)
        y_true_px = y_true * self.image_scale
        y_pred_px = y_pred * self.image_scale

        diff = y_true_px - y_pred_px
        abs_diff = tf.abs(diff)

        loss1 = self.w * tf.math.log(1.0 + abs_diff / self.epsilon)
        loss2 = abs_diff - self.c
        elementwise_loss = tf.where(tf.less(abs_diff, self.w), loss1, loss2)

        # Apply Biometric Weights & Chuẩn hóa chia 44.0 để Coord Loss ở mức hợp lý (~20.0)
        weighted_elem_loss = elementwise_loss * self.weights_tensor
        coord_loss = tf.reduce_mean(tf.reduce_sum(weighted_elem_loss, axis=-1)) / 44.0

        # 2. Geometric EAR Loss với Asymmetric Focal Weight (phạt gấp 3.0 lần khi nhắm mắt mà đoán mở)
        ear_l_true, ear_r_true = compute_tensor_ear(y_true)
        ear_l_pred, ear_r_pred = compute_tensor_ear(y_pred)
        focal_ear_l = tf.where(ear_l_true < 0.20, 3.0, 1.0)
        focal_ear_r = tf.where(ear_r_true < 0.20, 3.0, 1.0)
        ear_diff = focal_ear_l * tf.abs(ear_l_true - ear_l_pred) + focal_ear_r * tf.abs(ear_r_true - ear_r_pred)
        ear_loss = tf.reduce_mean(ear_diff)

        # 3. Geometric MAR Loss với Focal Weight (phạt gấp 2.5 lần khi ngáp mà đoán ngậm)
        mar_true = compute_tensor_mar(y_true)
        mar_pred = compute_tensor_mar(y_pred)
        focal_mar = tf.where(mar_true > 0.45, 2.5, 1.0)
        mar_diff = focal_mar * tf.abs(mar_true - mar_pred)
        mar_loss = tf.reduce_mean(mar_diff)

        # 4. Geometric Mouth Width Loss (Khóe miệng P12 - P13 chuẩn hóa)
        pts_true_px = tf.reshape(y_true_px, [-1, 22, 2])
        pts_pred_px = tf.reshape(y_pred_px, [-1, 22, 2])
        w_mouth_true = _compute_point_distance(pts_true_px[:, 12, :], pts_true_px[:, 13, :])
        w_mouth_pred = _compute_point_distance(pts_pred_px[:, 12, :], pts_pred_px[:, 13, :])
        mouth_w_diff = tf.abs(w_mouth_true - w_mouth_pred)
        mouth_w_loss = tf.reduce_mean(mouth_w_diff) / 96.0

        total_loss = coord_loss + (self.ear_weight * ear_loss) + (self.mar_weight * mar_loss) + (1.0 * mouth_w_loss)
        return total_loss

    def get_config(self):
        config = super(AdaptiveBiometricWingLoss, self).get_config()
        config.update({
            "w": self.w,
            "epsilon": self.epsilon,
            "image_scale": self.image_scale,
            "ear_weight": self.ear_weight,
            "mar_weight": self.mar_weight
        })
        return config


# Aliases for seamless imports
WingLoss = AdaptiveBiometricWingLoss
BiometricWeightedWingLoss = AdaptiveBiometricWingLoss
