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
# - Mí mắt di động (P1, P2, P4, P5, P7, P8, P10, P11): 5.0 (ép cực mạnh vi chuyển động chớp mắt/nhắm mắt)
# - Khóe mắt (P0, P3, P6, P9)                         : 3.0 (neo giữ hốc mắt)
# - Khóe miệng (P12, P13)                             : 3.5 (neo giữ khóe mép)
# - Môi trên (P14, P16)                               : 4.0 (bám sát vòm môi trên)
# - Môi dưới (P15, P17)                               : 5.5 (ép bám sát viền môi dưới khi hạ miệng)
# - Trục mũi (P18, P19, P20)                          : 1.5 (neo giữ trục đối xứng khuôn mặt)
# - Đáy cằm (P21)                                     : 5.5 (buộc cằm phải di chuyển đồng bộ theo hàm dưới khi ngáp)
POINT_WEIGHTS = [
    3.0, 4.5, 4.5, 3.0, 4.5, 4.5,  # P0..P5: Mắt trái (Mí trên P1,P2 và Mí dưới P4,P5 x4.5, Khóe mắt x3.0)
    3.0, 4.5, 4.5, 3.0, 4.5, 4.5,  # P6..P11: Mắt phải (Mí trên P7,P8 và Mí dưới P10,P11 x4.5, Khóe mắt x3.0)
    3.5, 3.5,                      # P12, P13: Khóe miệng trái & phải (x3.5)
    4.0, 5.0,                      # P14, P15: Môi trên ngoài (x4.0), Môi dưới ngoài (x5.0)
    4.0, 5.0,                      # P16, P17: Môi trên trong (x4.0), Môi dưới trong (x5.0)
    2.0, 2.0, 2.0,                 # P18, P19, P20: Nasion, Chóp mũi, Nhân trung (x2.0)
    4.5                            # P21: Đáy cằm (Gnathion bám theo hàm dưới khi ngáp x4.5)
]
BIOMETRIC_WEIGHTS_44 = np.array(
    [w for pt_w in POINT_WEIGHTS for w in (pt_w, pt_w)],
    dtype=np.float32
)


def _compute_point_distance(p1, p2):
    """Computes Euclidean distance between points p1 and p2 of shape (batch, 2)."""
    return tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=-1) + 1e-7)


def compute_tensor_ear(pts_44, min_width=0.05, detach_width=True):
    """
    [v2.3.0] Differentiable Eye Aspect Ratio (EAR) for both eyes from (batch, 44).
    Points:
      Left Eye: P0 (outer), P1 (top-out), P2 (top-in), P3 (inner), P4 (bot-in), P5 (bot-out)
      Right Eye: P6 (inner), P7 (top-in), P8 (top-out), P9 (outer), P10 (bot-out), P11 (bot-in)

    Vì sao khôi phục detach_width cho MẮT:
      - Mắt cần phản hồi nhắm/mở rất nhạy -> phải có loss tỉ lệ EAR MẠNH để kéo mí mắt.
      - Nếu không detach, gradient mẫu số 1/w^2 có thể kéo khóe mắt (P0,P3,P6,P9) co lại.
      - stop_gradient(w) chỉ cho gradient chảy vào mí (P1,P2,P4,P5) -> khóe mắt được
        bảo vệ bởi coordinate loss, mí mắt chịu trách nhiệm mở/đóng. Mẫu số vẫn được
        clamp để tuyệt đối không nổ gradient.
    Returns:
      ear_l, ear_r: shape (batch,)
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
    if detach_width:
        w_l = tf.stop_gradient(w_l)
    w_l = tf.maximum(w_l, min_width)
    h1_l = _compute_point_distance(p1, p5)
    h2_l = _compute_point_distance(p2, p4)
    ear_l = (h1_l + h2_l) / (2.0 * w_l)

    # Right Eye
    p6 = pts[:, 6, :]
    p7 = pts[:, 7, :]
    p8 = pts[:, 8, :]
    p9 = pts[:, 9, :]
    p10 = pts[:, 10, :]
    p11 = pts[:, 11, :]

    w_r = _compute_point_distance(p6, p9)
    if detach_width:
        w_r = tf.stop_gradient(w_r)
    w_r = tf.maximum(w_r, min_width)
    h1_r = _compute_point_distance(p7, p11)
    h2_r = _compute_point_distance(p8, p10)
    ear_r = (h1_r + h2_r) / (2.0 * w_r)

    return ear_l, ear_r


def compute_tensor_mar(pts_44, min_width=0.15):
    """
    [v2.2.0] Differentiable Mouth Aspect Ratio (MAR = (h_outer + h_inner) / (2*w_m)).
    Points:
      P12 (left corner), P13 (right corner), P14 (top outer), P15 (bot outer),
      P16 (top inner), P17 (bot inner)

    Mẫu số w_m = |P12-P13| được KẸP xuống `min_width` để chặn đạo hàm 1/w^2 nổ.
    Khi khóe miệng co cụm, mẫu số bị kẹp -> MAR không còn là "đường tắt" để hạ loss,
    buộc mạng phải định vị đúng P12/P13.
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

    w_m = tf.maximum(_compute_point_distance(p12, p13), min_width)
    h_inner = _compute_point_distance(p16, p17)
    h_outer = _compute_point_distance(p14, p15)

    mar = (h_outer + h_inner) / (2.0 * w_m)
    return mar


class AdaptiveBiometricWingLoss(tf.keras.losses.Loss):
    """
    [v2.3.0 - COORD + EAR-MANH + MAR-YEU]

    L_total = L_wing(toạ độ) + ear_w * L_EAR + mar_w * L_MAR
              + gap_w * L_lip_gap + width_w * L_mouth_width

    Triết lý (đồng thời sửa 2 lỗi đối lập):
      1. L_wing (hồi quy trực tiếp 44 toạ độ) là NỀN TẢNG.
      2. L_EAR MẠNH (mặc định 20) + stop_gradient mẫu số: gradient chỉ chảy vào MÍ MẮT
         (P1,P2,P4,P5,P7,P8,P10,P11) -> mắt phản hồi nhắm/mở nhạy; khóe mắt được bảo vệ
         bởi coordinate loss nên không co cụm. (Bản v2.2.0 để EAR=1.5 làm mắt bị nén dải.)
      3. L_MAR YẾU (1.5) + mẫu số KẸP (clamp) + KHÔNG stop_gradient: đây là điều đã sửa
         lỗi sập khóe miệng P12/P13. Ratio yếu nên không lấn át; clamp chặn nổ 1/w^2;
         L_mouth_width neo cứng P12-P13.
      4. Loại bỏ Focal bất đối xứng (nguồn xung gradient đột biến làm mạng "ăn gian").

    Lỗi bản cũ: EAR x25/MAR x20 (focal x2) khiến gradient tỉ lệ lấn át toạ độ; MAR dùng
    stop_gradient tạo thế cân bằng suy biến ép P12/P13 co cụm. Bản v2.2.0 hạ cả EAR xuống
    1.5 -> hết sập miệng nhưng mắt bị nén dải. v2.3.0 tách đúng vai trò: EAR mạnh, MAR yếu.
    """

    def __init__(self, w=10.0, epsilon=1.5, image_scale=96.0,
                 ear_weight=20.0, mar_weight=1.5, gap_weight=3.0, width_weight=5.0,
                 name="adaptive_biometric_wing_loss", **kwargs):
        super(AdaptiveBiometricWingLoss, self).__init__(name=name, **kwargs)
        self.w = float(w)
        self.epsilon = float(epsilon)
        self.image_scale = float(image_scale)
        self.ear_weight = float(ear_weight)
        self.mar_weight = float(mar_weight)
        self.gap_weight = float(gap_weight)
        self.width_weight = float(width_weight)

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
        # 1. PRIMARY: Base Wing Loss in pixel space (96x96)
        y_true_px = y_true * self.image_scale
        y_pred_px = y_pred * self.image_scale

        abs_diff = tf.abs(y_true_px - y_pred_px)
        loss1 = self.w * tf.math.log(1.0 + abs_diff / self.epsilon)
        loss2 = abs_diff - self.c
        elementwise_loss = tf.where(tf.less(abs_diff, self.w), loss1, loss2)

        weighted_elem_loss = elementwise_loss * self.weights_tensor
        coord_loss = tf.reduce_mean(tf.reduce_sum(weighted_elem_loss, axis=-1)) / 44.0

        # 2. AUXILIARY: EAR loss (MẠNH, mẫu số kẹp + detach bảo vệ khóe mắt)
        #    -> mí mắt (P1,P2,P4,P5...) chịu trách nhiệm nhắm/mở, cho phản hồi sinh trắc nhạy.
        ear_l_true, ear_r_true = compute_tensor_ear(y_true, detach_width=False)
        ear_l_pred, ear_r_pred = compute_tensor_ear(y_pred, detach_width=True)
        ear_loss = tf.reduce_mean(
            0.5 * (tf.abs(ear_l_true - ear_l_pred) + tf.abs(ear_r_true - ear_r_pred))
        )

        # 3. AUXILIARY: MAR loss (yếu, mẫu số kẹp, KHÔNG detach — tránh sập khóe miệng)
        mar_true = compute_tensor_mar(y_true)
        mar_pred = compute_tensor_mar(y_pred)
        mar_loss = tf.reduce_mean(tf.abs(mar_true - mar_pred))

        # 4. Direct Linear Lip Gap Loss (pixel space, gradient hằng số ổn định)
        pts_true_px = tf.reshape(y_true_px, [-1, 22, 2])
        pts_pred_px = tf.reshape(y_pred_px, [-1, 22, 2])
        gap_outer_true = _compute_point_distance(pts_true_px[:, 14, :], pts_true_px[:, 15, :])
        gap_outer_pred = _compute_point_distance(pts_pred_px[:, 14, :], pts_pred_px[:, 15, :])
        gap_inner_true = _compute_point_distance(pts_true_px[:, 16, :], pts_true_px[:, 17, :])
        gap_inner_pred = _compute_point_distance(pts_pred_px[:, 16, :], pts_pred_px[:, 17, :])
        gap_loss = tf.reduce_mean(tf.abs(gap_outer_true - gap_outer_pred) + tf.abs(gap_inner_true - gap_inner_pred)) / self.image_scale

        # 5. Mouth Width Anchor Loss (NEO CỨNG P12-P13, chống sập khóe miệng)
        w_mouth_true = _compute_point_distance(pts_true_px[:, 12, :], pts_true_px[:, 13, :])
        w_mouth_pred = _compute_point_distance(pts_pred_px[:, 12, :], pts_pred_px[:, 13, :])
        width_loss = tf.reduce_mean(tf.abs(w_mouth_true - w_mouth_pred)) / self.image_scale

        total_loss = (
            coord_loss +
            (self.ear_weight * ear_loss) +
            (self.mar_weight * mar_loss) +
            (self.gap_weight * gap_loss) +
            (self.width_weight * width_loss)
        )
        return total_loss

    def get_config(self):
        config = super(AdaptiveBiometricWingLoss, self).get_config()
        config.update({
            "w": self.w,
            "epsilon": self.epsilon,
            "image_scale": self.image_scale,
            "ear_weight": self.ear_weight,
            "mar_weight": self.mar_weight,
            "gap_weight": self.gap_weight,
            "width_weight": self.width_weight
        })
        return config


# Aliases for seamless imports
WingLoss = AdaptiveBiometricWingLoss
BiometricWeightedWingLoss = AdaptiveBiometricWingLoss
