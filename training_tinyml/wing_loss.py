"""
Biometric-Weighted Wing Loss for Robust Facial Landmark Regression.
Reference: Feng et al., "Wing Loss for Robust Facial Landmark Localisation with
Convolutional Neural Networks", IEEE CVPR 2018.

Enhanced for Edge AI Driver Drowsiness Detection:
Applies adaptive biometric weights to prioritize eye landmarks (EAR) and mouth
landmarks (MAR) over rigid nose and chin anchor points.
"""

import numpy as np
import tensorflow as tf

# Default 22-point Biometric Weights (x2 coords per point = 44 values)
#   - 6 Left eye landmarks   (0-5,   indices  0-11): weight = 2.0 (EAR sensitivity)
#   - 6 Right eye landmarks  (6-11,  indices 12-23): weight = 2.0 (EAR sensitivity)
#   - 6 Mouth landmarks      (12-17, indices 24-35): weight = 1.8 (MAR yawn sensitivity)
#   - 4 Nose/Chin landmarks  (18-21, indices 36-43): weight = 1.0 (Head Pose solver)
DEFAULT_BIOMETRIC_WEIGHTS_44 = np.array(
    [2.0] * 12 +   # Left Eye:  6 pts * 2 coords
    [2.0] * 12 +   # Right Eye: 6 pts * 2 coords
    [1.8] * 12 +   # Mouth:     6 pts * 2 coords
    [1.0] * 8,     # Nose/Chin: 4 pts * 2 coords
    dtype=np.float32
)


class WingLoss(tf.keras.losses.Loss):
    """
    Biometric-Weighted Wing Loss implementation in TensorFlow / Keras.
    
    Penalizes small errors much more heavily than L1 / L2 loss, which is essential
    for micro-movements of eye lids (EAR) and mouth lips (MAR).
    """
    def __init__(self, w=10.0, epsilon=2.0, image_scale=96.0,
                 use_biometric_weights=True, name="wing_loss", **kwargs):
        super(WingLoss, self).__init__(name=name, **kwargs)
        self.w = float(w)
        self.epsilon = float(epsilon)
        self.image_scale = float(image_scale)
        self.use_biometric_weights = bool(use_biometric_weights)
        
        # Precompute constant C = w - w * ln(1 + w / epsilon)
        self.c = self.w - self.w * tf.math.log(1.0 + self.w / self.epsilon)
        
        # Biometric weighting vector (1, 44)
        if self.use_biometric_weights:
            self.weights_tensor = tf.constant(
                np.expand_dims(DEFAULT_BIOMETRIC_WEIGHTS_44, axis=0),
                dtype=tf.float32
            )
        else:
            self.weights_tensor = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground-truth coordinates normalized to [0.0, 1.0], shape (batch, 44)
            y_pred: Predicted coordinates normalized to [0.0, 1.0], shape (batch, 44)
        """
        # Scale normalized coordinates back to pixel space so that w and epsilon match the CVPR paper
        y_true_px = y_true * self.image_scale
        y_pred_px = y_pred * self.image_scale

        diff = y_true_px - y_pred_px
        abs_diff = tf.abs(diff)

        # Region 1: |diff| < w
        loss1 = self.w * tf.math.log(1.0 + abs_diff / self.epsilon)
        # Region 2: |diff| >= w
        loss2 = abs_diff - self.c

        elementwise_loss = tf.where(tf.less(abs_diff, self.w), loss1, loss2)

        # Apply Biometric Weights (Eyes x2.0, Mouth x1.5, Pose x1.0)
        if self.use_biometric_weights:
            elementwise_loss = elementwise_loss * self.weights_tensor

        # Sum loss over all landmark coordinates, then average across batch
        return tf.reduce_mean(tf.reduce_sum(elementwise_loss, axis=-1))

    def get_config(self):
        config = super(WingLoss, self).get_config()
        config.update({
            "w": self.w,
            "epsilon": self.epsilon,
            "image_scale": self.image_scale,
            "use_biometric_weights": self.use_biometric_weights
        })
        return config


# Alias for explicit semantic usage
BiometricWeightedWingLoss = WingLoss
