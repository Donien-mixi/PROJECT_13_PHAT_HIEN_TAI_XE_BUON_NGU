"""
Knowledge Distillation & Teacher-Student Learning Module for TinyDriverNet.
Transfers representational knowledge from a large, high-capacity Vision Teacher
(e.g., Google MediaPipe Face Mesh / Pretrained Landmark Expert) to the
ultra-compact TinyDriver-LandmarkNet (~120K params) running on ESP32-S3.
"""

import sys
import numpy as np
import tensorflow as tf

# Mapping from MediaPipe 468-point Face Mesh to Project 13 (22 Landmarks)
# Ordered as: Left Eye (6), Right Eye (6), Mouth (6), Nose & Chin (4)
MEDIAPIPE_TO_22_INDICES = [
    # Left Eye (6 landmarks)
    33,   # Left eye outer corner
    160,  # Left eye top-outer
    158,  # Left eye top-inner
    133,  # Left eye inner corner
    153,  # Left eye bottom-inner
    144,  # Left eye bottom-outer
    # Right Eye (6 landmarks)
    362,  # Right eye inner corner
    385,  # Right eye top-inner
    387,  # Right eye top-outer
    263,  # Right eye outer corner
    373,  # Right eye bottom-outer
    380,  # Right eye bottom-inner
    # Mouth (6 landmarks)
    61,   # Mouth left corner
    291,  # Mouth right corner
    0,    # Upper lip outer center
    17,   # Lower lip outer center
    13,   # Upper lip inner center
    14,   # Lower lip inner center
    # Nose and Chin (4 landmarks)
    168,  # Nasion (between eyebrows / top of nose bridge)
    1,    # Nose tip (Chóp mũi)
    2,    # Subnasale / Philtrum (Nhân trung)
    152   # Chin bottom (Chóp cằm)
]


class MediaPipeTeacher:
    """
    Teacher Model wrapper using Google MediaPipe Face Mesh.
    Extracts high-fidelity 22-point biometric soft labels from arbitrary face images.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.available = False
        self.face_mesh = None
        try:
            try:
                import mediapipe as mp
                solutions = getattr(mp, 'solutions', None)
                if solutions is None:
                    from mediapipe.python import solutions
            except (ImportError, AttributeError):
                try:
                    from mediapipe import solutions
                except Exception:
                    solutions = None

            if solutions is None or not hasattr(solutions, 'face_mesh'):
                raise AttributeError("Không tìm thấy solutions.face_mesh trong mediapipe")

            self.mp_face_mesh = solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.available = True
            print("[Teacher] MediaPipe Face Mesh Teacher đã sẵn sàng (468 -> 22 Landmarks)!")
        except Exception as e:
            print(f"[Teacher] MediaPipe không khả dụng ({e}). Chuyển sang chế độ Autonomous Student.")

    def extract_22_landmarks(self, image_rgb):
        """
        Extracts 22 normalized landmarks from an RGB image.
        Returns:
            np.ndarray shape (22, 2) in [0.0, 1.0] or None if face not found.
        """
        if not self.available or self.face_mesh is None:
            return None

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = image_rgb.shape[:2]

        pts_22 = []
        for idx in MEDIAPIPE_TO_22_INDICES:
            pt = face_landmarks.landmark[idx]
            pts_22.append([pt.x, pt.y])

        return np.array(pts_22, dtype=np.float32)


class DistillationModel(tf.keras.Model):
    """
    Keras Model subclass implementing Teacher-Student Knowledge Distillation.
    Combines Ground-Truth Biometric Wing Loss with Teacher Smooth L1 Soft Loss.
    """
    def __init__(self, student_model, teacher_model=None, alpha=0.35, **kwargs):
        super(DistillationModel, self).__init__(**kwargs)
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = float(alpha)  # Weight for distillation loss vs student loss
        self.distill_loss_fn = tf.keras.losses.MeanSquaredError()

    def compile(self, optimizer, loss_fn, metrics=None):
        super(DistillationModel, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = loss_fn

    def train_step(self, data):
        x, y = data

        # Teacher predictions (no gradient)
        teacher_pred = None
        if self.teacher is not None:
            teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student forward pass
            student_pred = self.student(x, training=True)

            # Compute primary student loss vs ground truth
            student_loss = self.student_loss_fn(y, student_pred)

            if teacher_pred is not None:
                # Distillation loss vs teacher soft-targets
                distill_loss = self.distill_loss_fn(teacher_pred, student_pred)
                total_loss = (1.0 - self.alpha) * student_loss + self.alpha * distill_loss
            else:
                distill_loss = tf.constant(0.0)
                total_loss = student_loss

        # Backpropagation
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": total_loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss
        })
        return results

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_pred)

        self.compiled_metrics.update_state(y, student_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": student_loss})
        return results

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)
