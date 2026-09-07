"""
Knowledge Distillation & Teacher-Student Learning Module for TinyDriverNet.
Transfers representational knowledge from a large, high-capacity Vision Teacher
(Google MediaPipe Face Mesh / Pretrained Landmark Expert) to the
compact TinyDriver-LandmarkNet running on ESP32-S3.
"""

import sys
import numpy as np
import cv2

try:
    import tensorflow as tf
    _BaseModel = tf.keras.Model
except ImportError:
    tf = None
    _BaseModel = object

# Mapping from MediaPipe 468/478-point Face Mesh to Project 13 (22 Landmarks)
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

# Standard Anthropometric 3D Human Face Model in millimeters for PnP
FACE_3D_MODEL = np.array([
    [  0.0,   0.0,   0.0],    # 0: Nose Tip (Index 19)
    [  0.0,  65.0, -35.0],    # 1: Chin (Index 21: +Y down)
    [-43.0, -32.0, -30.0],    # 2: Left eye outer (Index 0)
    [ 43.0, -32.0, -30.0],    # 3: Right eye outer (Index 9)
    [-30.0,  30.0, -20.0],    # 4: Mouth left (Index 12)
    [ 30.0,  30.0, -20.0]     # 5: Mouth right (Index 13)
], dtype=np.float64)


def estimate_pose_from_landmarks(pts_22_norm):
    """
    Estimates 3D Head Pose (Yaw, Pitch, Roll in radians) from 22 normalized landmarks.
    """
    pts_px = pts_22_norm.copy() * 96.0
    pts_2d = np.array([
        pts_px[19],
        pts_px[21],
        pts_px[0],
        pts_px[9],
        pts_px[12],
        pts_px[13]
    ], dtype=np.float64)

    focal_length = 96.0 * 1.1
    center = (48.0, 48.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return np.zeros(3, dtype=np.float32)

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) / 90.0  # Normalize to approx [-1.0, 1.0]
    yaw = float(angles[1]) / 90.0
    roll = float(angles[2]) / 90.0
    return np.array([yaw, pitch, roll], dtype=np.float32)


class MediaPipeTeacher:
    """
    Teacher Model wrapper using Google MediaPipe Face Mesh / Face Landmarker.
    Extracts high-fidelity 22-point biometric soft labels from arbitrary face images.
    Supports both classic MediaPipe solutions and MediaPipe tasks (1.0+).
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.available = False
        self.face_mesh = None
        self.task_landmarker = None
        self.use_tasks_api = False

        import importlib
        importlib.invalidate_caches()

        # 1. Direct import from mediapipe.python.solutions
        mp_face_mesh = None
        try:
            from mediapipe.python.solutions import face_mesh as fm
            mp_face_mesh = fm
        except Exception:
            pass

        # 2. Via mediapipe.solutions
        if mp_face_mesh is None:
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                    mp_face_mesh = mp.solutions.face_mesh
                else:
                    from mediapipe import solutions
                    mp_face_mesh = getattr(solutions, 'face_mesh', None)
            except Exception:
                pass

        if mp_face_mesh is not None:
            try:
                self.mp_face_mesh = mp_face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.available = True
                print("[Teacher] MediaPipe Face Mesh Teacher da san sang (468 -> 22 Landmarks)!")
                return
            except Exception as e:
                print(f"[Teacher] Loi khoi tao FaceMesh classic: {e}")

        # 3. Via MediaPipe Tasks
        try:
            import os
            import urllib.request
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            task_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
            if not os.path.exists(task_model_path):
                print("[Teacher] Dang dong bo mo hinh MediaPipe Tasks Face Landmarker (~3.7MB)...")
                task_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                req = urllib.request.Request(task_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as resp, open(task_model_path, "wb") as f:
                    f.write(resp.read())

            base_options = mp_python.BaseOptions(model_asset_path=task_model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence
            )
            self.task_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.use_tasks_api = True
            self.available = True
            print("[Teacher] MediaPipe Tasks FaceLandmarker Teacher da san sang (478 -> 22 Landmarks)!")
        except Exception as e:
            print(f"[Teacher] Khong khoi tao duoc MediaPipe Tasks ({e}). Chuyen sang che do Autonomous Student.")

    def extract_22_landmarks(self, image_rgb):
        """
        Extracts 22 normalized landmarks from an RGB image.
        Returns:
            np.ndarray shape (22, 2) in [0.0, 1.0] or None if face not found.
        """
        if not self.available:
            return None

        if self.use_tasks_api and self.task_landmarker is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                res = self.task_landmarker.detect(mp_image)
                if not res.face_landmarks:
                    return None
                face_landmarks = res.face_landmarks[0]
                pts_22 = []
                for idx in MEDIAPIPE_TO_22_INDICES:
                    pt = face_landmarks[idx]
                    pts_22.append([pt.x, pt.y])
                return np.array(pts_22, dtype=np.float32)
            except Exception:
                return None

        if self.face_mesh is not None:
            try:
                results = self.face_mesh.process(image_rgb)
                if not results.multi_face_landmarks:
                    return None

                face_landmarks = results.multi_face_landmarks[0]
                pts_22 = []
                for idx in MEDIAPIPE_TO_22_INDICES:
                    pt = face_landmarks.landmark[idx]
                    pts_22.append([pt.x, pt.y])

                return np.array(pts_22, dtype=np.float32)
            except Exception:
                return None

        return None


class PFLDMultiTaskModel(_BaseModel):
    """
    Keras Model wrapper handling Multi-Task Training:
      - Primary Task: 22 Facial Landmarks (AWing + Geometric EAR/MAR Loss)
      - Auxiliary Task: 3D Head Pose Estimation (MSE Loss on Yaw, Pitch, Roll)
    During inference or TFLite export, the landmark head is extracted directly.
    """
    def __init__(self, full_model, landmark_loss_fn, pose_weight=15.0, **kwargs):
        super(PFLDMultiTaskModel, self).__init__(**kwargs)
        self.model = full_model
        self.landmark_loss_fn = landmark_loss_fn
        self.pose_loss_fn = tf.keras.losses.MeanSquaredError()
        self.pose_weight = float(pose_weight)

    def train_step(self, data):
        # Unpack data: x is image, y is dict {"landmarks_output": ..., "pose_output": ...}
        # or tuple (y_landmarks, y_pose)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            x, y = data
        else:
            x = data
            y = None

        if isinstance(y, dict):
            y_lm = y["landmarks_output"]
            y_pose = y["pose_output"]
        elif isinstance(y, (list, tuple)):
            y_lm = y[0]
            y_pose = y[1]
        else:
            y_lm = y
            y_pose = None

        with tf.GradientTape() as tape:
            preds = self.model(x, training=True)
            if isinstance(preds, dict):
                pred_lm = preds["landmarks_output"]
                pred_pose = preds.get("pose_output", None)
            elif isinstance(preds, (list, tuple)):
                pred_lm = preds[0]
                pred_pose = preds[1] if len(preds) > 1 else None
            else:
                pred_lm = preds
                pred_pose = None

            # Primary landmark loss
            lm_loss = self.landmark_loss_fn(y_lm, pred_lm)

            # Auxiliary pose loss
            if pred_pose is not None and y_pose is not None:
                pose_loss = self.pose_loss_fn(y_pose, pred_pose)
                total_loss = lm_loss + self.pose_weight * pose_loss
            else:
                pose_loss = tf.constant(0.0)
                total_loss = lm_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        results = {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "pose_loss": pose_loss
        }
        return results

    def test_step(self, data):
        if isinstance(data, (list, tuple)) and len(data) == 2:
            x, y = data
        else:
            x = data
            y = None

        if isinstance(y, dict):
            y_lm = y["landmarks_output"]
            y_pose = y["pose_output"]
        elif isinstance(y, (list, tuple)):
            y_lm = y[0]
            y_pose = y[1]
        else:
            y_lm = y
            y_pose = None

        preds = self.model(x, training=False)
        if isinstance(preds, dict):
            pred_lm = preds["landmarks_output"]
            pred_pose = preds.get("pose_output", None)
        elif isinstance(preds, (list, tuple)):
            pred_lm = preds[0]
            pred_pose = preds[1] if len(preds) > 1 else None
        else:
            pred_lm = preds
            pred_pose = None

        lm_loss = self.landmark_loss_fn(y_lm, pred_lm)
        if pred_pose is not None and y_pose is not None:
            pose_loss = self.pose_loss_fn(y_pose, pred_pose)
            total_loss = lm_loss + self.pose_weight * pose_loss
        else:
            pose_loss = tf.constant(0.0)
            total_loss = lm_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "pose_loss": pose_loss
        }

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
