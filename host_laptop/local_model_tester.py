#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🖥️ MODULE KIỂM THỬ MÔ HÌNH TINYML TRỰC TIẾP TRÊN LAPTOP (LOCAL MODEL TESTER)
=============================================================================
Đồ án 13: Phát hiện tài xế ngủ gật & mất tập trung (Edge AI ADAS)

Mục đích:
  Cho phép người dùng kiểm thử trực tiếp mô hình TFLite (tải về từ Google Colab)
  với Webcam thật của Laptop TRƯỚC KHI nạp sang vi điều khiển ESP32-S3!

Các tính năng nâng cấp V2:
  1. Tự động Dò & Bám khuôn mặt thời gian thực (Multi-cascade Face Tracker + EMA Smoothing).
     Khung hình 1:1 luôn bám sát khuôn mặt người lái, 22 điểm mốc sinh học bám dính
     theo cử động mắt, mũi, miệng, cằm thật.
  2. Giao diện Cyberpunk 2 màn hình (Widescreen 960x480):
     - Bên trái (640x480): Camera thô trực diện + 22 điểm mốc + 3D Pose vector,
       KHÔNG BỊ BẢNG ĐIỀU KHIỂN CHE MẶT.
     - Bên phải (320x480): Bảng điều khiển ADAS Telemetry chuyên nghiệp,
       kèm ô soi trực tiếp Tensor 96x96 Grayscale đầu vào AI thời gian thực.
  3. Thanh đo động trực quan (Dynamic Bar Gauge) cho EAR và MAR với vạch ngưỡng,
     đổi màu nhạy bén (Xanh lá -> Vàng -> Đỏ hú còi) kèm huy hiệu trạng thái.
  4. Giải góc quay đầu 3D Head Pose (PnP) chuẩn trục ảnh (RQDecomp3x3),
     nhìn thẳng = 0°, quay trái/phải phản hồi mượt mà không bị lỗi lộn 180°.
  5. Đồng bộ 100% logic với vi điều khiển ESP32-S3.
=============================================================================
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path
import numpy as np
import cv2

# Đảm bảo in tiếng Việt UTF-8 trên Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Âm thanh còi hú qua winsound trên Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Hỗ trợ nạp TFLite / LiteRT Interpreter
HAS_TFLITE = False
TFLiteInterpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
    HAS_TFLITE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        HAS_TFLITE = True
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
            HAS_TFLITE = True
        except ImportError:
            HAS_TFLITE = False

# Hỗ trợ nạp MediaPipe (Hỗ trợ cả Tasks Vision API v1.0+ và Solutions API cũ)
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# -----------------------------------------------------------------------------
# CẤU HÌNH THÔNG SỐ CHUẨN ĐỒNG BỘ VỚI PROJECT_CONFIG.JSON & ESP32-S3
# -----------------------------------------------------------------------------
INPUT_WIDTH = 96
INPUT_HEIGHT = 96
NUM_LANDMARKS = 22

# Chỉ số 22 điểm mốc
LEFT_EYE_PTS = [0, 1, 2, 3, 4, 5]
RIGHT_EYE_PTS = [6, 7, 8, 9, 10, 11]
MOUTH_PTS = [12, 13, 14, 15, 16, 17]
NASION_PT = 18
NOSE_TIP_PT = 19
NOSE_WING_PT = 20
CHIN_PT = 21

# Mô hình nhân trắc học 3D 6 điểm chuẩn cho PnP đồng bộ trục ảnh (+X sang phải, +Y hướng xuống, -Z hướng ra xa)
FACE_3D_MODEL = np.array([
    [  0.0,   0.0,   0.0],    # 0: Chóp mũi (Nose Tip P19)
    [  0.0,  65.0, -35.0],    # 1: Chóp cằm (Chin P21: +Y hướng xuống)
    [-43.0, -32.0, -30.0],    # 2: Khóe mắt ngoài bên trái (Index 0: -X, -Y hướng lên)
    [ 43.0, -32.0, -30.0],    # 3: Khóe mắt ngoài bên phải (Index 9: +X, -Y hướng lên)
    [-30.0,  30.0, -20.0],    # 4: Khóe miệng bên trái (Index 12: -X, +Y hướng xuống)
    [ 30.0,  30.0, -20.0]     # 5: Khóe miệng bên phải (Index 13: +X, +Y hướng xuống)
], dtype=np.float64)


def enhance_low_light(frame_bgr, target_luma=115.0):
    """
    Tự động cân bằng sáng và tăng cường tương phản trong điều kiện cabin thiếu sáng/ngược sáng/ban đêm.
    Áp dụng Adaptive Gamma Curve + CLAHE trên kênh Luminance (LAB), không gây bết màu hay chói lóa.
    """
    if frame_bgr is None:
        return frame_bgr
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))

    # Nếu độ sáng đã tốt (>= target_luma) thì giữ nguyên frame gốc để tối ưu FPS
    if mean_luma >= target_luma:
        return frame_bgr

    # 1. Tính hệ số Gamma nâng sáng vùng tối lên mức mục tiêu
    gamma = math.log(target_luma / 255.0) / math.log(max(mean_luma, 4.0) / 255.0)
    gamma = float(np.clip(gamma, 0.35, 0.90))

    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
    brightened = cv2.LUT(frame_bgr, lut)

    # 2. Nâng tương phản cục bộ các chi tiết mắt, mũi, miệng bằng CLAHE
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = 2.5 if mean_luma > 60.0 else 3.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


class MediaPipeLandmarkExtractor:
    """Bộ trích xuất 22 điểm mốc sinh học chuẩn mực từ MediaPipe (Hỗ trợ Tasks API 1.0+ & Solutions API cũ, thích ứng Low-Light)."""

    def __init__(self):
        self.is_loaded = False
        self.engine_type = None  # "TASKS" hoặc "SOLUTIONS"
        self.landmarker = None
        self.face_mesh = None

        # Ánh xạ chuẩn xác 22 điểm từ 468 điểm MediaPipe Face Mesh:
        # Mắt trái: 33 (ngoài), 160 (mí trên 1), 158 (mí trên 2), 133 (trong), 153 (mí dưới 2), 144 (mí dưới 1)
        # Mắt phải: 362 (trong), 385 (mí trên 1), 387 (mí trên 2), 263 (ngoài), 373 (mí dưới 2), 380 (mí dưới 1)
        # Miệng: 61 (khóe trái), 291 (khóe phải), 0 (môi trên ngoài), 17 (môi dưới ngoài), 13 (môi trên trong), 14 (môi dưới trong)
        # Mũi & Cằm: 168 (Nasion/gốc mũi), 1 (Chóp mũi), 2 (Nhân trung), 152 (Đáy cằm)
        self.indices = [
            33, 160, 158, 133, 153, 144,
            362, 385, 387, 263, 373, 380,
            61, 291, 0, 17, 13, 14,
            168, 1, 2, 152
        ]

        if not HAS_MEDIAPIPE:
            return

        # 1. Thử nạp MediaPipe Tasks Vision API (MediaPipe 1.0+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_candidates = [
                os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker.task'),
                os.path.join(os.path.dirname(__file__), 'face_landmarker.task'),
                os.path.join(os.path.dirname(__file__), '..', 'training_tinyml', 'face_landmarker.task'),
            ]
            task_path = next((p for p in task_candidates if os.path.exists(p)), None)
            if task_path:
                base_options = python.BaseOptions(model_asset_path=task_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.engine_type = "TASKS"
                self.is_loaded = True
                print(f"💎 [MediaPipe Engine] Đã nạp MediaPipe Tasks FaceLandmarker từ: {os.path.basename(task_path)}")
                return
        except Exception:
            pass

        # 2. Thử nạp MediaPipe Solutions Face Mesh (MediaPipe < 1.0)
        try:
            import mediapipe as mp
            mp_fm = None
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                mp_fm = mp.solutions.face_mesh
            elif hasattr(mp, 'face_mesh'):
                mp_fm = mp.face_mesh
            if mp_fm is not None:
                self.face_mesh = mp_fm.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.35
                )
                self.engine_type = "SOLUTIONS"
                self.is_loaded = True
                print("💎 [MediaPipe Engine] Đã nạp MediaPipe Solutions Face Mesh (Ground-Truth).")
                return
        except Exception:
            pass

    def extract(self, frame_bgr):
        """Trích xuất 22 điểm mốc và bounding box chuẩn từ frame webcam (Thích ứng tự động với bóng tối/ngược sáng)."""
        if not self.is_loaded:
            return None, None
        h, w = frame_bgr.shape[:2]

        if self.engine_type == "TASKS" and self.landmarker is not None:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)

            if not res.face_landmarks:
                enhanced_bgr = enhance_low_light(frame_bgr, target_luma=120.0)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_rgb)
                res = self.landmarker.detect(mp_img)

            if not res.face_landmarks:
                return None, None

            lms = res.face_landmarks[0]
            pts_px = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
            for i, idx in enumerate(self.indices):
                pts_px[i, 0] = lms[idx].x * w
                pts_px[i, 1] = lms[idx].y * h

            eye_y = float(np.mean(pts_px[:12, 1]))
            eye_x = float(np.mean(pts_px[:12, 0]))
            nose_x = float(pts_px[19, 0])
            nose_y = float(pts_px[19, 1])

            eye_l_center = np.mean(pts_px[:6, :], axis=0)
            eye_r_center = np.mean(pts_px[6:12, :], axis=0)
            eyes_center = (eye_l_center + eye_r_center) / 2.0
            eye_x = float(eyes_center[0])
            eye_y = float(eyes_center[1])
            d_eyes = float(np.linalg.norm(eye_r_center - eye_l_center))
            d_eye_nose = float(max(nose_y - eye_y, d_eyes * 0.45, 1.0))
            chin_y = float(pts_px[21, 1])
            d_eye_chin = float(max(chin_y - eye_y, d_eye_nose, 1.0))

            # [v2.0.5 SYNC] h_skull đồng bộ isomorphic_transform (có term cằm /1.20 tự giãn khi ngáp)
            h_skull = float(max(d_eye_nose * 2.10, d_eyes * 1.40, d_eye_chin / 1.20))
            face_size = float(round(h_skull * 2.05))
            face_cx = float((eye_x + nose_x) / 2.0)
            face_cy = float(eye_y + 0.32 * h_skull)
            return pts_px, (face_cx, face_cy, face_size)

        elif self.engine_type == "SOLUTIONS" and self.face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                enhanced_bgr = enhance_low_light(frame_bgr, target_luma=120.0)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(enhanced_rgb)

            if not results.multi_face_landmarks:
                return None, None

            lms = results.multi_face_landmarks[0].landmark
            pts_px = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
            for i, idx in enumerate(self.indices):
                pts_px[i, 0] = lms[idx].x * w
                pts_px[i, 1] = lms[idx].y * h

            eye_y = float(np.mean(pts_px[:12, 1]))
            eye_x = float(np.mean(pts_px[:12, 0]))
            nose_x = float(pts_px[19, 0])
            nose_y = float(pts_px[19, 1])

            eye_l_center = np.mean(pts_px[:6, :], axis=0)
            eye_r_center = np.mean(pts_px[6:12, :], axis=0)
            eyes_center = (eye_l_center + eye_r_center) / 2.0
            eye_x = float(eyes_center[0])
            eye_y = float(eyes_center[1])
            d_eyes = float(np.linalg.norm(eye_r_center - eye_l_center))
            d_eye_nose = float(max(nose_y - eye_y, d_eyes * 0.45, 1.0))
            chin_y = float(pts_px[21, 1])
            d_eye_chin = float(max(chin_y - eye_y, d_eye_nose, 1.0))

            # [v2.0.5 SYNC] h_skull đồng bộ isomorphic_transform (có term cằm /1.20 tự giãn khi ngáp)
            h_skull = float(max(d_eye_nose * 2.10, d_eyes * 1.40, d_eye_chin / 1.20))
            face_size = float(round(h_skull * 2.05))
            face_cx = float((eye_x + nose_x) / 2.0)
            face_cy = float(eye_y + 0.32 * h_skull)
            return pts_px, (face_cx, face_cy, face_size)

        return None, None

    def close(self):
        """Giải phóng tài nguyên MediaPipe an toàn khi thoát."""
        try:
            if self.landmarker is not None:
                self.landmarker.close()
                self.landmarker = None
        except Exception:
            pass
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None
        except Exception:
            pass


class OneEuroFilter:
    """
    Bộ lọc 1€ (One-Euro Filter) - Casiez et al., CHI 2012.
    Bộ lọc thích ứng tần số cắt theo vận tốc:
    - Khi đứng yên (vận tốc ~ 0): fc = min_cutoff (lọc cực mạnh, triệt tiêu 100% rung giật).
    - Khi chuyển động nhanh (xoay đầu): fc tăng theo beta * |v| (mở rộng băng thông, đáp ứng tức thì, ZERO LAG).
    """
    def __init__(self, t0=0.0, x0=0.0, min_cutoff=0.8, beta=0.025, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None or self.t_prev == 0.0:
            self.t_prev = t
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)

        dt = t - self.t_prev
        if dt <= 1e-5:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self):
        self.t_prev = 0.0
        self.dx_prev = 0.0


class FaceTracker:
    """
    Bộ theo dõi khuôn mặt thời gian thực:
    1. Bộ lọc One-Euro: Triệt tiêu rung giật khi đứng yên, tức thì theo kịp khi xoay đầu nhanh (Zero Lag).
    2. Deadband: Triệt tiêu nhiễu sensor ISO trong điều kiện thiếu sáng.
    3. Dò đa tầng Haar Multi-Cascade có CLAHE: Bắt dính khuôn mặt cả khi nghiêng đầu/đeo kính/thiếu sáng.
    4. Giữ quán tính an toàn (Inertial Hold): Tuyệt đối KHÔNG trôi dạt và KHÔNG co nhỏ bất thường.
    """

    def __init__(self, expansion_ratio=1.50, **kwargs):
        self.expansion_ratio = expansion_ratio
        # One-Euro Filter: min_cutoff=0.70 giữ ổn định khi đứng yên, beta=0.08 mở rộng tức thì khi cử động/xoay đầu (Zero-Lag)
        self.filter_cx = OneEuroFilter(min_cutoff=0.70, beta=0.080)
        self.filter_cy = OneEuroFilter(min_cutoff=0.70, beta=0.080)
        self.filter_S = OneEuroFilter(min_cutoff=0.50, beta=0.040)

        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_S = None
        self.hysteresis_pos = 1.5    # Ngưỡng trễ vị trí: dịch chuyển < 1.5px giữ nguyên box chống rung (nhạy bén theo chuyển động)
        self.hysteresis_scale = 0.020 # Ngưỡng trễ tỉ lệ: thay đổi scale < 2% giữ nguyên box chống giật
        self.last_detection_time = 0.0
        self.is_tracking = False
        self.enabled = True
        self.cascades = []

        candidate_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_alt2.xml'),
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt2.xml',
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt.xml',
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        ]

        if hasattr(cv2, 'CascadeClassifier'):
            for p in candidate_paths:
                if p and os.path.exists(p):
                    try:
                        cc = cv2.CascadeClassifier(p)
                        if not cc.empty():
                            self.cascades.append(cc)
                    except Exception:
                        pass

        if self.cascades:
            print(f"🎯 [Face Tracker] Đã nạp {len(self.cascades)} bộ dò Haar Cascade dự phòng.")
        else:
            print("⚠️ [Face Tracker] Không tìm thấy file Haar Cascade.")

    def _get_crop_canonical(self, frame, cx, cy, S):
        h, w = frame.shape[:2]
        default_S = min(w, h)
        # Giới hạn kích thước tối thiểu 160px để không bao giờ bị co nhỏ thành một chấm
        S = int(round(np.clip(S, 160, default_S)))
        if S % 2 != 0:
            S += 1

        # Kẹp chặt tâm cx, cy an toàn trong khung hình
        cx = float(np.clip(cx, S * 0.35, w - S * 0.35))
        cy = float(np.clip(cy, S * 0.35, h - S * 0.35))

        x1 = int(round(cx - S / 2.0))
        y1 = int(round(cy - S / 2.0))
        x2 = x1 + S
        y2 = y1 + S

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cropped = padded[y1 + pad_t:y2 + pad_t, x1 + pad_l:x2 + pad_l]
        else:
            cropped = frame[y1:y2, x1:x2]

        crop_box = (x1, y1, x2, y2)
        return cropped, crop_box, S, self.is_tracking

    def detect_face_haar(self, frame):
        """Dò tìm khuôn mặt bằng Haar Multi-Cascade với CLAHE cân bằng sáng."""
        if not self.enabled or not self.cascades:
            return None

        h, w = frame.shape[:2]
        default_S = min(w, h)
        enh_frame = enhance_low_light(frame, target_luma=115.0)
        gray = cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        for cc in self.cascades:
            faces = cc.detectMultiScale(
                gray_clahe, scaleFactor=1.06, minNeighbors=2, minSize=(50, 50)
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                # [v2.0.5 SYNC] Haar box -> mỏ neo Canonical (đồng bộ
                # compute_canonical_anchor của isomorphic_transform.py, có term cằm):
                # d_eyes ≈ 0.46*fw; d_eye_nose ≈ 0.45*d_eyes; d_eye_chin ≈ 2.2*d_eye_nose
                # -> h = max(0.45d*2.10, d*1.40, 2.2*0.45d/1.345) ≈ 1.40*d_eyes
                d_eyes_est = 0.46 * float(fw)
                h_skull = max(d_eyes_est * 0.45 * 2.10, d_eyes_est * 1.40,
                              d_eyes_est * 0.45 * 2.2 / 1.20)
                target_S = float(np.clip(h_skull * 2.05, 160, default_S))
                target_cx = float(fx + fw / 2.0)
                # eye_y ≈ fy + 0.40*fh; cy = eye_y + 0.32*h_skull (giống canonical crop)
                target_cy = float(fy + fh * 0.40 + 0.32 * h_skull)
                return (target_cx, target_cy, target_S)

        return None

    def get_target_from_landmarks(self, landmarks_px):
        """
        Tính toán tâm sọ mặt và kích thước chuẩn hóa trực tiếp từ 22 landmarks của frame trước
        (Vòng lặp Landmark-Driven Tracking chuẩn mực của MediaPipe Face Mesh).
        [v2.0.5 SYNC] Công thức mỏ neo Đồng bộ 1:1 với
        training_tinyml/isomorphic_transform.compute_canonical_anchor() (có term cằm
        d_eye_chin/1.345 — hộp tự giãn khi ngáp, không clip P21).
        """
        if landmarks_px is None or len(landmarks_px) != NUM_LANDMARKS:
            return None
        pts = np.array(landmarks_px, dtype=np.float32)
        eye_l_center = np.mean(pts[:6, :], axis=0)
        eye_r_center = np.mean(pts[6:12, :], axis=0)
        eyes_center = (eye_l_center + eye_r_center) / 2.0
        eye_x = float(eyes_center[0])
        eye_y = float(eyes_center[1])
        nose_x = float(pts[19, 0])
        nose_y = float(pts[19, 1])
        chin_y = float(pts[21, 1])

        d_eyes = float(np.linalg.norm(eye_r_center - eye_l_center))
        d_eye_nose = float(max(nose_y - eye_y, d_eyes * 0.45, 1.0))
        d_eye_chin = float(max(chin_y - eye_y, d_eye_nose, 1.0))
        h_skull = float(max(d_eye_nose * 2.10, d_eyes * 1.40, d_eye_chin / 1.20))
        face_size = float(round(h_skull * 2.05))
        face_cx = float((eye_x + nose_x) / 2.0)
        face_cy = float(eye_y + 0.32 * h_skull)
        return (face_cx, face_cy, face_size)

    def update_with_target(self, target_cx, target_cy, target_S, frame):
        """Cập nhật vị trí mặt từ bộ dò ngoài (MediaPipe / Haar) kết hợp deadband và lọc One-Euro."""
        h, w = frame.shape[:2]
        default_S = min(w, h)
        target_S = float(np.clip(target_S, 160, default_S))
        self.is_tracking = True
        now = time.time()
        self.last_detection_time = now

        if self.smooth_cx is not None and self.smooth_cy is not None and self.smooth_S is not None:
            # Hysteresis lọc rung micro-jitter: Nếu dịch chuyển tâm < 4.5px, giữ nguyên tọa độ cũ
            move_dist = math.hypot(target_cx - self.smooth_cx, target_cy - self.smooth_cy)
            if move_dist < self.hysteresis_pos:
                target_cx = self.smooth_cx
                target_cy = self.smooth_cy
            # Hysteresis kích thước: Nếu scale thay đổi < 4.5%, giữ nguyên kích thước cũ
            if abs(target_S - self.smooth_S) / max(self.smooth_S, 1.0) < self.hysteresis_scale:
                target_S = self.smooth_S

        self.smooth_cx = self.filter_cx.filter(target_cx, now)
        self.smooth_cy = self.filter_cy.filter(target_cy, now)
        self.smooth_S = self.filter_S.filter(target_S, now)
        return self._get_crop_canonical(frame, self.smooth_cx, self.smooth_cy, self.smooth_S)

    def update_idle(self, frame):
        """
        Xử lý khi frame hiện tại detector chưa thấy mặt:
        - Trong 1.5s: Giữ quán tính an toàn (Inertial Hold), giữ nguyên khung vàng tại chỗ, không trôi dạt.
        - Quá 1.5s: Nhẹ nhàng lùi về khung trung tâm người lái (Center Crop).
        """
        h, w = frame.shape[:2]
        now = time.time()
        default_S = float(min(w, h))

        if self.smooth_cx is not None and (now - self.last_detection_time < 1.5):
            self.is_tracking = True
        else:
            self.is_tracking = False
            center_cx = w / 2.0
            center_cy = h / 2.0
            center_S = default_S * 0.85
            if self.smooth_cx is None:
                self.smooth_cx = center_cx
                self.smooth_cy = center_cy
                self.smooth_S = center_S
            else:
                self.smooth_cx = 0.90 * self.smooth_cx + 0.10 * center_cx
                self.smooth_cy = 0.90 * self.smooth_cy + 0.10 * center_cy
                self.smooth_S = 0.90 * self.smooth_S + 0.10 * center_S

        return self._get_crop_canonical(frame, self.smooth_cx, self.smooth_cy, self.smooth_S)

    def update(self, frame):
        """Cập nhật frame: Dò tìm bằng Haar Cascade và lọc quán tính."""
        face_info = self.detect_face_haar(frame)
        if face_info is not None:
            cx, cy, s = face_info
            return self.update_with_target(cx, cy, s, frame)
        return self.update_idle(frame)


class LocalTFLiteModel:
    """Quản lý và thực thi mô hình TFLite trên Laptop."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_quantized = False
        self.is_loaded = False

        if not HAS_TFLITE:
            print("⚠️ [CẢNH BÁO] Không tìm thấy thư viện tensorflow hoặc ai_edge_litert.")
            print("   👉 Sẽ tự động kích hoạt chế độ mô phỏng giải thuật sinh trắc học.")
            return

        if not os.path.exists(model_path):
            print(f"⚠️ [CẢNH BÁO] Chưa tìm thấy file mô hình tại: {model_path}")
            print("   👉 Hãy train trên Colab và dùng lệnh: py tools/project_manager.py --deploy-model <file.zip>")
            return

        try:
            self.interpreter = TFLiteInterpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            self.input_is_quantized = (self.input_details['dtype'] in [np.int8, np.uint8])
            self.output_is_quantized = (self.output_details['dtype'] in [np.int8, np.uint8])
            self.is_loaded = True
            print(f"✅ ĐÃ NẠP MÔ HÌNH THÀNH CÔNG: {model_path}")
            print(f"   • Input Shape : {self.input_details['shape']} ({self.input_details['dtype'].__name__})")
            print(f"   • Output Shape: {self.output_details['shape']} ({self.output_details['dtype'].__name__})")
            precision_desc = "Full INT8" if (self.input_is_quantized and self.output_is_quantized) else \
                             "Mixed-Precision (INT8 In / Float32 Out)" if self.input_is_quantized else "Float32"
            print(f"   • Kiểu lượng tử: {precision_desc}")
        except Exception as e:
            print(f"❌ [LỖI] Không thể khởi tạo mô hình TFLite: {e}")
            self.is_loaded = False

    def predict(self, face_gray_96x96: np.ndarray) -> np.ndarray:
        """Thực hiện suy luận trả về mảng (22, 2) tọa độ landmarks chuẩn hóa [0.0, 1.0]."""
        if not self.is_loaded:
            return None

        img = face_gray_96x96.astype(np.float32)

        if self.input_is_quantized:
            scale, zero_point = self.input_details['quantization']
            if scale == 0.0:
                scale = 1.0 / 128.0
                zero_point = 0
            # Chuẩn hóa về [-1.0, 1.0] rồi lượng tử hóa sang int8: (img - 128) / 128
            norm = (img - 128.0) / 128.0
            quant = np.clip(np.round(norm / scale) + zero_point, -128, 127).astype(self.input_details['dtype'])
            input_tensor = np.expand_dims(np.expand_dims(quant, axis=0), axis=-1)
        else:
            norm = (img - 128.0) / 128.0
            input_tensor = np.expand_dims(np.expand_dims(norm, axis=0), axis=-1).astype(np.float32)

        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details['index'])[0]

        if self.output_is_quantized:
            o_scale, o_zero_point = self.output_details['quantization']
            if o_scale != 0.0:
                output_data = (output_data.astype(np.float32) - o_zero_point) * o_scale
            else:
                output_data = (output_data.astype(np.float32) + 128.0) / 255.0
        else:
            output_data = output_data.astype(np.float32)

        landmarks = np.clip(output_data.reshape((NUM_LANDMARKS, 2)), 0.0, 1.0)
        return landmarks


class LocalADASController:
    """Máy trạng thái ADAS phản ánh 100% logic firmware C++ trên ESP32-S3."""

    def __init__(self):
        self.calibrated = False
        self.calib_start_time = time.time()
        self.calib_duration = 5.0       # [SYNC 2025] = 5.0s đúng như firmware adas_controller.cpp

        self.ear_samples = []
        self.mar_samples = []

        # [SYNC 2025] Toàn bộ ngưỡng đồng bộ 1:1 với firmware_esp32/main/adas_controller.cpp
        # và project_config.json -> laptop test ra số liệu nào, ESP32 chạy y hệt số liệu đó.
        self.ear_threshold = 0.21       # = ear_default_threshold (project_config.json)
        self.mar_threshold = 0.45       # = mar_default_threshold (công thức MAR mới outer+inner/2w)
        self.yaw_threshold = 30.0       # = yaw_distraction_threshold_deg

        self.closed_eyes_start = None
        self.yawn_start = None
        self.distraction_start = None

        self.yawn_timestamps = []
        self.current_state = "CALIBRATING"
        self.alarm_active = False
        self.alarm_reason = ""
        self.last_beep_time = 0.0

    def recalibrate(self):
        """Khởi động lại quá trình hiệu chuẩn ngưỡng."""
        self.calibrated = False
        self.calib_start_time = time.time()
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.closed_eyes_start = None
        self.yawn_start = None
        self.distraction_start = None
        self.current_state = "CALIBRATING"
        print("🎯 [ADAS] Bắt đầu tự hiệu chuẩn lại ngưỡng trong 5.0 giây...")

    def update(self, ear: float, mar: float, yaw: float, pitch: float):
        now = time.time()
        self.alarm_active = False
        self.alarm_reason = ""

        # 1. Giai đoạn Tự Hiệu Chuẩn (Calibration 3.5s)
        if not self.calibrated:
            self.ear_samples.append(ear)
            self.mar_samples.append(mar)
            elapsed = now - self.calib_start_time
            if elapsed >= self.calib_duration:
                base_ear = np.median(self.ear_samples) if self.ear_samples else 0.28
                base_mar = np.median(self.mar_samples) if self.mar_samples else 0.18
                # [SYNC 2025] Hệ số hiệu chuẩn khớp firmware: EAR*0.75 clip [0.18,0.25], MAR*1.60 floor 0.40
                self.ear_threshold = max(0.18, min(0.25, float(base_ear * 0.75)))
                self.mar_threshold = max(0.40, float(base_mar * 1.60))
                self.calibrated = True
                print(f"\n🎯 [ADAS] HIỆU CHUẨN HOÀN TẤT: EAR_thresh={self.ear_threshold:.2f}, MAR_thresh={self.mar_threshold:.2f}")
            else:
                self.current_state = f"CALIBRATING ({self.calib_duration - elapsed:.1f}s)"
                return

        # 2. Kiểm tra Buồn Ngủ (Mắt nhắm liên tục)
        is_microsleep = False
        is_slow_blink = False
        if ear < self.ear_threshold:
            if self.closed_eyes_start is None:
                self.closed_eyes_start = now
            closed_duration = now - self.closed_eyes_start
            if closed_duration >= 1.5:
                self.current_state = "ALARM: MICROSLEEP!"
                self.alarm_active = True
                self.alarm_reason = f"Ngủ gật nhắm mắt {closed_duration:.1f}s"
                is_microsleep = True
            elif closed_duration >= 0.50:   # [SYNC 2025] = SLOW_BLINK 0.5s của firmware
                self.current_state = "WARNING: SLOW BLINK"
                is_slow_blink = True
        else:
            self.closed_eyes_start = None

        # 3. Kiểm tra Ngáp / Mệt Mỏi (Há miệng)
        is_yawn = False
        if mar > self.mar_threshold:
            if self.yawn_start is None:
                self.yawn_start = now
            yawn_dur = now - self.yawn_start
            if yawn_dur >= 1.5:   # [SYNC 2025] = YAWN_EVENT 1.5s của firmware
                self.current_state = "YAWNING DETECTED"
                is_yawn = True
                if not self.yawn_timestamps or (now - self.yawn_timestamps[-1] > 4.0):
                    self.yawn_timestamps.append(now)
        else:
            self.yawn_start = None

        # Đếm ngáp trong cửa sổ trượt 3 phút (180s)
        self.yawn_timestamps = [t for t in self.yawn_timestamps if (now - t) <= 180.0]
        if len(self.yawn_timestamps) >= 3:
            self.current_state = "ALARM: FATIGUE!"
            self.alarm_active = True
            self.alarm_reason = f"Mệt mỏi: Ngáp {len(self.yawn_timestamps)} lần / 3 phút"

        # 4. Kiểm tra Mất Tập Trung (Quay đầu góc lớn) - [SYNC 2025] Yaw 30°/Pitch 25°/3.0s như firmware
        is_distracted = False
        if abs(yaw) > self.yaw_threshold or abs(pitch) > 25.0:
            if self.distraction_start is None:
                self.distraction_start = now
            distract_dur = now - self.distraction_start
            if distract_dur >= 3.0:
                self.current_state = "ALARM: DISTRACTED!"
                self.alarm_active = True
                self.alarm_reason = f"Quay mặt góc {yaw:+.0f}° quá {distract_dur:.1f}s"
                is_distracted = True
        else:
            self.distraction_start = None

        # 5. Tự động chuyển về trạng thái bình thường khi không còn sự kiện nào
        if not self.alarm_active and not is_slow_blink and not is_yawn and not is_distracted:
            self.current_state = "NORMAL (ATTENTIVE)"

        # Kích hoạt còi hú qua loa máy tính
        if self.alarm_active and HAS_WINSOUND:
            if now - self.last_beep_time >= 0.25:
                winsound.Beep(1200, 100)
                self.last_beep_time = now


def compute_ear(landmarks_px, eye_indices):
    """Tính tỉ lệ mắt mở EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)."""
    pts = [landmarks_px[idx] for idx in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


def compute_mar(landmarks_px, mouth_indices):
    """Tính tỉ lệ miệng há MAR = (||p2-p3|| + ||p4-p5||) / (2 * ||p0-p1||)."""
    pts = [landmarks_px[idx] for idx in mouth_indices]
    A = np.linalg.norm(pts[2] - pts[3])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[0] - pts[1])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


def solve_head_pose_pnp(landmarks_px, img_w, img_h):
    """Ước lượng góc xoay đầu 3D (Yaw, Pitch, Roll) bằng PnP đối chiếu với mô hình nhân trắc học kết hợp Robust Gating."""
    p_nose = landmarks_px[NOSE_TIP_PT]
    p_chin = landmarks_px[CHIN_PT]
    p_eye_l = landmarks_px[LEFT_EYE_PTS[0]]
    p_eye_r = landmarks_px[RIGHT_EYE_PTS[3]]
    p_mouth_l = landmarks_px[MOUTH_PTS[0]]
    p_mouth_r = landmarks_px[MOUTH_PTS[1]]

    # 1. Tính toán góc Yaw hình học từ độ bất đối xứng của chóp mũi so với 2 khóe mắt (Geometric Yaw Anchor)
    # Đây là mỏ neo hình học sọ cứng bất biến trước hiện tượng chụm miệng hay suy biến PnP Levenberg-Marquardt.
    d_eyes = float(np.linalg.norm(p_eye_r - p_eye_l))
    mid_eyes = (p_eye_l + p_eye_r) / 2.0
    dx_nose = float(p_nose[0] - mid_eyes[0])

    geom_yaw = 0.0
    if d_eyes > 10.0:
        ratio = np.clip(dx_nose / (0.35 * d_eyes), -1.0, 1.0)
        geom_yaw = float(math.degrees(math.asin(ratio)))

    # 2. Kiểm tra tính toàn vẹn của khóe miệng (Mouth Width Degeneracy Gating)
    mouth_w = float(np.linalg.norm(p_mouth_r - p_mouth_l))
    mouth_valid = (d_eyes > 10.0) and ((mouth_w / d_eyes) >= 0.35)

    # 6 điểm đối xứng chuẩn
    pts_2d = np.array([
        p_nose,
        p_chin,
        p_eye_l,
        p_eye_r,
        p_mouth_l,
        p_mouth_r
    ], dtype=np.float64)

    focal_length = img_w * 1.1
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return geom_yaw, 0.0, 0.0, None, camera_matrix, dist_coeffs

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    # 3. Robust Gating: Chỉ can thiệp khi PnP bị lật nghiệm suy biến thực sự (|yaw| > 80° hoặc |yaw - geom_yaw| > 45°)
    # Không để việc há miệng khi ngáp hay sai số nhỏ của chóp mũi kích hoạt ghi đè góc ảo.
    if abs(yaw) > 80.0 or abs(yaw - geom_yaw) > 45.0:
        yaw = geom_yaw
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(math.radians(pitch)), -math.sin(math.radians(pitch))],
            [0.0, math.sin(math.radians(pitch)), math.cos(math.radians(pitch))]
        ])
        ry = np.array([
            [math.cos(math.radians(yaw)), 0.0, math.sin(math.radians(yaw))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))]
        ])
        rz = np.array([
            [math.cos(math.radians(roll)), -math.sin(math.radians(roll)), 0.0],
            [math.sin(math.radians(roll)), math.cos(math.radians(roll)), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_clean = rz @ ry @ rx
        rvec, _ = cv2.Rodrigues(R_clean)

    return yaw, pitch, roll, (rvec, tvec), camera_matrix, dist_coeffs


def draw_head_pose_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_len=50.0):
    """
    Vẽ 3 trục tọa độ không gian 3D Head Pose bằng phép chiếu phối cảnh cv2.projectPoints chuẩn xác:
      - Trục X (Đỏ): Trục ngang sang phải (+X), phản ánh góc Pitch / ngửa-cúi.
      - Trục Y (Xanh lá): Trục dọc hướng xuống cằm (+Y), phản ánh góc Yaw / quay trái-phải.
      - Trục Z (Xanh dương): Trục đâm thẳng ra ngoài mặt về phía camera (+Z), phản ánh góc Roll / nghiêng đầu.
    """
    if rvec is None or tvec is None or camera_matrix is None:
        return
    axes_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],      # +X (Sang phải)
        [0.0, axis_len, 0.0],      # +Y (Hướng xuống cằm)
        [0.0, 0.0, axis_len]       # +Z (Hướng ra ngoài mặt về camera)
    ], dtype=np.float64)

    imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2)
    p_origin = (int(round(imgpts[0][0])), int(round(imgpts[0][1])))
    p_x = (int(round(imgpts[1][0])), int(round(imgpts[1][1])))
    p_y = (int(round(imgpts[2][0])), int(round(imgpts[2][1])))
    p_z = (int(round(imgpts[3][0])), int(round(imgpts[3][1])))

    # Vẽ các trục có khử răng cưa LINE_AA
    cv2.line(frame, p_origin, p_x, (0, 0, 255), 2, cv2.LINE_AA)  # X (Đỏ)
    cv2.line(frame, p_origin, p_y, (0, 255, 0), 2, cv2.LINE_AA)  # Y (Xanh lá)
    cv2.line(frame, p_origin, p_z, (255, 0, 0), 2, cv2.LINE_AA)  # Z (Xanh dương)


class TerminalDiagnosticLogger:
    """
    Bộ in kết quả chẩn đoán chuyên sâu thời gian thực ra Terminal (Console Debugger).
    Được thiết kế trực quan, xúc tích, chống cuộn tràn màn hình và hỗ trợ đối chiếu song song:
      - 'FULL': Bảng báo cáo định dạng chuẩn, chia 2 cột đối xứng, so sánh song song TinyDriver vs MediaPipe.
      - 'COMPACT': 1 dòng tóm tắt xúc tích mỗi chu kỳ.
      - 'OFF': Tắt in terminal để giữ màn hình console yên tĩnh.
    """
    def __init__(self, mode="FULL", interval_sec=1.2):
        self.mode = mode  # "FULL", "COMPACT", "OFF"
        self.interval_sec = interval_sec
        self.last_log_time = 0.0
        self.last_alarm_state = None
        self.frame_counter = 0

    def toggle_mode(self):
        modes = ["FULL", "COMPACT", "OFF"]
        idx = (modes.index(self.mode) + 1) % len(modes)
        self.mode = modes[idx]
        print(f"\n📢 [Terminal Debugger] Đã chuyển chế độ hiển thị: {self.mode}\n")
        return self.mode

    def should_log(self, now, current_state=None):
        if self.mode == "OFF":
            return False
        # Nếu có sự kiện báo động mới phát sinh -> kích hoạt in ngay lập tức
        if current_state is not None and self.last_alarm_state is not None:
            if current_state != self.last_alarm_state and "ALARM" in current_state:
                return True
        return (now - self.last_log_time) >= self.interval_sec

    def log(self, frame_idx, fps, active_mode, is_tracking, crop_box,
            landmarks_px, mp_landmarks_px,
            ear_l, ear_r, ear, mar, yaw, pitch, roll,
            adas_ctrl, force=False, frame_w=640, frame_h=480):
        if self.mode == "OFF" and not force:
            return

        now = time.time()
        self.frame_counter += 1
        if not force and not self.should_log(now, adas_ctrl.current_state):
            return

        self.last_log_time = now
        self.last_alarm_state = adas_ctrl.current_state
        x1, y1, x2, y2 = crop_box
        box_w = max(1, x2 - x1)

        # Tính toán dữ liệu MediaPipe Ground-Truth nếu có
        has_gt = (active_mode == "TINYDRIVER" and mp_landmarks_px is not None and landmarks_px is not None)
        mp_ear_l, mp_ear_r, mp_ear, mp_mar = 0.0, 0.0, 0.0, 0.0
        mp_yaw, mp_pitch, mp_roll = 0.0, 0.0, 0.0
        deltas = None

        if has_gt:
            deltas = np.linalg.norm(landmarks_px - mp_landmarks_px, axis=1)  # (22,)
            mp_ear_l = compute_ear(mp_landmarks_px, LEFT_EYE_PTS)
            mp_ear_r = compute_ear(mp_landmarks_px, RIGHT_EYE_PTS)
            mp_ear = (mp_ear_l + mp_ear_r) / 2.0
            mp_mar = compute_mar(mp_landmarks_px, MOUTH_PTS)
            mp_yaw, mp_pitch, mp_roll, _, _, _ = solve_head_pose_pnp(mp_landmarks_px, frame_w, frame_h)

        # -------------------------------------------------------------
        # 1. CHẾ ĐỘ COMPACT (1 DÒNG XÚC TÍCH)
        # -------------------------------------------------------------
        if self.mode == "COMPACT":
            alarm_tag = f"🚨 {adas_ctrl.current_state}" if adas_ctrl.alarm_active else f"✅ {adas_ctrl.current_state}"
            if has_gt:
                d_l2 = np.mean(deltas)
                print(f"[TD-vs-MP] #{frame_idx:04d} | {fps:4.1f}FPS | TD[EAR:{ear:.3f}, MAR:{mar:.3f}, Yaw:{yaw:+5.1f}°] | MP[EAR:{mp_ear:.3f}, MAR:{mp_mar:.3f}, Yaw:{mp_yaw:+5.1f}°] | Δ_L2:{d_l2:4.1f}px | {alarm_tag}")
            elif active_mode == "MEDIAPIPE":
                print(f"[MEDIAPIPE] #{frame_idx:04d} | {fps:4.1f}FPS | EAR:{ear:.3f} (L:{ear_l:.3f}, R:{ear_r:.3f}) | MAR:{mar:.3f} | Yaw:{yaw:+5.1f}° Pitch:{pitch:+5.1f}° | {alarm_tag}")
            else:
                print(f"[TINYDRIVER] #{frame_idx:04d} | {fps:4.1f}FPS | EAR:{ear:.3f} | MAR:{mar:.3f} | Yaw:{yaw:+5.1f}° Pitch:{pitch:+5.1f}° | {alarm_tag}")
            return

        # -------------------------------------------------------------
        # 2. CHẾ ĐỘ FULL (BẢNG TRỰC QUAN ĐẦY ĐỦ THÔNG TIN, 2 CỘT GỌN)
        # -------------------------------------------------------------
        pt_names = [
            "P0  Khóe ngoài", "P1  Mí trên 1", "P2  Mí trên 2", "P3  Khóe trong", "P4  Mí dưới 2", "P5  Mí dưới 1",
            "P6  Khóe trong", "P7  Mí trên 1", "P8  Mí trên 2", "P9  Khóe ngoài", "P10 Mí dưới 2", "P11 Mí dưới 1",
            "P12 Khóe trái", "P13 Khóe phải", "P14 Môi tr ngoài", "P15 Môi d ngoài", "P16 Môi tr trong", "P17 Môi d trong",
            "P18 Gốc mũi", "P19 Chóp mũi", "P20 Nhân trung", "P21 Đáy cằm"
        ]

        col_w = 54
        total_w = col_w * 2 + 3

        # Header Badge
        print("\n" + "╔" + "═" * total_w + "╗")
        if active_mode == "MEDIAPIPE":
            title = f" 🌟 [ADAS DEBUG] #{frame_idx:04d} | FPS: {fps:4.1f} | ENGINE: MEDIAPIPE TEACHER (CHUẨN GROUND-TRUTH) | BÁM MẶT: {'ON' if is_tracking else 'OFF'}"
        else:
            gt_tag = "ĐANG ĐỐI CHIẾU MEDIAPIPE ✅" if has_gt else "MEDIAPIPE OFFLINE"
            title = f" ⚡ [ADAS DEBUG] #{frame_idx:04d} | FPS: {fps:4.1f} | ENGINE: TINYDRIVERNET (INT8 EDGE) | {gt_tag}"
        print(f"║{title:<{total_w}}║")
        print("╚" + "═" * total_w + "╝")

        calib_str = "Đã xong ✅" if adas_ctrl.calibrated else "Đang Calib..."
        print(f" 📦 Hộp Cắt 1:1  : [{x1}, {y1}, {x2}, {y2}] ({box_w}x{box_w} px) | FSM: {adas_ctrl.current_state}")
        print(f" 🎯 Ngưỡng ADAS  : EAR_thresh={adas_ctrl.ear_threshold:.3f} | MAR_thresh={adas_ctrl.mar_threshold:.3f} | Calib: {calib_str}\n")

        if landmarks_px is None and mp_landmarks_px is None:
            print(" ⚠️  [CHƯA PHÁT HIỆN ĐƯỢC 22 ĐIỂM MỐC TRÊN KHUÔN MẶT - ĐANG QUÉT...]\n")
            print(" ⌨️  Phím: [d] Chế độ Log | [p] In ngay | [m] Đổi Engine | [f] Bám mặt | [r] Calib | [q] Thoát\n")
            return

        # =============================================================
        # NHÁNH A: CHẾ ĐỘ MEDIAPIPE (THAM KHẢO CHUẨN THỰC TẾ)
        # =============================================================
        if active_mode == "MEDIAPIPE":
            cur_pts = landmarks_px if landmarks_px is not None else mp_landmarks_px
            w_eye_l = np.linalg.norm(cur_pts[0] - cur_pts[3])
            h1_eye_l = np.linalg.norm(cur_pts[1] - cur_pts[5])
            h2_eye_l = np.linalg.norm(cur_pts[2] - cur_pts[4])
            w_eye_r = np.linalg.norm(cur_pts[6] - cur_pts[9])
            h1_eye_r = np.linalg.norm(cur_pts[7] - cur_pts[11])
            h2_eye_r = np.linalg.norm(cur_pts[8] - cur_pts[10])
            w_mouth = np.linalg.norm(cur_pts[12] - cur_pts[13])
            h_outer = np.linalg.norm(cur_pts[14] - cur_pts[15])
            h_inner = np.linalg.norm(cur_pts[16] - cur_pts[17])

            dir_str = "CHÍNH DIỆN 🟢"
            if yaw > 18: dir_str = "QUAY TRÁI ⬅️"
            elif yaw < -18: dir_str = "QUAY PHẢI ➡️"
            elif pitch > 15: dir_str = "CÚI ĐẦU ⬇️"
            elif pitch < -15: dir_str = "NGỬA ĐẦU ⬆️"

            t1, t2, t3, t4 = 27, 13, 38, 22
            sep_top = f"┌{'─'*t1}┬{'─'*t2}┬{'─'*t3}┬{'─'*t4}┐"
            sep_mid = f"├{'─'*t1}┼{'─'*t2}┼{'─'*t3}┼{'─'*t4}┤"
            sep_bot = f"└{'─'*t1}┴{'─'*t2}┴{'─'*t3}┴{'─'*t4}┘"

            print(sep_top)
            print(f"│ {'CHỈ SỐ SINH TRẮC CHUẨN':<{t1-1}}│ {'GIÁ TRỊ':<{t2-1}}│ {'CHI TIẾT ĐO ĐẠC HÌNH HỌC (PIXEL)':<{t3-1}}│ {'ĐÁNH GIÁ':<{t4-1}}│")
            print(sep_mid)
            eval_el = "MẮT MỞ ✅" if ear_l >= adas_ctrl.ear_threshold else "NHẮM MẮT ⚠️"
            eval_er = "MẮT MỞ ✅" if ear_r >= adas_ctrl.ear_threshold else "NHẮM MẮT ⚠️"
            eval_e = "TỈNH TÁO ✅" if ear >= adas_ctrl.ear_threshold else "CẢNH BÁO NHẮM ⚠️"
            eval_m = "NGÁP HÁ TO ⚠️" if mar > adas_ctrl.mar_threshold else "NGẬM BÌNH THƯỜNG ✅"
            print(f"│ {'EAR Mắt Trái (P0..P5)':<{t1-1}}│ {ear_l:<{t2-1}.3f}│ {f'W: {w_eye_l:4.1f}px | H1: {h1_eye_l:4.1f}, H2: {h2_eye_l:4.1f}':<{t3-1}}│ {eval_el:<{t4-1}}│")
            print(f"│ {'EAR Mắt Phải (P6..P11)':<{t1-1}}│ {ear_r:<{t2-1}.3f}│ {f'W: {w_eye_r:4.1f}px | H1: {h1_eye_r:4.1f}, H2: {h2_eye_r:4.1f}':<{t3-1}}│ {eval_er:<{t4-1}}│")
            print(f"│ {'EAR Trung Bình 2 Mắt':<{t1-1}}│ {ear:<{t2-1}.3f}│ {f'Ngưỡng hiệu chuẩn: {adas_ctrl.ear_threshold:.3f}':<{t3-1}}│ {eval_e:<{t4-1}}│")
            print(f"│ {'MAR Há Miệng (P12..P17)':<{t1-1}}│ {mar:<{t2-1}.3f}│ {f'W: {w_mouth:4.1f}px | Dày: {h_outer:4.1f}, Hở: {h_inner:4.1f}':<{t3-1}}│ {eval_m:<{t4-1}}│")
            print(f"│ {'3D Pose Yaw (Trái/Phải)':<{t1-1}}│ {f'{yaw:+5.1f}°':<{t2-1}}│ {'Góc quay ngang (Ngưỡng: ±30.0°)':<{t3-1}}│ {dir_str:<{t4-1}}│")
            print(f"│ {'3D Pose Pitch (Cúi/Ngửa)':<{t1-1}}│ {f'{pitch:+5.1f}°':<{t2-1}}│ {'Góc ngửa/cúi   (Ngưỡng: ±25.0°)':<{t3-1}}│ {dir_str:<{t4-1}}│")
            print(f"│ {'3D Pose Roll (Nghiêng)':<{t1-1}}│ {f'{roll:+5.1f}°':<{t2-1}}│ {'Góc nghiêng mặt theo phương ngang':<{t3-1}}│ {'CÂN BẰNG 🟢':<{t4-1}}│")
            print(sep_bot)

            # Bảng tọa độ 22 điểm chuẩn chia 2 cột
            print(f"\n┌{'─' * (col_w + 2)}┬{'─' * (col_w + 2)}┐")
            print(f"│ {'MẮT TRÁI (P0 - P5)          (X, Y) px      [u, v]':<{col_w}} │ {'MẮT PHẢI (P6 - P11)         (X, Y) px      [u, v]':<{col_w}} │")
            print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
            for i in range(6):
                idx_l = i
                xl, yl = cur_pts[idx_l]
                ul, vl = (xl - x1) / box_w, (yl - y1) / box_w
                left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) px [u={ul:5.3f},v={vl:5.3f}]"

                idx_r = i + 6
                xr, yr = cur_pts[idx_r]
                ur, vr = (xr - x1) / box_w, (yr - y1) / box_w
                right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) px [u={ur:5.3f},v={vr:5.3f}]"
                print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")

            print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
            print(f"│ {'MIỆNG (P12 - P17)           (X, Y) px      [u, v]':<{col_w}} │ {'MŨI & CẰM (P18 - P21)       (X, Y) px      [u, v]':<{col_w}} │")
            print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
            for i in range(6):
                idx_l = 12 + i
                xl, yl = cur_pts[idx_l]
                ul, vl = (xl - x1) / box_w, (yl - y1) / box_w
                left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) px [u={ul:5.3f},v={vl:5.3f}]"

                if i < 4:
                    idx_r = 18 + i
                    xr, yr = cur_pts[idx_r]
                    ur, vr = (xr - x1) / box_w, (yr - y1) / box_w
                    right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) px [u={ur:5.3f},v={vr:5.3f}]"
                else:
                    right_str = ""
                print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")
            print(f"└{'─' * (col_w + 2)}┴{'─' * (col_w + 2)}┘")

        # =============================================================
        # NHÁNH B: CHẾ ĐỘ TINYDRIVER (ĐỐI CHIẾU SONG SONG VỚI MEDIAPIPE)
        # =============================================================
        else:
            cur_pts = landmarks_px
            if has_gt:
                # 1. Bảng đối chiếu song song
                t1, t2, t3, t4, t5 = 26, 15, 15, 15, 30
                sep_top = f"┌{'─'*t1}┬{'─'*t2}┬{'─'*t3}┬{'─'*t4}┬{'─'*t5}┐"
                sep_mid = f"├{'─'*t1}┼{'─'*t2}┼{'─'*t3}┼{'─'*t4}┼{'─'*t5}┤"
                sep_bot = f"└{'─'*t1}┴{'─'*t2}┴{'─'*t3}┴{'─'*t4}┴{'─'*t5}┘"

                dir_str = "CHÍNH DIỆN 🟢"
                if yaw > 18: dir_str = "QUAY TRÁI ⬅️"
                elif yaw < -18: dir_str = "QUAY PHẢI ➡️"
                elif pitch > 15: dir_str = "CÚI ĐẦU ⬇️"
                elif pitch < -15: dir_str = "NGỬA ĐẦU ⬆️"

                d_ear_l = ear_l - mp_ear_l
                d_ear_r = ear_r - mp_ear_r
                d_ear = ear - mp_ear
                d_mar = mar - mp_mar
                d_yaw = yaw - mp_yaw
                d_pitch = pitch - mp_pitch
                d_roll = roll - mp_roll

                print(sep_top)
                print(f"│ {'CHỈ SỐ SINH TRẮC & POSE':<{t1-1}}│ {'TINYDRIVER':<{t2-1}}│ {'MEDIAPIPE':<{t3-1}}│ {'ĐỘ LỆCH (Δ)':<{t4-1}}│ {'ĐÁNH GIÁ & CẢNH BÁO':<{t5-1}}│")
                print(sep_mid)
                eval_ear_l = "Khớp tốt ✅" if abs(d_ear_l) < 0.035 else "Lệch mi ⚠️"
                eval_ear_r = "Khớp tốt ✅" if abs(d_ear_r) < 0.035 else "Lệch mi ⚠️"
                eval_ear = "MẮT MỞ TỈNH TÁO ✅" if ear >= adas_ctrl.ear_threshold else "⚠️ CẢNH BÁO NHẮM MẮT"
                eval_mar = "NGẬM BÌNH THƯỜNG ✅" if mar <= adas_ctrl.mar_threshold else "⚠️ CẢNH BÁO NGÁP"
                print(f"│ {'EAR Mắt Trái':<{t1-1}}│ {ear_l:<{t2-1}.3f}│ {mp_ear_l:<{t3-1}.3f}│ {f'{d_ear_l:+6.3f}':<{t4-1}}│ {eval_ear_l:<{t5-1}}│")
                print(f"│ {'EAR Mắt Phải':<{t1-1}}│ {ear_r:<{t2-1}.3f}│ {mp_ear_r:<{t3-1}.3f}│ {f'{d_ear_r:+6.3f}':<{t4-1}}│ {eval_ear_r:<{t5-1}}│")
                print(f"│ {'EAR Trung Bình':<{t1-1}}│ {ear:<{t2-1}.3f}│ {mp_ear:<{t3-1}.3f}│ {f'{d_ear:+6.3f}':<{t4-1}}│ {eval_ear:<{t5-1}}│")
                print(f"│ {'MAR Há Miệng':<{t1-1}}│ {mar:<{t2-1}.3f}│ {mp_mar:<{t3-1}.3f}│ {f'{d_mar:+6.3f}':<{t4-1}}│ {eval_mar:<{t5-1}}│")
                print(f"│ {'3D Pose Yaw (Trái/Phải)':<{t1-1}}│ {f'{yaw:+5.1f}°':<{t2-1}}│ {f'{mp_yaw:+5.1f}°':<{t3-1}}│ {f'{d_yaw:+5.1f}°':<{t4-1}}│ {dir_str:<{t5-1}}│")
                print(f"│ {'3D Pose Pitch (Cúi/Ngửa)':<{t1-1}}│ {f'{pitch:+5.1f}°':<{t2-1}}│ {f'{mp_pitch:+5.1f}°':<{t3-1}}│ {f'{d_pitch:+5.1f}°':<{t4-1}}│ {dir_str:<{t5-1}}│")
                print(f"│ {'3D Pose Roll (Nghiêng)':<{t1-1}}│ {f'{roll:+5.1f}°':<{t2-1}}│ {f'{mp_roll:+5.1f}°':<{t3-1}}│ {f'{d_roll:+5.1f}°':<{t4-1}}│ {'CÂN BẰNG 🟢':<{t5-1}}│")
                print(sep_bot)

                mean_all = float(np.mean(deltas))
                max_all = float(np.max(deltas))
                max_idx = int(np.argmax(deltas))
                qual = "XUẤT SẮC (<4px)" if mean_all < 4.0 else ("TỐT (<8px)" if mean_all < 8.0 else "TRUNG BÌNH (<15px)" if mean_all < 15.0 else "LỆCH NHIỀU (Cần Retrain)")
                print(f" 📊 Sai số L2 so với MediaPipe: TB = {mean_all:4.2f}px [{qual}] | Lớn nhất: {max_all:4.2f}px (tại {pt_names[max_idx]})")
                print(f"    • Mắt T: {np.mean(deltas[LEFT_EYE_PTS]):4.2f}px | Mắt P: {np.mean(deltas[RIGHT_EYE_PTS]):4.2f}px | Miệng: {np.mean(deltas[MOUTH_PTS]):4.2f}px | Mũi/Cằm: {np.mean(deltas[18:]):4.2f}px")

                # 2. Bảng 22 điểm đối chiếu 2 cột song song
                print(f"\n┌{'─' * (col_w + 2)}┬{'─' * (col_w + 2)}┐")
                print(f"│ {'MẮT TRÁI (P0 - P5)     TINYDRIVER vs MP    Δ(px)':<{col_w}} │ {'MẮT PHẢI (P6 - P11)    TINYDRIVER vs MP    Δ(px)':<{col_w}} │")
                print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                for i in range(6):
                    idx_l = i
                    xl, yl = cur_pts[idx_l]
                    mxl, myl = mp_landmarks_px[idx_l]
                    dl = deltas[idx_l]
                    fl = "!" if dl > 8.0 else " "
                    left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) MP:({mxl:5.1f},{myl:5.1f}) {fl}Δ{dl:4.1f}"

                    idx_r = i + 6
                    xr, yr = cur_pts[idx_r]
                    mxr, myr = mp_landmarks_px[idx_r]
                    dr = deltas[idx_r]
                    fr = "!" if dr > 8.0 else " "
                    right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) MP:({mxr:5.1f},{myr:5.1f}) {fr}Δ{dr:4.1f}"
                    print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")

                print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                print(f"│ {'MIỆNG (P12 - P17)      TINYDRIVER vs MP    Δ(px)':<{col_w}} │ {'MŨI & CẰM (P18 - P21)  TINYDRIVER vs MP    Δ(px)':<{col_w}} │")
                print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                for i in range(6):
                    idx_l = 12 + i
                    xl, yl = cur_pts[idx_l]
                    mxl, myl = mp_landmarks_px[idx_l]
                    dl = deltas[idx_l]
                    fl = "!" if dl > 8.0 else " "
                    left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) MP:({mxl:5.1f},{myl:5.1f}) {fl}Δ{dl:4.1f}"

                    if i < 4:
                        idx_r = 18 + i
                        xr, yr = cur_pts[idx_r]
                        mxr, myr = mp_landmarks_px[idx_r]
                        dr = deltas[idx_r]
                        fr = "!" if dr > 8.0 else " "
                        right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) MP:({mxr:5.1f},{myr:5.1f}) {fr}Δ{dr:4.1f}"
                    else:
                        right_str = ""
                    print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")
                print(f"└{'─' * (col_w + 2)}┴{'─' * (col_w + 2)}┘")
            else:
                # Trường hợp không có MediaPipe Teacher (chạy độc lập TinyDriver)
                t1, t2, t3, t4 = 27, 13, 38, 22
                sep_top = f"┌{'─'*t1}┬{'─'*t2}┬{'─'*t3}┬{'─'*t4}┐"
                sep_mid = f"├{'─'*t1}┼{'─'*t2}┼{'─'*t3}┼{'─'*t4}┤"
                sep_bot = f"└{'─'*t1}┴{'─'*t2}┴{'─'*t3}┴{'─'*t4}┘"

                dir_str = "CHÍNH DIỆN 🟢"
                if yaw > 18: dir_str = "QUAY TRÁI ⬅️"
                elif yaw < -18: dir_str = "QUAY PHẢI ➡️"
                elif pitch > 15: dir_str = "CÚI ĐẦU ⬇️"
                elif pitch < -15: dir_str = "NGỬA ĐẦU ⬆️"

                eval_e = "TỈNH TÁO ✅" if ear >= adas_ctrl.ear_threshold else "CẢNH BÁO NHẮM ⚠️"
                eval_m = "NGÁP HÁ TO ⚠️" if mar > adas_ctrl.mar_threshold else "NGẬM BÌNH THƯỜNG ✅"

                print(sep_top)
                print(f"│ {'CHỈ SỐ SINH TRẮC (TINYDRIVER)':<{t1-1}}│ {'GIÁ TRỊ':<{t2-1}}│ {'THÔNG SỐ SO SÁNH':<{t3-1}}│ {'ĐÁNH GIÁ':<{t4-1}}│")
                print(sep_mid)
                print(f"│ {'EAR Mắt Trái (P0..P5)':<{t1-1}}│ {ear_l:<{t2-1}.3f}│ {'Khoảng cách mi mắt trái':<{t3-1}}│ {'MẮT MỞ ✅' if ear_l >= adas_ctrl.ear_threshold else 'NHẮM ⚠️':<{t4-1}}│")
                print(f"│ {'EAR Mắt Phải (P6..P11)':<{t1-1}}│ {ear_r:<{t2-1}.3f}│ {'Khoảng cách mi mắt phải':<{t3-1}}│ {'MẮT MỞ ✅' if ear_r >= adas_ctrl.ear_threshold else 'NHẮM ⚠️':<{t4-1}}│")
                print(f"│ {'EAR Trung Bình':<{t1-1}}│ {ear:<{t2-1}.3f}│ {f'Ngưỡng hiệu chuẩn: {adas_ctrl.ear_threshold:.3f}':<{t3-1}}│ {eval_e:<{t4-1}}│")
                print(f"│ {'MAR Há Miệng':<{t1-1}}│ {mar:<{t2-1}.3f}│ {f'Ngưỡng ngáp: {adas_ctrl.mar_threshold:.3f}':<{t3-1}}│ {eval_m:<{t4-1}}│")
                print(f"│ {'3D Pose Yaw (Trái/Phải)':<{t1-1}}│ {f'{yaw:+5.1f}°':<{t2-1}}│ {'Ngưỡng quay đầu: ±30.0°':<{t3-1}}│ {dir_str:<{t4-1}}│")
                print(f"│ {'3D Pose Pitch (Cúi/Ngửa)':<{t1-1}}│ {f'{pitch:+5.1f}°':<{t2-1}}│ {'Ngưỡng cúi/ngửa: ±25.0°':<{t3-1}}│ {dir_str:<{t4-1}}│")
                print(f"│ {'3D Pose Roll (Nghiêng)':<{t1-1}}│ {f'{roll:+5.1f}°':<{t2-1}}│ {'Góc nghiêng ngang':<{t3-1}}│ {'CÂN BẰNG 🟢':<{t4-1}}│")
                print(sep_bot)
                print(" ℹ️  MediaPipe Teacher: Không khả dụng để đối chiếu song song.")

                if cur_pts is not None:
                    print(f"\n┌{'─' * (col_w + 2)}┬{'─' * (col_w + 2)}┐")
                    print(f"│ {'MẮT TRÁI (P0 - P5)          (X, Y) px      [u, v]':<{col_w}} │ {'MẮT PHẢI (P6 - P11)         (X, Y) px      [u, v]':<{col_w}} │")
                    print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                    for i in range(6):
                        idx_l = i
                        xl, yl = cur_pts[idx_l]
                        ul, vl = (xl - x1) / box_w, (yl - y1) / box_w
                        left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) px [u={ul:5.3f},v={vl:5.3f}]"

                        idx_r = i + 6
                        xr, yr = cur_pts[idx_r]
                        ur, vr = (xr - x1) / box_w, (yr - y1) / box_w
                        right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) px [u={ur:5.3f},v={vr:5.3f}]"
                        print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")

                    print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                    print(f"│ {'MIỆNG (P12 - P17)           (X, Y) px      [u, v]':<{col_w}} │ {'MŨI & CẰM (P18 - P21)       (X, Y) px      [u, v]':<{col_w}} │")
                    print(f"├{'─' * (col_w + 2)}┼{'─' * (col_w + 2)}┤")
                    for i in range(6):
                        idx_l = 12 + i
                        xl, yl = cur_pts[idx_l]
                        ul, vl = (xl - x1) / box_w, (yl - y1) / box_w
                        left_str = f"{pt_names[idx_l]:<16}: ({xl:5.1f},{yl:5.1f}) px [u={ul:5.3f},v={vl:5.3f}]"

                        if i < 4:
                            idx_r = 18 + i
                            xr, yr = cur_pts[idx_r]
                            ur, vr = (xr - x1) / box_w, (yr - y1) / box_w
                            right_str = f"{pt_names[idx_r]:<16}: ({xr:5.1f},{yr:5.1f}) px [u={ur:5.3f},v={vr:5.3f}]"
                        else:
                            right_str = ""
                        print(f"│ {left_str:<{col_w}} │ {right_str:<{col_w}} │")
                    print(f"└{'─' * (col_w + 2)}┴{'─' * (col_w + 2)}┘")

        print(" ⌨️  Phím: [d] Chế độ Log | [p] In ngay | [m] Đổi Engine | [f] Bám mặt | [r] Calib | [q] Thoát\n")



def draw_hud(frame, crop_box, landmarks_px, gray_96x96, ear, mar, yaw, pitch, roll, adas_ctrl, fps, is_real_ai, is_tracking, active_mode="TINYDRIVER", log_mode="FULL", pnp_data=None, mp_landmarks_px=None):
    """
    Tạo bố cục màn hình đôi Cyberpunk Widescreen (960x480):
      - Khung hình bên trái (640x480): Hình camera thô + Bám mặt + 22 điểm + 3D Pose vector.
      - Khung điều khiển bên phải (320x480): Bảng đo ADAS Telemetry + Tensor Preview 96x96.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box

    # Tạo canvas toàn cảnh 960x480
    side_w = 320
    canvas = np.zeros((h, w + side_w, 3), dtype=np.uint8)

    # 1. Vẽ khung vuông Isomorphic 1:1 bám mặt trên khung hình camera
    roi_color = (0, 0, 255) if adas_ctrl.alarm_active else ((0, 240, 255) if is_tracking else (70, 70, 70))
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

    # Bo góc trang trí HUD công nghệ cao
    corner_len = max(15, int((x2 - x1) * 0.12))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), roi_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), roi_color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), roi_color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), roi_color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), roi_color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), roi_color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), roi_color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), roi_color, 3)

    box_tag = "BAM MAT 1:1" if is_tracking else "CAT TAM CO DINH"
    cv2.putText(frame, box_tag, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1)

    # 2. Vẽ 22 điểm mốc sinh học và 3 trục quay đầu 3D trên camera
    # Nếu ở chế độ TINYDRIVER và có MediaPipe Ground-Truth, vẽ trước các điểm mốc chuẩn màu xám nhạt để quan sát độ trùng khớp
    if active_mode == "TINYDRIVER" and mp_landmarks_px is not None:
        for (mpx, mpy) in mp_landmarks_px:
            cv2.circle(frame, (int(round(mpx)), int(round(mpy))), 2, (140, 140, 140), 1)

    if landmarks_px is not None:
        left_pts = np.array([landmarks_px[i] for i in LEFT_EYE_PTS], dtype=np.int32)
        right_pts = np.array([landmarks_px[i] for i in RIGHT_EYE_PTS], dtype=np.int32)
        mouth_pts = np.array([landmarks_px[i] for i in MOUTH_PTS], dtype=np.int32)

        cv2.polylines(frame, [left_pts], True, (0, 200, 0), 1)
        cv2.polylines(frame, [right_pts], True, (0, 200, 0), 1)
        cv2.polylines(frame, [mouth_pts], True, (200, 0, 200), 1)

        for idx, (px, py) in enumerate(landmarks_px):
            pt = (int(round(px)), int(round(py)))
            if idx in LEFT_EYE_PTS or idx in RIGHT_EYE_PTS:
                color = (0, 255, 0)
                radius = 3
            elif idx in MOUTH_PTS:
                color = (255, 0, 255)
                radius = 3
            else:
                color = (0, 255, 255)
                radius = 4
            cv2.circle(frame, pt, radius, color, -1)

        # Vẽ 3 trục quay đầu 3D từ chóp mũi (Nose Tip P19) bằng cv2.projectPoints chuẩn xác
        axis_len = max(35.0, (x2 - x1) * 0.28)
        if pnp_data is not None and pnp_data[0] is not None and pnp_data[1] is not None:
            rvec, tvec, cam_mat, dist_c = pnp_data
            draw_head_pose_axes(frame, rvec, tvec, cam_mat, dist_c, axis_len=axis_len)
        else:
            # Dự phòng khi chưa có PnP (ví dụ chế độ Synthetic)
            nose_px, nose_py = int(landmarks_px[NOSE_TIP_PT][0]), int(landmarks_px[NOSE_TIP_PT][1])
            yaw_rad = math.radians(yaw)
            pitch_rad = math.radians(pitch)
            roll_rad = math.radians(roll)
            x_end = (int(nose_px + axis_len * math.cos(yaw_rad)), int(nose_py + axis_len * math.sin(roll_rad)))
            y_end = (int(nose_px - axis_len * math.sin(roll_rad)), int(nose_py + axis_len * math.cos(pitch_rad)))
            z_end = (int(nose_px + axis_len * math.sin(yaw_rad)), int(nose_py - axis_len * math.sin(pitch_rad)))
            cv2.line(frame, (nose_px, nose_py), x_end, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (nose_px, nose_py), y_end, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (nose_px, nose_py), z_end, (255, 0, 0), 2, cv2.LINE_AA)

    # Chớp viền đỏ cảnh báo nếu có báo động
    if adas_ctrl.alarm_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(frame, f"🚨 {adas_ctrl.alarm_reason}", (w // 2 - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Đưa ảnh camera vào nửa bên trái của canvas
    canvas[:, :w] = frame

    # 3. Vẽ Bảng Điều Khiển ADAS Telemetry vào nửa bên phải của canvas (width: 320)
    side = canvas[:, w:]
    side[:] = [22, 22, 28] # Màu nền Dark Cyberpunk
    cv2.line(canvas, (w, 0), (w, h), (60, 60, 80), 2)

    # Tiêu đề
    title_text = "TINYDRIVER ADAS"
    if active_mode == "MEDIAPIPE":
        title_sub = "[MEDIAPIPE 22-PTS]"
        sub_color = (255, 230, 0) # Neon Cyan
        track_tag = "BAM MAT: MEDIAPIPE GT" if is_tracking else "DANG TIM KHUON MAT..."
        track_color = (255, 230, 0) if is_tracking else (0, 180, 255)
    else:
        title_sub = "[REAL AI INT8]" if is_real_ai else "[MO PHONG]"
        sub_color = (0, 255, 180) # Emerald green
        track_tag = "BAM MAT: TINYDRIVER AI" if is_tracking else "DANG TIM KHUON MAT..."
        track_color = (0, 255, 120) if is_tracking else (0, 180, 255)

    cv2.putText(side, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
    cv2.putText(side, title_sub, (200, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sub_color, 1)
    cv2.putText(side, f"FPS: {fps:.1f} | 96x96 Grayscale Tensor", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.line(side, (20, 60), (300, 60), (50, 50, 65), 1)

    cv2.putText(side, track_tag, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, track_color, 1)

    # Hiển thị Tensor 96x96 Crop AI thu nhỏ trực tiếp (Kích thước 96x96)
    if gray_96x96 is not None:
        tensor_vis = cv2.cvtColor(gray_96x96, cv2.COLOR_GRAY2BGR)
        tensor_vis = cv2.resize(tensor_vis, (96, 96), interpolation=cv2.INTER_NEAREST)
        side[90:90+96, 20:20+96] = tensor_vis
        cv2.rectangle(side, (19, 89), (20+96, 90+96), (0, 240, 255), 1)
        cv2.putText(side, "TENSOR 96x96", (24, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Thông tin bên cạnh Tensor Crop
    cv2.putText(side, "DAU VAO AI:", (130, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)
    cv2.putText(side, "Can canh 1:1", (130, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.putText(side, f"Khuon mat: {crop_box[2]-crop_box[0]}px", (130, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(side, "Mat/Mieng ro net", (130, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 150), 1)

    # --- THANH ĐO EAR (MẮT) ---
    cv2.line(side, (20, 212), (300, 212), (50, 50, 65), 1)
    is_eyes_closed = (ear < adas_ctrl.ear_threshold)
    ear_status_text = "[NHAM MAT]" if is_eyes_closed else "MO BINH THUONG"
    ear_text_color = (0, 0, 255) if is_eyes_closed else (0, 255, 0)
    cv2.putText(side, f"EAR (Mat): {ear:.3f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(side, ear_status_text, (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ear_text_color, 1)

    bar_w_max = 280
    bar_ear_w = int(np.clip(ear / 0.40, 0.0, 1.0) * bar_w_max)
    bar_ear_color = (0, 0, 255) if is_eyes_closed else ((0, 220, 255) if ear < 0.25 else (0, 255, 0))
    cv2.rectangle(side, (20, 238), (20 + bar_w_max, 252), (40, 40, 50), -1)
    cv2.rectangle(side, (20, 238), (20 + bar_ear_w, 252), bar_ear_color, -1)
    thresh_ear_x = int(20 + np.clip(adas_ctrl.ear_threshold / 0.40, 0.0, 1.0) * bar_w_max)
    cv2.line(side, (thresh_ear_x, 236), (thresh_ear_x, 254), (255, 255, 255), 2)
    cv2.putText(side, f"Nguong: {adas_ctrl.ear_threshold:.2f}", (thresh_ear_x - 20, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    # --- THANH ĐO MAR (MIỆNG) ---
    cv2.line(side, (20, 276), (300, 276), (50, 50, 65), 1)
    is_yawning = (mar > adas_ctrl.mar_threshold)
    mar_status_text = "[DANG NGAP]" if is_yawning else "NGAM BINH THUONG"
    mar_text_color = (0, 0, 255) if is_yawning else (255, 0, 255)
    cv2.putText(side, f"MAR (Mieng): {mar:.3f}", (20, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(side, mar_status_text, (160, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mar_text_color, 1)

    bar_mar_w = int(np.clip(mar / 0.80, 0.0, 1.0) * bar_w_max)
    bar_mar_color = (0, 0, 255) if is_yawning else (255, 0, 255)
    cv2.rectangle(side, (20, 302), (20 + bar_w_max, 316), (40, 40, 50), -1)
    cv2.rectangle(side, (20, 302), (20 + bar_mar_w, 316), bar_mar_color, -1)
    thresh_mar_x = int(20 + np.clip(adas_ctrl.mar_threshold / 0.80, 0.0, 1.0) * bar_w_max)
    cv2.line(side, (thresh_mar_x, 300), (thresh_mar_x, 318), (255, 255, 255), 2)
    cv2.putText(side, f"Nguong: {adas_ctrl.mar_threshold:.2f}", (thresh_mar_x - 20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    # --- GÓC QUAY ĐẦU HEAD POSE ---
    cv2.line(side, (20, 340), (300, 340), (50, 50, 65), 1)
    dir_tag = "CHINH DIEN"
    if yaw > 18.0:
        dir_tag = "QUAY TRAI"
    elif yaw < -18.0:
        dir_tag = "QUAY PHAI"
    elif pitch > 15.0:
        dir_tag = "CUI DAU"
    elif pitch < -15.0:
        dir_tag = "NGUA DAU"

    pose_color = (0, 0, 255) if abs(yaw) > adas_ctrl.yaw_threshold or abs(pitch) > 22.0 else (0, 255, 255)
    cv2.putText(side, f"Goc Dau: Yaw={yaw:+.1f} | Pitch={pitch:+.1f}", (20, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pose_color, 1)
    cv2.putText(side, f"Huong nhin: {dir_tag}", (20, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # --- TRẠNG THÁI ADAS FSM ---
    cv2.line(side, (20, 388), (300, 388), (50, 50, 65), 1)
    status_bg = (0, 0, 200) if adas_ctrl.alarm_active else ((0, 130, 255) if "WARNING" in adas_ctrl.current_state else (0, 140, 0))
    cv2.rectangle(side, (20, 398), (300, 432), status_bg, -1)
    cv2.putText(side, adas_ctrl.current_state, (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # Dòng phím bấm hướng dẫn
    cv2.putText(side, f"[m] Mode | [f] Bam mat | [d] Log: {log_mode} | [r] Calib | [q] Thoat", (8, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 170), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử mô hình AI TinyDriver trên Laptop với Webcam")
    parser.add_argument("--cam", type=int, default=0, help="ID cổng camera vật lý (mặc định: 0)")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn đến file .tflite tùy chỉnh")
    parser.add_argument("--synthetic", action="store_true", help="Chạy chế độ giả lập mô phỏng tài xế (không cần camera)")
    parser.add_argument("--log-mode", type=str, default="FULL", choices=["FULL", "COMPACT", "OFF"], help="Chế độ in Terminal (FULL, COMPACT, OFF)")
    parser.add_argument("--log-interval", type=float, default=1.2, help="Khoảng thời gian giữa 2 lần xuất log Terminal (giây, mặc định 1.2s)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    model_path = args.model or str(root_dir / "host_laptop" / "models" / "tinydriver_model.tflite")

    print("=" * 70)
    print("🚀 KHỞI ĐỘNG BỘ KIỂM THỬ MÔ HÌNH TINYDRIVER TRÊN LAPTOP (V2 - DUAL ENGINE)")
    print("=" * 70)

    ai_model = LocalTFLiteModel(model_path)
    mp_engine = MediaPipeLandmarkExtractor()
    adas_controller = LocalADASController()
    face_tracker = FaceTracker(expansion_ratio=1.35, ema_alpha=0.15, deadband_pos=6.0, deadband_size=8.0)
    terminal_logger = TerminalDiagnosticLogger(mode=args.log_mode, interval_sec=args.log_interval)

    active_mode = "MEDIAPIPE" if mp_engine.is_loaded else "TINYDRIVER"

    cap = None
    if not args.synthetic:
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            print(f"⚠️ Không mở được camera ID {args.cam}. Tự động chuyển sang chế độ mô phỏng (--synthetic).")
            args.synthetic = True
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Khử độ trễ đệm 4-5 frame nội bộ của OpenCV DirectShow

    print("\n👉 Bảng điều khiển đang hiển thị trên màn hình (Bố cục Widescreen 960x480).")
    print(f"💎 Chế độ khởi tạo: {active_mode}")
    print("⌨️  Phím bấm:")
    print("   • 'm': Chuyển đổi giữa MediaPipe (Ground-Truth 22-pts) và TinyDriverNet (INT8 Edge)")
    print("   • 'f': Bật/Tắt chế độ tự động bám mặt (Dynamic Face Tracking)")
    print("   • 'd': Đổi chế độ Terminal Debug Log (FULL -> COMPACT -> OFF)")
    print("   • 'p': Chụp nhanh và in chi tiết toàn bộ 22 điểm mốc ra Terminal ngay lập tức")
    print("   • 'r': Hiệu chuẩn lại ngưỡng (Recalibrate)")
    print("   • 'q' hoặc ESC: Thoát chương trình\n")

    sim_frame_idx = 0
    frame_count = 0
    t_prev = time.time()
    last_tracked_landmarks = None
    last_mp_landmarks_px = None

    # Tách bộ lọc One-Euro thích ứng theo từng vùng giải phẫu sinh học:
    #   • Mắt: Cực nhạy (min_cutoff=1.8, beta=0.06) bắt chớp mắt 100-300ms, không làm trễ hay méo EAR
    #   • Miệng: Cân bằng (min_cutoff=1.0, beta=0.03) mượt mà theo sát cử động ngáp há to mà không rung viền
    #   • Mũi & Cằm: Cực đầm (min_cutoff=0.35, beta=0.015) giữ vững mỏ neo sọ cứng, triệt tiêu rung giật cho PnP
    landmark_filters = []
    for pt_idx in range(NUM_LANDMARKS):
        if pt_idx in LEFT_EYE_PTS or pt_idx in RIGHT_EYE_PTS:
            fx = OneEuroFilter(min_cutoff=1.8, beta=0.06)
            fy = OneEuroFilter(min_cutoff=1.8, beta=0.06)
        elif pt_idx in MOUTH_PTS:
            fx = OneEuroFilter(min_cutoff=1.0, beta=0.03)
            fy = OneEuroFilter(min_cutoff=1.0, beta=0.03)
        else:
            fx = OneEuroFilter(min_cutoff=0.35, beta=0.015)
            fy = OneEuroFilter(min_cutoff=0.35, beta=0.015)
        landmark_filters.extend([fx, fy])

    # Bộ lọc One-Euro riêng cho 3 góc quay đầu 3D PnP (giữ số đo góc và trục hiển thị êm ái, zero-jitter)
    filter_yaw = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    filter_pitch = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    filter_roll = OneEuroFilter(min_cutoff=0.5, beta=0.02)

    while True:
        frame_count += 1
        if args.synthetic:
            frame = np.full((480, 640, 3), 35, dtype=np.uint8)
            sim_frame_idx += 1
            cycle = (sim_frame_idx % 600) / 30.0
            if cycle < 8.0:
                mock_ear = 0.30 + 0.02 * math.sin(cycle * 3)
                mock_mar = 0.16 + 0.01 * math.cos(cycle * 2)
                mock_yaw = 2.0 * math.sin(cycle)
            elif cycle < 13.0:
                mock_ear = 0.08  # Nhắm mắt
                mock_mar = 0.16
                mock_yaw = 0.0
            elif cycle < 16.0:
                mock_ear = 0.28
                mock_mar = 0.58  # Ngáp há miệng
                mock_yaw = 0.0
            else:
                mock_ear = 0.28
                mock_mar = 0.16
                mock_yaw = 35.0  # Quay đầu
            mock_pitch = 2.0
            mock_roll = 0.0
            time.sleep(0.033)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            # Lật gương ngang cho tự nhiên như gương chiếu hậu
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # 1. Thu thập điểm mốc và bám khuôn mặt Landmark-Driven (MediaPipe / OpenFace)
        mp_landmarks_px = None
        landmarks_px = None

        if not args.synthetic:
            # 1. Định vị khuôn mặt toàn cảnh (Full-Frame Face Detection & Tracking)
            # MediaPipe Teacher hoặc Haar Cascade luôn là nguồn định vị mặt chuẩn xác trên khung hình 640x480:
            detected_face_info = None

            if mp_engine.is_loaded:
                # MediaPipe trích xuất khuôn mặt và 22 điểm Ground-Truth
                mp_landmarks_px, mp_face_info = mp_engine.extract(frame)
                if mp_landmarks_px is not None:
                    last_mp_landmarks_px = mp_landmarks_px
                if mp_face_info is not None:
                    detected_face_info = mp_face_info

            if detected_face_info is None and face_tracker.enabled:
                # Fallback nếu không có MediaPipe: Dò bằng Haar Cascade đa tầng
                detected_face_info = face_tracker.detect_face_haar(frame)

            if detected_face_info is None and last_tracked_landmarks is not None and face_tracker.enabled:
                # Fallback quán tính ngắn hạn nếu cả 2 bộ dò tạm thời miss 1 frame
                detected_face_info = face_tracker.get_target_from_landmarks(last_tracked_landmarks)

            # Cập nhật khung cắt 1:1 theo vị trí khuôn mặt thực tế
            if face_tracker.enabled:
                if detected_face_info is not None:
                    face_cx, face_cy, face_size = detected_face_info
                    cropped_square, crop_box, crop_size, is_tracking = face_tracker.update_with_target(
                        face_cx, face_cy, face_size, frame
                    )
                else:
                    # Giữ quán tính ổn định (Inertial Hold 1.5s), không bao giờ co nhỏ hay trôi dạt
                    cropped_square, crop_box, crop_size, is_tracking = face_tracker.update_idle(frame)
            else:
                cropped_square, crop_box, crop_size, is_tracking = face_tracker.update_idle(frame)

            # 2. Suy luận 22 điểm mốc theo Chế Độ
            if active_mode == "MEDIAPIPE" and mp_engine.is_loaded and mp_landmarks_px is not None:
                landmarks_px = mp_landmarks_px
            else:
                # Chế độ TINYDRIVER INT8 (Thực thi mô hình Edge AI)
                if ai_model.is_loaded:
                    gray_square_tmp = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
                    # Nếu vùng mặt tối (mean < 95.0), áp dụng CLAHE để chi tiết mắt/mũi/miệng sắc nét
                    if np.mean(gray_square_tmp) < 95.0:
                        clahe_face = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                        gray_square_tmp = clahe_face.apply(gray_square_tmp)
                    gray_96x96_tmp = cv2.resize(gray_square_tmp, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    landmarks_norm = ai_model.predict(gray_96x96_tmp)
                    if landmarks_norm is not None:
                        x1, y1, x2, y2 = crop_box
                        raw_px = np.zeros_like(landmarks_norm)
                        raw_px[:, 0] = x1 + landmarks_norm[:, 0] * crop_size
                        raw_px[:, 1] = y1 + landmarks_norm[:, 1] * crop_size

                        # Lọc 22 điểm mốc bằng One-Euro Filter (Triệt tiêu rung giật, Zero-Lag)
                        now_ts = time.time()
                        landmarks_px = np.zeros_like(raw_px)
                        for pt_i in range(NUM_LANDMARKS):
                            landmarks_px[pt_i, 0] = landmark_filters[pt_i * 2].filter(raw_px[pt_i, 0], now_ts)
                            landmarks_px[pt_i, 1] = landmark_filters[pt_i * 2 + 1].filter(raw_px[pt_i, 1], now_ts)
                    else:
                        landmarks_px = None
                else:
                    landmarks_px = None
        else:
            crop_size = min(w, h)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            crop_box = (x1, y1, x1 + crop_size, y1 + crop_size)
            cropped_square = frame[y1:y1 + crop_size, x1:x1 + crop_size]
            is_tracking = True
            landmarks_px = None

        pnp_data = None
        if landmarks_px is not None:
            ear_l = compute_ear(landmarks_px, LEFT_EYE_PTS)
            ear_r = compute_ear(landmarks_px, RIGHT_EYE_PTS)
            ear = (ear_l + ear_r) / 2.0
            mar = compute_mar(landmarks_px, MOUTH_PTS)
            raw_yaw, raw_pitch, raw_roll, pnp_res, cam_mat, dist_c = solve_head_pose_pnp(landmarks_px, w, h)
            now_ts = time.time()
            yaw = float(filter_yaw.filter(raw_yaw, now_ts))
            pitch = float(filter_pitch.filter(raw_pitch, now_ts))
            roll = float(filter_roll.filter(raw_roll, now_ts))
            pnp_data = (pnp_res[0], pnp_res[1], cam_mat, dist_c) if pnp_res is not None else None
            last_tracked_landmarks = landmarks_px
        elif mp_landmarks_px is not None:
            ear_l = compute_ear(mp_landmarks_px, LEFT_EYE_PTS)
            ear_r = compute_ear(mp_landmarks_px, RIGHT_EYE_PTS)
            ear = (ear_l + ear_r) / 2.0
            mar = compute_mar(mp_landmarks_px, MOUTH_PTS)
            raw_yaw, raw_pitch, raw_roll, pnp_res, cam_mat, dist_c = solve_head_pose_pnp(mp_landmarks_px, w, h)
            now_ts = time.time()
            yaw = float(filter_yaw.filter(raw_yaw, now_ts))
            pitch = float(filter_pitch.filter(raw_pitch, now_ts))
            roll = float(filter_roll.filter(raw_roll, now_ts))
            pnp_data = (pnp_res[0], pnp_res[1], cam_mat, dist_c) if pnp_res is not None else None
            last_tracked_landmarks = mp_landmarks_px
        else:
            ear_l = 0.28
            ear_r = 0.28
            ear = mock_ear if args.synthetic else 0.28
            mar = mock_mar if args.synthetic else 0.18
            yaw = mock_yaw if args.synthetic else 0.0
            pitch = mock_pitch if args.synthetic else 0.0
            roll = mock_roll if args.synthetic else 0.0

        # Cập nhật tensor 96x96 Grayscale cho HUD Preview (dùng INTER_LINEAR giữ độ tương phản viền mí mắt)
        gray_square = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
        gray_96x96 = cv2.resize(gray_square, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # 2. Cập nhật Máy Trạng Thái ADAS
        adas_controller.update(ear, mar, yaw, pitch)

        # 3. Đo FPS
        now = time.time()
        fps = 1.0 / max(1e-5, (now - t_prev))
        t_prev = now

        # 4. In kết quả chẩn đoán chuyên sâu 22 điểm ra Terminal (Console Debugger)
        if mp_landmarks_px is None and last_mp_landmarks_px is not None:
            mp_landmarks_px = last_mp_landmarks_px

        terminal_logger.log(
            frame_idx=frame_count,
            fps=fps,
            active_mode=active_mode,
            is_tracking=is_tracking,
            crop_box=crop_box,
            landmarks_px=landmarks_px,
            mp_landmarks_px=mp_landmarks_px,
            ear_l=ear_l,
            ear_r=ear_r,
            ear=ear,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            adas_ctrl=adas_controller,
            frame_w=w,
            frame_h=h
        )

        # 5. Vẽ HUD Màn Hình Đôi Cyberpunk (960x480)
        hud_canvas = draw_hud(frame, crop_box, landmarks_px, gray_96x96, ear, mar, yaw, pitch, roll,
                              adas_controller, fps, ai_model.is_loaded, is_tracking,
                              active_mode=active_mode, log_mode=terminal_logger.mode, pnp_data=pnp_data,
                              mp_landmarks_px=mp_landmarks_px)

        cv2.imshow("TinyDriver ADAS - Local Model Tester (Project 13)", hud_canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('m'):
            if mp_engine.is_loaded:
                active_mode = "TINYDRIVER" if active_mode == "MEDIAPIPE" else "MEDIAPIPE"
                mode_desc = "MediaPipe Ground-Truth 22-PTS" if active_mode == "MEDIAPIPE" else "TinyDriverNet INT8 Edge"
                last_tracked_landmarks = None
                for f in landmark_filters:
                    f.reset()
                filter_yaw.reset()
                filter_pitch.reset()
                filter_roll.reset()
                print(f"🔄 [Chế Độ Suy Luận] Đã chuyển sang: {mode_desc}")
            else:
                print("⚠️ MediaPipe không khả dụng để chuyển đổi.")
        elif key == ord('f'):
            face_tracker.enabled = not face_tracker.enabled
            status_str = "BẬT (Bám mặt tự động)" if face_tracker.enabled else "TẮT (Cắt tâm cố định)"
            print(f"🎯 [Face Tracking] Đã chuyển chế độ: {status_str}")
        elif key == ord('d'):
            terminal_logger.toggle_mode()
        elif key == ord('p'):
            # Chụp nhanh và in chi tiết toàn bộ 22 điểm mốc ra Terminal ngay lập tức
            if active_mode == "TINYDRIVER" and mp_engine.is_loaded and mp_landmarks_px is None and not args.synthetic:
                mp_landmarks_px, _ = mp_engine.extract(frame)
                if mp_landmarks_px is not None:
                    last_mp_landmarks_px = mp_landmarks_px
                elif last_mp_landmarks_px is not None:
                    mp_landmarks_px = last_mp_landmarks_px
            terminal_logger.log(
                frame_idx=frame_count, fps=fps, active_mode=active_mode, is_tracking=is_tracking,
                crop_box=crop_box, landmarks_px=landmarks_px, mp_landmarks_px=mp_landmarks_px,
                ear_l=ear_l, ear_r=ear_r, ear=ear, mar=mar, yaw=yaw, pitch=pitch, roll=roll,
                adas_ctrl=adas_controller, force=True, frame_w=w, frame_h=h
            )
        elif key == ord('r'):
            adas_controller.recalibrate()

    if cap:
        cap.release()
    mp_engine.close()
    cv2.destroyAllWindows()
    print("👋 Đã dừng chương trình kiểm thử.")


if __name__ == "__main__":
    main()
