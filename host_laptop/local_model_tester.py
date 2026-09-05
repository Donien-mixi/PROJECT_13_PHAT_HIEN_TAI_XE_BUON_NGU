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

# Hỗ trợ nạp MediaPipe Face Mesh (Teacher / Ground-Truth)
HAS_MEDIAPIPE = False
mp_face_mesh = None
try:
    import mediapipe as mp
    mp_face_mesh = getattr(mp, 'solutions', None)
    if mp_face_mesh is not None and hasattr(mp_face_mesh, 'face_mesh'):
        mp_face_mesh = mp_face_mesh.face_mesh
        HAS_MEDIAPIPE = True
    else:
        from mediapipe.python.solutions import face_mesh as mp_fm
        mp_face_mesh = mp_fm
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


class MediaPipeLandmarkExtractor:
    """Bộ trích xuất 22 điểm mốc sinh học chuẩn mực từ MediaPipe Face Mesh (Mô hình Thầy)."""

    def __init__(self):
        self.is_loaded = False
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
        if HAS_MEDIAPIPE and mp_face_mesh is not None:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.is_loaded = True
                print("💎 [MediaPipe Engine] Đã nạp thành công bộ trích xuất MediaPipe Face Mesh (22 điểm Ground-Truth).")
            except Exception as e:
                print(f"⚠️ [MediaPipe Engine] Lỗi khởi tạo Face Mesh: {e}")
                self.is_loaded = False

    def extract(self, frame_bgr):
        """Trích xuất 22 điểm mốc và bounding box chuẩn từ frame webcam."""
        if not self.is_loaded or self.face_mesh is None:
            return None, None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, None
        lms = results.multi_face_landmarks[0].landmark
        pts_px = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        for i, idx in enumerate(self.indices):
            pts_px[i, 0] = lms[idx].x * w
            pts_px[i, 1] = lms[idx].y * h

        # Tính toán tâm và kích thước khung mặt từ 22 điểm mốc
        min_x = np.min(pts_px[:, 0])
        max_x = np.max(pts_px[:, 0])
        min_y = np.min(pts_px[:, 1])
        max_y = np.max(pts_px[:, 1])
        face_cx = (min_x + max_x) / 2.0
        face_cy = (min_y + max_y) / 2.0
        face_size = max(max_x - min_x, max_y - min_y) * 1.42

        return pts_px, (face_cx, face_cy, face_size)


class FaceTracker:
    """Bộ dò tìm và bám khuôn mặt thời gian thực với bộ lọc vùng chết (Deadband) chống rung giật."""

    def __init__(self, expansion_ratio=1.35, ema_alpha=0.15, deadband_pos=6.0, deadband_size=8.0):
        self.expansion_ratio = expansion_ratio
        self.ema_alpha = ema_alpha
        self.deadband_pos = deadband_pos
        self.deadband_size = deadband_size
        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_S = None
        self.last_detection_time = 0.0
        self.is_tracking = False
        self.enabled = True
        self.cascades = []

        # Tìm các file cascade có sẵn theo thứ tự ưu tiên
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_alt2.xml'),
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt2.xml',
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        ]

        for p in candidate_paths:
            if p and os.path.exists(p):
                try:
                    cc = cv2.CascadeClassifier(p)
                    if not cc.empty():
                        self.cascades.append(cc)
                        print(f"🎯 [Face Tracker] Đã nạp bộ dò khuôn mặt: {os.path.basename(p)}")
                except Exception:
                    pass

        if not self.cascades:
            print("⚠️ [Face Tracker] Không tìm thấy file Haar Cascade. Sử dụng Center-Crop cố định.")

    def _apply_deadband_and_smooth(self, target_cx, target_cy, target_S):
        if self.smooth_cx is None:
            self.smooth_cx = float(target_cx)
            self.smooth_cy = float(target_cy)
            self.smooth_S = float(target_S)
        else:
            dcx = abs(target_cx - self.smooth_cx)
            dcy = abs(target_cy - self.smooth_cy)
            dS = abs(target_S - self.smooth_S)

            # Khóa cứng tọa độ nếu độ lệch nằm trong vùng chết (Deadband) -> Triệt tiêu hoàn toàn rung giật
            if dcx > self.deadband_pos:
                self.smooth_cx = self.ema_alpha * target_cx + (1.0 - self.ema_alpha) * self.smooth_cx
            if dcy > self.deadband_pos:
                self.smooth_cy = self.ema_alpha * target_cy + (1.0 - self.ema_alpha) * self.smooth_cy
            if dS > self.deadband_size:
                self.smooth_S = self.ema_alpha * target_S + (1.0 - self.ema_alpha) * self.smooth_S

    def _get_crop(self, frame, shift_y_ratio=0.06):
        h, w = frame.shape[:2]
        cx = self.smooth_cx
        # Dịch nhẹ tâm cy xuống dưới shift_y_ratio * S để bù trừ phần tóc/trán,
        # giúp 2 mắt nằm chính xác tại dải 35% của tensor crop (chuẩn sinh trắc học)
        cy = self.smooth_cy + (shift_y_ratio * self.smooth_S)
        S = int(round(self.smooth_S))

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

    def update_with_target(self, target_cx, target_cy, target_S, frame):
        """Cập nhật vị trí mặt từ bộ dò ngoài (MediaPipe) kết hợp lọc vùng chết."""
        h, w = frame.shape[:2]
        default_S = min(w, h)
        target_S = float(np.clip(target_S, 120, default_S))
        self.is_tracking = True
        self.last_detection_time = time.time()
        self._apply_deadband_and_smooth(target_cx, target_cy, target_S)
        return self._get_crop(frame, shift_y_ratio=0.0)

    def update_idle(self, frame):
        """Cập nhật trạng thái khi mất dấu khuôn mặt."""
        h, w = frame.shape[:2]
        now = time.time()
        if self.smooth_cx is not None and (now - self.last_detection_time < 2.0):
            self.is_tracking = True
        else:
            self.is_tracking = False
            default_S = min(w, h)
            self.smooth_cx = w / 2.0
            self.smooth_cy = h / 2.0
            self.smooth_S = float(default_S)
        return self._get_crop(frame, shift_y_ratio=0.0)

    def update(self, frame):
        h, w = frame.shape[:2]
        default_S = min(w, h)
        target_cx = w / 2.0
        target_cy = h / 2.0
        target_S = float(default_S)
        found = False

        if self.enabled and self.cascades:
            scale = 0.5
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            small_gray = clahe.apply(small_gray)

            for cc in self.cascades:
                faces = cc.detectMultiScale(
                    small_gray, scaleFactor=1.15, minNeighbors=3, minSize=(35, 35)
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    target_cx = (fx + fw / 2.0) / scale
                    target_cy = (fy + fh / 2.0) / scale
                    face_size = max(fw, fh) / scale
                    target_S = float(np.clip(face_size * self.expansion_ratio, 120, default_S))
                    found = True
                    self.last_detection_time = time.time()
                    break

        now = time.time()
        if self.enabled and (found or (now - self.last_detection_time < 2.0 and self.smooth_cx is not None)):
            self.is_tracking = True
            self._apply_deadband_and_smooth(target_cx, target_cy, target_S)
        else:
            self.is_tracking = False
            self.smooth_cx = target_cx
            self.smooth_cy = target_cy
            self.smooth_S = float(default_S)

        return self._get_crop(frame, shift_y_ratio=0.06)


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

            self.is_quantized = (self.input_details['dtype'] in [np.int8, np.uint8])
            self.is_loaded = True
            print(f"✅ ĐÃ NẠP MÔ HÌNH THÀNH CÔNG: {model_path}")
            print(f"   • Input Shape : {self.input_details['shape']} ({self.input_details['dtype'].__name__})")
            print(f"   • Output Shape: {self.output_details['shape']} ({self.output_details['dtype'].__name__})")
            print(f"   • Quantized   : {self.is_quantized}")
        except Exception as e:
            print(f"❌ [LỖI] Không thể khởi tạo mô hình TFLite: {e}")
            self.is_loaded = False

    def predict(self, face_gray_96x96: np.ndarray) -> np.ndarray:
        """Thực hiện suy luận trả về mảng (22, 2) tọa độ landmarks chuẩn hóa [0.0, 1.0]."""
        if not self.is_loaded:
            return None

        img = face_gray_96x96.astype(np.float32)

        if self.is_quantized:
            scale, zero_point = self.input_details['quantization']
            if scale == 0.0:
                scale = 1.0 / 128.0
                zero_point = 0
            # Chuẩn hóa về [-1.0, 1.0] rồi lượng tử hóa sang int8: (img - 128) / 128
            norm = (img - 128.0) / 128.0
            quant = np.clip(np.round(norm / scale) + zero_point, -128, 127).astype(np.int8)
            input_tensor = np.expand_dims(np.expand_dims(quant, axis=0), axis=-1)
        else:
            norm = (img - 128.0) / 128.0
            input_tensor = np.expand_dims(np.expand_dims(norm, axis=0), axis=-1).astype(np.float32)

        self.interpreter.set_tensor(self.input_details['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details['index'])[0]

        if self.is_quantized:
            o_scale, o_zero_point = self.output_details['quantization']
            if o_scale != 0.0:
                output_data = (output_data.astype(np.float32) - o_zero_point) * o_scale
            else:
                output_data = (output_data.astype(np.float32) + 128.0) / 255.0

        landmarks = np.clip(output_data.reshape((NUM_LANDMARKS, 2)), 0.0, 1.0)
        return landmarks


class LocalADASController:
    """Máy trạng thái ADAS phản ánh 100% logic firmware C++ trên ESP32-S3."""

    def __init__(self):
        self.calibrated = False
        self.calib_start_time = time.time()
        self.calib_duration = 3.5
        self.ear_samples = []
        self.mar_samples = []

        self.ear_threshold = 0.20
        self.mar_threshold = 0.40
        self.yaw_threshold = 25.0

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
        print("🎯 [ADAS] Bắt đầu tự hiệu chuẩn lại ngưỡng trong 3.5 giây...")

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
                self.ear_threshold = max(0.16, min(0.25, float(base_ear * 0.72)))
                self.mar_threshold = max(0.35, float(base_mar * 1.65))
                self.calibrated = True
                print(f"\n🎯 [ADAS] HIỆU CHUẨN HOÀN TẤT: EAR_thresh={self.ear_threshold:.2f}, MAR_thresh={self.mar_threshold:.2f}")
            else:
                self.current_state = f"CALIBRATING ({self.calib_duration - elapsed:.1f}s)"
                return

        # 2. Kiểm tra Buồn Ngủ (Mắt nhắm liên tục)
        if ear < self.ear_threshold:
            if self.closed_eyes_start is None:
                self.closed_eyes_start = now
            closed_duration = now - self.closed_eyes_start
            if closed_duration >= 1.5:
                self.current_state = "ALARM: MICROSLEEP!"
                self.alarm_active = True
                self.alarm_reason = f"Ngủ gật nhắm mắt {closed_duration:.1f}s"
            elif closed_duration >= 0.35:
                self.current_state = "WARNING: SLOW BLINK"
        else:
            self.closed_eyes_start = None

        # 3. Kiểm tra Ngáp / Mệt Mỏi (Há miệng)
        if mar > self.mar_threshold:
            if self.yawn_start is None:
                self.yawn_start = now
            yawn_dur = now - self.yawn_start
            if yawn_dur >= 1.2:
                self.current_state = "YAWNING DETECTED"
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

        # 4. Kiểm tra Mất Tập Trung (Quay đầu góc lớn)
        if abs(yaw) > self.yaw_threshold or abs(pitch) > 22.0:
            if self.distraction_start is None:
                self.distraction_start = now
            distract_dur = now - self.distraction_start
            if distract_dur >= 2.5:
                self.current_state = "ALARM: DISTRACTED!"
                self.alarm_active = True
                self.alarm_reason = f"Quay mặt góc {yaw:+.0f}° quá {distract_dur:.1f}s"
        else:
            self.distraction_start = None

        if not self.alarm_active and "WARNING" not in self.current_state and "YAWN" not in self.current_state and "CALIB" not in self.current_state:
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
    """Ước lượng góc xoay đầu 3D (Yaw, Pitch, Roll) bằng PnP đối chiếu với mô hình nhân trắc học."""
    # 6 điểm đối xứng: Nose Tip (19), Chin (21), Left Eye Outer (0), Right Eye Outer (9), Mouth Left (12), Mouth Right (13)
    pts_2d = np.array([
        landmarks_px[NOSE_TIP_PT],
        landmarks_px[CHIN_PT],
        landmarks_px[LEFT_EYE_PTS[0]],
        landmarks_px[RIGHT_EYE_PTS[3]], # P9: Khóe ngoài mắt phải
        landmarks_px[MOUTH_PTS[0]],
        landmarks_px[MOUTH_PTS[1]]
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
        return 0.0, 0.0, 0.0, None, camera_matrix, dist_coeffs

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    return yaw, pitch, roll, (rvec, tvec), camera_matrix, dist_coeffs


def draw_hud(frame, crop_box, landmarks_px, gray_96x96, ear, mar, yaw, pitch, roll, adas_ctrl, fps, is_real_ai, is_tracking, active_mode="TINYDRIVER"):
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

    box_tag = "🎯 BÁM MẶT 1:1" if is_tracking else "CẮT TÂM CỐ ĐỊNH"
    cv2.putText(frame, box_tag, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 1)

    # 2. Vẽ 22 điểm mốc sinh học và 3 trục quay đầu 3D trên camera
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

        # Vẽ 3 trục quay đầu 3D từ chóp mũi (Nose Tip P19)
        nose_px, nose_py = int(landmarks_px[NOSE_TIP_PT][0]), int(landmarks_px[NOSE_TIP_PT][1])
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        axis_len = max(35.0, (x2 - x1) * 0.25)

        x_end = (int(nose_px + axis_len * math.cos(yaw_rad)),
                 int(nose_py + axis_len * math.sin(roll_rad)))
        y_end = (int(nose_px - axis_len * math.sin(roll_rad)),
                 int(nose_py + axis_len * math.cos(pitch_rad)))
        z_end = (int(nose_px + axis_len * math.sin(yaw_rad)),
                 int(nose_py - axis_len * math.sin(pitch_rad)))

        cv2.line(frame, (nose_px, nose_py), x_end, (0, 0, 255), 2)  # X (Đỏ)
        cv2.line(frame, (nose_px, nose_py), y_end, (0, 255, 0), 2)  # Y (Xanh lá)
        cv2.line(frame, (nose_px, nose_py), z_end, (255, 0, 0), 2)  # Z (Xanh dương)

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
        track_tag = "💎 BÁM MẶT: MEDIAPIPE GROUND-TRUTH" if is_tracking else "⚠️ BÁM MẶT: TÌM KIẾM KHUÔN MẶT..."
        track_color = (255, 230, 0) if is_tracking else (0, 180, 255)
    else:
        title_sub = "[REAL AI INT8]" if is_real_ai else "[MÔ PHỎNG]"
        sub_color = (0, 255, 180) # Emerald green
        track_tag = "🎯 BÁM MẶT: TINYDRIVER AI" if is_tracking else "⚠️ BÁM MẶT: TÌM KIẾM KHUÔN MẶT..."
        track_color = (0, 255, 120) if is_tracking else (0, 180, 255)

    cv2.putText(side, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
    cv2.putText(side, title_sub, (200, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sub_color, 1)
    cv2.putText(side, f"FPS: {fps:.1f} | 96x96 Grayscale Tensor", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.line(side, (20, 60), (300, 60), (50, 50, 65), 1)

    cv2.putText(side, track_tag, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, track_color, 1)

    # Hiển thị Tensor 96x96 Crop AI thu nhỏ trực tiếp (Kích thước 100x100)
    if gray_96x96 is not None:
        tensor_vis = cv2.cvtColor(gray_96x96, cv2.COLOR_GRAY2BGR)
        tensor_vis = cv2.resize(tensor_vis, (96, 96), interpolation=cv2.INTER_NEAREST)
        side[90:90+96, 20:20+96] = tensor_vis
        cv2.rectangle(side, (19, 89), (20+96, 90+96), (0, 240, 255), 1)
        cv2.putText(side, "TENSOR 96x96", (24, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Thông tin bên cạnh Tensor Crop
    cv2.putText(side, "ĐẦU VÀO AI:", (130, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)
    cv2.putText(side, "Cận cảnh 1:1", (130, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.putText(side, f"Khuôn mặt: {crop_box[2]-crop_box[0]}px", (130, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.putText(side, "Mắt/Miệng rõ nét", (130, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 150), 1)

    # --- THANH ĐO EAR (MẮT) ---
    cv2.line(side, (20, 212), (300, 212), (50, 50, 65), 1)
    is_eyes_closed = (ear < adas_ctrl.ear_threshold)
    ear_status_text = "⚠️ [NHẮM MẮT]" if is_eyes_closed else "MỞ BÌNH THƯỜNG"
    ear_text_color = (0, 0, 255) if is_eyes_closed else (0, 255, 0)
    cv2.putText(side, f"EAR (Mắt): {ear:.3f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(side, ear_status_text, (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ear_text_color, 1)

    bar_w_max = 280
    bar_ear_w = int(np.clip(ear / 0.40, 0.0, 1.0) * bar_w_max)
    bar_ear_color = (0, 0, 255) if is_eyes_closed else ((0, 220, 255) if ear < 0.25 else (0, 255, 0))
    cv2.rectangle(side, (20, 238), (20 + bar_w_max, 252), (40, 40, 50), -1)
    cv2.rectangle(side, (20, 238), (20 + bar_ear_w, 252), bar_ear_color, -1)
    thresh_ear_x = int(20 + np.clip(adas_ctrl.ear_threshold / 0.40, 0.0, 1.0) * bar_w_max)
    cv2.line(side, (thresh_ear_x, 236), (thresh_ear_x, 254), (255, 255, 255), 2)
    cv2.putText(side, f"Ngưỡng: {adas_ctrl.ear_threshold:.2f}", (thresh_ear_x - 20, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    # --- THANH ĐO MAR (MIỆNG) ---
    cv2.line(side, (20, 276), (300, 276), (50, 50, 65), 1)
    is_yawning = (mar > adas_ctrl.mar_threshold)
    mar_status_text = "⚠️ [ĐANG NGÁP]" if is_yawning else "NGẬM BÌNH THƯỜNG"
    mar_text_color = (0, 0, 255) if is_yawning else (255, 0, 255)
    cv2.putText(side, f"MAR (Miệng): {mar:.3f}", (20, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(side, mar_status_text, (160, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mar_text_color, 1)

    bar_mar_w = int(np.clip(mar / 0.80, 0.0, 1.0) * bar_w_max)
    bar_mar_color = (0, 0, 255) if is_yawning else (255, 0, 255)
    cv2.rectangle(side, (20, 302), (20 + bar_w_max, 316), (40, 40, 50), -1)
    cv2.rectangle(side, (20, 302), (20 + bar_mar_w, 316), bar_mar_color, -1)
    thresh_mar_x = int(20 + np.clip(adas_ctrl.mar_threshold / 0.80, 0.0, 1.0) * bar_w_max)
    cv2.line(side, (thresh_mar_x, 300), (thresh_mar_x, 318), (255, 255, 255), 2)
    cv2.putText(side, f"Ngưỡng: {adas_ctrl.mar_threshold:.2f}", (thresh_mar_x - 20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    # --- GÓC QUAY ĐẦU HEAD POSE ---
    cv2.line(side, (20, 340), (300, 340), (50, 50, 65), 1)
    dir_tag = "🟢 CHÍNH DIỆN"
    if yaw > 18.0:
        dir_tag = "⬅️ QUAY TRÁI"
    elif yaw < -18.0:
        dir_tag = "➡️ QUAY PHẢI"
    elif pitch > 15.0:
        dir_tag = "⬇️ CÚI ĐẦU"
    elif pitch < -15.0:
        dir_tag = "⬆️ NGỬA ĐẦU"

    pose_color = (0, 0, 255) if abs(yaw) > adas_ctrl.yaw_threshold or abs(pitch) > 22.0 else (0, 255, 255)
    cv2.putText(side, f"Góc Đầu: Yaw={yaw:+.1f}° | Pitch={pitch:+.1f}°", (20, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pose_color, 1)
    cv2.putText(side, f"Hướng nhìn: {dir_tag}", (20, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # --- TRẠNG THÁI ADAS FSM ---
    cv2.line(side, (20, 388), (300, 388), (50, 50, 65), 1)
    status_bg = (0, 0, 200) if adas_ctrl.alarm_active else ((0, 130, 255) if "WARNING" in adas_ctrl.current_state else (0, 140, 0))
    cv2.rectangle(side, (20, 398), (300, 432), status_bg, -1)
    cv2.putText(side, adas_ctrl.current_state, (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # Dòng phím bấm hướng dẫn
    cv2.putText(side, "[m] Đổi Mode | [f] Bám mặt | [r] Calib | [q] Thoát", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Kiểm thử mô hình AI TinyDriver trên Laptop với Webcam")
    parser.add_argument("--cam", type=int, default=0, help="ID cổng camera vật lý (mặc định: 0)")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn đến file .tflite tùy chỉnh")
    parser.add_argument("--synthetic", action="store_true", help="Chạy chế độ giả lập mô phỏng tài xế (không cần camera)")
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

    print("\n👉 Bảng điều khiển đang hiển thị trên màn hình (Bố cục Widescreen 960x480).")
    print(f"💎 Chế độ khởi tạo: {active_mode}")
    print("⌨️  Phím bấm:")
    print("   • 'm': Chuyển đổi giữa MediaPipe (Ground-Truth 22-pts) và TinyDriverNet (INT8 Edge)")
    print("   • 'f': Bật/Tắt chế độ tự động bám mặt (Dynamic Face Tracking)")
    print("   • 'r': Hiệu chuẩn lại ngưỡng (Recalibrate)")
    print("   • 'q' hoặc ESC: Thoát chương trình\n")

    sim_frame_idx = 0
    t_prev = time.time()

    while True:
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

        # 1. Thu thập điểm mốc và cắt khung mặt đồng bộ (Loại bỏ hoàn toàn rung giật khung vàng)
        mp_landmarks_px = None
        if not args.synthetic:
            if mp_engine.is_loaded:
                mp_landmarks_px, mp_face_info = mp_engine.extract(frame)
                if mp_face_info is not None and face_tracker.enabled:
                    face_cx, face_cy, face_size = mp_face_info
                    cropped_square, crop_box, crop_size, is_tracking = face_tracker.update_with_target(
                        face_cx, face_cy, face_size, frame
                    )
                else:
                    cropped_square, crop_box, crop_size, is_tracking = face_tracker.update_idle(frame)
            else:
                cropped_square, crop_box, crop_size, is_tracking = face_tracker.update(frame)
        else:
            crop_size = min(w, h)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            crop_box = (x1, y1, x1 + crop_size, y1 + crop_size)
            cropped_square = frame[y1:y1 + crop_size, x1:x1 + crop_size]
            is_tracking = True

        landmarks_px = None
        if active_mode == "MEDIAPIPE" and mp_engine.is_loaded and not args.synthetic:
            landmarks_px = mp_landmarks_px
            if landmarks_px is not None:
                ear_l = compute_ear(landmarks_px, LEFT_EYE_PTS)
                ear_r = compute_ear(landmarks_px, RIGHT_EYE_PTS)
                ear = (ear_l + ear_r) / 2.0
                mar = compute_mar(landmarks_px, MOUTH_PTS)
                yaw, pitch, roll, _, _, _ = solve_head_pose_pnp(landmarks_px, w, h)
            else:
                ear, mar, yaw, pitch, roll = 0.28, 0.18, 0.0, 0.0, 0.0
        else:
            # Chế độ TINYDRIVER INT8 hoặc Giả lập
            if ai_model.is_loaded:
                gray_square_tmp = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
                gray_96x96_tmp = cv2.resize(gray_square_tmp, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                landmarks_norm = ai_model.predict(gray_96x96_tmp)
                if landmarks_norm is not None:
                    x1, y1, x2, y2 = crop_box
                    landmarks_px = np.zeros_like(landmarks_norm)
                    landmarks_px[:, 0] = x1 + landmarks_norm[:, 0] * crop_size
                    landmarks_px[:, 1] = y1 + landmarks_norm[:, 1] * crop_size

                    ear_l = compute_ear(landmarks_px, LEFT_EYE_PTS)
                    ear_r = compute_ear(landmarks_px, RIGHT_EYE_PTS)
                    ear = (ear_l + ear_r) / 2.0
                    mar = compute_mar(landmarks_px, MOUTH_PTS)
                    yaw, pitch, roll, _, _, _ = solve_head_pose_pnp(landmarks_px, w, h)
                else:
                    ear, mar, yaw, pitch, roll = 0.28, 0.18, 0.0, 0.0, 0.0
            else:
                ear = mock_ear if args.synthetic else 0.28
                mar = mock_mar if args.synthetic else 0.18
                yaw = mock_yaw if args.synthetic else 0.0
                pitch = mock_pitch if args.synthetic else 0.0
                roll = mock_roll if args.synthetic else 0.0

        # Cập nhật tensor 96x96 Grayscale cho HUD Preview
        gray_square = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)
        gray_96x96 = cv2.resize(gray_square, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)

        # 2. Cập nhật Máy Trạng Thái ADAS
        adas_controller.update(ear, mar, yaw, pitch)

        # 3. Đo FPS
        now = time.time()
        fps = 1.0 / max(1e-5, (now - t_prev))
        t_prev = now

        # 4. Vẽ HUD Màn Hình Đôi Cyberpunk (960x480)
        hud_canvas = draw_hud(frame, crop_box, landmarks_px, gray_96x96, ear, mar, yaw, pitch, roll,
                              adas_controller, fps, ai_model.is_loaded, is_tracking, active_mode=active_mode)

        cv2.imshow("TinyDriver ADAS - Local Model Tester (Project 13)", hud_canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('m'):
            if mp_engine.is_loaded:
                active_mode = "TINYDRIVER" if active_mode == "MEDIAPIPE" else "MEDIAPIPE"
                mode_desc = "MediaPipe Ground-Truth 22-PTS" if active_mode == "MEDIAPIPE" else "TinyDriverNet INT8 Edge"
                print(f"🔄 [Chế Độ Suy Luận] Đã chuyển sang: {mode_desc}")
            else:
                print("⚠️ MediaPipe không khả dụng để chuyển đổi.")
        elif key == ord('f'):
            face_tracker.enabled = not face_tracker.enabled
            status_str = "BẬT (Bám mặt tự động)" if face_tracker.enabled else "TẮT (Cắt tâm cố định)"
            print(f"🎯 [Face Tracking] Đã chuyển chế độ: {status_str}")
        elif key == ord('r'):
            adas_controller.recalibrate()

    if cap:
        cap.release()
    print("👋 Đã dừng chương trình kiểm thử.")


if __name__ == "__main__":
    main()
