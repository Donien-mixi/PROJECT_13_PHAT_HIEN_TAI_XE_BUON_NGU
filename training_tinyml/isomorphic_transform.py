"""
Isomorphic Preprocessing & Landmark Transformation.
Guarantees 100% aspect ratio preservation (scale_x == scale_y)
from training dataset to real-time ESP32-S3 inference.

Enhanced with Translation & Scale Jitter to eliminate Mean Face / Static Landmark Collapse!
"""

import cv2
import numpy as np
try:
    from config import IMAGE_WIDTH, IMAGE_HEIGHT, BBOX_EXPANSION_RATIO
except (ImportError, ValueError):
    from .config import IMAGE_WIDTH, IMAGE_HEIGHT, BBOX_EXPANSION_RATIO


def compute_canonical_anchor(pts_px):
    """
    [v2.0.5 - DỒN MỎ NEO CHÀN] Tính (cx, cy, S) hộp canonical từ 22 điểm pixel.
    Đồng bộ 1 nơi duy nhất cho: canonical_face_crop (train/label) và
    host_laptop (get_target_from_landmarks / detect_face_haar / camera_streamer).

    Công thức:
      d_eyes    = khoảng cách tâm 2 mắt
      d_eye_nose= max(nose_y - eye_y, 0.45*d_eyes, 1.0)
      d_eye_chin= max(chin_y - eye_y, d_eye_nose, 1.0)   [MỚI v2.0.5]
      h_skull   = max(d_eye_nose*2.10, d_eyes*1.40, d_eye_chin/1.20)  [MỚI]
      S         = 2.05 * h_skull (số chẵn)
      cx        = (eye_x + nose_x) / 2
      cy        = eye_y + 0.32 * h_skull

    Term d_eye_chin/1.20 đến từ điều kiện "đáy hộp phải chứa được cằm CÓ DƯ
    an toàn":
      đáy = cy + S/2 = eye_y + 0.32h + 1.025h >= chin_y + 0.07*S
      => h >= d_eye_chin / 1.20  (cho cằm dư ~7% chiều cao hộp)
    [LỖI v2.0.5a ĐÃ SỬA: dùng /1.345 làm cằm CHẠM ĐÚNG biên (margin=0)
    -> vẫn bị gate margin loại oan 33-39% mẫu — giữ nguyên đếm rejection!]
    Với mặt nhắm miệng bình thường (d_eye_chin ~ 2.2*d_eye_nose) term này
    KHÔNG chiếm ưu tiên (2.1*d_eye_nose vẫn lớn hơn) -> layout không đổi cho
    mặt chính diện; chỉ TỰ GIÃN khi miệng há to (ngáp) -> cằm luôn nằm trong
    hộp với biên an toàn 7%.
    """
    pts = np.asarray(pts_px, dtype=np.float32)
    eye_left_center = np.mean(pts[:6, :], axis=0)
    eye_right_center = np.mean(pts[6:12, :], axis=0)
    eyes_center = (eye_left_center + eye_right_center) / 2.0

    eye_x = float(eyes_center[0])
    eye_y = float(eyes_center[1])
    nose_x = float(pts[19, 0])
    nose_y = float(pts[19, 1])
    chin_y = float(pts[21, 1])

    d_eyes = float(np.linalg.norm(eye_right_center - eye_left_center))
    d_eye_nose = float(max(nose_y - eye_y, d_eyes * 0.45, 1.0))
    d_eye_chin = float(max(chin_y - eye_y, d_eye_nose, 1.0))

    # [v2.0.5b] /1.20 cho cằm dư 7% chiều cao hộp (bản /1.345 làm cằm chạm biên)
    h_skull = float(max(d_eye_nose * 2.10, d_eyes * 1.40, d_eye_chin / 1.20))
    cx = float((eye_x + nose_x) / 2.0)
    cy = float(eye_y + 0.32 * h_skull)
    return cx, cy, h_skull


def canonical_face_crop(image, landmarks_px, jitter=False):
    """
    Chuẩn hóa hình học khuôn mặt Canonical Anatomical Anchor (InsightFace & MediaPipe).
    Đo đạc cấu trúc xương sọ cứng bất biến (Upper Skull Rigid Anchor):
        - Khoảng cách giữa 2 mắt d_eyes và khoảng cách từ mắt đến chóp mũi d_eye_nose.
        - Chiều cao sọ chuẩn H_ref = max(d_eye_nose * 2.10, d_eyes * 1.40).
        - Kích thước hộp vuông S = round(H_ref * 2.05) (đảm bảo bao trọn từ trán đến cằm khi ngáp).
        - Tâm hộp vuông: cx = (eye_x + nose_x) / 2.0, cy = eye_y + 0.32 * H_ref.
    
    Khi jitter=True (chế độ huấn luyện):
        Áp dụng Micro-Translation Jitter (dx, dy ngẫu nhiên +/- 2.5% S ~ 2px)
        và Micro-Scale Jitter (0.97 - 1.03) để mô hình hấp thụ nhiễu sub-pixel của detector
        mà KHÔNG làm phá vỡ vị trí chuẩn hóa sinh học (triệt tiêu 100% Mean Face Collapse)!
    """
    h_img, w_img = image.shape[:2]
    pts = np.array(landmarks_px, dtype=np.float32)

    # [v2.0.5] Mỏ neo canonical dùng chung (có term cằm — chống clip P21 khi ngáp)
    cx, cy, h_skull = compute_canonical_anchor(pts)
    square_size = int(round(h_skull * 2.05))
    if square_size % 2 != 0:
        square_size += 1

    # Micro-Jitter (Chống nhiễu detector nhưng giữ nguyên vùng mắt/miệng chuẩn MediaPipe)
    if jitter:
        shift_x = float(np.random.uniform(-0.025, 0.025) * square_size)
        shift_y = float(np.random.uniform(-0.025, 0.025) * square_size)
        scale_j = float(np.random.uniform(0.97, 1.03))
        square_size = int(round(square_size * scale_j))
        if square_size % 2 != 0:
            square_size += 1
        cx += shift_x
        cy += shift_y

    x0 = int(round(cx - square_size / 2.0))
    y0 = int(round(cy - square_size / 2.0))
    x1 = x0 + square_size
    y1 = y0 + square_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w_img)
    pad_bottom = max(0, y1 - h_img)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(image.shape) == 3:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
        crop_x0 = x0 + pad_left
        crop_y0 = y0 + pad_top
        crop = padded_img[crop_y0:crop_y0 + square_size, crop_x0:crop_x0 + square_size]
    else:
        crop = image[y0:y1, x0:x1]

    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop

    crop_resized = cv2.resize(crop_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    crop_resized = np.expand_dims(crop_resized, axis=-1)

    norm_landmarks = np.zeros_like(pts, dtype=np.float32)
    norm_landmarks[:, 0] = (pts[:, 0] - x0) / float(square_size)
    norm_landmarks[:, 1] = (pts[:, 1] - y0) / float(square_size)

    # Đảm bảo kẹp tọa độ an toàn trong dải khả kiến
    norm_landmarks = np.clip(norm_landmarks, 0.0, 1.0)

    transform_meta = {
        'x0': x0,
        'y0': y0,
        'square_size': square_size
    }
    return crop_resized, norm_landmarks, transform_meta


def square_crop_and_resize(image, bbox, landmarks=None, shift_y_ratio=0.0):
    """
    Expands a bounding box into a symmetric square (1:1), crops the region with padding
    if needed, and resizes to (IMAGE_WIDTH, IMAGE_HEIGHT) Grayscale.
    Nếu landmarks 22 điểm được truyền vào, ưu tiên sử dụng canonical_face_crop.
    """
    if landmarks is not None and len(landmarks) == 22:
        return canonical_face_crop(image, landmarks)

    h_img, w_img = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    box_w = xmax - xmin
    box_h = ymax - ymin
    cx = xmin + box_w / 2.0
    # Căn chỉnh tâm sọ mặt chuẩn (phù hợp với vị trí mắt ở y=0.34)
    base_cy = ymin + (box_h * 0.44 if shift_y_ratio == 0.0 else box_h / 2.0)
    cy = base_cy + (shift_y_ratio * max(box_w, box_h))

    square_size = int(round(max(box_w, box_h) * BBOX_EXPANSION_RATIO))
    if square_size % 2 != 0:
        square_size += 1

    x0 = int(round(cx - square_size / 2.0))
    y0 = int(round(cy - square_size / 2.0))
    x1 = x0 + square_size
    y1 = y0 + square_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w_img)
    pad_bottom = max(0, y1 - h_img)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(image.shape) == 3:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
        crop_x0 = x0 + pad_left
        crop_y0 = y0 + pad_top
        crop = padded_img[crop_y0:crop_y0 + square_size, crop_x0:crop_x0 + square_size]
    else:
        crop = image[y0:y1, x0:x1]

    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop

    crop_resized = cv2.resize(crop_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    crop_resized = np.expand_dims(crop_resized, axis=-1)

    norm_landmarks = None
    if landmarks is not None:
        pts = np.array(landmarks, dtype=np.float32)
        norm_landmarks = np.zeros_like(pts)
        norm_landmarks[:, 0] = (pts[:, 0] - x0) / float(square_size)
        norm_landmarks[:, 1] = (pts[:, 1] - y0) / float(square_size)
        norm_landmarks = np.clip(norm_landmarks, 0.0, 1.0)

    transform_meta = {
        'x0': x0,
        'y0': y0,
        'square_size': square_size
    }
    return crop_resized, norm_landmarks, transform_meta
