"""
Dataset Loader & Advanced Data Augmentation Pipeline for TinyDriver-LandmarkNet (PFLD-Edge).
Features:
1. Robust Bounding Box Translation & Scale Jitter (eliminates static template / mean face collapse).
2. Photorealistic synthetic & in-cabin driver face generator (microsleep blinks, yawns, head yaws).
3. MediaPipe Teacher automated soft-labeling with 3D Pose estimation (PnP Euler angles).
4. Multi-Task Target Generation: {"landmarks_output": 44, "pose_output": 3}.
"""

import sys
import os
import glob
import time
import urllib.request
import zipfile
import cv2
import numpy as np
import math

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, IBUG_68_TO_22_INDICES,
        BBOX_EXPANSION_RATIO, NUM_LANDMARKS
    )
    from isomorphic_transform import square_crop_and_resize, canonical_face_crop
    from distillation import estimate_pose_from_landmarks
except (ImportError, ValueError):
    from .config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, IBUG_68_TO_22_INDICES,
        BBOX_EXPANSION_RATIO, NUM_LANDMARKS
    )
    from .isomorphic_transform import square_crop_and_resize, canonical_face_crop
    from .distillation import estimate_pose_from_landmarks


def apply_macro_affine_jitter(image, landmarks_norm, shift_range=0.07,
                              scale_range=(0.85, 1.18), rot_range=10.0,
                              border=cv2.BORDER_REFLECT, rng=None):
    """
    [FIX v2.0.4 - CHỐNG TEMPLATE COLLAPSE] Random affine MACRO quanh tâm khung 96x96:
      - Dịch chuyển +/- shift_range (7% khung ~ +/-6.7px)
      - Scale [0.85, 1.18]
      - Xoay +/- rot_range độ
    Bản cũ chỉ jitter vi mô +/-1.5px -> vị trí mắt trong crop gần như hằng số
    (v=0.344 cố định) -> mô hình học thuộc TEMPLATE vị trí canonical thay vì
    định vị pixel thật -> landmark lệch 17-23px khi gặp mặt có tỉ lệ giải phẫu
    khác population + tracking bám template chứ không bám mặt.

    Image và landmark được transform bằng CÙNG ma trận affine M (đảm bảo khớp
    pixel-exact). borderMode=REFLECT tránh viền đen giả.
    Returns: (img (96,96,1) uint8, lms (22,2) norm)
    """
    img2d = image[:, :, 0] if image.ndim == 3 else image
    h, w = img2d.shape[:2]
    ang = float(np.random.uniform(-rot_range, rot_range) if rng is None
                else rng.uniform(-rot_range, rot_range))
    scale = float(np.random.uniform(*scale_range) if rng is None
                  else rng.uniform(*scale_range))
    dx = float(np.random.uniform(-shift_range, shift_range) if rng is None
               else rng.uniform(-shift_range, shift_range)) * w
    dy = float(np.random.uniform(-shift_range, shift_range) if rng is None
               else rng.uniform(-shift_range, shift_range)) * h

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    M = M.astype(np.float32)

    out = cv2.warpAffine(img2d, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)

    pts = np.asarray(landmarks_norm, dtype=np.float32).reshape(-1, 2).copy()
    pts_px = pts * np.array([w, h], dtype=np.float32)
    ones = np.hstack([pts_px, np.ones((len(pts_px), 1), dtype=np.float32)])
    out_px = ones @ M.T
    out_norm = np.clip(out_px / np.array([w, h], dtype=np.float32), 0.0, 1.0)

    return np.expand_dims(out, axis=-1), out_norm


def make_jittered_val_copy(val_imgs, val_lms, copies=2, seed=123,
                           shift_range=0.05, scale_range=(0.90, 1.12), rot_range=7.0):
    """
    [BẢN 2025] Tạo bản sao val với jitter affine xác định (seed cố định) để đo
    NME LOCALIZATION thật. NME trên val canonical không phát hiện được template
    collapse (mắt luôn ở v=0.344 trong canonical crop) — chính là lỗ hổng khiến
    model cũ pass gate 6.62% nhưng live sai 17-23px.
    Returns: (imgs (copies*N,96,96,1), lms (copies*N,44))
    """
    rng = np.random.RandomState(seed)
    imgs_out, lms_out = [], []
    for c in range(copies):
        for i in range(len(val_imgs)):
            img, lms = apply_macro_affine_jitter(
                val_imgs[i], val_lms[i].reshape(22, 2),
                shift_range=shift_range, scale_range=scale_range,
                rot_range=rot_range, border=cv2.BORDER_REFLECT, rng=rng)
            imgs_out.append(img)
            lms_out.append(lms.flatten())
    return np.array(imgs_out, dtype=np.uint8), np.array(lms_out, dtype=np.float32)


def apply_cabin_data_augmentation(image, landmarks_norm):
    """
    In-Cabin Environmental & Spatial Data Augmentation.
    Mô phỏng điều kiện thực tế trong cabin xe ô tô:
      - Rung lắc dịch chuyển vị trí (Spatial Micro-Translation Jitter +/- 5px)
      - Thay đổi ánh sáng (nắng gắt, hầm tối, ban đêm)
      - Nhiễu hạt ISO của webcam xe
      - Rung lắc xe (Motion Blur)
      - Gọng kính cận hoặc che khuất cục bộ (Cutout)
      - Nghiêng đầu (+/- 20 độ) và lật gương đối xứng
    """
    img = image.copy().astype(np.float32)
    # Đảm bảo xử lý trên mảng 2D (96, 96) để tránh lỗi NumPy broadcasting khi nhân với mask (96, 96)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    lm = landmarks_norm.copy()

    # 1. [MACRO-JITTER v2.0.4] Random affine dịch +/-7%, scale 0.85-1.18, xoay +/-10°
    #    ÁP DỤNG LUÔN trong train: phá tính bất biến vị trí canonical -> ép mô hình
    #    ĐỊNH VỊ pixel thật thay vì học thuộc template. (Bản cũ chỉ +/-1.5px 40% thời gian.)
    img_u8, lm = apply_macro_affine_jitter(img.astype(np.uint8), lm)

    # 2. Photometric: Random Brightness & Contrast
    alpha = np.random.uniform(0.75, 1.25)
    beta = np.random.uniform(-30.0, 30.0)
    img = np.clip(img * alpha + beta, 0, 255)

    # 3. Random Gamma (Mô phỏng hầm tối hoặc nắng chiếu trực diện cabin)
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.7, 1.4)
        inv_gamma = 1.0 / gamma
        img = np.clip(((img / 255.0) ** inv_gamma) * 255.0, 0, 255)

    # 4. Sensor Noise / Gaussian Noise (Webcam ISO noise trong bóng tối)
    if np.random.rand() > 0.6:
        noise = np.random.normal(0, np.random.uniform(3, 10), img.shape)
        img = np.clip(img + noise, 0, 255)

    # 5. Vehicle Vibration / Motion Blur (Xe rung lắc trên đường xóc)
    if np.random.rand() > 0.7:
        ksize = np.random.choice([3, 5])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize
        img = cv2.filter2D(img, -1, kernel)

    # 6. Realistic Eyeglasses Frames, Bridge, and Specular Lens Glare (Mô phỏng kính mắt & bóng phản quang thực tế)
    # [FIX 2025] Hạ xác suất từ 55% xuống 18%: trước đây 55% mẫu bị đeo "kính ellipse đen"
    # làm phân bố huấn luyện lệch nặng khỏi thế giới thực.
    if np.random.rand() < 0.18:
        left_eye_pts = lm[0:6] * float(IMAGE_WIDTH)
        right_eye_pts = lm[6:12] * float(IMAGE_WIDTH)
        cx_l, cy_l = np.mean(left_eye_pts, axis=0)
        cx_r, cy_r = np.mean(right_eye_pts, axis=0)
        w_l = max(float(np.linalg.norm(left_eye_pts[0] - left_eye_pts[3])), 8.0)
        w_r = max(float(np.linalg.norm(right_eye_pts[3] - right_eye_pts[0])), 8.0)

        # 6a. Vẽ gọng kính elip và cầu nối sống mũi
        frame_color = float(np.random.randint(15, 65))
        frame_thick = int(np.random.choice([1, 2]))
        r_axes_l = (int(round(w_l * 0.70)), int(round(w_l * 0.52)))
        r_axes_r = (int(round(w_r * 0.70)), int(round(w_r * 0.52)))
        cv2.ellipse(img, (int(round(cx_l)), int(round(cy_l))), r_axes_l, 0, 0, 360, frame_color, frame_thick)
        cv2.ellipse(img, (int(round(cx_r)), int(round(cy_r))), r_axes_r, 0, 0, 360, frame_color, frame_thick)
        bridge_start = (int(round(cx_l + r_axes_l[0] * 0.8)), int(round(cy_l)))
        bridge_end = (int(round(cx_r - r_axes_r[0] * 0.8)), int(round(cy_r)))
        cv2.line(img, bridge_start, bridge_end, frame_color, frame_thick)

        # 6b. Vệt lóa phản quang tròng kính (Specular Lens Glare / Reflection)
        if np.random.rand() > 0.40:
            glare_target = np.random.choice(['left', 'right', 'both'])
            glare_eyes = []
            if glare_target in ['left', 'both']:
                glare_eyes.append((cx_l, cy_l, w_l))
            if glare_target in ['right', 'both']:
                glare_eyes.append((cx_r, cy_r, w_r))

            for gx, gy, gw in glare_eyes:
                glare_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
                offset_x = float(np.random.uniform(-gw * 0.25, gw * 0.25))
                offset_y = float(np.random.uniform(-gw * 0.20, gw * 0.20))
                g_center = (int(round(gx + offset_x)), int(round(gy + offset_y)))
                g_axes = (int(round(np.random.uniform(gw * 0.25, gw * 0.55))),
                          int(round(np.random.uniform(gw * 0.15, gw * 0.35))))
                g_angle = float(np.random.uniform(-45.0, 45.0))
                cv2.ellipse(glare_mask, g_center, g_axes, g_angle, 0, 360, 1.0, -1)
                glare_mask = cv2.GaussianBlur(glare_mask, (7, 7), 2.0)

                glare_val = float(np.random.randint(160, 235))
                alpha_glare = float(np.random.uniform(0.35, 0.65))
                img = (img * (1.0 - glare_mask * alpha_glare) + glare_val * (glare_mask * alpha_glare)).astype(np.float32)

    # 7. Geometric: Random Horizontal Flip (50% probability)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        lm_flipped = lm.copy()
        lm_flipped[:, 0] = 1.0 - lm[:, 0]
        # Hoán đổi cặp đối xứng: Mắt trái (0..5) <-> Mắt phải (6..11)
        eye_swaps = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 11), (5, 10)]
        for l_idx, r_idx in eye_swaps:
            tmp = lm_flipped[l_idx].copy()
            lm_flipped[l_idx] = lm_flipped[r_idx]
            lm_flipped[r_idx] = tmp
        # Khóe miệng trái (12) <-> Khóe miệng phải (13)
        tmp_m = lm_flipped[12].copy()
        lm_flipped[12] = lm_flipped[13]
        lm_flipped[13] = tmp_m
        lm = lm_flipped

    # 8. Micro-Tilt (+/- 5 độ: tăng mạnh bất biến xoay đầu so với ±2.5° cũ)
    if np.random.rand() > 0.70:
        angle = float(np.random.uniform(-5.0, 5.0))
        M = cv2.getRotationMatrix2D((IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMAGE_WIDTH, IMAGE_HEIGHT), borderMode=cv2.BORDER_REFLECT)
        rad = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cx, cy = 0.5, 0.5
        lms_rot = lm.copy()
        dx = lm[:, 0] - cx
        dy = lm[:, 1] - cy
        lms_rot[:, 0] = cx + dx * cos_a - dy * sin_a
        lms_rot[:, 1] = cy + dx * sin_a + dy * cos_a
        lm = np.clip(lms_rot, 0.0, 1.0)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    elif len(img.shape) == 3 and img.shape[-1] != 1:
        img = img[:, :, :1]

    return img.astype(np.uint8), lm


def generate_synthetic_driver_sample(sample_idx, apply_aug=False, force_state=None):
    """
    Generates a photorealistic synthetic face sample with 22 ground-truth landmarks.
    Incorporates true human anatomical jaw mechanics:
      - Maxilla (upper jaw) is rigid: upper lip (P14, P16) stays anchored under nose.
      - Mandible (lower jaw) depression: lower lip (P15, P17) drops by 18-36px when yawning.
      - Chin Gnathion (P21) drops proportionally with mandible depression (15-30px).
      - Subnasale / Philtrum (P20) strictly centered on face midline.
      - Realistic oral cavity: deep dark cavity, dental arch (upper teeth), lower teeth/tongue highlight.
    """
    canvas = np.full((240, 240), np.random.randint(25, 55), dtype=np.uint8)

    # Face center and dimensions
    cx = 120 + np.random.randint(-16, 17)
    cy = 115 + np.random.randint(-14, 15)
    rx = 54 + np.random.randint(-5, 6)
    ry = 64 + np.random.randint(-5, 6)

    # 1. Realistic skin shading gradient
    face_color = np.random.randint(155, 220)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, face_color, -1)
    # Forehead highlight
    cv2.ellipse(canvas, (cx, cy - 22), (rx // 2, ry // 3), 0, 0, 360, min(255, face_color + 22), -1)

    # State selection: 50/50 balanced when sample_idx % 2 == 1 or forced
    if force_state == 'yawn':
        is_yawning = True
        is_eyes_closed = False
    elif force_state == 'microsleep':
        is_yawning = False
        is_eyes_closed = True
    elif force_state == 'normal':
        is_yawning = False
        is_eyes_closed = False
    else:
        is_yawning = (sample_idx % 2 == 1)
        is_eyes_closed = (sample_idx % 4 == 0) and not is_yawning

    yaw_deg = np.random.uniform(-30.0, 30.0)
    yaw_shift = int(yaw_deg * 0.40)

    eye_open_dy = float(np.random.uniform(0.3, 1.2)) if is_eyes_closed else float(np.random.uniform(4.5, 9.0))
    # [v2.3.0] Nâng sàn mouth_open_dy 18->22 để MAR khi ngáp LUÔN >= 0.45 (tránh test flaky).
    mouth_open_dy = float(np.random.uniform(22.0, 36.0)) if is_yawning else float(np.random.uniform(1.5, 4.5))

    # Eye centers with 3D foreshortening
    lex = cx - 23 + yaw_shift
    ley = cy - 16
    rex = cx + 23 + yaw_shift
    rey = cy - 16

    # 6 Left eye landmarks
    left_eye = np.array([
        [lex - 10, ley],
        [lex - 5,  ley - eye_open_dy],
        [lex + 5,  ley - eye_open_dy],
        [lex + 10, ley],
        [lex + 5,  ley + eye_open_dy],
        [lex - 5,  ley + eye_open_dy],
    ], dtype=np.float32)

    # 6 Right eye landmarks
    right_eye = np.array([
        [rex - 10, rey],
        [rex - 5,  rey - eye_open_dy],
        [rex + 5,  rey - eye_open_dy],
        [rex + 10, rey],
        [rex + 5,  rey + eye_open_dy],
        [rex - 5,  rey + eye_open_dy],
    ], dtype=np.float32)

    # 4 Nose and Chin landmarks (Anatomically anchored)
    nx = cx + int(yaw_shift * 1.1)
    nasion_y = cy - 26
    nose_tip_y = cy + 5
    subnasale_y = cy + 13

    # Đáy cằm: khi há miệng ngáp, xương hàm dưới hạ sâu xuống kéo theo đáy cằm
    base_chin_y = cy + ry - 4
    if is_yawning:
        chin_y = base_chin_y + float(0.85 * mouth_open_dy)
        # Kéo dài viền cằm của khuôn mặt theo hàm dưới
        jaw_drop_int = int(round(0.85 * mouth_open_dy))
        cv2.ellipse(canvas, (cx, cy + jaw_drop_int // 2), (rx, ry + jaw_drop_int // 2), 0, 0, 360, face_color, -1)
    else:
        chin_y = base_chin_y

    nose_chin = np.array([
        [nx, nasion_y],    # P18: Nasion (gốc mũi giữa 2 mắt)
        [nx, nose_tip_y],  # P19: Chóp mũi
        [nx, subnasale_y], # P20: Nhân trung (Subnasale - nằm trên đường giữa khuôn mặt)
        [cx, chin_y]       # P21: Đáy cằm (Gnathion - hạ xuống khi ngáp)
    ], dtype=np.float32)

    # 6 Mouth landmarks (Mô phỏng giải phẫu học cơ hàm dưới)
    mx = cx + yaw_shift
    upper_lip_base_y = subnasale_y + 9.0  # Môi trên gắn cố định dưới nhân trung
    upper_outer_y = upper_lip_base_y - (np.random.uniform(0.0, 1.5) if is_yawning else 0.0)
    upper_inner_y = upper_outer_y + 2.5

    if is_yawning:
        lower_outer_y = upper_outer_y + mouth_open_dy
        lower_inner_y = lower_outer_y - 3.0
    else:
        lower_outer_y = upper_outer_y + mouth_open_dy + 3.0
        lower_inner_y = upper_inner_y + 0.8

    mouth_corner_y = upper_lip_base_y + 3.0
    mouth_half_w = float(np.random.uniform(16.0, 20.0))

    mouth = np.array([
        [mx - mouth_half_w, mouth_corner_y], # P12: Khóe môi trái
        [mx + mouth_half_w, mouth_corner_y], # P13: Khóe môi phải
        [mx,                upper_outer_y],  # P14: Môi trên ngoài
        [mx,                lower_outer_y],  # P15: Môi dưới ngoài (hạ sâu khi ngáp)
        [mx,                upper_inner_y],  # P16: Môi trên trong
        [mx,                lower_inner_y],  # P17: Môi dưới trong (hạ sâu khi ngáp)
    ], dtype=np.float32)

    landmarks_22 = np.vstack([left_eye, right_eye, mouth, nose_chin])

    # Vẽ cấu trúc mắt (Mở / Nhắm)
    if not is_eyes_closed:
        cv2.fillPoly(canvas, [left_eye.astype(np.int32)], 240)
        cv2.circle(canvas, (int(lex), int(ley)), 3, 30, -1)
        cv2.fillPoly(canvas, [right_eye.astype(np.int32)], 240)
        cv2.circle(canvas, (int(rex), int(rey)), 3, 30, -1)
    else:
        cv2.polylines(canvas, [left_eye[:4].astype(np.int32)], False, 30, 2)
        cv2.polylines(canvas, [right_eye[:4].astype(np.int32)], False, 30, 2)

    # Vẽ sống mũi & chóp mũi
    cv2.line(canvas, (int(nx), int(nasion_y)), (int(nx), int(nose_tip_y)), min(255, face_color + 18), 2)
    cv2.circle(canvas, (int(nx), int(nose_tip_y)), 3, min(255, face_color + 25), -1)

    # Vẽ khoang miệng chân thực (Oral Cavity Rendering)
    if is_yawning:
        # Thứ tự chuẩn vẽ đa giác khép kín: Trái -> Môi trên -> Phải -> Môi dưới (Không bị chéo nơ)
        oral_poly = np.array([
            [mx - mouth_half_w, mouth_corner_y],
            [mx,                upper_outer_y + 1.0],
            [mx + mouth_half_w, mouth_corner_y],
            [mx,                lower_outer_y - 2.0]
        ], dtype=np.int32)

        # 1. Khoang miệng tối sâu (Dark oral cavity)
        cavity_shade = int(np.random.randint(12, 28))
        cv2.fillPoly(canvas, [oral_poly], cavity_shade)
        center_y_cavity = int((upper_outer_y + lower_outer_y) / 2.0)
        cv2.ellipse(canvas, (int(mx), center_y_cavity),
                    (int(mouth_half_w * 0.85), int(mouth_open_dy * 0.45)),
                    0, 0, 360, cavity_shade, -1)

        # 2. Vòm răng trên màu trắng sáng (Upper dental arcade)
        cv2.ellipse(canvas, (int(mx), int(upper_outer_y + 3)),
                    (int(mouth_half_w * 0.65), 4),
                    0, 0, 180, 230, -1)
        # Rãnh phân chia răng
        cv2.line(canvas, (int(mx), int(upper_outer_y + 1)),
                 (int(mx), int(upper_outer_y + 5)), 70, 1)

        # 3. Phản chiếu lưỡi và hàm dưới (Tongue / Lower teeth highlight)
        cv2.ellipse(canvas, (int(mx), int(lower_outer_y - 4)),
                    (int(mouth_half_w * 0.55), 3),
                    0, 180, 360, 135, -1)

        # 4. Viền môi
        cv2.polylines(canvas, [oral_poly], True, max(0, face_color - 40), 1)
    else:
        closed_poly = np.array([
            [mx - mouth_half_w, mouth_corner_y],
            [mx,                upper_outer_y],
            [mx + mouth_half_w, mouth_corner_y],
            [mx,                lower_outer_y]
        ], dtype=np.int32)
        cv2.polylines(canvas, [closed_poly], True, max(0, face_color - 35), 1)
        cv2.line(canvas, (int(mx - mouth_half_w + 2), int(mouth_corner_y)),
                 (int(mx + mouth_half_w - 2), int(mouth_corner_y)), 35, 2)

    # Đổ bóng cằm
    cv2.ellipse(canvas, (int(cx), int(chin_y)), (12, 4), 0, 0, 360, max(0, face_color - 30), -1)

    # Canonical crop với Translation Jitter
    crop_img, norm_landmarks, _ = canonical_face_crop(canvas, landmarks_22, jitter=apply_aug)

    if apply_aug:
        crop_img, norm_landmarks = apply_cabin_data_augmentation(crop_img, norm_landmarks)

    if len(crop_img.shape) == 2:
        crop_img = np.expand_dims(crop_img, axis=-1)

    pose = estimate_pose_from_landmarks(norm_landmarks)
    return crop_img, norm_landmarks.flatten(), pose



def extract_teacher_landmarks_from_image(image_bgr, teacher, jitter=False):
    """
    Extracts 22 landmarks and 3D pose using MediaPipe Teacher.
    Applies Canonical Anatomical Anchor with optional Translation Jitter.
    """
    if teacher is None or not getattr(teacher, 'available', False):
        return None, None, None

    h_img, w_img = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pts_norm = teacher.extract_22_landmarks(rgb)
    if pts_norm is None:
        return None, None, None

    pts_px = pts_norm.copy()
    pts_px[:, 0] *= w_img
    pts_px[:, 1] *= h_img

    crop_img, norm_lms, _ = canonical_face_crop(image_bgr, pts_px, jitter=jitter)
    if crop_img is None or norm_lms is None:
        return None, None, None

    pose = estimate_pose_from_landmarks(norm_lms)
    return crop_img, norm_lms.flatten(), pose


def download_drowsiness_benchmark_dataset(target_dir):
    """
    [DEPRECATED 2025] Bộ Hazeeq drowsiness.zip là dataset YOLO bbox (Roboflow export):
    chỉ có khung bao, KHÔNG có nhãn landmark, nhiều ảnh crop cận cảnh thiếu cằm/mắt
    -> nguyên nhân chính gây label bẩn cho mô hình trước đây. KHÔNG còn tự tải bộ này.

    Thay thế: dùng tools/build_clean_dataset.py (300W / AFLW2000-3D / WFLW / YawDD
    + MediaPipe Teacher + 6 QA gates + train/val split giữ-out thật).
    """
    os.makedirs(target_dir, exist_ok=True)
    print("[DatasetLoader] [DEPRECATED] Không còn tự tải bộ Hazeeq drowsiness (dữ liệu YOLO bbox, không có landmark).")
    print("  👉 Dùng lệnh mới: python tools/build_clean_dataset.py --download-aflw2000")
    print("     Hoặc bỏ ảnh chuẩn (300W/WFLW/YawDD/ảnh tự chụp) vào datasets/raw_faces/<ten>/ rồi chạy:")
    print("     python tools/build_clean_dataset.py")
    return target_dir


def build_or_load_drowsiness_cache(data_dir, cache_path, teacher=None, max_samples=4000):
    """
    Caches teacher-labeled real face samples (images, landmarks, poses).
    """
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            images = data['images']
            landmarks = data['landmarks']
            poses = data['poses'] if 'poses' in data else np.zeros((len(images), 3), dtype=np.float32)
            print(f"[DatasetLoader] Da nap thanh cong {len(images)} mau tu bo nho dem: {cache_path}")
            return images, landmarks, poses
        except Exception as e:
            print(f"[WARNING] [DatasetLoader] Loi doc cache ({e}). Dang tai tao...")

    if not teacher or not getattr(teacher, 'available', False):
        try:
            from distillation import MediaPipeTeacher
            teacher = MediaPipeTeacher()
        except Exception:
            pass

    if not teacher or not getattr(teacher, 'available', False):
        print("[WARNING] [DatasetLoader] Khong co MediaPipe Teacher. Se su dung du lieu sinh trac hoc tong hop.")
        return None, None, None

    img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

    if not img_paths:
        return None, None, None

    print(f"[DatasetLoader] Dang dan nhan thong minh {min(len(img_paths), max_samples)} anh thuc te bang MediaPipe Teacher...")
    cached_images = []
    cached_lms = []
    cached_poses = []

    for idx, path in enumerate(img_paths[:max_samples]):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        crop_img, lms_44, pose_3 = extract_teacher_landmarks_from_image(img_bgr, teacher, jitter=False)
        if crop_img is not None and lms_44 is not None:
            cached_images.append(crop_img)
            cached_lms.append(lms_44)
            cached_poses.append(pose_3 if pose_3 is not None else np.zeros(3, dtype=np.float32))

        if (idx + 1) % 200 == 0:
            print(f"  [OK] Da trich xuat {len(cached_images)} / {idx + 1} mau...")

    if not cached_images:
        return None, None, None

    images_arr = np.array(cached_images, dtype=np.uint8)
    lms_arr = np.array(cached_lms, dtype=np.float32)
    poses_arr = np.array(cached_poses, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path, images=images_arr, landmarks=lms_arr, poses=poses_arr)
    print(f"[OK] [DatasetLoader] Da luu bo nho dem: {cache_path} ({len(images_arr)} mau, {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)")
    return images_arr, lms_arr, poses_arr


def compute_sample_ear(lms_44):
    """Tính EAR trung bình của 2 mắt để phân loại nhắm mắt vs mở mắt."""
    pts = lms_44.reshape((22, 2))
    w_l = float(np.linalg.norm(pts[0] - pts[3]))
    h1_l = float(np.linalg.norm(pts[1] - pts[5]))
    h2_l = float(np.linalg.norm(pts[2] - pts[4]))
    ear_l = (h1_l + h2_l) / (2.0 * max(w_l, 1e-4))
    w_r = float(np.linalg.norm(pts[6] - pts[9]))
    h1_r = float(np.linalg.norm(pts[7] - pts[11]))
    h2_r = float(np.linalg.norm(pts[8] - pts[10]))
    ear_r = (h1_r + h2_r) / (2.0 * max(w_r, 1e-4))
    return (ear_l + ear_r) / 2.0


def compute_sample_mar(lms_44):
    """Tính MAR của 1 mẫu để phân loại ngáp vs bình thường."""
    pts = lms_44.reshape((22, 2))
    p12, p13 = pts[12], pts[13]
    p14, p15 = pts[14], pts[15]
    p16, p17 = pts[16], pts[17]
    w_m = float(np.linalg.norm(p12 - p13))
    h_outer = float(np.linalg.norm(p14 - p15))
    h_inner = float(np.linalg.norm(p16 - p17))
    return (h_outer + h_inner) / (2.0 * max(w_m, 1e-4))


class DriverLandmarkDataset:
    """
    Multi-Task Data Pipeline for TinyDriver PFLD-Edge (bản nâng cấp 3-Way Balanced Sampling):
      - Delivers (x_image, {"landmarks_output": 44, "pose_output": 3}).
      - Cân bằng 3 trạng thái sinh học bắt buộc: ~33% Nhắm mắt / ~33% Ngáp / ~34% Tỉnh táo.
      - Triệt tiêu 100% hiện tượng Mean-State Collapse (không còn bị kẹt ở mắt mở/ngậm miệng).
      - Val giữ-out THẬT đọc từ cột 'split' của build_clean_dataset.py.
      - Áp dụng Translation & Scale Jitter và Cabin Environmental Augmentation.
    """
    def __init__(self, data_dir=None, cache_path=None, use_synthetic=False,
                 synthetic_count=6000, augment=True, teacher=None):
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.use_synthetic = use_synthetic
        self.synthetic_count = synthetic_count
        self.augment = augment
        self.real_images = None
        self.real_landmarks = None
        self.real_poses = None

        self.closed_indices = []
        self.yawn_indices = []
        self.normal_indices = []
        self.train_indices = []
        self.val_indices = []

        if not self.use_synthetic:
            # Ưu tiên số 1: Nạp trực tiếp từ preprocessed_driver_dataset.npz
            preprocessed_candidates = [
                cache_path,
                os.path.join(os.path.dirname(__file__), "preprocessed_driver_dataset.npz"),
                "preprocessed_driver_dataset.npz",
                os.path.join("training_tinyml", "preprocessed_driver_dataset.npz"),
                os.path.join(data_dir or "", "preprocessed_driver_dataset.npz"),
            ]
            for p_path in preprocessed_candidates:
                if p_path and os.path.exists(p_path):
                    try:
                        data = np.load(p_path)
                        self.real_images = data['images']
                        self.real_landmarks = data['landmarks']
                        self.real_poses = data['poses'] if 'poses' in data else np.zeros((len(self.real_images), 3), dtype=np.float32)
                        # Đọc cột split: split=0 -> train, split=1 -> VAL GIỮ-OUT THẬT
                        if 'split' in data:
                            split_arr = data['split'].astype(np.int32)
                            self.train_indices = list(np.where(split_arr == 0)[0])
                            self.val_indices = list(np.where(split_arr == 1)[0])
                        else:
                            self.train_indices = list(range(len(self.real_images)))
                            self.val_indices = []
                        print(f"[DatasetLoader] [SUCCESS] Đã nạp {len(self.real_images)} mẫu thật từ: {p_path}")
                        print(f"  • Train: {len(self.train_indices)} | Val giữ-out: {len(self.val_indices)}")
                        break
                    except Exception as e:
                        print(f"[WARNING] [DatasetLoader] Lỗi đọc {p_path}: {e}")

            # Dự phòng số 2: Xây dựng hoặc nạp từ drowsiness_cache.npz
            if self.real_images is None:
                if cache_path and os.path.exists(cache_path):
                    self.real_images, self.real_landmarks, self.real_poses = build_or_load_drowsiness_cache(
                        data_dir or "", cache_path, teacher=teacher
                    )
                elif data_dir and os.path.exists(data_dir):
                    default_cache = os.path.join(data_dir, "drowsiness_cache.npz")
                    self.real_images, self.real_landmarks, self.real_poses = build_or_load_drowsiness_cache(
                        data_dir, default_cache, teacher=teacher
                    )

        if self.real_images is not None and len(self.real_images) > 0:
            # Phân tách 3 nhóm độc lập từ tập TRAIN:
            pool_for_pools = self.train_indices if self.train_indices else list(range(len(self.real_images)))
            for idx in pool_for_pools:
                lm = self.real_landmarks[idx]
                ear_val = compute_sample_ear(lm)
                mar_val = compute_sample_mar(lm)
                if ear_val < 0.20:
                    self.closed_indices.append(idx)
                elif mar_val >= 0.40:
                    self.yawn_indices.append(idx)
                else:
                    self.normal_indices.append(idx)

            print(f"[DatasetLoader] [REAL-DATA] Kích hoạt luồng huấn luyện Thực Tế (3-Way Balanced Sampling): {len(self.train_indices)} mẫu train!")
            print(f"  • Mẫu nhắm mắt/microsleep thật (EAR < 0.20): {len(self.closed_indices)} mẫu")
            print(f"  • Mẫu ngáp/há miệng thật      (MAR >= 0.40): {len(self.yawn_indices)} mẫu")
            print(f"  • Mẫu tỉnh táo/bình thường    (Attentive)   : {len(self.normal_indices)} mẫu")
            self.total_samples = max(len(self.real_images) * 4, self.synthetic_count, 4000)
        else:
            print(f"[DatasetLoader] [SYNTHETIC] Kích hoạt luồng huấn luyện Sinh Trắc Học Tăng Cường ({self.synthetic_count} mẫu)...")
            self.use_synthetic = True
            self.total_samples = self.synthetic_count

    @staticmethod
    def _normalize_image(img):
        """Chuẩn hóa [-1.0, 1.0] đồng bộ 100% với ESP32-S3: (pixel - 128) / 128."""
        img_norm = (img.astype(np.float32) - 128.0) / 128.0
        if len(img_norm.shape) == 2:
            img_norm = np.expand_dims(img_norm, axis=-1)
        elif len(img_norm.shape) == 3 and img_norm.shape[-1] != 1:
            img_norm = img_norm[:, :, :1]
        return img_norm

    def get_val_arrays(self):
        """Trả về (images, landmarks, poses) của tập VAL GIỮ-OUT THẬT để đo NME."""
        if self.val_indices:
            idxs = np.array(self.val_indices, dtype=np.int64)
            return (self.real_images[idxs], self.real_landmarks[idxs], self.real_poses[idxs])
        if self.real_images is not None and len(self.real_images) > 0:
            n_val = max(int(len(self.real_images) * 0.08), 8)
            return (self.real_images[-n_val:], self.real_landmarks[-n_val:], self.real_poses[-n_val:])
        return None

    def _generate_real_val_generator(self, num_samples):
        """VAL GIỮ-OUT THẬT: duyệt tuần tự các mẫu split=1, KHÔNG augment, KHÔNG mixup."""
        np.random.seed(42)
        if self.val_indices:
            order = np.random.permutation(np.array(self.val_indices, dtype=np.int64))
        elif self.real_images is not None and len(self.real_images) > 0:
            print("[DatasetLoader] [VAL-WARN] npz thiếu cột 'split' -> dùng 8% cuối làm val.")
            n_val = max(int(len(self.real_images) * 0.08), 8)
            order = np.arange(len(self.real_images) - n_val, len(self.real_images))
        else:
            print("[DatasetLoader] [VAL-WARN] Không có dữ liệu thật -> val fallback synthetic.")
            for i in range(num_samples):
                force_st = 'normal' if (i % 2 == 0) else 'yawn'
                img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=False, force_state=force_st)
                yield self._normalize_image(img), {"landmarks_output": lms.astype(np.float32),
                                                   "pose_output": pose.astype(np.float32)}
            return

        for i in range(num_samples):
            idx = int(order[i % len(order)])
            img = self.real_images[idx].copy()
            lms = self.real_landmarks[idx].copy()
            pose = self.real_poses[idx].copy()
            yield self._normalize_image(img), {"landmarks_output": lms.astype(np.float32),
                                               "pose_output": pose.astype(np.float32)}

    def generate_data_generator(self, num_samples, split='train'):
        """Generator yielding (image_norm, {"landmarks_output": lms, "pose_output": pose}).
        split='train': Cân bằng 3 trạng thái (33% Nhắm mắt : 33% Ngáp : 34% Tỉnh táo) + cabin augment.
        split='val': giữ-out thật."""
        if split == 'val':
            yield from self._generate_real_val_generator(num_samples)
            return
        apply_aug = (split == 'train') and self.augment

        has_real_closed = len(self.closed_indices) > 0
        has_real_yawns = len(self.yawn_indices) > 0
        has_real_normals = len(self.normal_indices) > 0

        for i in range(num_samples):
            slot = i % 3  # Cân bằng 3 trạng thái đồng đều

            if slot == 0:
                # Slot nhắm mắt (EAR < 0.20): 100% mẫu mắt nhắm
                if not self.use_synthetic and has_real_closed and (np.random.rand() > 0.15):
                    idx = np.random.choice(self.closed_indices)
                    img = self.real_images[idx].copy()
                    lms = self.real_landmarks[idx].copy()
                    pose = self.real_poses[idx].copy()
                    if apply_aug:
                        lms_22 = lms.reshape((22, 2))
                        img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                        lms = lms_22.flatten()
                        pose = estimate_pose_from_landmarks(lms_22)
                else:
                    img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=apply_aug, force_state='microsleep')
            elif slot == 1:
                # Slot ngáp (MAR >= 0.40): 100% mẫu ngáp há miệng
                if not self.use_synthetic and has_real_yawns and (np.random.rand() > 0.10):
                    idx = np.random.choice(self.yawn_indices)
                    img = self.real_images[idx].copy()
                    lms = self.real_landmarks[idx].copy()
                    pose = self.real_poses[idx].copy()
                    if apply_aug:
                        lms_22 = lms.reshape((22, 2))
                        img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                        lms = lms_22.flatten()
                        pose = estimate_pose_from_landmarks(lms_22)
                else:
                    img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=apply_aug, force_state='yawn')
            else:
                # Slot tỉnh táo / mắt mở / ngậm miệng bình thường
                if not self.use_synthetic and has_real_normals:
                    idx = np.random.choice(self.normal_indices)
                    img = self.real_images[idx].copy()
                    lms = self.real_landmarks[idx].copy()
                    pose = self.real_poses[idx].copy()
                    if apply_aug:
                        lms_22 = lms.reshape((22, 2))
                        img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                        lms = lms_22.flatten()
                        pose = estimate_pose_from_landmarks(lms_22)
                else:
                    img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=apply_aug, force_state='normal')

            # Chuẩn hóa về [-1.0, 1.0] đồng bộ 100% với ESP32-S3 (pixel - 128) / 128
            img_norm = self._normalize_image(img)

            yield img_norm, {"landmarks_output": lms.astype(np.float32), "pose_output": pose.astype(np.float32)}

    def _augment_and_normalize(self, img_t, lms_t, pose_t):
        """Chạy trong worker song song: augment NumPy/CV2 + chuẩn hóa [-1,1]."""
        img = img_t.numpy()
        lms = lms_t.numpy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        lms_22 = lms.reshape(22, 2)
        img_aug, lms_aug = apply_cabin_data_augmentation(img, lms_22)
        pose_aug = estimate_pose_from_landmarks(lms_aug)
        img_norm = (img_aug.astype(np.float32) - 128.0) / 128.0
        if len(img_norm.shape) == 2:
            img_norm = np.expand_dims(img_norm, axis=-1)
        elif len(img_norm.shape) == 3 and img_norm.shape[-1] != 1:
            img_norm = img_norm[:, :, :1]
        return img_norm, lms_aug.flatten().astype(np.float32), pose_aug.astype(np.float32)

    def _augment_map_fn(self, img_t, lms_t, pose_t):
        out = tf.py_function(
            self._augment_and_normalize, [img_t, lms_t, pose_t],
            [tf.float32, tf.float32, tf.float32]
        )
        img_a, lms_a, pose_a = out
        return (
            tf.reshape(img_a, [IMAGE_HEIGHT, IMAGE_WIDTH, 1]),
            {
                "landmarks_output": tf.reshape(lms_a, [NUM_LANDMARKS * 2]),
                "pose_output": tf.reshape(pose_a, [3]),
            },
        )

    def _build_raw_pools(self):
        """Pool ảnh THÔ tách làm 3 nhóm: closed (nhắm mắt) / yawn (ngáp) / normal (tỉnh táo)."""
        if self.real_images is not None and len(self.real_images) > 0:
            c_idx = np.array(self.closed_indices, dtype=np.int64) if self.closed_indices else np.array(self.normal_indices[:50], dtype=np.int64)
            y_idx = np.array(self.yawn_indices, dtype=np.int64) if self.yawn_indices else np.array(self.normal_indices[:50], dtype=np.int64)
            n_idx = np.array(self.normal_indices, dtype=np.int64)

            # [100% REAL HUMAN DATASET]
            # Nếu tập dữ liệu đã có dồi dào mẫu người thật (>= 200 mẫu ngáp và nhắm mắt)
            # thì sử dụng 100% dữ liệu người thật, không cần chèn mẫu vẽ giả lập!
            if len(c_idx) >= 200 and len(y_idx) >= 200:
                return (
                    (self.real_images[c_idx], self.real_landmarks[c_idx], self.real_poses[c_idx]),
                    (self.real_images[y_idx], self.real_landmarks[y_idx], self.real_poses[y_idx]),
                    (self.real_images[n_idx], self.real_landmarks[n_idx], self.real_poses[n_idx]),
                )

            # Fallback nếu dữ liệu thiếu hụt cực đoan:
            n_synth_inject = max(len(c_idx) * 3, 350)
            c_synth_i, c_synth_l, c_synth_p = [], [], []
            y_synth_i, y_synth_l, y_synth_p = [], [], []
            for i in range(n_synth_inject):
                img_c, lm_c, pose_c = generate_synthetic_driver_sample(i, apply_aug=False, force_state='microsleep')
                c_synth_i.append(img_c); c_synth_l.append(lm_c); c_synth_p.append(pose_c)
                img_y, lm_y, pose_y = generate_synthetic_driver_sample(i, apply_aug=False, force_state='yawn')
                y_synth_i.append(img_y); y_synth_l.append(lm_y); y_synth_p.append(pose_y)

            all_c_i = np.concatenate([self.real_images[c_idx], np.array(c_synth_i, np.uint8)], axis=0)
            all_c_l = np.concatenate([self.real_landmarks[c_idx], np.array(c_synth_l, np.float32)], axis=0)
            all_c_p = np.concatenate([self.real_poses[c_idx], np.array(c_synth_p, np.float32)], axis=0)

            all_y_i = np.concatenate([self.real_images[y_idx], np.array(y_synth_i, np.uint8)], axis=0)
            all_y_l = np.concatenate([self.real_landmarks[y_idx], np.array(y_synth_l, np.float32)], axis=0)
            all_y_p = np.concatenate([self.real_poses[y_idx], np.array(y_synth_p, np.float32)], axis=0)

            return (
                (all_c_i, all_c_l, all_c_p),
                (all_y_i, all_y_l, all_y_p),
                (self.real_images[n_idx], self.real_landmarks[n_idx], self.real_poses[n_idx]),
            )
        # Synthetic fallback
        n_third = max(self.synthetic_count // 3, 64)
        c_i, c_l, c_p = [], [], []
        y_i, y_l, y_p = [], [], []
        n_i, n_l, n_p = [], [], []
        for i in range(n_third):
            img, lm, pose = generate_synthetic_driver_sample(i, apply_aug=False, force_state='microsleep')
            c_i.append(img); c_l.append(lm); c_p.append(pose)
        for i in range(n_third):
            img, lm, pose = generate_synthetic_driver_sample(i, apply_aug=False, force_state='yawn')
            y_i.append(img); y_l.append(lm); y_p.append(pose)
        for i in range(n_third):
            img, lm, pose = generate_synthetic_driver_sample(i, apply_aug=False, force_state='normal')
            n_i.append(img); n_l.append(lm); n_p.append(pose)
        return (
            (np.array(c_i, np.uint8), np.array(c_l, np.float32), np.array(c_p, np.float32)),
            (np.array(y_i, np.uint8), np.array(y_l, np.float32), np.array(y_p, np.float32)),
            (np.array(n_i, np.uint8), np.array(n_l, np.float32), np.array(n_p, np.float32))
        )

    def _light_graph_jitter_fn(self, img_t, lms_t, pose_t):
        """Photometric jitter bằng TF graph ops (C++, không GIL) — giữ tính động photometric."""
        img = tf.cast(img_t, tf.float32)
        img = tf.image.random_brightness(img, 0.12)
        img = tf.image.random_contrast(img, 0.90, 1.10)
        noise = tf.random.normal(tf.shape(img), stddev=2.5)
        img = tf.clip_by_value(img + noise, 0.0, 255.0)
        img = (img - 128.0) / 128.0
        return img, {"landmarks_output": lms_t, "pose_output": pose_t}

    def _preexpand_pools(self, expand_factor=6):
        """Pre-expand pool closed/yawn/normal cân bằng 30/30/40 với augment cabin đa dạng, song song thread."""
        from concurrent.futures import ThreadPoolExecutor
        (nc_img, nc_lms, nc_pose), (ny_img, ny_lms, ny_pose), (nn_img, nn_lms, nn_pose) = self._build_raw_pools()
        total = max(len(self.real_images) * expand_factor, 4000) if self.real_images is not None \
            else max(self.synthetic_count, 4000)
        n_closed = int(total * 0.30)
        n_yawn = int(total * 0.30)
        n_norm = total - n_closed - n_yawn

        def expand_pool(pool, count, tag):
            out_i, out_l, out_p = [], [], []
            if len(pool[0]) == 0:
                return out_i, out_l, out_p
            def _one(k):
                idx = k % len(pool[0])
                img = pool[0][idx].copy()
                lms22 = pool[1][idx].reshape(22, 2).copy()
                img_a, lms_a = apply_cabin_data_augmentation(img, lms22)
                pose_a = estimate_pose_from_landmarks(lms_a)
                return img_a, lms_a.flatten().astype(np.float32), pose_a.astype(np.float32)
            with ThreadPoolExecutor(max_workers=4) as ex:
                for r in ex.map(_one, range(count)):
                    out_i.append(r[0]); out_l.append(r[1]); out_p.append(r[2])
            return out_i, out_l, out_p

        print(f"    [STATIC-EXPAND] Đang tiền-tính {total} bản augment cân bằng 3 trạng thái "
              f"({n_closed} nhắm mắt + {n_yawn} ngáp + {n_norm} tỉnh táo, expand_factor={expand_factor})...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            fc = ex.submit(expand_pool, (nc_img, nc_lms, nc_pose), n_closed, "closed")
            fy = ex.submit(expand_pool, (ny_img, ny_lms, ny_pose), n_yawn, "yawn")
            fn = ex.submit(expand_pool, (nn_img, nn_lms, nn_pose), n_norm, "normal")
            ci, cl, cp = fc.result()
            yi, yl, yp = fy.result()
            ni, nl, np_ = fn.result()

        # [ANTI-MEAN-COLLAPSE] Đan xen đều 3 trạng thái theo bộ ba [nhắm_k, ngáp_k, bình_thường_k]
        # Tránh lỗi dồn 22.000 mẫu cùng loại thành một khối làm tê liệt gradient hoặc trôi về trạng thái trung bình
        n_min = min(len(ci), len(yi), len(ni))
        interleaved_i = []
        interleaved_l = []
        interleaved_p = []
        for k in range(n_min):
            interleaved_i.append(ci[k])
            interleaved_i.append(yi[k])
            interleaved_i.append(ni[k])
            interleaved_l.append(cl[k])
            interleaved_l.append(yl[k])
            interleaved_l.append(nl[k])
            interleaved_p.append(cp[k])
            interleaved_p.append(yp[k])
            interleaved_p.append(np_[k])

        # Nối phần dư nếu các nhóm có độ dài chênh lệch nhẹ
        for k in range(n_min, len(ci)):
            interleaved_i.append(ci[k]); interleaved_l.append(cl[k]); interleaved_p.append(cp[k])
        for k in range(n_min, len(yi)):
            interleaved_i.append(yi[k]); interleaved_l.append(yl[k]); interleaved_p.append(yp[k])
        for k in range(n_min, len(ni)):
            interleaved_i.append(ni[k]); interleaved_l.append(nl[k]); interleaved_p.append(np_[k])

        all_i = np.array(interleaved_i, dtype=np.uint8)
        all_l = np.array(interleaved_l, dtype=np.float32)
        all_p = np.array(interleaved_p, dtype=np.float32)

        # Xáo trộn ngẫu nhiên toàn cục trước khi đưa vào TF Dataset
        perm = np.random.RandomState(42).permutation(len(all_i))
        all_i = all_i[perm]
        all_l = all_l[perm]
        all_p = all_p[perm]

        print(f"    [STATIC-EXPAND] Xong trong {time.time() - t0:.1f}s "
              f"({len(all_i)} mẫu: {len(ci)} nhắm mắt / {len(yi)} ngáp / {len(ni)} tỉnh táo, RAM ~{all_i.nbytes/1e6:.0f}MB, ĐÃ TRỘN TOÀN CỤC)")
        return all_i, all_l, all_p

    def get_static_expanded_dataset(self, batch_size=64, val_split=0.15, expand_factor=6):
        """Pipeline NHANH NHẤT: augmentation pre-computed + jitter nhẹ bằng TF graph.
        Trả về cùng cấu trúc get_tf_dataset. Không đổi kiến trúc/loss/logic model."""
        if tf is None:
            raise RuntimeError("TensorFlow không khả dụng.")
        all_i, all_l, all_p = self._preexpand_pools(expand_factor=expand_factor)

        ds = tf.data.Dataset.from_tensor_slices((all_i, all_l, all_p))
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
        ds = ds.map(self._light_graph_jitter_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Val giữ-out thật: giữ nguyên generator (không augment)
        val_samples = max(int(all_i.shape[0] * val_split), 64)
        val_ds = tf.data.Dataset.from_generator(
            lambda: self.generate_data_generator(val_samples, split='val'),
            output_signature=(
                tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
                {
                    "landmarks_output": tf.TensorSpec(shape=(NUM_LANDMARKS * 2,), dtype=tf.float32),
                    "pose_output": tf.TensorSpec(shape=(3,), dtype=tf.float32)
                }
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_samples = all_i.shape[0] - val_samples
        return ds, val_ds, train_samples, val_samples

    def get_fast_tf_dataset(self, batch_size=64, val_split=0.15):
        """Pipeline huấn luyện SONG SONG: chọn mẫu bằng tf ops + augmentation qua
        py_function num_parallel_calls=AUTOTUNE. Trả về cùng cấu trúc get_tf_dataset."""
        if tf is None:
            raise RuntimeError("TensorFlow không khả dụng. Vui lòng cài tensorflow.")

        (nc_img, nc_lms, nc_pose), (ny_img, ny_lms, ny_pose), (nn_img, nn_lms, nn_pose) = self._build_raw_pools()
        n_closed, n_yawn, n_norm = len(nc_img), len(ny_img), len(nn_img)
        cat_img = tf.constant(np.concatenate([nc_img, ny_img, nn_img], axis=0), dtype=tf.uint8)
        cat_lms = tf.constant(np.concatenate([nc_lms, ny_lms, nn_lms], axis=0), dtype=tf.float32)
        cat_pose = tf.constant(np.concatenate([nc_pose, ny_pose, nn_pose], axis=0), dtype=tf.float32)

        offset_closed = tf.constant(0, tf.int64)
        offset_yawn = tf.constant(n_closed, tf.int64)
        offset_norm = tf.constant(n_closed + n_yawn, tf.int64)

        def _pick(i):
            slot = tf.random.uniform([], 0, 3, tf.int32)
            rc = tf.random.uniform([], 0, max(n_closed, 1), tf.int64) + offset_closed
            ry = tf.random.uniform([], 0, max(n_yawn, 1), tf.int64) + offset_yawn
            rn = tf.random.uniform([], 0, max(n_norm, 1), tf.int64) + offset_norm
            return tf.switch_case(slot, {
                0: lambda: rc,
                1: lambda: ry,
                2: lambda: rn
            }, default=lambda: rn)

        ds = tf.data.Dataset.range(max(self.total_samples, 2048))
        ds = ds.map(_pick, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(4096)
        ds = ds.map(
            lambda idx: (tf.gather(cat_img, idx), tf.gather(cat_lms, idx), tf.gather(cat_pose, idx)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Augmentation SONG SONG (thuật toán nguyên bản trong py_function)
        ds = ds.map(self._augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Val giữ-out thật: giữ nguyên generator (không augment, đã đủ nhanh)
        val_samples = max(int(self.total_samples * val_split), 64)
        val_ds = tf.data.Dataset.from_generator(
            lambda: self.generate_data_generator(val_samples, split='val'),
            output_signature=(
                tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
                {
                    "landmarks_output": tf.TensorSpec(shape=(NUM_LANDMARKS * 2,), dtype=tf.float32),
                    "pose_output": tf.TensorSpec(shape=(3,), dtype=tf.float32)
                }
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_samples = self.total_samples - val_samples
        return ds, val_ds, train_samples, val_samples

    def get_tf_dataset(self, batch_size=64, val_split=0.15):
        """Builds optimized tf.data.Dataset for train and validation."""
        if tf is None:
            raise RuntimeError("TensorFlow không khả dụng. Vui lòng cài tensorflow.")

        total_samples = self.total_samples
        val_samples = max(int(total_samples * val_split), 64)
        train_samples = total_samples - val_samples

        output_signature = (
            tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
            {
                "landmarks_output": tf.TensorSpec(shape=(NUM_LANDMARKS * 2,), dtype=tf.float32),
                "pose_output": tf.TensorSpec(shape=(3,), dtype=tf.float32)
            }
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: self.generate_data_generator(train_samples, split='train'),
            output_signature=output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: self.generate_data_generator(val_samples, split='val'),
            output_signature=output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, train_samples, val_samples

