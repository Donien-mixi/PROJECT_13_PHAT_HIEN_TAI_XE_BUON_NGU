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

    # 1. Spatial Sub-Pixel Micro-Jitter (+/- 1.5px trong ô 96x96)
    # Hấp thụ nhiễu sub-pixel của detector nhưng giữ nguyên vùng mắt/mũi/miệng căn giữa
    if np.random.rand() > 0.60:
        dx = float(np.random.uniform(-1.5, 1.5))
        dy = float(np.random.uniform(-1.5, 1.5))
        M_shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        img = cv2.warpAffine(img, M_shift, (96, 96), borderMode=cv2.BORDER_REFLECT)
        lm[:, 0] = np.clip(lm[:, 0] + (dx / 96.0), 0.0, 1.0)
        lm[:, 1] = np.clip(lm[:, 1] + (dy / 96.0), 0.0, 1.0)

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
    if np.random.rand() > 0.45:
        left_eye_pts = lm[0:6] * 96.0
        right_eye_pts = lm[6:12] * 96.0
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
                glare_mask = np.zeros((96, 96), dtype=np.float32)
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

    # 8. Micro-Tilt (+/- 2.5 độ, giữ nguyên quy chuẩn xoay thẳng mặt MediaPipe)
    if np.random.rand() > 0.70:
        angle = float(np.random.uniform(-2.5, 2.5))
        M = cv2.getRotationMatrix2D((48.0, 48.0), angle, 1.0)
        img = cv2.warpAffine(img, M, (96, 96), borderMode=cv2.BORDER_REFLECT)
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
    mouth_open_dy = float(np.random.uniform(18.0, 35.0)) if is_yawning else float(np.random.uniform(1.5, 4.5))

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
    Downloads & extracts full-face benchmark datasets:
    1. Driver Drowsiness Dataset (In-Cabin Full Face)
    2. Human Yawning & Fatigue Dataset (Full Face Yawning)
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"[DatasetLoader] Kiểm tra thư mục dữ liệu buồn ngủ: {target_dir}")

    # 1. Check local yawn_faces.zip
    candidate_yawn_zips = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "yawn_faces.zip"),
        "yawn_faces.zip",
        os.path.join("training_tinyml", "yawn_faces.zip"),
        os.path.join(target_dir, "yawn_faces.zip")
    ]
    for bz in candidate_yawn_zips:
        if os.path.exists(bz) and zipfile.is_zipfile(bz):
            print(f"  [BUNDLE] Đã tìm thấy gói ảnh ngáp người thật: {bz}. Đang giải nén...")
            try:
                with zipfile.ZipFile(bz, 'r') as zf:
                    zf.extractall(target_dir)
                print("  [OK] Đã giải nén thành công dữ liệu ngáp ngủ thực tế!")
            except Exception as e:
                print(f"  [WARNING] Lỗi giải nén {bz}: {e}")
            break

    # 2. Download In-Cabin Driver Drowsiness
    cabin_url = "https://raw.githubusercontent.com/hazeeq911/Driver-Drowsiness-Detection/main/drowsiness.zip"
    cabin_zip_path = os.path.join(target_dir, "drowsiness_cabin.zip")

    existing_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        existing_images.extend(glob.glob(os.path.join(target_dir, '**', ext), recursive=True))

    if len(existing_images) < 100:
        print("  [DOWNLOAD] Đang tải bộ dữ liệu tài xế trong cabin ô tô (Full Face Drowsiness, ~6.8MB)...")
        try:
            req = urllib.request.Request(cabin_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=45) as resp, open(cabin_zip_path, 'wb') as out_f:
                import shutil
                shutil.copyfileobj(resp, out_f)

            if os.path.exists(cabin_zip_path) and zipfile.is_zipfile(cabin_zip_path):
                with zipfile.ZipFile(cabin_zip_path, 'r') as zf:
                    zf.extractall(target_dir)
                os.remove(cabin_zip_path)
                print("  [OK] Đã giải nén thành công bộ dữ liệu tài xế trong cabin!")
        except Exception as e:
            print(f"  [WARNING] Không tải được từ {cabin_url} ({e})")
            if os.path.exists(cabin_zip_path):
                try: os.remove(cabin_zip_path)
                except Exception: pass

    found_imgs = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        found_imgs.extend(glob.glob(os.path.join(target_dir, '**', ext), recursive=True))
    print(f"[DatasetLoader] [DATA-READY] Tổng cộng tìm thấy {len(found_imgs)} ảnh FULL MẶT thực tế trong {target_dir}!")
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
    Multi-Task Data Pipeline for TinyDriver PFLD-Edge:
      - Delivers (x_image, {"landmarks_output": 44, "pose_output": 3}).
      - Triệt tiêu Mean-Face Collapse bằng cơ chế Balanced 50/50 Class-Conditional Sampling:
        + 50% mẫu Ngáp / Há miệng to (MAR >= 0.45).
        + 50% mẫu Tỉnh táo / Ngủ gật nhắm mắt (MAR < 0.35).
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

        self.yawn_indices = []
        self.normal_indices = []

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
                        print(f"[DatasetLoader] [SUCCESS] Đã nạp thành công {len(self.real_images)} mẫu ảnh người thật từ: {p_path}")
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
            for idx, lm in enumerate(self.real_landmarks):
                mar_val = compute_sample_mar(lm)
                if mar_val >= 0.40:
                    self.yawn_indices.append(idx)
                else:
                    self.normal_indices.append(idx)

            print(f"[DatasetLoader] [REAL-DATA] Kích hoạt luồng huấn luyện Thực Tế: {len(self.real_images)} mẫu người thật!")
            print(f"  • Mẫu ngáp/há miệng thật (MAR >= 0.40): {len(self.yawn_indices)} mẫu")
            print(f"  • Mẫu ngậm miệng/nhắm mắt thật       : {len(self.normal_indices)} mẫu")
            self.total_samples = max(len(self.real_images) * 8, self.synthetic_count, 6000)
        else:
            print(f"[DatasetLoader] [SYNTHETIC] Kích hoạt luồng huấn luyện Sinh Trắc Học Tăng Cường ({self.synthetic_count} mẫu)...")
            self.use_synthetic = True
            self.total_samples = self.synthetic_count

    def generate_data_generator(self, num_samples, split='train'):
        """Generator yielding (image_norm, {"landmarks_output": lms, "pose_output": pose}) with 50/50 balance."""
        np.random.seed(42 if split == 'val' else None)
        apply_aug = (split == 'train') and self.augment

        has_real_yawns = len(self.yawn_indices) > 0
        has_real_normals = len(self.normal_indices) > 0

        for i in range(num_samples):
            is_yawn_slot = (i % 2 == 1) # 50% thời gian là slot ngáp / há miệng

            if is_yawn_slot:
                # Slot ngáp (MAR >= 0.40): 100% dữ liệu người thật khi khả dụng
                if not self.use_synthetic and has_real_yawns:
                    idx = np.random.choice(self.yawn_indices)
                    img = self.real_images[idx].copy()
                    lms = self.real_landmarks[idx].copy()
                    pose = self.real_poses[idx].copy()
                    # Biometric Mixup giữa 2 ảnh ngáp thật để tăng sự đa dạng biểu cảm
                    if apply_aug and np.random.rand() > 0.50 and len(self.yawn_indices) > 1:
                        idx2 = np.random.choice(self.yawn_indices)
                        lam = float(np.random.uniform(0.30, 0.70))
                        img = (lam * img.astype(np.float32) + (1.0 - lam) * self.real_images[idx2].astype(np.float32)).astype(np.uint8)
                        lms = lam * lms + (1.0 - lam) * self.real_landmarks[idx2]
                        pose = lam * pose + (1.0 - lam) * self.real_poses[idx2]
                    if apply_aug:
                        lms_22 = lms.reshape((22, 2))
                        img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                        lms = lms_22.flatten()
                        pose = estimate_pose_from_landmarks(lms_22)
                else:
                    img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=apply_aug, force_state='yawn')
            else:
                # Slot ngậm miệng / nhắm mắt ngủ gật: 100% dữ liệu người thật khi khả dụng
                if not self.use_synthetic and has_real_normals:
                    idx = np.random.choice(self.normal_indices)
                    img = self.real_images[idx].copy()
                    lms = self.real_landmarks[idx].copy()
                    pose = self.real_poses[idx].copy()
                    # Biometric Mixup giữa 2 ảnh mắt bình thường / nhắm mắt thật
                    if apply_aug and np.random.rand() > 0.50 and len(self.normal_indices) > 1:
                        idx2 = np.random.choice(self.normal_indices)
                        lam = float(np.random.uniform(0.30, 0.70))
                        img = (lam * img.astype(np.float32) + (1.0 - lam) * self.real_images[idx2].astype(np.float32)).astype(np.uint8)
                        lms = lam * lms + (1.0 - lam) * self.real_landmarks[idx2]
                        pose = lam * pose + (1.0 - lam) * self.real_poses[idx2]
                    if apply_aug:
                        lms_22 = lms.reshape((22, 2))
                        img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                        lms = lms_22.flatten()
                        pose = estimate_pose_from_landmarks(lms_22)
                else:
                    force_st = 'microsleep' if (i % 4 == 0) else 'normal'
                    img, lms, pose = generate_synthetic_driver_sample(i, apply_aug=apply_aug, force_state=force_st)

            # Chuẩn hóa về [-1.0, 1.0] đồng bộ 100% với ESP32-S3 (pixel - 128) / 128
            img_norm = (img.astype(np.float32) - 128.0) / 128.0
            if len(img_norm.shape) == 2:
                img_norm = np.expand_dims(img_norm, axis=-1)
            elif len(img_norm.shape) == 3 and img_norm.shape[-1] != 1:
                img_norm = img_norm[:, :, :1]

            yield img_norm, {"landmarks_output": lms.astype(np.float32), "pose_output": pose.astype(np.float32)}

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

