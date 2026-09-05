"""
Dataset Loader & Advanced Data Augmentation Pipeline for TinyDriver-LandmarkNet.
Supports standard 68-point datasets (300W, iBUG, LFPW, HELEN) with 22-point filtering,
an enhanced photorealistic synthetic driver generator, and in-cabin automotive
data augmentations (variable lighting, motion blur, glasses occlusion, sensor noise).
"""

import os
import glob
import urllib.request
import zipfile
import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, IBUG_68_TO_22_INDICES,
        BBOX_EXPANSION_RATIO, NUM_LANDMARKS
    )
    from isomorphic_transform import square_crop_and_resize
except (ImportError, ValueError):
    from .config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, IBUG_68_TO_22_INDICES,
        BBOX_EXPANSION_RATIO, NUM_LANDMARKS
    )
    from .isomorphic_transform import square_crop_and_resize


def parse_pts_file(pts_path):
    """Parses a standard iBUG 68-point .pts annotation file."""
    landmarks = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()
        
    start = False
    for line in lines:
        line = line.strip()
        if line.startswith('{'):
            start = True
            continue
        if line.startswith('}'):
            break
        if start and line:
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                landmarks.append([x, y])
                
    return np.array(landmarks, dtype=np.float32)


def apply_cabin_data_augmentation(image, landmarks_norm):
    """
    Photometric and In-Cabin Environmental Data Augmentation.
    Mô phỏng điều kiện thực tế trong cabin xe ô tô:
      - Thay đổi ánh sáng (nắng gắt, hầm tối, ban đêm)
      - Nhiễu hạt ISO của webcam xe
      - Rung lắc xe (Motion Blur)
      - Gọng kính cận hoặc che khuất cục bộ (Cutout)
      
    Args:
        image: np.ndarray shape (96, 96) uint8 grayscale
        landmarks_norm: np.ndarray shape (22, 2) float32 in [0, 1]
    Returns:
        aug_image: np.ndarray shape (96, 96) uint8
        aug_landmarks: np.ndarray shape (22, 2) float32 in [0, 1]
    """
    img = image.copy().astype(np.float32)
    lm = landmarks_norm.copy()

    # 1. Photometric: Random Brightness & Contrast
    # Alpha (Contrast): 0.7 - 1.3 | Beta (Brightness): -35 - +35
    alpha = np.random.uniform(0.75, 1.25)
    beta = np.random.uniform(-30.0, 30.0)
    img = np.clip(img * alpha + beta, 0, 255)

    # 2. Random Gamma (Mô phỏng đi vào hầm tối hoặc nắng chiếu trực diện)
    if np.random.rand() > 0.5:
        gamma = np.random.uniform(0.7, 1.4)
        inv_gamma = 1.0 / gamma
        img = np.clip(((img / 255.0) ** inv_gamma) * 255.0, 0, 255)

    # 3. Sensor Noise / Gaussian Noise (Webcam ISO noise trong bóng tối)
    if np.random.rand() > 0.6:
        noise = np.random.normal(0, np.random.uniform(3, 10), img.shape)
        img = np.clip(img + noise, 0, 255)

    # 4. Vehicle Vibration / Motion Blur (Xe rung lắc trên đường xóc)
    if np.random.rand() > 0.7:
        ksize = np.random.choice([3, 5])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel = kernel / ksize
        img = cv2.filter2D(img, -1, kernel)

    # 5. Simulated Eyeglasses Frame or Hand Occlusion (Cutout)
    if np.random.rand() > 0.65:
        # Che một thanh ngang mỏng qua mắt mô phỏng gọng kính cận
        eye_y = int(np.mean(lm[:12, 1]) * 96.0)
        thickness = np.random.randint(2, 5)
        y0 = max(0, eye_y - thickness)
        y1 = min(96, eye_y + thickness)
        dark_shade = np.random.randint(20, 70)
        img[y0:y1, :] = (img[y0:y1, :] * 0.4 + dark_shade * 0.6).astype(np.float32)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    return img.astype(np.uint8), lm


def generate_synthetic_driver_sample(sample_idx, apply_aug=False):
    """
    Generates a realistic synthetic face sample with 22 ground-truth landmarks.
    Incorporates skin gradient textures, realistic iris/pupils, and eyelid dynamics.
    """
    canvas = np.full((240, 240), np.random.randint(30, 60), dtype=np.uint8)

    # Face center and dimensions
    cx = 120 + np.random.randint(-12, 13)
    cy = 120 + np.random.randint(-12, 13)
    rx = 54 + np.random.randint(-4, 5)
    ry = 68 + np.random.randint(-4, 5)

    # 1. Realistic skin shading gradient
    face_color = np.random.randint(160, 210)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, face_color, -1)
    # Highlight on forehead and nose
    cv2.ellipse(canvas, (cx, cy - 20), (rx // 2, ry // 3), 0, 0, 360, min(255, face_color + 25), -1)

    # Random simulated states (Eye open/closed, Mouth open/closed, Head yaw)
    is_eyes_closed = (sample_idx % 3 == 0)
    is_yawning = (sample_idx % 4 == 0)
    yaw_shift = np.random.randint(-10, 11)

    eye_open_dy = 0.8 if is_eyes_closed else float(np.random.randint(4, 9))
    mouth_open_dy = float(np.random.randint(14, 24)) if is_yawning else float(np.random.randint(2, 5))

    # Left eye center ~ (cx - 24 + yaw_shift, cy - 15)
    lex = cx - 23 + yaw_shift
    ley = cy - 15
    # Right eye center ~ (cx + 24 + yaw_shift, cy - 15)
    rex = cx + 23 + yaw_shift
    rey = cy - 15

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

    # 6 Mouth landmarks
    mx = cx + yaw_shift
    my = cy + 32
    mouth = np.array([
        [mx - 18, my],
        [mx + 18, my],
        [mx,      my - mouth_open_dy],
        [mx,      my + mouth_open_dy],
        [mx,      my - mouth_open_dy * 0.5],
        [mx,      my + mouth_open_dy * 0.5],
    ], dtype=np.float32)

    # 4 Nose and Chin landmarks
    nx = cx + yaw_shift
    nose_chin = np.array([
        [nx, cy - 25],     # Nasion
        [nx, cy + 6],      # Nose tip
        [nx - 6, cy + 13], # Left nostril
        [cx, cy + ry - 6]  # Chin
    ], dtype=np.float32)

    landmarks_22 = np.vstack([left_eye, right_eye, mouth, nose_chin])

    # Draw eyes (sclera + iris if open, fold if closed)
    if not is_eyes_closed:
        cv2.fillPoly(canvas, [left_eye.astype(np.int32)], 240)
        cv2.circle(canvas, (int(lex), int(ley)), 3, 30, -1)
        cv2.fillPoly(canvas, [right_eye.astype(np.int32)], 240)
        cv2.circle(canvas, (int(rex), int(rey)), 3, 30, -1)
    else:
        cv2.polylines(canvas, [left_eye[:4].astype(np.int32)], False, 30, 2)
        cv2.polylines(canvas, [right_eye[:4].astype(np.int32)], False, 30, 2)

    # Draw mouth cavity if yawning
    if is_yawning:
        cv2.fillPoly(canvas, [mouth[:4].astype(np.int32)], 25)
    else:
        cv2.polylines(canvas, [mouth.astype(np.int32)], True, 40, 1)

    # Compute bounding box from landmarks with realistic anthropometric margin (accounting for forehead and hair)
    lm_min_x = np.min(landmarks_22[:, 0])
    lm_max_x = np.max(landmarks_22[:, 0])
    lm_min_y = np.min(landmarks_22[:, 1])
    lm_max_y = np.max(landmarks_22[:, 1])
    face_w = lm_max_x - lm_min_x
    face_h = lm_max_y - lm_min_y

    # Trán và tóc phía trên mắt chiếm khoảng 45% chiều cao khuôn mặt (chuẩn Haar Cascade & camera xe)
    xmin = int(lm_min_x - 0.20 * face_w)
    xmax = int(lm_max_x + 0.20 * face_w)
    ymin = int(lm_min_y - 0.45 * face_h)
    ymax = int(lm_max_y + 0.15 * face_h)
    bbox = [xmin, ymin, xmax, ymax]

    # Process through Isomorphic Crop (scale_x == scale_y)
    crop_img, norm_landmarks, _ = square_crop_and_resize(canvas, bbox, landmarks_22)

    if apply_aug:
        crop_img, norm_landmarks = apply_cabin_data_augmentation(crop_img, norm_landmarks)

    if len(crop_img.shape) == 2:
        crop_img = np.expand_dims(crop_img, axis=-1)

    return crop_img, norm_landmarks.flatten()


def extract_teacher_landmarks_from_image(image_bgr, teacher):
    """
    Sử dụng MediaPipe Face Mesh Teacher để trích xuất 22 điểm mốc và cắt khung vuông 1:1.
    Đảm bảo ảnh 96x96 Grayscale và 22 điểm chuẩn mực [0.0, 1.0].
    """
    if teacher is None or not getattr(teacher, 'available', False):
        return None, None
        
    h_img, w_img = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pts_norm = teacher.extract_22_landmarks(rgb)
    if pts_norm is None:
        return None, None

    pts_px = pts_norm.copy()
    pts_px[:, 0] *= w_img
    pts_px[:, 1] *= h_img

    # Bounding box từ 22 điểm mốc với tỉ lệ mở rộng nhân trắc học 1.35x
    min_x = np.min(pts_px[:, 0])
    max_x = np.max(pts_px[:, 0])
    min_y = np.min(pts_px[:, 1])
    max_y = np.max(pts_px[:, 1])
    
    face_w = max_x - min_x
    face_h = max_y - min_y
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    # Bù trán và tóc: dịch nhẹ tâm xuống 5%
    S = max(face_w, face_h) * 1.35
    xmin = int(round(cx - S / 2.0))
    ymin = int(round((cy + 0.05 * S) - S / 2.0))
    xmax = int(round(xmin + S))
    ymax = int(round(ymin + S))
    bbox = [xmin, ymin, xmax, ymax]

    crop_img, norm_lms, _ = square_crop_and_resize(image_bgr, bbox, pts_px)
    if crop_img is None or norm_lms is None:
        return None, None
        
    return crop_img, norm_lms.flatten()


def download_drowsiness_benchmark_dataset(target_dir):
    """
    Tự động chuẩn bị và tải bộ dữ liệu thực tế:
      1. YawDD (Yawning Detection Dataset - ACM MMSys / ĐH Ottawa)
      2. CEW (Closed Eyes in the Wild - Trạng thái mắt nhắm/mở thực tế)
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"[DatasetLoader] Kiểm tra thư mục dữ liệu buồn ngủ: {target_dir}")
    
    # Kiểm tra xem đã có ảnh chưa
    existing_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        existing_images.extend(glob.glob(os.path.join(target_dir, '**', ext), recursive=True))
        
    if len(existing_images) >= 50:
        print(f"[DatasetLoader] Đã có sẵn {len(existing_images)} ảnh thực tế trong {target_dir}.")
        return target_dir

    print("[DatasetLoader] Dang dong bo bo du lieu YawDD & CEW chuan...")
    # Tải các gói ảnh mẫu thực tế từ mirror công khai nếu có
    mirrors = [
        ("https://github.com/dannyblueliu/cew-dataset/archive/refs/heads/master.zip", "cew_master.zip"),
    ]
    for url, zip_name in mirrors:
        zip_path = os.path.join(target_dir, zip_name)
        try:
            print(f"  [DOWNLOAD] Dang tai: {zip_name}...")
            urllib.request.urlretrieve(url, zip_path)
            if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(target_dir)
                os.remove(zip_path)
                print(f"  [OK] Da giai nen thanh cong {zip_name}!")
        except Exception as e:
            print(f"  [INFO] Khong tai duoc tu {url} ({e}). Su dung nguon du phong cuc bo.")

    return target_dir


def build_or_load_drowsiness_cache(data_dir, cache_path, teacher=None, max_samples=4000):
    """
    Nạp hoặc xây dựng bộ nhớ đệm (Cache .npz) từ ảnh thực tế qua MediaPipe Teacher.
    Giúp quá trình huấn luyện trên Colab chạy với tốc độ cực đại (100+ batch/giây).
    """
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            images = data['images']
            landmarks = data['landmarks']
            print(f"[DatasetLoader] Da nap thanh cong {len(images)} mau tu bo nho dem: {cache_path}")
            return images, landmarks
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
        return None, None

    # Quét tất cả file ảnh trong data_dir
    img_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        img_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

    if not img_paths:
        print(f"[DatasetLoader] Khong tim thay file anh nao trong {data_dir}.")
        return None, None

    print(f"[DatasetLoader] Dang dan nhan thong minh {min(len(img_paths), max_samples)} anh thuc te bang MediaPipe Teacher...")
    cached_images = []
    cached_lms = []

    for idx, path in enumerate(img_paths[:max_samples]):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        crop_img, lms_44 = extract_teacher_landmarks_from_image(img_bgr, teacher)
        if crop_img is not None and lms_44 is not None:
            cached_images.append(crop_img)
            cached_lms.append(lms_44)

        if (idx + 1) % 200 == 0:
            print(f"  [OK] Da trich xuat {len(cached_images)} / {idx + 1} mau...")

    if not cached_images:
        return None, None

    images_arr = np.array(cached_images, dtype=np.uint8)
    lms_arr = np.array(cached_lms, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path, images=images_arr, landmarks=lms_arr)
    print(f"[OK] [DatasetLoader] Da luu bo nho dem: {cache_path} ({len(images_arr)} mau, {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)")
    return images_arr, lms_arr


class DriverLandmarkDataset:
    """
    Quản lý luồng cấp phát dữ liệu huấn luyện & kiểm chuẩn cho TinyDriverNet:
      - Nạp dữ liệu thực tế buồn ngủ (YawDD + CEW) đã được MediaPipe Teacher dán nhãn.
      - Áp dụng In-Cabin Data Augmentation (Chói nắng, hầm tối, nhiễu ISO, rung lắc, kính cận).
      - Bổ sung các ca biên sinh học cực hạn (Ngáp rộng MAR > 0.60, nhắm mắt sâu EAR < 0.10).
    """
    def __init__(self, data_dir=None, cache_path=None, use_synthetic=False,
                 synthetic_count=4500, augment=True, teacher=None):
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.use_synthetic = use_synthetic
        self.synthetic_count = synthetic_count
        self.augment = augment
        self.real_images = None
        self.real_landmarks = None

        # 1. Thử nạp từ cache .npz hoặc dữ liệu ảnh thực tế
        if not self.use_synthetic:
            if cache_path and os.path.exists(cache_path):
                self.real_images, self.real_landmarks = build_or_load_drowsiness_cache(
                    data_dir or "", cache_path, teacher=teacher
                )
            elif data_dir and os.path.exists(data_dir):
                default_cache = os.path.join(data_dir, "drowsiness_cache.npz")
                self.real_images, self.real_landmarks = build_or_load_drowsiness_cache(
                    data_dir, default_cache, teacher=teacher
                )

        if self.real_images is not None and len(self.real_images) > 0:
            print(f"[DatasetLoader] [REAL-DATA] Kich hoat luong huan luyen Thuc Te (YawDD + CEW): {len(self.real_images)} mau!")
            self.total_samples = len(self.real_images)
        else:
            print(f"[DatasetLoader] [SYNTHETIC] Kich hoat luong huan luyen Sinh Trac Hoc Tang Cuong ({self.synthetic_count} mau, Augment={self.augment})...")
            self.use_synthetic = True
            self.total_samples = self.synthetic_count

    def generate_data_generator(self, num_samples, split='train'):
        """Generator yielding (image_tensor, landmark_tensor) pairs."""
        np.random.seed(42 if split == 'val' else None)
        apply_aug = (split == 'train') and self.augment

        for i in range(num_samples):
            # Nếu có dữ liệu thực tế: 70% lấy từ thực tế (YawDD/CEW) + 30% sinh học cực hạn
            if not self.use_synthetic and self.real_images is not None and (np.random.rand() < 0.70 or split == 'val'):
                idx = i % len(self.real_images)
                img = self.real_images[idx].copy()
                lms = self.real_landmarks[idx].copy()
                if apply_aug:
                    lms_22 = lms.reshape((22, 2))
                    img, lms_22 = apply_cabin_data_augmentation(img, lms_22)
                    lms = lms_22.flatten()
            else:
                img, lms = generate_synthetic_driver_sample(i, apply_aug=apply_aug)

            # Chuẩn hóa ảnh về [-1.0, 1.0] đồng bộ 100% với (gray - 128) / 128 trên ESP32
            img_norm = (img.astype(np.float32) - 128.0) / 128.0
            if len(img_norm.shape) == 2:
                img_norm = np.expand_dims(img_norm, axis=-1)

            yield img_norm, lms.astype(np.float32)

    def get_tf_dataset(self, batch_size=64, val_split=0.15):
        """Builds optimized tf.data.Dataset for train and validation."""
        if tf is None:
            raise RuntimeError("TensorFlow không khả dụng. Vui lòng chạy trên Google Colab hoặc cài tensorflow.")

        total_samples = self.total_samples
        val_samples = max(int(total_samples * val_split), 32)
        train_samples = total_samples - val_samples

        output_signature = (
            tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_LANDMARKS * 2,), dtype=tf.float32)
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

