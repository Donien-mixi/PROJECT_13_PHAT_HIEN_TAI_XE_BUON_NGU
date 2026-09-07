#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🧠 CÔNG CỤ TIỀN XỬ LÝ & GÁN NHÃN SINH HỌC 22 ĐIỂM (PREPROCESS DATASET)
=============================================================================
Đồ án 13: Hệ thống phát hiện tài xế ngủ gật & mất tập trung (Edge AI ESP32-S3)

Chức năng:
1. Thu gom và trích xuất dữ liệu ảnh người thật:
   - Bộ ảnh ngáp người thật (yawn_faces.zip).
   - Bộ ảnh tài xế trong cabin ô tô (Driver Drowsiness Dataset).
   - Bộ ảnh người dùng tự chụp qua Webcam (--capture-webcam).
2. Dùng mô hình Thầy MediaPipe FaceMesh trích xuất 22 điểm Landmark chuẩn giải phẫu.
3. Cắt ô vuông chuẩn hóa Isomorphic Skull Anchor 96x96 Grayscale (đồng bộ 100% với ESP32-S3).
4. Xuất ảnh trực quan kiểm tra (Visual Preview) vẽ 22 điểm để kiểm tra độ bám dính.
5. Đóng gói ra file nén preprocessed_driver_dataset.npz phục vụ huấn luyện Colab / Laptop.
=============================================================================
"""

import sys
import os
import glob
import time
import zipfile
import urllib.request
import argparse
from pathlib import Path
import numpy as np
import cv2

# Đảm bảo in tiếng Việt UTF-8 trên Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
TRAINING_DIR = ROOT_DIR / "training_tinyml"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from distillation import MediaPipeTeacher, estimate_pose_from_landmarks
    from isomorphic_transform import canonical_face_crop
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, NUM_LANDMARKS,
        LEFT_EYE_22, RIGHT_EYE_22, MOUTH_22, NOSE_CHIN_22
    )
except ImportError as e:
    print(f"❌ [LỖI IMPORT]: {e}")
    sys.exit(1)


def compute_ear_and_mar(pts_22_norm):
    """Tính EAR và MAR từ 22 điểm landmark đã chuẩn hóa [0, 1]."""
    pts = pts_22_norm.reshape((22, 2)) * 96.0

    # Mắt trái: P0..P5
    d_l_v1 = float(np.linalg.norm(pts[1] - pts[5]))
    d_l_v2 = float(np.linalg.norm(pts[2] - pts[4]))
    d_l_h  = float(np.linalg.norm(pts[0] - pts[3]))
    ear_left = (d_l_v1 + d_l_v2) / (2.0 * max(d_l_h, 1e-4))

    # Mắt phải: P6..P11
    d_r_v1 = float(np.linalg.norm(pts[7] - pts[11]))
    d_r_v2 = float(np.linalg.norm(pts[8] - pts[10]))
    d_r_h  = float(np.linalg.norm(pts[6] - pts[9]))
    ear_right = (d_r_v1 + d_r_v2) / (2.0 * max(d_r_h, 1e-4))

    ear_avg = (ear_left + ear_right) / 2.0

    # Miệng: P12(khóe trái), P13(khóe phải), P14(trên ngoài), P15(dưới ngoài), P16(trên trong), P17(dưới trong)
    w_mouth = float(np.linalg.norm(pts[12] - pts[13]))
    h_outer = float(np.linalg.norm(pts[14] - pts[15]))
    h_inner = float(np.linalg.norm(pts[16] - pts[17]))
    mar = (h_outer + h_inner) / (2.0 * max(w_mouth, 1e-4))

    return ear_avg, mar, w_mouth


def draw_landmarks_preview(crop_gray_96, pts_22_norm, ear, mar, yaw_deg, sample_idx):
    """
    Vẽ trực quan 22 điểm landmark lên ảnh crop (phóng to 3x: 288x288)
    để người dùng mở xem trực tiếp độ bám dính của các bộ phận.
    """
    canvas_gray = cv2.resize(crop_gray_96, (288, 288), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)

    pts = (pts_22_norm.copy() * 288.0).astype(np.int32)

    # 1. Vẽ Mắt Trái (P0 -> P5) - Màu Xanh Lá
    eye_l_poly = pts[LEFT_EYE_22]
    cv2.polylines(canvas, [eye_l_poly], True, (0, 255, 0), 1, cv2.LINE_AA)
    for p in eye_l_poly:
        cv2.circle(canvas, tuple(p), 3, (0, 255, 0), -1)

    # 2. Vẽ Mắt Phải (P6 -> P11) - Màu Xanh Lá
    eye_r_poly = pts[RIGHT_EYE_22]
    cv2.polylines(canvas, [eye_r_poly], True, (0, 255, 0), 1, cv2.LINE_AA)
    for p in eye_r_poly:
        cv2.circle(canvas, tuple(p), 3, (0, 255, 0), -1)

    # 3. Vẽ Miệng (P12 -> P17) - Màu Đỏ Cam
    # Viền ngoài: P12 -> P14 -> P13 -> P15
    outer_poly = np.array([pts[12], pts[14], pts[13], pts[15]], dtype=np.int32)
    cv2.polylines(canvas, [outer_poly], True, (0, 70, 255), 2, cv2.LINE_AA)
    # Khoang trong: P16 -> P17
    cv2.line(canvas, tuple(pts[16]), tuple(pts[17]), (0, 200, 255), 1, cv2.LINE_AA)
    for idx in MOUTH_22:
        # Làm nổi bật khóe môi P12, P13
        if idx in [12, 13]:
            cv2.circle(canvas, tuple(pts[idx]), 4, (0, 0, 255), -1)
        else:
            cv2.circle(canvas, tuple(pts[idx]), 3, (0, 140, 255), -1)

    # 4. Vẽ Sống Mũi & Cằm (P18 -> P21) - Màu Vàng & Xanh Dương
    cv2.line(canvas, tuple(pts[18]), tuple(pts[19]), (255, 200, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[19]), tuple(pts[20]), (255, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[18]), 3, (255, 255, 0), -1) # Nasion
    cv2.circle(canvas, tuple(pts[19]), 4, (255, 255, 0), -1) # Tip
    cv2.circle(canvas, tuple(pts[20]), 3, (255, 255, 0), -1) # Subnasale
    cv2.circle(canvas, tuple(pts[21]), 5, (255, 50, 50), -1)  # Chin Gnathion

    # Viết nhãn thông tin
    state_txt = "NGÁP" if mar >= 0.40 else ("NHẮM" if ear < 0.21 else "TỈNH")
    info_line = f"#{sample_idx:03d} | EAR:{ear:.2f} | MAR:{mar:.2f} | {state_txt}"
    cv2.rectangle(canvas, (0, 0), (288, 24), (20, 20, 20), -1)
    cv2.putText(canvas, info_line, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def extract_sample_from_face_bgr(img_bgr, teacher):
    """
    Trích xuất 1 mẫu hoàn chỉnh từ ảnh khuôn mặt BGR bằng MediaPipe Teacher:
    - Trả về: (crop_gray_96, landmarks_44, pose_3, ear, mar, w_mouth) hoặc None
    """
    if img_bgr is None or teacher is None or not getattr(teacher, 'available', False):
        return None

    h_img, w_img = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pts_norm = teacher.extract_22_landmarks(rgb)
    if pts_norm is None:
        return None

    pts_px = pts_norm.copy()
    pts_px[:, 0] *= w_img
    pts_px[:, 1] *= h_img

    # Cắt chuẩn hình học sọ mặt (Skull Anchor) 96x96
    crop_gray, norm_lms, _ = canonical_face_crop(img_bgr, pts_px, jitter=False)
    if crop_gray is None or norm_lms is None:
        return None

    pose = estimate_pose_from_landmarks(norm_lms)
    ear, mar, w_mouth = compute_ear_and_mar(norm_lms)

    return crop_gray, norm_lms.flatten(), pose, ear, mar, w_mouth


def prepare_benchmark_datasets(raw_datasets_dir):
    """
    Thu thập ảnh từ:
    1. yawn_faces.zip (ảnh ngáp người thật có sẵn).
    2. drowsiness_cabin (bộ dữ liệu tài xế trong cabin ô tô).
    """
    os.makedirs(raw_datasets_dir, exist_ok=True)

    # 1. Giải nén yawn_faces.zip
    candidate_yawn_zips = [
        TRAINING_DIR / "yawn_faces.zip",
        ROOT_DIR / "yawn_faces.zip",
        raw_datasets_dir / "yawn_faces.zip"
    ]
    for bz in candidate_yawn_zips:
        if bz.exists() and zipfile.is_zipfile(str(bz)):
            target_unzip = raw_datasets_dir / "yawn_faces"
            if not target_unzip.exists() or len(list(target_unzip.glob("*.jpg"))) == 0:
                print(f"  [BUNDLE] Tìm thấy {bz.name}. Đang giải nén vào {target_unzip.name}...")
                try:
                    with zipfile.ZipFile(str(bz), 'r') as zf:
                        zf.extractall(str(raw_datasets_dir))
                    print("  ✓ Đã giải nén thành công ảnh ngáp thật!")
                except Exception as e:
                    print(f"  ⚠️ Lỗi giải nén: {e}")
            break

    # 2. Tải Driver Drowsiness Cabin Dataset nếu cần
    existing_imgs = list(raw_datasets_dir.rglob("*.jpg")) + list(raw_datasets_dir.rglob("*.png"))
    if len(existing_imgs) < 100:
        cabin_url = "https://raw.githubusercontent.com/hazeeq911/Driver-Drowsiness-Detection/main/drowsiness.zip"
        cabin_zip = raw_datasets_dir / "drowsiness_cabin.zip"
        print(f"  [DOWNLOAD] Đang tải bộ dữ liệu tài xế trong cabin ô tô (~6.8MB)...")
        try:
            req = urllib.request.Request(cabin_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=40) as resp, open(str(cabin_zip), 'wb') as f:
                f.write(resp.read())
            if cabin_zip.exists() and zipfile.is_zipfile(str(cabin_zip)):
                with zipfile.ZipFile(str(cabin_zip), 'r') as zf:
                    zf.extractall(str(raw_datasets_dir / "cabin_drowsiness"))
                cabin_zip.unlink()
                print("  ✓ Đã tải và giải nén bộ dữ liệu cabin!")
        except Exception as e:
            print(f"  ⚠️ Không thể tải dataset tự động ({e}). Tiếp tục với các dữ liệu hiện có.")
            if cabin_zip.exists():
                try: cabin_zip.unlink()
                except Exception: pass


def run_preprocessing_pipeline(data_dirs, output_npz, preview_dir, max_previews=50):
    """
    Tiến hành quét toàn bộ ảnh thật, dán nhãn 22 điểm bằng MediaPipe,
    cắt chuẩn 96x96 Skull Anchor, xuất ảnh preview và đóng gói file .npz.
    """
    print("\n" + "=" * 75)
    print("🚀 BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ & GÁN NHÃN 22 ĐIỂM DỮ LIỆU THỰC TẾ")
    print("=" * 75)

    # 1. Khởi tạo Mô hình Thầy MediaPipe
    print("[1/4] Đang khởi tạo mô hình Thầy MediaPipe FaceMesh...")
    teacher = MediaPipeTeacher()
    if not teacher.available:
        print("❌ [LỖI] Không thể khởi tạo MediaPipe FaceMesh! Hãy cài đặt: pip install mediapipe")
        sys.exit(1)
    print("  ✓ MediaPipe Teacher đã sẵn sàng trích xuất 22 điểm giải phẫu!")

    # 2. Quét tập tin ảnh
    all_img_paths = []
    for d in data_dirs:
        p = Path(d)
        if p.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                all_img_paths.extend(list(p.rglob(ext)))

    # Loại bỏ ảnh trùng lặp
    all_img_paths = sorted(list(set(all_img_paths)))
    print(f"\n[2/4] Đã tìm thấy tổng cộng: {len(all_img_paths)} ảnh thô từ các thư mục dữ liệu.")
    if len(all_img_paths) == 0:
        print("⚠️ Không có ảnh nào để xử lý!")
        return None

    # 3. Tiến hành trích xuất và dán nhãn
    print(f"\n[3/4] Đang tiến hành tiền xử lý, cắt Isomorphic 96x96 và gán nhãn Ground-Truth...")
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    processed_images = []
    processed_landmarks = []
    processed_poses = []
    processed_mars = []
    processed_ears = []

    yawn_count = 0
    sleep_count = 0
    normal_count = 0
    preview_saved = 0

    t0 = time.time()
    for idx, img_path in enumerate(all_img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        res = extract_sample_from_face_bgr(img_bgr, teacher)
        if res is None:
            continue

        crop_gray, lms_44, pose_3, ear, mar, w_mouth = res

        # Phân loại trạng thái
        if mar >= 0.40:
            yawn_count += 1
        elif ear < 0.21:
            sleep_count += 1
        else:
            normal_count += 1

        processed_images.append(crop_gray)
        processed_landmarks.append(lms_44)
        processed_poses.append(pose_3)
        processed_mars.append(mar)
        processed_ears.append(ear)

        # Lưu ảnh preview trực quan cho người dùng kiểm tra
        if preview_dir and preview_saved < max_previews:
            # Chọn đều các mẫu: cả ngáp, nhắm mắt và bình thường
            should_save = (preview_saved < 15) or (mar >= 0.40 and preview_saved < max_previews) or (ear < 0.21 and preview_saved < max_previews)
            if should_save:
                preview_saved += 1
                canvas = draw_landmarks_preview(
                    crop_gray, lms_44.reshape((22, 2)), ear, mar,
                    yaw_deg=float(pose_3[0] * 90.0),
                    sample_idx=preview_saved
                )
                prev_path = Path(preview_dir) / f"preview_{preview_saved:03d}_{'YAWN' if mar>=0.40 else ('SLEEP' if ear<0.21 else 'NORMAL')}.jpg"
                cv2.imwrite(str(prev_path), canvas)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_img_paths):
            print(f"  • Đã duyệt: {idx + 1}/{len(all_img_paths)} | Hợp lệ: {len(processed_images)} mẫu (Ngáp: {yawn_count}, Nhắm: {sleep_count}, Bình thường: {normal_count})")

    dur = time.time() - t0
    total_valid = len(processed_images)
    print(f"\n✓ Tiền xử lý hoàn tất sau {dur:.1f}s. Tỷ lệ thành công: {total_valid}/{len(all_img_paths)} ({total_valid/max(len(all_img_paths),1)*100:.1f}%)")

    if total_valid == 0:
        print("❌ Không trích xuất được mẫu hợp lệ nào!")
        return None

    # 4. Đóng gói ra file .npz
    print(f"\n[4/4] Đang đóng gói dữ liệu vào: {Path(output_npz).resolve()}...")
    images_arr = np.array(processed_images, dtype=np.uint8)
    landmarks_arr = np.array(processed_landmarks, dtype=np.float32)
    poses_arr = np.array(processed_poses, dtype=np.float32)
    mars_arr = np.array(processed_mars, dtype=np.float32)
    ears_arr = np.array(processed_ears, dtype=np.float32)

    os.makedirs(Path(output_npz).parent, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        images=images_arr,
        landmarks=landmarks_arr,
        poses=poses_arr,
        mars=mars_arr,
        ears=ears_arr
    )

    size_mb = os.path.getsize(str(output_npz)) / (1024 * 1024)
    print("=" * 75)
    print(f"🎉 XUẤT TẬP DỮ LIỆU TIỀN XỬ LÝ THÀNH CÔNG: {Path(output_npz).name}")
    print(f"   • Kích thước file : {size_mb:.2f} MB")
    print(f"   • Tổng số mẫu     : {total_valid} mẫu")
    print(f"   • Mẫu Ngáp (Há to): {yawn_count} mẫu ({yawn_count/total_valid*100:.1f}%)")
    print(f"   • Mẫu Nhắm mắt    : {sleep_count} mẫu ({sleep_count/total_valid*100:.1f}%)")
    print(f"   • Mẫu Bình thường : {normal_count} mẫu ({normal_count/total_valid*100:.1f}%)")
    if preview_dir:
        print(f"   • Đã lưu {preview_saved} ảnh trực quan kiểm tra tại: {Path(preview_dir).resolve()}")
    print("=" * 75)
    return str(output_npz)


def main():
    parser = argparse.ArgumentParser(description="Công cụ Tiền Xử Lý & Gán Nhãn 22 Điểm Dữ Liệu Cộng Đồng Chuẩn")
    parser.add_argument("--data-dir", type=str, default=None, help="Thư mục chứa tập dữ liệu ảnh khuôn mặt bổ sung (nếu có)")
    parser.add_argument("--output-npz", type=str, default=str(TRAINING_DIR / "preprocessed_driver_dataset.npz"),
                        help="Đường dẫn file .npz kết quả")
    parser.add_argument("--preview-dir", type=str, default=str(ROOT_DIR / "output" / "preprocessed_preview"),
                        help="Thư mục lưu ảnh preview trực quan")
    parser.add_argument("--max-previews", type=int, default=50, help="Số lượng ảnh preview tối đa")
    args = parser.parse_args()

    raw_datasets_dir = ROOT_DIR / "datasets" / "raw_faces"

    # Chuẩn bị dữ liệu benchmark (yawn_faces + cabin)
    prepare_benchmark_datasets(raw_datasets_dir)

    # Thu thập tất cả các thư mục nguồn tập dữ liệu cộng đồng
    search_dirs = [raw_datasets_dir]
    if args.data_dir:
        search_dirs.append(Path(args.data_dir))

    # Chạy quy trình tiền xử lý
    run_preprocessing_pipeline(
        data_dirs=search_dirs,
        output_npz=args.output_npz,
        preview_dir=args.preview_dir,
        max_previews=args.max_previews
    )


if __name__ == "__main__":
    main()
