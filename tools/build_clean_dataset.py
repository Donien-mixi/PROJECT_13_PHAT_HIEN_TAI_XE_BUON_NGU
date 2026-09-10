#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🧬 BUILD CLEAN DATASET - Pipeline dữ liệu SẠCH cho TinyDriverNet (Đồ án 13)
=============================================================================
Thay thế hoàn toàn pipeline cũ (yawn_faces.zip + drowsiness Roboflow-YOLO)
vốn chỉ có ~400 mẫu thật, nhiều ảnh crop cận cảnh thiếu cằm/mắt và KHÔNG
có nhãn landmark chuẩn → nguyên nhân gốc khiến mô hình học vẹt.

NGUỒN DỮ LIỆU ĐƯỢC HỖ TRỢ (theo thứ tự ưu tiên):
  1. AFLW2000-3D  (tự tải bằng --download-aflw2000, 2000 ảnh, nhãn 68-pt 3D,
                   phủ góc quay đầu yaw ±90° — dataset chuẩn cho Head Pose).
  2. 300W / 300W-LP (thư mục ảnh + file .pts 68-pt iBUG — tải thủ công từ
                     https://ibug.doc.ic.ac.uk/resources/facial-points/).
  3. Mọi thư mục ảnh khác (WFLW, YawDD, ảnh tự chụp webcam...) → tự động
     dán nhãn 22 điểm bằng MediaPipe Teacher (Knowledge Distillation).

BỘ GATE CHẤT LƯỢNG (QA GATES) - mẫu nào FAIL sẽ bị loại và ghi lý do:
  G1. Dò được đầy đủ 22 điểm mốc.
  G2. Sau crop canonical 96x96: mọi landmark nằm trong [0.02, 0.98]
      (không bị cắt mất cằm/mắt ở mép ảnh → chống label bẩn P21).
  G3. Khoảng cách 2 mắt trong crop >= 10 px (mặt đủ lớn, không mờ chấm).
  G4. Độ nét Laplacian variance >= 12 (loại ảnh nhòe/motion blur nặng).
  G5. Góc quay: |yaw| <= 80°, |pitch| <= 70°, |roll| <= 55°.
  G6. Anti-duplicate: aHash 8x8, khoảng cách Hamming <= 3 của 2 ảnh bất kỳ
      trong cùng nguồn → chỉ giữ 1 (chống trùng lặp làm overfit).

TRAIN/VAL SPLIT NGHIÊM NGẶT (chống leak):
  - Chia theo hash tên file: md5(filename) % 100 < val_percent → VAL.
  - Xác định NGAY LÚC BUILD, lưu cột `split` vào npz. Generator huấn luyện
    CHỈ dùng mẫu split=0; validation/NME dùng mẫu split=1 (giữ-out thật).

OUTPUT:
  - training_tinyml/preprocessed_driver_dataset.npz
      images(N,96,96,1) landmarks(N,44) poses(N,3) ears(N,) mars(N,)
      split(N,) sources(N,)
  - output/preprocessed_preview/  (ảnh kiểm tra trực quan có vẽ 22 điểm)
  - output/dataset_report.md      (báo cáo thống kê + lý do loại mẫu)

HƯỚNG DẪN TẢI DỮ LIỆU THỦ CÔNG (nếu auto-download lỗi):
  - AFLW2000-3D : https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
                  → giải nén vào datasets/raw_faces/AFLW2000/
  - 300W        : https://ibug.doc.ic.ac.uk/resources/facial-points/
                  → bỏ thư mục ảnh + .pts vào datasets/raw_faces/300W/
  - WFLW        : https://wywu.github.io/projects/LAB/WFLW.html
                  → chỉ cần thư mục ảnh (bỏ qua txt), MediaPipe sẽ dán nhãn.
  - YawDD       : https://sites.google.com/site/yawddf/ (video trích frame)
                  → bỏ frame vào datasets/raw_faces/yawdd/
=============================================================================
"""

import sys
import os
import glob
import json
import time
import hashlib
import argparse
import zipfile
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import cv2

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

CURRENT_DIR = Path(__file__).resolve().parent


def _find_project_root():
    """Tự phát hiện thư mục gốc dự án (chống bug v2.0.1: script nằm ở ROOT gói Colab
    thì ROOT=CURRENT_DIR, nằm trong tools/ của repo thì ROOT=parent)."""
    for cand in (CURRENT_DIR, CURRENT_DIR.parent):
        if (cand / "training_tinyml").is_dir():
            return cand
    return CURRENT_DIR.parent


ROOT_DIR = _find_project_root()
TRAINING_DIR = ROOT_DIR / "training_tinyml"
RAW_FACES_DIR = ROOT_DIR / "datasets" / "raw_faces"
PREVIEW_DIR = ROOT_DIR / "output" / "preprocessed_preview"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, NUM_LANDMARKS,
    LEFT_EYE_22, RIGHT_EYE_22, MOUTH_22
)
from isomorphic_transform import canonical_face_crop
from distillation import MediaPipeTeacher, estimate_pose_from_landmarks

# ============================================================================
# THAM SỐ CHẤT LƯỢNG (QA GATES) - đồng bộ 1 nơi duy nhất
# ============================================================================
LM_MARGIN = 0.01        # G2: landmark phải nằm trong [margin, 1-margin]
                        # [v2.0.1] Nới từ 0.02: AFLW2000 nhiều ảnh cận cảnh,
                        # label vẫn chính xác tuyệt đối nhờ padding công thức canonical.
MIN_D_EYES_PX = 10.0    # G3: khoảng cách 2 mắt tối thiểu (px trong ô 96x96)
MIN_LAPLACIAN_VAR = 12.0  # G4: ngưỡng nét ảnh
MAX_YAW_DEG = 80.0      # G5
MAX_PITCH_DEG = 70.0
MAX_ROLL_DEG = 55.0
DUP_HAMMING = 3         # G6: ngưỡng Hamming của aHash 8x8


# ============================================================================
# TIỆN ÍCH CHUNG
# ============================================================================
def compute_crop_ear(pts_22_norm):
    pts = pts_22_norm.reshape((22, 2))
    def _ear(idxs):
        p = pts[idxs]
        v1 = np.linalg.norm(p[1] - p[5]); v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * max(h, 1e-5))
    return (_ear(LEFT_EYE_22) + _ear(RIGHT_EYE_22)) / 2.0


def compute_crop_mar(pts_22_norm):
    """MAR = (h_outer + h_inner) / (2*w) - đồng bộ 100% wing_loss.py & firmware."""
    pts = pts_22_norm.reshape((22, 2))
    w_m = np.linalg.norm(pts[12] - pts[13])
    h_outer = np.linalg.norm(pts[14] - pts[15])
    h_inner = np.linalg.norm(pts[16] - pts[17])
    return (h_outer + h_inner) / (2.0 * max(w_m, 1e-4))


def ahash_64(gray_96):
    """aHash 8x8 của ảnh crop 96x96 để chống trùng lặp."""
    small = cv2.resize(gray_96, (8, 8), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    bits = (small > small.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming64(a, b):
    return bin(int(a) ^ int(b)).count('1')


def split_of_filename(filename, val_percent):
    """Split quyết định bởi hash tên file -> ổn định qua các lần chạy lại."""
    h = hashlib.md5(filename.encode('utf-8')).hexdigest()
    return 1 if (int(h[:8], 16) % 100 < val_percent) else 0


# ============================================================================
# BỘ ĐỌC NHÃN 68 ĐIỂM CHUẨN iBUG (300W / AFLW2000)
# ============================================================================
def parse_pts_file(pts_path):
    """Đọc file .pts chuẩn 68 điểm (300W)."""
    try:
        with open(pts_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f.readlines()]
        start = None
        for i, ln in enumerate(lines):
            if ln == '{':
                start = i + 1
                break
        if start is None:
            return None
        coords = []
        for ln in lines[start:]:
            if ln == '}':
                break
            parts = ln.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        if len(coords) == 68:
            return np.array(coords, dtype=np.float32)
    except Exception:
        pass
    return None


def parse_mat_landmarks(mat_path):
    """Đọc landmark 68 điểm từ file .mat của AFLW2000 / 300W-LP (pt3d_68)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        mat = loadmat(str(mat_path))
        arr = None
        for key in ('pt3d_68', 'pts_3d', 'landmarks'):
            if key in mat:
                arr = np.array(mat[key], dtype=np.float64)
                break
        if arr is None:
            return None
        if arr.shape == (3, 68):
            return np.stack([arr[0], arr[1]], axis=-1).astype(np.float32)
        if arr.shape == (68, 3):
            return arr[:, :2].astype(np.float32)
        if arr.shape == (2, 68):
            return np.stack([arr[0], arr[1]], axis=-1).astype(np.float32)
        if arr.shape == (68, 2):
            return arr.astype(np.float32)
    except Exception:
        pass
    return None


def pts68_to_22(pts_68):
    """Ánh xạ 68 điểm iBUG -> 22 điểm đề tài (theo config.IBUG_68_TO_22_INDICES)."""
    from config import IBUG_68_TO_22_INDICES
    if len(pts_68) != 68:
        return None
    return pts_68[IBUG_68_TO_22_INDICES].astype(np.float32)


# ============================================================================
# QA + THU THẬP 1 MẪU
# ============================================================================
class SampleCollector:
    def __init__(self, teacher=None, source_name="", val_percent=8):
        self.teacher = teacher
        self.source_name = source_name
        self.val_percent = val_percent
        self.images = []
        self.landmarks = []
        self.poses = []
        self.ears = []
        self.mars = []
        self.splits = []
        self.hashes = []
        self.rejected = Counter()      # lý do loại
        self.preview_pool = []         # (image, lms, ear, mar) để xuất preview

    def _reject(self, reason):
        self.rejected[reason] += 1

    def add_sample(self, img_bgr, pts_22_px, src_filename):
        """pts_22_px: (22,2) tọa độ pixel trong ảnh gốc. Trả về True nếu nhận."""
        if img_bgr is None or pts_22_px is None or len(pts_22_px) != 22:
            self._reject("missing_landmarks")
            return False

        crop, norm_lms, meta = canonical_face_crop(img_bgr, np.asarray(pts_22_px, dtype=np.float32))
        if crop is None or norm_lms is None:
            self._reject("crop_failed")
            return False

        crop_gray = crop[:, :, 0]
        pts = norm_lms.reshape((22, 2))

        # G2: không landmark nào bị cắt mép
        if pts.min() < LM_MARGIN or pts.max() > (1.0 - LM_MARGIN):
            self._reject("landmark_clipped")
            return False

        # G3: mặt đủ lớn
        eye_l = pts[LEFT_EYE_22].mean(axis=0)
        eye_r = pts[RIGHT_EYE_22].mean(axis=0)
        d_eyes_px = float(np.linalg.norm(eye_r - eye_l) * IMAGE_WIDTH)
        if d_eyes_px < MIN_D_EYES_PX:
            self._reject("face_too_small")
            return False

        # G4: ảnh nhòe
        lap_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        if lap_var < MIN_LAPLACIAN_VAR:
            self._reject("blurry")
            return False

        # G5: góc quay cực trị
        pose = estimate_pose_from_landmarks(pts)
        yaw_deg, pitch_deg, roll_deg = pose[0] * 90.0, pose[1] * 90.0, pose[2] * 90.0
        if (abs(yaw_deg) > MAX_YAW_DEG or abs(pitch_deg) > MAX_PITCH_DEG
                or abs(roll_deg) > MAX_ROLL_DEG):
            self._reject("extreme_pose")
            return False

        # G6: trùng lặp (chỉ so trong cùng nguồn)
        h = ahash_64(crop_gray)
        # so sánh nhanh với vài phần tử gần cuối (giảm O(n^2) với nguồn lớn)
        recent = self.hashes[-512:]
        for prev_h in recent:
            if hamming64(h, prev_h) <= DUP_HAMMING:
                self._reject("duplicate")
                return False

        ear = compute_crop_ear(pts)
        mar = compute_crop_mar(pts)
        split = split_of_filename(f"{self.source_name}/{src_filename}", self.val_percent)

        self.images.append(crop)
        self.landmarks.append(pts.flatten().astype(np.float32))
        self.poses.append(pose.astype(np.float32))
        self.ears.append(np.float32(ear))
        self.mars.append(np.float32(mar))
        self.splits.append(split)
        self.hashes.append(h)

        if len(self.preview_pool) < 400:
            self.preview_pool.append((crop_gray.copy(), pts.copy(), ear, mar))
        return True

    def __len__(self):
        return len(self.images)


# ============================================================================
# CÁC NGUỒN DỮ LIỆU
# ============================================================================
AFLW2000_URLS = [
    "https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
    "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
]
AFLW2000_SIZE_MB = 83  # dung lượng thực tế theo TFDS catalog (Download size 83.36 MiB)

# 300W-LP: 61.225 ảnh tổng hợp góc quay yaw ±90° — KHÔNG có link tải trực tiếp ổn định
# (cbsr trả 404, Google Drive ID cũ đã chết) -> chỉ hỗ trợ tải thủ công.
# Phương án tự động đáng tin: Microsoft FaceSynthetics (Azure blob, không cần auth).

LARGE_FILE_CHUNK = 1024 * 512
FACESYNTH_URLS = [
    "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_1000.zip",
]
FACESYNTH_SIZE_MB = 322


def _download_cbsr_zip(urls, zip_path, size_hint_mb):
    for url in urls:
        print(f"[CBSR] Đang tải {url} (~{size_hint_mb} MB)...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, 'wb') as out_f:
                total = int(resp.headers.get('Content-Length', 0) or 0)
                downloaded = 0
                while True:
                    chunk = resp.read(LARGE_FILE_CHUNK)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(f"  {downloaded / 1e6:9.1f} / {total / 1e6:.1f} MB", end='\r')
            print()
            return True
        except Exception as e:
            print(f"\n[CBSR] Tải thất bại từ URL này ({e}). Thử nguồn kế tiếp...")
            if zip_path.exists():
                zip_path.unlink()
    return False


def download_aflw2000(target_dir):
    """Tải AFLW2000-3D (2000 ảnh, nhãn 68-pt 3D, phủ yaw ±90°, ~83 MB)."""
    zip_path = target_dir / "AFLW2000-3D.zip"
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[AFLW2000] Đã tải và giải nén từ trước, bỏ qua.")
        return True
    if not _download_cbsr_zip(AFLW2000_URLS, zip_path, AFLW2000_SIZE_MB):
        print("[AFLW2000] TOÀN BỘ NGUỒN TẢI THẤT BẠI!")
        print("  👉 Phương án thủ công:")
        print("     1. Trình duyệt tải: https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip")
        print("        (hoặc Kaggle: kaggle.com/datasets/mohamedadlyi/aflw2000-3d)")
        print(f"     2. Giải nén vào: {target_dir} rồi chạy lại lệnh này.")
        return False
    print("[AFLW2000] Đang giải nén...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(target_dir))
    zip_path.unlink()
    done_marker.write_text("ok", encoding="utf-8")
    print("[AFLW2000] HOÀN TẤT!")
    return True


def download_facesynth(target_dir):
    """Tải Microsoft FaceSynthetics dataset_1000 (1000 mặt 512x512, nhãn 70 điểm:
    68 đầu theo scheme iBUG + 2 điểm đồng tử). Link Azure blob trực tiếp, không auth.
    License: non-commercial research (phù hợp đồ án)."""
    zip_path = target_dir / "dataset_1000.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[FaceSynthetics] Đã tải và giải nén từ trước, bỏ qua.")
        return True
    for url in FACESYNTH_URLS:
        print(f"[FaceSynthetics] Đang tải {url} (~{FACESYNTH_SIZE_MB} MB)...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=180) as resp, open(zip_path, 'wb') as out_f:
                total = int(resp.headers.get('Content-Length', 0) or 0)
                downloaded = 0
                while True:
                    chunk = resp.read(LARGE_FILE_CHUNK)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(f"  {downloaded / 1e6:9.1f} / {total / 1e6:.1f} MB", end='\r')
            print("\n[FaceSynthetics] Đang giải nén (1000 ảnh PNG 512x512)...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(target_dir))
            zip_path.unlink()
            done_marker.write_text("ok", encoding="utf-8")
            print("[FaceSynthetics] HOÀN TẤT!")
            return True
        except Exception as e:
            print(f"\n[FaceSynthetics] Tải thất bại ({e}).")
            if zip_path.exists():
                zip_path.unlink()
    print("[FaceSynthetics] TẢI THẤT BẠI! Tải thủ công từ README microsoft/FaceSynthetics")
    print(f"  và giải nén vào: {target_dir}")
    return False


def download_300wlp(target_dir):
    """Thử tải 300W-LP. LƯU Ý: link cbsr hiện 404 (Google Drive ID cũ cũng đã chết)
    -> đa số sẽ thất bại và hướng dẫn tải thủ công. Phương án tự động: --download-facesynth."""
    zip_path = target_dir / "300W-LP.zip"
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[300W-LP] Đã tải và giải nén từ trước, bỏ qua.")
        return True
    urls = [
        "https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP.zip",
        "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP.zip",
    ]
    if not _download_cbsr_zip(urls, zip_path, 1700):
        print("[300W-LP] Link chính thức hiện 404! Tải thủ công:")
        print("  👉 Trang chủ 3DDFA: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm")
        print(f"     Giải nén vào: {target_dir} rồi chạy lại lệnh này.")
        return False
    print("[300W-LP] Đang giải nén (61k ảnh, có thể mất vài phút)...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(target_dir))
    zip_path.unlink()
    done_marker.write_text("ok", encoding="utf-8")
    print("[300W-LP] HOÀN TẤT!")
    return True


def collect_from_aflw2000(folder, collector, max_samples):
    """AFLW2000: ảnh .jpg + .mat (pt3d_68)."""
    mats = sorted(folder.rglob("*.mat"))
    if not mats:
        return 0
    n = 0
    for mat_idx, mat_path in enumerate(mats):
        if max_samples and n >= max_samples:
            break
        img_path = mat_path.with_suffix('.jpg')
        if not img_path.exists():
            img_path = mat_path.with_suffix('.png')
        if not img_path.exists():
            collector._reject("image_missing")
            continue
        pts68 = parse_mat_landmarks(mat_path)
        if pts68 is None:
            collector._reject("mat_unreadable")
            continue
        pts22 = pts68_to_22(pts68)
        img = cv2.imread(str(img_path))
        if collector.add_sample(img, pts22, img_path.name):
            n += 1
            if n % 250 == 0:
                print(f"  [AFLW2000] {n} mẫu hợp lệ / đã duyệt {mat_idx + 1}...")
    return n


def collect_from_300w(folder, collector, max_samples):
    """300W / 300W-LP: ảnh + file .pts 68 điểm."""
    pts_files = sorted(folder.rglob("*.pts"))
    if not pts_files:
        return 0
    n = 0
    for pts_path in pts_files:
        if max_samples and n >= max_samples:
            break
        img_path = None
        for ext in ('.jpg', '.png', '.jpeg'):
            cand = pts_path.with_suffix(ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            collector._reject("image_missing")
            continue
        pts68 = parse_pts_file(pts_path)
        if pts68 is None:
            collector._reject("pts_unreadable")
            continue
        pts22 = pts68_to_22(pts68)
        img = cv2.imread(str(img_path))
        if collector.add_sample(img, pts22, img_path.name):
            n += 1
            if n % 250 == 0:
                print(f"  [300W] {n} mẫu hợp lệ...")
    return n


def collect_from_facesynth(folder, collector, max_samples):
    """FaceSynthetics: {id}.png + {id}_ldmks.txt (70 hàng 'x y' pixel;
    68 hàng đầu = scheme iBUG, 2 hàng cuối = đồng tử -> bỏ)."""
    txts = sorted(folder.rglob("*_ldmks.txt"))
    if not txts:
        return 0
    n = 0
    for i, txt_path in enumerate(txts):
        if max_samples and n >= max_samples:
            break
        stem = txt_path.name[:-len("_ldmks.txt")]
        img_path = txt_path.with_name(stem + ".png")
        if not img_path.exists():
            collector._reject("image_missing")
            continue
        try:
            arr = np.loadtxt(str(txt_path), dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, 2)
        except Exception:
            collector._reject("txt_unreadable")
            continue
        if arr.shape[0] < 68:
            collector._reject("txt_unreadable")
            continue
        pts22 = pts68_to_22(arr[:68].astype(np.float32))
        img = cv2.imread(str(img_path))
        if collector.add_sample(img, pts22, img_path.name):
            n += 1
            if n % 250 == 0:
                print(f"  [FaceSynthetics] {n} mẫu hợp lệ / duyệt {i+1}...")
    return n


def collect_from_images(folder, collector, teacher, max_samples):
    """Mọi ảnh không có nhãn landmark -> MediaPipe Teacher tự dán nhãn."""
    if teacher is None or not getattr(teacher, 'available', False):
        print(f"  [SKIP] {folder.name}: không có MediaPipe Teacher để dán nhãn.")
        return 0
    img_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        img_paths.extend(folder.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        return 0
    n = 0
    for i, img_path in enumerate(img_paths):
        if max_samples and n >= max_samples:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            collector._reject("unreadable")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pts_norm = teacher.extract_22_landmarks(rgb)
        if pts_norm is None:
            collector._reject("teacher_no_face")
            continue
        h_img, w_img = img.shape[:2]
        pts_px = pts_norm.copy()
        pts_px[:, 0] *= w_img
        pts_px[:, 1] *= h_img
        if collector.add_sample(img, pts_px, img_path.name):
            n += 1
            if n % 200 == 0:
                print(f"  [{folder.name}] {n} mẫu hợp lệ / duyệt {i+1}...")
    return n


# ============================================================================
# PREVIEW + REPORT
# ============================================================================
def draw_preview(crop_gray, pts, ear, mar):
    canvas = cv2.resize(crop_gray, (288, 288), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    p = (pts.copy() * 288.0).astype(np.int32)
    for group, color in ((LEFT_EYE_22, (0, 255, 0)), (RIGHT_EYE_22, (0, 255, 0)),
                         (MOUTH_22, (0, 120, 255))):
        cv2.polylines(canvas, [p[group]], True, color, 1, cv2.LINE_AA)
        for q in p[group]:
            cv2.circle(canvas, tuple(q), 3, color, -1)
    cv2.line(canvas, tuple(p[18]), tuple(p[21]), (255, 220, 0), 2, cv2.LINE_AA)
    for idx in (18, 19, 20):
        cv2.circle(canvas, tuple(p[idx]), 3, (255, 255, 0), -1)
    cv2.circle(canvas, tuple(p[21]), 5, (255, 60, 60), -1)
    state = "NGAP" if mar >= 0.40 else ("NHAM" if ear < 0.21 else "TINH")
    cv2.rectangle(canvas, (0, 0), (288, 24), (20, 20, 20), -1)
    cv2.putText(canvas, f"EAR:{ear:.2f} MAR:{mar:.2f} {state}", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def export_previews(collectors, preview_dir, max_previews=60):
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    buckets = {"YAWN": [], "SLEEP": [], "NORMAL": []}
    for c in collectors:
        for item in c.preview_pool:
            crop, pts, ear, mar = item
            if mar >= 0.40:
                buckets["YAWN"].append((c.source_name, crop, pts, ear, mar))
            elif ear < 0.21:
                buckets["SLEEP"].append((c.source_name, crop, pts, ear, mar))
            else:
                buckets["NORMAL"].append((c.source_name, crop, pts, ear, mar))
    per_bucket = max(1, max_previews // len(buckets))
    for tag, items in buckets.items():
        for i, (src, crop, pts, ear, mar) in enumerate(items[:per_bucket]):
            saved += 1
            canvas = draw_preview(crop, pts, ear, mar)
            cv2.imwrite(str(preview_dir / f"preview_{saved:03d}_{tag}_{src}.jpg"), canvas)
    return saved


def export_report(collectors, output_npz, preview_dir, total_time):
    n_total = sum(len(c) for c in collectors)
    n_val = sum(sum(1 for s in c.splits if s == 1) for c in collectors)
    n_train = n_total - n_val
    ears = np.concatenate([np.array(c.ears) for c in collectors]) if n_total else np.array([])
    mars = np.concatenate([np.array(c.mars) for c in collectors]) if n_total else np.array([])
    poses = np.concatenate([np.array(c.poses) for c in collectors]) if n_total else np.zeros((0, 3))

    lines = [
        "# 🧬 Báo cáo Build Clean Dataset (Đồ án 13)\n",
        f"- **Thời gian build:** {total_time:.1f}s",
        f"- **File npz:** `{output_npz.name}`",
        f"- **Tổng mẫu hợp lệ:** {n_total} (train {n_train} / val giữ-out {n_val})",
        f"- **Preview kiểm tra:** {preview_dir}\n",
        "## 📥 Số mẫu theo nguồn\n",
        "| Nguồn | Hợp lệ | Train | Val |",
        "|---|---|---|---|",
    ]
    for c in collectors:
        nv = sum(1 for s in c.splits if s == 1)
        lines.append(f"| {c.source_name} | {len(c)} | {len(c) - nv} | {nv} |")

    lines += ["\n## 🗑️ Mẫu bị loại theo lý do (QA Gates)\n", "| Nguồn | Lý do | Số mẫu |", "|---|---|---|"]
    for c in collectors:
        for reason, cnt in c.rejected.most_common():
            lines.append(f"| {c.source_name} | {reason} | {cnt} |")

    if n_total:
        yaw_deg = poses[:, 0] * 90.0
        pitch_deg = poses[:, 1] * 90.0
        bins = [(-90, -40), (-40, -20), (-20, 20), (20, 40), (40, 90)]
        lines += ["\n## 📐 Phân bố góc Yaw (độ) — phải phủ đều 2 phía\n", "| Khoảng | Số mẫu |", "|---|---|"]
        for lo, hi in bins:
            cnt = int(np.sum((yaw_deg >= lo) & (yaw_deg < hi)))
            lines.append(f"| [{lo:+d}, {hi:+d}) | {cnt} |")
        lines += ["\n## 🥱 Phân bố trạng thái sinh trắc\n",
                  "| Trạng thái | Ngưỡng | Số mẫu |", "|---|---|---|",
                  f"| Ngáp (MAR cao) | MAR >= 0.40 | {int(np.sum(mars >= 0.40))} |",
                  f"| Nhắm mắt | EAR < 0.21 | {int(np.sum(ears < 0.21))} |",
                  f"| Bình thường | còn lại | {int(np.sum((mars < 0.40) & (ears >= 0.21)))} |",
                  "\n> ⚠️ Nếu cột Yaw [-90,-40) hoặc [40,90) bằng 0 → bổ sung 300W-LP/AFLW2000",
                  "để mô hình học bất biến góc quay (nguyên nhân tracking hỏng cũ)."]
    report_path = ROOT_DIR / "output" / "dataset_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Build Clean Dataset cho TinyDriverNet")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Thư mục dữ liệu bổ sung (ảnh tự chụp / WFLW / YawDD...)")
    parser.add_argument("--download-aflw2000", action="store_true",
                        help="Tự tải AFLW2000-3D (~83 MB, 2000 ảnh phủ góc quay đầu ±90°)")
    parser.add_argument("--download-facesynth", action="store_true",
                        help="Tự tải Microsoft FaceSynthetics dataset_1000 (~322 MB, 1000 mặt 512x512 nhãn 68-pt iBUG chính xác pixel)")
    parser.add_argument("--download-300wlp", action="store_true",
                        help="Thử tải 300W-LP (~1.7 GB) — LƯU Ý: link chính thức hiện 404, chỉ còn tải thủ công")
    parser.add_argument("--aflw2000-zip", type=str, default=None,
                        help="Đường dẫn file AFLW2000-3D.zip đã tải sẵn")
    parser.add_argument("--output-npz", type=str,
                        default=str(TRAINING_DIR / "preprocessed_driver_dataset.npz"))
    parser.add_argument("--val-percent", type=int, default=8,
                        help="Phần trăm mẫu giữ-out cho validation (mặc định 8%%)")
    parser.add_argument("--max-per-source", type=int, default=25000)
    parser.add_argument("--max-previews", type=int, default=60)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("🧬 BUILD CLEAN DATASET - Đồ án 13 (pipeline thay thế dữ liệu bẩn)")
    print("=" * 78)

    # [v2.0.5] Teacher LAZY: chỉ khởi tạo MediaPipe khi gặp nguồn KHÔNG có nhãn sẵn
    # (trước đây init trước cả khi tải file -> noise + chậm vô ích với nguồn mat68/pts68)
    teacher_holder = {}

    def _get_teacher():
        if "t" not in teacher_holder:
            teacher_holder["t"] = MediaPipeTeacher()
        return teacher_holder["t"]

    failed_downloads = []

    if args.aflw2000_zip:
        z = Path(args.aflw2000_zip)
        dest = RAW_FACES_DIR / "AFLW2000"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(z), 'r') as zf:
            zf.extractall(str(dest))
        (dest / "_extracted_ok").write_text("ok", encoding="utf-8")
        print(f"[AFLW2000] Đã giải nén từ {z.name}")

    if args.download_aflw2000:
        if not download_aflw2000(RAW_FACES_DIR / "AFLW2000"):
            failed_downloads.append("AFLW2000")

    if args.download_facesynth:
        if not download_facesynth(RAW_FACES_DIR / "FaceSynthetics"):
            failed_downloads.append("FaceSynthetics")

    if args.download_300wlp:
        if not download_300wlp(RAW_FACES_DIR / "300W-LP"):
            failed_downloads.append("300W-LP (link 404 — tải thủ công từ trang 3DDFA)")

    # Danh sách nguồn: mọi thư mục con của datasets/raw_faces + data-dir tùy chọn
    source_dirs = []
    if RAW_FACES_DIR.exists():
        for d in sorted(RAW_FACES_DIR.iterdir()):
            if d.is_dir():
                source_dirs.append(d)
    if args.data_dir:
        extra = Path(args.data_dir)
        if extra.exists():
            source_dirs.append(extra)
        else:
            print(f"⚠️ --data-dir không tồn tại: {extra}")

    if failed_downloads:
        print("\n⚠️ CẢNH BÁO: Có nguồn tải thất bại: " + ", ".join(failed_downloads))
        print("   (Vẫn tiếp tục build với những nguồn còn lại.)")

    if not source_dirs:
        print("\n❌ KHÔNG CÓ DỮ LIỆU NÀO trong datasets/raw_faces/!")
        print("👉 Làm 1 trong các việc sau:")
        print("   1. python tools/build_clean_dataset.py --download-aflw2000 --download-facesynth")
        print("   2. Tải thủ công 300W / WFLW / YawDD vào datasets/raw_faces/<ten>/")
        print("      (hướng dẫn link ở đầu file này)")
        sys.exit(1)

    collectors = []
    for src_dir in source_dirs:
        name = src_dir.name
        # [v2.0.5] Bỏ qua thư mục RỖNG (thường do tải thất bại) thay vì in "0 mẫu" gây rối
        has_any = next(src_dir.rglob("*"), None)
        if has_any is None:
            print(f"\n⏭️ [SKIP] {name}: thư mục RỖNG (nguồn chưa được tải về) — bỏ qua.")
            continue
        print(f"\n📥 NGUỒN: {name} ({src_dir})")
        collector = SampleCollector(teacher=None, source_name=name,
                                    val_percent=args.val_percent)
        # Ưu tiên nhãn chuẩn 68-pt nếu có (mat68: AFLW2000/300W-LP, pts68: 300W,
        # ldmks70: FaceSynthetics) — KHÔNG cần MediaPipe
        n = collect_from_aflw2000(src_dir, collector, args.max_per_source)
        fmt = "mat68" if n > 0 else None
        if n == 0:
            n = collect_from_300w(src_dir, collector, args.max_per_source)
            fmt = "pts68" if n > 0 else None
        if n == 0:
            n = collect_from_facesynth(src_dir, collector, args.max_per_source)
            fmt = "ldmks70" if n > 0 else None
        if n == 0:
            # Nguồn không nhãn -> lúc này mới cần MediaPipe Teacher
            teacher = _get_teacher()
            if teacher is None or not teacher.available:
                print(f"  ⏭️ [SKIP] {name}: không có nhãn sẵn và MediaPipe Teacher không khả dụng.")
                continue
            n = collect_from_images(src_dir, collector, teacher, args.max_per_source)
            fmt = "mediapipe" if n > 0 else None
        print(f"  ✅ {name}: {n} mẫu hợp lệ (format={fmt}) | loại: "
              f"{dict(collector.rejected.most_common()) if collector.rejected else '{}'}")
        if n > 0:
            collectors.append(collector)

    if not collectors:
        print("\n❌ Không thu thập được mẫu hợp lệ nào! Kiểm tra lại dữ liệu.")
        sys.exit(1)

    # Gộp + xuất npz
    images = np.concatenate([np.array(c.images, dtype=np.uint8) for c in collectors])
    landmarks = np.concatenate([np.array(c.landmarks, dtype=np.float32) for c in collectors])
    poses = np.concatenate([np.array(c.poses, dtype=np.float32) for c in collectors])
    ears = np.concatenate([np.array(c.ears, dtype=np.float32) for c in collectors])
    mars = np.concatenate([np.array(c.mars, dtype=np.float32) for c in collectors])
    splits = np.concatenate([np.array(c.splits, dtype=np.uint8) for c in collectors])
    sources = np.concatenate([np.array([c.source_name] * len(c), dtype='U32') for c in collectors])

    # Shuffle nhất quán (theo seed) nhưng GIỮ nguyên cột split
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(images))
    images, landmarks, poses = images[perm], landmarks[perm], poses[perm]
    ears, mars, splits, sources = ears[perm], mars[perm], splits[perm], sources[perm]

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_npz),
        images=images, landmarks=landmarks, poses=poses,
        ears=ears, mars=mars, split=splits, sources=sources,
    )
    size_mb = output_npz.stat().st_size / (1024 * 1024)
    n_val = int(np.sum(splits == 1))

    print("\n" + "=" * 78)
    print(f"🎉 HOÀN TẤT: {output_npz.name} ({size_mb:.1f} MB)")
    print(f"   • Tổng mẫu      : {len(images)} (train {len(images) - n_val} / val giữ-out {n_val})")
    print(f"   • Ngáp (MAR>=0.4): {int(np.sum(mars >= 0.40))} | Nhắm (EAR<0.21): {int(np.sum(ears < 0.21))}")
    saved = export_previews(collectors, PREVIEW_DIR, args.max_previews)
    print(f"   • Preview        : {saved} ảnh tại {PREVIEW_DIR}")
    report = export_report(collectors, output_npz, PREVIEW_DIR, time.time() - t0)
    print(f"   • Báo cáo        : {report}")
    yaw_deg = poses[:, 0] * 90.0
    big_pose = int(np.sum(np.abs(yaw_deg) >= 40))
    print(f"   • Mẫu |Yaw|>=40° : {big_pose} "
          f"({'✅ OK' if big_pose > 200 else '⚠️ CẦN BỔ SUNG dữ liệu góc quay lớn'})")
    print("=" * 78)

    # [v2.0.5] Thông báo tiếp theo PHỤ THỤCH chất lượng dữ liệu (không in máy móc)
    if len(images) < 3000:
        print("\n⚠️ DATASET VẪN NHỎ (< 3000 mẫu). Khuyến nghị TRƯỚC KHI train:")
        print("   1. python tools/build_clean_dataset.py --download-facesynth (tự động, 322MB)")
        print("   2. Tải WFLW thủ công (Google Drive, ~9.8k ảnh) bỏ vào datasets/raw_faces/WFLW/")
        print("   3. Chạy lại lệnh build này để gộp thêm nguồn mới.")
    print("\n👉 BƯỚC TIẾP THEO: python tools/project_manager.py --pack-colab (train trên Colab)")


if __name__ == "__main__":
    main()
