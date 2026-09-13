#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ðŸ§¬ BUILD CLEAN DATASET - Pipeline dá»¯ liá»‡u Sáº CH cho TinyDriverNet (Äá»“ Ã¡n 13)
=============================================================================
Thay tháº¿ hoÃ n toÃ n pipeline cÅ© (yawn_faces.zip + drowsiness Roboflow-YOLO)
vá»‘n chá»‰ cÃ³ ~400 máº«u tháº­t, nhiá»u áº£nh crop cáº­n cáº£nh thiáº¿u cáº±m/máº¯t vÃ  KHÃ”NG
cÃ³ nhÃ£n landmark chuáº©n â†’ nguyÃªn nhÃ¢n gá»‘c khiáº¿n mÃ´ hÃ¬nh há»c váº¹t.

NGUá»’N Dá»® LIá»†U ÄÆ¯á»¢C Há»– TRá»¢:
  1. AFLW2000-3D  (tá»± táº£i báº±ng --download-aflw2000, 2000 áº£nh, nhÃ£n 68-pt 3D,
                    phá»§ gÃ³c quay Ä‘áº§u yaw Â±90Â° â€” dataset chuáº©n cho Head Pose).
  2. YawDD        (video cabin tÃ i xáº¿ tháº­t, dÃ¡n nhÃ£n báº±ng MediaPipe Teacher).
  3. CEW          (Closed Eyes in the Wild, dÃ¡n nhÃ£n báº±ng MediaPipe Teacher).
  4. áº¢nh 68-pt (.pts) / áº£nh thÃ´ báº¥t ká»³ â†’ dÃ¡n nhÃ£n báº±ng MediaPipe Teacher.

âš ï¸ [v2.3.0] 300W-LP ÄÃƒ Bá»Š VÃ” HIá»†U HÃ“A (skip tá»± Ä‘á»™ng): file .mat cá»§a 300W-LP chá»‰ cÃ³
   `pt2d` á»Ÿ há»‡ toáº¡ Ä‘á»™ áº¢NH Gá»C frontal, KHÃ”NG khá»›p áº£nh render pose (cÃ¹ng 1 máº·t á»Ÿ 18 pose
   cÃ³ pt2d y há»‡t; sai tá»›i 45px á»Ÿ yaw 50Â°). DÃ¹ng nÃ³ sáº½ há»ng ~31% dá»¯ liá»‡u. Thay báº±ng
   AFLW2000_3D + YawDD cho gÃ³c quay lá»›n.

Bá»˜ GATE CHáº¤T LÆ¯á»¢NG (QA GATES) - máº«u nÃ o FAIL sáº½ bá»‹ loáº¡i vÃ  ghi lÃ½ do:
  G1. DÃ² Ä‘Æ°á»£c Ä‘áº§y Ä‘á»§ 22 Ä‘iá»ƒm má»‘c.
  G2. Sau crop canonical 96x96: má»i landmark náº±m trong [0.02, 0.98]
      (khÃ´ng bá»‹ cáº¯t máº¥t cáº±m/máº¯t á»Ÿ mÃ©p áº£nh â†’ chá»‘ng label báº©n P21).
  G3. Khoáº£ng cÃ¡ch 2 máº¯t trong crop >= 10 px (máº·t Ä‘á»§ lá»›n, khÃ´ng má» cháº¥m).
  G4. Äá»™ nÃ©t Laplacian variance >= 12 (loáº¡i áº£nh nhÃ²e/motion blur náº·ng).
  G5. GÃ³c quay: |yaw| <= 80Â°, |pitch| <= 70Â°, |roll| <= 55Â°.
  G6. Anti-duplicate: aHash 8x8, khoáº£ng cÃ¡ch Hamming <= 3 cá»§a 2 áº£nh báº¥t ká»³
      trong cÃ¹ng nguá»“n â†’ chá»‰ giá»¯ 1 (chá»‘ng trÃ¹ng láº·p lÃ m overfit).

TRAIN/VAL SPLIT NGHIÃŠM NGáº¶T (chá»‘ng leak):
  - Chia theo hash tÃªn file: md5(filename) % 100 < val_percent â†’ VAL.
  - XÃ¡c Ä‘á»‹nh NGAY LÃšC BUILD, lÆ°u cá»™t `split` vÃ o npz. Generator huáº¥n luyá»‡n
    CHá»ˆ dÃ¹ng máº«u split=0; validation/NME dÃ¹ng máº«u split=1 (giá»¯-out tháº­t).

OUTPUT:
  - training_tinyml/preprocessed_driver_dataset.npz
      images(N,96,96,1) landmarks(N,44) poses(N,3) ears(N,) mars(N,)
      split(N,) sources(N,)
  - output/preprocessed_preview/  (áº£nh kiá»ƒm tra trá»±c quan cÃ³ váº½ 22 Ä‘iá»ƒm)
  - output/dataset_report.md      (bÃ¡o cÃ¡o thá»‘ng kÃª + lÃ½ do loáº¡i máº«u)

HÆ¯á»šNG DáºªN Táº¢I Dá»® LIá»†U THá»¦ CÃ”NG (náº¿u auto-download lá»—i):
  - AFLW2000-3D : https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
                  â†’ giáº£i nÃ©n vÃ o datasets/raw_faces/AFLW2000/
  - 300W        : https://ibug.doc.ic.ac.uk/resources/facial-points/
                  â†’ bá» thÆ° má»¥c áº£nh + .pts vÃ o datasets/raw_faces/300W/
  - WFLW        : https://wywu.github.io/projects/LAB/WFLW.html
                  â†’ chá»‰ cáº§n thÆ° má»¥c áº£nh (bá» qua txt), MediaPipe sáº½ dÃ¡n nhÃ£n.
  - YawDD       : https://sites.google.com/site/yawddf/ (video trÃ­ch frame)
                  â†’ bá» frame vÃ o datasets/raw_faces/yawdd/
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
    """Tá»± phÃ¡t hiá»‡n thÆ° má»¥c gá»‘c dá»± Ã¡n (chá»‘ng bug v2.0.1: script náº±m á»Ÿ ROOT gÃ³i Colab
    thÃ¬ ROOT=CURRENT_DIR, náº±m trong tools/ cá»§a repo thÃ¬ ROOT=parent)."""
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
# THAM Sá» CHáº¤T LÆ¯á»¢NG (QA GATES) - Ä‘á»“ng bá»™ 1 nÆ¡i duy nháº¥t
# ============================================================================
LM_MARGIN = 0.01        # G2: landmark pháº£i náº±m trong [margin, 1-margin]
                        # [v2.0.1] Ná»›i tá»« 0.02: AFLW2000 nhiá»u áº£nh cáº­n cáº£nh,
                        # label váº«n chÃ­nh xÃ¡c tuyá»‡t Ä‘á»‘i nhá» padding cÃ´ng thá»©c canonical.
MIN_D_EYES_PX = 10.0    # G3: khoáº£ng cÃ¡ch 2 máº¯t tá»‘i thiá»ƒu (px trong Ã´ 96x96)
MIN_LAPLACIAN_VAR = 12.0  # G4: ngÆ°á»¡ng nÃ©t áº£nh
MAX_YAW_DEG = 80.0      # G5
MAX_PITCH_DEG = 70.0
MAX_ROLL_DEG = 55.0
DUP_HAMMING = 3         # G6: ngÆ°á»¡ng Hamming cá»§a aHash 8x8


# ============================================================================
# TIá»†N ÃCH CHUNG
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
    """MAR = (h_outer + h_inner) / (2*w) - Ä‘á»“ng bá»™ 100% wing_loss.py & firmware."""
    pts = pts_22_norm.reshape((22, 2))
    w_m = np.linalg.norm(pts[12] - pts[13])
    h_outer = np.linalg.norm(pts[14] - pts[15])
    h_inner = np.linalg.norm(pts[16] - pts[17])
    return (h_outer + h_inner) / (2.0 * max(w_m, 1e-4))


def ahash_64(gray_96):
    """aHash 8x8 cá»§a áº£nh crop 96x96 Ä‘á»ƒ chá»‘ng trÃ¹ng láº·p."""
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
    """Split quyáº¿t Ä‘á»‹nh bá»Ÿi hash tÃªn file -> á»•n Ä‘á»‹nh qua cÃ¡c láº§n cháº¡y láº¡i."""
    h = hashlib.md5(filename.encode('utf-8')).hexdigest()
    return 1 if (int(h[:8], 16) % 100 < val_percent) else 0


# ============================================================================
# Bá»˜ Äá»ŒC NHÃƒN 68 ÄIá»‚M CHUáº¨N iBUG (300W / AFLW2000)
# ============================================================================
def parse_pts_file(pts_path):
    """Äá»c file .pts chuáº©n 68 Ä‘iá»ƒm (300W)."""
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
    """Äá»c landmark 68 Ä‘iá»ƒm tá»« file .mat cá»§a AFLW2000 / 300W-LP (pt3d_68 hoáº·c pt2d)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        mat = loadmat(str(mat_path))
        arr = None
        for key in ('pt3d_68', 'pts_3d', 'landmarks', 'pt2d'):
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
    """Ãnh xáº¡ 68 Ä‘iá»ƒm iBUG -> 22 Ä‘iá»ƒm Ä‘á» tÃ i (theo config.IBUG_68_TO_22_INDICES)."""
    from config import IBUG_68_TO_22_INDICES
    if len(pts_68) != 68:
        return None
    return pts_68[IBUG_68_TO_22_INDICES].astype(np.float32)


# ============================================================================
# QA + THU THáº¬P 1 MáºªU
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
        self.rejected = Counter()      # lÃ½ do loáº¡i
        self.preview_pool = []         # (image, lms, ear, mar) Ä‘á»ƒ xuáº¥t preview

    def _reject(self, reason):
        self.rejected[reason] += 1

    def add_sample(self, img_bgr, pts_22_px, src_filename):
        """pts_22_px: (22,2) tá»a Ä‘á»™ pixel trong áº£nh gá»‘c. Tráº£ vá» True náº¿u nháº­n."""
        if img_bgr is None or pts_22_px is None or len(pts_22_px) != 22:
            self._reject("missing_landmarks")
            return False

        crop, norm_lms, meta = canonical_face_crop(img_bgr, np.asarray(pts_22_px, dtype=np.float32))
        if crop is None or norm_lms is None:
            self._reject("crop_failed")
            return False

        crop_gray = crop[:, :, 0]
        pts = norm_lms.reshape((22, 2))

        # G2: khÃ´ng landmark nÃ o bá»‹ cáº¯t mÃ©p
        if pts.min() < LM_MARGIN or pts.max() > (1.0 - LM_MARGIN):
            self._reject("landmark_clipped")
            return False

        # G3: máº·t Ä‘á»§ lá»›n
        eye_l = pts[LEFT_EYE_22].mean(axis=0)
        eye_r = pts[RIGHT_EYE_22].mean(axis=0)
        d_eyes_px = float(np.linalg.norm(eye_r - eye_l) * IMAGE_WIDTH)
        if d_eyes_px < MIN_D_EYES_PX:
            self._reject("face_too_small")
            return False

        # G4: áº£nh nhÃ²e
        lap_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        if lap_var < MIN_LAPLACIAN_VAR:
            self._reject("blurry")
            return False

        # G5: gÃ³c quay cá»±c trá»‹
        pose = estimate_pose_from_landmarks(pts)
        yaw_deg, pitch_deg, roll_deg = pose[0] * 90.0, pose[1] * 90.0, pose[2] * 90.0
        if (abs(yaw_deg) > MAX_YAW_DEG or abs(pitch_deg) > MAX_PITCH_DEG
                or abs(roll_deg) > MAX_ROLL_DEG):
            self._reject("extreme_pose")
            return False

        # G6: trÃ¹ng láº·p (chá»‰ so trong cÃ¹ng nguá»“n)
        h = ahash_64(crop_gray)
        # so sÃ¡nh nhanh vá»›i vÃ i pháº§n tá»­ gáº§n cuá»‘i (giáº£m O(n^2) vá»›i nguá»“n lá»›n)
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
# CÃC NGUá»’N Dá»® LIá»†U
# ============================================================================
AFLW2000_URLS = [
    "https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
    "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip",
]
AFLW2000_SIZE_MB = 83  # dung lÆ°á»£ng thá»±c táº¿ theo TFDS catalog (Download size 83.36 MiB)

# 300W-LP: 61.225 áº£nh tá»•ng há»£p gÃ³c quay yaw Â±90Â° â€” KHÃ”NG cÃ³ link táº£i trá»±c tiáº¿p á»•n Ä‘á»‹nh
# (cbsr tráº£ 404, Google Drive ID cÅ© Ä‘Ã£ cháº¿t) -> chá»‰ há»— trá»£ táº£i thá»§ cÃ´ng.
# PhÆ°Æ¡ng Ã¡n tá»± Ä‘á»™ng Ä‘Ã¡ng tin: Microsoft FaceSynthetics (Azure blob, khÃ´ng cáº§n auth).

LARGE_FILE_CHUNK = 1024 * 512
FACESYNTH_URLS = [
    "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_1000.zip",
]
FACESYNTH_SIZE_MB = 322


def _download_cbsr_zip(urls, zip_path, size_hint_mb):
    for url in urls:
        print(f"[CBSR] Äang táº£i {url} (~{size_hint_mb} MB)...")
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
            print(f"\n[CBSR] Táº£i tháº¥t báº¡i tá»« URL nÃ y ({e}). Thá»­ nguá»“n káº¿ tiáº¿p...")
            if zip_path.exists():
                zip_path.unlink()
    return False


def download_aflw2000(target_dir):
    """Táº£i AFLW2000-3D (2000 áº£nh, nhÃ£n 68-pt 3D, phá»§ yaw Â±90Â°, ~83 MB)."""
    zip_path = target_dir / "AFLW2000-3D.zip"
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[AFLW2000] ÄÃ£ táº£i vÃ  giáº£i nÃ©n tá»« trÆ°á»›c, bá» qua.")
        return True
    if not _download_cbsr_zip(AFLW2000_URLS, zip_path, AFLW2000_SIZE_MB):
        print("[AFLW2000] TOÃ€N Bá»˜ NGUá»’N Táº¢I THáº¤T Báº I!")
        print("  ðŸ‘‰ PhÆ°Æ¡ng Ã¡n thá»§ cÃ´ng:")
        print("     1. TrÃ¬nh duyá»‡t táº£i: https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip")
        print("        (hoáº·c Kaggle: kaggle.com/datasets/mohamedadlyi/aflw2000-3d)")
        print(f"     2. Giáº£i nÃ©n vÃ o: {target_dir} rá»“i cháº¡y láº¡i lá»‡nh nÃ y.")
        return False
    print("[AFLW2000] Äang giáº£i nÃ©n...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(target_dir))
    zip_path.unlink()
    done_marker.write_text("ok", encoding="utf-8")
    print("[AFLW2000] HOÃ€N Táº¤T!")
    return True


def download_facesynth(target_dir):
    """Táº£i Microsoft FaceSynthetics dataset_1000 (1000 máº·t 512x512, nhÃ£n 70 Ä‘iá»ƒm:
    68 Ä‘áº§u theo scheme iBUG + 2 Ä‘iá»ƒm Ä‘á»“ng tá»­). Link Azure blob trá»±c tiáº¿p, khÃ´ng auth.
    License: non-commercial research (phÃ¹ há»£p Ä‘á»“ Ã¡n)."""
    zip_path = target_dir / "dataset_1000.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[FaceSynthetics] ÄÃ£ táº£i vÃ  giáº£i nÃ©n tá»« trÆ°á»›c, bá» qua.")
        return True
    for url in FACESYNTH_URLS:
        print(f"[FaceSynthetics] Äang táº£i {url} (~{FACESYNTH_SIZE_MB} MB)...")
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
            print("\n[FaceSynthetics] Äang giáº£i nÃ©n (1000 áº£nh PNG 512x512)...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(target_dir))
            zip_path.unlink()
            done_marker.write_text("ok", encoding="utf-8")
            print("[FaceSynthetics] HOÃ€N Táº¤T!")
            return True
        except Exception as e:
            print(f"\n[FaceSynthetics] Táº£i tháº¥t báº¡i ({e}).")
            if zip_path.exists():
                zip_path.unlink()
    print("[FaceSynthetics] Táº¢I THáº¤T Báº I! Táº£i thá»§ cÃ´ng tá»« README microsoft/FaceSynthetics")
    print(f"  vÃ  giáº£i nÃ©n vÃ o: {target_dir}")
    return False


def download_300wlp(target_dir):
    """Thá»­ táº£i 300W-LP. LÆ¯U Ã: link cbsr hiá»‡n 404 (Google Drive ID cÅ© cÅ©ng Ä‘Ã£ cháº¿t)
    -> Ä‘a sá»‘ sáº½ tháº¥t báº¡i vÃ  hÆ°á»›ng dáº«n táº£i thá»§ cÃ´ng. PhÆ°Æ¡ng Ã¡n tá»± Ä‘á»™ng: --download-facesynth."""
    zip_path = target_dir / "300W-LP.zip"
    done_marker = target_dir / "_extracted_ok"
    if done_marker.exists():
        print("[300W-LP] ÄÃ£ táº£i vÃ  giáº£i nÃ©n tá»« trÆ°á»›c, bá» qua.")
        return True
    urls = [
        "https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP.zip",
        "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP.zip",
    ]
    if not _download_cbsr_zip(urls, zip_path, 1700):
        print("[300W-LP] Link chÃ­nh thá»©c hiá»‡n 404! Táº£i thá»§ cÃ´ng:")
        print("  ðŸ‘‰ Trang chá»§ 3DDFA: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm")
        print(f"     Giáº£i nÃ©n vÃ o: {target_dir} rá»“i cháº¡y láº¡i lá»‡nh nÃ y.")
        return False
    print("[300W-LP] Äang giáº£i nÃ©n (61k áº£nh, cÃ³ thá»ƒ máº¥t vÃ i phÃºt)...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(target_dir))
    zip_path.unlink()
    done_marker.write_text("ok", encoding="utf-8")
    print("[300W-LP] HOÃ€N Táº¤T!")
    return True


def collect_from_mat68(folder, collector, max_samples, shuffle=False):
    """AFLW2000-3D / 300W-LP: áº£nh .jpg + .mat (pt3d_68 hoáº·c pt2d)."""
    mats = sorted(folder.rglob("*.mat"))
    if not mats:
        return 0
    if shuffle:
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(mats))
        mats = [mats[i] for i in perm]
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
                print(f"  [{folder.name}] {n} máº«u há»£p lá»‡ / Ä‘Ã£ duyá»‡t {mat_idx + 1}...")
    return n

collect_from_aflw2000 = collect_from_mat68


def collect_from_aflw_teacher(folder, collector, teacher, max_samples=2000,
                              sanity_max_diff_px=15.0):
    """[v2.4.2 - FIX QUY ƯỚC NHÃN] AFLW2000: DÁN NHÃN LẠI bằng MediaPipe Teacher.

    Vì sao: nhãn native iBUG-68 của AFLW lệch MediaPipe trung bình ~6.2px (cả mắt/miệng/mũi).
    Trộn 2 quy ước -> model học giá trị trung bình -> output lệch MediaPipe, nặng nhất ở MIỆNG.
    Dán lại bằng MediaPipe để MỌI nguồn (AFLW/CEW/YawDD) cùng 1 quy ước 22 điểm.

    Nhãn native chỉ dùng SANITY CHECK: nếu MediaPipe lệch quá `sanity_max_diff_px` px so với
    nhãn native (dấu hiệu mis-detect, ảnh xoay/lạ) thì LOẠI.
    """
    if teacher is None or not getattr(teacher, 'available', False):
        print(f"  [SKIP] {folder.name}: không có MediaPipe Teacher.")
        return 0
    mats = sorted(folder.rglob("*.mat"))
    if not mats:
        return 0
    n = 0
    for mat_path in mats:
        if max_samples and n >= max_samples:
            break
        img_path = mat_path.with_suffix('.jpg')
        if not img_path.exists():
            img_path = mat_path.with_suffix('.png')
        if not img_path.exists():
            collector._reject("image_missing")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            collector._reject("unreadable")
            continue
        h_img, w_img = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pts_norm = teacher.extract_22_landmarks(rgb)
        if pts_norm is None:
            collector._reject("teacher_no_face")
            continue
        pts_px = pts_norm.copy()
        pts_px[:, 0] *= w_img
        pts_px[:, 1] *= h_img
        # Sanity check chống mis-detect (so với nhãn native nếu đọc được)
        pts68 = parse_mat_landmarks(mat_path)
        if pts68 is not None:
            native22 = pts68_to_22(pts68)
            d = float(np.linalg.norm(native22 - pts_px, axis=1).mean())
            if d > sanity_max_diff_px:
                collector._reject("teacher_native_mismatch")
                continue
        if collector.add_sample(img, pts_px, img_path.name):
            n += 1
            if n % 250 == 0:
                print(f"  [AFLW-MP] {n} mẫu hợp lệ...")
    return n


def collect_from_300w(folder, collector, max_samples):
    """300W / 300W-LP: áº£nh + file .pts 68 Ä‘iá»ƒm."""
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
                print(f"  [300W] {n} máº«u há»£p lá»‡...")
    return n


def collect_from_facesynth(folder, collector, max_samples):
    """FaceSynthetics: {id}.png + {id}_ldmks.txt (70 hÃ ng 'x y' pixel;
    68 hÃ ng Ä‘áº§u = scheme iBUG, 2 hÃ ng cuá»‘i = Ä‘á»“ng tá»­ -> bá»)."""
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
                print(f"  [FaceSynthetics] {n} máº«u há»£p lá»‡ / duyá»‡t {i+1}...")
    return n


def collect_from_cew(folder, collector, teacher, max_samples=3500):
    """CEW: ChuyÃªn sÃ¢u nháº¯m máº¯t (closed) vÃ  má»Ÿ máº¯t (open) ngÆ°á»i tháº­t."""
    if teacher is None or not getattr(teacher, 'available', False):
        print(f"  [SKIP] {folder.name}: khÃ´ng cÃ³ MediaPipe Teacher.")
        return 0
    closed_paths = sorted(folder.rglob("*closed*.*"))
    closed_paths = [p for p in closed_paths if p.suffix.lower() in ('.jpg', '.png', '.jpeg')]
    open_paths = sorted(folder.rglob("*open*.*"))
    open_paths = [p for p in open_paths if p.suffix.lower() in ('.jpg', '.png', '.jpeg')]

    rng = np.random.RandomState(42)
    if closed_paths:
        perm = rng.permutation(len(closed_paths))
        closed_paths = [closed_paths[i] for i in perm]
    if open_paths:
        perm = rng.permutation(len(open_paths))
        open_paths = [open_paths[i] for i in perm]

    n_closed_target = int(max_samples * 0.70)
    n_open_target = max_samples - n_closed_target

    n = 0
    # 1. Thu tháº­p áº£nh nháº¯m máº¯t
    for i, img_path in enumerate(closed_paths):
        if n >= n_closed_target:
            break
        img = cv2.imread(str(img_path))
        if img is None:
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
            if n % 250 == 0:
                print(f"  [CEW-Closed] {n}/{n_closed_target} máº«u nháº¯m máº¯t...")

    # 2. Thu tháº­p áº£nh má»Ÿ máº¯t
    n_open = 0
    for i, img_path in enumerate(open_paths):
        if n_open >= n_open_target:
            break
        img = cv2.imread(str(img_path))
        if img is None:
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
            n_open += 1
            if n_open % 250 == 0:
                print(f"  [CEW-Open] {n_open}/{n_open_target} máº«u má»Ÿ máº¯t...")
    return n


def collect_from_yawdd_videos(folder, collector, teacher, max_samples=3500):
    """YawDD: TrÃ­ch xuáº¥t frame cabin xe chuyÃªn sÃ¢u vá» ngÃ¡p hÃ¡ miá»‡ng & lÃ¡i xe tháº­t."""
    if teacher is None or not getattr(teacher, 'available', False):
        print(f"  [SKIP] {folder.name}: khÃ´ng cÃ³ MediaPipe Teacher.")
        return 0
    vids = sorted(folder.rglob("*.avi"))
    if not vids:
        return 0

    # PhÃ¢n loáº¡i video ngÃ¡p vÃ  video thÆ°á»ng
    yawn_vids = [v for v in vids if "yawn" in v.name.lower() or "dash" in str(v).lower()]
    other_vids = [v for v in vids if v not in yawn_vids]

    rng = np.random.RandomState(42)
    if yawn_vids:
        perm = rng.permutation(len(yawn_vids))
        yawn_vids = [yawn_vids[i] for i in perm]
    if other_vids:
        perm = rng.permutation(len(other_vids))
        other_vids = [other_vids[i] for i in perm]

    n_yawn_target = int(max_samples * 0.70)
    n_normal_target = max_samples - n_yawn_target

    n = 0
    n_yawns = 0
    # 1. TrÃ­ch xuáº¥t frame tá»« video ngÃ¡p
    print(f"  [YawDD] Báº¯t Ä‘áº§u quÃ©t {len(yawn_vids)} video cÃ³ hÃ nh vi ngÃ¡p...")
    for vid_idx, v_path in enumerate(yawn_vids):
        if n_yawns >= n_yawn_target:
            break
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            continue
        frame_idx = 0
        while cap.isOpened() and n_yawns < n_yawn_target:
            ret, frame = cap.read()
            if not ret:
                break
            # Láº¥y máº«u má»—i 6 frames
            if frame_idx % 6 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pts_norm = teacher.extract_22_landmarks(rgb)
                if pts_norm is not None:
                    h_img, w_img = frame.shape[:2]
                    pts_px = pts_norm.copy()
                    pts_px[:, 0] *= w_img
                    pts_px[:, 1] *= h_img
                    # Äo nhanh MAR
                    w_m = np.linalg.norm(pts_px[12] - pts_px[13]) + 1e-6
                    h_m = np.linalg.norm(pts_px[14] - pts_px[15]) + np.linalg.norm(pts_px[16] - pts_px[17])
                    mar = float(h_m / (2.0 * w_m))
                    # Æ¯u tiÃªn ngÃ¡p vÃ  nháº¯m máº¯t
                    if mar >= 0.38 or (frame_idx % 18 == 0):
                        if collector.add_sample(frame, pts_px, v_path.stem):
                            n += 1
                            if mar >= 0.40:
                                n_yawns += 1
                            if n % 150 == 0:
                                print(f"  [YawDD-Yawn] {n} frame há»£p lá»‡ ({n_yawns} ngÃ¡p MAR>=0.40) / video {vid_idx+1}...")
            frame_idx += 1
        cap.release()

    # 2. TrÃ­ch xuáº¥t frame lÃ¡i xe bÃ¬nh thÆ°á»ng
    n_norm = 0
    print(f"  [YawDD] Báº¯t Ä‘áº§u quÃ©t {len(other_vids)} video lÃ¡i xe cabin bÃ¬nh thÆ°á»ng...")
    for vid_idx, v_path in enumerate(other_vids):
        if n_norm >= n_normal_target or n >= max_samples:
            break
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            continue
        frame_idx = 0
        while cap.isOpened() and n_norm < n_normal_target and n < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            # Láº¥y máº«u má»—i 15 frames
            if frame_idx % 15 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pts_norm = teacher.extract_22_landmarks(rgb)
                if pts_norm is not None:
                    h_img, w_img = frame.shape[:2]
                    pts_px = pts_norm.copy()
                    pts_px[:, 0] *= w_img
                    pts_px[:, 1] *= h_img
                    if collector.add_sample(frame, pts_px, v_path.stem):
                        n += 1
                        n_norm += 1
                        if n_norm % 150 == 0:
                            print(f"  [YawDD-Normal] {n_norm}/{n_normal_target} frame bÃ¬nh thÆ°á»ng...")
            frame_idx += 1
        cap.release()

    return n


def collect_from_images(folder, collector, teacher, max_samples):
    """Má»i áº£nh khÃ´ng cÃ³ nhÃ£n landmark -> MediaPipe Teacher tá»± dÃ¡n nhÃ£n."""
    if teacher is None or not getattr(teacher, 'available', False):
        print(f"  [SKIP] {folder.name}: khÃ´ng cÃ³ MediaPipe Teacher Ä‘á»ƒ dÃ¡n nhÃ£n.")
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
                print(f"  [{folder.name}] {n} máº«u há»£p lá»‡ / duyá»‡t {i+1}...")
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
        "# ðŸ§¬ BÃ¡o cÃ¡o Build Clean Dataset (Äá»“ Ã¡n 13)\n",
        f"- **Thá»i gian build:** {total_time:.1f}s",
        f"- **File npz:** `{output_npz.name}`",
        f"- **Tá»•ng máº«u há»£p lá»‡:** {n_total} (train {n_train} / val giá»¯-out {n_val})",
        f"- **Preview kiá»ƒm tra:** {preview_dir}\n",
        "## ðŸ“¥ Sá»‘ máº«u theo nguá»“n\n",
        "| Nguá»“n | Há»£p lá»‡ | Train | Val |",
        "|---|---|---|---|",
    ]
    for c in collectors:
        nv = sum(1 for s in c.splits if s == 1)
        lines.append(f"| {c.source_name} | {len(c)} | {len(c) - nv} | {nv} |")

    lines += ["\n## ðŸ—‘ï¸ Máº«u bá»‹ loáº¡i theo lÃ½ do (QA Gates)\n", "| Nguá»“n | LÃ½ do | Sá»‘ máº«u |", "|---|---|---|"]
    for c in collectors:
        for reason, cnt in c.rejected.most_common():
            lines.append(f"| {c.source_name} | {reason} | {cnt} |")

    if n_total:
        yaw_deg = poses[:, 0] * 90.0
        pitch_deg = poses[:, 1] * 90.0
        bins = [(-90, -40), (-40, -20), (-20, 20), (20, 40), (40, 90)]
        lines += ["\n## ðŸ“ PhÃ¢n bá»‘ gÃ³c Yaw (Ä‘á»™) â€” pháº£i phá»§ Ä‘á»u 2 phÃ­a\n", "| Khoáº£ng | Sá»‘ máº«u |", "|---|---|"]
        for lo, hi in bins:
            cnt = int(np.sum((yaw_deg >= lo) & (yaw_deg < hi)))
            lines.append(f"| [{lo:+d}, {hi:+d}) | {cnt} |")
        lines += ["\n## ðŸ¥± PhÃ¢n bá»‘ tráº¡ng thÃ¡i sinh tráº¯c\n",
                  "| Tráº¡ng thÃ¡i | NgÆ°á»¡ng | Sá»‘ máº«u |", "|---|---|---|",
                  f"| NgÃ¡p (MAR cao) | MAR >= 0.40 | {int(np.sum(mars >= 0.40))} |",
                  f"| Nháº¯m máº¯t | EAR < 0.21 | {int(np.sum(ears < 0.21))} |",
                  f"| BÃ¬nh thÆ°á»ng | cÃ²n láº¡i | {int(np.sum((mars < 0.40) & (ears >= 0.21)))} |",
                  "\n> âš ï¸ Náº¿u cá»™t Yaw [-90,-40) hoáº·c [40,90) báº±ng 0 â†’ bá»• sung 300W-LP/AFLW2000",
                  "Ä‘á»ƒ mÃ´ hÃ¬nh há»c báº¥t biáº¿n gÃ³c quay (nguyÃªn nhÃ¢n tracking há»ng cÅ©)."]
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
                        help="ThÆ° má»¥c dá»¯ liá»‡u bá»• sung (áº£nh tá»± chá»¥p / WFLW / YawDD...)")
    parser.add_argument("--download-aflw2000", action="store_true",
                        help="Tá»± táº£i AFLW2000-3D (~83 MB, 2000 áº£nh phá»§ gÃ³c quay Ä‘áº§u Â±90Â°)")
    parser.add_argument("--download-facesynth", action="store_true",
                        help="Tá»± táº£i Microsoft FaceSynthetics dataset_1000 (~322 MB, 1000 máº·t 512x512 nhÃ£n 68-pt iBUG chÃ­nh xÃ¡c pixel)")
    parser.add_argument("--download-300wlp", action="store_true",
                        help="Thá»­ táº£i 300W-LP (~1.7 GB) â€” LÆ¯U Ã: link chÃ­nh thá»©c hiá»‡n 404, chá»‰ cÃ²n táº£i thá»§ cÃ´ng")
    parser.add_argument("--aflw2000-zip", type=str, default=None,
                        help="ÄÆ°á»ng dáº«n file AFLW2000-3D.zip Ä‘Ã£ táº£i sáºµn")
    parser.add_argument("--output-npz", type=str,
                        default=str(TRAINING_DIR / "preprocessed_driver_dataset.npz"))
    parser.add_argument("--val-percent", type=int, default=8,
                        help="Pháº§n trÄƒm máº«u giá»¯-out cho validation (máº·c Ä‘á»‹nh 8%%)")
    parser.add_argument("--max-per-source", type=int, default=25000)
    parser.add_argument("--max-previews", type=int, default=60)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("ðŸ§¬ BUILD CLEAN DATASET - Äá»“ Ã¡n 13 (pipeline thay tháº¿ dá»¯ liá»‡u báº©n)")
    print("=" * 78)

    # [v2.0.5] Teacher LAZY: chá»‰ khá»Ÿi táº¡o MediaPipe khi gáº·p nguá»“n KHÃ”NG cÃ³ nhÃ£n sáºµn
    # (trÆ°á»›c Ä‘Ã¢y init trÆ°á»›c cáº£ khi táº£i file -> noise + cháº­m vÃ´ Ã­ch vá»›i nguá»“n mat68/pts68)
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
        print(f"[AFLW2000] ÄÃ£ giáº£i nÃ©n tá»« {z.name}")

    if args.download_aflw2000:
        if not download_aflw2000(RAW_FACES_DIR / "AFLW2000"):
            failed_downloads.append("AFLW2000")

    if args.download_facesynth:
        if not download_facesynth(RAW_FACES_DIR / "FaceSynthetics"):
            failed_downloads.append("FaceSynthetics")

    if args.download_300wlp:
        if not download_300wlp(RAW_FACES_DIR / "300W-LP"):
            failed_downloads.append("300W-LP (link 404 â€” táº£i thá»§ cÃ´ng tá»« trang 3DDFA)")

    # Danh sÃ¡ch nguá»“n: má»i thÆ° má»¥c con cá»§a datasets/raw_faces + data-dir tÃ¹y chá»n
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
            print(f"âš ï¸ --data-dir khÃ´ng tá»“n táº¡i: {extra}")

    if failed_downloads:
        print("\nâš ï¸ Cáº¢NH BÃO: CÃ³ nguá»“n táº£i tháº¥t báº¡i: " + ", ".join(failed_downloads))
        print("   (Váº«n tiáº¿p tá»¥c build vá»›i nhá»¯ng nguá»“n cÃ²n láº¡i.)")

    if not source_dirs:
        print("\nâŒ KHÃ”NG CÃ“ Dá»® LIá»†U NÃ€O trong datasets/raw_faces/!")
        print("ðŸ‘‰ LÃ m 1 trong cÃ¡c viá»‡c sau:")
        print("   1. python tools/build_clean_dataset.py --download-aflw2000 --download-facesynth")
        print("   2. Táº£i thá»§ cÃ´ng 300W / WFLW / YawDD vÃ o datasets/raw_faces/<ten>/")
        print("      (hÆ°á»›ng dáº«n link á»Ÿ Ä‘áº§u file nÃ y)")
        sys.exit(1)

    collectors = []
    for src_dir in source_dirs:
        name = src_dir.name
        # [v2.0.5] Bá» qua thÆ° má»¥c Rá»–NG (thÆ°á»ng do táº£i tháº¥t báº¡i) thay vÃ¬ in "0 máº«u" gÃ¢y rá»‘i
        has_any = next(src_dir.rglob("*"), None)
        if has_any is None:
            print(f"\nâ­ï¸ [SKIP] {name}: thÆ° má»¥c Rá»–NG (nguá»“n chÆ°a Ä‘Æ°á»£c táº£i vá») â€” bá» qua.")
            continue
        print(f"\nðŸ“¥ NGUá»’N: {name} ({src_dir})")
        collector = SampleCollector(teacher=None, source_name=name,
                                    val_percent=args.val_percent)
        low_name = name.lower()

        # [v2.3.0 - FIX DATA NGHIÃŠM TRá»ŒNG] LOáº I Bá»Ž 300W-LP.
        # ÄÃ£ kiá»ƒm chá»©ng: file .mat cá»§a 300W-LP chá»‰ chá»©a `pt2d` á»Ÿ há»‡ toáº¡ Ä‘á»™ áº¢NH Gá»C (frontal),
        # KHÃ”NG khá»›p vá»›i áº£nh Ä‘Ã£ render pose. CÃ¹ng má»™t khuÃ´n máº·t á»Ÿ 18 pose khÃ¡c nhau cÃ³ pt2d
        # Y Há»†T NHAU; táº¡i yaw=50Â° sai lá»‡ch lÃªn tá»›i 45px so vá»›i MediaPipe. ÄÃ¢y lÃ  3500/11174
        # máº«u (31%) bá»‹ há»ng nhÃ£n -> dáº¡y sai mÃ´ hÃ¬nh. Thay báº±ng AFLW2000_3D + YawDD (ngÆ°á»i tháº­t,
        # cÃ³ gÃ³c quay lá»›n tá»›i Â±90Â° vÃ  nhÃ£n chuáº©n).
        if "300w" in low_name:
            print(f"  â­ï¸ [SKIP] {name}: nhÃ£n 300W-LP á»Ÿ há»‡ toáº¡ Ä‘á»™ frontal, KHÃ”NG khá»›p áº£nh render pose "
                  f"(sai tá»›i 45px á»Ÿ yaw 50Â°). Bá» Ä‘á»ƒ trÃ¡nh há»ng dá»¯ liá»‡u.")
            continue

        if "yawdd" in low_name:
            teacher = _get_teacher()
            n = collect_from_yawdd_videos(src_dir, collector, teacher, max_samples=3500)
            fmt = "yawdd_video"
        elif "cew" in low_name:
            teacher = _get_teacher()
            n = collect_from_cew(src_dir, collector, teacher, max_samples=3500)
            fmt = "cew_images"
        elif "aflw" in low_name:
            # [v2.4.2] Dán nhãn lại bằng MediaPipe để đồng bộ quy ước 22 điểm với CEW/YawDD.
            teacher = _get_teacher()
            n = collect_from_aflw_teacher(src_dir, collector, teacher, max_samples=2000)
            fmt = "mediapipe_aflw"
        else:
            n = collect_from_mat68(src_dir, collector, args.max_per_source)
            fmt = "mat68" if n > 0 else None
            if n == 0:
                n = collect_from_300w(src_dir, collector, args.max_per_source)
                fmt = "pts68" if n > 0 else None
            if n == 0:
                n = collect_from_facesynth(src_dir, collector, args.max_per_source)
                fmt = "ldmks70" if n > 0 else None
            if n == 0:
                teacher = _get_teacher()
                if teacher is None or not teacher.available:
                    print(f"  â­ï¸ [SKIP] {name}: khÃ´ng cÃ³ nhÃ£n sáºµn vÃ  MediaPipe Teacher khÃ´ng kháº£ dá»¥ng.")
                    continue
                n = collect_from_images(src_dir, collector, teacher, args.max_per_source)
                fmt = "mediapipe" if n > 0 else None
        print(f"  âœ… {name}: {n} máº«u há»£p lá»‡ (format={fmt}) | loáº¡i: "
              f"{dict(collector.rejected.most_common()) if collector.rejected else '{}'}")
        if n > 0:
            collectors.append(collector)

    if not collectors:
        print("\nâŒ KhÃ´ng thu tháº­p Ä‘Æ°á»£c máº«u há»£p lá»‡ nÃ o! Kiá»ƒm tra láº¡i dá»¯ liá»‡u.")
        sys.exit(1)

    # Gá»™p + xuáº¥t npz
    images = np.concatenate([np.array(c.images, dtype=np.uint8) for c in collectors])
    landmarks = np.concatenate([np.array(c.landmarks, dtype=np.float32) for c in collectors])
    poses = np.concatenate([np.array(c.poses, dtype=np.float32) for c in collectors])
    ears = np.concatenate([np.array(c.ears, dtype=np.float32) for c in collectors])
    mars = np.concatenate([np.array(c.mars, dtype=np.float32) for c in collectors])
    splits = np.concatenate([np.array(c.splits, dtype=np.uint8) for c in collectors])
    sources = np.concatenate([np.array([c.source_name] * len(c), dtype='U32') for c in collectors])

    # Shuffle nháº¥t quÃ¡n (theo seed) nhÆ°ng GIá»® nguyÃªn cá»™t split
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
    print(f"ðŸŽ‰ HOÃ€N Táº¤T: {output_npz.name} ({size_mb:.1f} MB)")
    print(f"   â€¢ Tá»•ng máº«u      : {len(images)} (train {len(images) - n_val} / val giá»¯-out {n_val})")
    print(f"   â€¢ NgÃ¡p (MAR>=0.4): {int(np.sum(mars >= 0.40))} | Nháº¯m (EAR<0.21): {int(np.sum(ears < 0.21))}")
    saved = export_previews(collectors, PREVIEW_DIR, args.max_previews)
    print(f"   â€¢ Preview        : {saved} áº£nh táº¡i {PREVIEW_DIR}")
    report = export_report(collectors, output_npz, PREVIEW_DIR, time.time() - t0)
    print(f"   â€¢ BÃ¡o cÃ¡o        : {report}")
    yaw_deg = poses[:, 0] * 90.0
    big_pose = int(np.sum(np.abs(yaw_deg) >= 40))
    print(f"   â€¢ Máº«u |Yaw|>=40Â° : {big_pose} "
          f"({'âœ… OK' if big_pose > 200 else 'âš ï¸ Cáº¦N Bá»” SUNG dá»¯ liá»‡u gÃ³c quay lá»›n'})")
    print("=" * 78)

    # [v2.0.5] ThÃ´ng bÃ¡o tiáº¿p theo PHá»¤ THá»¤CH cháº¥t lÆ°á»£ng dá»¯ liá»‡u (khÃ´ng in mÃ¡y mÃ³c)
    if len(images) < 3000:
        print("\nâš ï¸ DATASET VáºªN NHá»Ž (< 3000 máº«u). Khuyáº¿n nghá»‹ TRÆ¯á»šC KHI train:")
        print("   1. python tools/build_clean_dataset.py --download-facesynth (tá»± Ä‘á»™ng, 322MB)")
        print("   2. Táº£i WFLW thá»§ cÃ´ng (Google Drive, ~9.8k áº£nh) bá» vÃ o datasets/raw_faces/WFLW/")
        print("   3. Cháº¡y láº¡i lá»‡nh build nÃ y Ä‘á»ƒ gá»™p thÃªm nguá»“n má»›i.")
    print("\nðŸ‘‰ BÆ¯á»šC TIáº¾P THEO: python tools/project_manager.py --pack-colab (train trÃªn Colab)")


if __name__ == "__main__":
    main()
