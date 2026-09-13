"""
Tool thu dữ liệu CAMERA THẬT để model hết lệch miền (domain gap) trên webcam/OV5640.
=====================================================================================
- Dùng ĐÚNG MediaPipe 22 điểm (y như Teacher lúc train) + ĐÚNG canonical crop
  (`isomorphic_transform.canonical_face_crop`) => nhãn cùng quy ước 100% với dataset.
- Lưu ra: training_tinyml/live_landmarks.npz
    images(N,96,96,1) uint8 | landmarks(N,44) float32 | poses(N,3) float32

Cách dùng:
    conda activate projet_13
    python tools/collect_live_landmarks.py --cam 0 --target 600

Phím:  [SPACE] tạm dừng/tiếp tục thu | [S] lưu | [Q]/[ESC] lưu & thoát
Gợi ý kịch bản (làm đủ để đa dạng): mở mắt thẳng -> nói chuyện -> nhắm/mở mắt ->
ngáp to -> quay trái/phải/lên/xuống -> lại gần/ra xa. (Ánh sáng khác nhau càng tốt.)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training_tinyml"))
sys.path.insert(0, str(ROOT / "host_laptop"))

from local_model_tester import MediaPipeLandmarkExtractor          # noqa: E402
from isomorphic_transform import canonical_face_crop                # noqa: E402
from distillation import estimate_pose_from_landmarks               # noqa: E402


def save_npz(out_path, images, landmarks, poses):
    if len(images) == 0:
        print("⚠️ Không có mẫu nào để lưu.")
        return
    images_arr = np.stack(images).astype(np.uint8)
    lms_arr = np.stack(landmarks).astype(np.float32)
    poses_arr = np.stack(poses).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, images=images_arr, landmarks=lms_arr, poses=poses_arr)
    print(f"💾 Đã lưu {len(images)} mẫu -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Thu dữ liệu camera thật (MediaPipe labels) cho domain adaptation")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "training_tinyml" / "live_landmarks.npz"))
    ap.add_argument("--target", type=int, default=600, help="Số mẫu mục tiêu")
    ap.add_argument("--append", action="store_true",
                    help="Nối thêm vào file có sẵn (dùng để thu NHIỀU NGƯỜI khác nhau cho đa dạng)")
    ap.add_argument("--min-interval", type=float, default=0.06, help="Giãn cách tối thiểu giữa 2 mẫu (giây)")
    ap.add_argument("--dedup-thresh", type=float, default=0.010, help="Bỏ frame quá giống mẫu vừa lưu (max |Δ| landmark)")
    args = ap.parse_args()

    print("=" * 70)
    print("🎥 THU DỮ LIỆU CAMERA THẬT (MediaPipe 22 điểm + canonical crop)")
    print("=" * 70)
    extractor = MediaPipeLandmarkExtractor()
    if not getattr(extractor, "is_loaded", False):
        print("❌ Không nạp được MediaPipe. Kiểm tra môi trường projet_13.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"❌ Không mở được webcam index={args.cam}")
        sys.exit(1)

    images, landmarks, poses = [], [], []
    if args.append and os.path.exists(args.out):
        prev = np.load(args.out)
        images = list(prev['images'])
        landmarks = list(prev['landmarks'])
        poses = list(prev['poses']) if 'poses' in prev else [np.zeros(3, dtype=np.float32)] * len(images)
        print(f"➕ Nối tiếp: đã có {len(images)} mẫu từ {args.out}")
    session_start = len(images)
    last_saved_lm = None
    capturing = True
    last_time = 0.0
    fps_t0, fps_n, fps = time.time(), 0, 0.0

    win = "Thu du lieu (SPACE=pause | S=luu | Q=thoat)"
    print("Hướng dẫn: SPACE tạm dừng | S lưu | Q thoát. Làm đa dạng biểu cảm/ánh sáng.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        pts_px, _ = extractor.extract(frame)
        crop_vis = None
        if pts_px is not None:
            crop, norm_lms, _ = canonical_face_crop(frame, pts_px, jitter=False)
            flat = norm_lms.flatten().astype(np.float32)

            now = time.time()
            is_new = True
            if last_saved_lm is not None:
                is_new = float(np.max(np.abs(flat - last_saved_lm))) >= args.dedup_thresh
            enough_gap = (now - last_time) >= args.min_interval

            if capturing and is_new and enough_gap and (len(images) - session_start) < args.target:
                images.append(crop.copy())
                landmarks.append(flat)
                poses.append(estimate_pose_from_landmarks(norm_lms).astype(np.float32))
                last_saved_lm = flat
                last_time = now

            # preview crop + landmark
            crop_vis = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
            crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
            for (x, y) in (norm_lms * 240.0):
                cv2.circle(crop_vis, (int(x), int(y)), 2, (0, 230, 255), -1)

        # HUD
        fps_n += 1
        if fps_n >= 15:
            fps = fps_n / max(1e-3, time.time() - fps_t0)
            fps_t0, fps_n = time.time(), 0
        hud = cv2.resize(frame, (480, 360))
        cv2.putText(hud, f"Phien nay: {len(images)-session_start}/{args.target}  tong {len(images)}  {'ON' if capturing else 'PAUSE'}  {fps:.1f}FPS",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if capturing else (0, 180, 255), 2)
        if pts_px is None:
            cv2.putText(hud, "KHONG THAY MAT", (10, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if crop_vis is not None:
            hud[110:350, 240:480] = crop_vis
        cv2.imshow(win, hud)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if key == ord(' '):
            capturing = not capturing
        if key in (ord('s'), ord('S')):
            save_npz(args.out, images, landmarks, poses)

        if (len(images) - session_start) >= args.target:
            print(f"✅ Phiên này đã đủ {args.target} mẫu (tổng {len(images)}).")
            break

    cap.release()
    cv2.destroyAllWindows()
    save_npz(args.out, images, landmarks, poses)


if __name__ == "__main__":
    main()
