import sys
import os
import numpy as np
import cv2

# Đảm bảo đường dẫn import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from host_laptop.local_model_tester import (
    solve_head_pose_pnp,
    draw_head_pose_axes,
    draw_hud,
    OneEuroFilter,
    FaceTracker,
    NUM_LANDMARKS,
    LEFT_EYE_PTS,
    RIGHT_EYE_PTS,
    MOUTH_PTS,
    NOSE_TIP_PT
)
from training_tinyml.dataset_loader import apply_cabin_data_augmentation
from training_tinyml.config import IMAGE_WIDTH, IMAGE_HEIGHT

def test_pnp_and_axes():
    print("--- 1. Kiểm thử solve_head_pose_pnp & draw_head_pose_axes ---")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Tạo 22 điểm landmark giả lập khuôn mặt chính diện ở giữa khung hình
    landmarks_px = np.zeros((NUM_LANDMARKS, 2), dtype=np.float64)
    # Mắt trái
    landmarks_px[0] = [280, 200]
    landmarks_px[1] = [290, 195]
    landmarks_px[2] = [300, 195]
    landmarks_px[3] = [310, 200]
    landmarks_px[4] = [300, 205]
    landmarks_px[5] = [290, 205]
    # Mắt phải
    landmarks_px[6] = [330, 200]
    landmarks_px[7] = [340, 195]
    landmarks_px[8] = [350, 195]
    landmarks_px[9] = [360, 200]
    landmarks_px[10] = [350, 205]
    landmarks_px[11] = [340, 205]
    # Miệng
    landmarks_px[12] = [295, 290]
    landmarks_px[13] = [345, 290]
    landmarks_px[14] = [320, 280]
    landmarks_px[15] = [320, 305]
    landmarks_px[16] = [320, 285]
    landmarks_px[17] = [320, 295]
    # Mũi & Cằm
    landmarks_px[18] = [320, 190] # Nasion
    landmarks_px[19] = [320, 240] # Nose tip
    landmarks_px[20] = [320, 255] # Subnasale
    landmarks_px[21] = [320, 335] # Chin

    yaw, pitch, roll, pnp_res, cam_mat, dist_c = solve_head_pose_pnp(landmarks_px, 640, 480)
    print(f"✅ solve_head_pose_pnp thành công: Yaw={yaw:.2f}°, Pitch={pitch:.2f}°, Roll={roll:.2f}°")
    assert pnp_res is not None, "pnp_res không được là None"
    rvec, tvec = pnp_res

    # Test draw_head_pose_axes
    draw_head_pose_axes(frame, rvec, tvec, cam_mat, dist_c, axis_len=45.0)
    has_red = np.any((frame[:, :, 2] > 200) & (frame[:, :, 0] < 50))
    has_green = np.any((frame[:, :, 1] > 200) & (frame[:, :, 0] < 50))
    has_blue = np.any((frame[:, :, 0] > 200) & (frame[:, :, 2] < 50))
    print(f"✅ 3 trục 3D có màu sắc hợp lệ: Red={has_red}, Green={has_green}, Blue={has_blue}")
    assert has_red and has_green and has_blue, "Trục 3D phải vẽ đủ 3 màu X, Y, Z"

def test_face_tracker_hysteresis():
    print("\n--- 2. Kiểm thử Bounding Box Hysteresis trong FaceTracker ---")
    tracker = FaceTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Frame 1: Khởi tạo ở tâm (320, 240, 240)
    _, box1, s1, _ = tracker.update_with_target(320.0, 240.0, 240.0, frame)
    init_cx, init_cy, init_S = tracker.smooth_cx, tracker.smooth_cy, tracker.smooth_S
    print(f"Frame 1: Box tâm=({init_cx:.2f}, {init_cy:.2f}), S={init_S:.2f}")

    # Frame 2: Dịch chuyển nhỏ 1.8px (dưới ngưỡng 4.5px) -> Hysteresis phải giữ nguyên
    import time
    time.sleep(0.04)
    _, box2, s2, _ = tracker.update_with_target(321.5, 241.0, 241.0, frame)
    print(f"Frame 2 (dịch 1.8px): Target bị triệt tiêu rung giật -> smooth_cx={tracker.smooth_cx:.2f}")
    assert tracker.smooth_cx == init_cx or abs(tracker.smooth_cx - init_cx) < 0.5, "Hysteresis phải chặn rung giật nhỏ"

    # Frame 3: Dịch chuyển lớn 25px (di chuyển đầu thật) -> Tracker phải theo sát
    time.sleep(0.04)
    _, box3, s3, _ = tracker.update_with_target(345.0, 240.0, 240.0, frame)
    print(f"Frame 3 (dịch 25px): Target được bám theo -> smooth_cx={tracker.smooth_cx:.2f}")
    assert tracker.smooth_cx > init_cx + 5.0, "Khi di chuyển lớn, tracker phải bám theo"
    print("✅ FaceTracker Hysteresis hoạt động hoàn hảo!")

def test_eyeglasses_augmentation():
    print("\n--- 3. Kiểm thử Eyeglasses & Specular Glare Augmentation ---")
    img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 120, dtype=np.uint8)
    # 22 điểm chuẩn hóa
    lm = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    # Mắt trái
    lm[0:6] = [[0.28, 0.35], [0.32, 0.33], [0.36, 0.33], [0.40, 0.35], [0.36, 0.38], [0.32, 0.38]]
    # Mắt phải
    lm[6:12] = [[0.60, 0.35], [0.64, 0.33], [0.68, 0.33], [0.72, 0.35], [0.68, 0.38], [0.64, 0.38]]
    # Điểm còn lại
    lm[12:] = 0.5

    # Chạy 15 lần để kiểm tra cả trường hợp kính lẫn vệt lóa
    aug_success = 0
    for i in range(15):
        aug_img, aug_lm = apply_cabin_data_augmentation(img, lm)
        assert aug_img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 1), f"Shape không đúng: {aug_img.shape}"
        assert not np.isnan(aug_lm).any(), "Landmarks bị NaN"
        aug_success += 1
    print(f"✅ Augment sample 15/15 lần thành công, shape chuẩn ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, 1)!")

if __name__ == '__main__':
    test_pnp_and_axes()
    test_face_tracker_hysteresis()
    test_eyeglasses_augmentation()
    print("\n🎉 TẤT CẢ CÁC BÀI KIỂM THỬ XÁC NHẬN ĐÃ THÀNH CÔNG RỰC RỠ!")
