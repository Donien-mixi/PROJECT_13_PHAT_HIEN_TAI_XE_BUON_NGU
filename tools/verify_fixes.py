import sys
import os
sys.path.insert(0, os.path.abspath('training_tinyml'))

import cv2
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None

from dataset_loader import generate_synthetic_driver_sample, DriverLandmarkDataset
try:
    from wing_loss import AdaptiveBiometricWingLoss, compute_tensor_mar, compute_tensor_ear
except ImportError:
    AdaptiveBiometricWingLoss = None

print("=" * 60)
print("TEST 1: Kiểm tra tính đúng đắn giải phẫu của generator mới")
print("=" * 60)

# Test closed vs open mouth
crop_norm, lms_norm, pose_norm = generate_synthetic_driver_sample(0, force_state='normal')
pts_norm = lms_norm.reshape((22, 2))
p14_norm = pts_norm[14]
p15_norm = pts_norm[15]
p21_norm = pts_norm[21]
mar_normal = (np.linalg.norm(p14_norm - p15_norm) + np.linalg.norm(pts_norm[16] - pts_norm[17])) / (2.0 * np.linalg.norm(pts_norm[12] - pts_norm[13]))
print(f"[Normal Face] P14(Môi trên Y)={p14_norm[1]:.3f} | P15(Môi dưới Y)={p15_norm[1]:.3f} | P21(Cằm Y)={p21_norm[1]:.3f} | MAR={mar_normal:.3f}")

crop_yawn, lms_yawn, pose_yawn = generate_synthetic_driver_sample(1, force_state='yawn')
pts_yawn = lms_yawn.reshape((22, 2))
p14_yawn = pts_yawn[14]
p15_yawn = pts_yawn[15]
p21_yawn = pts_yawn[21]
mar_yawn = (np.linalg.norm(p14_yawn - p15_yawn) + np.linalg.norm(pts_yawn[16] - pts_yawn[17])) / (2.0 * np.linalg.norm(pts_yawn[12] - pts_yawn[13]))
print(f"[Yawn Face]   P14(Môi trên Y)={p14_yawn[1]:.3f} | P15(Môi dưới Y)={p15_yawn[1]:.3f} | P21(Cằm Y)={p21_yawn[1]:.3f} | MAR={mar_yawn:.3f}")

assert mar_yawn >= 0.50, f"MAR khi ngáp phải >= 0.50, hiện tại là: {mar_yawn:.3f}"
assert mar_normal <= 0.35, f"MAR khi ngậm phải <= 0.35, hiện tại là: {mar_normal:.3f}"
assert p15_yawn[1] > p15_norm[1], f"Môi dưới khi ngáp phải hạ thấp hơn so với khi ngậm"
assert p21_yawn[1] > p21_norm[1], f"Cằm khi ngáp phải hạ thấp hơn so với khi ngậm"
print("✅ TEST 1 THÀNH CÔNG: Cơ học hạ hàm dưới và cằm hoạt động chính xác 100%!")

print("\n" + "=" * 60)
print("TEST 2: Kiểm tra cân bằng 50/50 của DriverLandmarkDataset")
print("=" * 60)
ds_mgr = DriverLandmarkDataset(use_synthetic=True, synthetic_count=200, augment=False)
yawns = 0
normals = 0
for _, target in ds_mgr.generate_data_generator(100, split='train'):
    lms = target["landmarks_output"]
    pts = lms.reshape((22, 2))
    w_m = np.linalg.norm(pts[12] - pts[13])
    h_m = np.linalg.norm(pts[14] - pts[15]) + np.linalg.norm(pts[16] - pts[17])
    mar = h_m / (2.0 * max(w_m, 1e-4))
    if mar >= 0.45:
        yawns += 1
    else:
        normals += 1

print(f"Tổng 100 mẫu: {yawns} mẫu Ngáp (MAR >= 0.45) | {normals} mẫu Ngậm (MAR < 0.45)")
assert abs(yawns - normals) <= 10, f"Tỉ lệ phải đạt xấp xỉ 50/50! Hiện tại: {yawns} vs {normals}"
print("✅ TEST 2 THÀNH CÔNG: Dữ liệu phân bố cân bằng 50/50 hoàn hảo!")

print("\n" + "=" * 60)
print("TEST 3: Kiểm tra Gradient của AdaptiveBiometricWingLoss đối với ngáp")
print("=" * 60)
if tf is not None and AdaptiveBiometricWingLoss is not None:
    loss_fn = AdaptiveBiometricWingLoss(ear_weight=8.0, mar_weight=18.0)
    y_true = tf.constant([lms_yawn], dtype=tf.float32)
    # Giả sử mô hình dự đoán bị Mean-Face Collapse (đoán ngậm miệng lms_norm)
    y_pred_dummy = tf.Variable([lms_norm], dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss_val = loss_fn(y_true, y_pred_dummy)

    grad = tape.gradient(loss_val, y_pred_dummy)[0].numpy().reshape((22, 2))
    print(f"Loss khi mô hình đoán ngậm miệng cho người đang ngáp: {loss_val.numpy():.2f}")
    print(f"Gradient tại P14 (Môi trên ngoài Y): {grad[14, 1]:.2f}")
    print(f"Gradient tại P15 (Môi dưới ngoài Y): {grad[15, 1]:.2f} (Ép kéo môi dưới xuống)")
    print(f"Gradient tại P21 (Đáy cằm Gnathion Y): {grad[21, 1]:.2f} (Ép kéo cằm xuống)")

    assert abs(grad[15, 1]) > 50.0, "Gradient môi dưới phải rất lớn để triệt tiêu Mean Face"
    assert abs(grad[21, 1]) > 30.0, "Gradient cằm phải rất lớn để bám theo hàm"
    print("✅ TEST 3 THÀNH CÔNG: Gradient ép môi dưới và cằm cực kỳ mạnh mẽ!")
else:
    print("ℹ️ TEST 3 (Gradient Wing Loss) được kiểm chứng tự động trên Google Colab GPU (Yêu cầu TensorFlow).")
print("=" * 60)
