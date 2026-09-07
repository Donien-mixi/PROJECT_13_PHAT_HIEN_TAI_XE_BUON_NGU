#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 SCRIPT HUẤN LUYỆN TỰ ĐỘNG TINYDRIVER PFLD-EDGE TRÊN GOOGLE COLAB
=============================================================================
Đồ án 13: Hệ thống phát hiện tài xế ngủ gật & mất tập trung (Edge AI ESP32-S3)

Phiên bản nâng cấp PFLD-Edge (Chống sụp đổ tọa độ / Mean Face Collapse):
  1. Kiến trúc Inverted Residual (MBConv) với Depthwise Separable Convolutions.
  2. Multi-Scale Spatial Fusion (Stage 3 + Stage 4) bảo toàn viền mí mắt và môi.
  3. Auxiliary 3D Pose Head: Ép mạng học góc quay đầu 3D (tự động cắt bỏ khi xuất TFLite).
  4. Adaptive Biometric Wing Loss kết hợp Differentiable Geometric EAR Loss (mí mắt)
     và MAR Loss (ngáp há miệng).
  5. Bounding Box Translation & Scale Jitter: Triệt tiêu 100% hiện tượng học vẹt tọa độ.
  6. Lượng tử hóa Full-Integer INT8 tương thích 100% với ESP32-S3 N16R8 (esp-nn SIMD).
  7. Tự động đóng gói và gửi lệnh tải tinydriver_esp32_package.zip.
=============================================================================
"""

import sys
import os
import time
import zipfile
import shutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Đảm bảo đường dẫn module
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
sub_dir = CURRENT_DIR / "training_tinyml"
if sub_dir.exists() and str(sub_dir) not in sys.path:
    sys.path.insert(0, str(sub_dir))

# Cài đặt mediapipe tự động nếu đang trên Google Colab để làm mô hình Thầy
try:
    import mediapipe
except ImportError:
    try:
        import google.colab
        print("⚡ [Colab] Đang cài đặt thư viện MediaPipe để kích hoạt Mô hình Thầy (Teacher)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe"], check=False)
        import importlib, site
        site.main()
        importlib.invalidate_caches()
    except ImportError:
        pass

# Nạp các module trong gói
try:
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, EPOCHS,
        LEARNING_RATE, WING_W, WING_EPSILON, NUM_LANDMARKS
    )
    from tinydriver_net import build_tinydriver_net
    from wing_loss import AdaptiveBiometricWingLoss
    from dataset_loader import (
        DriverLandmarkDataset, generate_synthetic_driver_sample,
        download_drowsiness_benchmark_dataset
    )
    from distillation import MediaPipeTeacher, PFLDMultiTaskModel
    from export_tflite import convert_model_to_c_array
except ImportError as e:
    import traceback
    print(f"❌ [LỖI IMPORT]: {e}")
    traceback.print_exc()
    sys.exit(1)


g_dataset_mgr = None


def representative_dataset_gen(num_samples=300):
    """Tạo tập mẫu hiệu chuẩn số nguyên INT8 với đa dạng góc quay và ánh sáng."""
    global g_dataset_mgr
    if g_dataset_mgr is not None:
        count = 0
        for img_norm, _ in g_dataset_mgr.generate_data_generator(num_samples, split='train'):
            yield [np.expand_dims(img_norm, axis=0)]
            count += 1
            if count >= num_samples:
                break
    else:
        for i in range(num_samples):
            img, _, _ = generate_synthetic_driver_sample(i, apply_aug=True)
            img_norm = (img.astype(np.float32) - 128.0) / 128.0
            if len(img_norm.shape) == 2:
                img_norm = np.expand_dims(img_norm, axis=-1)
            yield [np.expand_dims(img_norm, axis=0)]


def compute_nme(y_true, y_pred):
    """Tính sai số Normalized Mean Error (NME) chuẩn quốc tế."""
    pts_true = tf.reshape(y_true, [-1, NUM_LANDMARKS, 2]) * float(IMAGE_WIDTH)
    pts_pred = tf.reshape(y_pred, [-1, NUM_LANDMARKS, 2]) * float(IMAGE_WIDTH)
    interocular_dist = tf.norm(pts_true[:, 0, :] - pts_true[:, 9, :], axis=-1)
    interocular_dist = tf.maximum(interocular_dist, 1e-4)
    point_errors = tf.norm(pts_true - pts_pred, axis=-1)
    mean_point_error = tf.reduce_mean(point_errors, axis=-1)
    return tf.reduce_mean(mean_point_error / interocular_dist)


def main():
    global g_dataset_mgr

    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TINYDRIVER PFLD-EDGE (PHIÊN BẢN CẢI TIẾN)")
    print("   Kiến trúc: MobileNetV2 MBConv + Multi-Scale Fusion + Auxiliary 3D Pose Head")
    print("   Hàm mất mát: Adaptive Biometric Wing Loss + Geometric EAR/MAR Constraint Loss")
    print("   Tăng cường: Bounding Box Translation & Scale Jitter (Triệt tiêu Mean Face)")
    print("   Mục tiêu:   Full-Integer INT8 cho ESP32-S3 N16R8 (<250 KB)")
    print("=" * 75)

    # 1. Kiểm tra phần cứng GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Đã kích hoạt phần cứng GPU: {gpus[0].name} (Khuyên dùng Tesla T4)")
    else:
        print("⚠️ Đang chạy trên CPU (Tốc độ sẽ chậm hơn so với GPU)")

    work_dir = Path("./colab_output").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2. Khởi tạo Mô hình Thầy MediaPipe
    print("\n[Bước 1/5] Khởi tạo Mô hình Thầy MediaPipe Face Mesh...")
    teacher = MediaPipeTeacher()
    if teacher.available:
        print("  ✓ Mô hình Thầy MediaPipe Face Mesh đã sẵn sàng dán nhãn sinh học!")
    else:
        print("  ✓ Chế độ: Huấn luyện độc lập tối ưu sinh học (Autonomous Biometric Training)")

    # 3. Chuẩn bị dữ liệu Full Mặt thực tế & Tiền xử lý
    print("\n[Bước 2/5] Chuẩn bị dữ liệu Full Mặt (Preprocessed Real Dataset) + In-Cabin Augmentation...")
    preprocessed_npz_candidates = [
        CURRENT_DIR / "preprocessed_driver_dataset.npz",
        CURRENT_DIR / "training_tinyml" / "preprocessed_driver_dataset.npz",
        Path("./preprocessed_driver_dataset.npz").resolve(),
        Path("./training_tinyml/preprocessed_driver_dataset.npz").resolve(),
    ]
    preprocessed_file = None
    for cand in preprocessed_npz_candidates:
        if cand.exists():
            preprocessed_file = cand
            break

    if preprocessed_file:
        print(f"  ✓ Đã phát hiện tập dữ liệu tiền xử lý chuẩn: {preprocessed_file.name}")
        dataset_mgr = DriverLandmarkDataset(
            cache_path=str(preprocessed_file),
            synthetic_count=5000,
            augment=True,
            teacher=teacher
        )
    else:
        print("  ℹ️ Không thấy preprocessed_driver_dataset.npz đóng gói sẵn, tiến hành tải & giải nén...")
        dataset_dir = Path("./datasets/drowsiness_benchmark").resolve()
        download_drowsiness_benchmark_dataset(str(dataset_dir))
        cache_path = work_dir / "drowsiness_cache.npz"
        dataset_mgr = DriverLandmarkDataset(
            data_dir=str(dataset_dir),
            cache_path=str(cache_path),
            synthetic_count=5000,
            augment=True,
            teacher=teacher
        )

    g_dataset_mgr = dataset_mgr

    train_ds, val_ds, n_train, n_val = dataset_mgr.get_tf_dataset(batch_size=BATCH_SIZE)
    print(f"  ✓ Mẫu huấn luyện (Augmented): {n_train} | Mẫu kiểm chuẩn: {n_val}")
    print("  ✓ Kích hoạt Bounding Box Translation Jitter: triệt tiêu học vẹt tọa độ cố định.")

    # 4. Xây dựng mô hình PFLD-Edge Multi-Task
    print("\n[Bước 3/5] Xây dựng kiến trúc PFLD-Edge với Auxiliary 3D Pose Head...")
    full_train_model = build_tinydriver_net(include_pose_head=True)
    print(f"  ✓ Kiến trúc PFLD-Edge: {full_train_model.count_params():,} tham số")
    print("  ✓ Tích hợp Inverted Residual Blocks (MBConv) + Nhánh ước lượng góc đầu 3D")

    # 5. Thiết lập Optimizer & Multi-Task Loss
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * (n_train // BATCH_SIZE + 1),
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    landmark_loss_fn = AdaptiveBiometricWingLoss(
        w=WING_W, epsilon=WING_EPSILON,
        image_scale=float(IMAGE_WIDTH),
        ear_weight=15.0, mar_weight=20.0
    )

    multi_task_model = PFLDMultiTaskModel(
        full_model=full_train_model,
        landmark_loss_fn=landmark_loss_fn,
        pose_weight=1.5
    )
    multi_task_model.compile(optimizer=optimizer)
    print("  ✓ Hàm mất mát: Adaptive Biometric Wing Loss (EAR x15.0, MAR x20.0, Pose Regularizer x1.5)")

    # 6. Huấn luyện mô hình
    print(f"\n[Bước 4/5] Bắt đầu huấn luyện qua {EPOCHS} epochs...")
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_weights_path = work_dir / "best_weights.weights.h5"

    steps_per_epoch = n_train // BATCH_SIZE
    val_steps = max(n_val // BATCH_SIZE, 1)

    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        # Train loop
        train_losses = []
        train_lm_losses = []
        for step, batch in enumerate(train_ds.take(steps_per_epoch)):
            res = multi_task_model.train_step(batch)
            train_losses.append(float(res["loss"]))
            train_lm_losses.append(float(res["lm_loss"]))

        mean_train_loss = np.mean(train_losses)
        mean_train_lm = np.mean(train_lm_losses)

        # Validation loop
        val_losses = []
        val_lm_losses = []
        for batch in val_ds.take(val_steps):
            res = multi_task_model.test_step(batch)
            val_losses.append(float(res["loss"]))
            val_lm_losses.append(float(res["lm_loss"]))

        mean_val_loss = np.mean(val_losses)
        mean_val_lm = np.mean(val_lm_losses)

        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)
        dur = time.time() - epoch_start

        star = " "
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            full_train_model.save_weights(str(best_weights_path))
            star = " ⭐ (Best)"

        if (epoch + 1) % 2 == 0 or epoch == 0 or star.strip():
            print(f"Epoch {epoch+1:2d}/{EPOCHS} [{dur:.1f}s] - Train Loss: {mean_train_loss:7.2f} (LM: {mean_train_lm:7.2f}) | Val Loss: {mean_val_loss:7.2f} (LM: {mean_val_lm:7.2f}){star}")

    train_duration = time.time() - t0
    print(f"\n✅ Huấn luyện hoàn tất sau: {train_duration:.1f} giây! Best Val Loss: {best_val_loss:.2f}")

    # Nạp lại trọng số tốt nhất
    if best_weights_path.exists():
        full_train_model.load_weights(str(best_weights_path))
        print("  ✓ Đã nạp lại trọng số tốt nhất (Best Weights).")

    # Vẽ biểu đồ Loss
    plot_path = work_dir / "training_loss.png"
    plt.figure(figsize=(9, 3.8))
    plt.plot(train_loss_history, label="Train PFLD-Edge Loss", color="#1E88E5", lw=2)
    plt.plot(val_loss_history, label="Val PFLD-Edge Loss", color="#D81B60", lw=2)
    plt.title("Tiến Trình Huấn Luyện TinyDriver PFLD-Edge (AWing + Geometric EAR/MAR)")
    plt.xlabel("Epoch")
    plt.ylabel("Multi-Task Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Đã lưu biểu đồ tiến trình: {plot_path.name}")

    # 7. Trích xuất mô hình Landmark-Only và Lượng tử hóa Full INT8
    print("\n[Bước 5/5] Cắt bỏ Auxiliary Head và Lượng tử hóa Full-Integer INT8...")
    deploy_input = full_train_model.input
    deploy_output = full_train_model.get_layer("landmarks_output").output
    deploy_model = tf.keras.Model(inputs=deploy_input, outputs=deploy_output, name="TinyDriver_PFLD_Deploy")
    print(f"  ✓ Mô hình xuất xưởng (Landmark-Only): {deploy_model.count_params():,} tham số")

    converter = tf.lite.TFLiteConverter.from_keras_model(deploy_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    tflite_path = work_dir / "tinydriver_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print(f"  ✓ Kích thước mô hình INT8: {len(tflite_quant_model):,} bytes ({size_kb:.1f} KB)")

    # Đọc tham số lượng tử hóa
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    in_scale, in_zp = in_details["quantization"]
    out_scale, out_zp = out_details["quantization"]

    # Xuất file C Header (alignas 16 bytes cho esp-nn SIMD)
    header_path = work_dir / "tinydriver_model_data.h"
    convert_model_to_c_array(
        tflite_quant_model,
        str(header_path),
        in_scale, in_zp, out_scale, out_zp
    )
    print(f"  ✓ Đã tạo file C Header: {header_path.name} (16-byte aligned)")

    # 8. Đóng gói ZIP xuất xưởng
    zip_path = Path("./tinydriver_esp32_package.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(header_path, arcname="tinydriver_model_data.h")
        zf.write(tflite_path, arcname="tinydriver_model.tflite")
        zf.write(plot_path, arcname="training_loss.png")

    print("\n" + "=" * 75)
    print(f"🎉 ĐÓNG GÓI THÀNH CÔNG: {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 75)

    # 9. Tự động sao chép vào thư mục dự án nếu chạy trên máy tính
    local_model_target = CURRENT_DIR.parent / "host_laptop" / "models" / "tinydriver_model.tflite"
    local_header_target = CURRENT_DIR.parent / "firmware_esp32" / "main" / "tinydriver_model_data.h"
    if local_model_target.parent.exists():
        shutil.copy2(tflite_path, local_model_target)
        print(f"  ✓ Đã đồng bộ sang: {local_model_target}")
    if local_header_target.parent.exists():
        shutil.copy2(header_path, local_header_target)
        print(f"  ✓ Đã đồng bộ sang: {local_header_target}")

    # 10. Kích hoạt tự động tải xuống nếu đang trên Google Colab
    try:
        from google.colab import files
        print("📥 Đang gửi lệnh tự động tải file tinydriver_esp32_package.zip về máy tính của bạn...")
        files.download(str(zip_path))
        print("✅ Đã kích hoạt tải xuống thành công!")
    except Exception:
        print(f"ℹ️ File kết quả đã sẵn sàng tại: {zip_path}")

    print("\n👉 BƯỚC TIẾP THEO TRÊN MÁY TÍNH CỦA BẠN:")
    print("  1. Chạy lệnh nạp model tự động:")
    print("     python tools/project_manager.py --deploy-model tinydriver_esp32_package.zip")
    print("  2. Chạy thử AI với Webcam trên Laptop:")
    print("     python host_laptop/local_model_tester.py --cam 0")
    print("=" * 75)


if __name__ == "__main__":
    main()
