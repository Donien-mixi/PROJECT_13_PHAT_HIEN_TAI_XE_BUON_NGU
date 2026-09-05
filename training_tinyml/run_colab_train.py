#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 SCRIPT THỰC THI HUẤN LUYỆN TỰ ĐỘNG TRÊN GOOGLE COLAB (RUN COLAB TRAIN)
=============================================================================
Đồ án 13: Hệ thống phát hiện tài xế ngủ gật & mất tập trung (Edge AI ESP32-S3)

Phiên bản nâng cấp tân tiến:
  1. Biometric-Weighted Wing Loss (Ưu tiên mí mắt x2.0, khóe miệng x1.5)
  2. Data Augmentation đa miền cabin xe (Chói nắng, hầm tối, rung lắc, gọng kính)
  3. Teacher-Student Knowledge Distillation (MediaPipe Face Mesh Teacher)
  4. Lượng tử hóa Full-Integer INT8 cho ESP32-S3 (Xtensa LX7 SIMD 128-bit)
  5. Đóng gói và tự động tải xuống tinydriver_esp32_package.zip
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
    except ImportError:
        pass

# Nạp các module trong gói
try:
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, EPOCHS,
        LEARNING_RATE, WING_W, WING_EPSILON, NUM_LANDMARKS
    )
    from tinydriver_net import build_tinydriver_net
    from wing_loss import WingLoss, BiometricWeightedWingLoss
    from dataset_loader import (
        DriverLandmarkDataset, generate_synthetic_driver_sample,
        download_drowsiness_benchmark_dataset
    )
    from distillation import MediaPipeTeacher, DistillationModel
    from export_tflite import convert_model_to_c_array
except ImportError as e:
    import traceback
    print(f"❌ [LỖI IMPORT]: {e}")
    traceback.print_exc()
    sys.exit(1)


# Biến toàn cục quản lý dataset để dùng cho representative dataset generator
g_dataset_mgr = None


def representative_dataset_gen(num_samples=300):
    """Tạo tập mẫu hiệu chuẩn số nguyên INT8 với ảnh thực tế và điều kiện cabin."""
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
            img, _ = generate_synthetic_driver_sample(i, apply_aug=True)
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TINYDRIVER-LANDMARKNET (PHIÊN BẢN NÂNG CẤP)")
    print("   Bộ dữ liệu: YawDD (Yawning) + CEW (Eye Closure) + MediaPipe Teacher")
    print("   Phương pháp: Biometric-Weighted Wing Loss (Mắt x2.0, Miệng x1.8)")
    print("   Cơ chế:      Spatial Head + Full-Integer INT8 cho ESP32-S3")
    print("=" * 75)

    # 1. Kiểm tra phần cứng GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Đã kích hoạt phần cứng GPU: {gpus[0].name} (Khuyên dùng Tesla T4)")
    else:
        print("⚠️ Đang chạy trên CPU (Tốc độ sẽ chậm hơn so với GPU)")

    work_dir = Path("./colab_output").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2. Khởi tạo Mô hình Thầy (Teacher)
    print("\n[Bước 1/5] Khởi tạo Mô hình Thầy MediaPipe Face Mesh...")
    teacher = MediaPipeTeacher()
    if teacher.available:
        print("  ✓ Mô hình Thầy (Teacher): MediaPipe Face Mesh đã kích hoạt dán nhãn sinh học!")
    else:
        print("  ✓ Chế độ: Huấn luyện độc lập tối ưu sinh học (Autonomous Biometric Training)")

    # 3. Chuẩn bị dữ liệu thực tế YawDD & CEW và trích xuất Ground-Truth
    print("\n[Bước 2/5] Chuẩn bị dữ liệu thực tế YawDD & CEW + In-Cabin Augmentation...")
    dataset_dir = Path("./datasets/drowsiness_benchmark").resolve()
    download_drowsiness_benchmark_dataset(str(dataset_dir))

    cache_path = work_dir / "drowsiness_cache.npz"
    dataset_mgr = DriverLandmarkDataset(
        data_dir=str(dataset_dir),
        cache_path=str(cache_path),
        synthetic_count=4500,
        augment=True,
        teacher=teacher
    )
    g_dataset_mgr = dataset_mgr

    train_ds, val_ds, n_train, n_val = dataset_mgr.get_tf_dataset(batch_size=BATCH_SIZE)
    print(f"  ✓ Mẫu huấn luyện (Augmented): {n_train} | Mẫu kiểm chuẩn: {n_val}")
    print("  ✓ Kích hoạt In-Cabin Augmentation: Chói sáng, hầm tối, nhiễu ISO, rung lắc, kính cận.")

    # 4. Xây dựng kiến trúc TinyDriverNet với Spatial Feature Head
    print("\n[Bước 3/5] Xây dựng kiến trúc TinyDriverNet (Spatial Feature Preservation Head)...")
    student_model = build_tinydriver_net()
    print(f"  ✓ Kiến trúc TinyDriverNet: {student_model.count_params():,} tham số (~190KB INT8)")
    print("  ✓ Bảo toàn không gian 2D cho mi mắt và khóe miệng (Không làm phẳng bằng GAP)")

    # 5. Thiết lập Optimizer & Biometric-Weighted Wing Loss
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * (n_train // BATCH_SIZE + 1),
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    loss_fn = BiometricWeightedWingLoss(
        w=WING_W, epsilon=WING_EPSILON,
        image_scale=float(IMAGE_WIDTH),
        use_biometric_weights=True
    )
    print("  ✓ Hàm mất mát: Biometric-Weighted Wing Loss (Mắt x2.0, Miệng x1.8, Dáng đầu x1.0)")

    student_model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae", compute_nme])

    # 6. Tiến hành huấn luyện
    print(f"\n[Bước 4/5] Bắt đầu huấn luyện qua {EPOCHS} epochs...")
    best_model_path = work_dir / "tinydriver_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=14,
            restore_best_weights=True,
            verbose=1
        )
    ]

    t0 = time.time()
    history = student_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - t0
    print(f"✅ Huấn luyện hoàn tất sau: {train_duration:.1f} giây!")

    # Vẽ và lưu biểu đồ Loss
    plot_path = work_dir / "training_loss.png"
    plt.figure(figsize=(9, 3.8))
    plt.plot(history.history["loss"], label="Train Weighted Wing Loss", color="#1E88E5", lw=2)
    plt.plot(history.history["val_loss"], label="Val Weighted Wing Loss", color="#D81B60", lw=2)
    plt.title("Tiến Trình Huấn Luyện TinyDriverNet (Biometric-Weighted Wing Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓ Đã lưu biểu đồ tiến trình: {plot_path.name}")

    # 7. Lượng tử hóa Full-Integer INT8 cho ESP32-S3
    print("\n[Bước 5/5] Lượng tử hóa Full-Integer INT8 và xuất C Header...")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
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

    # 7. Đóng gói ZIP xuất xưởng
    zip_path = Path("./tinydriver_esp32_package.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(header_path, arcname="tinydriver_model_data.h")
        zf.write(tflite_path, arcname="tinydriver_model.tflite")
        zf.write(plot_path, arcname="training_loss.png")

    print("\n" + "=" * 75)
    print(f"🎉 ĐÓNG GÓI THÀNH CÔNG: {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 75)

    # 8. Kích hoạt tự động tải xuống nếu đang trên Google Colab
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
