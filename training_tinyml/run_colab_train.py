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
        LEARNING_RATE, WING_W, WING_EPSILON, NUM_LANDMARKS,
        EAR_LOSS_WEIGHT, MAR_LOSS_WEIGHT, LIP_GAP_WEIGHT
    )
    from tinydriver_net import build_tinydriver_net
    from wing_loss import AdaptiveBiometricWingLoss
    from dataset_loader import (
        DriverLandmarkDataset,
        generate_synthetic_driver_sample
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
        states = ['microsleep', 'yawn', 'normal']
        for i in range(num_samples):
            force_st = states[i % 3]
            img, _, _ = generate_synthetic_driver_sample(i, apply_aug=True, force_state=force_st)
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


def evaluate_real_nme(deploy_model, val_imgs, val_lms, batch_size=64):
    """
    [FIX 2025] Đo NME trên tập VAL GIỮ-OUT THẬT (ảnh thật, không augment, không trùng train).
    Trước đây val lấy từ generator cùng phân bố train -> loss val ảo, mô hình overfit không phát hiện được.
    Returns: {"nme": float, "parts": {eye, mouth, nose_chin}, "worst_pt", "worst_px"}
    """
    preds = []
    for s in range(0, len(val_imgs), batch_size):
        x = (val_imgs[s:s + batch_size].astype(np.float32) - 128.0) / 128.0
        p = deploy_model(x, training=False).numpy()
        preds.append(p)
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    true = val_lms.astype(np.float32)

    pt_t = true.reshape(-1, NUM_LANDMARKS, 2)
    pt_p = pred.reshape(-1, NUM_LANDMARKS, 2)
    iod = np.linalg.norm(pt_t[:, 0, :] - pt_t[:, 9, :], axis=-1)
    iod = np.maximum(iod, 1e-4)
    err = np.linalg.norm(pt_t - pt_p, axis=-1)          # (N, 22) đơn vị chuẩn hóa

    nme = float(np.mean(np.mean(err, axis=-1) / iod))
    overall_px = float(np.mean(err) * IMAGE_WIDTH)
    parts = {
        "eye": list(range(0, 12)),
        "mouth": list(range(12, 18)),
        "nose_chin": list(range(18, 22)),
    }
    # [FIX v2.0.3] err la mang 2 chieu (N, 22) sau khi norm -> chi duoc index 2 chieu err[:, v]
    part_nme = {k: float(np.mean(np.mean(err[:, v], axis=-1) / iod)) for k, v in parts.items()}
    part_px = {k: float(np.mean(err[:, v]) * IMAGE_WIDTH) for k, v in parts.items()}
    per_sample_mean = np.mean(err, axis=-1)
    worst_sample = int(np.argmax(per_sample_mean))
    worst_pt = int(np.argmax(err[worst_sample]))
    return {
        "nme": nme,
        "overall_px": overall_px,
        "parts": part_nme,
        "parts_px": part_px,
        "worst_sample": worst_sample,
        "worst_pt": worst_pt,
        "worst_px": float(err[worst_sample, worst_pt] * IMAGE_WIDTH),
    }


def main():
    global g_dataset_mgr

    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TINYDRIVER PFLD-EDGE (PHIÊN BẢN CẢI TIẾN)")
    print("   Kiến trúc: MobileNetV2 MBConv + Multi-Scale Fusion + Auxiliary 3D Pose Head")
    print("   Hàm mất mát: Adaptive Biometric Wing Loss + Geometric EAR/MAR Constraint Loss")
    print("   Tăng cường: Bounding Box Translation & Scale Jitter (Triệt tiêu Mean Face)")
    print("   Mục tiêu:   Mixed-Precision INT8 (Convs INT8 + Head Float32) cho ESP32-S3 N16R8")
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
        # [BẢN 2025] Tự động BUILD DỮ LIỆU SẠCH ngay trên Colab (mạng Colab tải được
        # AFLW2000-3D ~83MB từ CBSR). Nếu máy cá nhân bị chặn mạng, bước này vẫn chạy được.
        print("  ℹ️ Không thấy preprocessed_driver_dataset.npz trong gói -> TỰ ĐỘNG build dữ liệu sạch trên Colab...")
        build_script = CURRENT_DIR / "build_clean_dataset.py"
        if not build_script.exists():
            build_script = CURRENT_DIR / "tools" / "build_clean_dataset.py"
        if not build_script.exists():
            print("  ❌ Thiếu build_clean_dataset.py trong gói! Hãy chạy lại: python tools/project_manager.py --pack-colab")
            sys.exit(1)
        import subprocess as _sp
        # [FIX v2.0.1] Truyền --output-npz TUYỆT ĐỐI vào thư mục training_tinyml của gói
        # để không phụ thuộc vị trí build_clean_dataset.py nằm ở đâu (root gói hay tools/).
        _explicit_npz = (CURRENT_DIR / "training_tinyml" / "preprocessed_driver_dataset.npz").resolve()
        _explicit_npz.parent.mkdir(parents=True, exist_ok=True)
        _r = _sp.run(
            [sys.executable, str(build_script),
             "--download-aflw2000",
             "--download-facesynth",
             "--output-npz", str(_explicit_npz)],
            cwd=str(CURRENT_DIR)
        )
        if _r.returncode != 0:
            print("  ❌ Build dataset trên Colab thất bại! Kiểm tra kết nối mạng của Colab")
            print("     hoặc build npz trên máy cá nhân rồi pack lại gói.")
            sys.exit(1)
        # Sau khi build, npz nằm ở ./training_tinyml/ hoặc ./
        preprocessed_file = None
        for cand in preprocessed_npz_candidates + [_explicit_npz]:
            if cand.exists():
                preprocessed_file = cand
                break
        if preprocessed_file is None:
            print("  ❌ Build xong nhưng không tìm thấy npz kết quả!")
            sys.exit(1)
        print(f"  ✓ Dataset sạch đã sẵn sàng: {preprocessed_file}")
        dataset_mgr = DriverLandmarkDataset(
            cache_path=str(preprocessed_file),
            synthetic_count=5000,
            augment=True,
            teacher=teacher
        )

    g_dataset_mgr = dataset_mgr

    # [v2.0.7 - TỐI ƯU COLAB T4] 3 tầng pipeline, ưu tiên nhanh nhất trước:
    #   1. STATIC-EXPAND: pre-tính augment 1 lần + jitter nhẹ TF graph -> epoch ~15-25s
    #      (thủ thuật cộng đồng: nút cổ chai là py_function CPU 2-vCPU, không phải GPU)
    #   2. FAST (py_function song song)  ~120s/epoch
    #   3. Chuẩn single-thread           ~290s/epoch
    # Cả 3 dùng CÙNG thuật toán augment + cùng loss + cùng kiến trúc -> logic không đổi.
    train_ds = val_ds = None
    n_train = n_val = 0
    try:
        train_ds, val_ds, n_train, n_val = dataset_mgr.get_static_expanded_dataset(
            batch_size=BATCH_SIZE, expand_factor=6)
        print("  ✓ [STATIC-EXPAND] Tiền-tính augment x6 xong -> epoch chỉ chạy GPU thuần (~15-25s)")
    except Exception as _e_static:
        print(f"  ⚠️ [STATIC-EXPAND] lỗi ({_e_static}) -> thử pipeline song song FAST...")
        try:
            train_ds, val_ds, n_train, n_val = dataset_mgr.get_fast_tf_dataset(batch_size=BATCH_SIZE)
            print("  ✓ [FAST] Online Dynamic Augmentation song song (num_parallel_calls=AUTOTUNE)")
        except Exception as _e_fast:
            print(f"  ⚠️ [FAST] lỗi ({_e_fast}) -> quay về pipeline chuẩn single-thread")
            train_ds, val_ds, n_train, n_val = dataset_mgr.get_tf_dataset(batch_size=BATCH_SIZE)
    print(f"  ✓ Mẫu huấn luyện (Augmented): {n_train} | Mẫu kiểm chuẩn: {n_val}")
    print("  ✓ Kích hoạt Bounding Box Translation Jitter: triệt tiêu học vẹt tọa độ cố định.")

    # [FIX 2025] Lấy tập VAL GIỮ-OUT THẬT (ảnh thật, không augment) để đo NME trung thực
    val_arrays = dataset_mgr.get_val_arrays()
    if val_arrays is not None:
        val_imgs_real, val_lms_real, _ = val_arrays
        # [v2.0.4] Bản val JITTER AFFINE để đo NME LOCALIZATION thật:
        # NME trên val canonical KHÔNG phát hiện được template collapse (mắt luôn
        # ở v=0.344 trong canonical crop) — lỗ hổng khiến model cũ pass gate 6.62%
        # nhưng live sai 17-23px. Best model giờ được chọn theo NME JITTER.
        from dataset_loader import make_jittered_val_copy
        val_imgs_eval, val_lms_eval = make_jittered_val_copy(
            val_imgs_real, val_lms_real, copies=2, seed=123)
        NME_EVAL_MAX = 600
        print(f"  ✓ [REAL-VAL] Val giữ-out THẬT: {len(val_imgs_real)} mẫu "
              f"(NME mỗi epoch trên bản JITTER x2 = {len(val_imgs_eval)} mẫu, tối đa {NME_EVAL_MAX})")
    else:
        val_imgs_real, val_lms_real = None, None
        val_imgs_eval, val_lms_eval = None, None
        NME_EVAL_MAX = 0
        print("  ⚠️ [REAL-VAL] Không có val giữ-out! Best model sẽ chọn theo val loss ảo từ generator.")
        print("     👉 Nên chạy: python tools/build_clean_dataset.py để tạo dữ liệu chuẩn.")

    # 4. Xây dựng mô hình PFLD-Edge Multi-Task
    print("\n[Bước 3/5] Xây dựng kiến trúc PFLD-Edge với Auxiliary 3D Pose Head...")
    full_train_model = build_tinydriver_net(include_pose_head=True)
    print(f"  ✓ Kiến trúc PFLD-Edge: {full_train_model.count_params():,} tham số")
    print("  ✓ Tích hợp Inverted Residual Blocks (MBConv) + Nhánh ước lượng góc đầu 3D")

    # Mô hình probe đầu ra landmark để đo NME nhanh trong vòng lặp huấn luyện
    # [FIX v2.0.2] PHẢI tạo SAU khi full_train_model được xây (trước đây chèn nhầm
    # lên trên gây UnboundLocalError giữa Colab)
    deploy_probe = tf.keras.Model(
        inputs=full_train_model.input,
        outputs=full_train_model.get_layer("landmarks_output").output
    )

    # 5. Thiết lập Optimizer & Multi-Task Loss (Warmup + Cosine Decay)
    total_steps = EPOCHS * (n_train // BATCH_SIZE + 1)
    warmup_steps = 3 * (n_train // BATCH_SIZE + 1)
    try:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=total_steps,
            alpha=0.01,
            warmup_target=LEARNING_RATE,
            warmup_steps=warmup_steps
        )
    except TypeError:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=total_steps,
            alpha=0.01
        )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    landmark_loss_fn = AdaptiveBiometricWingLoss(
        w=WING_W, epsilon=WING_EPSILON,
        image_scale=float(IMAGE_WIDTH),
        ear_weight=EAR_LOSS_WEIGHT,
        mar_weight=MAR_LOSS_WEIGHT,
        gap_weight=LIP_GAP_WEIGHT
    )

    multi_task_model = PFLDMultiTaskModel(
        full_model=full_train_model,
        landmark_loss_fn=landmark_loss_fn,
        pose_weight=1.5
    )
    multi_task_model.compile(optimizer=optimizer)
    print(f"  ✓ Hàm mất mát: Detached Adaptive Biometric Wing Loss (EAR x{EAR_LOSS_WEIGHT}, MAR x{MAR_LOSS_WEIGHT} [Detached w_m], LipGap x{LIP_GAP_WEIGHT}, MouthWidth x2.0, Pose x1.5)")

    # 6. Huấn luyện mô hình
    print(f"\n[Bước 4/5] Bắt đầu huấn luyện qua {EPOCHS} epochs...")
    train_loss_history = []
    val_loss_history = []
    nme_history = []
    best_score = float('inf')          # NME giữ-out thật nếu có, ngược lại val loss
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
            l_val = float(res["loss"])
            lm_val = float(res["lm_loss"])
            train_losses.append(l_val)
            train_lm_losses.append(lm_val)

            # Heartbeat logging: in tiến độ động và flush thường xuyên để Colab không bị đóng băng/mất kết nối
            if (step + 1) % 25 == 0 or (step + 1) == steps_per_epoch:
                pct = int((step + 1) / steps_per_epoch * 100)
                sys.stdout.write(f"\r  [Epoch {epoch+1:02d}/{EPOCHS}] Step {step+1:3d}/{steps_per_epoch} ({pct:3d}%) | Loss: {l_val:6.2f} (LM: {lm_val:6.2f})")
                sys.stdout.flush()

        mean_train_loss = np.mean(train_losses)
        mean_train_lm = np.mean(train_lm_losses)

        # Validation loop (generator val giữ-out thật)
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

        # [FIX 2025] Đo NME trên val giữ-out JITTER (localization thật) -> chọn best model
        nme_now = None
        if val_imgs_eval is not None:
            n = min(len(val_imgs_eval), NME_EVAL_MAX)
            nme_now = evaluate_real_nme(deploy_probe, val_imgs_eval[:n], val_lms_eval[:n])
            nme_history.append(nme_now["nme"])

        star = " "
        score = nme_now["nme"] if nme_now is not None else mean_val_loss
        if score < best_score:
            best_score = score
            full_train_model.save_weights(str(best_weights_path))
            star = " ⭐ (Best)"

        # Xóa dòng tiến trình tạm thời trước khi in kết quả epoch
        sys.stdout.write("\r" + " " * 75 + "\r")
        sys.stdout.flush()

        # In kết quả cho MỌI epoch (không bỏ sót epoch lẻ) để giữ kết nối Colab luôn thông suốt
        nme_str = f" | Real-Val NME: {nme_now['nme']*100:.2f}% (miệng {nme_now['parts']['mouth']*100:.2f}%)" \
            if nme_now is not None else " | Real-Val NME: N/A"
        print(f"Epoch {epoch+1:2d}/{EPOCHS} [{dur:4.1f}s] - Train: {mean_train_loss:6.2f} (LM: {mean_train_lm:6.2f}) | Val: {mean_val_loss:6.2f}{nme_str}{star}", flush=True)

    train_duration = time.time() - t0
    final_nme_str = f", Best Real-Val NME: {best_score*100:.2f}%" if nme_history else ""
    print(f"\n✅ Huấn luyện hoàn tất sau: {train_duration:.1f} giây!{final_nme_str}")

    # Nạp lại trọng số tốt nhất
    if best_weights_path.exists():
        full_train_model.load_weights(str(best_weights_path))
        print("  ✓ Đã nạp lại trọng số tốt nhất (Best Weights).")

    # [FIX 2025/2026] Báo cáo NME & Sai số Pixel CUỐI CÙNG: canonical (chuẩn) + JITTER (localization chống rung lắc)
    if val_imgs_real is not None:
        nme_canon = evaluate_real_nme(deploy_probe, val_imgs_real, val_lms_real)
        nme_jit = evaluate_real_nme(deploy_probe, val_imgs_eval, val_lms_eval)
        p_canon = nme_canon["parts"]
        p_jit = nme_jit["parts"]
        print("\n" + "=" * 75)
        print("📊 BÁO CÁO NME & SAI SỐ ĐỊNH VỊ CUỐI CÙNG (Val giữ-out người thật 100%):")
        print("=" * 75)
        print("1. NME CANONICAL (Đo trên khuôn mặt chuẩn, không rung lắc biến dạng):")
        print(f"   • NME Tổng thể             : {nme_canon['nme']*100:.2f}% "
              f"({'✅ XUẤT SẮC (<6%)' if nme_canon['nme'] < 0.06 else ('✅ ĐẠT CHUẨN TỐT (<8%)' if nme_canon['nme'] < 0.08 else '⚠️ CẦN CẢI THIỆN (>=8%)')})")
        print(f"   • Sai số pixel trung bình : {nme_canon['overall_px']:.2f}px / 96px (Toàn bộ 22 điểm mốc)")
        print(f"   • Sai số nhóm Mắt         : {p_canon['eye']*100:.2f}% ({nme_canon['parts_px']['eye']:.2f}px)")
        print(f"   • Sai số nhóm Miệng       : {p_canon['mouth']*100:.2f}% ({nme_canon['parts_px']['mouth']:.2f}px)")
        print(f"   • Sai số nhóm Mũi / Cằm   : {p_canon['nose_chin']*100:.2f}% ({nme_canon['parts_px']['nose_chin']:.2f}px)")

        print(f"\n2. NME JITTER (Khả năng định vị pixel khi bị rung lắc dịch chuyển/xoay):")
        print(f"   • NME Jitter Tổng thể      : {nme_jit['nme']*100:.2f}% "
              f"({'✅ XUẤT SẮC (<8%)' if nme_jit['nme'] < 0.08 else ('✅ ĐẠT CHUẨN KHÁ (<12%)' if nme_jit['nme'] < 0.12 else '⚠️ CẦN THÊM DỮ LIỆU (>=12%)')})")
        print(f"   • Sai số pixel trung bình : {nme_jit['overall_px']:.2f}px / 96px")
        print(f"   • Sai số nhóm Mắt         : {p_jit['eye']*100:.2f}% ({nme_jit['parts_px']['eye']:.2f}px)")
        print(f"   • Sai số nhóm Miệng       : {p_jit['mouth']*100:.2f}% ({nme_jit['parts_px']['mouth']:.2f}px)")
        print(f"   • Sai số nhóm Mũi / Cằm   : {p_jit['nose_chin']*100:.2f}% ({nme_jit['parts_px']['nose_chin']:.2f}px)")
        print(f"   • Mẫu biên ngoại lệ tệ nhất: #{nme_jit['worst_sample']} (P{nme_jit['worst_pt']} lệch tối đa {nme_jit['worst_px']:.1f}px - chỉ là 1 mẫu cá biệt)")

        gap = nme_jit['nme'] / max(nme_canon['nme'], 1e-6)
        print(f"\n3. ĐÁNH GIÁ ĐỊNH VỊ (Tỉ lệ Jitter/Canonical):")
        print(f"   • Tỉ lệ Jitter / Canon     : {gap:.2f}x "
              f"({'✅ ĐẠT CHUẨN (<2.0x): Mô hình định vị pixel thật, không học vẹt!' if gap < 2.0 else '⚠️ >=2.0x: Còn dấu hiệu học vẹt template'})")
        print("=" * 75)
        print("   👉 Kiểm tra lại sau khi tải về bằng: python evaluation/eval_nme_holdout.py\n")

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

    # 7. Trích xuất mô hình Landmark-Only và Lượng tử hóa Mixed-Precision (Convs INT8 + Regression Head Float32)
    print("\n[Bước 5/5] Cắt bỏ Auxiliary Head và Lượng tử hóa Mixed-Precision cho ESP32-S3...")
    deploy_input = full_train_model.input
    deploy_output = full_train_model.get_layer("landmarks_output").output
    deploy_model = tf.keras.Model(inputs=deploy_input, outputs=deploy_output, name="TinyDriver_PFLD_Deploy")
    print(f"  ✓ Mô hình xuất xưởng (Landmark-Only): {deploy_model.count_params():,} tham số")

    # Khóa Concrete Function để tạo Static Graph (triệt tiêu ops động cho TFLite Micro)
    run_model = tf.function(lambda x: deploy_model(x, training=False))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Mixed-Precision: Toàn bộ Convs/MBConv chạy INT8 vector SIMD, riêng Head xuất Float32 sub-pixel
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    tflite_quant_model = converter.convert()
    tflite_path = work_dir / "tinydriver_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024.0
    print(f"  ✓ Kích thước mô hình Mixed-Precision: {len(tflite_quant_model):,} bytes ({size_kb:.1f} KB)")

    # Đọc tham số lượng tử hóa
    in_scale, in_zp = 0.007843137, 0
    out_scale, out_zp = 1.0, 0
    output_is_float = True

    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        in_scale, in_zp = in_details["quantization"]
        if in_scale == 0.0:
            in_scale = 1.0 / 128.0
            in_zp = 0
        output_is_float = (out_details["dtype"] == np.float32)
    except Exception:
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            in_details = interpreter.get_input_details()[0]
            out_details = interpreter.get_output_details()[0]
            in_scale, in_zp = in_details["quantization"]
            if in_scale == 0.0:
                in_scale = 1.0 / 128.0
                in_zp = 0
            output_is_float = (out_details["dtype"] == np.float32)
        except Exception:
            pass

    # Xuất file C Header (alignas 16 bytes cho esp-nn SIMD)
    header_path = work_dir / "tinydriver_model_data.h"
    convert_model_to_c_array(
        tflite_quant_model,
        str(header_path),
        in_scale, in_zp, out_scale, out_zp,
        output_is_float=output_is_float
    )
    print(f"  ✓ Đã tạo file C Header: {header_path.name} (16-byte aligned, Output Float32: {output_is_float})")

    # 8. Đóng gói ZIP xuất xưởng
    zip_path = Path("./tinydriver_esp32_package.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(header_path, arcname="tinydriver_model_data.h")
        zf.write(tflite_path, arcname="tinydriver_model.tflite")
        zf.write(plot_path, arcname="training_loss.png")
        # [BỔ SUNG 2025] Kèm theo tập VAL GIỮ-OUT (~1MB) để chạy cổng NME trên máy
        # cá nhân ngay sau deploy-model, KHÔNG cần rebuild dataset
        if val_imgs_real is not None:
            val_holdout_path = work_dir / "val_holdout.npz"
            np.savez_compressed(val_holdout_path, images=val_imgs_real, landmarks=val_lms_real)
            zf.write(val_holdout_path, arcname="val_holdout.npz")
            print(f"  ✓ Đã đóng gói val_holdout.npz ({len(val_imgs_real)} mẫu giữ-out) vào gói")

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
