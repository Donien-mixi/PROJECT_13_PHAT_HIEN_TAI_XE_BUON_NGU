"""
Training Pipeline for TinyDriver-LandmarkNet.
Optimized for Google Colab GPU (Tesla T4) and CPU execution.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    from config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, EPOCHS,
        LEARNING_RATE, WING_W, WING_EPSILON
    )
    from tinydriver_net import build_tinydriver_net
    from wing_loss import WingLoss
    from dataset_loader import DriverLandmarkDataset
except (ImportError, ValueError):
    from .config import (
        IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, EPOCHS,
        LEARNING_RATE, WING_W, WING_EPSILON
    )
    from .tinydriver_net import build_tinydriver_net
    from .wing_loss import WingLoss
    from .dataset_loader import DriverLandmarkDataset

def compute_nme(y_true, y_pred):
    """
    Computes Normalized Mean Error (NME) using inter-ocular distance.
    y_true, y_pred: shape (batch, 44) representing 22 points (x, y) in [0.0, 1.0].
    """
    pts_true = tf.reshape(y_true, [-1, 22, 2]) * float(IMAGE_WIDTH)
    pts_pred = tf.reshape(y_pred, [-1, 22, 2]) * float(IMAGE_WIDTH)

    # Point 0: Left eye outer corner; Point 9: Right eye outer corner
    interocular_dist = tf.norm(pts_true[:, 0, :] - pts_true[:, 9, :], axis=-1)
    interocular_dist = tf.maximum(interocular_dist, 1e-4)

    point_errors = tf.norm(pts_true - pts_pred, axis=-1) # (batch, 22)
    mean_point_error = tf.reduce_mean(point_errors, axis=-1) # (batch,)
    nme = tf.reduce_mean(mean_point_error / interocular_dist)
    return nme


def main():
    parser = argparse.ArgumentParser(description="Train TinyDriver-LandmarkNet for ESP32-S3 Edge AI")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to 300W / iBUG dataset folder")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic driver dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TINYDRIVER-LANDMARKNET (TINYML ADAS)")
    print(f"👉 Kích thước đầu vào: {IMAGE_WIDTH}x{IMAGE_HEIGHT} Grayscale (Isomorphic 1:1)")
    print(f"👉 Số lượng điểm mốc: 22 Facial Landmarks")
    print(f"👉 Thiết bị phần cứng: {tf.config.list_physical_devices()}")
    print("=" * 60)

    # 1. Dataset Initialization
    dataset_mgr = DriverLandmarkDataset(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        synthetic_count=3500
    )
    train_ds, val_ds, n_train, n_val = dataset_mgr.get_tf_dataset(batch_size=args.batch_size)
    print(f"[Dataset] Mẫu huấn luyện: {n_train} | Mẫu kiểm chuẩn: {n_val}")

    # 2. Build Model
    model = build_tinydriver_net()
    print(f"[Model] Tổng số tham số: {model.count_params():,} params")

    # 3. Compile Model with Wing Loss
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.epochs * (n_train // args.batch_size + 1),
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    loss_fn = WingLoss(w=WING_W, epsilon=WING_EPSILON, image_scale=float(IMAGE_WIDTH))

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mae", compute_nme]
    )

    # 4. Callbacks
    best_model_path = os.path.join(args.output_dir, "tinydriver_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 5. Execute Training Loop
    print(f"\n[Training] Bắt đầu huấn luyện qua {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save Final Model and Training Plot
    final_model_path = os.path.join(args.output_dir, "tinydriver_final.keras")
    model.save(final_model_path)
    print(f"[Save] Đã lưu mô hình tốt nhất tại: {best_model_path}")

    # Plot Loss Curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Wing Loss")
    plt.plot(history.history["val_loss"], label="Val Wing Loss")
    plt.title("Wing Loss qua các Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if "compute_nme" in history.history:
        plt.plot(history.history["compute_nme"], label="Train NME")
        plt.plot(history.history["val_compute_nme"], label="Val NME")
        plt.title("Normalized Mean Error (NME)")
        plt.xlabel("Epoch")
        plt.ylabel("NME")
        plt.legend()
        plt.grid(True)

    plot_path = os.path.join(args.output_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[Plot] Đã xuất biểu đồ học tập tại: {plot_path}")
    print("=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print("=" * 60)


if __name__ == "__main__":
    main()
