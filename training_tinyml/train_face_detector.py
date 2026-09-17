"""
D5-v5 — Train FACE DETECTOR NHẸ (96x96 grayscale -> box), CÔNG THỨC HỒI QUY CHUẨN
================================================================================
Vì sao v3/v4 loss đứng yên: dùng loss tùy biến (Huber+IoU) + bias-khởi-tạo-bằng-trung-bình
-> mạng mắc kẹt ở "hộp trung bình" (không học đặc trưng).

v5 dùng công thức CHUẨN, tin cậy:
  1) CHUẨN HOÁ NHÃN (z-score: t = (box - mean)/std) -> bài toán hồi quy "dễ".
  2) Model: backbone MBConv nhẹ, đầu ra Dense(4) LINEAR, khởi tạo mặc định.
  3) Loss: MSE (mean_squared_error) thuần -> ổn định, không "trung bình hoá" kỳ dị.
  4) Denormalize khi đánh giá/ chạy thực tế (đã xuất mean/std vào header cho firmware).
  5) INT8 mixed-precision (input INT8, output float32).

Chạy Colab:
  !python train_face_detector.py --save-to /content/drive/MyDrive/PROJECT_13/fd_out
"""

import os
import shutil
import sys

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 2026
SIZE = 96
NPZ = "face_detection_dataset.npz"
OUT_TFLITE = "face_detector_lite.tflite"
OUT_HEADER = "face_detector_lite_model_data.h"
SYMBOL = "g_face_detector_lite"
EPOCHS = 300
BATCH = 64
LR = 1e-3

tf.random.set_seed(SEED)
np.random.seed(SEED)


def mbconv(x, exp, out, stride):
    ch = x.shape[-1]
    y = layers.Conv2D(exp, 1, 1, "same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.DepthwiseConv2D(3, stride, "same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(out, 1, 1, "same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride == 1 and ch == out:
        y = layers.Add()([x, y])
    return y


def build_model():
    inp = keras.Input((SIZE, SIZE, 1))
    x = layers.Conv2D(24, 3, 2, "same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = mbconv(x, 1, 24, 2)
    x = mbconv(x, 2, 32, 2)
    x = mbconv(x, 2, 48, 2)
    x = layers.Conv2D(96, 1, 1, "same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, activation=None)(x)        # LINEAR, khởi tạo mặc định
    return keras.Model(inp, out, name="FaceDetectorLite")


def _augment(img, tgt):
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.clip_by_value(img, -1.0, 1.0)
    if tf.random.uniform(()) < 0.5:
        img = tf.image.flip_left_right(img)
        tgt = tf.stack([1.0 - tgt[0], tgt[1], tgt[2], tgt[3]])
    return img, tgt


def iou_np(a, b):
    x1 = np.maximum(a[:, 0] - a[:, 2] / 2, b[:, 0] - b[:, 2] / 2)
    y1 = np.maximum(a[:, 1] - a[:, 3] / 2, b[:, 1] - b[:, 3] / 2)
    x2 = np.minimum(a[:, 0] + a[:, 2] / 2, b[:, 0] + b[:, 2] / 2)
    y2 = np.minimum(a[:, 1] + a[:, 3] / 2, b[:, 1] + b[:, 3] / 2)
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    un = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-7
    return inter / un


def main():
    assert os.path.exists(NPZ), f"Thiếu {NPZ}"
    z = np.load(NPZ, allow_pickle=True)
    images, box, obj, split = z["images"], z["box"], z["obj"], z["split"]

    m_tr = (split == 0) & (obj == 1)
    m_va = (split == 1) & (obj == 1)
    m_ov = (split == 2) & (obj == 1)
    Xtr = (images[m_tr].astype(np.float32) - 127.5) / 128.0
    Xva = (images[m_va].astype(np.float32) - 127.5) / 128.0
    Xov = (images[m_ov].astype(np.float32) - 127.5) / 128.0
    Ytr, Yva, Yov = box[m_tr].astype(np.float32), box[m_va].astype(np.float32), box[m_ov].astype(np.float32)
    print(f"train {len(Xtr)} | val {len(Xva)} | val_OV5640 {len(Xov)}")

    # [v5] CHUẨN HOÁ NHÃN (z-score) -> hồi quy dễ hội tụ
    mean = Ytr.mean(axis=0)
    std = Ytr.std(axis=0) + 1e-6
    Ttr = (Ytr - mean) / std
    print("mean box:", np.round(mean, 3).tolist(), "| std:", np.round(std, 3).tolist())

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    model.summary()

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr[..., None].astype(np.float32), Ttr))
             .shuffle(min(len(Xtr), 20000), seed=SEED, reshuffle_each_iteration=True)
             .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva[..., None].astype(np.float32), (Yva - mean) / std)).batch(BATCH)

    # --- SMOKE TEST: xác nhận mạng HỌC được ngay từ đầu (tránh tốn 1 lượt train) ---
    print("\n--- SMOKE TEST (30 batch x 2 epoch) ---")
    hs = model.fit(ds_tr.take(30), epochs=2, verbose=2)
    l0, l1 = hs.history["loss"][0], hs.history["loss"][-1]
    if l1 < l0 * 0.98:
        print(f"SMOKE: OK - loss {l0:.4f} -> {l1:.4f} (dang hoc)")
    else:
        print(f"SMOKE: !!! CANH BAO - loss {l0:.4f} -> {l1:.4f} KHONG giam (bao lai log nay)")
    print("-" * 40 + "\n")

    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ]
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=cbs, verbose=2)

    def report(tag, X, Y):
        p = model.predict(X[..., None], verbose=0) * std + mean
        c = np.linalg.norm((p[:, :2] - Y[:, :2]) * SIZE, axis=1)
        iou = iou_np(Y, p)
        print(f"  [{tag}] box_IoU={iou.mean():.3f} (p10 {np.percentile(iou,10):.3f}) | "
              f"center_err={c.mean():.2f}px (p90 {np.percentile(c,90):.2f})")

    print("\n=== Đánh giá ===")
    report("val", Xva, Yva)
    if len(Xov):
        report("val_OV5640 (in-domain)", Xov, Yov)

    # Lượng tử INT8 (input int8, output float32) + nhúng mean/std vào header
    rep = Xtr[np.random.choice(len(Xtr), min(300, len(Xtr)), replace=False)][..., None].astype(np.float32)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.float32
    conv.representative_dataset = lambda: ([rep[i:i + 1]] for i in range(len(rep)))
    tfl = conv.convert()
    open(OUT_TFLITE, "wb").write(tfl)
    print(f"\n✅ {OUT_TFLITE}: {len(tfl)} bytes")

    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    idet = it.get_input_details()[0]
    odet = it.get_output_details()[0]
    print(f"   INPUT  {idet['shape']} {idet['dtype'].__name__} quant={idet['quantization']}")
    print(f"   OUTPUT {odet['shape']} {odet['dtype'].__name__} quant={odet['quantization']}")
    print("   -> Firmware: INPUT_W=96 INPUT_H=96 SCALE=%.10f ZP=%d | OUT=4 float32"
          % (idet["quantization"][0], idet["quantization"][1]))
    print("   -> Denormalize: box = out*std + mean | mean=%s std=%s"
          % (np.round(mean, 6).tolist(), np.round(std, 6).tolist()))

    lines = []
    for i in range(0, len(tfl), 12):
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in tfl[i:i + 12]) + ",")
    mstr = ", ".join(f"{v:.8f}f" for v in mean)
    sstr = ", ".join(f"{v:.8f}f" for v in std)
    hdr = (f"// Auto-generated from {OUT_TFLITE} ({len(tfl)} bytes)\n"
           f"#ifndef {SYMBOL.upper()}_H_\n#define {SYMBOL.upper()}_H_\n\n"
           f"#define FDL_INPUT_W       96\n#define FDL_INPUT_H       96\n"
           f"#define FDL_INPUT_SCALE   {idet['quantization'][0]:.10f}f\n"
           f"#define FDL_INPUT_ZP      {idet['quantization'][1]}\n"
           f"// box = out * FDL_STD + FDL_MEAN  (out la 4 gia tri float32 cua model)\n"
           f"static const float FDL_MEAN[4] = {{ {mstr} }};\n"
           f"static const float FDL_STD[4]  = {{ {sstr} }};\n\n"
           f"extern const unsigned char {SYMBOL}[];\nconst unsigned char {SYMBOL}[] = {{\n"
           + "\n".join(lines) + "\n};\n\n#endif\n")
    open(OUT_HEADER, "w", encoding="utf-8").write(hdr)
    print(f"✅ {OUT_HEADER}")

    save_to = None
    for i, a in enumerate(sys.argv):
        if a == "--save-to" and i + 1 < len(sys.argv):
            save_to = sys.argv[i + 1]
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        for f in (OUT_TFLITE, OUT_HEADER):
            shutil.copy(f, os.path.join(save_to, f))
        print(f"💾 Đã lưu 2 file vào: {save_to}")
    print("XONG!")


if __name__ == "__main__":
    main()
