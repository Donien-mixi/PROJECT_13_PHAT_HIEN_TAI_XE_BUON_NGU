#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
📏 EVAL NME HOLDOUT - Cổng kiểm tra chất lượng mô hình TRƯỚC KHI NẠP ESP32
=============================================================================
BẢN v2.0.4 — Đo HAI chỉ số trên tập VAL GIỮ-OUT THẬT:

  1. NME CANONICAL  : val giữ nguyên crop canonical → chỉ số "template-fit".
     Mọi canonical crop đều có mắt ở v≈0.344 (crop được dựng TỪ landmark GT)
     nên model học-thuộc-template vẫn đạt điểm cao ở đây.

  2. NME JITTER (CHỈ SỐ QUYẾT ĐỊNH) : val được jitter affine ngẫu nhiên
     (dịch ±5%, scale 0.90-1.12, xoay ±7°, seed cố định) → vị trí mắt/mũi/miệng
     THAY ĐỔI trong khung → CHỈ model ĐỊNH VỊ pixel thật mới đạt điểm.
     Đây là chỉ số bắt được lỗi "model cũ pass 6.62% nhưng live sai 17-23px".

  3. TỈ LỆ Jitter/Canon ≥ 2.0x  → cảnh báo template collapse.

Chuẩn đắn trước khi nạp ESP32:
  - NME JITTER < 6%   : XUẤT SẮC — nạp ESP32
  - NME JITTER < 8%   : ĐẠT — nạp được
  - NME JITTER < 10%  : TRUNG BÌNH — nên bổ sung dữ liệu rồi train lại
  - NME JITTER >= 10% : YẾU — KHÔNG nạp ESP32

Cách chạy:
  python evaluation/eval_nme_holdout.py
=============================================================================
"""

import sys
import argparse
from pathlib import Path

import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT_DIR / "training_tinyml"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

NUM_LANDMARKS = 22
INPUT_W = 96
INPUT_H = 96
LEFT_EYE = list(range(0, 6))
RIGHT_EYE = list(range(6, 12))
MOUTH = list(range(12, 18))
NOSE_CHIN = list(range(18, 22))
PARTS = {"Mắt": LEFT_EYE + RIGHT_EYE, "Miệng": MOUTH, "Mũi/Cằm": NOSE_CHIN}


def get_interpreter_cls():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        return None


def load_val_holdout(npz_path):
    """Nạp tập val giữ-out. Hỗ trợ 2 định dạng:
    - val_holdout.npz (đi kèm gói Colab): toàn bộ mẫu đều là val.
    - preprocessed_driver_dataset.npz: lọc split==1 (fallback 8% cuối)."""
    data = np.load(npz_path, allow_pickle=False)
    images = data['images']
    landmarks = data['landmarks']
    if 'split' in data:
        split = data['split'].astype(np.int32)
        idx = np.where(split == 1)[0]
        if len(idx) == 0:
            print("⚠️ npz không có mẫu split=1! Chạy lại: python tools/build_clean_dataset.py")
            return None, None
        return images[idx], landmarks[idx]
    if 'val_holdout' in str(npz_path):
        return images, landmarks
    n_val = max(int(len(images) * 0.08), 8)
    print("⚠️ npz thiếu cột 'split' -> dùng 8% cuối. Nên build lại dataset.")
    return images[-n_val:], landmarks[-n_val:]


def main():
    parser = argparse.ArgumentParser(description="Đo NME giữ-out (canonical + jitter) của mô hình TFLite")
    parser.add_argument("--model", type=str,
                        default=str(ROOT_DIR / "host_laptop" / "models" / "tinydriver_model.tflite"))
    parser.add_argument("--npz", type=str, default=None,
                        help="Đường dẫn npz val (mặc định tự tìm val_holdout.npz hoặc preprocessed_driver_dataset.npz)")
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    Interpreter = get_interpreter_cls()
    if Interpreter is None:
        print("❌ Cần cài: pip install ai-edge-litert (hoặc tensorflow)")
        sys.exit(2)
    if not Path(args.model).exists():
        print(f"❌ Không tìm thấy model: {args.model}")
        print("   👉 Train xong chạy: python tools/project_manager.py deploy-model tinydriver_esp32_package.zip")
        sys.exit(2)

    if args.npz:
        npz_path = Path(args.npz)
    else:
        candidates = [
            TRAINING_DIR / "val_holdout.npz",
            TRAINING_DIR / "preprocessed_driver_dataset.npz",
        ]
        npz_path = next((c for c in candidates if c.exists()), None)
    if npz_path is None or not Path(npz_path).exists():
        print("❌ Không tìm thấy tập val giữ-out! Đã thử:")
        print("   - training_tinyml/val_holdout.npz (nạp từ gói Colab qua deploy-model)")
        print("   - training_tinyml/preprocessed_driver_dataset.npz")
        print("   👉 Hoặc: python tools/build_clean_dataset.py --download-aflw2000")
        sys.exit(2)

    val_imgs, val_lms = load_val_holdout(npz_path)
    if val_imgs is None:
        sys.exit(2)
    n = min(len(val_imgs), args.max_samples)
    val_imgs, val_lms = val_imgs[:n], val_lms[:n]

    print("=" * 74)
    print("📏 ĐO NME GIỮ-OUT: CANONICAL (template-fit) + JITTER (localization thật)")
    print("=" * 74)
    print(f"• Model : {args.model}")
    print(f"• Val   : {n} mẫu giữ-out thật | NME JITTER đo trên bản x2 jitter affine\n")

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    if in_scale == 0.0:
        in_scale, in_zp = 1.0 / 128.0, 0
    out_is_float = (out_det['dtype'] == np.float32)
    print(f"• Input : {in_det['shape']} {in_det['dtype'].__name__} | "
          f"Output: {'Float32 (mixed-precision)' if out_is_float else 'INT8'}\n")

    def predict_all(images):
        preds = []
        for i in range(len(images)):
            img = images[i, :, :, 0].astype(np.float32)
            norm = (img - 128.0) / 128.0
            quant = np.clip(np.round(norm / in_scale) + in_zp, -128, 127).astype(in_det['dtype'])
            tensor = np.expand_dims(np.expand_dims(quant, 0), -1)
            interp.set_tensor(in_det['index'], tensor)
            interp.invoke()
            out = interp.get_tensor(out_det['index'])[0].astype(np.float32)
            if not out_is_float:
                o_scale, o_zp = out_det['quantization']
                if o_scale != 0.0:
                    out = (out - o_zp) * o_scale
                else:
                    out = (out + 128.0) / 255.0
            preds.append(np.clip(out, 0.0, 1.0))
        return np.array(preds, dtype=np.float32)

    def compute_metrics(pred, lms):
        pt_t = lms.reshape(-1, NUM_LANDMARKS, 2)
        pt_p = pred.reshape(-1, NUM_LANDMARKS, 2)
        iod = np.maximum(np.linalg.norm(pt_t[:, 0, :] - pt_t[:, 9, :], axis=-1), 1e-4)
        err = np.linalg.norm(pt_t - pt_p, axis=-1)      # (N, 22)
        nme = float(np.mean(np.mean(err, axis=-1) / iod))
        part_nme = {k: float(np.mean(np.mean(err[:, v], axis=-1) / iod)) for k, v in PARTS.items()}
        part_px = {k: float(np.mean(err[:, v]) * INPUT_W) for k, v in PARTS.items()}
        worst_i = int(np.argmax(np.mean(err, axis=-1)))
        worst_pt = int(np.argmax(err[worst_i]))
        return {"nme": nme, "parts": part_nme, "px": part_px,
                "overall_px": float(np.mean(err) * INPUT_W),
                "worst_sample": worst_i, "worst_pt": worst_pt,
                "worst_px": float(err[worst_i, worst_pt] * INPUT_W)}

    def print_table(tag, m):
        print(f"--- {tag} ---")
        print(f"{'Nhóm điểm mốc':<14} | {'NME':>8} | {'Sai px':>8}")
        print("-" * 42)
        for name in PARTS:
            print(f"{name:<14} | {m['parts'][name]*100:>7.2f}% | {m['px'][name]:>7.2f}px")
        print("-" * 42)
        print(f"{'TỔNG THỂ':<14} | {m['nme']*100:>7.2f}% | {m['overall_px']:>7.2f}px")
        print(f"• Mẫu tệ nhất : #{m['worst_sample']} | P{m['worst_pt']} lệch {m['worst_px']:.1f}px\n")

    def compute_biometric_metrics(preds, lms):
        pt_t = lms.reshape(-1, NUM_LANDMARKS, 2)
        pt_p = preds.reshape(-1, NUM_LANDMARKS, 2)

        def _calc_ear(pts):
            def _eye(idx):
                h1 = np.linalg.norm(pts[:, idx[1]] - pts[:, idx[5]], axis=-1)
                h2 = np.linalg.norm(pts[:, idx[2]] - pts[:, idx[4]], axis=-1)
                w = np.linalg.norm(pts[:, idx[0]] - pts[:, idx[3]], axis=-1) + 1e-6
                return (h1 + h2) / (2.0 * w)
            return (_eye([0, 1, 2, 3, 4, 5]) + _eye([6, 7, 8, 9, 10, 11])) / 2.0

        def _calc_mar(pts):
            w = np.linalg.norm(pts[:, 12] - pts[:, 13], axis=-1) + 1e-6
            h_out = np.linalg.norm(pts[:, 14] - pts[:, 15], axis=-1)
            h_in = np.linalg.norm(pts[:, 16] - pts[:, 17], axis=-1)
            return (h_out + h_in) / (2.0 * w)

        ear_t = _calc_ear(pt_t)
        ear_p = _calc_ear(pt_p)
        mar_t = _calc_mar(pt_t)
        mar_p = _calc_mar(pt_p)

        ear_mae = float(np.mean(np.abs(ear_t - ear_p)))
        mar_mae = float(np.mean(np.abs(mar_t - mar_p)))

        closed_idx = np.where(ear_t < 0.20)[0]
        yawn_idx = np.where(mar_t >= 0.40)[0]
        normal_idx = np.where((ear_t >= 0.20) & (mar_t < 0.40))[0]

        return {
            "ear_mae": ear_mae,
            "mar_mae": mar_mae,
            "closed": {
                "count": len(closed_idx),
                "mean_true": float(np.mean(ear_t[closed_idx])) if len(closed_idx) else 0.0,
                "mean_pred": float(np.mean(ear_p[closed_idx])) if len(closed_idx) else 0.0,
                "recall": float(np.mean(ear_p[closed_idx] < 0.22)) * 100.0 if len(closed_idx) else 100.0
            },
            "yawn": {
                "count": len(yawn_idx),
                "mean_true": float(np.mean(mar_t[yawn_idx])) if len(yawn_idx) else 0.0,
                "mean_pred": float(np.mean(mar_p[yawn_idx])) if len(yawn_idx) else 0.0,
                "recall": float(np.mean(mar_p[yawn_idx] >= 0.40)) * 100.0 if len(yawn_idx) else 100.0
            },
            "normal": {
                "count": len(normal_idx),
                "mean_true_ear": float(np.mean(ear_t[normal_idx])) if len(normal_idx) else 0.0,
                "mean_pred_ear": float(np.mean(ear_p[normal_idx])) if len(normal_idx) else 0.0,
                "mean_true_mar": float(np.mean(mar_t[normal_idx])) if len(normal_idx) else 0.0,
                "mean_pred_mar": float(np.mean(mar_p[normal_idx])) if len(normal_idx) else 0.0
            }
        }

    def print_biometrics_table(bio):
        print("--- ĐÁNH GIÁ ĐỘ NHẠY SINH TRẮC HỌC (EAR & MAR BIOMETRIC METRICS) ---")
        print(f"• Sai số tuyệt đối (MAE)        : EAR MAE = {bio['ear_mae']:.4f} | MAR MAE = {bio['mar_mae']:.4f}")
        c = bio['closed']
        print(f"• Mẫu Nhắm mắt (EAR_true < 0.20): {c['count']} mẫu | GT={c['mean_true']:.3f} -> Pred={c['mean_pred']:.3f} | Bắt ngủ gật (EAR<0.22): {c['recall']:.1f}%")
        y = bio['yawn']
        print(f"• Mẫu Ngáp to (MAR_true >= 0.40): {y['count']} mẫu | GT={y['mean_true']:.3f} -> Pred={y['mean_pred']:.3f} | Bắt ngáp (MAR>=0.40): {y['recall']:.1f}%")
        n = bio['normal']
        print(f"• Mẫu Tỉnh táo bình thường      : {n['count']} mẫu | EAR: GT={n['mean_true_ear']:.3f} -> {n['mean_pred_ear']:.3f} | MAR: GT={n['mean_true_mar']:.3f} -> {n['mean_pred_mar']:.3f}")

        if c['count'] > 0 and c['mean_pred'] > 0.28:
            print("  ⚠️ CẢNH BÁO: Mô hình bị kẹt trạng thái mắt mở (Mean-State Collapse trên EAR)!")
        if y['count'] > 0 and y['mean_pred'] < 0.35:
            print("  ⚠️ CẢNH BÁO: Mô hình bị kẹt trạng thái ngậm miệng (Mean-State Collapse trên MAR)!")
        if (c['count'] == 0 or c['mean_pred'] <= 0.25) and (y['count'] == 0 or y['mean_pred'] >= 0.38):
            print("  ✅ Mô hình phản hồi sắc nét, không bị kẹt giá trị trung bình!")
        print("-" * 74 + "\n")

    # 1) Canonical
    preds_c = predict_all(val_imgs)
    m_c = compute_metrics(preds_c, val_lms)
    print_table("NME CANONICAL (template-fit — điểm cao chưa chắc tốt)", m_c)

    # Biometrics on Canonical
    bio_c = compute_biometric_metrics(preds_c, val_lms)
    print_biometrics_table(bio_c)

    # 2) Jittered (localization thật)
    from dataset_loader import make_jittered_val_copy
    jit_imgs, jit_lms = make_jittered_val_copy(val_imgs, val_lms, copies=2, seed=123)
    preds_j = predict_all(jit_imgs)
    m_j = compute_metrics(preds_j, jit_lms)
    print_table("NME JITTER (LOCALIZATION — chỉ số quyết định)", m_j)

    ratio = m_j["nme"] / max(m_c["nme"], 1e-6)
    print("=" * 74)
    print(f"• TỈ LỆ Jitter/Canonical : {ratio:.2f}x "
          f"({'✅ <2.0x — mô hình ĐỊNH VỊ pixel thật' if ratio < 2.0 else '⚠️ >=2.0x — CÒN HỌC TEMPLATE: cần jitter mạnh hơn / thêm dữ liệu'})")

    print("\n" + "=" * 74)
    nme_j = m_j["nme"]
    if nme_j < 0.06:
        verdict = "✅ XUẤT SẮC (<6%) — tự tin nạp ESP32-S3!"
        code = 0
    elif nme_j < 0.08:
        verdict = "✅ ĐẠT (<8%) — được phép nạp ESP32-S3."
        code = 0
    elif nme_j < 0.10:
        verdict = "⚠️ TRUNG BÌNH (<10%) — nên bổ sung 300W-LP/WFLW rồi train lại trước khi nạp."
        code = 1
    else:
        verdict = "❌ YẾU (>=10%) — KHÔNG nạp ESP32! Bổ sung dữ liệu + train lại."
        code = 1
    print(f"🏆 KẾT LUẬN (theo NME JITTER): {verdict}")
    print("=" * 74)
    sys.exit(code)


if __name__ == "__main__":
    main()
