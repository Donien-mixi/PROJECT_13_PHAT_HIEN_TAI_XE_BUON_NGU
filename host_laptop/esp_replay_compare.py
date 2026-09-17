"""
ESP32 Replay-Compare (GROUND-TRUTH SO SÁNH ESP32 ↔ LAPTOP)
==========================================================
Mục đích: TRẢ LỜI CHÍNH XÁC câu hỏi "model/firmware trên ESP32 có khớp với
bản chạy trên Laptop không?" bằng cách so sánh **CÙNG MỘT ĐẦU VÀO**.

Cách hoạt động:
  - Lắng nghe UDP :8889 (JSON + gói ảnh 96x96 mà ESP32 gửi lên).
  - Lấy CHÍNH ảnh 96x96 đó, chạy lại model .tflite (cùng file) trên Laptop.
  - So sánh OUTPUT suy luận: 22 landmarks (L2 px), EAR, MAR (ESP32 vs Laptop).

Kết quả đọc:
  - Δ_L2 ≈ 0 (vài phần trăm px)  -> MODEL + TIỀN/HẬU XỬ LÝ firmware ĐÚNG (khác biệt thực tế đến từ crop/cảm biến).
  - Δ_L2 lớn (> ~2 px)          -> có lỗi ở firmware (decode/normalize/quantize/hậu xử lý).

Tuỳ chọn --save-dir: lưu ảnh 96x96 ra PNG để bổ sung dữ liệu miền OV5640 (domain adaptation).

Dùng:
    conda activate projet_13
    python host_laptop/esp_replay_compare.py
    python host_laptop/esp_replay_compare.py --save-dir output\esp_frames
Phím: [ESC]/[Q] thoát  (cần cửa sổ OpenCV).
"""

import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

# --- Chỉ số landmark (giống hệt firmware adas_controller.cpp và host local_model_tester.py) ---
LEFT_EYE = [0, 1, 2, 3, 4, 5]
RIGHT_EYE = [6, 7, 8, 9, 10, 11]
MOUTH = [12, 13, 14, 15, 16, 17]


def compute_ear(pts, eye_idx):
    p = [pts[i] for i in eye_idx]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0


def compute_mar(pts, mouth_idx):
    p = [pts[i] for i in mouth_idx]
    a = np.linalg.norm(p[2] - p[3])
    b = np.linalg.norm(p[4] - p[5])
    c = np.linalg.norm(p[0] - p[1])
    return float((a + b) / (2.0 * c)) if c > 1e-6 else 0.0


class HostModel:
    """Chạy đúng file model .tflite của firmware trên ảnh 96x96 (giống host laptop)."""

    def __init__(self, model_path):
        self.interp = None
        self.in_det = None
        self.out_det = None
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter
            except Exception:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                except Exception as e:
                    print(f"❌ Không có runtime TFLite: {e}")
                    sys.exit(1)
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        print(f"✅ Model: {model_path}")
        print(f"   Input {self.in_det['shape']} {self.in_det['dtype'].__name__} | "
              f"Output {self.out_det['shape']} {self.out_det['dtype'].__name__} | "
              f"quant_in={self.in_det['quantization']}")

    def predict(self, gray96):
        """gray96: uint8 (96,96). Trả (22,2) normalized [0,1] + thời gian ms."""
        t0 = time.perf_counter()
        img = gray96.astype(np.float32)
        norm = (img - 128.0) / 128.0
        if self.in_det['dtype'] in (np.int8, np.uint8):
            scale, zp = self.in_det['quantization']
            if scale == 0.0:
                scale, zp = 1.0 / 128.0, 0
            q = np.clip(np.round(norm / scale) + zp, -128, 127).astype(self.in_det['dtype'])
            inp = q[None, :, :, None]
        else:
            inp = norm[None, :, :, None].astype(np.float32)
        self.interp.set_tensor(self.in_det['index'], inp)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det['index'])[0]
        if self.out_det['dtype'] in (np.int8, np.uint8):
            s, z = self.out_det['quantization']
            if s != 0.0:
                out = (out.astype(np.float32) - z) * s
            else:
                out = (out.astype(np.float32) + 128.0) / 255.0
        out = out.astype(np.float32).reshape((22, 2))
        return np.clip(out, 0.0, 1.0), (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser(description="So sánh suy luận ESP32 vs Laptop trên CÙNG ảnh 96x96")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "models", "tinydriver_model.tflite"))
    ap.add_argument("--save-dir", default=None, help="Lưu ảnh 96x96 ra thư mục này (PNG)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Không thấy model: {args.model}")
        sys.exit(1)

    model = HostModel(args.model)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"💾 Sẽ lưu ảnh vào: {args.save_dir}")

    # [v2] Tự động lưu log ra output/replay_log_<ts>.txt (+ con trỏ latest_replay_log.txt)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, time.strftime("replay_log_%Y%m%d_%H%M%S.txt"))
    log_fp = open(log_path, "w", encoding="utf-8")
    try:
        with open(os.path.join(out_dir, "latest_replay_log.txt"), "w", encoding="utf-8") as p:
            p.write(log_path)
    except Exception:
        pass

    def log(msg):
        print(msg, flush=True)
        try:
            log_fp.write(msg + "\n")
            log_fp.flush()
        except Exception:
            pass

    log(f"# REPLAY-COMPARE log | {time.strftime('%Y-%m-%d %H:%M:%S')} | model={args.model}")
    log("# cot: EAR esp vs host | MAR esp vs host | ΔLM = khoang cach 22 diem (px tren 96x96)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"❌ Không bind được UDP {args.bind}:{args.port} -> {e}")
        sys.exit(1)
    sock.settimeout(0.1)
    print(f"🔎 [Replay-Compare] Nghe UDP {args.bind}:{args.port} ... (ESC/Q thoát)")

    latest_img = None
    latest_full = None
    latest_json = None
    img_seq = 0
    frame_id = 0
    n_match = 0
    n_total = 0

    while True:
        try:
            while True:
                data, _ = sock.recvfrom(65536)
                if len(data) >= 8 and data[:4] == b"\xAA\x56\xAA\x56":
                    w = data[4] | (data[5] << 8)
                    h = data[6] | (data[7] << 8)
                    if 0 < w <= 256 and 0 < h <= 256 and len(data) >= 8 + w * h:
                        latest_img = np.frombuffer(data[8:8 + w * h], dtype=np.uint8).reshape(h, w).copy()
                        img_seq += 1
                elif len(data) >= 8 and data[:4] == b"\xAA\x56\xA0\x01":
                    # [v2] Anh xam TOAN KHUNG (preview 128x96) -> de doi chieu KHUNG CROP nam o dau
                    w = data[4] | (data[5] << 8)
                    h = data[6] | (data[7] << 8)
                    if 0 < w <= 512 and 0 < h <= 512 and len(data) >= 8 + w * h:
                        latest_full = np.frombuffer(data[8:8 + w * h], dtype=np.uint8).reshape(h, w).copy()
                else:
                    try:
                        latest_json = json.loads(data.decode("utf-8", errors="ignore"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
        except socket.timeout:
            pass

        # Khi có cả ảnh + JSON: chạy model trên CHÍNH ảnh ESP32 gửi và so sánh
        if latest_img is not None and latest_json is not None and latest_img.shape == (96, 96):
            esp_lm = np.array(latest_json.get("landmarks", []), dtype=np.float32)
            if esp_lm.size == 44 and img_seq != frame_id:
                frame_id = img_seq
                host_lm, ai_ms = model.predict(latest_img)

                d = np.linalg.norm((esp_lm.reshape(22, 2) - host_lm) * 96.0, axis=1)
                d_mean, d_max = float(d.mean()), float(d.max())

                e_esp = float(latest_json.get("ear", 0.0))
                e_host = (compute_ear(host_lm, LEFT_EYE) + compute_ear(host_lm, RIGHT_EYE)) / 2.0
                m_esp = float(latest_json.get("mar", 0.0))
                m_host = compute_mar(host_lm, MOUTH)

                n_total += 1
                ok = d_mean < 1.0            # <1px coi như khớp hoàn hảo
                n_match += 1 if ok else 0
                verdict = "KHỚP ✅" if ok else "LỆCH ❌"
                rx = int(latest_json.get("rx", 0)); ry = int(latest_json.get("ry", 0))
                rs = int(latest_json.get("rs", 0)); roi_on = int(latest_json.get("roi", 0))

                log(f"[REPLAY] #{n_total:04d} | AI_host {ai_ms:5.1f}ms | "
                      f"EAR esp {e_esp:.3f} vs host {e_host:.3f} (Δ{e_esp - e_host:+.3f}) | "
                      f"MAR esp {m_esp:.3f} vs host {m_host:.3f} (Δ{m_esp - m_host:+.3f}) | "
                      f"ΔLM mean {d_mean:.2f}px max {d_max:.2f}px | "
                      f"ROI({roi_on}) {rx},{ry},{rs} | {verdict} | "
                      f"khớp {n_match}/{n_total}")

                if args.save_dir:
                    fn = os.path.join(args.save_dir, f"esp_{n_total:05d}.png")
                    cv2.imwrite(fn, latest_img)
                    if latest_full is not None:
                        big = cv2.resize(latest_full, (320, 240), interpolation=cv2.INTER_NEAREST)
                        if rs > 0:
                            cv2.rectangle(big, (rx, ry), (rx + rs, ry + rs), (0, 0, 255), 1)
                        cv2.imwrite(os.path.join(args.save_dir, f"full_{n_total:05d}.png"), big)

        # Thoát nếu bấm phím (nếu môi trường có GUI)
        try:
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
                break
        except cv2.error:
            pass

    sock.close()


if __name__ == "__main__":
    main()
