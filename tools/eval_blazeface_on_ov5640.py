"""
D1 — Đánh giá BlazeFace INT8 trên ẢNH OV5640 THẬT (offline, không cần mạch)
==========================================================================
Mục đích (theo lộ trình D1 trong KIEN_TRUC_NEN_TANG.md):
  - Kiểm tra BlazeFace có "thấy" mặt trong ảnh camera OV5640 của bạn không.
  - Đo phân bố box (tâm/kích thước) => chọn tham số crop cố định (TD_CROP_*).
  - Xem 6 keypoints có đúng mắt/mũi/miệng (để dùng canonical crop ở D3).
  - Chọn ngưỡng conf trước khi port lên ESP32 (D2).

TÁI HIỆN ĐÚNG THIẾT KẾ:
  - Model + decode giống hệt `PROJECT_5/host_laptop/detector/blazeface_esp32.py`
    và firmware `ai_face_detector.cpp` (anchor 896, sigmoid logits, EMA không dùng khi eval).
  - Input: lấy ĐÚNG vùng mà ESP32 sẽ đưa vào detector = crop vuông GIỮA khung
    (240x240 của khung 320x240) -> resize 128x128.

KHÔNG đụng TinyDriverNet (không nạp, không sửa model 22 điểm).

Dùng:
    conda activate projet_13
    python tools/eval_blazeface_on_ov5640.py
    python tools/eval_blazeface_on_ov5640.py --conf 0.8 --montage output/blazeface_check.png
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_MODEL = os.path.join(
    ROOT, "PROJECT_5_DIEM_DANH_KHUON_MAT", "host_laptop", "detector",
    "face_detection_short_range_int8.tflite")

# Kích thước khung gốc / preview (theo telemetry_sender_preview: 128x96 từ 320x240)
FULL_W, FULL_H = 320, 240
PREV_W, PREV_H = 128, 96


def load_interp(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=model_path)
    it.allocate_tensors()
    return it


class BlazeFace:
    """Bản sao trung thực của emulator project 5 / firmware ai_face_detector.cpp."""

    def __init__(self, model_path, conf=0.80):
        self.input_size = 128
        self.conf = conf
        self.it = load_interp(model_path)
        self.in_det = self.it.get_input_details()[0]
        self.out_det = self.it.get_output_details()
        self.anchors = self._gen_anchors()

    @staticmethod
    def _gen_anchors():
        a = []
        for y in range(16):
            for x in range(16):
                cx, cy = (x + 0.5) / 16.0, (y + 0.5) / 16.0
                a.append((cx, cy)); a.append((cx, cy))
        for y in range(8):
            for x in range(8):
                cx, cy = (x + 0.5) / 8.0, (y + 0.5) / 8.0
                for _ in range(6):
                    a.append((cx, cy))
        return a  # 896

    @staticmethod
    def _sig(x):
        x = np.clip(x, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _deq(t, d):
        if np.issubdtype(d['dtype'], np.integer):
            s, z = d.get('quantization', (0.0, 0))
            if s:
                return (t.astype(np.float32) - z) * s
        return t.astype(np.float32, copy=False)

    def detect(self, gray128):
        """gray128: uint8 (128,128) đúng vùng detector nhìn (đã là crop vuông giữa khung).
        Trả (score, cx, cy, w, h, kps6) trong toạ độ 128, hoặc None."""
        rgb = cv2.cvtColor(gray128, cv2.COLOR_GRAY2RGB)
        norm = (rgb.astype(np.float32) - 127.5) / 128.0
        s = self.in_det['quantization'][0] or 1.0
        z = self.in_det['quantization'][1]
        inp = np.clip(np.round(norm / s) + z, -128, 127).astype(self.in_det['dtype'])
        self.it.set_tensor(self.in_det['index'], np.expand_dims(inp, 0))
        self.it.invoke()

        o0d, o1d = self.out_det[0], self.out_det[1]
        o0 = self.it.get_tensor(o0d['index']); o1 = self.it.get_tensor(o1d['index'])
        if o0.shape[-1] == 1:
            scores = self._deq(o0[0], o0d); boxes = self._deq(o1[0], o1d)
        else:
            scores = self._deq(o1[0], o1d); boxes = self._deq(o0[0], o0d)

        sv = self._sig(np.asarray(scores, np.float32).reshape(-1))
        best = int(np.argmax(sv)); score = float(sv[best])
        if score < self.conf:
            return None
        row = np.asarray(boxes, np.float32)[best]
        ax, ay = self.anchors[best]
        cx = float(row[0] + ax * 128.0); cy = float(row[1] + ay * 128.0)
        w = float(row[2]); h = float(row[3])
        if not np.isfinite([cx, cy, w, h]).all() or w <= 0 or h <= 0:
            return None
        # 6 keypoints (nếu model có) — offsets pixel so với anchor
        kps = []
        for i in range(6):
            kx = float(row[4 + i * 2] + ax * 128.0)
            ky = float(row[5 + i * 2] + ay * 128.0)
            kps.append((kx, ky))
        return score, cx, cy, w, h, kps


def prev_to_det(prev_gray):
    """Ảnh preview 128x96 (toàn khung) -> ảnh 128x128 ĐÚNG vùng detector nhìn (crop vuông giữa)."""
    S = FULL_H                         # 240 = cạnh vuông giữa
    x0 = (FULL_W - S) // 2             # 40
    # preview = toàn khung co 320->128, 240->96  (tỉ lệ 0.4)
    px0 = int(round(x0 * PREV_W / FULL_W))          # 16
    px1 = int(round((x0 + S) * PREV_W / FULL_W))    # 112
    sq = prev_gray[0:PREV_H, px0:px1]               # 96x96
    return cv2.resize(sq, (128, 128), interpolation=cv2.INTER_LINEAR)


def det_to_full(cx, cy, w, h):
    """Toạ độ detector (128) -> toạ độ khung gốc 320x240."""
    S = FULL_H
    x0 = (FULL_W - S) // 2
    k = S / 128.0
    return x0 + cx * k, 0.0 + cy * k, w * k, h * k


def main():
    ap = argparse.ArgumentParser(description="D1: BlazeFace INT8 trên ảnh OV5640 đã lưu")
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "output", "esp_frames"))
    ap.add_argument("--frames-glob", default="cam*.pgm",
                    help="mẫu file ảnh TOÀN KHUNG trong frames-dir (mặc định cam*.pgm)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.80, help="ngưỡng conf (theo project 5 = 0.80)")
    ap.add_argument("--limit", type=int, default=200, help="số frame tối đa đem đánh giá")
    ap.add_argument("--montage", default=os.path.join(ROOT, "output", "blazeface_check.png"))
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[LOI] Khong thay model: {args.model}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.frames_dir, args.frames_glob)))
    if not files:
        print(f"[LOI] Khong co file '{args.frames_glob}' trong: {args.frames_dir}")
        print("      Can anh TOAN KHUNG tu camera. Cach tao:")
        print("        1) Nap firmware moi (co preview): tools\\flash_esp32.ps1 -Port COM3")
        print("        2) python tools\\esp32_log_capture.py --save-frames")
        print("      (khi do se sinh cam*.pgm = toan khung va f*.pgm = 96x96 model)")
        sys.exit(1)
    files = files[: args.limit]

    det = BlazeFace(args.model, conf=args.conf)
    print(f"✅ Model: {os.path.basename(args.model)}")
    print(f"   input {det.in_det['shape']} {det.in_det['dtype'].__name__} quant={det.in_det['quantization']}")
    print(f"   conf ngưỡng = {args.conf} | đánh giá {len(files)} ảnh toàn khung\n")

    n_det = 0
    scores = []
    cs, ss = [], []   # tâm% và size% (theo khung gốc) để gợi ý TD_CROP_*
    eye_pct, nose_pct, size_ratio = [], [], []   # kiểm chứng crop CANONICAL từ 6 keypoints
    tiles = []
    for i, f in enumerate(files):
        g = cv2.imread(f, 0)
        if g is None:
            continue
        g = cv2.resize(g, (PREV_W, PREV_H), interpolation=cv2.INTER_AREA)
        inp128 = prev_to_det(g)
        r = det.detect(inp128)
        if r is None:
            if len(tiles) < 9 and i % max(1, len(files) // 12) == 0:
                tiles.append((os.path.basename(f)[:11], g.copy(), None))
            continue
        score, cx, cy, w, h, kps = r
        n_det += 1
        scores.append(score)
        fx, fy, fw, fh = det_to_full(cx, cy, w, h)
        cs.append((fx / FULL_W * 100.0, fy / FULL_H * 100.0))
        ss.append(max(fw, fh) / FULL_H * 100.0)

        # ---- Canonical anchor từ 6 keypoints (giống isomorphic_transform.compute_canonical_anchor) ----
        S = FULL_H
        x0 = (FULL_W - S) // 2
        k = S / 128.0

        def kp_full(kx, ky):
            return x0 + kx * k, ky * k

        e0 = kp_full(*kps[0]); e1 = kp_full(*kps[1]); nz = kp_full(*kps[2])
        ex, ey = (e0[0] + e1[0]) * 0.5, (e0[1] + e1[1]) * 0.5
        d_eyes = ((e1[0] - e0[0]) ** 2 + (e1[1] - e0[1]) ** 2) ** 0.5
        d_eye_nose = max(nz[1] - ey, 0.45 * d_eyes, 1.0)
        h_skull = max(2.10 * d_eye_nose, 1.40 * d_eyes)
        S_can = 2.05 * h_skull
        cx_can = (ex + nz[0]) * 0.5
        cy_can = ey + 0.32 * h_skull
        top = cy_can - S_can / 2.0
        eye_pct.append((ey - top) / S_can * 100.0)     # kỳ vọng ~34.4%
        nose_pct.append((nz[1] - top) / S_can * 100.0)
        size_ratio.append(S_can / max(fw, fh))

        if len(tiles) < 9 and i % max(1, len(files) // 12) == 0:
            tiles.append((os.path.basename(f)[:11], g.copy(), (fx, fy, fw, fh, kps)))

    rate = 100.0 * n_det / max(1, len(files))
    print("=" * 64)
    print(f"📊 Tỉ lệ phát hiện: {n_det}/{len(files)}  ({rate:.1f}%)  @ conf>={args.conf}")
    if scores:
        print(f"   score : min {min(scores):.2f}  p50 {np.median(scores):.2f}  max {max(scores):.2f}")
        cxs = [c[0] for c in cs]; cys = [c[1] for c in cs]
        print(f"   tâm X : p50 {np.median(cxs):.1f}%   tâm Y: p50 {np.median(cys):.1f}%   (so với khung)")
        print(f"   size  : min {min(ss):.0f}%  p50 {np.median(ss):.0f}%  max {max(ss):.0f}%  (/% cạnh ngắn)")
        print()
        print("   👉 Gợi ý TD_CROP_ để khớp người/vị trí hiện tại:")
        print(f"      TD_CROP_CX_PCT  = {int(round(np.median(cxs)))}")
        print(f"      TD_CROP_CY_PCT  = {int(round(np.median(cys)))}")
        print(f"      TD_CROP_SIZE_PCT= {int(min(100, round(np.median(ss) * 1.25)))}   (thêm ~25% biên)")
        print()
        print("   🔎 Kiểm chứng CROP CANONICAL (từ 6 keypoints):")
        print(f"      mắt ở {np.median(eye_pct):.1f}% chiều cao crop   (kỳ vọng ~34.4%)")
        print(f"      mũi ở {np.median(nose_pct):.1f}% chiều cao crop")
        print(f"      S_canonical / box = {np.median(size_ratio):.2f}   (≈1.3 nếu đúng chuẩn)")
    print("=" * 64)

    if tiles:
        canvas_tiles = []
        sx = 256.0 / PREV_W   # preview -> tile
        sy = 192.0 / PREV_H
        for name, img, box in tiles:
            t = cv2.resize(img, (256, 192), interpolation=cv2.INTER_NEAREST)
            if box is not None:
                fx, fy, fw, fh, _kps = box
                # khung gốc(320x240) -> preview(128x96): nhân 0.4 ; -> tile: nhân sx,sy
                px = (fx - fw / 2.0) * (PREV_W / FULL_W)
                py = (fy - fh / 2.0) * (PREV_H / FULL_H)
                pw = fw * (PREV_W / FULL_W)
                ph = fh * (PREV_H / FULL_H)
                cv2.rectangle(t, (int(px * sx), int(py * sy)),
                              (int((px + pw) * sx), int((py + ph) * sy)), (0, 255, 0), 2)
            cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            canvas_tiles.append(t)
        rows = [np.hstack(canvas_tiles[j:j + 3]) for j in range(0, len(canvas_tiles), 3)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows]
        cv2.imwrite(args.montage, np.vstack(rows))
        print(f"🖼️  Ảnh minh hoạ (box): {args.montage}")


if __name__ == "__main__":
    main()
