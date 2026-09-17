"""
D5-v3 — Dataset FACE DETECTOR nhẹ (96x96 grayscale), QUY ƯỚC ĐỒNG NHẤT
=====================================================================
Vấn đề của v2: 3 nguồn dùng 3 quy ước hộp khác nhau (68-điểm+20% / MediaPipe / BlazeFace)
-> model học "trung bình" các quy ước -> lệch hệ thống trên miền OV5640 (IoU thấp).

GIẢI PHÁP (nhất quán tuyệt đối): dùng **CHÍNH BlazeFace** gán nhãn cho MỌI ảnh
  - BlazeFace là mốc neo on-device -> lite detector học "bắt chước BlazeFace" (distillation).
  - Lấy cả **6 keypoints** (2 mắt, mũi, miệng, 2 tai) -> crop canonical mỗi frame
    bằng ĐÚNG công thức lúc train (isomorphic_transform.compute_canonical_anchor).

Đầu ra: training_tinyml/face_detection_dataset.npz
  images (N,96,96) uint8 ; box (N,4) [cx,cy,w,h] norm ; keypoints (N,12) norm
  obj (N,) uint8 ; source (N,) <U16 ; split (N,) uint8 (0=train,1=val,2=val_ov)

Dùng:
  python tools/build_face_detection_dataset.py --max-300w 14000 --yawdd-max 2600
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "datasets", "raw_faces")
OUT_DEFAULT = os.path.join(ROOT, "training_tinyml", "face_detection_dataset.npz")
MODEL = os.path.join(ROOT, "PROJECT_5_DIEM_DANH_KHUON_MAT", "host_laptop", "detector",
                     "face_detection_short_range_int8.tflite")
SIZE = 96
SEED = 2026


# ------------------------------------------------------------------ helpers
def _gray96(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


class BlazeLabeler:
    """Dùng BlazeFace (PROJECT_5) gán nhãn: box + 6 keypoints, trên ảnh bất kỳ."""

    def __init__(self, model_path, conf=0.5):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from eval_blazeface_on_ov5640 import BlazeFace  # noqa
        self.det = BlazeFace(model_path, conf=conf)

    def label(self, bgr):
        """Trả (box_xywh, kp6) theo PIXEL ảnh gốc, hoặc None."""
        H, W = bgr.shape[:2]
        S = min(H, W)
        x0, y0 = (W - S) // 2, (H - S) // 2
        sq = bgr[y0:y0 + S, x0:x0 + S]
        g = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY) if sq.ndim == 3 else sq
        g128 = cv2.resize(g, (128, 128), interpolation=cv2.INTER_LINEAR)
        r = self.det.detect(g128)
        if r is None:
            return None
        score, cx, cy, w, h, kps = r
        k = S / 128.0
        box = (x0 + (cx - w / 2.0) * k, y0 + (cy - h / 2.0) * k, w * k, h * k)
        kp6 = [(x0 + kx * k, y0 + ky * k) for (kx, ky) in kps]
        # Hợp lệ: box trong ảnh, keypoints hữu hạn
        if box[2] < 6 or box[3] < 6:
            return None
        for (px, py) in kp6:
            if not (np.isfinite(px) and np.isfinite(py)):
                return None
        return box, kp6


def crop42(bgr, x, y, w, h, rng, kmin=1.6, kmax=3.2, jitter=0.22):
    """Cắt vùng 4:3 quanh hộp mặt -> đồng nhất hình học với khung OV5640."""
    H, W = bgr.shape[:2]
    face = max(w, h)
    for _ in range(8):
        Hc = int(max(48.0, face * rng.uniform(kmin, kmax)))
        Wc = int(Hc * 4.0 / 3.0)
        if Wc > W or Hc > H:
            Hc = min(Hc, H); Wc = int(Hc * 4.0 / 3.0)
        if Wc > W:
            Wc = W; Hc = int(Wc * 3.0 / 4.0)
        if Wc < w * 1.15 or Hc < h * 1.15 or Wc < 32 or Hc < 24:
            continue
        fx, fy = x + w / 2.0, y + h / 2.0
        ccx = min(max(fx + rng.uniform(-jitter, jitter) * Wc, Wc / 2.0), W - Wc / 2.0)
        ccy = min(max(fy + rng.uniform(-jitter, jitter) * Hc, Hc / 2.0), H - Hc / 2.0)
        x0 = int(round(ccx - Wc / 2.0)); y0 = int(round(ccy - Hc / 2.0))
        crop = bgr[y0:y0 + Hc, x0:x0 + Wc]
        if crop.size:
            return crop, (x - x0, y - y0), (x0, y0)
    return None


def _add(out, bgr, box, kp6, source, split, rng, use_crop=True):
    """Thêm 1 mẫu (crop 4:3 hoặc toàn khung) vào out."""
    Hc, Wc = (bgr.shape[0], bgr.shape[1])
    if use_crop:
        rc = crop42(bgr, *box, rng)
        if rc is None:
            return False
        crop, (bx, by), (ox, oy) = rc
        Hc, Wc = crop.shape[:2]
    else:
        crop, bx, by, ox, oy = bgr, box[0], box[1], 0, 0
    out["images"].append(_gray96(crop))
    out["box"].append([(bx + box[2] / 2.0) / Wc, (by + box[3] / 2.0) / Hc,
                       box[2] / Wc, box[3] / Hc])
    flat = []
    for (px, py) in kp6:
        flat += [(px - ox) / Wc, (py - oy) / Hc]   # keypoint -> toạ độ trong crop
    out["kp"].append(flat)
    out["obj"].append(1)
    out["source"].append(source)
    out["split"].append(split)
    return True


def _new():
    return {"images": [], "box": [], "kp": [], "obj": [], "source": [], "split": []}


# ------------------------------------------------------------------ sources
def src_300w(out, lab, max_n, rng):
    pairs = []
    base = os.path.join(RAW, "300W_LP", "300W_LP")
    for sub in sorted(os.listdir(base)):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            for jpg in glob.glob(os.path.join(d, "*.jpg")):
                pairs.append(jpg)
    rng.shuffle(pairs)
    pairs = pairs[:max_n]
    print(f"  300W_LP: thử gán nhãn {len(pairs)} ảnh...")
    ok = 0
    for jpg in pairs:
        bgr = cv2.imread(jpg)
        if bgr is None:
            continue
        r = lab.label(bgr)
        if r is None:
            continue
        if _add(out, bgr, r[0], r[1], "300W_LP", 0, rng):
            ok += 1
    print(f"  -> {ok} mẫu")


def src_yawdd(out, lab, max_frames, rng):
    vids = glob.glob(os.path.join(RAW, "YawDD", "**", "*.avi"), recursive=True)
    print(f"  YawDD: {len(vids)} video")
    per_vid = max(1, max_frames // max(1, len(vids)))
    ok = 0
    for v in vids:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // per_vid) if total > 0 else 25
        taken = 0
        i = 0
        while taken < per_vid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            good, bgr = cap.read()
            if not good:
                break
            i += step
            r = lab.label(bgr)
            if r is None:
                continue
            if _add(out, bgr, r[0], r[1], "YawDD", 0, rng):
                taken += 1; ok += 1
        cap.release()
    print(f"  -> {ok} frame YawDD")


def src_ov5640(out, lab, train_frac, repeat, rng):
    """Ảnh thật ESP32: giữ NGUYÊN toàn khung (đúng như lúc chạy) + nhãn BlazeFace."""
    frames = sorted(glob.glob(os.path.join(ROOT, "output", "esp_frames", "cam*.pgm")))
    print(f"  OV5640: {len(frames)} ảnh")
    if not frames:
        return
    imgs, boxes, kps = [], [], []
    for f in frames:
        g = cv2.imread(f, 0)
        if g is None:
            continue
        H, W = g.shape[:2]
        S = min(H, W)
        x0, y0 = (W - S) // 2, (H - S) // 2
        g128 = cv2.resize(g[y0:y0 + S, x0:x0 + S], (128, 128), interpolation=cv2.INTER_LINEAR)
        r = lab.det.detect(g128)
        if r is None:
            continue
        score, cx, cy, w, h, k6 = r
        k = S / 128.0
        imgs.append(g)
        boxes.append((x0 + (cx - w / 2.0) * k, y0 + (cy - h / 2.0) * k, w * k, h * k))
        kps.append([(x0 + a * k, y0 + b * k) for (a, b) in k6])
    n = len(imgs)
    if n == 0:
        print("  -> 0"); return
    order = list(range(n)); rng.shuffle(order)
    n_tr = int(n * train_frac)
    for j, idx in enumerate(order):
        rep = repeat if j < n_tr else 1
        split = 0 if j < n_tr else 2
        for _ in range(rep):
            _add(out, imgs[idx], boxes[idx], kps[idx], "OV5640", split, rng, use_crop=False)
    print(f"  -> {n_tr} train (x{repeat}) + {n - n_tr} val_ov")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="D5-v3: dataset face detector (BlazeFace nhất quán)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-300w", type=int, default=14000)
    ap.add_argument("--yawdd-max", type=int, default=2600)
    ap.add_argument("--conf", type=float, default=0.5, help="ngưỡng conf BlazeFace khi gán nhãn")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--ov5640-train-frac", type=float, default=0.75)
    ap.add_argument("--ov5640-repeat", type=int, default=4)
    ap.add_argument("--preview", default=os.path.join(ROOT, "output", "fd_preview.png"))
    args = ap.parse_args()

    if not os.path.exists(MODEL):
        print(f"[LOI] Thiếu model BlazeFace: {MODEL}"); sys.exit(1)

    rng = random.Random(SEED)
    lab = BlazeLabeler(MODEL, conf=args.conf)
    print("=== D5-v3: dataset (nhãn BlazeFace nhất quán) ===")

    out = _new()
    src_300w(out, lab, args.max_300w, rng)
    src_yawdd(out, lab, args.yawdd_max, rng)

    n_main = len(out["obj"])
    src_ov5640(out, lab, args.ov5640_train_frac, args.ov5640_repeat, rng)

    # Chia val trên nhóm chính (bỏ qua OV5640 - đã có split riêng)
    idx = list(range(n_main)); rng.shuffle(idx)
    for i in idx[:int(n_main * args.val_frac)]:
        out["split"][i] = 1

    images = np.stack(out["images"]).astype(np.uint8)
    box = np.asarray(out["box"], np.float32)
    kp = np.asarray(out["kp"], np.float32)
    obj = np.asarray(out["obj"], np.uint8)
    source = np.asarray(out["source"], dtype="<U16")
    split = np.asarray(out["split"], np.uint8)
    np.savez_compressed(args.out, images=images, box=box, keypoints=kp,
                        obj=obj, source=source, split=split)

    print(f"\n✅ Đã lưu: {args.out}")
    print(f"   Tổng {len(images)} | train={int((split==0).sum())} val={int((split==1).sum())} val_ov={int((split==2).sum())}")
    for s in np.unique(source):
        print(f"   - {s}: {int((source==s).sum())}")

    if args.preview:
        tiles = []
        picks = [i for i in range(len(images)) if obj[i] == 1][:12]
        for i in picks:
            t = cv2.resize(images[i], (192, 192), interpolation=cv2.INTER_NEAREST)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cx, cy, w, h = box[i]
            cv2.rectangle(t, (int((cx - w / 2) * 192), int((cy - h / 2) * 192)),
                          (int((cx + w / 2) * 192), int((cy + h / 2) * 192)), (0, 255, 0), 2)
            for j in range(6):
                px, py = kp[i][j * 2] * 192, kp[i][j * 2 + 1] * 192
                cv2.circle(t, (int(px), int(py)), 3, (0, 200, 255), -1)
            tiles.append(t)
        rows = [np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows]
        cv2.imwrite(args.preview, np.vstack(rows))
        print(f"   🖼️  Preview: {args.preview}")


if __name__ == "__main__":
    main()
