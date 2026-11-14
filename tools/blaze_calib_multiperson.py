"""Build a MULTI-PERSON calibration set for BlazeFace INT8 (replaces the single-person set).

Why: the 13/11 quantisation was calibrated on 6 frames of one driver. Measured on 100 other
identities (300W_LP, face ~98 px in 320x240 — the board's working range) the INT8 detector
scored >= 0.60 on only 56 % of them, while the same driver scores 0.855-0.937.

Sources (all real photos, many identities):
  * 300W_LP (AFW/HELEN/LFPW/IBUG)  : diverse identities, backgrounds, poses
  * CEW open/closed                : diverse identities, two eye states
  * the 6 board/laptop frames + 40 frames captured from the board on 13/11 (same domain)

Output: calib_mp128_nhwc.npy / .nchw.npy (values 0..1) + calib_mp_index.json

Usage:
  py tools\\blaze_calib_multiperson.py --per-set 12
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, r"output\port_models_20261110\blaze_calib_mp")
SIZE = 128
W, H = 320, 240


def letterbox(img, fill=0):
    w, h = img.size
    s = float(SIZE) / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (SIZE, SIZE), fill)
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas


def board_like(img, face_frac, face_px=98):
    """Place the photo on a 320x240 canvas with the face ~face_px, then letterbox to 128.

    This reproduces the on-board framing (QVGA, 60-70 cm) instead of feeding a 450 px crop.
    """
    w, h = img.size
    s = float(face_px) / (face_frac * float(min(w, h)))
    nw, nh = max(16, int(round(w * s))), max(16, int(round(h * s)))
    im = img.convert("L").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (W, H), 40)
    canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))
    return canvas


def crop_resize(img, fx, fy, fside):
    w, h = img.size
    side = max(16.0, fside * min(w, h))
    cx, cy = fx * w, fy * h
    x0 = min(max(0.0, cx - side / 2.0), w - side)
    y0 = min(max(0.0, cy - side / 2.0), h - side)
    box = (int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side)))
    return img.crop(box).resize((SIZE, SIZE), Image.BILINEAR)


def bright(img, k):
    return img.point(lambda p: max(0, min(255, int(round(p * k)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=12, help="identities sampled per dataset")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    samples, index = [], []

    def add(tag, img128):
        arr = np.asarray(img128, dtype=np.float32) / 255.0
        samples.append(np.stack([arr] * 3, -1))
        index.append({"src": tag})

    # ---- 1) 300W_LP: diverse identities, real backgrounds ----
    wlp = os.path.join(ROOT, r"datasets\raw_faces\300W_LP\300W_LP")
    per_sub = {}
    for sub in ("AFW", "HELEN", "LFPW", "IBUG"):
        files = glob.glob(os.path.join(wlp, sub, "*.jpg"))
        random.shuffle(files)
        # identity = prefix before the numeric id
        seen = set()
        picked = []
        for f in files:
            parts = os.path.basename(f).split("_")
            # LFPW_image_test_0001_0 / IBUG_image_003_1_0 / HELEN_100032540_1_0
            ident = "_".join(parts[:4]) if parts[0] == "LFPW" else "_".join(parts[:3])
            if ident in seen:
                continue
            seen.add(ident)
            picked.append(f)
            if len(picked) >= a.per_set:
                break
        per_sub[sub] = len(picked)
        for f in picked:
            img = Image.open(f).convert("L")
            base = board_like(img, 0.70)          # 300W_LP crops: face ~0.7 of the frame
            add(f"300W_LP/{sub}/{os.path.basename(f)}", letterbox(base))
            add(f"300W_LP/{sub}/{os.path.basename(f)}#crop0.8", letterbox(crop_resize(base, 0.5, 0.5, 0.8)))
            add(f"300W_LP/{sub}/{os.path.basename(f)}#wide", letterbox(board_like(img, 0.55)))

    # ---- 2) CEW: other identities, open + closed eyes ----
    cew = os.path.join(ROOT, r"datasets\raw_faces\CEW\CEW")
    for sub in ("open", "closed"):
        files = glob.glob(os.path.join(cew, sub, "*.jpg"))
        random.shuffle(files)
        picked, seen = [], set()
        for f in files:
            ident = os.path.basename(f).split("_")[2] if sub == "open" else "_".join(os.path.basename(f).split("_")[3:5])
            if ident in seen:
                continue
            seen.add(ident)
            picked.append(f)
            if len(picked) >= a.per_set:
                break
        for f in picked:
            img = Image.open(f).convert("L")      # 100x100 face crop: face fills the frame
            base = board_like(img, 1.0)
            add(f"CEW/{sub}/{os.path.basename(f)}", letterbox(base))
            add(f"CEW/{sub}/{os.path.basename(f)}#dark", letterbox(bright(base, 0.75)))

    # ---- 3) board / laptop frames (same domain as deployment) ----
    for src in [
        r"output\compare_live_20261112\still_open.png",
        r"output\compare_live_20261112\still_closed.png",
        r"output\compare_live_20261112\board_frame.png",
        r"output\compare_live_20261112\esp32_still_open.png",
        r"output\compare_live_20261112\esp32_still_closed.png",
    ]:
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("L")
        add(src, letterbox(img))
        add(src + "#crop0.8", crop_resize(img, 0.5, 0.5, 0.75))
        add(src + "#mirror", letterbox(img.transpose(Image.FLIP_LEFT_RIGHT)))

    # ---- 4) the 40 frames captured from the board (close range, driver included) ----
    cap = os.path.join(ROOT, r"output\calib_frames_20261113")
    frames = sorted(glob.glob(os.path.join(cap, "frame_*.png")))
    for f in frames[:40]:
        img = Image.open(f).convert("L")
        add("board_capture/" + os.path.basename(f), letterbox(img))
        if len(frames) % 4 == 0:
            add("board_capture/" + os.path.basename(f) + "#crop0.9", crop_resize(img, 0.5, 0.45, 0.9))

    data = np.stack(samples, 0).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "calib_mp128_nhwc.npy"), data)
    np.save(os.path.join(OUT_DIR, "calib_mp128_nchw.npy"), np.transpose(data, (0, 3, 1, 2)).copy())
    with open(os.path.join(OUT_DIR, "calib_mp_index.json"), "w", encoding="utf-8") as fh:
        json.dump({"count": len(index), "size": SIZE, "mean": 0.5, "std": 0.5,
                   "wlp_per_set": per_sub, "samples": index}, fh, indent=1)
    print("samples", data.shape, "->", OUT_DIR)
    print("per -set", per_sub)


if __name__ == "__main__":
    main()
