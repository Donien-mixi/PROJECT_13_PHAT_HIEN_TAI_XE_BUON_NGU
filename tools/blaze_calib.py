"""Build representative calibration data for BlazeFace INT8 (onnx2tf -oiqt).

Sources are real OV5640 / laptop frames captured on 12/11 (compare_live_20261112).
Values are stored in [0, 1] (pixel/255); onnx2tf applies (x - mean) / std, so the
caller passes mean=0.5 std=0.5 to obtain the [-1, 1] range BlazeFace expects.

Outputs (under output/port_models_20261110/blaze_calib/):
  calib_128_nchw.npy  float32 [N,3,128,128]
  calib_128_nhwc.npy  float32 [N,128,128,3]
  calib_index.json    provenance of every sample
"""
import json
import os

import numpy as np
from PIL import Image

SIZE = 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "port_models_20261110", "blaze_calib")

SOURCES = [
    r"output\compare_live_20261112\still_open.png",
    r"output\compare_live_20261112\still_closed.png",
    r"output\compare_live_20261112\board_frame.png",
    r"output\compare_live_20261112\esp32_still_open.png",
    r"output\compare_live_20261112\esp32_still_closed.png",
    r"output\compare_live_20261112\board.jpg",
]


def letterbox(img, fill=0):
    w, h = img.size
    s = float(SIZE) / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (SIZE, SIZE), fill)
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas, s, (SIZE - nw) // 2, (SIZE - nh) // 2


def crop_resize(img, fx, fy, fside):
    """Crop a square window of side fside*min(w,h) centred at (fx*w, fy*h)."""
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
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = []
    index = []
    for src in SOURCES:
        path = os.path.join(ROOT, src)
        if not os.path.exists(path):
            print("skip missing", src)
            continue
        img = Image.open(path).convert("L")
        variants = []
        lb, s, padx, pady = letterbox(img)
        variants.append(("letterbox", lb))
        # scale / offset variants around the face
        for tag, fx, fy, fs in [
            ("c0.90", 0.50, 0.50, 0.90),
            ("c0.75", 0.50, 0.50, 0.75),
            ("c0.60", 0.50, 0.50, 0.60),
            ("offL", 0.38, 0.50, 0.90),
            ("offR", 0.62, 0.50, 0.90),
            ("offU", 0.50, 0.38, 0.90),
            ("offD", 0.50, 0.62, 0.90),
            # background-only corners: teach the classifier its negative end
            ("bgTL", 0.12, 0.12, 0.45),
            ("bgTR", 0.88, 0.12, 0.45),
            ("bgBL", 0.12, 0.88, 0.45),
            ("bgBR", 0.88, 0.88, 0.45),
        ]:
            variants.append((tag, crop_resize(img, fx, fy, fs)))
        for tag, im in (("dark", bright(lb, 0.72)), ("light", bright(lb, 1.28))):
            variants.append((tag, im))
        variants.append(("mirror", lb.transpose(Image.FLIP_LEFT_RIGHT)))

        for tag, im in variants:
            arr = np.asarray(im, dtype=np.float32) / 255.0
            rgb = np.stack([arr, arr, arr], axis=-1)  # HWC
            samples.append(rgb)
            index.append({"src": src, "variant": tag, "letterbox_scale": s,
                          "pad": [padx, pady]})

    data = np.stack(samples, axis=0)  # N,H,W,C in [0,1]
    nhwc = data.astype(np.float32)
    nchw = np.transpose(nhwc, (0, 3, 1, 2)).copy()
    np.save(os.path.join(OUT_DIR, "calib_128_nchw.npy"), nchw)
    np.save(os.path.join(OUT_DIR, "calib_128_nhwc.npy"), nhwc)
    with open(os.path.join(OUT_DIR, "calib_index.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(index), "size": SIZE, "value_range": "[0,1] pix/255",
                   "mean_for_onnx2tf": 0.5, "std_for_onnx2tf": 0.5,
                   "samples": index}, f, indent=1)
    print("samples", nchw.shape, nhwc.shape)
    print("nchw min/max", float(nchw.min()), float(nchw.max()))


if __name__ == "__main__":
    main()
