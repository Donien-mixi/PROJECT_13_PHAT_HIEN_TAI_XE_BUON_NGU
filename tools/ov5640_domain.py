"""OV5640 QVGA domain adapter (Project 13).

The datasets in datasets/ (YawDD dash-cam, CEW, 300W_LP) are NOT what the ESP32 sees:
the board gets 320x240 GRAY frames from an OV5640 behind a wide lens, with sensor noise,
soft focus, vignetting and JPEG-ish blocking. This module

  1. measures those statistics from REAL board frames (output/calib_frames_20261113/*.png),
  2. converts any dataset image into the same domain (QVGA, grey, face ~= target px,
     matched contrast/noise/blur/vignette).

Usage:
  py tools\\ov5640_domain.py --measure                 # print the measured domain stats
  py tools\\ov5640_domain.py --convert <img> --out x.png --face-px 98
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD_FRAMES = os.path.join(ROOT, r"output\calib_frames_20261113")
STATS_JSON = os.path.join(ROOT, r"output\ov5640_domain_20261114\stats.json")
W, H = 320, 240


# ---------------------------------------------------------------- measuring
def frame_stats(gray):
    """Noise / sharpness / contrast / vignette signature of one board frame."""
    g = gray.astype(np.float32)
    # noise: median absolute deviation of the Laplacian (robust to edges)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    noise = float(np.median(np.abs(lap - np.median(lap))) * 1.4826)
    # sharpness: 95th pct of |gradient| (edges), normalised by contrast
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    sharp = float(np.percentile(grad, 95))
    # local contrast: std inside 16x16 blocks (texture level)
    blk = g[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    local = float(np.median(blk.std(axis=(1, 3))))
    # vignette: mean of border ring vs centre disc
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    centre = float(g[r < 0.5].mean())
    border = float(g[r > 1.1].mean()) if (r > 1.1).any() else centre
    return {"mean": float(g.mean()), "std": float(g.std()), "noise": noise,
            "sharp": sharp, "local_std": local,
            "vignette": round(border / max(1.0, centre), 3)}


def measure():
    files = sorted(glob.glob(os.path.join(BOARD_FRAMES, "frame_*.png")))
    rows = [frame_stats(np.asarray(cv2.imread(f, cv2.IMREAD_GRAYSCALE))) for f in files]
    keys = rows[0].keys()
    agg = {k: round(float(np.median([r[k] for r in rows])), 3) for k in keys}
    agg["n_frames"] = len(rows)
    agg["source"] = BOARD_FRAMES
    os.makedirs(os.path.dirname(STATS_JSON), exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"aggregate": agg, "per_frame": rows}, fh, indent=1)
    print(json.dumps(agg, indent=1))
    return agg


# ---------------------------------------------------------------- converting
def load_stats():
    if not os.path.exists(STATS_JSON):
        return measure()
    with open(STATS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["aggregate"]


def to_ov5640(img, stats, face_px=98, face_frac=1.0, seed=None):
    """Dataset image -> OV5640-like 320x240 grey frame with the face ~face_px wide.

    face_frac = face size / min(image side) of the source (0.70 for 300W_LP crops,
                1.0 for CEW face crops, 0.30 for YawDD dash frames).
    """
    rng = np.random.default_rng(seed)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    s = float(face_px) / max(1.0, face_frac * float(min(h, w)))
    nw, nh = max(16, int(round(w * s))), max(16, int(round(h * s)))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if face_frac >= 0.99:
        # CEW-style face crop: scale it to COVER the QVGA frame (like a face at ~40 cm),
        # so there is no artificial rectangle border
        cov = max(W / float(w), H / float(h))
        cw2, ch2 = int(round(w * cov)) + 2, int(round(h * cov)) + 2
        big = cv2.resize(img, (cw2, ch2), interpolation=cv2.INTER_LINEAR)
        x0 = (cw2 - W) // 2
        y0 = (ch2 - H) // 2
        canvas = big[y0:y0 + H, x0:x0 + W].copy()
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
        g = canvas.astype(np.float32)
        g = g - g.mean() + float(stats["mean"])
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        v = 1.0 - 0.12 * np.clip(r / 1.4, 0, 1) ** 2
        g = g * v + rng.normal(0.0, max(0.8, float(stats["noise"]) * 0.6), g.shape)
        return np.clip(g, 0, 255).astype(np.uint8)

    canvas = np.full((H, W), 45, dtype=np.uint8)          # board background level
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    # paste with clipping: the scaled frame can be larger than the QVGA canvas
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(nw - sx0, W - dx0)
    ch = min(nh - sy0, H - dy0)
    if cw > 0 and ch > 0:
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = im[sy0:sy0 + ch, sx0:sx0 + cw]

    # soft focus like the OV5640 module
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    # match BRIGHTNESS only: stretching a face crop to the board's global std (which is
    # inflated by the bright window in real frames) washed the eyes out
    g = canvas.astype(np.float32)
    g = g - g.mean() + float(stats["mean"])
    # vignetting
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = 1.0 - (1.0 - float(stats["vignette"])) * np.clip(r / 1.4, 0, 1) ** 2
    g = g * v
    # sensor noise (sigma from the measured Laplacian MAD)
    sigma = float(stats["noise"]) * 0.6
    g = g + rng.normal(0.0, max(0.4, sigma), g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--convert")
    ap.add_argument("--out")
    ap.add_argument("--face-px", type=int, default=98)
    ap.add_argument("--face-frac", type=float, default=1.0)
    a = ap.parse_args()
    if a.measure:
        measure()
        return 0
    if a.convert and a.out:
        st = load_stats()
        img = cv2.imread(a.convert, cv2.IMREAD_GRAYSCALE)
        out = to_ov5640(img, st, a.face_px, a.face_frac)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cv2.imwrite(a.out, out)
        print("wrote", a.out, out.shape)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
