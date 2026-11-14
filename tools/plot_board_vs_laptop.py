"""Overlay figure: ESP32 landmarks vs laptop landmarks on the same still frames.

Usage: py tools\\plot_board_vs_laptop.py --log <serial_multiid_pts.log> \
         --frames output/multiid_from_header_20261114 \
         --json output/board_vs_laptop_20261114/board_vs_laptop.json \
         --out output/board_vs_laptop_20261114/overlay.png --ids 4 11
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_98 = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
W, H = 320, 240
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"MULTIID: (\d+) \[([^\]]+)\].*side=([\d.]+) roi=(\d+),(\d+),(\d+)")
PTS = re.compile(r"MULTIIDPTS: (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)")


def parse(log):
    boxes, pts = {}, {}
    for line in open(log, encoding="utf-8", errors="replace"):
        L = ANSI.sub("", line)
        m = LINE.search(L)
        if m:
            i = int(m.group(1))
            boxes[i] = {"name": m.group(2), "side": float(m.group(3)),
                        "roi": (int(m.group(4)), int(m.group(5)), int(m.group(6)))}
            continue
        m = PTS.search(L)
        if m:
            pts.setdefault(int(m.group(1)), np.zeros((98, 2), np.float32))[int(m.group(2))] = \
                (float(m.group(3)), float(m.group(4)))
    return boxes, pts


def laptop_landmarks(sess, gray, roi):
    x0, y0, side = roi
    cx, cy = x0 + side * 0.5, y0 + side * 0.5
    s = min(side * 1.5, float(min(W, H)))
    x1 = max(0.0, min(cx - s / 2, W - s))
    y1 = max(0.0, min(cy - s / 2, H - s))
    step = s / 256.0
    yy = np.clip((y1 + (np.arange(256) + 0.5) * step + 0.5).astype(int), 0, H - 1)
    xx = np.clip((x1 + (np.arange(256) + 0.5) * step + 0.5).astype(int), 0, W - 1)
    crop = gray[np.ix_(yy, xx)]
    rgb = np.stack([crop] * 3, -1).astype(np.float32)
    blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
    cls, xo, yo, _, _ = sess.run(None, {"input": blob})
    G = cls.shape[-1]
    cls = cls.reshape(98, G * G)
    xo = xo.reshape(98, G * G)
    yo = yo.reshape(98, G * G)
    ids = cls.argmax(1)
    nx = ((ids % G) + xo[np.arange(98), ids]) / G
    ny = ((ids // G) + yo[np.arange(98), ids]) / G
    return np.stack([x1 + nx * s, y1 + ny * s], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="+", type=int, default=[4, 11])
    a = ap.parse_args()
    boxes, pts = parse(a.log)
    meta = {r["i"]: r for r in json.load(open(a.json, encoding="utf-8"))["rows"]}
    sess = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])

    n = len(a.ids)
    fig, axs = plt.subplots(1, n, figsize=(6.0 * n, 5.0))
    if n == 1:
        axs = [axs]
    for ax, idx in zip(axs, a.ids):
        gray = cv2.imread(os.path.join(a.frames, f"frame_{idx:02d}.png"), cv2.IMREAD_GRAYSCALE)
        b = boxes[idx]
        lp = laptop_landmarks(sess, gray, b["roi"])
        bp = pts[idx]
        d = np.linalg.norm(bp - lp, axis=1)
        ax.imshow(gray, cmap="gray")
        ax.plot(bp[:, 0], bp[:, 1], "o", ms=2.6, mfc="none", mec="#00e676", mew=0.9, label="ESP32 (INT8)")
        ax.plot(lp[:, 0], lp[:, 1], "x", ms=3.0, mec="#ff1744", mew=0.9, label="laptop (float)")
        for k in np.argsort(-d)[:3]:
            ax.plot([bp[k, 0], lp[k, 0]], [bp[k, 1], lp[k, 1]], "-", color="#ffea00", lw=0.8)
        m = meta.get(idx, {})
        ax.set_title(f"id{idx:02d}  L2 trung bình {m.get('L2_mean_px', 0):.2f} px · "
                     f"p95 {m.get('L2_p95_px', 0):.2f} px · EAR {m.get('ear_board', 0):.3f} vs "
                     f"{m.get('ear_laptop', 0):.3f}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("98 điểm: ESP32 (INT8, ESP-DL) vs laptop (float ONNX) — cùng ảnh, cùng ROI", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
