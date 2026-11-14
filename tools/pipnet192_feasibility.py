"""Can PIPNet-98 run at 192x192 with the same physical field of view?

PIPNet-98 is fully convolutional (25 Conv + 1 MaxPool, no Gemm/Reshape), so the ONNX can be
re-shaped. At 192 the stride-32 grid becomes 6x6 (cell is still 32 px), so the crop has to be
256/192 = 1.333x wider to keep the same face region — which costs 0.75x sampling resolution.

This script measures that trade-off on the 32 still frames used before:
  baseline : 256 input, crop = 1.5 * ROI side          (what the firmware runs today)
  variant  : 192 input, crop = 1.5 * ROI * 4/3
Reports landmark L2 error (px) and EAR delta, so the speed/accuracy decision is made on numbers.

Usage: py tools\\pipnet192_feasibility.py
"""
import glob
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
OUT = os.path.join(ROOT, r"output\pipnet192_20261114")
BATCHES = [
    (r"output\blaze_iq_20261113\serial_multiid_pts.log", r"output\multiid_from_header_20261114"),
    (r"output\blaze_iq_20261113\serial_multiid_b2.log", r"output\multiid_b2_frames"),
    (r"output\blaze_iq_20261113\serial_multiid_b3.log", r"output\multiid_b3_frames"),
]
W, H = 320, 240
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"MULTIID: (\d+) \[([^\]]+)\].*side=([\d.]+) roi=(\d+),(\d+),(\d+)")


def make_192_model():
    """Re-shape the input to 192 and let ONNX infer every intermediate shape."""
    dst = os.path.join(OUT, "pipnet98_resnet18_192.onnx")
    m = onnx.load(ONNX)
    dim = m.graph.input[0].type.tensor_type.shape.dim
    dim[2].dim_value = 192
    dim[3].dim_value = 192
    # wipe stale inferred shapes so shape inference recomputes them from the new input
    del m.graph.value_info[:]
    m2 = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    onnx.save(m2, dst)
    shapes = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
              for o in m2.graph.output]
    return dst, shapes


def parse_roi(log):
    out = {}
    for line in open(log, encoding="utf-8", errors="replace"):
        L = ANSI.sub("", line)
        m = LINE.search(L)
        if m:
            out[int(m.group(1))] = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
    return out


def crop256(gray, roi, scale, in_size):
    """Firmware crop: centre 1.5x ROI (times `scale`), nearest, then to `in_size`."""
    x0, y0, side = roi
    cx, cy = x0 + side * 0.5, y0 + side * 0.5
    s = min(side * 1.5 * scale, float(min(W, H)))
    x1 = max(0.0, min(cx - s / 2, W - s))
    y1 = max(0.0, min(cy - s / 2, H - s))
    step = s / float(in_size)
    yy = np.clip((y1 + (np.arange(in_size) + 0.5) * step + 0.5).astype(int), 0, H - 1)
    xx = np.clip((x1 + (np.arange(in_size) + 0.5) * step + 0.5).astype(int), 0, W - 1)
    return gray[np.ix_(yy, xx)], (x1, y1, s)


def run(sess, crop, in_size, x1, y1, s):
    rgb = np.stack([crop] * 3, -1).astype(np.float32)
    blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
    cls, xo, yo, _, _ = sess.run(None, {"input": blob})
    G = cls.shape[-1]
    cells = G * G
    cls = cls.reshape(98, cells)
    xo = xo.reshape(98, cells)
    yo = yo.reshape(98, cells)
    ids = cls.argmax(1)
    nx = ((ids % G) + xo[np.arange(98), ids]) / G
    ny = ((ids // G) + yo[np.arange(98), ids]) / G
    return np.stack([x1 + nx * s, y1 + ny * s], 1)


def ear98(p):
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    D = d(60, 68)
    if D < 1e-4:
        return 0.0
    return float(((d(61, 66) + d(62, 65)) * 0.5 / D + (d(69, 74) + d(70, 73)) * 0.5 / D) * 0.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    m192, shapes = make_192_model()
    print("192 model outputs:", shapes)
    s256 = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    s192 = ort.InferenceSession(m192, providers=["CPUExecutionProvider"])

    rows = []
    for log, frames in BATCHES:
        rois = parse_roi(os.path.join(ROOT, log))
        for i, roi in sorted(rois.items()):
            f = os.path.join(ROOT, frames, f"frame_{i:02d}.png")
            if not os.path.exists(f):
                continue
            gray = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
            c256, (x1, y1, s) = crop256(gray, roi, 1.0, 256)
            p256 = run(s256, c256, 256, x1, y1, s)
            c192, (x1b, y1b, s192c) = crop256(gray, roi, 256.0 / 192.0, 192)
            p192 = run(s192, c192, 192, x1b, y1b, s192c)
            d = np.linalg.norm(p256 - p192, axis=1)
            rows.append({"file": os.path.basename(frames) + f"/{i}", "L2_mean": round(float(d.mean()), 2),
                         "L2_p95": round(float(np.percentile(d, 95)), 2), "L2_max": round(float(d.max()), 2),
                         "ear256": round(ear98(p256), 4), "ear192": round(ear98(p192), 4),
                         "d_ear": round(abs(ear98(p256) - ear98(p192)), 4)})

    L = np.array([r["L2_mean"] for r in rows]); E = np.array([r["d_ear"] for r in rows])
    summary = {"n": len(rows), "L2_mean_of_means": round(float(L.mean()), 2),
               "L2_median": round(float(np.median(L)), 2), "L2_worst": round(float(L.max()), 2),
               "d_ear_median": round(float(np.median(E)), 4), "d_ear_max": round(float(E.max()), 4)}
    with open(os.path.join(OUT, "feasibility.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    worst = sorted(rows, key=lambda r: -r["L2_mean"])[:3]
    print("worst:", [(w["file"], w["L2_mean"], w["d_ear"]) for w in worst])
    print("saved", os.path.join(OUT, "feasibility.json"))


if __name__ == "__main__":
    main()
