"""Sweep the ROI scale policy (k in td_crop_from_box) on the 40 frames captured from the board.

Question: at close range (detector box ~145 px) PIPNet-98 landmark quality collapsed
(EAR up to 2.74, bbox 1.67x the detector box). Which k restores sane landmarks?

For every frame and every k:
  detector (INT8 tflite, firmware preprocessing) -> box -> ROI (k, centre offsets)
  -> PIPNet crop (1.5x ROI, nearest) -> 98 points -> EAR + bbox ratio vs detector box
Reports per k: EAR median/range, bbox ratio median, and how many frames are "plausible"
(EAR in 0.05..0.45 and 0.6 <= bbox/det <= 1.4).

Usage:
  py tools\\roi_scale_sweep.py --frames output/calib_frames_20261113 --k 1.08 1.20 1.35 1.50
"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL_INT8 = os.path.join(ROOT, r"output\port_models_20261110\blaze_iq\blazeface_nchw_full_integer_quant.tflite")
ONNX_98 = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
SIZE = 128
W, H = 320, 240


def anchors():
    out = []
    for y in range(16):
        for x in range(16):
            cx, cy = (x + 0.5) / 16.0, (y + 0.5) / 16.0
            out += [(cx, cy), (cx, cy)]
    for y in range(8):
        for x in range(8):
            cx, cy = (x + 0.5) / 8.0, (y + 0.5) / 8.0
            out += [(cx, cy)] * 6
    return np.asarray(out, dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def detect(interp, det, outs, anc, gray):
    qs, qz = det["quantization"]
    s = float(SIZE) / float(max(W, H))
    nw, nh = int(round(W * s)), int(round(H * s))
    im = Image.fromarray(gray).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    q = np.clip(np.rint(x / qs) + qz, -128, 127).astype(np.int8)
    interp.set_tensor(det["index"], np.stack([q] * 3, -1)[None])
    interp.invoke()
    c = (interp.get_tensor(outs[1]["index"]).reshape(-1).astype(np.float32) - outs[1]["quantization"][1]) * outs[1]["quantization"][0]
    r = (interp.get_tensor(outs[16]["index"])[0].astype(np.float32) - outs[16]["quantization"][1]) * outs[16]["quantization"][0]
    sc = sigmoid(c)
    idx = np.where(sc >= 0.5)[0]
    if idx.size == 0:
        return 0.0, None
    padx, pady = (SIZE - nw) // 2, (SIZE - nh) // 2
    cx = (r[idx, 0] + anc[idx, 0] * SIZE - padx) / s
    cy = (r[idx, 1] + anc[idx, 1] * SIZE - pady) / s
    w = r[idx, 2] / s
    h = r[idx, 3] / s
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    j = int(np.argmax(sc[idx]))
    return float(sc[idx][j]), [float(v) for v in b[j]]


def crop_from_box(box, k):
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    sz = max(box[2] - box[0], box[3] - box[1])
    cs = k * sz
    fc = (cx + 0.040 * sz, cy - 0.028 * sz)
    side = float(cs) * 1.5
    if side > W: side = float(W)
    if side > H: side = float(H)
    x0 = max(0.0, min(fc[0] - side / 2, W - side))
    y0 = max(0.0, min(fc[1] - side / 2, H - side))
    return x0, y0, side, cs


def run98(sess, gray, x0, y0, side):
    step = side / 256.0
    yy = np.clip((y0 + (np.arange(256) + 0.5) * step).astype(int), 0, H - 1)
    xx = np.clip((x0 + (np.arange(256) + 0.5) * step).astype(int), 0, W - 1)
    crop = gray[np.ix_(yy, xx)]
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
    return np.stack([x0 + nx * side, y0 + ny * side], 1)


def ear98(p):
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    D = d(60, 68)
    if D < 1e-4:
        return 0.0
    return float(((d(61, 66) + d(62, 65)) * 0.5 / D + (d(69, 74) + d(70, 73)) * 0.5 / D) * 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--k", nargs="+", type=float, default=[1.08, 1.20, 1.35, 1.50])
    ap.add_argument("--out", default=os.path.join(ROOT, r"output\multiperson_eval_20261113\roi_scale_sweep.json"))
    a = ap.parse_args()

    interp = tf.lite.Interpreter(model_path=TFL_INT8)
    interp.allocate_tensors()
    det = interp.get_input_details()[0]
    outs = {int(t["shape"][-1]): t for t in interp.get_output_details()}
    anc = anchors()
    s98 = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])

    files = sorted(glob.glob(os.path.join(a.frames, "*.png")))
    res = {k: [] for k in a.k}
    dets = []
    for f in files:
        gray = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
        sc, box = detect(interp, det, outs, anc, gray)
        if box is None:
            dets.append({"file": os.path.basename(f), "score": round(sc, 3), "box": None})
            continue
        dets.append({"file": os.path.basename(f), "score": round(sc, 3), "box": [round(v, 1) for v in box]})
        for k in a.k:
            x0, y0, side, cs = crop_from_box(box, k)
            pts = run98(s98, gray, x0, y0, side)
            e = ear98(pts)
            x1 = pts[:, 0].max() - pts[:, 0].min()
            y1 = pts[:, 1].max() - pts[:, 1].min()
            ratio = float(max(x1, y1) / max(1e-3, max(box[2] - box[0], box[3] - box[1])))
            ok = 0.05 <= e <= 0.45 and 0.60 <= ratio <= 1.40
            res[k].append({"file": os.path.basename(f), "ear": round(e, 4),
                           "bbox_ratio": round(ratio, 3), "plausible": ok,
                           "roi": [round(x0, 1), round(y0, 1), round(side, 1)]})

    summary = {}
    for k in a.k:
        rows = res[k]
        if not rows:
            continue
        ears = [r["ear"] for r in rows]
        ratios = [r["bbox_ratio"] for r in rows]
        summary[f"k={k:.2f}"] = {
            "n": len(rows),
            "ear_median": round(st.median(ears), 3),
            "ear_min": round(min(ears), 3), "ear_max": round(max(ears), 3),
            "ear_in_0.10_0.35": round(sum(1 for e in ears if 0.10 <= e <= 0.35) / len(ears), 3),
            "ratio_median": round(st.median(ratios), 2),
            "plausible_rate": round(sum(1 for r in rows if r["plausible"]) / len(rows), 3),
        }
    out = {"frames": len(files), "detections": dets, "summary": summary, "rows": {str(k): v for k, v in res.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"frames={len(files)} detected={sum(1 for d in dets if d['box'])}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
