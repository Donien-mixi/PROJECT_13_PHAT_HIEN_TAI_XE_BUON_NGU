"""Board (ESP32) vs laptop: same 12 still images, same ROI, two model stacks.

Board  : BlazeFace INT8 (TFLM) -> ROI -> PIPNet-98 INT8 (ESP-DL, 7 parts)
Laptop : BlazeFace INT8 (same .tflite) -> ROI -> PIPNet-98 float ONNX

Reads the board log (MULTIID / MULTIIDPTS lines) and the frames dumped from the header,
re-runs the laptop stack with the SAME crop policy (side*1.5, nearest, clamped), then reports:
  * detector: score delta, box IoU (board box reconstructed from ROI + side)
  * landmarks: per-point L2 error (mean / p95 / max, in QVGA pixels)
  * EAR: board vs laptop
Aggregate + per-identity table + an overlay figure.

Usage:
  py tools\\board_vs_laptop_stills.py --log output/blaze_iq_20261113/serial_multiid_pts.log \
      --frames output/multiid_from_header_20261114 --out output/board_vs_laptop_20261114
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL = os.path.join(ROOT, r"output\port_models_20261110\blaze_iq_mp\blazeface_nchw_full_integer_quant.tflite")
ONNX_98 = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
W, H = 320, 240

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"MULTIID: (\d+) \[([^\]]+)\] score=([\d.]+) side=([\d.]+) roi=(\d+),(\d+),(\d+) ear=([\d.]+)")
PTS = re.compile(r"MULTIIDPTS: (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)")


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


class Laptop:
    def __init__(self):
        self.it = tf.lite.Interpreter(model_path=TFL)
        self.it.allocate_tensors()
        self.det = self.it.get_input_details()[0]
        self.outs = {int(t["shape"][-1]): t for t in self.it.get_output_details()}
        self.anc = anchors()
        self.s98 = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])

    def detect(self, gray):
        qs, qz = self.det["quantization"]
        s = 128.0 / max(W, H)
        nw, nh = int(round(W * s)), int(round(H * s))
        im = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
        c = np.zeros((128, 128), np.uint8)
        c[(128 - nh) // 2:(128 - nh) // 2 + nh, (128 - nw) // 2:(128 - nw) // 2 + nw] = im
        x = c.astype(np.float32) / 127.5 - 1.0
        q = np.clip(np.rint(x / qs) + qz, -128, 127).astype(np.int8)
        self.it.set_tensor(self.det["index"], np.stack([q] * 3, -1)[None])
        self.it.invoke()
        c1, r1 = self.outs[1], self.outs[16]
        cc = (self.it.get_tensor(c1["index"]).reshape(-1).astype(np.float32) - c1["quantization"][1]) * c1["quantization"][0]
        rr = (self.it.get_tensor(r1["index"])[0].astype(np.float32) - r1["quantization"][1]) * r1["quantization"][0]
        sc = 1.0 / (1.0 + np.exp(-np.clip(cc, -60, 60)))
        i = int(np.argmax(sc))
        padx, pady = (128 - nw) // 2, (128 - nh) // 2
        cx = (rr[i, 0] + self.anc[i, 0] * 128 - padx) / s
        cy = (rr[i, 1] + self.anc[i, 1] * 128 - pady) / s
        w = rr[i, 2] / s
        h = rr[i, 3] / s
        return float(sc[i]), np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def landmarks(self, gray, roi):
        """Same crop as firmware face_espdl_landmarks(): 1.5x ROI, nearest, q = p - 128."""
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
        cls, xo, yo, _, _ = self.s98.run(None, {"input": blob})
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


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-6 else 0.0


def parse_board(log):
    boxes, pts = {}, {}
    for line in open(log, encoding="utf-8", errors="replace"):
        L = ANSI.sub("", line)
        m = LINE.search(L)
        if m:
            i = int(m.group(1))
            boxes[i] = {"name": m.group(2), "score": float(m.group(3)), "side": float(m.group(4)),
                        "roi": (int(m.group(5)), int(m.group(6)), int(m.group(7))),
                        "ear": float(m.group(8))}
            continue
        m = PTS.search(L)
        if m:
            pts.setdefault(int(m.group(1)), np.zeros((98, 2), np.float32))[int(m.group(2))] = \
                (float(m.group(3)), float(m.group(4)))
    return boxes, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    board_boxes, board_pts = parse_board(a.log)
    lap = Laptop()
    rows, errors = [], []
    for i in sorted(board_boxes):
        f = os.path.join(a.frames, f"frame_{i:02d}.png")
        if not os.path.exists(f):
            continue
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        b = board_boxes[i]
        roi = b["roi"]
        side = b["side"]
        cs = round(1.08 * side)
        # board box reconstructed from the logged ROI (td_crop_from_box is invertible)
        bcx = roi[0] + cs / 2 - 0.040 * side
        bcy = roi[1] + cs / 2 + 0.028 * side
        bbox = np.array([bcx - side / 2, bcy - side / 2, bcx + side / 2, bcy + side / 2])

        lsc, lbox = lap.detect(gray)
        lpts = lap.landmarks(gray, roi)
        bpts = board_pts.get(i)
        if bpts is None:
            continue
        d = np.linalg.norm(bpts - lpts, axis=1)
        ear_b, ear_l = ear98(bpts), ear98(lpts)
        rows.append({"i": i, "name": b["name"], "score_board": b["score"], "score_laptop": round(lsc, 3),
                     "box_iou": round(iou(bbox, lbox), 3),
                     "L2_mean_px": round(float(d.mean()), 2), "L2_p95_px": round(float(np.percentile(d, 95)), 2),
                     "L2_max_px": round(float(d.max()), 2),
                     "ear_board": round(ear_b, 4), "ear_laptop": round(ear_l, 4),
                     "d_ear": round(abs(ear_b - ear_l), 4)})
        errors.append(d)

    allerr = np.concatenate(errors) if errors else np.zeros(1)
    summary = {
        "n_identities": len(rows),
        "detector": {
            "score_delta_median": round(float(np.median([abs(r["score_board"] - r["score_laptop"]) for r in rows])), 3),
            "box_iou_median": round(float(np.median([r["box_iou"] for r in rows])), 3),
            "box_iou_min": round(min(r["box_iou"] for r in rows), 3),
        },
        "landmarks": {
            "L2_mean_px": round(float(allerr.mean()), 2),
            "L2_p95_px": round(float(np.percentile(allerr, 95)), 2),
            "L2_max_px": round(float(allerr.max()), 2),
            "per_image_mean_px_median": round(float(np.median([r["L2_mean_px"] for r in rows])), 2),
        },
        "ear": {"d_ear_median": round(float(np.median([r["d_ear"] for r in rows])), 4),
                "d_ear_max": round(max(r["d_ear"] for r in rows), 4)},
    }
    with open(os.path.join(a.out, "board_vs_laptop.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1)

    print(f"{'#':>2} {'name':<34} {'sc_b':>5} {'sc_l':>5} {'IoU':>5} {'L2mean':>6} {'L2p95':>6} {'EAR_b':>6} {'EAR_l':>6}")
    for r in rows:
        print(f"{r['i']:2d} {r['name'][:34]:<34} {r['score_board']:5.2f} {r['score_laptop']:5.2f} "
              f"{r['box_iou']:5.2f} {r['L2_mean_px']:6.2f} {r['L2_p95_px']:6.2f} {r['ear_board']:6.3f} {r['ear_laptop']:6.3f}")
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", os.path.join(a.out, "board_vs_laptop.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
