"""Does the current pipeline work on OTHER drivers' faces? (offline, multi-person)

Runs the exact firmware preprocessing/decoding on public multi-identity data:
  * 300W_LP  (real photos, many identities)      -> BlazeFace INT8 detection rate
  * CEW      (12.9k closed + 14.3k open crops, many identities) -> PIPNet EAR open vs closed

Pipeline replicated from firmware:
  detector: letterbox 128 bilinear, x = p/127.5 - 1, q = round(x/scale)+zp ; 896 anchors ;
            sigmoid ; NMS 0.3 -> top-1
  ROI     : td_crop_from_box(): side = 1.08*box, centre offset (+0.040, -0.028)*box
  PIPNet  : crop 1.5x ROI (clamped), resize 256 nearest -> ONNX float -> decode 5 heads
  EAR     : D = |p60-p68|, left = mean(|p61-p66|,|p62-p65|)/D, right = mean(|p69-p74|,|p70-p73|)/D

Usage:
  py tools\\multiperson_eval.py --n-wlp 120 --n-cew 120
"""
import argparse
import glob
import json
import os
import random
import statistics as st

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL_INT8 = os.environ.get("BLAZE_TFLITE", os.path.join(ROOT, r"output\port_models_20261110\blaze_iq\blazeface_nchw_full_integer_quant.tflite"))
ONNX_98 = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
WLP = os.path.join(ROOT, r"datasets\raw_faces\300W_LP\300W_LP")
CEW = os.path.join(ROOT, r"datasets\raw_faces\CEW\CEW")
OUT = os.path.join(ROOT, r"output\multiperson_eval_20261113")
SIZE = 128
CANVAS_W, CANVAS_H = 320, 240
FACE_TARGET_PX = 100          # board working range (Block B: 98 px)
DET_SCORE = 0.50              # "detected" for the rate report
ACQ_SCORE = 0.60              # firmware acquire gate


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


def letterbox(img):
    w, h = img.size
    s = float(SIZE) / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return np.asarray(canvas, dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def detect(interp, det, anc, gray):
    qs, qz = det["quantization"]
    x = letterbox(gray) / 127.5 - 1.0     # firmware: (pixel/127.5 - 1), pixel 0..255
    q = np.clip(np.rint(x / qs) + qz, -128, 127).astype(np.int8)
    q3 = np.stack([q] * 3, -1)[None]
    interp.set_tensor(det["index"], q3)
    interp.invoke()
    cls_t, reg_t = OUTS[1], OUTS[16]
    c = (interp.get_tensor(cls_t["index"]).reshape(-1).astype(np.float32) - cls_t["quantization"][1]) * cls_t["quantization"][0]
    r = (interp.get_tensor(reg_t["index"])[0].astype(np.float32) - reg_t["quantization"][1]) * reg_t["quantization"][0]
    sc = sigmoid(c)
    idx = np.where(sc >= DET_SCORE)[0]
    if idx.size == 0:
        return 0.0, None, 0
    cx = r[idx, 0] + anc[idx, 0] * SIZE
    cy = r[idx, 1] + anc[idx, 1] * SIZE
    w = r[idx, 2]
    h = r[idx, 3]
    lb = float(SIZE) / float(max(gray.size))
    nw = max(1, int(round(gray.size[0] * lb)))
    nh = max(1, int(round(gray.size[1] * lb)))
    padx, pady = (SIZE - nw) // 2, (SIZE - nh) // 2
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    # back to source pixels
    boxes = (boxes - np.array([padx, pady, padx, pady])) / lb
    sc_i = sc[idx]
    order = np.argsort(-sc_i)
    best = boxes[order[0]]
    keep_side = max(0.0, float(best[2] - best[0]))
    if keep_side < 8.0 or keep_side > 100.0 / lb:   # firmware box-size gate (128-space 8..100)
        return float(sc_i.max()), None, int(idx.size)
    return float(sc_i.max()), [float(v) for v in best], int(idx.size)


def td_crop(box):
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    sz = max(box[2] - box[0], box[3] - box[1])
    cs = 1.08 * sz
    fcx = cx + 0.040 * sz
    fcy = cy - 0.028 * sz
    csi = int(round(cs))
    x0 = int(round(fcx)) - csi // 2
    y0 = int(round(fcy)) - csi // 2
    x0 = max(0, min(x0, CANVAS_W - csi))
    y0 = max(0, min(y0, CANVAS_H - csi))
    return x0, y0, csi


def run98(sess, gray_np, x0, y0, side):
    """Firmware crop: side*1.5 centred on the ROI, nearest sampling, q = p-128."""
    cx, cy = x0 + side * 0.5, y0 + side * 0.5
    s = side * 1.5
    x1 = cx - s * 0.5
    y1 = cy - s * 0.5
    x1 = max(0.0, min(x1, CANVAS_W - s))
    y1 = max(0.0, min(y1, CANVAS_H - s))
    step = s / 256.0
    yy = np.clip((y1 + (np.arange(256) + 0.5) * step).astype(int), 0, CANVAS_H - 1)
    xx = np.clip((x1 + (np.arange(256) + 0.5) * step).astype(int), 0, CANVAS_W - 1)
    crop = gray_np[np.ix_(yy, xx)]
    rgb = np.stack([crop] * 3, -1).astype(np.float32)
    blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
    cls, xo, yo, _, _ = sess.run(None, {"input": blob})
    G = cls.shape[-1]
    cells = G * G
    cls = cls.reshape(98, cells)
    xo = xo.reshape(98, cells)
    yo = yo.reshape(98, cells)
    ids = cls.argmax(1)
    cxg = (ids % G).astype(np.float32)
    cyg = (ids // G).astype(np.float32)
    nx = (cxg + xo[np.arange(98), ids]) / G
    ny = (cyg + yo[np.arange(98), ids]) / G
    pts = np.stack([x1 + nx * s, y1 + ny * s], 1)
    return pts


def ear98(pts):
    def d(i, j):
        return float(np.linalg.norm(pts[i] - pts[j]))
    D = d(60, 68)
    if D < 1e-4:
        return 0.0
    vl = (d(61, 66) + d(62, 65)) * 0.5
    vr = (d(69, 74) + d(70, 73)) * 0.5
    return float((vl / D + vr / D) * 0.5)


def paste_face(img, face_px=FACE_TARGET_PX, face_frac=1.0):
    """Scale so the face is ~face_px wide (face_frac = face size / min(img side)), paste on QVGA."""
    w, h = img.size
    s = float(face_px) / (float(face_frac) * float(min(w, h)))
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    im = img.convert("L").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (CANVAS_W, CANVAS_H), 40)
    canvas.paste(im, ((CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2))
    return canvas


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wlp", type=int, default=120)
    ap.add_argument("--n-cew", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    random.seed(a.seed)

    global OUTS
    interp = tf.lite.Interpreter(model_path=TFL_INT8)
    interp.allocate_tensors()
    det = interp.get_input_details()[0]
    OUTS = {int(t["shape"][-1]): t for t in interp.get_output_details()}
    anc = anchors()
    s98 = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])

    # ---------- 1) detector on many identities (300W_LP real photos) ----------
    wlp = []
    for sub in ("AFW", "HELEN", "LFPW", "IBUG"):
        wlp += glob.glob(os.path.join(WLP, sub, "*.jpg"))
    random.shuffle(wlp)
    det_rows = []
    for f in wlp[: a.n_wlp]:
        img = Image.open(f).convert("L")
        canvas = paste_face(img, FACE_TARGET_PX, 0.70)   # 300W_LP: face ~0.70 of the 450px crop
        sc, box, ncand = detect(interp, det, anc, canvas)
        det_rows.append({"file": os.path.basename(f), "score": round(sc, 3),
                         "detected": box is not None, "cand": ncand})
    n = len(det_rows)
    hit = sum(1 for r in det_rows if r["detected"])
    hit_acq = sum(1 for r in det_rows if r["score"] >= ACQ_SCORE)
    scores = [r["score"] for r in det_rows]
    det_summary = {"n": n, "detected_ge_0.50": hit, "detected_ge_0.60": hit_acq,
                   "rate_0.50": round(hit / n, 3), "rate_0.60": round(hit_acq / n, 3),
                   "score_median": pct(scores, 0.5), "score_p10": pct(scores, 0.1),
                   "score_min": round(min(scores), 3)}
    print("== detector on 300W_LP (multi-identity) ==")
    print(json.dumps(det_summary, ensure_ascii=False, indent=1))

    # ---------- 2) EAR open vs closed on CEW (many identities) ----------
    def cew_eval(sub, n_max):
        files = glob.glob(os.path.join(CEW, sub, "*.jpg"))
        random.shuffle(files)
        rows = []
        for f in files[:n_max]:
            img = Image.open(f).convert("L")
            face = paste_face(img, FACE_TARGET_PX)      # face crop fills ~100 px
            g = np.asarray(face, dtype=np.uint8)
            # ROI = the pasted face box (isolates the landmark model, not the detector)
            side = FACE_TARGET_PX
            x0 = (CANVAS_W - side) // 2
            y0 = (CANVAS_H - side) // 2
            pts = run98(s98, g, x0, y0, side)
            e = ear98(pts)
            rows.append({"file": os.path.basename(f), "ear": round(e, 4)})
        return rows

    opens = cew_eval("open", a.n_cew)
    closeds = cew_eval("closed", a.n_cew)
    eo = [r["ear"] for r in opens]
    ec = [r["ear"] for r in closeds]

    def stats(v):
        return {"n": len(v), "median": pct(v, 0.5), "p10": pct(v, 0.1), "p90": pct(v, 0.9),
                "min": round(min(v), 3), "max": round(max(v), 3)}

    # firmware rule: threshold = clamp(0.70 * baseline_p75, 0.15, 0.25)
    base75 = pct(eo, 0.75)
    thr = min(0.25, max(0.15, 0.70 * base75))
    fr_closed = sum(1 for v in ec if v < thr) / len(ec)
    fr_open = sum(1 for v in eo if v < thr) / len(eo)
    ear_summary = {
        "open": stats(eo), "closed": stats(ec),
        "baseline_p75_open": base75, "firmware_threshold": round(thr, 3),
        "closed_below_threshold": round(fr_closed, 3),
        "open_below_threshold(false_alarm)": round(fr_open, 3),
        "floor_0.15_would_flag_open": round(sum(1 for v in eo if v < 0.15) / len(eo), 3),
    }
    print("\n== PIPNet EAR on CEW (multi-identity) ==")
    print(json.dumps(ear_summary, ensure_ascii=False, indent=1))

    with open(os.path.join(OUT, "multiperson_eval.json"), "w", encoding="utf-8") as fh:
        json.dump({"detector_300W_LP": det_summary, "detector_rows": det_rows,
                   "ear_cew": ear_summary, "ear_open_rows": opens, "ear_closed_rows": closeds}, fh,
                  ensure_ascii=False, indent=1)
    print("\nsaved", os.path.join(OUT, "multiperson_eval.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
