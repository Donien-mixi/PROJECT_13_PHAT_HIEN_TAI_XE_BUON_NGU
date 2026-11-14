"""BlazeFace parity gate: float ONNX  vs  INT8 TFLite (onnx2tf -oiqt).

Runs the exact preprocessing the firmware will use (letterbox 128 + x/127.5 - 1),
quantizes with the scale/zero-point read from the TFLite input tensor, decodes
anchors -> boxes -> NMS, then reports:
  * max |delta logit| on the winning anchor, delta score
  * IoU(float box, int8 box)   (gate: > 0.9)
  * box coordinate error in input pixels

Usage: py -3.11 tools/blaze_parity.py
"""
import json
import os

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_FLOAT = os.path.join(ROOT, r"output\port_models_20261110\blazeface_nchw.onnx")
TFL_INT8 = os.environ.get("BLAZE_TFLITE", os.path.join(ROOT, r"output\port_models_20261110\blaze_iq\blazeface_nchw_full_integer_quant.tflite"))
OUT_JSON = os.environ.get("BLAZE_PARITY_OUT", os.path.join(ROOT, r"output\port_models_20261110\blaze_iq\parity.json"))

IMAGES = [
    ("still_open", r"output\compare_live_20261112\still_open.png"),
    ("still_closed", r"output\compare_live_20261112\still_closed.png"),
    ("board_frame", r"output\compare_live_20261112\board_frame.png"),
    ("esp32_still_open", r"output\compare_live_20261112\esp32_still_open.png"),
    ("esp32_still_closed", r"output\compare_live_20261112\esp32_still_closed.png"),
    ("board_jpg", r"output\compare_live_20261112\board.jpg"),
]
SIZE = 128
SCORE_TH = 0.5
NMS_IOU = 0.3


def anchors():
    out = []
    for y in range(16):
        for x in range(16):
            cx, cy = (x + 0.5) / 16.0, (y + 0.5) / 16.0
            out.append((cx, cy))
            out.append((cx, cy))
    for y in range(8):
        for x in range(8):
            cx, cy = (x + 0.5) / 8.0, (y + 0.5) / 8.0
            for _ in range(6):
                out.append((cx, cy))
    return np.asarray(out, dtype=np.float32)  # [896,2] in [0,1]


def letterbox_gray(img):
    w, h = img.size
    s = float(SIZE) / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (SIZE, SIZE), 0)
    canvas.paste(im, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0, s, (SIZE - nw) // 2, (SIZE - nh) // 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def decode(reg, cls, anc):
    """reg [896,16] pixels-unit, cls [896] logits -> (score, box[4], n_cand)."""
    score = sigmoid(cls)
    idx = np.where(score >= SCORE_TH)[0]
    if idx.size == 0:
        return 0.0, None, 0, -1
    cx = reg[idx, 0] + anc[idx, 0] * SIZE
    cy = reg[idx, 1] + anc[idx, 1] * SIZE
    w = reg[idx, 2]
    h = reg[idx, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    sc = score[idx]
    order = np.argsort(-sc)
    boxes, sc = boxes[order], sc[order]
    keep = []
    for i in range(len(sc)):
        ok = True
        for j in keep:
            x0 = max(boxes[i, 0], boxes[j, 0]); y0 = max(boxes[i, 1], boxes[j, 1])
            x1 = min(boxes[i, 2], boxes[j, 2]); y1 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            ua = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + \
                 (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]) - inter
            if ua > 1e-6 and inter / ua > NMS_IOU:
                ok = False
                break
        if ok:
            keep.append(i)
    best = keep[0]
    return float(sc[best]), boxes[best].astype(np.float32), len(idx), int(idx[order[best]])


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-6 else 0.0


def main():
    anc = anchors()
    sess = ort.InferenceSession(ONNX_FLOAT, providers=["CPUExecutionProvider"])
    interp = tf.lite.Interpreter(model_path=TFL_INT8)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    qs, qz = in_det["quantization"]
    outs = {int(t["shape"][-1]): t for t in interp.get_output_details()}

    rows = []
    for name, rel in IMAGES:
        img = Image.open(os.path.join(ROOT, rel)).convert("L")
        x01, s_lb, padx, pady = letterbox_gray(img)
        x = (x01 - 0.5) / 0.5  # [-1,1]
        nhwc = np.stack([x] * 3, axis=-1)[None].astype(np.float32)

        o = sess.run(None, {sess.get_inputs()[0].name: np.transpose(nhwc, (0, 3, 1, 2))})
        named = {v.name: val for v, val in zip(sess.get_outputs(), o)}
        f_reg = named["regressors"][0]
        f_cls = named["classificators"][0][:, 0]

        q = np.clip(np.rint(nhwc / qs) + qz, -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], q)
        interp.invoke()
        q_cls = interp.get_tensor(outs[1]["index"]).reshape(-1).astype(np.float32)
        q_reg = interp.get_tensor(outs[16]["index"])[0].astype(np.float32)
        q_cls = (q_cls - outs[1]["quantization"][1]) * outs[1]["quantization"][0]
        q_reg = (q_reg - outs[16]["quantization"][1]) * outs[16]["quantization"][0]

        f_score, f_box, f_n, f_win = decode(f_reg, f_cls, anc)
        q_score, q_box, q_n, q_win = decode(q_reg, q_cls, anc)

        # Saturation-aware parity: compare the SAME (float winner) anchor.
        d_logit_win = abs(float(f_cls[f_win]) - float(q_cls[f_win]))
        d_score_win = abs(float(sigmoid(f_cls[f_win])) - float(sigmoid(q_cls[f_win])))
        rel = f_cls > 0.0  # anchors the float model considers plausibly a face
        d_cls_rel_max = float(np.max(np.abs(f_cls[rel] - q_cls[rel]))) if rel.any() else 0.0
        d_reg_win = [abs(float(f_reg[f_win, k] - q_reg[f_win, k])) for k in range(4)]

        row = {
            "image": name,
            "letterbox_scale": round(s_lb, 4),
            "float": {"score": round(f_score, 4),
                      "box": None if f_box is None else [round(float(v), 1) for v in f_box],
                      "cand": f_n, "win_anchor": f_win},
            "int8": {"score": round(q_score, 4),
                     "box": None if q_box is None else [round(float(v), 1) for v in q_box],
                     "cand": q_n, "win_anchor": q_win},
            "d_score_top1": round(abs(f_score - q_score), 4),
            "d_score_winner": round(d_score_win, 4),
            "d_logit_winner": round(d_logit_win, 3),
            "d_cls_max_on_pos": round(d_cls_rel_max, 3),
            "d_reg_winner_xywh": [round(v, 2) for v in d_reg_win],
            "iou": round(iou(f_box, q_box), 4),
            "d_logit_max_all": round(float(np.max(np.abs(f_cls - q_cls))), 3),
            "d_reg_max_all": round(float(np.max(np.abs(f_reg - q_reg))), 2),
            "input_q_range": [int(q.min()), int(q.max())],
        }
        if f_box is not None and q_box is not None:
            row["d_box_px"] = [round(float(v), 2) for v in (q_box - f_box)]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=float))

    pos = [r for r in rows if r.get("d_box_px") is not None]
    neg = [r for r in rows if r.get("d_box_px") is None]
    disagree = [r for r in rows if (r["float"]["box"] is None) != (r["int8"]["box"] is None)]
    ok = (not disagree) and all(
        r["iou"] > 0.9 and max(abs(v) for v in r["d_box_px"]) < 3.0 for r in pos)
    report = {
        "gate": {
            "iou_gt": 0.9,
            "box_err_px_lt": 3.0,
            "detection_agreement": "both find a face, or both find none",
            "passed": bool(ok),
            "positive_pairs": len(pos),
            "no_face_agreements": len(neg),
            "disagreements": len(disagree),
            "score_delta_max": max([r["d_score_winner"] for r in pos] or [0.0]),
        },
        "score_delta_note": (
            "Classifier output is int8 with scale 0.888 logit/LSB, so the smallest "
            "resolvable score change near a confident face is ~0.07; score deltas are "
            "reported for information, IoU/box parity is the operative gate."),
        "onnx_float": os.path.basename(ONNX_FLOAT),
        "tflite_int8": os.path.basename(TFL_INT8),
        "input_quant": {"scale": qs, "zero_point": qz},
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=float)
    print("GATE", "PASS" if ok else "FAIL", "->", OUT_JSON)


if __name__ == "__main__":
    main()
