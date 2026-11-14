"""Merge the three board-vs-laptop batches and draw an overlay sheet.

Usage:
  py tools\\merge_board_vs_laptop.py
"""
import glob
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
OUT = os.path.join(ROOT, r"output\board_vs_laptop_20261114")
W, H = 320, 240
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"MULTIID: (\d+) \[([^\]]+)\].*side=([\d.]+) roi=(\d+),(\d+),(\d+)")
PTS = re.compile(r"MULTIIDPTS: (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)")

BATCHES = [
    ("L1", r"output\blaze_iq_20261113\serial_multiid_pts.log", r"output\multiid_from_header_20261114",
     r"output\board_vs_laptop_20261114\board_vs_laptop.json"),
    ("L2", r"output\blaze_iq_20261113\serial_multiid_b2.log", r"output\multiid_b2_frames",
     r"output\board_vs_laptop_20261114\batch2\board_vs_laptop.json"),
    ("L3", r"output\blaze_iq_20261113\serial_multiid_b3.log", r"output\multiid_b3_frames",
     r"output\board_vs_laptop_20261114\batch3\board_vs_laptop.json"),
]


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


def overlay_sheet(picks):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for ax, (tag, idx) in zip(axs.flat, picks):
        log, frames, js = {b[0]: (b[1], b[2], b[3]) for b in BATCHES}[tag]
        boxes, pts = parse(os.path.join(ROOT, log))
        rows = json.load(open(os.path.join(ROOT, js), encoding='utf-8'))['rows']
        meta = next((r for r in rows if r['i'] == idx), {})
        gray = cv2.imread(os.path.join(ROOT, frames, f'frame_{idx:02d}.png'), cv2.IMREAD_GRAYSCALE)
        b = boxes[idx]
        lp = laptop_landmarks(ORT_SESS, gray, b['roi'])
        bp = pts[idx]
        d = np.linalg.norm(bp - lp, axis=1)
        ax.imshow(gray, cmap='gray')
        ax.plot(bp[:, 0], bp[:, 1], 'o', ms=2.4, mfc='none', mec='#00e676', mew=0.8, label='ESP32 INT8')
        ax.plot(lp[:, 0], lp[:, 1], 'x', ms=2.8, mec='#ff1744', mew=0.8, label='laptop float')
        for k in np.argsort(-d)[:2]:
            ax.plot([bp[k,0], lp[k,0]], [bp[k,1], lp[k,1]], '-', color='#ffea00', lw=0.7)
        ax.set_title('%s id%02d | L2 %.2f px (p95 %.2f) | EAR %.3f vs %.3f' % (
            tag, idx, d.mean(), np.percentile(d, 95), meta.get('ear_board', 0), meta.get('ear_laptop', 0)), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower right', fontsize=7)
    fig.suptitle('32 ảnh tĩnh (12+10+10 người) chạy trên ESP32 và laptop — 6 ví dụ điển hình', fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, 'overlay_sheet_32.png')
    fig.savefig(p, dpi=120)
    print('wrote', p)


def main():
    overlay_sheet([('L1', 4), ('L2', 4), ('L3', 4), ('L1', 7), ('L3', 6), ('L1', 11)])
    all_rows, all_err = [], []
    for tag, log, frames, js in BATCHES:
        d = json.load(open(os.path.join(ROOT, js), encoding="utf-8"))
        for r in d["rows"]:
            r["batch"] = tag
            all_rows.append(r)
        # recompute point errors for the global distribution
        boxes, pts = parse(os.path.join(ROOT, log))
        for i, b in boxes.items():
            if i not in pts:
                continue
            f = os.path.join(ROOT, frames, f"frame_{i:02d}.png")
            if not os.path.exists(f):
                continue
            gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            lp = laptop_landmarks(ORT_SESS, gray, b["roi"])
            all_err.append(np.linalg.norm(pts[i] - lp, axis=1))

    err = np.concatenate(all_err) if all_err else np.zeros(1)
    summary = {
        "n_images": len(all_rows),
        "n_points": int(err.size),
        "score_delta_median": round(float(np.median([abs(r["score_board"] - r["score_laptop"]) for r in all_rows])), 3),
        "box_iou_median": round(float(np.median([r["box_iou"] for r in all_rows])), 3),
        "box_iou_min": round(min(r["box_iou"] for r in all_rows), 3),
        "L2_mean_px": round(float(err.mean()), 2),
        "L2_median_px": round(float(np.median(err)), 2),
        "L2_p95_px": round(float(np.percentile(err, 95)), 2),
        "L2_max_px": round(float(err.max()), 2),
        "per_image_median_px": round(float(np.median([r["L2_mean_px"] for r in all_rows])), 2),
        "ear_delta_median": round(float(np.median([r["d_ear"] for r in all_rows])), 4),
        "ear_delta_max": round(max(r["d_ear"] for r in all_rows), 4),
        "per_batch": {},
    }
    for tag in ("L1", "L2", "L3"):
        rows = [r for r in all_rows if r["batch"] == tag]
        if rows:
            summary["per_batch"][tag] = {
                "n": len(rows),
                "L2_mean_px": round(float(np.mean([r["L2_mean_px"] for r in rows])), 2),
                "iou_median": round(float(np.median([r["box_iou"] for r in rows])), 3),
                "ear_delta_median": round(float(np.median([r["d_ear"] for r in rows])), 4),
            }

    with open(os.path.join(OUT, "all_batches.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": all_rows}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "all_batches.csv"), "w", encoding="utf-8") as fh:
        fh.write("batch,idx,name,score_board,score_laptop,box_iou,L2_mean_px,L2_p95_px,ear_board,ear_laptop,d_ear\n")
        for r in all_rows:
            fh.write(f"{r['batch']},{r['i']},\"{r['name']}\",{r['score_board']},{r['score_laptop']},"
                     f"{r['box_iou']},{r['L2_mean_px']},{r['L2_p95_px']},{r['ear_board']},{r['ear_laptop']},{r['d_ear']}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", os.path.join(OUT, "all_batches.json"), "and .csv")


if __name__ == "__main__":
    ORT_SESS = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])
    main()
