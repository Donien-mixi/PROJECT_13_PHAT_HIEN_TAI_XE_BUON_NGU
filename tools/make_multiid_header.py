"""Build a MULTI-IDENTITY still set from YawDD (real drivers) for the on-board test.

YawDD gives 348 clips from many drivers (glasses / no glasses, male / female) recorded on a
dash-cam — the same domain as this project. This script picks one good frame per driver,
resizes it to the board's QVGA framebuffer (320x240) and writes a C header so the firmware
can run the WHOLE chain (BlazeFace INT8 -> ROI -> PIPNet-98 -> EAR) on other people's faces
with no live subject.

Selection criteria (offline, using the deployed INT8 BlazeFace):
  detector score >= 0.60 and face box side 60..130 px at 320x240  (board working range)

Usage:
  py tools\\make_multiid_header.py --n 10
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL_INT8 = os.path.join(ROOT, r"output\port_models_20261110\blaze_iq_mp\blazeface_nchw_full_integer_quant.tflite")
OUT_H = os.path.join(ROOT, r"firmware_esp32\main\multiid_images.h")
OUT_PNG = os.path.join(ROOT, r"output\multiid_stills_20261114")
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


def detect(interp, det, outs, anc, gray):
    qs, qz = det["quantization"]
    s = 128.0 / float(max(W, H))
    nw, nh = int(round(W * s)), int(round(H * s))
    im = Image.fromarray(gray).resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (128, 128), 0)
    c.paste(im, ((128 - nw) // 2, (128 - nh) // 2))
    x = np.asarray(c, dtype=np.float32) / 127.5 - 1.0
    q = np.clip(np.rint(x / qs) + qz, -128, 127).astype(np.int8)
    interp.set_tensor(det["index"], np.stack([q] * 3, -1)[None])
    interp.invoke()
    cc = (interp.get_tensor(outs[1]["index"]).reshape(-1).astype(np.float32) - outs[1]["quantization"][1]) * outs[1]["quantization"][0]
    rr = (interp.get_tensor(outs[16]["index"])[0].astype(np.float32) - outs[16]["quantization"][1]) * outs[16]["quantization"][0]
    sc = 1.0 / (1.0 + np.exp(-np.clip(cc, -60, 60)))
    i = int(np.argmax(sc))
    if sc[i] < 0.5:
        return float(sc[i]), None
    padx, pady = (128 - nw) // 2, (128 - nh) // 2
    cx = (rr[i, 0] + anc[i, 0] * 128 - padx) / s
    cy = (rr[i, 1] + anc[i, 1] * 128 - pady) / s
    side = max(rr[i, 2], rr[i, 3]) / s
    return float(sc[i]), (cx, cy, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--cew-closed", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(OUT_PNG, exist_ok=True)

    interp = tf.lite.Interpreter(model_path=TFL_INT8)
    interp.allocate_tensors()
    det = interp.get_input_details()[0]
    outs = {int(t["shape"][-1]): t for t in interp.get_output_details()}
    anc = anchors()

    vids = sorted(glob.glob(os.path.join(ROOT, r"datasets\raw_faces\YawDD\**\*.avi"), recursive=True))
    random.shuffle(vids)
    picked, seen_dirs = [], {}
    for v in vids:
        if len(picked) >= a.n:
            break
        # one clip per driver folder keeps identities distinct
        key = os.path.dirname(v)
        if seen_dirs.get(key, 0) >= 2:      # max 2 drivers per folder
            continue
        glasses = "Glasses" in os.path.basename(v) and "NoGlasses" not in os.path.basename(v)
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            cap.release()
            continue
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        best = None
        for idx in range(20, max(60, nfr - 20), max(10, nfr // 40)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            sc, box = detect(interp, det, outs, anc, gray)
            if box is None or sc < 0.55:
                continue
            cx, cy, side = box
            if not (60.0 <= side <= 130.0):
                continue
            if not (0.25 * W < cx < 0.75 * W and 0.2 * H < cy < 0.8 * H):
                continue
            if best is None or sc > best[0]:
                best = (sc, gray, side, os.path.basename(v), glasses)
        cap.release()
        if best:
            seen_dirs[key] = seen_dirs.get(key, 0) + 1
            picked.append(best)

    # ---- CEW: closed eyes of OTHER people (label CLOSED) ----
    cew_closed = sorted(glob.glob(os.path.join(ROOT, r"datasets\raw_faces\CEW\CEW\closed\*.jpg")))
    random.shuffle(cew_closed)
    n_closed = 0
    for f in cew_closed:
        if n_closed >= a.cew_closed:
            break
        im = Image.open(f).convert("L")
        im = im.resize((100, 100), Image.BILINEAR)
        canvas = Image.new("L", (W, H), 40)
        canvas.paste(im, ((W - 100) // 2, (H - 100) // 2))
        gray = np.asarray(canvas, dtype=np.uint8)
        sc, box = detect(interp, det, outs, anc, gray)
        if box is None or sc < 0.55:
            continue
        cx, cy, side = box
        if not (60.0 <= side <= 130.0):
            continue
        picked.append((sc, gray, side, "CEW_closed_" + os.path.basename(f)[:12], False, "CLOSED"))
        n_closed += 1
    print("CEW closed added:", n_closed)

    if len(picked) < a.n:
        print(f"WARNING: only {len(picked)} identities found (asked {a.n})")

    names, arrays = [], []
    for i, item in enumerate(picked):
        sc, gray, side, fname, glasses = item[:5]
        kind = item[5] if len(item) > 5 else ("G" if glasses else "-")
        p = os.path.join(OUT_PNG, f"id{i:02d}_{'glasses' if glasses else 'noglass'}.png")
        Image.fromarray(gray).save(p)
        names.append(f"{a.tag}id{i:02d} {kind} {fname[:16]} s{side:.0f}px sc{sc:.2f}")
        arrays.append(gray.reshape(-1))

    with open(OUT_H, "w", encoding="ascii", newline="\n") as fh:
        fh.write("// AUTO-GENERATED by tools/make_multiid_header.py (YawDD driver frames)\n")
        fh.write(f"// {len(arrays)} identities, {W}x{H} grey, one frame per driver\n")
        fh.write("#pragma once\n#include <stdint.h>\n\n")
        fh.write(f"#define MULTIID_COUNT {len(arrays)}\n")
        fh.write(f"static const char* const kMultiIdName[MULTIID_COUNT] = {{\n")
        for n in names:
            fh.write(f'  "{n}",\n')
        fh.write("};\n\n")
        fh.write(f"static const uint8_t kMultiId[MULTIID_COUNT][{W * H}] = {{\n")
        for arr in arrays:
            fh.write("  {")
            fh.write(",".join(str(int(v)) for v in arr))
            fh.write("},\n")
        fh.write("};\n")
    print("wrote", OUT_H, "count", len(arrays))
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
