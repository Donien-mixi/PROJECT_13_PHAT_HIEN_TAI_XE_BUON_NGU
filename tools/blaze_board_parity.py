"""Reproduce the ESP32-side BlazeFace INT8 dump on the host, for byte-level parity.

Mirrors firmware_esp32/main/blaze_tracker.cpp:
  * letterbox 128 (bilinear resize of the grey frame, black pad)
  * x = pixel/127.5 - 1 ; q = round_half_away(x/scale) + zp, clamped
  * int8 inference with the TFLite reference kernels
Then prints the same fields the firmware logs under tag BLAZE "DUMP ...":
  input FNV-1a, top-5 anchors, q-cls/reg at fixed anchors, #anchors above zp.

Usage: py -3.11 tools/blaze_board_parity.py [--serial output/.../serial.log]
"""
import argparse
import os
import re

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL_INT8 = os.path.join(ROOT, r"output\port_models_20261110\blaze_iq\blazeface_nchw_full_integer_quant.tflite")
IMG = os.path.join(ROOT, r"output\compare_live_20261112\still_open.png")
SIZE = 128
FIXED_ANCHORS = [241, 273, 275, 305, 728]


def rint_away(x):
    """C lrintf: round half away from zero, as the firmware uses."""
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def build_input(gray, w, h, qs, qz, padx, pady, lb):
    """Same mapping as prep_input(): float coords, integer truncation."""
    out = np.full((SIZE, SIZE), qz, dtype=np.int32)
    nw = int(w * lb + 0.5)
    nh = int(h * lb + 0.5)
    q_pad = int(rint_away(np.float32(-1.0) / np.float32(qs))) + qz
    q_pad = max(-128, min(127, q_pad))
    out[:] = q_pad
    for dy in range(SIZE):
        sy = int((dy - pady) / lb)
        row_ok = pady <= dy < pady + nh
        for dx in range(SIZE):
            if row_ok and padx <= dx < padx + nw:
                sx = int((dx - padx) / lb)
                sy2 = max(0, min(h - 1, sy))
                sx2 = max(0, min(w - 1, sx))
                x = np.float32(gray[sy2, sx2]) / np.float32(127.5) - np.float32(1.0)
                q = int(rint_away(np.float32(x) / np.float32(qs))) + qz
                out[dy, dx] = max(-128, min(127, q))
    return np.stack([out] * 3, axis=-1).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    a = ap.parse_args()

    gray = np.asarray(Image.open(IMG).convert("L"))
    h, w = gray.shape
    it = tf.lite.Interpreter(model_path=TFL_INT8,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    qs, qz = ind["quantization"]
    lb = float(SIZE) / float(max(w, h))
    nw, nh = int(w * lb + 0.5), int(h * lb + 0.5)
    padx, pady = (SIZE - nw) // 2, (SIZE - nh) // 2
    q = build_input(gray, w, h, qs, qz, padx, pady, lb)
    fnv = 2166136261
    for b in q.reshape(-1):
        fnv ^= int(b) & 0xFF
        fnv = (fnv * 16777619) & 0xFFFFFFFF

    it.set_tensor(ind["index"], q[None])
    it.invoke()
    outs = {int(t["shape"][-1]): t for t in it.get_output_details()}
    cls = it.get_tensor(outs[1]["index"]).reshape(-1)
    reg = it.get_tensor(outs[16]["index"])[0]
    cscale, czp = outs[1]["quantization"]
    rscale, rzp = outs[16]["quantization"]

    host = {"fnv": fnv, "top": [], "anchors": {}, "above": int((cls > czp).sum())}
    order = np.argsort(-cls.astype(np.int16))
    for rank in range(5):
        i = int(order[rank])
        logit = (float(cls[i]) - czp) * cscale
        host["top"].append((i, int(cls[i]), logit, 1.0 / (1.0 + np.exp(-logit))))
    for i in FIXED_ANCHORS:
        host["anchors"][i] = (int(cls[i]), [int(v) for v in reg[i, :4]])

    print(f"HOST fnv={fnv:08x} above_zp={host['above']}")
    for rank, (i, qq, logit, sc) in enumerate(host["top"]):
        print(f"HOST top{rank} anchor={i} q={qq} logit={logit:.3f} score={sc:.4f}")
    for i in FIXED_ANCHORS:
        qq, rr = host["anchors"][i]
        print(f"HOST anchor={i} qcls={qq} reg={','.join(str(v) for v in rr)}")

    if a.serial and os.path.exists(a.serial):
        txt = open(a.serial, encoding="utf-8", errors="replace").read()
        txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
        board = [ln.split("BLAZE: ", 1)[1] for ln in txt.splitlines() if "BLAZE: DUMP" in ln]
        print("\n--- BOARD (last boot only) ---")
        seen = False
        for ln in board:
            print("BOARD", ln)
        print("\n--- DIFF ---")
        m = re.search(r"DUMP in fnv=([0-9a-f]{8})", txt)
        if m:
            bfnv = int(m.group(1), 16)
            print(f"input fnv: board={bfnv:08x} host={fnv:08x} match={bfnv == fnv}")


if __name__ == "__main__":
    main()
