"""Reconstruct frames dumped by the firmware CAPTURE mode into PNGs.

Firmware (BLAZE_CAPTURE_MODE=1) prints, per frame:
    CAPTURE: BEGIN <idx> <w> <h>
    CAPTURE: ROW <y> <x0> <hex...>      (64 px per row chunk)
    CAPTURE: END <idx>
This script rebuilds every complete frame, writes PNG + a manifest with the wall-clock
timestamp of the BEGIN line (so poses can be aligned with the cue timeline).

Usage:
  py tools\\blaze_frame_capture.py --log output/blaze_iq_20261113/serial_capture1.log \
      --out-dir output/calib_frames_20261113
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^\[(\d\d:\d\d:\d\d)\]")
BEGIN = re.compile(r"CAPTURE: BEGIN (\d+) (\d+) (\d+)")
ROW = re.compile(r"CAPTURE: ROW (\d+) (\d+) ([0-9a-fA-F]+)")
END = re.compile(r"CAPTURE: END (\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    frames = {}      # idx -> {"w":..,"h":..,"rows":{y:(x0,bytes)},"t":..}
    cur = None
    stamp = "?"
    for raw in open(a.log, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip())
        m = TS.match(line)
        if m:
            stamp = m.group(1)
        m = BEGIN.search(line)
        if m:
            cur = {"idx": int(m.group(1)), "w": int(m.group(2)), "h": int(m.group(3)),
                   "rows": {}, "t": stamp}
            continue
        m = ROW.search(line)
        if m and cur is not None:
            cur["rows"][(int(m.group(1)), int(m.group(2)))] = bytes.fromhex(m.group(3))
            continue
        m = END.search(line)
        if m and cur is not None and int(m.group(1)) == cur["idx"]:
            frames[cur["idx"]] = cur
            cur = None

    manifest = []
    ok = 0
    for idx in sorted(frames):
        f = frames[idx]
        w, h = f["w"], f["h"]
        img = np.zeros((h, w), dtype=np.uint8)
        filled = 0
        for y in range(h):
            row = bytearray(w)
            got = False
            for x0 in range(0, w, 64):
                data = f["rows"].get((y, x0))
                if data is not None:
                    row[x0:x0 + len(data)] = data
                    got = True
            if got:
                img[y, :] = row[:w]
                filled += 1
        name = f"frame_{idx:02d}.png"
        Image.fromarray(img).save(os.path.join(a.out_dir, name))
        manifest.append({"file": name, "t": f["t"], "w": w, "h": h,
                         "rows_complete": filled, "mean": int(img.mean())})
        if filled == h:
            ok += 1
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"log": os.path.abspath(a.log), "frames": manifest,
                   "complete": ok, "total": len(manifest)}, fh, ensure_ascii=False, indent=1)
    print(f"frames={len(manifest)} complete={ok} -> {a.out_dir}")
    for m in manifest:
        print(f"  {m['file']} t={m['t']} rows={m['rows_complete']}/{m['h']} mean={m['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
