"""Dump the frames embedded in firmware_esp32/main/multiid_images.h back to PNGs.

Guarantees the evidence sheet shows exactly what the board ran (no stale files), and
returns the names the firmware logs used.

Usage: py tools/dump_multiid_header.py --out output/multiid_from_header_20261114
"""
import argparse
import os
import re

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = os.path.join(ROOT, r"firmware_esp32\main\multiid_images.h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    txt = open(HDR, encoding="ascii", errors="replace").read()

    names = re.findall(r'^\s*"([^"]+)",\s*$', txt, re.M)
    m = re.search(r"kMultiId\[MULTIID_COUNT\]\[320 \* 240\] = \{(.*)\};", txt, re.S)
    if not m:
        m = re.search(r"kMultiId\[MULTIID_COUNT\]\[76800\] = \{(.*)\};", txt, re.S)
    body = m.group(1)
    frames = re.findall(r"\{([^{}]+)\}", body)
    print(f"names={len(names)} frames={len(frames)}")
    manifest = []
    for i, f in enumerate(frames):
        vals = np.fromstring(f.replace("\n", ""), dtype=np.int32, sep=",")
        if vals.size != 320 * 240:
            print("skip", i, vals.size)
            continue
        img = vals.astype(np.uint8).reshape(240, 320)
        p = os.path.join(a.out, f"frame_{i:02d}.png")
        Image.fromarray(img).save(p)
        manifest.append({"i": i, "file": f"frame_{i:02d}.png",
                         "name": names[i] if i < len(names) else "?"})
        print(f"  {i:2d} {manifest[-1]['name']}")
    import json
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
