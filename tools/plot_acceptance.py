"""Draw the acceptance evidence figures from the board serial logs (v2).

Figure 1 acceptance_timeline.png : EAR samples (per logged inference + per FSM frame),
                                   calibrated threshold, cue windows, alarm markers.
Figure 2 multiid_sheet.png       : the 12 still frames embedded in the firmware with the
                                   on-board detector score and the on-device EAR.

Usage:
  py tools\\plot_acceptance.py --log <serial_accept3.log> --start 20:37:13 \
      --closed-seconds 40 --out output/blaze_iq_20261113/acceptance_timeline.png
  py tools\\plot_acceptance.py --sheet output/blaze_iq_20261113/multiid_sheet.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"\[(\d\d:\d\d:\d\d)\]")
EAR = re.compile(r"EARDIRECT L=([\d.]+) R=([\d.]+) avg=([\d.]+)")
FRAME = re.compile(r"\[Frame #\s*(\d+)\].*EAR: ([\d.]+) \(thr ([\d.]+) base ([\d.]+)\).*eye: (\w+) cl=([\d.]+)ms.*Status: (.+?)\s*$")

# results measured ON THE BOARD with MULTI_ID_TEST (serial_multiid2.log, k=1.08)
MULTIID = [
    ("id00 nữ + kính", "glasses", 0.704, 0.145), ("id01 nam", "noglass", 0.930, 0.171),
    ("id02 nữ", "noglass", 0.849, 0.165), ("id03 nam", "noglass", 0.849, 0.164),
    ("id04 nam + kính", "glasses", 0.970, 0.148), ("id05 nam + kính", "glasses", 0.970, 0.170),
    ("id06 nữ", "noglass", 0.970, 0.183), ("id07 nữ", "noglass", 0.970, 0.193),
    ("id08 NHẮM", "closed", 0.849, 0.150), ("id09 NHẮM", "closed", 0.849, 0.106),
    ("id10 NHẮM", "closed", 0.849, 0.092), ("id11 NHẮM", "closed", 0.704, 0.125),
]


def secs(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def parse(path):
    ears, frames, t = [], [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        L = ANSI.sub("", line.rstrip())
        m = TS.match(L)
        if m:
            t = m.group(1)
        if t is None:
            continue
        m = EAR.search(L)
        if m:
            ears.append((secs(t), float(m.group(3))))
            continue
        m = FRAME.search(L)
        if m:
            frames.append({"t": secs(t), "ear": float(m.group(2)), "thr": float(m.group(3)),
                           "base": float(m.group(4)), "eye": m.group(5), "cl": float(m.group(6)),
                           "status": m.group(7)})
    return ears, frames


def timeline(a):
    ears, frames = parse(a.log)
    t0 = secs(a.start)
    plan = [("nhìn thẳng", 25, "#c8e6c9"), ("NHẮM MẮT (kịch bản)", a.closed_seconds, "#ffcdd2"),
            ("mở mắt", 20, "#c8e6c9"), ("ngáp", 15, "#fff9c4"),
            ("quay phải", 20, "#e1bee7"), ("quay trái", 20, "#e1bee7"),
            ("cúi", 20, "#e1bee7"), ("ra khỏi khung", 20, "#eeeeee")]
    total = sum(d for _, d, _ in plan)
    thr = frames[-1]["thr"] if frames else 0.15
    base = frames[-1]["base"] if frames else 0.20

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    x = 0
    for label, dur, col in plan:
        ax.add_patch(Rectangle((x, 0), dur, 0.5, color=col, alpha=0.6, zorder=0))
        ax.text(x + dur / 2, 0.30, label, ha="center", va="bottom", fontsize=8.5, color="#37474f")
        x += dur

    ok = [(t - t0, e) for t, e in ears if t - t0 <= total and e < 0.9]
    ax.plot([p[0] for p in ok], [p[1] for p in ok], "-o", color="#1a237e", lw=1.8, ms=7,
            label="EAR đo trên board (mỗi 2 lần suy luận)")
    fok = [(f["t"] - t0, f["ear"]) for f in frames if f["t"] - t0 <= total and f["ear"] < 0.9]
    ax.plot([p[0] for p in fok], [p[1] for p in fok], "x", color="#00838f", ms=9, mew=2,
            label="EAR tại khung FSM quyết định")
    ax.axhline(thr, color="#c62828", ls="--", lw=1.5,
               label=f"ngưỡng = 0,75 × baseline = {thr:.3f} (baseline {base:.2f})")

    for f in frames:
        s = f["status"]
        tt = f["t"] - t0
        if tt > total:
            continue
        if "MICROSLEEP" in s:
            ax.axvline(tt, color="#b71c1c", lw=1.2, alpha=0.8)
            ax.annotate("MICROSLEEP ALARM\n(2 mẫu nhắm trong 3 mẫu)", (tt, f["ear"]),
                        xytext=(tt - 34, 0.245), fontsize=9.5, color="#b71c1c", weight="bold",
                        arrowprops=dict(arrowstyle="->", color="#b71c1c"))
        elif "DISTRACTION" in s:
            ax.axvline(tt, color="#4a148c", lw=1.0, ls=":", alpha=0.8)
            ax.annotate("DISTRACTION", (tt, 0.03), xytext=(tt - 30, 0.06), fontsize=9,
                        color="#4a148c", arrowprops=dict(arrowstyle="->", color="#4a148c"))

    ax.set_xlabel("thời gian từ lúc bắt đầu kịch bản (giây)")
    ax.set_ylabel("EAR")
    ax.set_ylim(0, 0.34)
    ax.set_xlim(0, total)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Nghiệm thu 20:37 — nhắm mắt báo đúng, không báo nhầm khi mở (ESP32-S3, 0,12 FPS)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


def sheet(out, frames_dir):
    import json
    files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png"))
    man_path = os.path.join(frames_dir, "manifest.json")
    hdr_names = []
    if os.path.exists(man_path):
        hdr_names = [m["name"] for m in json.load(open(man_path, encoding="utf-8"))]
    cols, rows = 4, 3
    fig, axs = plt.subplots(rows, cols, figsize=(13.5, 8.4))
    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if i >= len(files):
            continue
        name, kind, score, ear = MULTIID[i] if i < len(MULTIID) else ("", "noglass", 0, 0)
        if i < len(hdr_names):
            name = hdr_names[i].replace("CLOSED ", "NHẮM ")
        img = plt.imread(os.path.join(frames_dir, files[i]))
        ax.imshow(img, cmap="gray")
        color = "#b71c1c" if kind == "closed" else ("#1b5e20" if kind == "noglass" else "#0d47a1")
        ax.set_title(f"{name}\nscore {score:.3f}   EAR {ear:.3f}", fontsize=10, color=color)
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("12 khung người KHÁC được nhúng vào firmware và chạy trên ESP32-S3 "
                 "(8 tài xế YawDD, 4 người mắt nhắm CEW)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--start", default="20:37:13")
    ap.add_argument("--closed-seconds", type=int, default=40)
    ap.add_argument("--out")
    ap.add_argument("--sheet")
    ap.add_argument("--frames", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), r"output\multiid_stills_20261114"))
    a = ap.parse_args()
    if a.sheet:
        sheet(a.sheet, a.frames)
    if a.log and a.out:
        timeline(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
