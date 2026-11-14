"""Parse an ESP32 serial log and report BlazeFace live-detector behaviour.

Reads the log produced by tools/serial_capture.py (lines prefixed with [HH:MM:SS])
and extracts:
  BLAZE  : cand=<n> score=<s> box=<x0>,<y0>,<x1>,<y1> ms=<t>
  DETLITE: frame <w>x<h> mean=<m> min=<a> max=<b> grid=...
  MAIN_APP: [Frame #n] ... AI: ...ms ... ROI: x,y,s | EAR: e ...

Outputs a per-detection table, a summary with median/p95, and pass/fail gates.
Usage:
  py -3.11 tools/live_blaze_measure.py output/blaze_iq_20261113/serial_live3.log
"""
import json
import os
import re
import statistics as st
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_BLAZE = re.compile(
    r"BLAZE: cand=(\d+) score=([\d.]+) box=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+) ms=(\d+)")
RE_FRAME = re.compile(r"DETLITE: frame (\d+)x(\d+) mean=(\d+) min=(\d+) max=(\d+)")
RE_MISS = re.compile(r"BLAZE: cand=0 \(no anchor above [\d.]+, max_logit=(-?[\d.]+)\) in (\d+) ms")
RE_STATS = re.compile(r"MAIN_APP: \[Frame #\s*(\d+)\].*AI:\s*([\d.]+)ms.*ROI: (\d+),(\d+),(\d+) \| EAR: ([\d.]+)")


def parse(path):
    dets, frames, stats = [], [], []
    t = "?"
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip())
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]", line)
        if m:
            t = m.group(1)
        m = RE_FRAME.search(line)
        if m:
            frames.append({"t": t, "w": int(m.group(1)), "h": int(m.group(2)),
                           "mean": int(m.group(3)), "min": int(m.group(4)), "max": int(m.group(5))})
        m = RE_BLAZE.search(line)
        if m:
            d = {"t": t, "cand": int(m.group(1)), "score": float(m.group(2)),
                 "x0": float(m.group(3)), "y0": float(m.group(4)),
                 "x1": float(m.group(5)), "y1": float(m.group(6)),
                 "ms": int(m.group(7))}
            d["w"] = round(d["x1"] - d["x0"], 1)
            d["h"] = round(d["y1"] - d["y0"], 1)
            d["cx"] = round((d["x0"] + d["x1"]) / 2, 1)
            d["cy"] = round((d["y0"] + d["y1"]) / 2, 1)
            dets.append(d)
            continue
        m = RE_MISS.search(line)
        if m:
            dets.append({"t": t, "cand": 0, "score": 0.0, "max_logit": float(m.group(1)),
                         "ms": int(m.group(2)), "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0})
        m = RE_STATS.search(line)
        if m:
            stats.append({"t": t, "frame": int(m.group(1)), "ai_ms": float(m.group(2)),
                          "roi": [int(m.group(3)), int(m.group(4)), int(m.group(5))],
                          "ear": float(m.group(6))})
    return dets, frames, stats


def summary(vals):
    if not vals:
        return {}
    s = sorted(vals)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"n": len(s), "median": round(st.median(s), 2), "p95": round(p95, 2),
            "min": round(min(s), 2), "max": round(max(s), 2)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    dets, frames, stats = parse(path)
    out = {"log": os.path.abspath(path), "detections": dets, "frames": frames, "pipeline": stats}

    print(f"{'t':>8} {'cand':>4} {'score':>6} {'w':>6} {'h':>6} {'cx':>6} {'cy':>6} {'ms':>4}")
    for d in dets:
        print(f"{d['t']:>8} {d['cand']:4d} {d['score']:6.3f} {d['w']:6.1f} {d['h']:6.1f} "
              f"{d['cx']:6.1f} {d['cy']:6.1f} {d['ms']:4d}")

    hit = [d for d in dets if d["score"] >= 0.9]
    miss = [d for d in dets if d["score"] < 0.5]
    gates = {
        "detections": len(dets),
        "frames_with_score_ge_0.90": len(hit),
        "frames_with_score_lt_0.50": len(miss),
        "hit_rate": round(len(hit) / len(dets), 3) if dets else 0.0,
        "cand_median": st.median([d["cand"] for d in dets]) if dets else 0,
        "box_w_range": [min((d["w"] for d in dets), default=0), max((d["w"] for d in dets), default=0)],
        "ms": summary([d["ms"] for d in dets]),
        "score": summary([d["score"] for d in dets]),
        "frame_mean_range": [min((f["mean"] for f in frames), default=0),
                             max((f["mean"] for f in frames), default=0)],
        "pipeline": {"n": len(stats),
                     "ai_ms": summary([s["ai_ms"] for s in stats]),
                     "ear": summary([s["ear"] for s in stats])},
        "pass": bool(dets) and len(hit) / max(1, len(dets)) >= 0.8,
    }
    out["gates"] = gates
    print("\n--- summary ---")
    print(json.dumps(gates, ensure_ascii=False, indent=1, default=float))
    dst = os.path.splitext(path)[0] + "_measure.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("saved", dst)
    return 0 if gates["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
