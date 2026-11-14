"""Score the BlazeFace live session (Block B) by segmenting the serial log
against the cue timeline printed by tools/live_session_cue.py.

Timeline per round (same order as live_session_cue.PLAN):
  NHIN THANG 20 | NHAM MAT 15 | MO MAT 10 | NGAP 8 | QUAY PHAI 15 |
  QUAY TRAI 15 | CUI 15 | RA KHOI KHUNG 15 | NGHI 7        = 120 s

Usage:
  py tools\\live_session_analyze.py output\\blaze_iq_20261113\\serial_live5.log \
      --start 12:08:30 --rounds 3
"""
import argparse
import json
import re
import statistics as st

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"\[(\d\d):(\d\d):(\d\d)\]")
RE_DET = re.compile(r"BLAZE: cand=(\d+) score=([\d.]+) box=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+) ms=(\d+)")
RE_MISS = re.compile(r"BLAZE: cand=0 \(no anchor above [\d.]+, max_logit=(-?[\d.]+)\) in (\d+) ms")
RE_EAR = re.compile(r"EARDIRECT L=([\d.]+) R=([\d.]+) avg=([\d.]+)")
RE_FRAME = re.compile(r"\[Frame #\s*(\d+)\].*ROI: (\d+),(\d+),(\d+) \| EAR: ([\d.]+) \| MAR: ([\d.]+)")

PLAN = [("NHIN THANG - MAT MO", 20), ("NHAM MAT", 15), ("MO MAT", 10), ("NGAP", 8),
        ("QUAY DAU SANG PHAI", 15), ("QUAY DAU SANG TRAI", 15), ("CUI XUONG BAN", 15),
        ("RA KHOI KHUNG", 15), ("NGHI", 7)]

ACCEPT_PLAN = [("NHIN THANG - MAT MO", 25), ("NHAM MAT", 30), ("MO MAT", 20), ("NGAP", 15),
               ("QUAY DAU SANG PHAI", 20), ("QUAY DAU SANG TRAI", 20), ("CUI XUONG BAN", 20),
               ("RA KHOI KHUNG", 20)]


def secs(ts):
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def parse(path):
    ev, t = [], None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip())
        m = TS.match(line)
        if m:
            t = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
        if t is None:
            continue
        e = secs(t)
        m = RE_DET.search(line)
        if m:
            x0, y0, x1, y1 = (float(m.group(i)) for i in (3, 4, 5, 6))
            ev.append({"t": t, "e": e, "kind": "det", "cand": int(m.group(1)),
                       "score": float(m.group(2)), "ms": int(m.group(7)),
                       "side": round(x1 - x0, 1), "cx": round((x0 + x1) / 2, 1),
                       "cy": round((y0 + y1) / 2, 1)})
            continue
        m = RE_MISS.search(line)
        if m:
            ev.append({"t": t, "e": e, "kind": "miss", "cand": 0, "score": 0.0,
                       "ms": int(m.group(2)), "side": 0.0, "cx": 0.0, "cy": 0.0})
            continue
        m = RE_EAR.search(line)
        if m:
            ev.append({"t": t, "e": e, "kind": "ear", "ear": float(m.group(3)),
                       "ear_l": float(m.group(1)), "ear_r": float(m.group(2))})
            continue
        m = RE_FRAME.search(line)
        if m:
            ev.append({"t": t, "e": e, "kind": "frame", "frame": int(m.group(1)),
                       "roi": [int(m.group(2)), int(m.group(3)), int(m.group(4))],
                       "ear": float(m.group(5)), "mar": float(m.group(6))})
    return ev


def seg(ev, a, b, kinds):
    return [x for x in ev if x["kind"] in kinds and a <= x["seg_e"] < b]


def stat(vals, nd=3):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "median": round(st.median(s), nd), "min": round(s[0], nd),
            "max": round(s[-1], nd),
            "p95": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], nd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--start", default="12:08:30", help="wall clock of round 1 action 1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--acceptance", action="store_true", help="kich ban nghiem thu 170 s/vong")
    ap.add_argument("--latency", type=float, default=7.7,
                    help="s between JPEG capture and the EAR log line (AI block, measured)")
    a = ap.parse_args()

    ev = parse(a.log)
    # The EAR/MAR line is printed AFTER inference, so its wall clock is ~AI ms later than
    # the frame it describes. Shift it back so it can be compared with the action timeline.
    for x in ev:
        if x["kind"] in ("ear", "frame"):
            x["seg_e"] = x["e"] - a.latency
        else:
            x["seg_e"] = x["e"] - 0.6   # decode+detect of the same frame
    t0 = secs(a.start)
    out = {"log": a.log, "start": a.start, "rounds": a.rounds, "latency_s": a.latency,
           "segments": []}

    plan = ACCEPT_PLAN if a.acceptance else PLAN
    round_len = sum(p[1] for p in plan)
    for r in range(a.rounds):
        base = t0 + r * round_len
        for label, dur in plan:
            lo, hi = base + (base - t0) % 1, base + (base - t0) % 1  # placeholder
            start = base + sum(d for _, d in plan[:plan.index((label, dur))])
            seg_ev = seg(ev, start, start + dur, {"det", "miss", "ear", "frame"})
            dets = [x for x in seg_ev if x["kind"] == "det"]
            misses = [x for x in seg_ev if x["kind"] == "miss"]
            ears = [x["ear"] for x in seg_ev if x["kind"] == "ear"]
            mars = [x["mar"] for x in seg_ev if x["kind"] == "frame"]
            rec = {
                "round": r + 1, "action": label, "t_from": start - t0 + r * 120,
                "detections": len(dets), "misses": len(misses),
                "score": stat([x["score"] for x in dets]),
                "cand": stat([x["cand"] for x in dets]),
                "side": stat([x["side"] for x in dets], 1),
                "cx": [round(min((x["cx"] for x in dets), default=0), 1),
                       round(max((x["cx"] for x in dets), default=0), 1)],
                "cy": [round(min((x["cy"] for x in dets), default=0), 1),
                       round(max((x["cy"] for x in dets), default=0), 1)],
                "ear": stat(ears), "mar": stat(mars),
            }
            # box jitter between consecutive detections (matters for head turns)
            if label in ("QUAY DAU SANG PHAI", "QUAY DAU SANG TRAI", "CUI XUONG BAN") and len(dets) > 1:
                jc, js = [], []
                for p, q in zip(dets, dets[1:]):
                    if p["side"] > 1:
                        jc.append(abs(q["cx"] - p["cx"]) / p["side"])
                        js.append(abs(q["side"] - p["side"]) / p["side"])
                rec["jump_cx_pct"] = round(100 * max(jc), 1) if jc else None
                rec["jump_side_pct"] = round(100 * max(js), 1) if js else None
            out["segments"].append(rec)
            print(f"R{r+1} {label:<22} det={len(dets):2d} miss={len(misses):2d} "
                  f"score_med={rec['score'].get('median','-')} cand_med={rec['cand'].get('median','-')} "
                  f"side_med={rec['side'].get('median','-')} EAR={rec['ear'].get('median','-')} "
                  f"MAR={rec['mar'].get('median','-')}")

    # ---- gates ----
    face_acts = {"NHIN THANG - MAT MO", "MO MAT", "NGAP", "QUAY DAU SANG PHAI",
                 "QUAY DAU SANG TRAI", "CUI XUONG BAN"}
    det_all = [x for x in ev if x["kind"] == "det"]
    det_face = [x for x in ev if x["kind"] == "det" and x["score"] > 0]
    closed_ear = [s["ear"]["median"] for s in out["segments"]
                  if s["action"] == "NHAM MAT" and s["ear"]["n"] > 0]
    open_ear = [s["ear"]["median"] for s in out["segments"]
                if s["action"] in ("NHIN THANG - MAT MO", "MO MAT") and s["ear"]["n"] > 0]
    scores_face = [s["score"]["median"] for s in out["segments"] if s["action"] in face_acts
                   and s["score"]["n"] > 0]
    gates = {
        "detector_runs": len([x for x in ev if x["kind"] in ("det", "miss")]),
        "frames_with_face_detected": len(det_face),
        "misses": len([x for x in ev if x["kind"] == "miss"]),
        "score_median_overall": stat([x["score"] for x in det_all]).get("median"),
        "score_p10_overall": round(sorted(x["score"] for x in det_all)[max(0, int(0.1 * (len(det_all) - 1)))], 3) if det_all else 0,
        "ms_median": stat([x["ms"] for x in det_all], 1).get("median"),
        "ear_open_median": stat(open_ear).get("median"),
        "ear_closed_median": stat(closed_ear).get("median"),
        "ear_drop_pct": None,
        "resets": sum(1 for line in open(a.log, encoding="utf-8", errors="replace")
                      if "ESP-ROM:esp32s3" in line or "Rebooting" in line),
        "watchdog_events": sum(1 for line in open(a.log, encoding="utf-8", errors="replace")
                               if "task_wdt" in line),
    }
    if gates["ear_open_median"] and gates["ear_closed_median"]:
        gates["ear_drop_pct"] = round(100.0 * (1 - gates["ear_closed_median"] / gates["ear_open_median"]), 1)
    gates["pass_detect"] = bool(det_all) and gates["score_median_overall"] >= 0.90
    gates["pass_ear"] = bool(gates["ear_drop_pct"] and gates["ear_drop_pct"] >= 25 and
                             gates["ear_closed_median"] < 0.15)
    gates["pass_stability"] = gates["resets"] <= 1 and gates["watchdog_events"] == 0
    out["gates"] = gates
    print("\n--- GATES ---")
    print(json.dumps(gates, ensure_ascii=False, indent=1, default=float))
    dst = a.log.rsplit(".", 1)[0] + "_session.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print("saved", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
