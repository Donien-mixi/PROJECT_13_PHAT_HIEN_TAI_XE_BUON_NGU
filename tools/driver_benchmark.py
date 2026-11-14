"""Driver-domain benchmark on OV5640-like frames (no live subjects needed).

Builds a benchmark from the local datasets AFTER converting them to the board's image
domain (tools/ov5640_domain.py), then evaluates the firmware chain:

  Phase A - 20 YawDD drivers (open eyes, dash-cam, glass/no-glass):
            detector hit-rate at 0.50/0.60 + ROI-scale sweep (k = 1.08 / 1.20 / 1.35)
            -> EAR plausibility and landmark-bbox ratio (fixes the "close range" bug).
  Phase B - 40 CEW open + 40 CEW closed identities in the same domain:
            EAR distributions + miss / false-alarm rates for three threshold rules.

Cached frames: output/driver_bench_20261114/

Usage:
  py tools\\driver_benchmark.py --n-yawdd 20 --n-cew 40
"""
import argparse
import glob
import json
import os
import random
import statistics as st

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf

from ov5640_domain import load_stats, to_ov5640

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFL = os.path.join(ROOT, r"output\port_models_20261110\blaze_iq_mp\blazeface_nchw_full_integer_quant.tflite")
ONNX_98 = os.path.join(ROOT, r"output\port_models_20261110\pipnet98_resnet18.onnx")
CACHE = os.path.join(ROOT, r"output\driver_bench_20261114")
OUT = os.path.join(CACHE, "benchmark.json")
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


class Pipe:
    def __init__(self):
        self.interp = tf.lite.Interpreter(model_path=TFL)
        self.interp.allocate_tensors()
        self.det = self.interp.get_input_details()[0]
        self.outs = {int(t["shape"][-1]): t for t in self.interp.get_output_details()}
        self.anc = anchors()
        self.s98 = ort.InferenceSession(ONNX_98, providers=["CPUExecutionProvider"])

    def detect(self, gray):
        qs, qz = self.det["quantization"]
        s = 128.0 / float(max(W, H))
        nw, nh = int(round(W * s)), int(round(H * s))
        im = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
        c = np.zeros((128, 128), np.uint8)
        c[(128 - nh) // 2:(128 - nh) // 2 + nh, (128 - nw) // 2:(128 - nw) // 2 + nw] = im
        x = c.astype(np.float32) / 127.5 - 1.0
        q = np.clip(np.rint(x / qs) + qz, -128, 127).astype(np.int8)
        self.interp.set_tensor(self.det["index"], np.stack([q] * 3, -1)[None])
        self.interp.invoke()
        c1, r1 = self.outs[1], self.outs[16]
        cc = (self.interp.get_tensor(c1["index"]).reshape(-1).astype(np.float32) - c1["quantization"][1]) * c1["quantization"][0]
        rr = (self.interp.get_tensor(r1["index"])[0].astype(np.float32) - r1["quantization"][1]) * r1["quantization"][0]
        sc = 1.0 / (1.0 + np.exp(-np.clip(cc, -60, 60)))
        i = int(np.argmax(sc))
        padx, pady = (128 - nw) // 2, (128 - nh) // 2
        cx = (rr[i, 0] + self.anc[i, 0] * 128 - padx) / s
        cy = (rr[i, 1] + self.anc[i, 1] * 128 - pady) / s
        w = rr[i, 2] / s
        h = rr[i, 3] / s
        return float(sc[i]), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def ear(self, gray, box, k, crop_scale=1.5):
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        sz = max(box[2] - box[0], box[3] - box[1])
        cs = k * sz
        fcx, fcy = cx + 0.040 * sz, cy - 0.028 * sz
        side = min(max(cs * crop_scale, 64.0), float(min(W, H)))
        x0 = max(0.0, min(fcx - side / 2, W - side))
        y0 = max(0.0, min(fcy - side / 2, H - side))
        step = side / 256.0
        yy = np.clip((y0 + (np.arange(256) + 0.5) * step).astype(int), 0, H - 1)
        xx = np.clip((x0 + (np.arange(256) + 0.5) * step).astype(int), 0, W - 1)
        crop = gray[np.ix_(yy, xx)]
        rgb = np.stack([crop] * 3, -1).astype(np.float32)
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
        cls, xo, yo, _, _ = self.s98.run(None, {"input": blob})
        G = cls.shape[-1]
        cells = G * G
        cls = cls.reshape(98, cells)
        xo = xo.reshape(98, cells)
        yo = yo.reshape(98, cells)
        ids = cls.argmax(1)
        nx = ((ids % G) + xo[np.arange(98), ids]) / G
        ny = ((ids // G) + yo[np.arange(98), ids]) / G
        pts = np.stack([x0 + nx * side, y0 + ny * side], 1)
        d = lambda i, j: float(np.linalg.norm(pts[i] - pts[j]))
        D = d(60, 68)
        if D < 1e-4:
            return 0.0, 0.0
        e = ((d(61, 66) + d(62, 65)) * 0.5 / D + (d(69, 74) + d(70, 73)) * 0.5 / D) * 0.5
        bw = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
        return float(e), float(bw / max(1e-3, sz))


def build_yawdd(n_want, stats, seed=3):
    """One OV5640-like frame per driver from YawDD, spread over face sizes 80-125 px."""
    rng = random.Random(seed)
    vids = sorted(glob.glob(os.path.join(ROOT, r"datasets\raw_faces\YawDD\**\*.avi"), recursive=True))
    rng.shuffle(vids)
    out, seen = [], {}
    for i, v in enumerate(vids):
        if len(out) >= n_want:
            break
        key = os.path.dirname(v)
        if seen.get(key, 0) >= 1:
            continue
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            cap.release()
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        got = None
        for idx in range(int(0.15 * n), max(30, int(0.85 * n)), max(12, n // 12)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            face_px = rng.choice([88, 98, 108, 118])
            img = to_ov5640(g, stats, face_px=face_px, face_frac=0.30, seed=rng.randrange(1 << 30))
            got = (img, os.path.basename(v), face_px)
            break
        cap.release()
        if got:
            seen[key] = 1
            out.append(got)
    return out


def build_cew(sub, n_want, stats, seed=4):
    rng = random.Random(seed)
    files = glob.glob(os.path.join(ROOT, rf"datasets\raw_faces\CEW\CEW\{sub}\*.jpg"))
    rng.shuffle(files)
    out = []
    for f in files[: n_want * 3]:
        if len(out) >= n_want:
            break
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out.append((to_ov5640(img, stats, face_px=100, face_frac=1.0, seed=rng.randrange(1 << 30)),
                    os.path.basename(f)))
    return out


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-yawdd", type=int, default=20)
    ap.add_argument("--n-cew", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    stats = load_stats()
    pipe = Pipe()

    # ---------------- Phase A: YawDD drivers, ROI-scale sweep ----------------
    frames = build_yawdd(a.n_yawdd, stats)
    rows = []
    for i, (img, name, face_px) in enumerate(frames):
        cv2.imwrite(os.path.join(CACHE, f"yaw{i:02d}.png"), img)
        sc, box = pipe.detect(img)
        rec = {"i": i, "clip": name, "face_target": face_px, "score": round(sc, 3)}
        if sc >= 0.50:
            rec["side"] = round(max(box[2] - box[0], box[3] - box[1]), 1)
            rec["k"] = {}
            for k in (1.08, 1.20, 1.35, 1.50):
                e, ratio = pipe.ear(img, box, k)
                rec["k"][str(k)] = {"ear": round(e, 4), "bbox_ratio": round(ratio, 3),
                                    "plausible": bool(0.05 <= e <= 0.45 and 0.60 <= ratio <= 1.40)}
        rows.append(rec)

    det = [r for r in rows if r["score"] >= 0.50]
    det60 = [r for r in rows if r["score"] >= 0.60]
    sweep = {}
    for k in ("1.08", "1.2", "1.35", "1.5"):
        ears = [r["k"][k]["ear"] for r in det if k in r.get("k", {})]
        rats = [r["k"][k]["bbox_ratio"] for r in det if k in r.get("k", {})]
        ok = [r["k"][k]["plausible"] for r in det if k in r.get("k", {})]
        if ears:
            sweep[k] = {"n": len(ears), "ear_median": pct(ears, 0.5),
                        "ear_in_0.10_0.35": round(sum(1 for e in ears if 0.10 <= e <= 0.35) / len(ears), 3),
                        "ratio_median": round(st.median(rats), 2),
                        "plausible_rate": round(sum(1 for v in ok if v) / len(ok), 3)}

    # ---------------- Phase B: EAR open vs closed in the same domain ----------------
    opens = build_cew("open", a.n_cew, stats)
    closeds = build_cew("closed", a.n_cew, stats)
    for tag, lst in (("open", opens), ("closed", closeds)):
        for i, (img, name) in enumerate(lst):
            cv2.imwrite(os.path.join(CACHE, f"cew_{tag}{i:02d}.png"), img)
    res = {}
    for tag, lst in (("open", opens), ("closed", closeds)):
        ears, hits = [], 0
        for img, name in lst:
            sc, box = pipe.detect(img)
            if sc < 0.50:
                continue
            hits += 1
            e, _ = pipe.ear(img, box, 1.08)
            ears.append(e)
        res[tag] = {"n_src": len(lst), "detected": hits, "ear": ears}
    eo, ec = res["open"]["ear"], res["closed"]["ear"]
    base75 = pct(eo, 0.75)
    rules = {
        "R1_fixed_0.15": 0.15,
        "R2_firmware_0.70xb75_floor0.10": round(min(0.25, max(0.10, 0.70 * base75)), 3),
        "R3_firmware_old_floor0.15": round(min(0.25, max(0.15, 0.70 * base75)), 3),
    }
    rule_stats = {name: {"threshold": thr,
                         "closed_below": round(sum(1 for v in ec if v < thr) / max(1, len(ec)), 3),
                         "open_below_false_alarm": round(sum(1 for v in eo if v < thr) / max(1, len(eo)), 3)}
                  for name, thr in rules.items()}

    summary = {
        "domain": stats,
        "phaseA_yawdd": {
            "n": len(rows), "detected_0.50": len(det), "detected_0.60": len(det60),
            "rate_0.50": round(len(det) / max(1, len(rows)), 3),
            "rate_0.60": round(len(det60) / max(1, len(rows)), 3),
            "score_median": pct([r["score"] for r in rows], 0.5),
            "side_median": round(st.median([r["side"] for r in det]), 1) if det else None,
            "roi_sweep": sweep,
        },
        "phaseB_ear": {
            "open": {"n": len(eo), "median": pct(eo, 0.5), "p10": pct(eo, 0.1), "p90": pct(eo, 0.9)},
            "closed": {"n": len(ec), "median": pct(ec, 0.5), "p10": pct(ec, 0.1), "p90": pct(ec, 0.9)},
            "detected": {"open": res["open"]["detected"], "closed": res["closed"]["detected"]},
            "threshold_rules": rule_stats,
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "yawdd_rows": rows}, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
