"""
Phan tich TRACKING/LAG cua ESP32 bang SO LIEU (khong cam nhan).
=============================================================
Doc file JSON tho do `esp32_log_capture.py` sinh ra (output/esp32_track_*.jsonl,
moi dong 1 frame co: rx,ry,rs + 22 landmark chuan hoa + ear/mar/yaw/fps + timestamp).

Tinh cac chi so KHACH QUAN:
  1) SAI SO KHUNG CAT (px): khoang cach giua TAM CANONICAL suy tu 22 diem (chuan
     luc train) va TAM KHUNG CAT thuc te. Dung yen -> ~0. Di chuyen -> tang.
  2) LAG (ms): chieu sai so len HUONG CHUYEN DONG roi chia van toc:
        err_along = |err| * cos(goc giua err va vector van toc)
        lag       = err_along / speed            (giay)
     => do TRUC TIEP do tre bam mat.
  3) RUNG (jitter) khi dung yen: do lech chuan sai so khung + do lech chuan 22 diem.
  4) SAI SO KICH THUOC khung cat so voi canonical.
  5) Thong ke FPS / Dec / AI / EAR / MAR / Yaw va cac canh bao.

Dung:
    python tools/analyze_tracking.py                       (file moi nhat)
    python tools/analyze_tracking.py --file output/esp32_track_XXXX.jsonl
    python tools/analyze_tracking.py --csv output/track.csv
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            lm = d.get("landmarks")
            rs = int(d.get("rs", 0))
            if not lm or len(lm) != 44 or rs <= 0:
                continue
            rows.append(d)
    return rows


def canonical_anchor(lm_frame):
    """lm_frame: (22,2) toa do PIXEL khung goc. Tra (cx, cy, S) chuan luc train."""
    eA = lm_frame[0:6].mean(axis=0)
    eB = lm_frame[6:12].mean(axis=0)
    eye_x, eye_y = (eA[0] + eB[0]) * 0.5, (eA[1] + eB[1]) * 0.5
    nose_x, nose_y = lm_frame[19, 0], lm_frame[19, 1]
    chin_y = lm_frame[21, 1]
    d_eyes = math.hypot(eB[0] - eA[0], eB[1] - eA[1])
    d_eye_nose = max(nose_y - eye_y, 0.45 * d_eyes, 1.0)
    d_eye_chin = max(chin_y - eye_y, d_eye_nose, 1.0)
    h_skull = max(2.10 * d_eye_nose, 1.40 * d_eyes, d_eye_chin / 1.20)
    return (eye_x + nose_x) * 0.5, eye_y + 0.32 * h_skull, 2.05 * h_skull


def main():
    ap = argparse.ArgumentParser(description="Phan tich tracking/lag ESP32 bang so lieu")
    ap.add_argument("--file", default=None, help="file .jsonl (mac dinh: moi nhat)")
    ap.add_argument("--still-speed", type=float, default=30.0, help="nguong coi la dung yen (px/s)")
    ap.add_argument("--csv", default=None, help="xuat CSV tung frame")
    args = ap.parse_args()

    path = args.file
    if path is None:
        ptr = os.path.join(ROOT, "output", "latest_esp32_track.txt")
        if os.path.exists(ptr):
            path = open(ptr, encoding="utf-8").read().strip()
        if not path or not os.path.exists(path):
            print("❌ Khong thay file jsonl. Chay: python tools/esp32_log_capture.py")
            sys.exit(1)

    rows = load(path)
    if len(rows) < 30:
        print(f"⚠️  Chi co {len(rows)} frame hop le trong {path} — can >=30 de phan tich.")
        sys.exit(1)

    t = np.array([float(d.get("_t", 0.0)) for d in rows])
    rx = np.array([float(d.get("rx", 0)) for d in rows])
    ry = np.array([float(d.get("ry", 0)) for d in rows])
    rs = np.array([float(d.get("rs", 0)) for d in rows])
    fps = np.array([float(d.get("fps", 0)) for d in rows])
    dec = np.array([float(d.get("dec", 0)) for d in rows])
    ai = np.array([float(d.get("ai", 0)) for d in rows])
    tot = np.array([float(d.get("total", 0)) for d in rows])
    ear = np.array([float(d.get("ear", 0)) for d in rows])
    mar = np.array([float(d.get("mar", 0)) for d in rows])
    yaw = np.array([float(d.get("yaw", 0)) for d in rows])

    fcx = np.zeros(len(rows)); fcy = np.zeros(len(rows)); fS = np.zeros(len(rows))
    for i, d in enumerate(rows):
        lm = np.array(d["landmarks"], dtype=np.float64).reshape(22, 2)
        lm_f = np.stack([rx[i] + lm[:, 0] * rs[i], ry[i] + lm[:, 1] * rs[i]], axis=1)
        cx, cy, S = canonical_anchor(lm_f)
        fcx[i], fcy[i], fS[i] = cx, cy, S

    # sai so khung cat: TAM CHUAN (tu 22 diem) so voi TAM KHUNG THUC TE
    err_x = fcx - (rx + rs * 0.5)
    err_y = fcy - (ry + rs * 0.5)
    err = np.hypot(err_x, err_y)
    err_size = fS - rs

    # van toc mat (px/s) tu vi tri mat toan khung
    dt = np.diff(t); dt[dt <= 0] = np.nan
    vx = np.diff(fcx) / dt; vy = np.diff(fcy) / dt
    speed = np.hypot(vx, vy)
    speed = np.concatenate([[np.nan], speed])
    ok = np.isfinite(speed) & np.isfinite(err)

    still = ok & (speed < args.still_speed)
    moving = ok & (speed >= args.still_speed)

    # [v2] Loai cac frame MIENG MO (ngap/noi chuyen) khoi thong ke "dung yen":
    # ngap lam ham/mieng di chuyen that -> neu tinh vao se bao rung gia.
    mouth_open = np.array([float(d.get("mar", 0.0)) for d in rows]) > 0.45
    still_clean = still & (~mouth_open) & (np.array([str(d.get("status", "")) for d in rows]) == "NORMAL")

    # [v2] Rung tung diem landmark (px khung goc) + chuan hoa theo canh crop
    L = np.zeros((len(rows), 22, 2))
    for i, d in enumerate(rows):
        a = np.array(d["landmarks"], dtype=np.float64).reshape(22, 2)
        L[i, :, 0] = rx[i] + a[:, 0] * rs[i]
        L[i, :, 1] = ry[i] + a[:, 1] * rs[i]
    S_mean = rs[still_clean].mean() if still_clean.sum() > 5 else rs.mean()

    def jitter_regions(mask):
        if mask.sum() < 5:
            return None, None
        tot2 = np.array([L[mask, j, :].std(axis=0).mean() for j in range(22)])
        hf2 = np.array([np.mean(np.hypot(np.diff(L[mask, j, 0]), np.diff(L[mask, j, 1])))
                        for j in range(22)])
        return tot2, hf2

    print("=" * 74)
    print(f"PHAN TICH TRACKING ESP32 | {os.path.basename(path)}")
    print(f"frame hop le: {len(rows)} | thoi luong: {t[-1]-t[0]:.1f}s | "
          f"FPS tb: {fps.mean():.2f} | Total tb: {tot.mean():.1f}ms "
          f"(Dec {dec.mean():.1f} | AI {ai.mean():.1f})")
    print("=" * 74)

    jr, jhf = jitter_regions(still_clean)
    if jr is not None:
        print(f"\n[1] KHI DUNG YEN THAT SU ({int(still_clean.sum())} frame: loc bo frame ngap/mieng mo)")
        print(f"    Rung khung cat   : err tb={err[still_clean].mean():5.2f}px  std={err[still_clean].std():5.2f}px")
        print(f"    Rung kich thuoc  : |dSize| tb={np.abs(err_size[still_clean]).mean():5.2f}px")
        print(f"    --- RUNG 22 DIEM (crop tb {S_mean:.0f}px) ---")
        print(f"        {'':>12} {'TONG(std)':>12} {'TAN SO CAO':>12}  (px / % crop)")
        for nm, sl in (("TB 22 diem", slice(0, 22)), ("Vung MAT", slice(0, 12)),
                       ("Vung MIENG", slice(12, 18)), ("MUI/CAM", slice(18, 22))):
            print(f"        {nm:>12} {jr[sl].mean():5.2f}px {100*jr[sl].mean()/S_mean:5.2f}% "
                  f"{jhf[sl].mean():7.2f}px {100*jhf[sl].mean()/S_mean:5.2f}%")
        print(f"    (TAN SO CAO = rung tung frame - day la 'rung' mat nhin thay duoc)")
    if moving.sum() > 5:
        ang = np.arctan2(vy, vx)
        proj = err_x[1:] * np.cos(ang) + err_y[1:] * np.sin(ang)   # err . huong di
        proj = np.concatenate([[np.nan], proj])
        m2 = moving & np.isfinite(proj)
        lag_ms = (proj[m2] / speed[m2]) * 1000.0
        lag_ms = lag_ms[np.abs(lag_ms) < 2000.0]
        print(f"\n[2] KHI DI CHUYEN ({int(moving.sum())} frame, speed>={args.still_speed:.0f}px/s)")
        print(f"    Toc do mat tb    : {speed[moving].mean():6.0f} px/s")
        print(f"    Sai so khung tb  : {err[moving].mean():5.2f}px")
        if lag_ms.size > 3:
            print(f"    >>> DO TRE (LAG) : tb={lag_ms.mean():6.0f} ms | trung vi={np.median(lag_ms):6.0f} ms")
            print(f"        (lag = sai so chieu len huong di / toc do. Ly tuong < ~150ms)")
    else:
        print("\n[2] KHI DI CHUYEN: khong du frame — hay quay/di chuyen dau nhieu hon khi test!")

    print(f"\n[3] KHUNG CAT: size thuc te tb={rs.mean():.0f}px | size canonical tb={fS.mean():.0f}px "
          f"| lech tb={err_size.mean():+.1f}px (std {err_size.std():.1f})")
    print(f"\n[4] SINH TON: EAR min={ear.min():.2f} tb={ear.mean():.2f} | "
          f"MAR min={mar.min():.2f} max={mar.max():.2f} | Yaw {yaw.min():+.0f}..{yaw.max():+.0f}")
    st = {}
    for d in rows:
        st[d.get("status", "?")] = st.get(d.get("status", "?"), 0) + 1
    for k, v in sorted(st.items(), key=lambda kv: -kv[1]):
        print(f"    {v:4d}x  {k}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("i,t,fcx,fcy,fS,rx,ry,rs,err,err_size,speed\n")
            for i in range(len(rows)):
                f.write(f"{i},{t[i]:.3f},{fcx[i]:.1f},{fcy[i]:.1f},{fS[i]:.1f},"
                        f"{rx[i]:.0f},{ry[i]:.0f},{rs[i]:.0f},{err[i]:.2f},{err_size[i]:.1f},"
                        f"{speed[i] if np.isfinite(speed[i]) else 0:.0f}\n")
        print(f"\n[CSV] Da luu: {args.csv}")


if __name__ == "__main__":
    main()
