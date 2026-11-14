"""Cue for the BlazeFace live measurement session (Project 13).

Prints a 2-minute action timeline per round with a live countdown, so the person
in front of the camera only has to follow the screen. The board samples once per
~8.2 s (PIPNet 0.12 FPS), therefore every action is held well over one frame.

Usage:
  py tools\\live_session_cue.py --rounds 3            # 3 rounds, real time
  py tools\\live_session_cue.py --rounds 1 --speed 20  # smoke test (6 s)
"""
import argparse
import sys
import time

# (label, seconds, hint printed once at the transition)
PLAN = [
    ("NHIN THANG - MAT MO", 20, "nhin thang vao camera, mat mo binh thuong"),
    ("NHAM MAT", 15, "nham mat, giu nguyen, KHONG nghieng dau"),
    ("MO MAT", 10, "mo mat, van nhin thang"),
    ("NGAP", 8, "ngap that, mieng mo rong"),
    ("QUAY DAU SANG PHAI", 15, "quay dau ~45 do sang phai, giu"),
    ("QUAY DAU SANG TRAI", 15, "quay dau ~45 do sang trai, giu"),
    ("CUI XUONG BAN", 15, "cui xuong nhin ban phim, giu"),
    ("RA KHOI KHUNG", 15, "dung day / lui ra ngoai khung hinh"),
    ("NGHI", 7, "ve tu the ban dau, thu gian"),
]
ACCEPT_PLAN = [
    ("NHIN THANG - MAT MO", 25, "nhin thang camera, mat mo (can 3 mau)"),
    ("NHAM MAT", 30, "nham mat, giu lau (can >= 2 mau ~16 s de bao)"),
    ("MO MAT", 20, "mo mat, van nhin thang"),
    ("NGAP", 15, "ngap that, mieng mo rong"),
    ("QUAY DAU SANG PHAI", 20, "quay ~45 do sang phai"),
    ("QUAY DAU SANG TRAI", 20, "quay ~45 do sang trai"),
    ("CUI XUONG BAN", 20, "cui xuong, giu"),
    ("RA KHOI KHUNG", 20, "dung day/lui ra ngoai"),
]
ROUND_SECONDS = sum(p[1] for p in PLAN)


def run_round(rnd, total, speed, plan=None):
    plan = plan or PLAN
    print(f"\n=== VONG {rnd}/{total} ({ROUND_SECONDS} s) ===")
    t0 = time.monotonic()
    for label, dur, hint in plan:
        offset = int(time.monotonic() - t0)
        print(f"\a[{offset:3d}s] >>> {label}  ({dur}s)  - {hint}")
        end = time.monotonic() + dur
        while True:
            left = end - time.monotonic()
            if left <= -0.05:
                break
            sys.stdout.write(f"\r      {label:<22} con {max(0.0, left):4.1f}s  ")
            sys.stdout.flush()
            time.sleep(min(0.2 / speed, max(0.01, left)))
        sys.stdout.write("\r" + " " * 60 + "\r")
    print(f"  --- het vong {rnd} ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0, help=">1 = chay nhanh de thu")
    ap.add_argument("--start-delay", type=int, default=5)
    ap.add_argument("--acceptance", action="store_true", help="dung kich ban nghiem thu (nham 30 s)")
    ap.add_argument("--closed-seconds", type=int, default=0, help="ghi de thoi gian NHAM MAT")
    a = ap.parse_args()
    plan = list(ACCEPT_PLAN if a.acceptance else PLAN)
    if a.closed_seconds > 0:
        plan = [(l, a.closed_seconds if l == "NHAM MAT" else d, h) for l, d, h in plan]
    round_seconds = sum(p[1] for p in plan)

    print("=" * 64)
    print("BUOI DO LIVE BLAZEFACE - PROJECT 13" + (" [NGHIEM THU]" if a.acceptance else ""))
    print(f"{a.rounds} vong x {round_seconds} s ~ {a.rounds * round_seconds // 60} phut")
    print("Yeu cau: camera ngang tam mat, cach 60-70 cm, mat TRON trong khung,")
    print("den chieu tu phia truoc, khong noi chuyen (tru dong tac ngap).")
    print("=" * 64)
    for i in range(a.start_delay, 0, -1):
        sys.stdout.write(f"\rBat dau sau {i} s ... (ngoi vao vi tri) ")
        sys.stdout.flush()
        time.sleep(1.0 / a.speed)
    sys.stdout.write("\r" + " " * 60 + "\r")

    for r in range(1, a.rounds + 1):
        run_round(r, a.rounds, a.speed, plan)

    print("\n" + "=" * 64)
    print("XONG. Dung yen 5 s roi bao agent: 'xong'")
    print("Log dang ghi: output/blaze_iq_20261113/serial_live5.log")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
