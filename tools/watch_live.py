"""Live dashboard for the Project 13 board (ESP32-S3) over USB serial.

Reads COM3, strips the ESP-IDF ANSI colour codes and prints one readable line per event:
  * detector result   (score / candidates / box / detector ms)
  * frame line        (FPS, decode/AI/total ms, ROI, EAR vs threshold, MAR, yaw, eye state, FSM status)
  * alarms            (MICROSLEEP / DISTRACTION / YAWN) highlighted
  * PIPNet phase split ([LM split] preprocess/model/decode)

Usage:
  py tools\\watch_live.py                  # dashboard, no timestamps (cleaner)
  py tools\\watch_live.py --port COM3 --baud 115200
  py tools\\watch_live.py --raw            # plain serial dump (everything)

Stop with Ctrl+C.
"""
import argparse
import re
import sys
import time

import serial

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^\[(\d\d:\d\d:\d\d)\]")
IDF = re.compile(r"^([IWED]) \((\d+)\) ([A-Z_0-9]+):\s*(.*)$")
DET = re.compile(r"blaze (OK|MISS) cand=(\d+) score=([\d.]+) box=([-\d.,]+).*det=(\d+)ms")
DETMISS = re.compile(r"blaze MISS cand=(\d+) -> centre ROI")
FRAME = re.compile(r"\[Frame #\s*(\d+)\] FPS:\s*([\d.]+).*?Dec:\s*([\d.]+)ms \| AI:\s*([\d.]+)ms.*?Total:\s*([\d.]+)ms"
                   r".*?ROI: ([\d,-]+).*?EAR: ([\d.]+) \(thr ([\d.]+) base ([\d.]+)\).*?MAR: ([\d.]+)"
                   r".*?Yaw: ([+-]?[\d.]+).*?bias ([+-]?[\d.]+)\).*?eye: (\w+).*?Status: (.+?)\s*$")
SPLIT = re.compile(r"\[LM split\] preprocess ([\d.]+) ms \| model ([\d.]+) ms \| decode ([\d.]+) ms")
CALIB = re.compile(r"Hiệu chuẩn thành công.*?EAR=([\d.]+)")
CALIBS = re.compile(r"calib sample (\d+)/(\d+)")

GREEN, RED, YELLOW, CYAN, GREY, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[0m"
ALARM_WORDS = ("MICROSLEEP", "DISTRACTION", "YAWN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # so the dashboard streams when redirected
    except Exception:
        pass

    try:
        ser = serial.Serial(a.port, a.baud, timeout=1.0)
    except Exception as e:                                    # noqa: BLE001
        print(f"Khong mo duoc {a.port}: {e}")
        print("Goi y: dong Serial Monitor / cua so khac dang giu cong.")
        return 1

    print(f"# listening on {a.port} @ {a.baud} - Ctrl+C to stop")
    print(f"{'time':8} {'frame':>5} {'FPS':>4} {'Dec':>5} {'AI':>5} {'Total':>6}  "
          f"{'EAR':>5} {'thr':>5} {'yaw-bias':>8}  {'eye':<10} status")
    print("-" * 104)
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace")
            if not line:
                continue
            txt = ANSI.sub("", line).rstrip()
            if a.raw:
                print(txt)
                continue
            m = TS.match(txt)
            stamp = m.group(1) if m else ""
            m = IDF.match(txt)
            if not m:
                continue
            lvl, tag, msg = m.group(1), m.group(3), m.group(4)

            if "LM split" in msg:
                s = SPLIT.search(msg)
                if s:
                    print(f"{GREY}{stamp} PIPNet: preprocess {float(s.group(1)):.0f} ms | "
                          f"model {float(s.group(2)):.0f} ms | decode {float(s.group(3)):.0f} ms{RESET}")
                continue

            if tag in ("BLAZE", "DETLITE"):
                d = DET.search(msg)
                if d:
                    ok = d.group(1) == "OK"
                    sc = float(d.group(3))
                    col = GREEN if (ok and sc >= 0.6) else YELLOW
                    print(f"{col}{stamp} detector {d.group(1)}  score {sc:.3f}  cand {d.group(2)}  "
                          f"box [{d.group(4)}]  {d.group(5)} ms{RESET}")
                    continue
                dm = DETMISS.search(msg)
                if dm:
                    print(f"{GREY}{stamp} detector MISS (cand {dm.group(1)}) -> ROI giua khung{RESET}")
                    continue

            if tag == "ADAS_CONTROLLER":
                c = CALIB.search(msg)
                if c:
                    print(f"{CYAN}{stamp} ✅ Hieu chuan xong — baseline EAR {float(c.group(1)):.2f}{RESET}")
                    continue
                cs = CALIBS.search(msg)
                if cs:
                    print(f"{GREY}{stamp} hieu chuan {cs.group(1)}/{cs.group(2)} mau{RESET}")
                    continue

            if "Frame #" in msg:
                f = FRAME.search(msg)
                if not f:
                    continue
                (idx, fps, dec, ai, total, roi, ear, thr, base,
                 mar, yaw, bias, eye, status) = f.groups()
                alarm = any(w in status for w in ALARM_WORDS)
                col = RED if alarm else GREEN
                print(f"{col}{stamp} #{idx:>5} {float(fps):>4.1f} {float(dec):>5.0f} {float(ai):>5.0f} "
                      f"{float(total):>6.0f}  {float(ear):>5.3f} {float(thr):>5.3f} "
                      f"{float(yaw) - float(bias):>+8.1f}  {eye:<10} {status}{RESET}")
    except KeyboardInterrupt:
        print("\n# dung.")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
