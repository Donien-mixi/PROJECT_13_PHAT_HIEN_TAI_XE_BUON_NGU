import argparse
import datetime
import sys
import time

import serial

p = argparse.ArgumentParser()
p.add_argument("--port", default="COM3")
p.add_argument("--baud", type=int, default=115200)
p.add_argument("--seconds", type=float, default=45.0)
p.add_argument("--out", required=True)
a = p.parse_args()

ser = serial.Serial(a.port, a.baud, timeout=0.3)
try:
    ser.setDTR(False)
    ser.setRTS(False)
except Exception:
    pass
t0 = time.time()
lines = []
with open(a.out, "w", encoding="utf-8", errors="replace") as f:
    while time.time() - t0 < a.seconds:
        try:
            raw = ser.readline()
        except Exception as e:  # noqa: BLE001
            f.write(f"# serial error: {e}\n")
            break
        if not raw:
            continue
        s = raw.decode("utf-8", errors="replace").rstrip()
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{stamp}] {s}\n")
        f.flush()
        lines.append(s)
ser.close()
tail = "\n".join(lines[-120:])
try:
    print(tail)
except UnicodeEncodeError:   # console in cp1252/cp437: log is fine, only the echo breaks
    print(tail.encode("ascii", "replace").decode("ascii"))
print(f"# saved {len(lines)} lines -> {a.out}")
