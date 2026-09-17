"""
ESP32 Edge-AI Realtime Monitor (DISPLAY-ONLY) - v2
==================================================
Nhận UDP :8889 từ ESP32 và vẽ realtime:
  - ẢNH TOÀN KHUNG camera OV5640 (packet magic AA56A001) + khung ROI đang crop
  - Ảnh 96x96 mà MODEL thực sự nhìn thấy (magic AA56AA56) + 22 điểm mốc
  - EAR / MAR / Yaw / Pitch / Roll / trạng thái / còi / ngưỡng / ROI
  - Đồ thị trượt EAR / MAR / Yaw / Pitch + thời gian Dec/AI/Total

Laptop là MÀN HÌNH, không chạy AI.
Dùng:
    conda activate projet_13
    python host_laptop/esp_display_monitor.py
Phím: [Q]/[ESC] thoát | [Space] tạm dừng đồ thị.
"""

import argparse
import json
import os
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np

if os.name == "nt":
    os.system("")

EYE_L = [0, 1, 2, 3, 4, 5, 0]
EYE_R = [6, 7, 8, 9, 10, 11, 6]
MOUTH = [12, 14, 15, 13, 12]
MOUTH_IN = [12, 16, 17, 13]
EAR_L = [(1, 5), (2, 4), (0, 3)]
EAR_R = [(7, 11), (8, 10), (6, 9)]
MAR_L = [(14, 15), (16, 17)]
NOSE_CHIN = [18, 19, 20, 21]

W, H = 1180, 660
HIST = 200
hist_ear = deque(maxlen=HIST)
hist_mar = deque(maxlen=HIST)
hist_yaw = deque(maxlen=HIST)
hist_pitch = deque(maxlen=HIST)


def draw_mesh(canvas, lm):
    h, w = canvas.shape[:2]
    P = (lm * np.array([w, h], dtype=np.float32)).astype(np.int32)

    def poly(idx, color, thick=1):
        cv2.polylines(canvas, [P[idx].reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)

    poly(EYE_L, (0, 220, 0))
    poly(EYE_R, (0, 220, 0))
    poly(MOUTH, (0, 180, 255))
    poly(MOUTH_IN, (0, 120, 200))
    for a, b in EAR_L + EAR_R:
        cv2.line(canvas, tuple(P[a]), tuple(P[b]), (255, 180, 0), 1, cv2.LINE_AA)
    for a, b in MAR_L + [(12, 13)]:
        cv2.line(canvas, tuple(P[a]), tuple(P[b]), (255, 120, 200), 1, cv2.LINE_AA)
    for i in NOSE_CHIN:
        cv2.circle(canvas, tuple(P[i]), 2, (0, 255, 255), -1, cv2.LINE_AA)
    for i in range(22):
        cv2.circle(canvas, tuple(P[i]), 2, (60, 60, 255), -1, cv2.LINE_AA)


def sparkline(canvas, x0, y0, w, h, values, vmin, vmax, color, title, thr=None, thr_color=(0, 0, 255)):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    cv2.putText(canvas, title, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

    def ypix(v):
        v = max(vmin, min(vmax, v))
        return int(y0 + h - (v - vmin) / (vmax - vmin) * h)

    if thr is not None:
        ty = ypix(thr)
        cv2.line(canvas, (x0, ty), (x0 + w, ty), thr_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{thr:.2f}", (x0 + w - 34, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, thr_color, 1, cv2.LINE_AA)
    if len(values) >= 2:
        pts = [(int(x0 + i / (HIST - 1) * w), ypix(v)) for i, v in enumerate(values)]
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{values[-1]:.2f}", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def render(d, last_rx, port, model_img, preview, pkt_rate, paused):
    canvas = np.full((H, W, 3), 22, dtype=np.uint8)

    ear = float(d.get("ear", 0.0)); mar = float(d.get("mar", 0.0))
    yaw = float(d.get("yaw", 0.0)); pitch = float(d.get("pitch", 0.0)); roll = float(d.get("roll", 0.0))
    fps = float(d.get("fps", 0.0)); status = str(d.get("status", "N/A")); alarm = bool(d.get("alarm", False))
    dec = float(d.get("dec", 0.0)); ai = float(d.get("ai", 0.0)); total = float(d.get("total", 0.0))
    ear_thr = float(d.get("ear_thr", 0.21)); mar_thr = float(d.get("mar_thr", 0.45))
    mouth_s = float(d.get("mouth_s", 0.0)); yawns = int(d.get("yawns", 0))
    roi_on = int(d.get("roi", 0)); rx = int(d.get("rx", 0)); ry = int(d.get("ry", 0)); rs = int(d.get("rs", 0))
    crop_label = {0: "CROP GIUA", 1: "CROP CO DINH", 2: "BAM MAT (ROI)"}.get(roi_on, "?")
    crop_col = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}.get(roi_on, (0, 0, 255))
    lm = np.array(d.get("landmarks", []), dtype=np.float32)
    age = (time.time() - last_rx) if last_rx else 0.0

    if not paused and last_rx:
        hist_ear.append(ear); hist_mar.append(mar); hist_yaw.append(yaw); hist_pitch.append(pitch)

    # ===== Panel 1: CAMERA FULL FRAME (OV5640) =====
    pw, ph = 420, 315
    px, py = 20, 52
    cv2.putText(canvas, "CAMERA OV5640 - toan khung (what ESP32 camera sees)", (px, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (px - 1, py - 1), (px + pw, py + ph), (90, 90, 90), 1)
    if preview is not None and preview.size > 0:
        big = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_LINEAR)
        canvas[py:py + ph, px:px + pw] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        # Vẽ khung ROI (toạ độ trong khung gốc 320x240) lên preview
        if rs > 0:
            sx = int(rx / 320.0 * pw); sy = int(ry / 240.0 * ph)
            sw = int(rs / 320.0 * pw); sh = int(rs / 240.0 * ph)
            col = crop_col
            cv2.rectangle(canvas, (px + sx, py + sy), (px + sx + sw, py + sy + sh), col, 2, cv2.LINE_AA)
            cv2.putText(canvas, crop_label, (px + sx + 2, py + sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "chua co preview...", (px + 90, py + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

    # ===== Panel 2: MODEL INPUT 96x96 + mesh =====
    mw = 200
    mx, my = 460, 52
    cv2.putText(canvas, "Anh MODEL nhin thay (96x96) + 22 moc", (mx, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (mx - 1, my - 1), (mx + mw, my + mw), (90, 90, 90), 1)
    sub = canvas[my:my + mw, mx:mx + mw]
    if model_img is not None and model_img.size > 0:
        sub[:] = cv2.cvtColor(cv2.resize(model_img, (mw, mw), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    if lm is not None and lm.size == 44:
        draw_mesh(sub, lm.reshape(22, 2))

    # ===== Panel 3: metrics =====
    x = mx + mw + 30
    acol = (0, 0, 255) if alarm else (0, 200, 0)
    cv2.putText(canvas, status, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2, cv2.LINE_AA)

    def line(txt, y, col=(235, 235, 235), sc=0.58):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

    line(f"FPS ESP32   : {fps:5.1f}   ({1000.0/fps if fps > 0.01 else 0:5.0f} ms/frame)", 84)
    line(f"Dec {dec:5.1f} | AI {ai:5.1f} | Total {total:5.1f} ms", 112, (150, 200, 255))
    line(f"EAR : {ear:5.3f}   (nguong < {ear_thr:.3f})", 150, (0, 0, 255) if ear < ear_thr else (235, 235, 235))
    line(f"MAR : {mar:5.3f}   (nguong > {mar_thr:.3f})", 178, (0, 0, 255) if mar > mar_thr else (235, 235, 235))
    line(f"Ha mieng {mouth_s:4.1f}s (>=1.5s) | Tong ngap: {yawns}", 206, (200, 200, 120))
    line(f"Yaw {yaw:+6.1f} | Pitch {pitch:+6.1f} | Roll {roll:+6.1f}", 234)
    line(f"ROI: {crop_label}  (x{rx},y{ry},s{rs})", 266, crop_col)
    line(f"UDP:{port} | goi/s {pkt_rate:4.1f} | cap nhat {age:4.1f}s", 300, (120, 120, 120), 0.5)
    if age > 1.5:
        cv2.putText(canvas, "MAT KET NOI UDP (ESP32 chua gui / sai IP)", (x, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # ===== Row: sparklines =====
    sy = 470; sw, sh = 250, 60
    sparkline(canvas, 24, sy, sw, sh, list(hist_ear), 0.0, 0.5, (0, 220, 0), "EAR", thr=ear_thr)
    sparkline(canvas, 24 + sw + 24, sy, sw, sh, list(hist_mar), 0.0, 1.2, (0, 180, 255), "MAR", thr=mar_thr)
    sparkline(canvas, 24 + 2 * (sw + 24), sy, sw, sh, list(hist_yaw), -60, 60, (255, 180, 0), "Yaw (do)")
    sparkline(canvas, 24 + 3 * (sw + 24), sy, sw, sh, list(hist_pitch), -60, 60, (200, 120, 255), "Pitch (do)")

    cv2.putText(canvas, "Q/[ESC] thoat | [Space] dung do thi", (24, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    cv2.imshow("ESP32 Edge-AI Monitor v2 (camera + model + metrics)", canvas)


def main():
    ap = argparse.ArgumentParser(description="Hiển thị realtime ESP32 (camera full + model 96x96)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8889)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"❌ Không bind được UDP {args.bind}:{args.port} -> {e}")
        print("   👉 Cổng 8889 đang bị tiến trình khác chiếm (monitor/capture/replay cũ).")
        print("      Chỉ được chạy MỘT trong: esp_display_monitor / esp_telemetry_terminal /")
        print("      esp_replay_compare / esp32_log_capture. Đóng cửa sổ cũ rồi chạy lại.")
        print("      Nếu cần: netstat -ano | findstr :8889   ->   taskkill /PID <PID> /F")
        sys.exit(1)
    sock.settimeout(0.05)
    print(f"📺 [Realtime v2] Nghe UDP {args.bind}:{args.port} ... (Q thoát)")

    last, last_rx, model_img, preview = None, 0.0, None, None
    pkt_times = deque(maxlen=200)
    paused = False

    while True:
        try:
            while True:
                data, _ = sock.recvfrom(65536)
                if len(data) >= 8 and data[:4] == b"\xAA\x56\xAA\x56":
                    w = data[4] | (data[5] << 8); h = data[6] | (data[7] << 8)
                    if 0 < w <= 256 and 0 < h <= 256 and len(data) >= 8 + w * h:
                        model_img = np.frombuffer(data[8:8 + w * h], dtype=np.uint8).reshape(h, w).copy()
                elif len(data) >= 8 and data[:4] == b"\xAA\x56\xA0\x01":
                    w = data[4] | (data[5] << 8); h = data[6] | (data[7] << 8)
                    if 0 < w <= 256 and 0 < h <= 256 and len(data) >= 8 + w * h:
                        preview = np.frombuffer(data[8:8 + w * h], dtype=np.uint8).reshape(h, w).copy()
                else:
                    try:
                        last = json.loads(data.decode("utf-8", errors="ignore"))
                        last_rx = time.time(); pkt_times.append(last_rx)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
        except socket.timeout:
            pass

        now = time.time()
        while pkt_times and now - pkt_times[0] > 1.0:
            pkt_times.popleft()

        render(last if last else {"status": "DANG CHO ESP32...", "landmarks": []},
               last_rx, args.port, model_img, preview, float(len(pkt_times)), paused)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if key == ord(' '):
            paused = not paused

    sock.close()
    cv2.destroyAllWindows()
    print("👋 Đã dừng monitor.")


if __name__ == "__main__":
    main()
