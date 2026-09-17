"""
ESP32 Log Capture (tự lưu log test ESP32, không cần copy tay)
==============================================================
Lắng nghe UDP :8889 mà ESP32 gửi telemetry JSON (song song với Serial Monitor,
KHÔNG chiếm cổng COM). Mỗi gói tin -> 1 dòng log in ra màn hình + ghi file:

    output/esp32_log_<YYYYmmdd_HHMMSS>.txt   (MỖI LẦN CHẠY = 1 FILE MỚI, không dùng log cũ)
    output/latest_esp32_log.txt              (con trỏ tới file mới nhất)

Chỉ dùng stdlib -> chạy được bằng mọi Python (không cần cv2/numpy).

Dùng:
    python tools/esp32_log_capture.py
    python tools/esp32_log_capture.py --port 8889
Dừng: Ctrl+C (file đã được flush từng dòng, không mất dữ liệu).
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Capture log telemetry ESP32 ra file trong output/")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--save-frames", action="store_true",
                    help="Luu anh 96x96 ESP32 gui len thanh file .pgm (de soi model nhin thay gi)")
    ap.add_argument("--frames-dir", default=None, help="Thu muc luu anh (mac dinh output/esp_frames)")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"esp32_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    fh = open(log_path, "w", encoding="utf-8")
    # [v2] JSONL tho tung frame (co 22 landmark + ROI + timestamp) -> de PHAN TICH LAG bang so
    jsonl_path = os.path.join(out_dir, f"esp32_track_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    fh_j = open(jsonl_path, "w", encoding="utf-8")
    try:
        with open(os.path.join(out_dir, "latest_esp32_log.txt"), "w", encoding="utf-8") as p:
            p.write(log_path)
        with open(os.path.join(out_dir, "latest_esp32_track.txt"), "w", encoding="utf-8") as p:
            p.write(jsonl_path)
    except OSError:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"❌ Không bind được UDP {args.bind}:{args.port} -> {e}\n"
              f"   Đóng viewer/replay đang chạy rồi chạy lại.")
        sys.exit(1)
    sock.settimeout(0.5)

    header = (f"# ESP32 LOG | bat dau {datetime.now().isoformat(timespec='seconds')} | "
              f"UDP :{args.port} | (ESP32 IP gui ve laptop nay)\n")
    fh.write(header)
    fh.flush()
    print(f"📥 [ESP32 Capture] Đang ghi log vao: {log_path}")
    print(f"📊 [ESP32 Capture] JSON tho (co 22 landmark) de phan tich lag: {jsonl_path}")
    print("💡 Giu Serial Monitor (flash_esp32.ps1) song song duoc - khong xung dot.")
    print("   Nhan Ctrl+C de dung.\n")

    n = 0
    n_img = 0
    n_cam = 0
    frames_dir = args.frames_dir or os.path.join(out_dir, "esp_frames")
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
        print(f"🖼️  Se luu anh 96x96 vao: {frames_dir}")
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            if len(data) >= 8 and data[:4] == b"\xAA\x56\xAA\x56":
                # goi anh 96x96 (model input) -> luu PGM neu duoc yeu cau
                if args.save_frames:
                    w = data[4] | (data[5] << 8)
                    h = data[6] | (data[7] << 8)
                    if 0 < w <= 256 and 0 < h <= 256 and len(data) >= 8 + w * h:
                        n_img += 1
                        fn = os.path.join(frames_dir,
                                          f"f{n_img:05d}_{datetime.now().strftime('%H%M%S_%f')[:-3]}.pgm")
                        with open(fn, "wb") as im:
                            im.write(b"P5\n%d %d\n255\n" % (w, h))
                            im.write(data[8:8 + w * h])
                continue
            if len(data) >= 8 and data[:4] == b"\xAA\x56\xA0\x01":
                # goi preview TOAN KHUNG camera -> luu PGM
                if args.save_frames:
                    w = data[4] | (data[5] << 8)
                    h = data[6] | (data[7] << 8)
                    if 0 < w <= 256 and 0 < h <= 256 and len(data) >= 8 + w * h:
                        n_cam += 1
                        fn = os.path.join(frames_dir,
                                          f"cam{n_cam:05d}_{datetime.now().strftime('%H%M%S_%f')[:-3]}.pgm")
                        with open(fn, "wb") as im:
                            im.write(b"P5\n%d %d\n255\n" % (w, h))
                            im.write(data[8:8 + w * h])
                continue
            try:
                d = json.loads(data.decode("utf-8", errors="ignore"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            n += 1
            line = (f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                    f"FPS: {float(d.get('fps', 0)):4.1f} | "
                    f"Dec: {float(d.get('dec', 0)):5.1f}ms | "
                    f"AI: {float(d.get('ai', 0)):5.1f}ms | "
                    f"Total: {float(d.get('total', 0)):5.1f}ms | "
                    f"EAR: {float(d.get('ear', 0)):.2f} | "
                    f"MAR: {float(d.get('mar', 0)):.2f} | "
                    f"Yaw: {float(d.get('yaw', 0)):+5.1f} | "
                    f"Pitch: {float(d.get('pitch', 0)):+5.1f} | "
                    f"EAR_thr: {float(d.get('ear_thr', 0)):.2f} | "
                    f"MAR_thr: {float(d.get('mar_thr', 0)):.2f} | "
                    f"mouth: {float(d.get('mouth_s', 0)):.1f}s | "
                    f"yawns: {int(d.get('yawns', 0))} | "
                    f"ROI: {int(d.get('roi', 0))} ({int(d.get('rx', 0))},{int(d.get('ry', 0))},{int(d.get('rs', 0))}) | "
                    f"Status: {d.get('status', 'N/A')}"
                    + (" [ALARM]" if d.get("alarm") else ""))
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()
            # [v2] ghi JSON tho + timestamp epoch (giu nguyen 22 landmark) de phan tich lag
            try:
                d["_t"] = time.time()
                fh_j.write(json.dumps(d) + "\n")
                fh_j.flush()
            except (TypeError, ValueError):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sys.__stdout__.write(f"\n👋 [ESP32 Capture] Da luu {n} dong log vao: {log_path}\n")
            if args.save_frames:
                sys.__stdout__.write(f"🖼️  Da luu {n_img} anh 96x96 + {n_cam} anh toan khung vao: {frames_dir}\n")
            sys.__stdout__.flush()
        except Exception:
            pass
        fh.close()
        fh_j.close()
        sock.close()


if __name__ == "__main__":
    main()
