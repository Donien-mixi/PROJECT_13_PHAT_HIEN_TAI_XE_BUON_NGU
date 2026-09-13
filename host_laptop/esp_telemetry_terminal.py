"""
Terminal Telemetry Receiver - Project 13
=========================================
Nhận gói UDP JSON từ ESP32-S3 (camera OV5640 onboard) và hiển thị dashboard
kết quả suy luận AI (EAR/MAR/Head Pose/ADAS) ngay trên cửa sổ terminal.

Cách dùng:
    conda activate projet_13
    python host_laptop/esp_telemetry_terminal.py

ESP32 gửi tới <LAPTOP_HOST_IP>:8889 (cấu hình trong idf.py menuconfig).
Đảm bảo IP laptop trong firmware trùng IP máy này (xem bằng lệnh `ipconfig`).
"""

import argparse
import json
import os
import socket
import sys
import time

# Bật mã ANSI trên Windows 10/11 console
if os.name == "nt":
    os.system("")

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
CLEAR = "\033[2J\033[H"


def color_for_status(status: str) -> str:
    s = status.upper()
    if "MICROSLEEP" in s or "DISTRACTION" in s or "FATIGUE" in s:
        return RED
    if "YAWN" in s or "SLOW" in s or "WARNING" in s:
        return YELLOW
    if "NORMAL" in s or "ATTENTIVE" in s:
        return GREEN
    return CYAN


def bar(value: float, vmax: float, width: int = 28) -> str:
    ratio = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    filled = int(round(ratio * width))
    return "#" * filled + "-" * (width - filled)


def render(d: dict, last_rx: float, port: int):
    now = time.time()
    age = now - last_rx if last_rx else 0.0
    ear = float(d.get("ear", 0.0))
    mar = float(d.get("mar", 0.0))
    yaw = float(d.get("yaw", 0.0))
    pitch = float(d.get("pitch", 0.0))
    roll = float(d.get("roll", 0.0))
    fps = float(d.get("fps", 0.0))
    status = str(d.get("status", "N/A"))
    alarm = bool(d.get("alarm", False))

    status_color = color_for_status(status)
    alarm_txt = f"{RED}{BOLD}🚨 CÒI HÚ (ALARM){RESET}" if alarm else f"{GREEN}---{RESET}"

    out = []
    out.append(f"{BOLD}{CYAN}====================================================================={RESET}")
    out.append(f"{BOLD}{CYAN}  PROJECT 13 - EDGE AI TREN ESP32-S3 N16R8 + OV5640 (UDP :{port}){RESET}")
    out.append(f"{BOLD}{CYAN}====================================================================={RESET}")
    out.append("")
    out.append(f"  {BOLD}FPS ESP32       :{RESET} {fps:5.1f}")
    out.append(f"  {BOLD}EAR             :{RESET} {ear:5.3f}  [{bar(ear, 0.40)}]")
    out.append(f"  {BOLD}MAR             :{RESET} {mar:5.3f}  [{bar(mar, 1.20)}]")
    out.append(f"  {BOLD}Yaw / Pitch/Roll:{RESET} {yaw:+6.1f}° / {pitch:+6.1f}° / {roll:+6.1f}°")
    out.append("")
    out.append(f"  {BOLD}Trang thai      :{RESET} {status_color}{BOLD}{status}{RESET}")
    out.append(f"  {BOLD}Canh bao        :{RESET} {alarm_txt}")
    out.append("")
    out.append(f"  {GREY}Cap nhat cach day {age:4.1f}s | Ctrl+C de thoat{RESET}")
    sys.stdout.write(CLEAR + "\n".join(out) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="ESP32 Edge AI telemetry terminal dashboard")
    ap.add_argument("--bind", default="0.0.0.0", help="Địa chỉ bind (mặc định 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8889, help="UDP port (mặc định 8889)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"❌ Không bind được UDP {args.bind}:{args.port} -> {e}")
        sys.exit(1)

    sock.settimeout(0.25)
    print(f"📡 [Telemetry] Đang lắng nghe UDP {args.bind}:{args.port} ... (Ctrl+C để thoát)")
    print("   Hãy đảm bảo IP laptop trong firmware (menuconfig > Laptop Host IP) trùng máy này.")

    last_packet = None
    last_rx = 0.0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                # Không có gói mới -> chỉ vẽ lại để cập nhật đồng hồ "cách đây"
                if last_packet is not None:
                    render(last_packet, last_rx, args.port)
                continue

            try:
                d = json.loads(data.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue

            last_packet = d
            last_rx = time.time()
            render(d, last_rx, args.port)
    except KeyboardInterrupt:
        print(f"\n{RESET}👋 Đã dừng terminal telemetry.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
