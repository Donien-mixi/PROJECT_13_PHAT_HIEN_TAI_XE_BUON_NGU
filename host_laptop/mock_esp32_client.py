"""
Mock ESP32-S3 Client for Testing Phase 2.
Connects to Laptop TCP Server (Port 8888), receives JPEG frames,
and sends simulated UDP Telemetry (Port 8889) to test the HUD.
"""

import sys
import socket
import struct
import json
import time
import math

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

TCP_MAGIC_HEADER = b'\xaa\x55\xaa\x55'

def main():
    server_ip = "127.0.0.1"
    tcp_port = 8888
    udp_port = 8889

    print("=" * 60)
    print("🤖 MOCK ESP32-S3 CLIENT SIMULATOR")
    print(f"👉 Đang kết nối TCP Server tại {server_ip}:{tcp_port}...")
    print("=" * 60)

    # 1. Connect TCP Socket for Video Stream
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_sock.connect((server_ip, tcp_port))
        print("✅ Đã kết nối thành công tới Camera TCP Server!")
    except Exception as e:
        print(f"❌ Không thể kết nối tới {server_ip}:{tcp_port}: {e}")
        print("Vui lòng khởi động host_ip_cam.py trước!")
        return

    # 2. Setup UDP Socket for Telemetry
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frame_count = 0
    start_time = time.time()
    t = 0.0

    try:
        while True:
            # Read 8-byte header: 4B Magic + 4B Length
            header = b''
            while len(header) < 8:
                chunk = tcp_sock.recv(8 - len(header))
                if not chunk:
                    raise ConnectionResetError("Mất kết nối với Server.")
                header += chunk

            magic = header[:4]
            if magic != TCP_MAGIC_HEADER:
                print("⚠️ Sai Magic Header! Bỏ qua byte đồng bộ...")
                continue

            payload_len = struct.unpack('>I', header[4:8])[0]

            # Read JPEG Payload
            jpeg_data = b''
            while len(jpeg_data) < payload_len:
                chunk = tcp_sock.recv(min(4096, payload_len - len(jpeg_data)))
                if not chunk:
                    raise ConnectionResetError("Mất kết nối khi đang nhận JPEG.")
                jpeg_data += chunk

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            # Generate simulated telemetry
            t += 0.05
            sim_ear = 0.28 + 0.08 * math.sin(t * 1.5)
            sim_mar = 0.15 + 0.10 * max(0.0, math.sin(t * 0.8))
            sim_yaw = 15.0 * math.sin(t * 0.6)
            sim_pitch = 8.0 * math.cos(t * 0.5)

            # Simulated landmarks normalized [0.0, 1.0]
            # Center around 0.5, 0.5
            cx, cy = 0.5 + 0.05 * math.sin(t * 0.6), 0.5 + 0.03 * math.cos(t * 0.5)
            landmarks = []
            # 6 left eye
            for dx, dy in [(-0.12, -0.06), (-0.08, -0.08), (-0.04, -0.08), (0.0, -0.06), (-0.04, -0.04), (-0.08, -0.04)]:
                landmarks.extend([cx + dx, cy + dy])
            # 6 right eye
            for dx, dy in [(0.04, -0.06), (0.08, -0.08), (0.12, -0.08), (0.16, -0.06), (0.12, -0.04), (0.08, -0.04)]:
                landmarks.extend([cx + dx, cy + dy])
            # 6 mouth
            for dx, dy in [(-0.08, 0.15), (0.08, 0.15), (0.0, 0.12), (0.0, 0.18), (0.0, 0.14), (0.0, 0.16)]:
                landmarks.extend([cx + dx, cy + dy])
            # 4 nose & chin
            for dx, dy in [(0.02, -0.02), (0.02, 0.06), (-0.02, 0.08), (0.02, 0.26)]:
                landmarks.extend([cx + dx, cy + dy])

            status = "NORMAL"
            alarm = False
            if sim_ear < 0.21:
                status = "MICROSLEEP!"
                alarm = True
            elif sim_mar > 0.40:
                status = "YAWNING!"
            elif abs(sim_yaw) > 25.0:
                status = "DISTRACTED!"
                alarm = True

            telemetry_packet = {
                "ear": round(sim_ear, 2),
                "mar": round(sim_mar, 2),
                "yaw": round(sim_yaw, 1),
                "pitch": round(sim_pitch, 1),
                "roll": 0.0,
                "status": status,
                "alarm": alarm,
                "fps": round(fps, 1),
                "landmarks": landmarks
            }

            # Send to UDP port 8889
            packet_bytes = json.dumps(telemetry_packet).encode('utf-8')
            udp_sock.sendto(packet_bytes, (server_ip, udp_port))

            if frame_count % 30 == 0:
                print(f"[Mock ESP32] Nhận Frame #{frame_count} ({payload_len:,} bytes) | FPS: {fps:.1f} | EAR: {sim_ear:.2f} | Status: {status}")

            time.sleep(0.04) # ~25 FPS

    except KeyboardInterrupt:
        print("\nĐã dừng Mock Client.")
    finally:
        tcp_sock.close()
        udp_sock.close()

if __name__ == "__main__":
    main()
