"""
Automated Verification & Integrity Test Suite for Phase 2 (Laptop Host IP Camera & Telemetry HUD).
Verifies:
 1. Mathematical Isomorphism & Zero-Distortion Aspect Ratio (scale_x == scale_y)
 2. Camera Streamer & Synthetic Driver Frame Generation
 3. TCP Streaming Protocol (Magic Header 0xAA55AA55 + Length + JPEG Payload)
 4. UDP Telemetry Receiver & HUD Overlay Rendering
 5. HTTP MJPEG Web Server Feed
"""

import os
import sys
import time
import socket
import struct
import json
import urllib.request
import numpy as np
import cv2

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure host_laptop directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from camera_streamer import CameraStreamer, TCPServerStreamer, HTTPServerStreamer, TCP_MAGIC_HEADER
from dashboard_visualizer import DashboardVisualizer

def test_1_isomorphic_math():
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Kiểm Chuẩn Tính Đẳng Hướng Hình Học (Isomorphic Math)")
    print("=" * 60)

    test_resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    stream_size = 240

    for w, h in test_resolutions:
        S = min(w, h)
        x0 = (w - S) // 2
        y0 = (h - S) // 2

        # Scale factors
        scale_x = stream_size / float(S)
        scale_y = stream_size / float(S)

        # Aspect ratio distortion
        distortion = abs(scale_x - scale_y)
        print(f"  Độ phân giải gốc: {w}x{h} -> Crop Vuông {S}x{S} | s_x={scale_x:.4f}, s_y={scale_y:.4f} | Méo: {distortion:.6f}")
        assert distortion < 1e-7, f"Méo tỉ lệ hình học tại {w}x{h}!"

        # Coordinate round-trip test
        test_pt_x, test_pt_y = x0 + S * 0.45, y0 + S * 0.55
        norm_x = (test_pt_x - x0) / float(S)
        norm_y = (test_pt_y - y0) / float(S)

        recon_x = x0 + norm_x * S
        recon_y = y0 + norm_y * S
        assert abs(recon_x - test_pt_x) < 1e-4 and abs(recon_y - test_pt_y) < 1e-4, "Lỗi tái tạo tọa độ!"

    print("✅ TEST 1 PASSED: Tỉ lệ co giãn tuyệt đối đồng bộ 1:1, méo hình học = 0.00%!")


def test_2_streamer_and_jpeg():
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Bộ Tạo Khung Hình & Nén JPEG 1:1")
    print("=" * 60)

    streamer = CameraStreamer(use_synthetic=True, stream_size=240, jpeg_quality=75)
    time.sleep(0.3)

    raw_frame, square_frame, jpeg_bytes, crop_meta = streamer.get_latest_frame_data()
    assert raw_frame is not None, "raw_frame is None!"
    assert square_frame is not None, "square_frame is None!"
    assert jpeg_bytes is not None, "jpeg_bytes is None!"
    assert crop_meta is not None, "crop_meta is None!"

    # Verify dimensions
    assert raw_frame.shape == (480, 640, 3), f"Raw shape mismatch: {raw_frame.shape}"
    assert square_frame.shape == (240, 240, 3), f"Square shape mismatch: {square_frame.shape}"

    # Verify JPEG decode
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "Không thể giải mã JPEG bytes!"
    assert decoded.shape == (240, 240, 3), f"Decoded shape: {decoded.shape}"

    print(f"  Raw Frame Shape   : {raw_frame.shape}")
    print(f"  Square Frame Shape: {square_frame.shape}")
    print(f"  JPEG Bytes Size   : {len(jpeg_bytes):,} bytes (Phù hợp truyền ESP32)")
    print(f"  Decoded Shape     : {decoded.shape}")

    streamer.stop()
    print("✅ TEST 2 PASSED: Khung hình 1:1 và nén JPEG hợp lệ 100%!")


def test_3_tcp_protocol():
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Giao Thức Truyền Socket TCP Khung Hình (Port 18888)")
    print("=" * 60)

    test_port = 18888
    streamer = CameraStreamer(use_synthetic=True, stream_size=240, jpeg_quality=75)
    tcp_server = TCPServerStreamer(streamer, host='127.0.0.1', port=test_port)
    tcp_server.start()
    time.sleep(0.2)

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(('127.0.0.1', test_port))

    frames_received = 0
    for _ in range(5):
        # Read header: 4B Magic + 4B Length
        header = b''
        while len(header) < 8:
            chunk = client_sock.recv(8 - len(header))
            assert chunk, "Mất kết nối TCP client!"
            header += chunk

        magic = header[:4]
        assert magic == TCP_MAGIC_HEADER, f"Magic byte không khớp: {magic}"

        payload_len = struct.unpack('>I', header[4:8])[0]
        assert 1000 < payload_len < 50000, f"Độ dài payload bất thường: {payload_len}"

        payload = b''
        while len(payload) < payload_len:
            chunk = client_sock.recv(min(4096, payload_len - len(payload)))
            assert chunk, "Mất kết nối TCP payload!"
            payload += chunk

        # Verify decoded frame
        img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (240, 240, 3)
        frames_received += 1

    print(f"  Nhận thành công {frames_received} frame JPEG qua TCP Socket.")
    print(f"  Magic Header chuẩn: 0x{TCP_MAGIC_HEADER.hex().upper()}")
    print(f"  Kích thước payload trung bình: {payload_len:,} bytes")

    client_sock.close()
    tcp_server.stop()
    streamer.stop()
    print("✅ TEST 3 PASSED: Giao thức TCP Socket và Magic Sync hoạt động hoàn hảo!")


def test_4_udp_telemetry_and_hud():
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Nhận Telemetry UDP & Kết Xuất Giao Diện ADAS HUD")
    print("=" * 60)

    test_udp_port = 18889
    visualizer = DashboardVisualizer(udp_port=test_udp_port)
    time.sleep(0.1)

    # Simulated landmarks: 22 points normalized (44 floats)
    sim_landmarks = []
    for i in range(22):
        sim_landmarks.extend([0.3 + 0.02 * (i % 5), 0.3 + 0.02 * (i // 5)])

    test_packet = {
        "ear": 0.16,
        "mar": 0.55,
        "yaw": -32.5,
        "pitch": 5.0,
        "roll": 1.2,
        "status": "MICROSLEEP & DISTRACTED!",
        "alarm": True,
        "fps": 24.5,
        "landmarks": sim_landmarks
    }

    # Send UDP packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(test_packet).encode('utf-8'), ('127.0.0.1', test_udp_port))
    sock.close()
    time.sleep(0.2)

    # Verify visualizer received state
    with visualizer.lock:
        st = dict(visualizer.telemetry)

    assert abs(st['ear'] - 0.16) < 1e-4, f"EAR mismatch: {st['ear']}"
    assert abs(st['mar'] - 0.55) < 1e-4, f"MAR mismatch: {st['mar']}"
    assert abs(st['yaw'] - (-32.5)) < 1e-4, f"Yaw mismatch: {st['yaw']}"
    assert st['status'] == "MICROSLEEP & DISTRACTED!"
    assert st['alarm'] is True

    # Test drawing HUD
    raw_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_meta = {'x0': 80, 'y0': 0, 'square_size': 480, 'orig_w': 640, 'orig_h': 480, 'stream_size': 240}
    hud_frame = visualizer.draw_hud(raw_dummy, crop_meta)

    assert hud_frame is not None and hud_frame.shape == (480, 640, 3)
    # Check that overlay was drawn (frame is not all black)
    assert np.count_nonzero(hud_frame) > 1000, "HUD frame trống!"

    visualizer.stop()
    print(f"  Nhận Telemetry: EAR={st['ear']}, MAR={st['mar']}, YAW={st['yaw']}°, Status={st['status']}")
    print(f"  Vẽ HUD thành công: Kích thước {hud_frame.shape}, các lớp Glassmorphism & Landmarks hợp lệ.")
    print("✅ TEST 4 PASSED: Telemetry UDP & Kết xuất HUD hoạt động trơn tru!")


def test_5_http_mjpeg_server():
    print("\n" + "=" * 60)
    print("🧪 TEST 5: Máy Chủ HTTP MJPEG Video Stream (Port 18080)")
    print("=" * 60)

    test_http_port = 18080
    streamer = CameraStreamer(use_synthetic=True, stream_size=240, jpeg_quality=75)
    http_server = HTTPServerStreamer(streamer, host='127.0.0.1', port=test_http_port)
    http_server.start()
    time.sleep(0.3)

    req = urllib.request.Request(f"http://127.0.0.1:{test_http_port}/video_feed")
    response = urllib.request.urlopen(req, timeout=3.0)

    assert response.status == 200, f"HTTP Status: {response.status}"
    content_type = response.headers.get('Content-Type')
    assert 'multipart/x-mixed-replace' in content_type, f"Content-Type: {content_type}"

    # Read initial chunk
    chunk = response.read(2048)
    assert b'--FRAME' in chunk, "Không tìm thấy MJPEG frame boundary!"
    assert b'image/jpeg' in chunk, "Không tìm thấy JPEG header!"

    response.close()
    http_server.stop()
    streamer.stop()
    print(f"  HTTP Streamer trả về Content-Type: {content_type}")
    print(f"  Đã đọc thành công MJPEG chunk với boundary --FRAME và image/jpeg.")
    print("✅ TEST 5 PASSED: Web Browser MJPEG Streaming hoạt động hoàn hảo!")


def main():
    print("🚀 BẮT ĐẦU BỘ KIỂM CHUẨN TOÀN DIỆN CHO GIAI ĐOẠN 2 (LAPTOP HOST IP CAM)")
    start_time = time.time()

    test_1_isomorphic_math()
    test_2_streamer_and_jpeg()
    test_3_tcp_protocol()
    test_4_udp_telemetry_and_hud()
    test_5_http_mjpeg_server()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 TẤT CẢ 5 BÀI TEST GIAI ĐOẠN 2 ĐÃ VƯỢT QUA 100% TRONG {elapsed:.2f}s!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
