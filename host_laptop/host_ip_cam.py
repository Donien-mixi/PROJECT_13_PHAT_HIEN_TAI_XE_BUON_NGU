"""
Master Laptop Host IP Camera & Telemetry Dashboard.
Entrypoint for Phase 2: Runs Camera Streamer, TCP Server for ESP32,
HTTP MJPEG Server, and renders the interactive ADAS Telemetry HUD.
"""

import sys
import cv2
import time
import argparse

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from camera_streamer import CameraStreamer, TCPServerStreamer, HTTPServerStreamer
from dashboard_visualizer import DashboardVisualizer

def main():
    parser = argparse.ArgumentParser(description="Laptop IP Camera & ADAS Dashboard for ESP32-S3")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--tcp_port", type=int, default=8888, help="TCP port for ESP32 stream (default: 8888)")
    parser.add_argument("--http_port", type=int, default=8080, help="HTTP MJPEG port (default: 8080)")
    parser.add_argument("--udp_port", type=int, default=8889, help="UDP port for ESP32 telemetry (default: 8889)")
    parser.add_argument("--stream_size", type=int, default=240, help="Square crop size for streaming (default: 240)")
    parser.add_argument("--synthetic", action="store_true", help="Run with simulated synthetic driver feed (no webcam needed)")
    parser.add_argument("--no_gui", action="store_true", help="Run in headless mode without OpenCV GUI window")
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 KHỞI ĐỘNG TRẠM CAMERA IP & GIÁM SÁT TELEMETRY (PHASE 2)")
    print(f"👉 Chế độ Camera  : {'Mô Phỏng (Synthetic)' if args.synthetic else f'Webcam Index {args.cam}'}")
    print(f"👉 TCP Port ESP32 : {args.tcp_port} (Gửi khung hình 1:1 qua Wi-Fi)")
    print(f"👉 HTTP Web Port  : http://localhost:{args.http_port}/video_feed")
    print(f"👉 UDP Telemetry  : {args.udp_port} (Nhận kết quả AI từ ESP32)")
    print("=" * 65)

    # 1. Start Camera Streamer
    camera = CameraStreamer(
        camera_src=args.cam,
        stream_size=args.stream_size,
        jpeg_quality=75,
        use_dynamic_face=True,
        use_synthetic=args.synthetic
    )

    # 2. Start TCP Server for ESP32
    tcp_server = TCPServerStreamer(camera, port=args.tcp_port)
    tcp_server.start()

    # 3. Start HTTP MJPEG Server for Web Browser
    http_server = HTTPServerStreamer(camera, port=args.http_port)
    http_server.start()

    # 4. Start Telemetry Dashboard Visualizer
    dashboard = DashboardVisualizer(udp_port=args.udp_port)

    print("\n✅ Hệ thống đã sẵn sàng!")
    print("⌨️  Phím bấm điều khiển:")
    print(" - Phím 'd': Bật/Tắt chế độ tự động bám tâm mặt (Dynamic Face Tracking)")
    print(" - Phím 's': Chụp ảnh màn hình (Snapshot)")
    print(" - Phím 'q': Thoát chương trình\n")

    if args.no_gui:
        print("Chạy ở chế độ nền (Headless). Nhấn Ctrl+C để dừng...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        window_name = "ADAS Host Dashboard (ESP32-S3 Edge AI)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        while True:
            raw_frame, _, _, crop_meta = camera.get_latest_frame_data()
            if raw_frame is not None and crop_meta is not None:
                hud_frame = dashboard.draw_hud(raw_frame, crop_meta)
                cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(15) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                camera.use_dynamic_face = not camera.use_dynamic_face
                mode_str = "BẬT" if camera.use_dynamic_face else "TẮT (Cố định tâm)"
                print(f"[Mode] Tự động bám tâm khuôn mặt: {mode_str}")
            elif key == ord('s'):
                snap_path = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, hud_frame)
                print(f"📸 Đã lưu ảnh chụp màn hình tại: {snap_path}")

        cv2.destroyAllWindows()

    # Graceful Shutdown
    print("\nĐang tắt các tiến trình...")
    camera.stop()
    tcp_server.stop()
    http_server.stop()
    dashboard.stop()
    print("Đã đóng hệ thống an toàn.")

if __name__ == "__main__":
    main()
