"""
Dashboard Visualizer for Laptop Host.
Receives UDP Telemetry from ESP32-S3 (Port 8889) and renders
a Cyberpunk ADAS HUD with 22 landmarks and 3D Head Pose vectors.
"""

import cv2
import json
import math
import socket
import threading
import time
import numpy as np

# Audio support for Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class DashboardVisualizer:
    def __init__(self, udp_port=8889):
        self.udp_port = udp_port
        self.udp_sock = None
        self.is_running = True

        # Telemetry State from ESP32
        self.telemetry = {
            "ear": 0.28,
            "mar": 0.15,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "status": "CONNECTING...",
            "alarm": False,
            "fps": 0.0,
            "landmarks": None
        }
        self.last_packet_time = 0
        self.lock = threading.Lock()

        # Audio debounce
        self.last_alarm_time = 0

        # Start UDP listener thread
        self.udp_thread = threading.Thread(target=self._udp_listener, daemon=True)
        self.udp_thread.start()

    def _udp_listener(self):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind(('0.0.0.0', self.udp_port))
        print(f"📥 [Telemetry] Đang lắng nghe gói tin từ ESP32 tại UDP cổng {self.udp_port}...")

        self.udp_sock.settimeout(0.5)
        while self.is_running:
            try:
                data, _ = self.udp_sock.recvfrom(4096)
                packet_str = data.decode('utf-8', errors='ignore')
                parsed = json.loads(packet_str)
                with self.lock:
                    self.telemetry.update(parsed)
                    self.last_packet_time = time.time()

                # Trigger sound if alarm
                if parsed.get('alarm', False):
                    now = time.time()
                    if now - self.last_alarm_time > 0.4:
                        self.last_alarm_time = now
                        if HAS_WINSOUND:
                            winsound.Beep(1200, 250)

            except socket.timeout:
                continue
            except Exception:
                pass

    def draw_hud(self, raw_frame, crop_meta):
        """Renders the comprehensive ADAS HUD onto the raw camera frame."""
        if raw_frame is None or crop_meta is None:
            return raw_frame

        display = raw_frame.copy()
        h, w = display.shape[:2]

        with self.lock:
            state = dict(self.telemetry)
            connected = (time.time() - self.last_packet_time) < 2.5

        if not connected:
            state['status'] = "WAITING FOR ESP32..."
            state['alarm'] = False

        x0 = crop_meta['x0']
        y0 = crop_meta['y0']
        S = crop_meta['square_size']

        # 1. Draw Isomorphic Square Crop Region
        is_alarm = state.get('alarm', False)
        box_color = (0, 0, 255) if is_alarm else (255, 240, 0) # Red if alarm, Cyan if normal
        cv2.rectangle(display, (x0, y0), (x0 + S, y0 + S), box_color, 2)
        cv2.putText(display, "ESP32 CROP 1:1", (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # 2. Draw 22 Landmarks if received
        landmarks_flat = state.get('landmarks', None)
        if landmarks_flat and len(landmarks_flat) == 44:
            pts_2d = []
            for i in range(0, 44, 2):
                nx = landmarks_flat[i]
                ny = landmarks_flat[i + 1]
                px = int(round(x0 + nx * S))
                py = int(round(y0 + ny * S))
                pts_2d.append((px, py))

            # Draw Eyes (Green)
            for pt in pts_2d[0:12]:
                cv2.circle(display, pt, 2, (0, 255, 0), -1)

            # Draw Mouth (Magenta)
            for pt in pts_2d[12:18]:
                cv2.circle(display, pt, 2, (255, 0, 255), -1)

            # Draw Nose & Chin (Cyan)
            for pt in pts_2d[18:22]:
                cv2.circle(display, pt, 3, (0, 255, 255), -1)

            # Draw 3D Head Pose Axes projecting from nose tip (pt 19) bằng cv2.projectPoints chuẩn xác
            nose_px, nose_py = pts_2d[19]
            pitch = float(state.get('pitch', 0.0))
            yaw = float(state.get('yaw', 0.0))
            roll = float(state.get('roll', 0.0))
            axis_len = 50.0

            # Chuyển góc Euler sang ma trận xoay Rodrigues (PnP convention: pitch=rx, yaw=ry, roll=rz)
            rx = np.array([
                [1.0, 0.0, 0.0],
                [0.0, math.cos(math.radians(pitch)), -math.sin(math.radians(pitch))],
                [0.0, math.sin(math.radians(pitch)), math.cos(math.radians(pitch))]
            ])
            ry = np.array([
                [math.cos(math.radians(yaw)), 0.0, math.sin(math.radians(yaw))],
                [0.0, 1.0, 0.0],
                [-math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))]
            ])
            rz = np.array([
                [math.cos(math.radians(roll)), -math.sin(math.radians(roll)), 0.0],
                [math.sin(math.radians(roll)), math.cos(math.radians(roll)), 0.0],
                [0.0, 0.0, 1.0]
            ])
            R = rz @ ry @ rx
            rvec, _ = cv2.Rodrigues(R)

            h_vis, w_vis = display.shape[:2]
            focal_len = w_vis * 1.1
            cam_mat = np.array([
                [focal_len, 0, nose_px],
                [0, focal_len, nose_py],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_c = np.zeros((4, 1))
            tvec = np.array([[0.0], [0.0], [focal_len]], dtype=np.float64)

            axes_3d = np.array([
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len]
            ], dtype=np.float64)

            imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam_mat, dist_c)
            imgpts = imgpts.reshape(-1, 2).astype(int)

            cv2.line(display, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 2, cv2.LINE_AA)  # Red: Pitch (X)
            cv2.line(display, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 2, cv2.LINE_AA)  # Green: Yaw (Y)
            cv2.line(display, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 2, cv2.LINE_AA)  # Blue: Roll/Forward (Z)

        # 3. Draw Telemetry Overlay Panel (Glassmorphism Dark)
        overlay = display.copy()
        cv2.rectangle(overlay, (15, 15), (280, 210), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        cv2.rectangle(display, (15, 15), (280, 210), (80, 80, 80), 1)

        # Title & Status Badge
        cv2.putText(display, "ADAS TELEMETRY", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)
        
        status_text = state.get('status', 'NORMAL')
        status_color = (0, 255, 0)
        if "MICROSLEEP" in status_text or "DISTRACT" in status_text or is_alarm:
            status_color = (0, 0, 255)
        elif "FATIGUE" in status_text or "SLOW BLINK" in status_text or "YAWN" in status_text:
            status_color = (0, 165, 255)
        elif not connected:
            status_color = (120, 120, 120)

        cv2.putText(display, f"STATUS: {status_text}", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Telemetry Gauges (EAR, MAR, Yaw)
        ear_val = float(state.get('ear', 0.0))
        mar_val = float(state.get('mar', 0.0))
        yaw_val = float(state.get('yaw', 0.0))
        fps_val = float(state.get('fps', 0.0))

        # EAR Bar
        cv2.putText(display, f"EAR: {ear_val:.2f}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        ear_bar_w = int(min(120, max(0, (ear_val / 0.4) * 120)))
        cv2.rectangle(display, (110, 85), (230, 97), (60, 60, 60), 1)
        ear_color = (0, 0, 255) if ear_val < 0.22 else (0, 255, 255)
        cv2.rectangle(display, (110, 85), (110 + ear_bar_w, 97), ear_color, -1)

        # MAR Bar
        cv2.putText(display, f"MAR: {mar_val:.2f}", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        mar_bar_w = int(min(120, max(0, (mar_val / 1.0) * 120)))
        cv2.rectangle(display, (110, 115), (230, 127), (60, 60, 60), 1)
        mar_color = (0, 0, 255) if mar_val > 0.50 else (255, 0, 255)
        cv2.rectangle(display, (110, 115), (110 + mar_bar_w, 127), mar_color, -1)

        # Yaw Angle
        cv2.putText(display, f"YAW: {yaw_val:+.1f} deg", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        yaw_color = (0, 0, 255) if abs(yaw_val) > 30.0 else (0, 255, 0)
        cv2.circle(display, (220, 150), 6, yaw_color, -1)

        # FPS & Hardware Badge
        cv2.putText(display, f"ESP32 FPS: {fps_val:.1f}", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        cv2.putText(display, "ESP32-S3 N16R8 ACTIVE", (25, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        # Flashing Red Border on Alarm
        if is_alarm:
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        return display

    def stop(self):
        self.is_running = False
        if self.udp_sock:
            self.udp_sock.close()
