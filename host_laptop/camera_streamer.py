"""
Camera Streamer for Laptop IP Camera.
Performs Isomorphic Square Crop (1:1) and serves video frames
to ESP32-S3 via TCP Socket (Port 8888) and HTTP MJPEG (Port 8080).
"""

import os
import cv2
import socket
import threading
import time
import math
import struct
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Magic protocol header for ESP32 TCP frame sync
TCP_MAGIC_HEADER = b'\xaa\x55\xaa\x55'

# [v2.4.3 SYNC] Dùng chung bộ trích xuất MediaPipe với local_model_tester để anchor
# khớp CHÍNH XÁC với lúc TRAIN (teacher) và lúc test laptop. Fallback Haar nếu không có.
try:
    from local_model_tester import MediaPipeLandmarkExtractor
    _HAS_MP_EXTRACTOR = True
except Exception:
    MediaPipeLandmarkExtractor = None
    _HAS_MP_EXTRACTOR = False


def enhance_low_light(frame_bgr, target_luma=115.0):
    """
    Tự động cân bằng sáng và tăng cường tương phản trong điều kiện cabin thiếu sáng/ngược sáng/ban đêm.
    Áp dụng Adaptive Gamma Curve + CLAHE trên kênh Luminance (LAB).
    """
    if frame_bgr is None:
        return frame_bgr
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))

    if mean_luma >= target_luma:
        return frame_bgr

    gamma = math.log(target_luma / 255.0) / math.log(max(mean_luma, 4.0) / 255.0)
    gamma = float(np.clip(gamma, 0.35, 0.90))

    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
    brightened = cv2.LUT(frame_bgr, lut)

    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip = 2.5 if mean_luma > 60.0 else 3.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


class CameraStreamer:
    def __init__(self, camera_src=0, stream_size=240, jpeg_quality=75, use_dynamic_face=True, use_synthetic=False):
        self.camera_src = camera_src
        self.stream_size = stream_size
        self.jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.use_dynamic_face = use_dynamic_face
        self.use_synthetic = use_synthetic
        self.synthetic_t = 0.0

        # Initialize Video Capture (or fallback to synthetic mode)
        self.cap = None
        if not self.use_synthetic:
            try:
                # Ưu tiên DirectShow trên Windows để khởi động nhanh và triệt tiêu độ trễ
                self.cap = cv2.VideoCapture(self.camera_src, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_src)
                if not self.cap.isOpened():
                    print(f"⚠️ [Camera] Không thể mở Webcam index={camera_src}. Tự động chuyển sang chế độ Mô Phỏng (Synthetic Driver Mode)!")
                    self.use_synthetic = True
                    self.cap = None
                else:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Khử trễ đệm 4-5 frame nội bộ của OpenCV DirectShow
                    print(f"📷 [Camera] Đã mở thành công Webcam index={camera_src} (640x480, Buffer=1)")
            except Exception as e:
                print(f"⚠️ [Camera] Lỗi mở Webcam ({e}). Chuyển sang chế độ Mô Phỏng!")
                self.use_synthetic = True
                self.cap = None
        else:
            print("🎬 [Camera] Đang chạy ở chế độ Mô Phỏng Tài Xế (Synthetic Driver Mode).")

        # Smooth center tracking for dynamic crop with Anti-Jitter Deadband
        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_S = None
        self.ema_alpha = 0.15 # Smoothing factor
        self.deadband_pos = 6.0
        self.deadband_size = 8.0

        # Initialize Haar Cascade for optional dynamic face tracking
        # Initialize Haar Multi-Cascade for robust dynamic face tracking
        self.face_cascades = []
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_alt2.xml'),
            os.path.join(os.path.dirname(__file__), 'models', 'haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt2.xml',
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_alt.xml',
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        ]

        if hasattr(cv2, 'CascadeClassifier'):
            for p in candidate_paths:
                if p and os.path.exists(p):
                    try:
                        cc = cv2.CascadeClassifier(p)
                        if not cc.empty():
                            self.face_cascades.append(cc)
                    except Exception:
                        pass

        if self.face_cascades:
            print(f"🎯 [Face Tracking] Đã nạp {len(self.face_cascades)} bộ dò Haar Cascade.")
        else:
            print("ℹ️ [Face Tracking] Sử dụng Square Center-Crop đẳng hướng 1:1.")

        # [v2.4.3 SYNC] Anchor MediaPipe (giống lúc train và test laptop)
        self.mp_extractor = None
        if _HAS_MP_EXTRACTOR and MediaPipeLandmarkExtractor is not None and self.use_dynamic_face:
            try:
                self.mp_extractor = MediaPipeLandmarkExtractor()
            except Exception:
                self.mp_extractor = None
        if self.mp_extractor is not None and getattr(self.mp_extractor, 'is_loaded', False):
            print("💎 [Face Tracking] Dùng ANCHOR MEDIAPIPE (đồng bộ TRAIN + test laptop).")
        elif self.face_cascades:
            print("🎯 [Face Tracking] Dùng anchor Haar Cascade (MediaPipe không khả dụng).")

        # Thread-safe buffer
        self.lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_square_frame = None
        self.latest_jpeg_bytes = None
        self.latest_crop_meta = None
        self.is_running = True

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _generate_synthetic_frame(self, t):
        """Generates realistic animated driver cockpit frame for testing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Cockpit & windshield gradient
        frame[:300, :] = [32, 24, 18] # Windshield
        frame[300:, :] = [20, 20, 22] # Steering dashboard

        # Driver torso silhouette
        cv2.ellipse(frame, (320, 520), (150, 110), 0, 0, 360, (55, 55, 65), -1)

        # Head motion: subtle sway simulating vehicle vibration
        head_cx = int(320 + 35 * math.sin(t * 0.7))
        head_cy = int(220 + 12 * math.cos(t * 0.5))

        # Face & Head
        cv2.ellipse(frame, (head_cx, head_cy), (65, 85), 0, 0, 360, (180, 195, 220), -1)
        cv2.ellipse(frame, (head_cx, head_cy), (65, 85), 0, 0, 360, (130, 145, 170), 2)
        cv2.ellipse(frame, (head_cx, head_cy - 45), (68, 45), 0, 180, 360, (40, 30, 20), -1)

        # Eyes (periodically blinking)
        blink = abs(math.cos(t * 1.5))
        eye_h = max(1, int(7 * blink))
        cv2.ellipse(frame, (head_cx - 24, head_cy - 12), (12, eye_h), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(frame, (head_cx + 24, head_cy - 12), (12, eye_h), 0, 0, 360, (30, 30, 30), -1)

        # Eyebrows
        cv2.line(frame, (head_cx - 36, head_cy - 26), (head_cx - 12, head_cy - 24), (40, 30, 20), 2)
        cv2.line(frame, (head_cx + 12, head_cy - 24), (head_cx + 36, head_cy - 26), (40, 30, 20), 2)

        # Nose
        cv2.line(frame, (head_cx, head_cy - 6), (head_cx, head_cy + 14), (120, 135, 160), 2)
        cv2.line(frame, (head_cx, head_cy + 14), (head_cx + 6, head_cy + 14), (120, 135, 160), 2)

        # Mouth (periodically yawning/talking)
        yawn = max(0.0, math.sin(t * 0.6) - 0.3)
        mouth_h = max(2, int(20 * yawn))
        cv2.ellipse(frame, (head_cx, head_cy + 38), (18, mouth_h), 0, 0, 360, (40, 40, 130), -1)

        # Overlay tag
        cv2.putText(frame, "SIMULATED DRIVER FEED (SYNTHETIC MODE)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)

        return frame, (head_cx, head_cy)

    def _capture_loop(self):
        """Continuously captures frames from webcam or synthetic generator and updates crop buffers."""
        while self.is_running:
            target_face_center = None

            if self.use_synthetic or self.cap is None:
                self.synthetic_t += 0.05
                frame, synth_center = self._generate_synthetic_frame(self.synthetic_t)
                target_face_center = synth_center
                time.sleep(0.033) # ~30 FPS
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                # Mirror frame for natural driver view
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            default_S = min(w, h)
            target_cx = w / 2.0
            target_cy = h / 2.0
            target_S = float(default_S)

            if self.use_dynamic_face:
                if target_face_center is not None:
                    target_cx, target_cy = target_face_center
                    target_S = 260.0
                else:
                    got_face = False
                    # (A) [v2.4.3 SYNC] Anchor MediaPipe — khớp công thức canonical_face_crop lúc TRAIN.
                    if self.mp_extractor is not None and getattr(self.mp_extractor, 'is_loaded', False):
                        try:
                            _mp_pts, mp_info = self.mp_extractor.extract(frame)
                        except Exception:
                            mp_info = None
                        if mp_info is not None:
                            target_cx = float(mp_info[0])
                            target_cy = float(mp_info[1])
                            target_S = float(mp_info[2])   # = round(h_skull * 2.05)
                            got_face = True
                    # (B) Fallback Haar Cascade
                    if (not got_face) and self.face_cascades:
                        enh_frame = enhance_low_light(frame, target_luma=115.0)
                        small_gray = cv2.resize(cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY), (w // 2, h // 2))
                        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                        small_gray = clahe.apply(small_gray)
                        for cc in self.face_cascades:
                            faces = cc.detectMultiScale(
                                small_gray, scaleFactor=1.06, minNeighbors=2, minSize=(30, 30)
                            )
                            if len(faces) > 0:
                                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                                fx, fy, fw, fh = faces[0]
                                # [v2.2.0 FIX] Haar dò trên ảnh HALF-RES -> quy về FULL-RES.
                                d_eyes_est = 0.46 * float(fw) * 2.0
                                h_skull = max(d_eyes_est * 0.45 * 2.10, d_eyes_est * 1.40,
                                              d_eyes_est * 0.45 * 2.2 / 1.20)
                                target_cx = (fx + fw / 2.0) * 2.0
                                target_cy = (fy + fh * 0.40) * 2.0 + 0.32 * h_skull
                                target_S = float(np.clip(h_skull * 2.05, 160, default_S * 1.5))
                                break

            # Apply Anti-Jitter Deadband and Exponential Moving Average (EMA) smoothing
            if self.smooth_cx is None:
                self.smooth_cx = float(target_cx)
                self.smooth_cy = float(target_cy)
                self.smooth_S = float(target_S)
            else:
                dcx = abs(target_cx - self.smooth_cx)
                dcy = abs(target_cy - self.smooth_cy)
                dS = abs(target_S - self.smooth_S)
                if dcx > self.deadband_pos:
                    self.smooth_cx = self.ema_alpha * target_cx + (1.0 - self.ema_alpha) * self.smooth_cx
                if dcy > self.deadband_pos:
                    self.smooth_cy = self.ema_alpha * target_cy + (1.0 - self.ema_alpha) * self.smooth_cy
                if dS > self.deadband_size:
                    self.smooth_S = self.ema_alpha * target_S + (1.0 - self.ema_alpha) * self.smooth_S

            cx = self.smooth_cx
            # [v2.2.0 FIX] Bỏ dịch +5%S: mỏ neo lúc TRAIN là cy = eye_y + 0.32*h_skull,
            # KHÔNG có offset này. Thêm offset gây lệch khung crop so với phân phối train.
            cy = self.smooth_cy
            S = int(round(self.smooth_S))

            # [v2.4.0] Cắt vuông 1:1 + PAD ĐEN khi crop vượt khung (S có thể tới 1.5*min(w,h)).
            # KHÔNG clamp x0/y0 vào trong khung: với S>w, clamp sẽ làm crop méo tỉ lệ.
            x0 = int(round(cx - S / 2.0))
            y0 = int(round(cy - S / 2.0))
            x1 = x0 + S
            y1 = y0 + S
            pad_l = max(0, -x0); pad_t = max(0, -y0)
            pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
            if pad_l or pad_t or pad_r or pad_b:
                padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
                crop = padded[y0 + pad_t:y1 + pad_t, x0 + pad_l:x1 + pad_l]
            else:
                crop = frame[y0:y1, x0:x1]

            square_resized = cv2.resize(crop, (self.stream_size, self.stream_size), interpolation=cv2.INTER_LINEAR)
            # [v2.2.0 SYNC] KHÔNG enhance/CLAHE crop trước khi encode: input lúc train là
            # canonical crop thô + photometric aug (brightness/gamma/noise), KHÔNG CLAHE.
            # (enhance_low_light chỉ dùng cho bộ DÒ mặt, không dùng cho input model.)

            # Encode to JPEG
            ret_enc, jpeg_buf = cv2.imencode('.jpg', square_resized, self.jpeg_quality)
            jpeg_bytes = jpeg_buf.tobytes() if ret_enc else None

            crop_meta = {
                'x0': x0,
                'y0': y0,
                'square_size': S,
                'orig_w': w,
                'orig_h': h,
                'stream_size': self.stream_size
            }

            with self.lock:
                self.latest_raw_frame = frame
                self.latest_square_frame = square_resized
                self.latest_jpeg_bytes = jpeg_bytes
                self.latest_crop_meta = crop_meta

            # Yield slightly to maintain ~30 FPS
            time.sleep(0.01)

    def get_latest_frame_data(self):
        """Returns thread-safe copies of raw frame, square frame, jpeg bytes and metadata."""
        with self.lock:
            return self.latest_raw_frame, self.latest_square_frame, self.latest_jpeg_bytes, self.latest_crop_meta

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()


# ==============================================================================
# TCP Server for High-Speed ESP32 Video Streaming (Port 8888)
# Protocol: [4B Magic: 0xAA55AA55] + [4B Length uint32 BE] + [JPEG Bytes]
# ==============================================================================
class TCPServerStreamer:
    def __init__(self, camera_streamer, host='0.0.0.0', port=8888):
        self.camera_streamer = camera_streamer
        self.host = host
        self.port = port
        self.server_sock = None
        self.is_running = True
        self.active_clients = 0

    def start(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)
        print(f"📡 [TCP Server] Đang lắng nghe kết nối ESP32 tại cổng {self.port}...")

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()

    def _accept_loop(self):
        self.server_sock.settimeout(0.5)
        while self.is_running:
            try:
                client_sock, addr = self.server_sock.accept()
                print(f"⚡ [TCP Server] ESP32 đã kết nối từ IP: {addr[0]}:{addr[1]}")
                client_thread = threading.Thread(target=self._client_handler, args=(client_sock,), daemon=True)
                client_thread.start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _client_handler(self, client_sock):
        self.active_clients += 1
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while self.is_running:
                _, _, jpeg_bytes, _ = self.camera_streamer.get_latest_frame_data()
                if jpeg_bytes is not None:
                    payload_len = len(jpeg_bytes)
                    header = TCP_MAGIC_HEADER + struct.pack('>I', payload_len)
                    # Send magic + length + jpeg data
                    client_sock.sendall(header + jpeg_bytes)
                time.sleep(0.04) # ~25 FPS stream
        except (socket.error, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            client_sock.close()
            self.active_clients -= 1
            print("🔌 [TCP Server] ESP32 đã ngắt kết nối.")

    def stop(self):
        self.is_running = False
        if self.server_sock:
            self.server_sock.close()


# ==============================================================================
# Optional HTTP MJPEG Server for Web Browser Inspection (Port 8080)
# ==============================================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGHandler(BaseHTTPRequestHandler):
    camera_streamer = None

    def do_GET(self):
        if self.path in ['/', '/stream', '/video_feed']:
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    _, _, jpeg_bytes, _ = self.camera_streamer.get_latest_frame_data()
                    if jpeg_bytes:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                pass
        else:
            self.send_error(404)


class HTTPServerStreamer:
    def __init__(self, camera_streamer, host='0.0.0.0', port=8080):
        self.camera_streamer = camera_streamer
        self.host = host
        self.port = port
        self.server = None

    def start(self):
        MJPEGHandler.camera_streamer = self.camera_streamer
        self.server = ThreadedHTTPServer((self.host, self.port), MJPEGHandler)
        print(f"🌐 [HTTP MJPEG] Luồng xem thử trên trình duyệt tại: http://localhost:{self.port}/video_feed")
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
