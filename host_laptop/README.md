# 📡 Trạm Camera IP & Giám Sát Telemetry ADAS (Laptop Host - Giai Đoạn 2)

Thư mục `host_laptop/` chứa toàn bộ mã nguồn của trạm máy tính đóng vai trò **Camera IP** và **Bảng điều khiển HUD Telemetry** cho đề tài *"Hệ thống phát hiện tài xế ngủ gật & mất tập trung chạy 100% Edge AI trên ESP32-S3"*.

---

## 🎯 Trách Nhiệm Của Laptop Host Trong Đồ Án

> [!IMPORTANT]
> **Ràng Buộc Kỹ Thuật Cốt Lõi:**
> Laptop **CHỈ** đóng vai trò là một camera IP truyền khung hình JPEG vuông qua mạng Wi-Fi và nhận kết quả JSON để vẽ HUD.
> **TUYỆT ĐỐI KHÔNG** chạy bất kỳ thuật toán AI nào (MediaPipe, PyTorch, TensorFlow...) trên Laptop. Mọi tính toán AI, giải mã ảnh, đo EAR/MAR và ước lượng Head Pose $Yaw, Pitch, Roll$ đều chạy 100% trên chip vi điều khiển **ESP32-S3 N16R8**.

---

## 🏗️ Cấu Trúc Mã Nguồn

```text
host_laptop/
├── camera_streamer.py                    # Thuật toán Square Center-Crop 1:1, TCP Server & HTTP MJPEG Server
├── dashboard_visualizer.py               # Lắng nghe UDP Telemetry từ ESP32, vẽ HUD Cyberpunk Glassmorphism
├── host_ip_cam.py                        # Điểm khởi chạy chính (Master Entrypoint)
├── local_model_tester.py                 # Kiểm thử mô hình TFLite/LiteRT với Webcam Laptop trước khi nạp ESP32
├── mock_esp32_client.py                  # Client giả lập ESP32-S3 để test hệ thống trước khi nạp mạch thật
├── test_phase2_pipeline.py               # Bộ kiểm chuẩn tự động 5 bài test (Unit & Integration)
├── haarcascade_frontalface_default.xml   # Bộ nhận diện khuôn mặt Haar Cascade bám tâm động
└── requirements.txt                      # opencv-python, numpy, ai-edge-litert
```

---

## 📐 Trọng Tâm Đồng Bộ: Thuật Toán Cắt Vuông Đẳng Hướng (1:1 Isomorphic Crop)

Để tránh hiện tượng hình ảnh khuôn mặt tài xế bị co bẹp làm lệch tỉ lệ mắt $EAR$, miệng $MAR$ và góc nghiêng $Yaw, Pitch$:
1. Khung hình gốc từ Webcam ($W \times H$, ví dụ $640 \times 480$ hoặc $1280 \times 720$) được tính cạnh vuông:
   $$S = \min(W, H)$$
2. Cắt vùng vuông trung tâm (hoặc bám tâm khuôn mặt với bộ lọc làm mượt EMA):
   $$x_0 = \text{clamp}\left(c_x - \frac{S}{2}, 0, W - S\right), \quad y_0 = \text{clamp}\left(c_y - \frac{S}{2}, 0, H - S\right)$$
3. Vùng ảnh cắt $S \times S$ được resize về kích thước chuẩn $240 \times 240$ (hoặc $192 \times 192$):
   $$s_x = s_y = \frac{240}{S}$$
   👉 **Hệ số co giãn đồng nhất $s_x = s_y$**, độ méo hình học bằng $0.00\%$.

---

## 🌐 Giao Thức Mạng & Truyền Nhận Dữ Liệu

### 1. Luồng Video TCP Frame Server (Port `8888`)
ESP32-S3 kết nối tới địa chỉ IP của Laptop qua TCP socket tại cổng `8888`:
```
┌────────────────────────┬──────────────────────┬─────────────────────────┐
│ Magic Bytes (4 bytes)  │ Payload Size (4 B)   │ JPEG Image Data         │
│ 0xAA 0x55 0xAA 0x55    │ uint32 Big-Endian    │ Buffer JPEG 1:1 (Bytes) │
└────────────────────────┴──────────────────────┴─────────────────────────┘
```
- **Tốc độ truyền:** $\approx 25$ FPS.
- **Dung lượng gói:** $\approx 3.5 - 6.5$ KB / frame (Băng thông chỉ $\approx 100 - 150$ KB/s, cực kỳ mượt mà qua Wi-Fi).

### 2. Luồng Xem Trực Tiếp Trên Trình Duyệt HTTP MJPEG (Port `8080`)
Truy cập qua bất kỳ trình duyệt nào trên cùng mạng LAN:
```text
http://localhost:8080/video_feed
hoặc http://<IP_LAPTOP>:8080/video_feed
```

### 3. Luồng Telemetry UDP Listener (Port `8889`)
ESP32-S3 sau khi suy luận AI sẽ gửi gói tin UDP JSON về lại Laptop tại cổng `8889`:
```json
{
  "ear": 0.28,
  "mar": 0.15,
  "yaw": 2.5,
  "pitch": -1.2,
  "roll": 0.0,
  "status": "NORMAL",
  "alarm": false,
  "fps": 21.3,
  "landmarks": [0.38, 0.42, 0.41, 0.40, ...]
}
```

---

## 🚀 Hướng Dẫn Vận Hành

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy trạm Camera IP & Dashboard HUD
- **Chạy với Webcam thật:**
  ```bash
  python host_ip_cam.py --cam 0
  ```
- **Chạy chế độ Mô Phỏng (Synthetic Driver Mode - Không cần cắm webcam):**
  ```bash
  python host_ip_cam.py --synthetic
  ```
- **Tùy chọn cổng mạng:**
  ```bash
  python host_ip_cam.py --tcp_port 8888 --udp_port 8889 --http_port 8080
  ```

### Bước 3: Phím tắt điều khiển trong giao diện
- **`d`**: Bật/Tắt tính năng tự động bám tâm khuôn mặt (Face-Guided Dynamic Centering).
- **`s`**: Chụp và lưu ảnh màn hình giao diện (`snapshot_<timestamp>.jpg`).
- **`q`**: Thoát chương trình một cách an toàn.

---

## 🧪 Kiểm Chuẩn & Giả Lập Hệ Thống

### 1. Chạy bộ kiểm chuẩn tự động 5 bài test:
```bash
python test_phase2_pipeline.py
```
Kết quả kiểm chuẩn:
- ✅ Test 1: Đẳng hướng hình học ($s_x = s_y$, sai số $0\%$).
- ✅ Test 2: Khung hình 1:1 và nén JPEG hợp lệ.
- ✅ Test 3: TCP Server & giao thức Magic Header `0xAA55AA55`.
- ✅ Test 4: UDP Telemetry & HUD Glassmorphism.
- ✅ Test 5: HTTP MJPEG web streaming.

### 2. Giả lập kết nối ESP32 bằng script test:
Nếu chưa có bo mạch ESP32-S3 cắm dây, bạn có thể kiểm thử toàn bộ hệ thống bằng 2 terminal:
- **Terminal 1:** Khởi động host:
  ```bash
  python host_ip_cam.py --synthetic
  ```
- **Terminal 2:** Khởi động client giả lập ESP32:
  ```bash
  python mock_esp32_client.py
  ```
Trình giả lập sẽ nhận frame ảnh thật qua TCP, tính toán giả lập và gửi gói tin UDP với các trạng thái bình thường, ngủ gật (Microsleep), ngáp (Yawn), quay đầu (Distracted), còi cảnh báo trên laptop sẽ kêu bíp tự động!

---

## 📌 Lấy Địa Chỉ IP Của Laptop Để Nạp Vào ESP32
Mở Command Prompt / PowerShell và gõ:
```powershell
ipconfig
```
Tìm mục **Wireless LAN adapter Wi-Fi** $\rightarrow$ dòng **IPv4 Address** (Ví dụ: `192.168.1.15`).
Điền địa chỉ IP này vào cấu hình Wi-Fi Client của firmware ESP32-S3 trong **Giai đoạn 3**.
