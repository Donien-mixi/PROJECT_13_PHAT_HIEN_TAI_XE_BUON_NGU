# 📺 Laptop Host: Kiểm Thử Mô Hình & Màn Hình Realtime (Display-Only)

Thư mục `host_laptop/` chứa mã nguồn phía máy tính cho đề tài *"Hệ thống phát hiện tài xế ngủ gật & mất tập trung chạy 100% Edge AI trên ESP32-S3"*.

> [!IMPORTANT]
> **Ràng buộc cốt lõi:** Laptop **KHÔNG** chạy AI cho ESP32. Mọi tính toán (JPEG decode, TinyDriverNet, EAR/MAR, POSIT Head Pose, FSM ADAS) chạy 100% trên **ESP32-S3**.
> Laptop đảm nhận 2 việc:
> 1. **Kiểm thử mô hình** trên webcam bằng chính model `.tflite` trước khi nạp ESP32 (`local_model_tester.py`).
> 2. **Hiển thị realtime** những gì ESP32 đã xử lý, nhận qua UDP `:8889` (`esp_display_monitor.py`).

---

## 🏗️ Cấu Trúc Mã Nguồn

```text
host_laptop/
├── local_model_tester.py           # Kiểm thử model .tflite với webcam (HUD + còi) trước khi nạp ESP32
├── esp_display_monitor.py          # [CHÍNH] Màn hình realtime: ảnh 96x96 + 22 mốc + số liệu + đồ thị trượt
├── esp_replay_compare.py           # Đối chiếu ESP32 ↔ Laptop trên CÙNG ảnh 96x96 (kiểm chứng model/firmware)
├── esp_telemetry_terminal.py       # Dashboard chữ gọn trên terminal (thay thế cho viewer đồ hoạ)
├── models/
│   └── tinydriver_model.tflite     # Model byte-identical với firmware (do deploy-model đặt vào)
├── requirements.txt                # opencv-python, numpy, ai-edge-litert
│
│   --- (Legacy Giai đoạn 2: laptop còn gửi ảnh qua TCP; KHÔNG dùng khi ESP32 có camera onboard) ---
├── camera_streamer.py              # Square Center-Crop 1:1, TCP Server & HTTP MJPEG Server
├── host_ip_cam.py                  # Entrypoint trạm camera IP cũ
├── dashboard_visualizer.py         # HUD cũ nhận UDP (đã thay bằng esp_display_monitor.py)
├── mock_esp32_client.py            # Giả lập ESP32 (test khi chưa có mạch)
├── test_phase2_pipeline.py         # Bộ 5 test pipeline Giai đoạn 2
└── haarcascade_frontalface_default.xml
```

---

## 🌐 Giao Thức Nhận Từ ESP32 (UDP Port `8889`)

ESP32 gửi 2 loại gói trên cùng port `8889`, phân biệt bằng magic:

### 1) JSON trạng thái (mỗi frame)
```json
{
  "ear": 0.28, "mar": 0.15,
  "yaw": 2.5, "pitch": -1.2, "roll": 0.0,
  "status": "NORMAL", "alarm": false, "fps": 6.5,
  "dec": 31.0, "ai": 116.0, "total": 153.0,
  "ear_thr": 0.21, "mar_thr": 0.70, "mouth_s": 0.0, "yawns": 0,
  "landmarks": [0.38, 0.42, 0.41, 0.40, ...]
}
```
- `dec/ai/total` = thời gian (ms) từng khâu — dùng để chẩn đoán độ trễ.
- `ear_thr/mar_thr` = ngưỡng ADAS hiện hành (sau hiệu chuẩn).
- `mouth_s/yawns` = thời lượng há miệng + tổng số ngáp.
- `landmarks` = 22 điểm × (x,y) chuẩn hoá [0,1].

### 2) Gói ảnh xám 96×96 (đúng cái model "nhìn thấy")
```
Magic "AA 56 AA 56" (4B) | w (uint16 LE) | h (uint16 LE) | pixel (w*h, uint8)
```

> [!NOTE]
> Giao thức **TCP 8888 / magic `0xAA55AA55`** thuộc Giai đoạn 2 (laptop gửi ảnh) — **không còn dùng**.

---

## 🚀 Hướng Dẫn Vận Hành

### Bước 1: Cài thư viện & kích hoạt môi trường
```powershell
# Chạy được trên cả CMD/Anaconda Prompt và PowerShell (hoặc dùng: cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU)
D:
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
conda activate projet_13
pip install -r host_laptop/requirements.txt
```

### Bước 2: Kiểm thử mô hình trên Laptop (trước khi nạp ESP32)
```powershell
python host_laptop/local_model_tester.py --cam 0
```
Phím tắt trong cửa sổ:
- **`d`**: đổi chế độ log (`FULL` → `COMPACT` → `OFF`).
- **`p`**: snapshot + in toạ độ 22 điểm mốc.
- **`m`**: đổi engine `TINYDRIVER` (model nhúng) ↔ `MEDIAPIPE` (ground-truth để đối chiếu).
- **`f`**: bật/tắt bám mặt 1:1.
- **`r`**: hiệu chuẩn lại baseline.
- **`q`/`ESC`**: thoát.

### Bước 3: Hiển thị realtime kết quả ESP32 (màn hình chính)
Mở **cửa sổ MỚI** (PowerShell hoặc CMD), kích hoạt môi trường rồi chạy:
```powershell
D:
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
conda activate projet_13
python host_laptop/esp_display_monitor.py
# Tuỳ chọn: --port 8889 --bind 0.0.0.0
```
Màn hình hiển thị:
- **Ảnh 96×96** (đúng cái ESP32 thấy) + **22 điểm mốc**.
- **Dec / AI / Total (ms)** + **FPS** + ms/frame.
- **EAR / MAR kèm ngưỡng**, thời lượng há miệng, tổng số ngáp.
- **4 đồ thị trượt**: EAR / MAR / Yaw / Pitch theo thời gian.
- **Tốc độ gói (gói/s)** + **độ trễ cập nhật** (cảnh báo mất kết nối).
- Phím: **Q/ESC** thoát, **Space** tạm dừng đồ thị.

### Bước 4 (thay thế): Dashboard chữ trên Terminal
```powershell
python host_laptop/esp_telemetry_terminal.py
```

### Bước 5: Kiểm chứng model/firmware — Replay-Compare trên CÙNG đầu vào
Chạy lại chính file model trên **đúng ảnh 96×96 mà ESP32 gửi lên**, so sánh output suy luận:
```powershell
python host_laptop/esp_replay_compare.py
# Lưu ảnh 96x96 để bổ sung dữ liệu miền OV5640 (domain adaptation):
python host_laptop/esp_replay_compare.py --save-dir output\esp_frames
```
- `ΔLM mean < 1px` → **model + tiền/hậu xử lý firmware ĐÚNG** (khác biệt số liệu thực tế đến từ **crop/cảm biến**, không phải lỗi firmware).
- `ΔLM` lớn (> ~2px) → có lỗi ở firmware (decode/normalize/quantize/hậu xử lý).

> [!WARNING]
> Cả 3 công cụ (`esp_display_monitor.py`, `esp_telemetry_terminal.py`, `esp_replay_compare.py`) đều **bind UDP `8889`** → chỉ chạy **MỘT** cái tại một thời điểm. Nếu báo "cổng đang bị chiếm", đóng cửa sổ kia.

---

## 📌 Lấy IP Laptop Để Nạp Vào ESP32
```powershell
ipconfig
```
Tìm **Wireless LAN adapter Wi-Fi → IPv4 Address** (ví dụ `192.168.2.1`) và điền vào `menuconfig → TinyDriver ADAS Configuration → Laptop Host IP Address`. Đảm bảo laptop và ESP32 **cùng mạng Wi-Fi 2.4GHz**, và mở firewall cho UDP `8889`.

---

## 🧪 Kiểm Chuẩn Pipeline Giai Đoạn 2 (legacy)
```powershell
python host_laptop/test_phase2_pipeline.py        # 5 bài test: crop 1:1, JPEG, TCP, UDP/HUD, MJPEG
```
Có thể test end-to-end không cần mạch bằng 2 terminal: `host_ip_cam.py --synthetic` + `mock_esp32_client.py`.
