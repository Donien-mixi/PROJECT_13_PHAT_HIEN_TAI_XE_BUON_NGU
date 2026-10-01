# 💻 Host Laptop: Bộ Công Cụ Kiểm Thử Ngoại Tuyến & Debug

Thư mục `host_laptop/` chứa các script Python phía máy tính cho đề tài *"Hệ thống phát hiện tài xế ngủ gật & mất tập trung chạy 100% Edge AI trên ESP32-S3"*.

> [!TIP]
> **TÍNH NĂNG MỚI (v3.0.0 - STANDALONE WEB STREAM):**
> ESP32-S3 hiện tại tự chạy **Web Server nhúng trên Cổng 80** (`http://<IP_ESP32>/`).
> Bạn **KHÔNG CẦN CHẠY SCRIPT NÀO TRÊN LAPTOP** để xem camera hay theo dõi trạng thái tài xế. Bất kỳ điện thoại, máy tính bảng hay laptop nào cùng mạng Wi-Fi chỉ cần mở trình duyệt là xem được ngay!
> 
> Các công cụ trong `host_laptop/` hiện đóng vai trò **kiểm thử ngoại tuyến (offline testing)** và **chẩn đoán sai số mô hình**.

---

## 🏗️ Cấu Trúc Các Công Cụ

```text
host_laptop/
├── local_model_tester.py           # [CHÍNH] Kiểm thử model .tflite bằng webcam laptop trước khi nạp mạch
├── esp_replay_compare.py           # Đối chiếu độ chính xác: chạy model trên laptop vs output của ESP32
├── esp_display_monitor.py          # [Tuỳ chọn] Màn hình hiển thị UDP :8889 đồ hoạ trên laptop (phụ trợ)
├── esp_telemetry_terminal.py       # [Tuỳ chọn] Dashboard dạng text trên Terminal qua UDP :8889
├── models/
│   └── tinydriver_model.tflite     # File model TFLite byte-identical với firmware ESP32
└── requirements.txt                # Thư viện: opencv-python, numpy, ai-edge-litert
```

---

## 🚀 Các Tác Vụ Thường Dùng

### 1. Kích hoạt môi trường Conda
```powershell
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
conda activate projet_13
```

### 2. Kiểm thử mô hình với Webcam Laptop (Trước khi nạp mạch)
Chạy AI trực tiếp trên webcam máy tính để kiểm tra độ nhạy của mắt, miệng và góc đầu:
```powershell
python host_laptop/local_model_tester.py --cam 0
```
**Phím tắt hữu ích:**
- **`r`**: Cân chỉnh lại baseline khuôn mặt tài xế (Recalibrate).
- **`m`**: Chuyển đổi giữa mô hình `TINYDRIVER` (nhúng) và `MEDIAPIPE` (ground-truth) để so sánh.
- **`d`**: Thay đổi mức log hiển thị trên màn hình.
- **`q` / `ESC`**: Thoát.

### 3. Đối chiếu sai số mô hình (Replay & Compare)
So sánh kết quả suy luận giữa máy tính và ESP32 trên cùng một khung hình để kiểm chứng lượng tử hóa INT8:
```powershell
python host_laptop/esp_replay_compare.py
```
- Nếu `ΔLM mean < 1px`: Mô hình và tiền/hậu xử lý trên ESP32 chính xác 100%.

### 4. (Tuỳ chọn) Màn hình phụ đồ hoạ trên Laptop qua UDP :8889
Nếu bạn muốn theo dõi đồ thị cuộn EAR/MAR/Yaw/Pitch trên laptop thay vì trình duyệt web:
```powershell
python host_laptop/esp_display_monitor.py
```
*(Thoát bằng phím `Q` hoặc `ESC`)*.
