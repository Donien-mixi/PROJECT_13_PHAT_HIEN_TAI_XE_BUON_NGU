# ⚡ Firmware ESP32-S3 Edge AI: Phát Hiện Ngủ Gật & Mất Tập Trung

Thư mục `firmware_esp32/` chứa toàn bộ mã nguồn firmware C/C++ chạy trực tiếp trên vi điều khiển **ESP32-S3 N16R8** (16MB Flash, 8MB Octal PSRAM), triển khai **100% các thuật toán Edge AI, giải thuật PnP và máy trạng thái ADAS** của **Đồ án 13**.

> [!IMPORTANT]
> **[Kiến trúc hiện tại] Camera OV5640 ONBOARD:** ESP32-S3 **tự thu hình** từ camera OV5640 gắn trực tiếp (DVP 24-pin). Laptop **KHÔNG** còn gửi ảnh — chỉ nhận telemetry UDP để **hiển thị**. Cơ chế TCP nhận ảnh từ laptop (Giai đoạn 2) chỉ còn là mã dự phòng (`wifi_stream_client`).
>
> **[v2.9.0] FACE ROI CROP:** Thay vì crop giữa khung, ESP32 bám **vùng khuôn mặt** (từ 22 landmark frame trước, công thức canonical anchor đúng như lúc train) → đầu vào 96×96 là **crop sát mặt** → độ nhạy MAR (ngáp) khớp laptop. Bootstrap bằng crop giữa khung khi chưa có ROI; tự reset về bootstrap nếu mất mặt >8 frame.

---

## 🎯 Kiến Trúc Phân Bổ Đa Nhân FreeRTOS (Dual-Core LX7 @ 240MHz)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     ESP32-S3 N16R8 SOC (Xtensa LX7 @ 240MHz)                │
├────────────────────────────────────┬───────────────────────────────────────┤
│   CORE 0: Web Server & Network      │   CORE 1: Edge AI & ADAS Task         │
│   (prio 3-4, esp_http_server)       │   (prio 6 - 16KB Stack)               │
├────────────────────────────────────┼───────────────────────────────────────┤
│ 1. Wi-Fi Station (2.4GHz)           │ 1. camera_capture_acquire (JPEG OV5640)│
│ 2. HTTP Web Server (Cổng 80)       │ 2. image_decoder → gray 96x96 INT8     │
│    - GET /       : Dashboard HTML5 │ 3. ai_inference: TinyDriverNet(esp-nn) │
│    - GET /stream : MJPEG Video Cam │ 4. pnp_solver: POSIT → Yaw/Pitch/Roll  │
│    - GET /status : JSON Telemetry  │ 5. adas_controller: FSM EAR/MAR/Pose  │
│    - POST /api/recalibrate        │ 6. Điều khiển còi GPIO2 & LED GPIO48  │
│ 3. UDP telemetry debug :8889       │ 7. Gửi dữ liệu sang Shared Buffer     │
└────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Mã Nguồn

```text
firmware_esp32/
├── CMakeLists.txt                 # Cấu hình dự án gốc ESP-IDF 5.3
├── sdkconfig / sdkconfig.defaults # 240MHz, Octal PSRAM 80MHz, cache, tối ưu bộ nhớ
├── partitions.csv                 # App partition 4MB
├── README.md                      # Tài liệu hướng dẫn nạp và cấu hình firmware
└── main/
    ├── CMakeLists.txt             # Đăng ký các file nguồn và thư viện phụ thuộc
    ├── idf_component.yml          # Dependency esp_new_jpeg + esp32-camera + esp-tflite-micro
    ├── Kconfig.projbuild          # Menu Wi-Fi, GPIO còi, bật camera onboard
    ├── main.cpp                   # app_main + vòng lặp Edge AI/ADAS (Core 1)
    ├── web_server.h/.cpp          # [v3.0.0] Web Server nhúng cổng 80: Dashboard HUD + MJPEG Stream
    ├── tinydriver_model_data.h    # Mảng byte mô hình INT8 (~249KB) căn lề 16-byte cho SIMD
    ├── camera_capture.h/.cpp      # Thu hình OV5640 onboard (JPEG QVGA, DMA PSRAM)
    ├── image_decoder.h/.cpp       # Giải mã JPEG THẬT (esp_new_jpeg) → gray 96x96 INT8 (crop ROI)
    ├── roi_tracker.h/.cpp         # Bám FACE ROI (crop sát mặt theo canonical anchor lúc train)
    ├── ai_inference.h/.cpp        # TFLite Micro Arena (SRAM nội / PSRAM fallback), SIMD esp-nn
    ├── pnp_solver.h/.cpp          # POSIT / PnP thuần C++ → Yaw, Pitch, Roll (<0.1ms)
    ├── adas_controller.h/.cpp     # FSM ADAS: EAR, MAR, Microsleep, Fatigue, Distraction
    ├── esp_nn_glue.h/.cpp         # Glue ESP-NN SIMD kernels vào TFLite Micro
    ├── esp_nn/                    # Nguồn ESP-NN vendor cục bộ
    ├── telemetry_sender.h/.cpp    # UDP JSON :8889 debug + điều khiển còi/LED
    └── wifi_stream_client.h/.cpp  # Quản lý kết nối Wi-Fi Station
```

---

## 🔌 Sơ Đồ Đấu Nối Phần Cứng (Pinout)

| Linh kiện | Chân trên ESP32-S3 | Chức năng |
| :--- | :---: | :--- |
| **Active Buzzer (Còi chíp 3.3V/5V)** | **GPIO 2** | Bíp khi chớp mắt chậm/ngáp; hú liên tục khi ngủ gật hoặc quay đầu |
| **Status / Warning LED** | **GPIO 48** | Sáng khi kích hoạt báo động ADAS |
| **Camera OV5640 (DVP 24-pin)** | XCLK 15, SIOD 4, SIOC 5, VSYNC 6, HREF 7, PCLK 13, D0..D7 = 11,9,8,10,12,18,17,16 | Thu hình onboard (JPEG QVGA) |
| **Nguồn cấp DevKit** | Cổng USB-C | 5V $\ge 1A$ (camera ăn dòng cao — nên dùng cáp dữ liệu + cổng USB 3.0) |

> [!CAUTION]
> **KHÔNG dùng GPIO4 cho còi** — GPIO4 là **SIOD** (I2C dữ liệu) của camera OV5640. Còi bắt buộc ở **GPIO2**.

---

## 🌐 Giao Diện Web Stream Trực Tiếp (Cổng 80)

ESP32-S3 tự động khởi chạy HTTP Server nhúng ngay khi kết nối Wi-Fi thành công. Bạn có thể truy cập từ bất kỳ trình duyệt nào trên điện thoại hoặc máy tính:

👉 **`http://<IP_ESP32>/`** (Ví dụ: `http://192.168.2.32/`)

### Các Endpoint Cung Cấp:
| Endpoint | Giao thức | Chức năng |
| :--- | :---: | :--- |
| **`/`** | `GET` (HTML5) | Giao diện Dashboard Cockpit ADAS tối màu, tự render 22 điểm mốc và phát còi cảnh báo Web |
| **`/stream`** | `GET` (MJPEG) | Luồng video camera OV5640 thời gian thực (`multipart/x-mixed-replace;boundary=...`) |
| **`/status`** | `GET` (JSON) | API trả về các chỉ số EAR, MAR, Head Pose 3D, mảng 22 mốc toạ độ, FPS và độ trễ (ms) |
| **`/api/recalibrate`** | `POST` (JSON) | Kích hoạt lại 5 giây tự hiệu chuẩn baseline cho tài xế từ xa |

### Cấu Trúc Gói JSON `/status`:
```json
{
  "ear": 0.28, "mar": 0.15,
  "yaw": 2.5, "pitch": -1.2, "roll": 0.0,
  "status": "NORMAL", "alarm": false, "fps": 6.5,
  "dec": 31.0, "ai": 116.0, "pnp": 0.1, "total": 153.0,
  "ear_thr": 0.21, "mar_thr": 0.60, "mouth_s": 0.0, "yawns": 0, "blinks": 12,
  "roi": {"active": 1, "x": 50, "y": 60, "s": 180},
  "landmarks": [0.38, 0.42, 0.41, 0.40, ...]   // 22 điểm × (x,y) = 44 số thực [0, 1]
}
```

> [!NOTE]
> ESP32 vẫn duy trì gửi song song gói UDP port `8889` làm kênh debug phụ (dành cho script `esp_display_monitor.py` trên laptop nếu cần).

---

## ⚙️ Cấu Hình Wi-Fi (menuconfig)

```bash
cd firmware_esp32
idf.py menuconfig
```
Vào mục **`TinyDriver ADAS Configuration`**:
- **Wi-Fi SSID / Password:** Đặt tên và mật khẩu mạng Wi-Fi **2.4GHz**.
- **Buzzer GPIO Pin:** `2` (Mặc định — còi báo phần cứng).
- **Use onboard OV5640 camera:** `y`.
- Bấm **S** (lưu) và **ESC** (thoát).

---

## 🛠️ Biên Dịch & Nạp Firmware

### Cách A — Khuyên dùng (chạy từ THƯ MỤC GỐC dự án)
```powershell
# Chạy được trên cả CMD/Anaconda Prompt và PowerShell (hoặc dùng: cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU)
D:
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1                 # tìm cổng COM
powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3         # build + nạp + monitor
# Tùy chọn: -BuildOnly (chỉ build) | -NoBuild (nạp không build lại)
```

### Cách B — Thủ công trong ESP-IDF 5.x PowerShell
```bash
cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU\firmware_esp32
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```
Thoát monitor: **Ctrl + ]**.

> [!IMPORTANT]
> **[v2.0] Dependency `esp_new_jpeg`:** lần build đầu cần internet để Component Manager tải
> (`main/idf_component.yml`). Nếu lỗi `esp_jpeg_dec.h not found`:
> `idf.py add-dependency "espressif/esp_new_jpeg^1.0.2"`. Chỉ hỗ trợ **baseline JPEG**.

---

## 🧠 Nạp Mô Hình AI Tự Động Từ Google Colab

Sau khi train xong trên Colab (xem [huong_dan_chay_project.md](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/huong_dan_chay_project.md)), chạy 1 lệnh tại thư mục gốc:
```powershell
python tools/project_manager.py deploy-model tinydriver_esp32_package.zip
```
Tự động cập nhật `main/tinydriver_model_data.h` + `host_laptop/models/tinydriver_model.tflite` → nạp lại firmware.

---

## 📊 Chỉ Số Kỹ Thuật Thực Tế Khi Vận Hành

Log in trạng thái **mỗi 5 frame** trên Serial Monitor:
```
[Frame #  15] FPS: x.x | Dec: xx ms | AI: xx ms | PnP: 0.1ms | Total: xxx ms | EAR: 0.xx | MAR: 0.xx | Yaw: +x.x° | Status: NORMAL
```

| Khâu | Thời gian | Ghi chú |
| :--- | :--- | :--- |
| **Dec** (giải mã + downsample) | **~30–33 ms** | JPEG ~24ms + Gray+Downsample ~8ms |
| **AI** (TinyDriverNet) | **~40–80 ms** nếu arena ở **SRAM nội** (tốt) • **~115 ms** nếu ở **PSRAM** | Xem mục tối ưu bên dưới |
| **PnP** (POSIT) | **~0.1 ms** | Không đáng kể |
| **Total / FPS** | $\approx$ Dec + AI | SRAM nội: ~80ms (~12 FPS) • PSRAM: ~153ms (~6.5 FPS) |

- **Mô hình:** $\approx 200$ K params, `tinydriver_model.tflite` $\approx 249$ KB (byte-identical với header C).
- **Tensor Arena thích ứng:** thử lần lượt các kích thước **SRAM nội** (384→128 KB, ưu tiên khối lớn nhất) — nếu đủ thì chạy tốc độ cao; nếu không đủ **khối liền mạch** thì fallback **PSRAM** (chậm hơn). Log kèm `khối liền mạch lớn nhất` để chẩn đoán.

---

## ⚡ Tối Ưu Bộ Nhớ Cho Arena (SRAM Nội vs PSRAM)

Aim: đưa arena (~162 KB) vào **SRAM nội** để esp-nn chạy nhanh (AI ~50ms thay vì ~115ms). Các knob trong `sdkconfig` / `sdkconfig.defaults`:
- Giảm buffer Wi-Fi: `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM`, `DYNAMIC_RX_BUFFER_NUM`, `STATIC_TX_BUFFER_NUM`, `CACHE_TX_BUFFER_NUM`; `CONFIG_ESP_WIFI_RX_BA_WIN` phải $\le$ DYNAMIC_RX và $\le 2\times$STATIC_RX.
- Giải phóng IRAM: `CONFIG_ESP_WIFI_RX_IRAM_OPT=n`.
- Tắt tính năng Wi-Fi không dùng: `SAE_PK`, `SOFTAP_SAE_SUPPORT`, `WPA3_OWE_STA`.
- Hạ `CONFIG_SPIRAM_MALLOC_RESERVE_INTERNAL` (nhường SRAM nội cho heap).

Đọc dòng log `AI_INFERENCE: SRAM nội trống: xxx KB (khối liền mạch lớn nhất: xxx KB)` để biết còn thiếu bao nhiêu.

## ❓ Sự Cố Thường Gặp (xem thêm tại `huong_dan_chay_project.md`)

| Hiện tượng | Cách xử lý |
| :--- | :--- |
| `Could not open COMx` | Chạy `tools\find_esp_port.ps1`; đóng monitor cũ; cắm lại cáp USB dữ liệu |
| `esp_camera_init 0x105` | Kiểm tra cáp FPC 24-pin của camera OV5640; thử XCLK 10MHz |
| Log `Tensor Arena ... PSRAM` | Arena bị phân mảnh SRAM nội → xem mục tối ưu bộ nhớ ở trên |
| Không mở được Web Dashboard | Đảm bảo điện thoại/laptop cùng mạng Wi-Fi 2.4GHz với ESP32; kiểm tra đúng IP trên Serial Monitor (`http://<IP_ESP32>/`) |
| Laptop viewer (UDP) không nhận số liệu | Chỉ mở 1 viewer duy nhất; kiểm tra IP laptop trong menuconfig (nếu dùng UDP 8889); mở firewall UDP |
