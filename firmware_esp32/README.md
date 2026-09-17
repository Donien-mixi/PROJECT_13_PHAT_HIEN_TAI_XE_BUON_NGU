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
│   CORE 0: Wi-Fi + Camera DMA        │   CORE 1: Edge AI & ADAS Task         │
│   (Wi-Fi task prio 23, esp32-cam)   │   (prio 6 - 16KB Stack)               │
├────────────────────────────────────┼───────────────────────────────────────┤
│ 1. Wi-Fi Station (2.4GHz)           │ 1. camera_capture_acquire (JPEG)      │
│ 2. esp32-camera DVP driver (OV5640) │ 2. image_decoder → gray 96x96 INT8     │
│    - Frame buffer trong PSRAM        │ 3. ai_inference: TinyDriverNet(esp-nn) │
│ 3. UDP socket gửi telemetry :8889    │ 4. pnp_solver: POSIT → Yaw/Pitch/Roll  │
│                                      │ 5. adas_controller: FSM EAR/MAR/Pose  │
│                                      │ 6. telemetry_sender + còi GPIO2       │
└────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Mã Nguồn

```text
firmware_esp32/
├── CMakeLists.txt                 # Cấu hình dự án gốc ESP-IDF
├── sdkconfig / sdkconfig.defaults # 240MHz, Octal PSRAM 80MHz, cache, tối ưu bộ nhớ
├── partitions.csv                 # App partition 4MB
├── README.md                      # Tài liệu hướng dẫn nạp và cấu hình firmware
└── main/
    ├── CMakeLists.txt             # Đăng ký các file nguồn và thư viện phụ thuộc
    ├── idf_component.yml          # Dependency espressif/esp_new_jpeg + esp32-camera + esp-tflite-micro
    ├── Kconfig.projbuild          # Menu Wi-Fi, IP Laptop, GPIO còi, bật camera onboard
    ├── main.cpp                   # app_main + vòng lặp Edge AI/ADAS (Core 1)
    ├── tinydriver_model_data.h    # Mảng byte mô hình INT8 (~249KB) căn lề 16-byte cho SIMD
    ├── camera_capture.h/.cpp      # [v2.7.0] Thu hình OV5640 onboard (JPEG QVGA, PSRAM)
    ├── image_decoder.h/.cpp       # Giải mã JPEG THẬT (esp_new_jpeg) → gray 96x96 INT8 (crop theo ROI)
    ├── roi_tracker.h/.cpp         # [v2.9.0] Bám FACE ROI (crop sát mặt theo canonical anchor lúc train)
    ├── ai_inference.h/.cpp        # TFLite Micro Arena (ưu tiên SRAM nội → fallback PSRAM), esp-nn
    ├── pnp_solver.h/.cpp          # POSIT / PnP thuần C++ → Yaw, Pitch, Roll (<0.3ms)
    ├── adas_controller.h/.cpp     # FSM ADAS: MAR=(h_outer+h_inner)/(2w) khớp laptop/train
    ├── esp_nn_glue.h/.cpp         # Glue ESP-NN SIMD kernels vào TFLite Micro
    ├── esp_nn/                    # Nguồn ESP-NN vendor cục bộ
    ├── telemetry_sender.h/.cpp    # UDP JSON :8889 + gói ảnh 96x96 + điều khiển còi/LED
    └── wifi_stream_client.h/.cpp  # [Legacy Giai đoạn 2] nhận JPEG qua TCP (không dùng khi có camera)
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

## 🌐 Giao Thức Telemetry UDP (Port `8889`)

ESP32 gửi qua UDP tới IP Laptop, cùng port `8889`, hai loại gói phân biệt bằng magic:

### 1) Gói JSON trạng thái (mỗi frame)
```json
{
  "ear": 0.28, "mar": 0.15,
  "yaw": 2.5, "pitch": -1.2, "roll": 0.0,
  "status": "NORMAL", "alarm": false, "fps": 6.5,
  "dec": 31.0, "ai": 116.0, "total": 153.0,
  "ear_thr": 0.21, "mar_thr": 0.70, "mouth_s": 0.0, "yawns": 0,
  "landmarks": [0.38, 0.42, 0.41, 0.40, ...]   // 22 điểm × (x,y) = 44 số
}
```
| Trường | Ý nghĩa |
| :--- | :--- |
| `ear`, `mar` | Chỉ số mắt/miệng hiện tại |
| `yaw`, `pitch`, `roll` | Góc đầu (độ) từ POSIT |
| `status`, `alarm` | Trạng thái FSM + cờ còi |
| `fps` | FPS cuộn của ESP32 |
| `dec`, `ai`, `total` | Thời gian (ms) từng khâu / cả frame |
| `ear_thr`, `mar_thr` | Ngưỡng ADAS hiện hành (sau hiệu chuẩn) |
| `mouth_s`, `yawns` | Thời lượng há miệng (s) + tổng số ngáp |
| `landmarks` | 22 điểm mốc chuẩn hóa [0,1] |

### 2) Gói ảnh 96×96 (đúng cái model "nhìn thấy")
```
┌────────────────────┬──────────────┬──────────────┬─────────────────────────┐
│ Magic  AA 56 AA 56 │ w (uint16 LE)│ h (uint16 LE)│ pixel (w*h, uint8 xám)  │
└────────────────────┴──────────────┴──────────────┴─────────────────────────┘
```
> [!NOTE]
> Giao thức **TCP Port 8888 / Magic `0xAA55AA55`** thuộc Giai đoạn 2 (laptop gửi ảnh) — **không còn dùng** khi chạy camera onboard.

---

## ⚙️ Cấu Hình (menuconfig)

```bash
idf.py menuconfig
```
Vào mục **`TinyDriver ADAS Configuration`**:
- **Wi-Fi SSID / Password:** mạng **2.4GHz**.
- **Laptop Host IP Address:** IP của laptop chạy viewer (lấy bằng `ipconfig`).
- **UDP Telemetry Port:** mặc định `8889`.
- **Buzzer GPIO Pin:** `2` (KHÔNG đổi).
- **Use onboard OV5640 camera:** `y`; các chân camera giữ mặc định.

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

---

## ❓ Sự Cố Thường Gặp (xem thêm tại `huong_dan_chay_project.md`)

| Hiện tượng | Cách xử lý |
| :--- | :--- |
| `Could not open COMx` | Chạy `tools\find_esp_port.ps1`; đóng monitor cũ; cắm lại cáp dữ liệu |
| `esp_camera_init 0x105` | Kiểm tra cáp FPC 24-pin; thử XCLK 10MHz |
| Log `Tensor Arena ... PSRAM` | Arena bị phân mảnh SRAM nội → xem mục tối ưu bộ nhớ ở trên |
| Laptop không thấy số liệu | Chỉ chạy 1 viewer; kiểm tra IP laptop trong menuconfig; mở firewall UDP 8889 |
