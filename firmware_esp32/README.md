# ⚡ Firmware ESP32-S3 Edge AI: Phát Hiện Ngủ Gật & Mất Tập Trung (Giai Đoạn 3)

Thư mục `firmware_esp32/` chứa toàn bộ mã nguồn firmware C/C++ chạy trực tiếp trên vi điều khiển **ESP32-S3 N16R8** (16MB Flash, 8MB Octal PSRAM), triển khai 100% các thuật toán Edge AI, giải thuật PnP và máy trạng thái ADAS của **Đồ án 13**.

---

## 🎯 Kiến Trúc Phân Bổ Đa Nhân FreeRTOS (Dual-Core LX7 @ 240MHz)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ESP32-S3 N16R8 SOC (Xtensa LX7 @ 240MHz)             │
├──────────────────────────────────┬─────────────────────────────────────┤
│   CORE 0: Network Ingestion Task │    CORE 1: Edge AI & ADAS Task      │
│   (Ưu tiên 5 - 8KB Stack)        │    (Ưu tiên 6 - 16KB Stack)         │
├──────────────────────────────────┼─────────────────────────────────────┤
│ 1. Kết nối Wi-Fi Station         │ 1. Giải nén JPEG bằng PSRAM Buffer  │
│ 2. TCP Client nhận frame từ Laptop│ 2. Isomorphic Downsample 96x96 INT8 │
│ 3. Đồng bộ Magic: 0xAA55AA55     │ 3. Suy luận TinyDriverNet (ESP-NN)  │
│ 4. Cơ chế Double Buffering PSRAM │ 4. Giải PnP Head Pose thuần C++     │
│    (Buffer A / Buffer B: 64KB)   │ 5. Máy trạng thái ADAS FSM          │
│ 5. Báo hiệu Semaphore sang Core 1│ 6. Gửi UDP Telemetry & Hú còi GPIO 4│
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Mã Nguồn

```text
firmware_esp32/
├── CMakeLists.txt                 # Cấu hình dự án gốc ESP-IDF
├── sdkconfig.defaults             # Cấu hình bật 8MB Octal PSRAM, 240MHz, tối ưu tốc độ
├── README.md                      # Tài liệu hướng dẫn nạp và cấu hình firmware
└── main/
    ├── CMakeLists.txt             # Đăng ký các file nguồn và thư viện phụ thuộc
    ├── Kconfig.projbuild          # Menu cấu hình Wi-Fi, IP Laptop, GPIO
    ├── main.cpp                   # Điểm khởi chạy app_main và khởi tạo 2 FreeRTOS Task
    ├── tinydriver_model_data.h    # Mảng byte mô hình INT8 (~140KB) căn lề 16-byte cho SIMD
    ├── wifi_stream_client.h/.cpp  # Nhận stream JPEG qua TCP, Double Buffering PSRAM (Core 0)
    ├── image_decoder.h/.cpp       # Giải nén JPEG 1:1, nội suy Bilinear về 96x96 INT8
    ├── ai_inference.h/.cpp        # Nạp TFLite Micro Arena (1.5MB PSRAM), nhân tăng tốc esp-nn
    ├── pnp_solver.h/.cpp          # Thuật toán POSIT/PnP thuần C++ tính Yaw, Pitch, Roll (<0.3ms)
    ├── adas_controller.h/.cpp     # Máy trạng thái ADAS FSM (Calibration, Microsleep, Yawn, Distraction)
    └── telemetry_sender.h/.cpp    # Gửi gói tin UDP JSON về Laptop và điều khiển còi Buzzer/LED
```

---

## 🔌 Sơ Đồ Đấu Nối Phần Cứng (Pinout)

| Linh kiện | Chân trên ESP32-S3 | Chức năng |
| :--- | :---: | :--- |
| **Active Buzzer (Còi chíp 5V/3.3V)** | **GPIO 4** | Kêu bíp cảnh báo chớp mắt chậm / ngáp, hú liên tục khi ngủ gật hoặc quay đầu |
| **Status / Warning LED** | **GPIO 48** (hoặc GPIO 2) | Sáng đỏ khi kích hoạt báo động ADAS |
| **Nguồn cấp DevKit** | Cổng USB-C | Cấp nguồn 5V $\ge 1A$ qua cổng USB Type-C |

---

## ⚙️ Cấu Hình Mạng & Tham Số Hệ Thống

Trước khi biên dịch, bạn cần cấu hình tên Wi-Fi và địa chỉ IP của Laptop Host. Có 2 cách:

### Cách 1: Qua giao diện menuconfig (Khuyên dùng)
```bash
idf.py menuconfig
```
Di chuyển tới mục: **`TinyDriver ADAS Configuration`**:
- **Wi-Fi SSID:** Tên Wi-Fi của bạn (hoặc điểm phát sóng di động từ điện thoại/laptop).
- **Wi-Fi Password:** Mật khẩu Wi-Fi.
- **Laptop Host IP Address:** Điền địa chỉ IP của máy tính chạy `host_ip_cam.py` (Lấy bằng lệnh `ipconfig`).
- **TCP Video Stream Port:** Mặc định `8888`.
- **UDP Telemetry Port:** Mặc định `8889`.
- **Buzzer GPIO Pin:** Mặc định `4`.

### Cách 2: Chỉnh sửa trực tiếp file `Kconfig.projbuild` hoặc file cấu hình `sdkconfig`.

---

## 🛠️ Hướng Dẫn Biên Dịch & Nạp Firmware (ESP-IDF v5.x)

### Bước 1: Thiết lập môi trường ESP-IDF
Mở **ESP-IDF Command Prompt** (hoặc chạy `. $HOME/esp/esp-idf/export.sh` trên Linux/macOS):
```bash
cd d:\PROJECT_13_PHAT_HIEN_BUON_NGU\firmware_esp32
```

### Bước 2: Thiết lập mục tiêu chip ESP32-S3
```bash
idf.py set-target esp32s3
```

### Bước 3: Biên dịch dự án
```bash
idf.py build
```

### Bước 4: Nạp firmware vào bo mạch và theo dõi Serial Monitor
Thay `COMx` bằng cổng COM thực tế của bo mạch (ví dụ `COM3` trên Windows hoặc `/dev/ttyUSB0` trên Linux):
```bash
idf.py -p COM3 flash monitor
```

---

## 🧠 Nạp Mô Hình AI Tự Động Từ Google Colab

Sau khi huấn luyện xong trên Google Colab (xem hướng dẫn chi tiết tại [huong_dan_chay_project.md](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/huong_dan_chay_project.md)):
Chỉ cần chạy lệnh nạp tự động (không cần copy thủ công):
```powershell
py tools/project_manager.py --deploy-model "C:\path\to\tinydriver_esp32_package.zip"
```
Hệ thống sẽ tự động cập nhật `tinydriver_model_data.h` vào thư mục `main/`. Sau đó bạn tiến hành nạp lại firmware (`idf.py flash monitor`).

---

## 📊 Chỉ Số Kỹ Thuật Dự Kiến Khi Vận Hành

- **Thời gian nhận & giải nén frame:** $\approx 6 - 8$ ms.
- **Thời gian suy luận AI (`TinyDriverNet` INT8):** $\approx 20 - 25$ ms.
- **Thời gian giải PnP góc đầu (Thuần C++ POSIT):** $< 0.3$ ms.
- **Thời gian cập nhật FSM & gửi Telemetry:** $< 0.5$ ms.
- **Tổng chu kỳ xử lý:** $\approx 28 - 34$ ms / frame ($\ge \mathbf{20 - 25}$ **FPS**).
- **Mức chiếm dụng bộ nhớ:**
  - SRAM nội: $< 180$ KB / 512 KB ($35\%$).
  - Octal PSRAM: $\approx 2.0$ MB / 8.0 MB ($25\%$).
