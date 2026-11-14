# ESP32-S3: bản đồ bộ nhớ, khả năng quan sát (webserver) và các hướng tăng tốc

> Số đo lấy từ log thật (`output/blaze_iq_20261113/serial_accept3.log`, `serial_multiid_pts.log`) và
> `firmware_esp32/sdkconfig`, `partitions.csv`, ngày 13/11/2026. Board: ESP32-S3 **N16R8** (16 MB flash,
> 8 MB PSRAM octal, 512 KB SRAM), camera OV5640 QVGA.

## 1. Bản đồ bộ nhớ

### 1.1 Flash 16 MB — phân vùng và nội dung thực tế

| Vùng | Offset | Kích thước | Đang chứa | Dùng |
|---|---|---|---|---|
| `nvs` | 0x009000 | 24 KB | cấu hình WiFi/calib (hiện gần như trống) | ~0 |
| `phy_init` | 0x00F000 | 4 KB | hiệu chuẩn RF | đủ |
| `factory` (app) | 0x010000 | **3,08 MB** | code + rodata: BlazeFace INT8 **187.416 B**, text/const, (bản test: thêm 0,92 MB ảnh nhúng) | app thật **1,79 MB (58 %)**, còn **1,29 MB** |
| `pipnet` | 0x300000 | **12 MB** | **PIPNet-98 .espdl 11,79 MB** (7 phần) — truy cập qua cache (XIP) mỗi lần suy luận | 98 % |
| `blaze` | 0xF00000 | **1 MB** | **trống** — BlazeFace giờ nằm trong rodata, không dùng partition này | **trống 1 MB** |

### 1.2 SRAM nội 512 KB — đo được

| Thời điểm | Trống | Ghi chú |
|---|---|---|
| Boot (heap_init) | 239 KiB + 21 KiB + 7 KiB RTC | 3 pool |
| Sau khi nạp đủ model + camera (chạy thật) | **252–293 KB** | đây là "trần" cho WiFi/web nếu bật |

Chứa: stack task (Core 0/1, camera, decode, detector), DMA descriptor camera, JPEG decoder, các buffer nhỏ.

### 1.3 PSRAM 8 MB octal 80 MHz — đo được

| Thời điểm | Trống | Đã dùng cho |
|---|---|---|
| Boot: pool cấp cho heap | **6 464 KB** (+50 KB gap) | ~1,5 MB còn lại bị giữ cho **rodata + instructions copy sang PSRAM** |
| Sau BlazeFace arena (602 664 B) | **5 492 KB** | arena TFLM của detector |
| Sau khi nạp PIPNet | **4 391 KB** (bản E2E) / **3 693 KB** (bản test 12 ảnh) | activation/buffer PIPNet ≈ **1,1 MB**; ảnh nhúng test ≈ 0,9 MB |

Cấu hình liên quan (sdkconfig): `SPIRAM_MODE_OCT=y`, `SPIRAM_SPEED=80`, `SPIRAM_RODATA=y`,
`SPIRAM_FETCH_INSTRUCTIONS=y` (code + const đang **chạy từ PSRAM**, nhanh hơn flash quad),
`FLASHFREQ=80M`, `FLASHMODE=QIO`, I-cache 32 KB 8-way, **D-cache 64 KB** 8-way, CPU 240 MHz, 2 core.

### 1.4 Ảnh đi qua bộ nhớ thế nào (mỗi khung)

```
OV5640 → JPEG (camera DMA, buffer PSRAM ~40 KB)
   → decode 0,60 s → khung xám 320×240 = 76,8 KB (PSRAM)
   → BlazeFace: letterbox 128×128×3 int8 (49 KB) → arena 602 KB PSRAM → box
   → crop ROI → 256×256×3 int8 (196 KB) → PIPNet (weights 11,79 MB trong flash, stream qua cache)
        activation ≈ 1,1 MB PSRAM → 98 điểm
   → EAR/MAR → FSM
```

## 2. Quan sát hệ thống: webserver có khả thi không?

| Hạng mục | Đánh giá |
|---|---|
| Dung lượng | **Đủ**: PSRAM trống 3,7–5,5 MB (1 snapshot BMP 320×240 chỉ 76,8 KB); SRAM trống 252–293 KB, WiFi+lwIP+httpd cần ~120–160 KB ⇒ **vừa, phải đo** |
| Tốc độ | MJPEG **vô nghĩa** ở 0,12 FPS. Chỉ nên làm **1 endpoint `/snapshot`** (ảnh đã vẽ ROI + 98 điểm) + `/status` JSON |
| Độ trễ HTTP | Task AI giữ Core 1 suốt 7,6 s/khung ⇒ phản hồi HTTP có thể tới **8,2 s**; httpd phải chạy task riêng, ưu tiên thấp |
| **Chặn thực tế** | **Thông tin WiFi của anh**: code mặc định `SSID mặc định (mật khẩu trống)` (`wifi_stream_client.cpp`). Muốn vào mạng nhà phải đặt `CONFIG_ESP_WIFI_SSID/PASSWORD` trong `idf.py menuconfig` — **tôi không ghi mật khẩu vào repo/log** |
| Rủi ro | Bật WiFi làm thay đổi timing (thêm ~50–70 KB SRAM, ngắt RF) ⇒ phải A/B lại tốc độ/độ ổn định |

**Phương án không cần WiFi (làm được ngay, không rủi ro):** *snapshot qua USB serial* — dùng lại cơ chế
`BLAZE_CAPTURE_MODE` (dump khung) + vẽ ROI và 98 điểm rồi gửi **BMP 76,8 KB** qua COM3 @921600
(mất ~1–2 s), mở bằng tool có sẵn (`tools/serial_capture.py` → `blaze_frame_capture.py`).
Ưu điểm: đúng ảnh board đang xử lý, không đụng WiFi, không đổi tầng mạng; nhược điểm: chỉ xem từ laptop nối dây.

## 3. Tăng tốc: các hướng phù hợp project (xếp theo hiệu quả/công sức)

| # | Hướng | Cơ chế / căn cứ | Kỳ vọng | Cần kiểm lại |
|---|---|---|---|---|
| 1 | **PIPNet input 256 → 192** | FLOPs ∝ diện tích ảnh (0,56×); ESP-DL hỗ trợ đổi shape khi quantize lại | 7,5 s → **~4,3 s**, khung ~5 s (**0,2 FPS**) | EAR/landmark phải đo lại; có thể lệch 1–2 px |
| 2 | **Bỏ decode JPEG**: cấu hình OV5640 xuất GRAYSCALE (kênh Y của YUV422) | camera DMA ghi trực tiếp khung xám, không cần decoder (đang tốn 0,60 s + buffer) | **−0,60 s** (−7 %) | xem lại pipeline decode/crop |
| 3 | **Chạy detector ở Core 0, PIPNet ở Core 1** | 2 core, detector chỉ 0,54 s | **−0,54 s** (−6,6 %) | đồng bộ ROI, latency |
| 4 | **PSRAM 80 → 120 MHz; flash 80 → 120 MHz** | ESP-IDF guide: octal PSRAM 120 MHz tăng băng thông; flash nhanh hơn giảm cache miss (>32 KB là cache thrash) | **5–20 %** (đo A/B) | module N16R8 phải chịu được; thử từng bước, đo lại |
| 5 | **Profile từng phần PIPNet (7 parts) / từng op** | ESP-DL tutorial "How to load & test & profile model" | biết phần nào nóng để tối ưu tiếp | – |
| 6 | **Ghim 1–2 part "nóng" vào PSRAM** (4,4 MB trống) | giảm đọc flash qua cache mỗi inference | ước 5–15 % nếu phần nóng chiếm nhiều | PSRAM còn đủ cho arena |
| 7 | **Giảm tần suất PIPNet** (detector mỗi khung, PIPNet mỗi 2–3 khung) | 0,12 FPS vốn đã thưa | không tăng FPS, nhưng giảm tải/điện | làm mất mẫu EAR → không khuyến nghị |
| 8 | **Prune kênh / distillation** | giảm 30–50 % FLOPs | 2–3× | **cần train** (ngoài chính sách hiện tại) |
| 9 | **Đổi board P4** | tham chiếu cũ: pipeline ESP-DL ~4,8 FPS | ~40× | đổi phần cứng, port lại |

**Thứ tự đề nghị:** (1) → (2) → (3) → (4). Ba việc đầu không cần train và có thể đưa khung từ
**8,2 s xuống ~4,5 s (0,22 FPS)** mà vẫn giữ độ chính xác nếu kiểm lại EAR đạt. Việc (4) là tinh chỉnh
cấu hình, làm sau cùng và đo A/B riêng.

Nguồn đã đọc: ESP-IDF *Support for External RAM (ESP32-S3)* — XIP từ PSRAM, PSRAM 120 MHz, giới hạn
cache khi đọc >32 KB, hạn chế DMA tới PSRAM; ESP-DL User Guide — tutorial
*How to load & test & profile model*, *How to quantize model*, *Deploy streaming models*.

