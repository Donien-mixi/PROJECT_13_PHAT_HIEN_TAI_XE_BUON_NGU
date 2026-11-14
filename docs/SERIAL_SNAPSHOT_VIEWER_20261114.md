# Xem ảnh trực quan từ ESP32 qua USB serial (phương án A) — 14/11

> Không cần WiFi, không cần mật khẩu mạng. Board gửi **ảnh đã vẽ ROI + 98 điểm** qua COM3,
> laptop dựng lại thành PNG để mở/spreadsheet/chèn báo cáo.

## 1. Có xem được ảnh không? — Có

Ảnh là **BMP/xám 320×240** chứa đúng khung mà board vừa xử lý, đã vẽ:
- **khung trắng** = ROI đưa vào PIPNet (sau `td_crop_from_box`, k=1,08);
- **98 chấm trắng** = 98 điểm PIPNet board tính ra trên khung đó (2 điểm con ngươi vẽ màu 200 để phân biệt).

Kết quả thật: `output/snapshots_20261114/` (10 ảnh) — xem `snapshot_sheet.png` để thấy 8 tài xế YawDD
(3 người đeo kính) và 2 khung CEW mắt nhắm, tất cả có ROI + điểm do **chính ESP32** vẽ.

## 2. Cách chạy (firmware snapshot)

Bản firmware này bật `BLAZE_CAPTURE_MODE 1` (console **921600**) và **tắt** dump khung thô
(`BLAZE_CAPTURE_RAW 0`):

| Ảnh nạp | Dùng khi |
|---|---|
| `output/blaze_iq_20261113/blaze_snapshot_app.bin` | chụp ảnh từ **camera live** (mỗi khung ~11 s một ảnh, tối đa 20) |
| `output/blaze_iq_20261113/blaze_snapshot_multiid.bin` | chụp ảnh cho **12 khung người khác đã nhúng** (không cần ai trước camera) |

```powershell
$py='C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe'
# 1) nạp bản snapshot
& $py -m esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x10000 `
      output\blaze_iq_20261113\blaze_snapshot_app.bin
# 2) ghi serial Ở 921600 (quan trọng: bản này console 921600, không phải 115200)
& $py tools\serial_capture.py --port COM3 --baud 921600 --seconds 160 `
      --out output\blaze_iq_20261113\serial_snapshot.log
# 3) dựng ảnh
& $py tools\blaze_frame_capture.py --log output\blaze_iq_20261113\serial_snapshot.log `
      --out-dir output\snapshots_20261114
```

Mỗi ảnh ~1,4 MB serial text (≈1,5 s ở 921600). Ảnh ra: `frame_00.png`, `frame_01.png`…

## 3. Ưu / nhược so với webserver

| | Serial snapshot (A) | Webserver (B) |
|---|---|---|
| Cần mật khẩu WiFi | **Không** | **Có** (anh tự đặt trong `idf.py menuconfig`) |
| Tốc độ xem | ~2 s/ảnh sau mỗi khung | tải ảnh qua HTTP, nhưng WiFi chiếm 120–160 KB SRAM |
| Rủi ro | thấp (không đổi tầng mạng) | phải A/B lại tốc độ/ổn định, HTTP trễ tới 8,2 s |
| Xem trên điện thoại | không (phải nối dây) | có |

## 4. Giới hạn

- Ảnh là **xám 320×240**, không có màu; không có chữ số trên ảnh (EAR/threshold nằm trong log, hoặc
  trong lệnh `CAPTURE: BEGIN <idx> 320 240 EAR .. THR ..`).
- Bản snapshot làm khung chậm thêm ~2 s (0,12 → 0,10 FPS) trong lúc chụp.
- Hiện tối đa **20 ảnh/lần khởi động** (`BLAZE_SNAP_COUNT`), đổi hằng số này nếu cần nhiều.

