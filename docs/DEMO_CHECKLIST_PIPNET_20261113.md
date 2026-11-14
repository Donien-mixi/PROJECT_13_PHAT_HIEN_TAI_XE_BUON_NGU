# Demo checklist — PIPNet-98 + BlazeFace INT8 trên ESP32-S3 (**bản chốt 14/11**)

> Thay cho bản 13/11 và cho `output/demo_ready_20261111/DEMO_CHECKLIST.md` (bản v3/TinyDriver).
> Chỉ ghi những gì **đã đo trên board**; phần chưa kiểm ghi rõ "chưa kiểm".

## 1. Bản chạy (bản nghiệm thu ĐẠT)

| Hạng mục | Giá trị |
|---|---|
| Firmware | `output/blaze_iq_20261113/blaze_accept_app3.bin` |
| SHA-256 | `866DFD9F59200F672D75CBB2A5867A18D39BFC6E4E4BCFCA517FA2BFD12595D2` |
| Nhịp | **8,2 s/khung (0,12 FPS)** = decode 0,61 s + BlazeFace 0,54 s + PIPNet 7,6 s |
| Bộ nhớ | Arena TFLM **602.664 B** PSRAM (đúng `arena_used+16`), PIPNet 7 phần, PSRAM còn ~5,5 MB |
| Cảnh báo mắt | 0,75 × baseline của chính tài xế; sàn 0,085; cửa sổ **2 mẫu trong 3 mẫu gần nhất** |
| Hoàn nguyên | `output/blaze_iq_20261113/restore_pipnet_app.bin` (PIPNet-only, SHA `28330A9E…028A77FD`) |
| Mạng | **Không WiFi/web** (`APP_ENABLE_WEB 0`) |

## 2. Số đo nghiệm thu (20:37, 13/11 — kịch bản 180 s, đeo kính, 55–60 cm)

| Hạng mục | Số đo |
|---|---|
| Phát hiện mặt | 25/25 lần detect có mặt; score trung vị **0,930**; cand trung vị 12; **541 ms** |
| Hiệu chuẩn | baseline EAR **0,20** → ngưỡng **0,15**; bias yaw +59,8° |
| Nhắm mắt | mẫu **0,124** và **0,101** → **`MICROSLEEP ALARM!`** (20:38:46) |
| Mắt mở | 0,171–0,199 → không cảnh báo mắt |
| Quay đầu / cúi | `DISTRACTION ALARM`; mẫu mí bị loại đúng (`eye: head_turned`) |
| Đa tài xế (thiết bị) | **12/12 khung người khác** (8 tài xế YawDD, 4 người đeo kính; 4 người mắt nhắm CEW): score 0,704–0,970, EAR mở 0,145–0,193, EAR nhắm 0,092–0,150 |
| Ổn định | 0 panic, 0 `task_wdt` (mở COM3 làm board reset 1 lần — không phải lỗi firmware) |

Hình: `output/blaze_iq_20261113/acceptance_timeline.png`, `multiid_sheet.png`.

## 3. Kịch bản demo (đúng nhịp 0,12 FPS)

> Mỗi khung 8,2 s ⇒ **mọi động tác giữ ≥15 s; nhắm mắt phải giữ ≥40 s** (30 s là biên, đã trượt 2 lần).

| Bước | Việc làm | Kỳ vọng trong log |
|---|---|---|
| 0 | Ngồi đúng khung (mặt ~70–110 px, cách **55–70 cm**), chờ ~40 s | `✅ Hiệu chuẩn thành công (baseline EAR≈0,19–0,22, bias yaw≈+60–75°)` |
| 1 | Nhìn thẳng, mắt mở 25 s | `blaze OK score 0,85–0,97`, `EAR ≈0,17–0,20`, `Status: NORMAL` |
| 2 | **Nhắm mắt 40 s** | EAR tụt (0,10–0,13) → **`MICROSLEEP ALARM!`** sau 2 mẫu (~16–24 s) |
| 3 | Mở mắt 20 s | về `NORMAL` |
| 4 | Ngáp 15 s | `YAWNING WARNING` (**chưa hiệu chỉnh — đừng hứa**) |
| 5 | Quay đầu phải/trái ≥20 s | `DISTRACTION ALARM`; `eye: head_turned` (không báo nhắm giả) |
| 6 | Cúi xuống 20 s | `DISTRACTION ALARM` (pitch) |
| 7 | Ra khỏi khung 20 s | `blaze MISS` → ROI giữa, không cảnh báo mắt |

## 4. Điều phải nói trước khi demo (giới hạn thật)

1. **0,12 FPS**: cảnh báo mắt cần 2 mẫu nhắm trong 3 mẫu gần nhất (~16–24 s). **Không** bắt được
   chớp mắt / nhắm 2–3 s. Muốn chắc: nhắm ≥40 s.
2. **Cự ly 55–75 cm** (box 69–127 px). Xa hơn: score tụt (benchmark 100 danh tính chỉ 63 % ở ngưỡng 0,60
   khi mặt < 80 px) và landmark kém; gần hơn ~40 cm (box >140 px) landmark văng (mục 9c báo cáo).
3. **Người khác**: đã kiểm **trên thiết bị** bằng 12 khung nhúng (kể cả 4 người đeo kính), **chưa kiểm
   live nhiều người** (không có người thử).
4. **Ngáp/MAR**: ngưỡng 0,48 báo cả khi không ngáp — **không demo như tính năng chắc chắn**.
5. Không dashboard/web, không còi vật lý; cảnh báo qua serial (và LED GPIO48 nếu bật).

## 4b. Xem board qua WiFi (web) — bản laze_web_app.bin

Firmware có **observation server**: Wi-Fi STA + HTTP, không đụng vào đường suy luận
(đo lại sau khi bật: Total vẫn **8.229 ms**, AI 7.624, Dec 601 — trùng bản không WiFi).

| Mục | Giá trị |
|---|---|
| Địa chỉ | **http://192.168.1.5/** (IP in ra serial: OBS: ==== MO http://…/ ====) |
 \/\ | \/\ | **trang cảnh báo**: banner TO đổi màu theo trạng thái (xám hiệu chuẩn / xanh bình thường / vàng NGÁP / đỏ BUỒN NGỦ / tím MẤT TẬP TRUNG), **nhấp nháy + đổi tiêu đề tab + bíp** khi có cảnh báo, kèm ảnh (ROI + 98 điểm), thanh EAR/MAR so ngưỡng, và danh sách 6 cảnh báo gần nhất |
| /status | JSON đầy đủ |
| /snapshot.bmp | **ảnh 320×240 xám BMP (77.878 B) có khung ROI + 98 điểm** do ESP32 vẽ |
| Yêu cầu | điện thoại/PC **cùng mạng WiFi Nhien**; board khởi động xong ~10 s mới có IP |
| Âm thanh | bấm nút **🔔 Bật âm thanh cảnh báo** một lần (trình duyệt chỉ cho phát tiếng sau 1 cú bấm) |
| Thanh EAR | thang đầy **0,40**, vạch trắng = ngưỡng; mắt mở 0,18 nên thanh ~45% là đúng |
| yaw/pitch | hiện đã trừ bias; trong ngoặc là raw + bias (raw lệch hệ thống ~43-46° là bình thường) |
| Nút ↻ | **Hiệu chuẩn lại** = POST /recalib, ngồi thẳng nhìn camera rồi bấm |
| Độ trễ | mỗi khung 8,2 s ⇒ ảnh/số liệu cập nhật chậm nhất ~8 s; KHÔNG phải video |
| Mật khẩu WiFi | nằm trong irmware_esp32/main/wifi_secret.h — **file này bị gitignore**, không vào repo |

Ghi chú: khi **chưa có mặt trong khung**, /status sẽ cho EAR/MAR rác (0,7 / 22) và state=CALIBRATING
— đó là bình thường; ngồi vào 55–60 cm thì sau ~40 s hiện NORMAL với số liệu thật.

## 5. Nạp / hoàn nguyên

```powershell
$py='C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe'
# nạp bản nghiệm thu
& $py -m esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x10000 `
      output\blaze_iq_20261113\blaze_accept_app3.bin
# hoàn nguyên PIPNet-only (bản trước khi có detector)
& $py -m esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x10000 `
      output\blaze_iq_20261113\restore_pipnet_app.bin
```

Đo: `& $py tools\serial_capture.py --port COM3 --seconds 230 --out <log>`, rồi
`& $py tools\live_session_cue.py --acceptance --closed-seconds 40 --rounds 1`
và chấm bằng `& $py tools\live_session_analyze.py <log> --acceptance --latency 7.7`.

## 6. Sự cố

| Hiện tượng | Xử lý |
|---|---|
| Không có `blaze OK` | kiểm hướng/ánh sáng camera; `cand` cao mà score 0,29–0,50 = mặt cắt mép hoặc quá xa |
| `calib sample 0/4 (waiting for a face)` kéo dài | chưa có mặt trong khung khi khởi động → ngồi vào khung rồi khởi động lại |
| EAR toàn 0,29–0,50 nhưng không có số | detector miss → PIPNet chạy ROI giữa; đưa mặt vào giữa khung |
| Cần dừng nhanh | nạp `restore_pipnet_app.bin` (mục 5) |
