# Đồ án 13: Hệ thống phát hiện tài xế ngủ gật & mất tập trung

## 1. Đề tài này làm gì?

Em làm một thiết bị nhỏ gắn trong xe để theo dõi khuôn mặt tài xế. Thiết bị dùng camera và một con chip nhỏ tên là **ESP32-S3**. Khi thấy tài xế buồn ngủ hoặc mất tập trung, thiết bị sẽ hú còi và hiện cảnh báo lên điện thoại/laptop.

Điểm đáng chú ý: **mọi tính toán chạy ngay trên con chip**, không cần máy tính hỗ trợ.

## 2. Hệ thống cảnh báo những trường hợp nào?

| Hành vi của tài xế | Cách nhận biết | Cảnh báo |
| :--- | :--- | :--- |
| Ngủ gật (nhắm mắt) | Mắt nhắm liên tục hơn 1.5 giây | Hú còi, đèn đỏ |
| Chớp mắt chậm | Mắt nhắm khoảng 0.5 giây | Nhắc nhẹ (cảnh báo vàng) |
| Ngáp / mệt mỏi | Ngáp há miệng lâu, hoặc ngáp ≥ 3 lần trong 3 phút | Cảnh báo mệt mỏi |
| Mất tập trung | Quay đầu quá 30° (hoặc cúi/ngửa quá 25°) trong 3 giây | Hú còi báo mất tập trung |

## 3. Cần chuẩn bị phần cứng gì?

- **Bo mạch ESP32-S3 N16R8** (16MB bộ nhớ trong, 8MB RAM ngoài).
- **Camera OV5640** gắn trực tiếp lên bo mạch (loại cáp DVP 24 chân).
- **Còi chip** nối vào chân **GPIO 2**.
- **Đèn LED báo** nối vào chân **GPIO 48**.
- Nguồn USB-C 5V (camera ăn khá nhiều điện, nên dùng cáp tốt).

> Lưu ý: **Không dùng chân GPIO4 cho còi**, vì chân này đã được camera dùng rồi.

## 4. Hệ thống hoạt động theo các bước nào?

```
Camera OV5640 chụp ảnh
      ↓
Giải nén ảnh JPEG
      ↓
Cắt một vùng vuông quanh khuôn mặt, thu nhỏ về 96 x 96
      ↓
Mô hình AI tìm 22 điểm trên khuôn mặt (mắt, miệng, mũi, cằm)
      ↓
Tính độ mở mắt (EAR) và độ mở miệng (MAR), tính góc quay đầu
      ↓
Máy trạng thái ADAS quyết định: bình thường / cảnh báo / báo động
      ↓
Hú còi GPIO2 + bật LED GPIO48 + gửi cảnh báo lên Web
```

**Giải thích vài từ viết tắt:**
- **EAR** (Eye Aspect Ratio) = tỉ lệ mở mắt. Giá trị nhỏ = mắt nhắm.
- **MAR** (Mouth Aspect Ratio) = tỉ lệ mở miệng. Giá trị lớn = đang ngáp.
- **Head Pose** = góc quay đầu gồm Yaw (quay trái/phải), Pitch (cúi/ngửa), Roll (nghiêng).

## 5. Mô hình AI nhận diện khuôn mặt

- **Đầu vào:** ảnh xám kích thước 96 x 96.
- **Đầu ra:** 22 điểm mốc trên khuôn mặt (mỗi điểm có tọa độ x, y).
- **Kích thước:** khoảng 200.000 tham số, file mô hình khoảng 249 KB.
- **Tốc độ trên chip:** khoảng 40–140 ms cho mỗi khung hình (tùy RAM dùng được).

Mô hình được huấn luyện bằng dữ liệu khuôn mặt người thật (AFLW2000-3D, CEW, YawDD) rồi lượng tử hóa về số nguyên 8-bit (INT8) để chạy nhẹ trên chip.

## 6. Web Dashboard (mở bằng trình duyệt)

Sau khi nạp xong, ESP32 tự phát một trang web. Chỉ cần mở trình duyệt trên điện thoại/laptop **cùng mạng Wi-Fi** và gõ:

```
http://<IP_ESP32>/
```

Trên trang này em xem được:
- Video trực tiếp từ camera.
- 22 điểm mốc vẽ đè lên mặt.
- Thước đo EAR, MAR và góc quay đầu 3D.
- Trạng thái ADAS và còi cảnh báo (chạy trên trình duyệt).
- Nút "Hiệu chuẩn lại" để đo lại khuôn mặt trong 5 giây.

## 7. Cấu trúc thư mục project

```
PROJECT_13_PHAT_HIEN_BUON_NGU/
├── training_tinyml/     # Code huấn luyện mô hình AI (Python)
├── host_laptop/         # Công cụ chạy thử trên laptop (kiểm tra trước khi nạp chip)
├── firmware_esp32/      # Firmware chạy trên chip ESP32-S3 (C/C++)
├── tools/               # Các script hỗ trợ (đóng gói, nạp, kiểm tra)
├── evaluation/          # Script đánh giá kết quả, đo độ chính xác
├── datasets/            # Dữ liệu huấn luyện (không đưa lên git)
├── output/              # Log, ảnh kiểm tra, báo cáo
└── project_config.json  # File cấu hình chung cho cả project
```

## 8. Cách chạy project (tóm tắt)

```powershell
# 1. Vào thư mục project và bật môi trường
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
conda activate projet_13

# 2. Kiểm tra tình trạng project
python tools/project_manager.py --status

# 3. (Tùy chọn) Chạy thử AI bằng webcam laptop trước
python host_laptop/local_model_tester.py --cam 0

# 4. Tìm cổng COM rồi nạp firmware vào ESP32
powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1
powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3

# 5. Mở trình duyệt vào http://<IP_ESP32>/ để xem kết quả
```

Hướng dẫn chi tiết từng bước xem ở file `huong_dan_chay_project.md`.

## 9. Kết quả đạt được

- Toàn bộ xử lý chạy trên ESP32-S3, laptop không cần tính toán.
- Tốc độ thực tế khoảng **7 FPS** (do RAM ngoài nhanh chậm hơn RAM trong chip).
- Độ trễ xử lý khoảng **140 ms** mỗi khung hình.
- Độ chính xác phát hiện trên bộ kiểm thử khoảng **96%**.
- Còi báo ngủ gật hoạt động đúng: nhắm mắt 1.5 giây là hú còi.

## 10. Ghi chú khi làm

- Trong 5 giây đầu, hệ thống tự hiệu chuẩn: nhớ **ngồi thẳng, miệng ngậm**.
- Chỉnh tham số trong `project_config.json` rồi chạy `python tools/project_manager.py --sync` để đồng bộ.
- Nếu camera bị tối, ngồi cách camera khoảng 45–65 cm và đảm bảo đủ sáng.
- Một số việc còn cần cải thiện: nhận diện mắt khi **đeo kính**, và hai điểm chóp mũi / nhân trung còn hơi gần nhau.
