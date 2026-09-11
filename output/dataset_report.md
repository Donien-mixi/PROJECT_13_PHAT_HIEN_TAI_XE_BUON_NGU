# 🧬 Báo cáo Build Clean Dataset (Đồ án 13)

- **Thời gian build:** 1301.9s
- **File npz:** `preprocessed_driver_dataset.npz`
- **Tổng mẫu hợp lệ:** 11174 (train 10305 / val giữ-out 869)
- **Preview kiểm tra:** D:\PROJECT_13_PHAT_HIEN_BUON_NGU\output\preprocessed_preview

## 📥 Số mẫu theo nguồn

| Nguồn | Hợp lệ | Train | Val |
|---|---|---|---|
| 300W_LP | 3500 | 3234 | 266 |
| AFLW2000_3D | 1645 | 1518 | 127 |
| CEW | 2450 | 2255 | 195 |
| YawDD | 3579 | 3298 | 281 |

## 🗑️ Mẫu bị loại theo lý do (QA Gates)

| Nguồn | Lý do | Số mẫu |
|---|---|---|
| 300W_LP | extreme_pose | 3764 |
| 300W_LP | image_missing | 3669 |
| 300W_LP | duplicate | 114 |
| 300W_LP | landmark_clipped | 3 |
| AFLW2000_3D | face_too_small | 156 |
| AFLW2000_3D | extreme_pose | 115 |
| AFLW2000_3D | duplicate | 76 |
| AFLW2000_3D | landmark_clipped | 8 |
| AFLW2000_3D | image_missing | 3 |
| CEW | duplicate | 1850 |
| CEW | teacher_no_face | 174 |
| CEW | extreme_pose | 58 |
| YawDD | duplicate | 7113 |
| YawDD | extreme_pose | 37 |
| YawDD | face_too_small | 1 |

## 📐 Phân bố góc Yaw (độ) — phải phủ đều 2 phía

| Khoảng | Số mẫu |
|---|---|
| [-90, -40) | 333 |
| [-40, -20) | 1338 |
| [-20, +20) | 6322 |
| [+20, +40) | 2309 |
| [+40, +90) | 872 |

## 🥱 Phân bố trạng thái sinh trắc

| Trạng thái | Ngưỡng | Số mẫu |
|---|---|---|
| Ngáp (MAR cao) | MAR >= 0.40 | 2226 |
| Nhắm mắt | EAR < 0.21 | 3661 |
| Bình thường | còn lại | 6008 |

> ⚠️ Nếu cột Yaw [-90,-40) hoặc [40,90) bằng 0 → bổ sung 300W-LP/AFLW2000
để mô hình học bất biến góc quay (nguyên nhân tracking hỏng cũ).