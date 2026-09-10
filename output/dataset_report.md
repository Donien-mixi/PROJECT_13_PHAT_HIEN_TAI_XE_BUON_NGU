# 🧬 Báo cáo Build Clean Dataset (Đồ án 13)

- **Thời gian build:** 17.9s
- **File npz:** `preprocessed_driver_dataset.npz`
- **Tổng mẫu hợp lệ:** 2306 (train 2111 / val giữ-out 195)
- **Preview kiểm tra:** D:\PROJECT_13_PHAT_HIEN_BUON_NGU\output\preprocessed_preview

## 📥 Số mẫu theo nguồn

| Nguồn | Hợp lệ | Train | Val |
|---|---|---|---|
| AFLW2000 | 1645 | 1499 | 146 |
| FaceSynthetics | 661 | 612 | 49 |

## 🗑️ Mẫu bị loại theo lý do (QA Gates)

| Nguồn | Lý do | Số mẫu |
|---|---|---|
| AFLW2000 | face_too_small | 156 |
| AFLW2000 | extreme_pose | 115 |
| AFLW2000 | duplicate | 76 |
| AFLW2000 | landmark_clipped | 8 |
| AFLW2000 | image_missing | 3 |
| FaceSynthetics | extreme_pose | 300 |
| FaceSynthetics | duplicate | 25 |
| FaceSynthetics | face_too_small | 14 |

## 📐 Phân bố góc Yaw (độ) — phải phủ đều 2 phía

| Khoảng | Số mẫu |
|---|---|
| [-90, -40) | 273 |
| [-40, -20) | 449 |
| [-20, +20) | 920 |
| [+20, +40) | 382 |
| [+40, +90) | 282 |

## 🥱 Phân bố trạng thái sinh trắc

| Trạng thái | Ngưỡng | Số mẫu |
|---|---|---|
| Ngáp (MAR cao) | MAR >= 0.40 | 515 |
| Nhắm mắt | EAR < 0.21 | 130 |
| Bình thường | còn lại | 1682 |

> ⚠️ Nếu cột Yaw [-90,-40) hoặc [40,90) bằng 0 → bổ sung 300W-LP/AFLW2000
để mô hình học bất biến góc quay (nguyên nhân tracking hỏng cũ).