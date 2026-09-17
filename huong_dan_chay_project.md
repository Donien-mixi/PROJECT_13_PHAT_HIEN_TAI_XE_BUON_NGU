# 📖 PROJECT 13 — LỆNH CẦN NHỚ (BẢN TỐI GIẢN)
Edge AI phát hiện ngủ gật & mất tập trung trên ESP32-S3 (OV5640 onboard, còi GPIO2).

---

## ⚡ TEST hằng ngày (PowerShell hoặc CMD / Anaconda Prompt)

> [!TIP]
> **Lưu ý chuyển ổ đĩa trên Windows:**
> - Nếu dùng **CMD / Anaconda Prompt**: bắt buộc dùng `cd /d D:\...` (hoặc gõ `D:` rồi mới `cd ...`), vì lệnh `cd` đơn thuần không tự đổi từ ổ `C:` sang `D:`.
> - Nếu dùng **PowerShell**: có thể gõ `cd D:\...` trực tiếp hoặc gõ `D:` rồi `cd ...`.

```powershell
# CỬA SỔ 1 — Laptop (tham chiếu, webcam):
# Chạy được trên cả CMD và PowerShell (hoặc dùng: cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU)
D:
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
conda activate projet_13
python host_laptop/local_model_tester.py --cam 0 --log-mode COMPACT --save-log
```
```powershell
# CỬA SỔ 2 — ESP32 (nạp + monitor + TỰ LƯU LOG + TỰ MỞ MÀN HÌNH TRỰC QUAN):
D:
cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU
powershell -ExecutionPolicy Bypass -File tools\flash_esp32.ps1 -Port COM3
```
📁 `flash_esp32.ps1` giờ **tự động**:
- **Mở màn hình trực quan** `esp_display_monitor.py` (cửa sổ mới) — tắt bằng `-NoDisplay`.
- **Lưu log Serial** → `output\esp32_serial_<timestamp>.txt` (con trỏ: `output\latest_esp32_serial.txt`) — tắt bằng `-NoSerialLog`.
- Cờ khác: `-BuildOnly`, `-NoBuild`, `-SaveLog` (lưu telemetry UDP — **không** chạy cùng lúc với màn hình trực quan vì tranh cổng 8889).

📁 Log tự lưu của laptop:
- `output\latest_laptop_log.txt` → `output\laptop_log_<timestamp>.txt`

Thoát Serial Monitor: **Ctrl + C** (vì log được tee ra file).

---

## 👀 XEM TRỰC QUAN ESP32 (chọn 1 — cùng giữ UDP 8889)
```powershell
python host_laptop/esp_display_monitor.py                      # viewer realtime: anh 96x96 + 22 moc + do thi
python host_laptop/esp_replay_compare.py                       # doi chieu cung dau vao (DM < 1px = firmware dung)
python tools/esp32_log_capture.py                              # chi luu log (flash_esp32.ps1 -SaveLog tu mo cai nay)
```

## 🔧 NẠP LẦN ĐẦU / ĐỔI MẠNG
```powershell
powershell -ExecutionPolicy Bypass -File tools\find_esp_port.ps1        # tim cong COM
cd firmware_esp32
idf.py menuconfig
cd ..                                                                   # SSID 2.4GHz, IP laptop, Buzzer GPIO=2
```

## 🧠 TRAIN LẠI (khi cần)
```powershell
python tools/project_manager.py --pack-colab                            # tai training_package.zip len Colab (GPU T4)
python tools/project_manager.py deploy-model tinydriver_esp32_package.zip
python evaluation/eval_nme_holdout.py                                   # NME < 6% moi nap ESP32
```
Colab cell:
```python
from google.colab import files
!pip install -q mediapipe
!rm -f training_package*.zip
uploaded = files.upload()
!unzip -q -o training_package*.zip && python run_colab_train.py
files.download('tinydriver_esp32_package.zip')
```
Thu thập thêm dữ liệu (khắc phục mốc lệch trên OV5640):
```powershell
python tools/collect_live_landmarks.py --cam 0 --target 600             # them --append
```

## ✅ KIỂM CHUẨN TỔNG
```powershell
python tools/run_all_tests.py
```

---

## ❓ SỰ CỐ NHANH
| Lỗi | Cách xử lý |
| :--- | :--- |
| `cd D:\...` không đổi thư mục (vẫn ở `C:\...`) | Trên CMD/Anaconda Prompt cần cờ `/d`: `cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU` hoặc gõ `D:` trước |
| `Could not open COMx` | `find_esp_port.ps1`; đóng monitor cũ; cáp dữ liệu + USB 3.0 |
| Không bind được UDP 8889 | Chỉ chạy 1 trong các công cụ ở mục 👀 |
| Laptop không nhận số liệu | Kiểm tra IP laptop trong `menuconfig`; firewall cho UDP 8889 |
| `esp_camera_init 0x105` | Cắm lại cáp FPC 24-pin; XCLK = 10MHz |
| `Arena ... PSRAM` (AI ~115ms) | Giảm buffer Wi-Fi / hạ `SPIRAM_MALLOC_RESERVE_INTERNAL` trong `sdkconfig` |
