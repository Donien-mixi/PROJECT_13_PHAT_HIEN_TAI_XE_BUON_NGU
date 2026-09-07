# 📖 CẨM NANG VẬN HÀNH TOÀN DIỆN PROJECT 13 (TỪ A ĐẾN Z)
### Hệ Thống Phát Hiện Tài Xế Ngủ Gật & Mất Tập Trung (100% Edge AI Trên ESP32-S3)
**Khoa Kỹ Thuật Máy Tính — Đồ Án Chuyên Ngành AIoT**

---

Tài liệu này là **hướng dẫn vận hành chính thức và duy nhất** mô tả chi tiết toàn bộ các bước để triển khai, huấn luyện mô hình, kiểm thử trên máy tính và nạp vào phần cứng ESP32-S3.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 QUY TRÌNH VẬN HÀNH 5 BƯỚC              │
                  └────────────────────────────────────────────────────────┘
                                              │
    [BƯỚC 1: ĐÓNG GÓI]  ──> python tools/project_manager.py --pack-colab
                                              │ (Tạo training_package.zip)
                                              ▼
    [BƯỚC 2: TRAIN COLAB]──> 1 Ô LỆNH DUY NHẤT TRÊN GOOGLE COLAB (GPU T4)
                                              │ (Tự động tải về tinydriver_esp32_package.zip)
                                              ▼
    [BƯỚC 3: NẠP MODEL] ──> python tools/project_manager.py --deploy-model <file.zip>
                                              │ (Tự giải nén phân phối vào Host & Firmware)
                                              ▼
    [BƯỚC 4: TEST LAPTOP]──> python host_laptop/local_model_tester.py --cam 0
                                              │ (Kiểm thử AI với Webcam Laptop + HUD + Còi hú)
                                              ▼
    [BƯỚC 5: FLASH ESP32]──> cd firmware_esp32 && idf.py build flash monitor
                                              │ (Chạy 100% Edge AI trên vi điều khiển)
                                              ▼
    [VẬN HÀNH HOÀN CHỈNH]──> Laptop làm Camera IP <---> ESP32-S3 xử lý AI thời gian thực
```

---

## 🛠️ BƯỚC 1: KÍCH HOẠT MÔI TRƯỜNG `projet_13` & KIỂM TRA HỆ THỐNG

### 1.1. Yêu cầu phần cứng
1. **Vi điều khiển:** Bo mạch **ESP32-S3 DevKit N16R8** (16MB Flash, 8MB Octal PSRAM).
2. **Máy tính / Laptop:** Có webcam tích hợp (hoặc webcam cắm ngoài cổng USB) và kết nối Wi-Fi cùng mạng với ESP32.
3. **Còi chíp báo động (Buzzer):** Còi chíp 5V hoặc 3.3V nối chân **GPIO 4** và **GND**.

### 1.2. Di chuyển vào thư mục dự án & Kích hoạt môi trường Conda `projet_13`
Nếu bạn đang mở terminal tại vị trí mặc định `(base) C:\Users\DONG NHIEN>`, hãy chạy các lệnh sau để chuyển sang ổ đĩa `D:` và kích hoạt môi trường:

```powershell
# 1. Chuyển sang thư mục dự án trên ổ D:
cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU
# (Nếu dùng PowerShell thông thường, bạn chỉ cần gõ: cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU)

# 2. Kích hoạt môi trường Conda đã cài đặt đầy đủ thư viện của đồ án:
conda activate projet_13
```
*(Khi thấy dấu nhắc lệnh chuyển thành `(projet_13) D:\PROJECT_13_PHAT_HIEN_BUON_NGU>` là bạn đã vào đúng vị trí và kích hoạt thành công)*

### 1.3. Kiểm tra sức khỏe toàn bộ project
Chạy lệnh quản trị:
```powershell
python tools/project_manager.py --status
```
Nếu hệ thống báo tất cả các module đều tồn tại, bạn sẵn sàng chuyển sang Bước 2!

---

## 🧠 BƯỚC 1.5: TIỀN XỬ LÝ CÁC TẬP DỮ LIỆU CỘNG ĐỒNG (BENCHMARK DATASETS)

Để mô hình AI có khả năng **tổng quát hóa cao nhất, nhận diện chính xác bất kỳ khuôn mặt tài xế nào trong cộng đồng**:

Chạy lệnh tiền xử lý tự động:
```powershell
python tools/project_manager.py --preprocess
```
*(Nếu có thêm thư mục ảnh dữ liệu tài xế khác của bạn, bạn có thể truyền: `python tools/preprocess_dataset.py --data-dir "duong_dan_thu_muc"`)*

👉 Lệnh này sẽ:
1. Quét qua toàn bộ các tập dữ liệu tài xế cộng đồng (bộ ảnh tài xế lái xe trong cabin ngày/đêm + bộ ảnh ngáp thật).
2. Dùng mô hình Thầy MediaPipe FaceMesh dán nhãn chuẩn 22 điểm Ground-Truth theo đúng giải phẫu học khuôn mặt người.
3. Cắt ô vuông chuẩn hóa Isomorphic Skull Anchor $96 \times 96$ Grayscale (đồng bộ 100% với ESP32-S3).
4. Xuất các ảnh kiểm tra trực quan có vẽ 22 điểm vào thư mục **`output/preprocessed_preview/`** để bạn mở xem trực tiếp độ bám dính của mí mắt và khóe môi.
5. Xuất file dữ liệu chuẩn **`training_tinyml/preprocessed_driver_dataset.npz`**.

---

## ☁️ BƯỚC 2: ĐÓNG GÓI & HUẤN LUYỆN TRÊN GOOGLE COLAB (GPU T4)

> [!TIP]
> **Đặc điểm nổi bật:** Bạn **không cần viết code trên ô lệnh Colab**! Code nằm hoàn toàn trong các file Python được đóng gói. Bạn chỉ cần nhấn Play một ô duy nhất để nạp file zip lên, Colab sẽ tự train và tự tải kết quả về!

### 2.1. Tạo file đóng gói trên máy tính của bạn
Tại thư mục gốc dự án (với môi trường `projet_13` đang kích hoạt), chạy lệnh:
```powershell
python tools/project_manager.py --pack-colab
```
👉 Lệnh này sẽ tự động thu gom mã nguồn và file dữ liệu tiền xử lý `preprocessed_driver_dataset.npz` (nếu có) để tạo file **`training_package.zip`** ngay tại thư mục dự án.

### 2.2. Huấn luyện trên Google Colab với 1 lệnh duy nhất
1. Mở trình duyệt và truy cập: **[https://colab.research.google.com/](https://colab.research.google.com/)**
2. Chọn **"Sổ tay mới" (New notebook)**.
3. Kích hoạt GPU miễn phí: Chọn menu **Thời gian chạy (Runtime)** $\rightarrow$ **Thay đổi loại thời gian chạy (Change runtime type)** $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ Bấm **Lưu (Save)**.
4. Tạo **1 ô lệnh (Cell) duy nhất** và dán đoạn code sau:

```python
# =====================================================================
# BẤM NÚT PLAY ĐỂ BẮT ĐẦU HUẤN LUYỆN TINYDRIVERNET (ĐỒ ÁN 13)
# =====================================================================
from google.colab import files

# Xóa file zip cũ nếu có trên Colab để đảm bảo luôn giải nén gói mới nhất
!rm -f training_package*.zip

print("📥 Hãy chọn file 'training_package.zip' từ máy tính của bạn:")
uploaded = files.upload()

print("🚀 Đang giải nén và bắt đầu huấn luyện...")
!unzip -q -o training_package*.zip && python run_colab_train.py

# Tự động tải gói model kết quả về máy tính
files.download('tinydriver_esp32_package.zip')
```

5. Nhấn nút **Play (Run cell)**:
   - Một nút **"Choose Files" (Chọn tệp)** sẽ xuất hiện: Bạn chọn file `training_package.zip` vừa tạo ở Bước 2.1.
   - Colab sẽ tự động giải nén, đồng bộ dữ liệu người thật **YawDD (ngáp tài xế)** và **CEW (mắt nhắm)**, dán nhãn qua **MediaPipe Teacher**, huấn luyện mạng `TinyDriverNet` (Spatial Head ~191K params) bằng hàm mất mát **Biometric-Weighted Wing Loss** (Mắt x2.0, Miệng x1.8), lượng tử hóa sang **Full-Integer INT8** (~303 KB), vẽ biểu đồ Loss và đóng gói thành file **`tinydriver_esp32_package.zip`**.
   - Khi hoàn tất, trình duyệt sẽ **tự động tải file `tinydriver_esp32_package.zip` về thư mục Downloads của máy bạn**!

---

## 📦 BƯỚC 3: NẠP MÔ HÌNH VÀO PROJECT (1 THAO TÁC TỰ ĐỘNG)

Sau khi file `tinydriver_esp32_package.zip` đã tải về máy tính của bạn, bạn **không cần giải nén hay copy thủ công rườm rà**.

Chỉ cần mở terminal tại thư mục gốc dự án và chạy câu lệnh:
```powershell
python tools/project_manager.py deploy-model tinydriver_esp32_package.zip
```
*(Nếu bạn để file zip ở thư mục khác, hãy truyền đường dẫn tới file đó, ví dụ: `python tools/project_manager.py deploy-model "C:\Users\...\Downloads\tinydriver_esp32_package.zip"`)*

Hệ thống sẽ tự động:
- Đặt file `tinydriver_model_data.h` (~1.92 MB) vào `firmware_esp32/main/` (để nạp vào ESP32).
- Đặt file `tinydriver_model.tflite` (~303 KB) vào `host_laptop/models/` và `training_tinyml/` (để chạy thử trên Laptop).
- Lưu biểu đồ huấn luyện `training_loss.png` vào `training_tinyml/`.

---

## 🖥️ BƯỚC 4: CHẠY THỬ MÔ HÌNH TRỰC TIẾP TRÊN LAPTOP TRƯỚC KHI NẠP ESP32

Trước khi mất thời gian nạp sang ESP32, bạn hãy kiểm tra chất lượng nhận diện của mô hình ngay trên Laptop bằng chính khuôn mặt thật của bạn qua Webcam:

### 4.1. Chạy với Webcam thật của máy tính
```powershell
python host_laptop/local_model_tester.py --cam 0
```
*(Nếu máy bạn có nhiều camera, bạn có thể thử `--cam 1` hoặc `--cam 2`)*

### 4.2. Chạy với chế độ mô phỏng (nếu không có camera ngoài)
```powershell
python host_laptop/local_model_tester.py --synthetic
```

### 4.3. Quan sát và kiểm thử các tính năng:
- **Giai đoạn 5 giây đầu:** Hệ thống ở trạng thái `CALIBRATING` để học hình dạng mắt và miệng của bạn khi nhìn thẳng.
- **Thử nhắm mắt $\ge 1.5$ giây:** Thanh EAR tụt xuống viền đỏ, dòng chữ `ALARM: MICROSLEEP!` kích hoạt và **loa laptop sẽ phát tiếng còi hú bíp bíp** (`winsound.Beep`).
- **Thử ngáp há to miệng $\ge 1.5$ giây:** Thanh MAR vọt lên màu tím, ghi nhận trạng thái `YAWNING DETECTED`. Nếu ngáp 3 lần trong 3 phút, còi hú báo động mệt mỏi `ALARM: FATIGUE!`.
- **Thử quay mặt sang trái/phải quá $30^\circ$ trong 3 giây:** Trục vector 3D nghiêng đi và kích hoạt báo động mất tập trung `ALARM: DISTRACTED!`.
- **Bấm phím `q` hoặc `ESC`** để thoát kiểm thử.

👉 Khi bạn đã thấy mô hình phản hồi chính xác và nhạy bén với khuôn mặt của mình, bạn tự tin chuyển sang nạp vào ESP32-S3!

---

## ⚡ BƯỚC 5: BIÊN DỊCH & NẠP FIRMWARE LÊN ESP32-S3

> [!TIP]
> **Lưu ý về Terminal biên dịch ESP32:**
> Để sử dụng lệnh `idf.py`, bạn hãy mở cửa sổ **ESP-IDF 5.x CMD** (hoặc ESP-IDF PowerShell) đã được cài đặt sẵn trên máy tính của bạn khi cài ESP-IDF.

### 5.1. Cấu hình Wi-Fi và IP máy tính cho ESP32
1. Mở PowerShell và kiểm tra địa chỉ IP nội bộ của Laptop:
   ```powershell
   ipconfig
   ```
   *(Tìm dòng **IPv4 Address**, ví dụ: `192.168.1.15`)*

2. Di chuyển vào thư mục firmware (trên cửa sổ ESP-IDF):
   ```powershell
   cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU\firmware_esp32
   ```

3. Mở menu cấu hình ESP-IDF:
   ```powershell
   idf.py menuconfig
   ```
   Chọn mục: **`TinyDriver ADAS Configuration`**:
   - **Wi-Fi SSID:** Tên Wi-Fi nhà bạn (hoặc Hotspot từ điện thoại).
   - **Wi-Fi Password:** Mật khẩu Wi-Fi.
   - **Laptop Host IP Address:** Điền địa chỉ IP máy tính bạn vừa xem ở trên.
   - Bấm phím **`S`** để Lưu $\rightarrow$ Bấm phím **`ESC`** để Thoát.

### 5.2. Nạp firmware vào mạch ESP32-S3
1. Cắm bo mạch ESP32-S3 vào cổng USB máy tính (kiểm tra cổng COM trong Device Manager, ví dụ `COM3`).
2. Chạy lệnh nạp:
   ```powershell
   idf.py set-target esp32s3
   idf.py build flash -p COM3 monitor
   ```
ESP32-S3 sẽ khởi động, kết nối Wi-Fi và chờ kết nối từ Laptop Host!

---

## 🚗 BƯỚC 6: VẬN HÀNH HỆ THỐNG THỰC TẾ HOÀN CHỈNH (LIVE DEMO)

Khi ESP32-S3 đã được nạp firmware và cắm nguồn:

### 6.1. Bật Trạm Camera IP & Bảng Điều Khiển Trên Laptop
Mở một cửa sổ PowerShell mới (nhớ kích hoạt `conda activate projet_13`) và chạy:
```powershell
python host_laptop/host_ip_cam.py --cam 0
```
*(Nếu muốn chạy mô phỏng, thêm tham số `--synthetic`)*

### 6.2. Quan sát hệ thống hoạt động tương tác 2 chiều:
1. **Chiều đi (Laptop $\rightarrow$ ESP32):**
   - Laptop bắt hình webcam, thực hiện **Square Center-Crop 1:1** đúng vị trí tài xế.
   - Nén JPEG gửi qua kết nối TCP tốc độ cao (Port 8888, Header `0xAA55AA55`).
2. **Chiều xử lý (100% tại ESP32-S3):**
   - Nhân Core 0 nhận frame qua cơ chế **PSRAM Double Buffer** (64KB).
   - Nhân Core 1 giải nén JPEG, đưa vào tensor $96 \times 96$ Grayscale INT8, chạy mô hình `TinyDriverNet` bằng tập lệnh SIMD `esp-nn`.
   - ESP32 tự giải thuật PnP thuần C++ tính góc quay đầu 3D trong $< 2\ \mu\text{s}$.
   - Máy trạng thái ADAS FSM ra quyết định: Nếu phát hiện ngủ gật/quay đầu, ESP32 **tự kích hoạt còi hú phần cứng ở chân GPIO 4**!
3. **Chiều về (ESP32 $\rightarrow$ Laptop HUD):**
   - ESP32 bắn gói tin UDP JSON (Port 8889) chứa EAR, MAR, Yaw, Pitch, Status về Laptop.
   - Bảng điều khiển Cyberpunk ADAS HUD trên màn hình laptop lập tức vẽ 22 điểm mốc, 3 trục xoay đầu 3D và hiển thị các thanh đo trạng thái thời gian thực.
4. **Xem trực tiếp trên trình duyệt Web:**
   - Bạn có thể mở trình duyệt bất kỳ và truy cập: `http://localhost:8080/video_feed` để xem luồng video 1:1.

---

## 🧪 BƯỚC 7: CHẠY BỘ KIỂM CHUẨN TỰ ĐỘNG 1-CLICK (CHO ĐỒ ÁN)

Bất cứ khi nào bạn cần kiểm tra lại toàn bộ mã nguồn hoặc thu thập số liệu phục vụ báo cáo tốt nghiệp, chỉ cần chạy 1 câu lệnh:
```powershell
python tools/run_all_tests.py
```
Hệ thống sẽ tự động thực thi 5 bài kiểm chuẩn trong chưa đầy 3 giây:
1. Kiểm tra tính đồng bộ cấu hình (`project_config.json`).
2. Kiểm thử pipeline truyền nhận TCP/UDP và HUD Giai đoạn 2.
3. Kiểm chuẩn sai số hình học ($0.00\%$ méo Isomorphic so với $77.8\%$ méo Naive).
4. Đo đạc độ trễ toàn chu trình ($31.7$ ms) và thông lượng đa nhân ($36.4$ FPS).
5. Kiểm thử độ chính xác trên $1.200$ mẫu biên khắc nghiệt ($96.50\%$ Accuracy).

👉 Báo cáo chi tiết dạng bài báo khoa học đã được xuất sẵn sàng tại: [bao_cao_danh_gia_dinh_luong.md](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/bao_cao_danh_gia_dinh_luong.md).

---

## ❓ BẢNG TRA CỨU SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING FAQ)

| Hiện tượng | Nguyên nhân khả dĩ | Cách khắc phục triệt để |
| :--- | :--- | :--- |
| **Báo lỗi `No module named cv2` hoặc `numpy`** | Chưa kích hoạt môi trường Conda `projet_13` | Chạy lệnh: `conda activate projet_13` trước khi thực thi bất kỳ lệnh Python nào. |
| **ESP32 không kết nối được Wi-Fi** | Sai tên Wi-Fi, mật khẩu hoặc dùng Wi-Fi 5GHz | ESP32 chỉ hỗ trợ Wi-Fi băng tần 2.4GHz. Hãy bật Hotspot 2.4GHz từ điện thoại hoặc kiểm tra lại `menuconfig`. |
| **ESP32 kết nối Wi-Fi nhưng báo `Connection refused`** | Sai địa chỉ IP của Laptop hoặc tường lửa Windows chặn cổng 8888 | 1. Kiểm tra lại IP bằng `ipconfig`.<br>2. Đảm bảo chạy `host_ip_cam.py` TRƯỚC KHI bật nguồn ESP32.<br>3. Cho phép Python đi qua Windows Firewall (hoặc tạm tắt Public Firewall khi demo). |
| **Hình ảnh camera bị tối hoặc không nhận được mặt** | Ánh sáng ngược hoặc ngồi quá lệch góc | Ngồi chính diện màn hình laptop ở cự ly $45 - 65$ cm, đảm bảo ánh sáng rọi đều khuôn mặt. |
| **Thay đổi tham số ADAS nhưng code không ăn theo** | Tham số chưa được đồng bộ | Mở file [project_config.json](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/project_config.json) sửa tham số, sau đó chạy: `python tools/project_manager.py --sync`. |
