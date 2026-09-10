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

# Hoặc chạy toàn bộ 8 bài kiểm chuẩn tự động 1-click (Đồng bộ, TCP/UDP, PnP C++, 3-Way, Biên):
python tools/run_all_tests.py
```
Nếu hệ thống báo tất cả các module đều tồn tại và bài test đạt 100%, bạn sẵn sàng chuyển sang Bước 2!

---

## 🧠 BƯỚC 1.5: XÂY DỰNG DỮ LIỆU SẠCH (BUILD CLEAN DATASET - BẢN 2025)

> [!CAUTION]
> Bộ dữ liệu cũ (yawn_faces.zip + drowsiness Roboflow) chỉ có ~400 mẫu thật, nhiều ảnh
> crop cận cảnh thiếu cằm, KHÔNG có nhãn landmark chuẩn, cộng thêm Mixup landmark bị lỗi
> → là nguyên nhân gốc khiến mô hình cũ học vẹt, landmark lệch vị trí và không tracking.
> Pipeline cũ đã bị XÓA SẠCH. Bắt buộc dùng pipeline mới dưới đây.

Để mô hình AI có khả năng **tổng quát hóa cao nhất, nhận diện chính xác bất kỳ khuôn mặt tài xế nào** (kể cả góc quay đầu lớn, đeo kính, ban đêm):

Chạy lệnh xây dựng dữ liệu sạch tự động:
```powershell
python tools/project_manager.py --preprocess
# hoặc trực tiếp (khuyến nghị: thêm FaceSynthetics của Microsoft — link trực tiếp, không auth):
python tools/build_clean_dataset.py --download-aflw2000 --download-facesynth
```

> [!WARNING]
> **[v2.0.5 — Mỏ neo cằm + chống template collapse]**
> 1. **Template collapse**: model v2.0.3 đạt NME canonical 6.62% nhưng LIVE sai 17-23px vì
>    canonical crop ép mắt luôn ở v≈0.344 → model học THUỘC template vị trí. Đã sửa bằng
>    **Macro-Jitter affine** (dịch ±7%, scale 0.85-1.18, xoay ±10° mọi mẫu train) + gate
>    **NME JITTER** (chỉ số quyết định, kèm tỉ lệ Jitter/Canon < 2.0x).
> 2. **Mỏ neo cằm**: công thức box cũ không chứa nổi cằm khi ngáp → 33-39% mẫu bị clip
>    P21 thành label bẩn. Đã thêm term `d_eye_chin/1.20` (đồng bộ 1 nơi:
>    `compute_canonical_anchor` — train/label/tracking/live demo dùng cùng công thức).
>    Kết quả: landmark_clipped 1044 → **8**; ngáp 200 → **515**; |Yaw|≥40°: 173 → **555**.
> 3. **300W-LP hiện không có link tải tự động** (cbsr 404, Drive ID cũ chết) — chỉ tải
>    thủ công. Phương án tự động đáng tin: **FaceSynthetics** (Microsoft, 1000 mặt 512×512
>    nhãn 68-pt iBUG chính xác pixel).

👉 Lệnh này sẽ:
1. **Tự tải AFLW2000-3D** (~83 MB, 2000 ảnh có nhãn 68-pt 3D chuẩn, phủ góc quay đầu yaw ±90° — giải quyết triệt để lỗi không tracking khi đầu quay). *Lưu ý: nếu mạng nhà bạn chặn server CBSR (kiểm tra bằng lệnh này báo "Tải thất bại"), đừng lo — Bước 2 trên Colab sẽ tự động build dữ liệu này vì mạng Colab tải được.*
2. Tự dán nhãn 22 điểm bằng **MediaPipe Teacher** cho mọi thư mục ảnh bạn bỏ vào `datasets/raw_faces/<ten>/` (WFLW, YawDD, ảnh tự chụp webcam...).
3. Lọc qua **6 cổng chất lượng (QA Gates)**: landmark không cắt mép, mặt đủ lớn (2 mắt ≥ 10px), không nhòe (Laplacian), góc quay trong giới hạn, chống trùng lặp (aHash).
4. **Chia Train/Val giữ-out NGHIÊM NGẶT theo hash tên file** — tập val không hề xuất hiện lúc train (trước đây val lấy từ generator là ảo).
5. Xuất ảnh kiểm tra trực quan có vẽ 22 điểm vào `output/preprocessed_preview/` và báo cáo `output/dataset_report.md` (kiểm tra phân bố góc Yaw — cột ±40..90° phải có mẫu, nếu trống phải bổ sung 300W-LP).

**Nguồn dữ liệu nên bổ sung thủ công** (tải về bỏ vào `datasets/raw_faces/`):
- [300W](https://ibug.doc.ic.ac.uk/resources/facial-points/) — 3.748 ảnh + nhãn .pts 68 điểm.
- [300W-LP](https://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/) — 61.225 ảnh tổng hợp góc quay ±90° (khuyến nghị mạnh).
- [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) — 9.8k ảnh đa điều kiện (chỉ cần thư mục ảnh).
- [YawDD](https://sites.google.com/site/yawddf/) — video tài xế ngáp thật, trích frame.

**Chuẩn chất lượng phải đạt trước khi train:** tổng mẫu ≥ 5.000 (lý tưởng ≥ 20.000), mẫu |Yaw| ≥ 40° ≥ 200. Sau train xong, bắt buộc chạy `python evaluation/eval_nme_holdout.py` — NME < 6% mới được nạp ESP32.

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
# BẤM NÚT PLAY ĐỂ BẮT ĐẦU HUẤN LUYỆN TINYDRIVERNET (ĐỒ ÁN 13 - BẢN 2025)
# =====================================================================
from google.colab import files

# Cài sẵn thư viện Mô hình Thầy MediaPipe (tránh phụ thuộc auto-install)
!pip install -q mediapipe

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
   - **Nếu gói chưa kèm `preprocessed_driver_dataset.npz`** (bạn chưa build dữ liệu ở Bước 1.5 vì mạng chặn): Colab **tự động tải AFLW2000-3D (~83 MB) và tự build dữ liệu sạch** với đầy đủ 6 QA Gates + Train/Val giữ-out — bạn không phải làm gì thêm!
   - Colab huấn luyện mạng `TinyDriverNet` PFLD-Edge (~191K params) bằng **Adaptive Biometric Wing Loss + Geometric EAR/MAR constraint**, đo **NME trên tập val giữ-out THẬT mỗi epoch** (không còn val ảo từ generator) và chọn best model theo NME, sau đó lượng tử hóa **Mixed-Precision INT8** (Convs INT8 + Regression Head Float32), xuất file C Header căn lề 16-byte cho ESP-NN và đóng gói thành **`tinydriver_esp32_package.zip`**.
   - ⚡ **[v2.0.7 - STATIC-EXPAND] Tối ưu Colab T4:** augmentation được **tiền-tính 1 lần** bằng đúng thuật toán gốc (x6 bản/ảnh, ~3 phút), phần train chỉ còn GPU thuần + photometric jitter bằng TF graph ops → **toàn bộ 60 epochs ≈ 15-25 phút** (thay vì 2-5 giờ). Tự động fallback: STATIC → FAST (song song py_function) → pipeline chuẩn. Thuật toán tăng cường/loss/kiến trúc **không đổi** — chỉ đổi cách thực thi.
   - Khi hoàn tất, trình duyệt sẽ **tự động tải file `tinydriver_esp32_package.zip` về thư mục Downloads của máy bạn**!
   - 📊 **Đọc kết quả train:** dòng `Real-Val NME` cuối cùng — **< 6% là ĐẠT**. Nếu ≥ 8%, đừng nạp ESP32 mà hãy bổ sung dữ liệu (300W-LP/WFLW) rồi train lại.

---

## 📦 BƯỚC 3: NẠP MÔ HÌNH VÀO PROJECT (1 THAO TÁC TỰ ĐỘNG)

Sau khi file `tinydriver_esp32_package.zip` đã tải về máy tính của bạn, bạn **không cần giải nén hay copy thủ công rườm rà**.

Chỉ cần mở terminal tại thư mục gốc dự án và chạy câu lệnh:
```powershell
python tools/project_manager.py deploy-model tinydriver_esp32_package.zip
```
*(Nếu bạn để file zip ở thư mục khác, hãy truyền đường dẫn tới file đó, ví dụ: `python tools/project_manager.py deploy-model "C:\Users\...\Downloads\tinydriver_esp32_package.zip"`)*

Hệ thống sẽ tự động:
- Đặt file `tinydriver_model_data.h` (~1.9 MB) vào `firmware_esp32/main/` (để nạp vào ESP32).
- Đặt file `tinydriver_model.tflite` (~300 KB) vào `host_laptop/models/` và `training_tinyml/` (để chạy thử trên Laptop).
- Lưu biểu đồ huấn luyện `training_loss.png` vào `training_tinyml/`.

### 3.1. ⛔ CỔNG NGHIỆM THU BẮT BUỘC: Đo NME giữ-out (KHÔNG ĐƯỢC BỎ QUA)
```powershell
python evaluation/eval_nme_holdout.py
```
- **NME < 6%** → ĐẠT, được phép sang Bước 4.
- **NME ≥ 8%** → ❌ DỪNG: mô hình sẽ lặp lại lỗi cũ (landmark lệch, tracking trôi). Bổ sung dữ liệu vào `datasets/raw_faces/` + build lại + train lại.

---

## 🖥️ BƯỚC 4: CHẠY THỬ MÔ HÌNH TRỰC TIẾP TRÊN LAPTOP TRƯỚC KHI NẠP ESP32

Trước khi mất thời gian nạp sang ESP32, bạn hãy kiểm tra chất lượng nhận diện của mô hình ngay trên Laptop bằng chính khuôn mặt thật của bạn qua Webcam:

### 4.1. Chạy với Webcam thật của máy tính
```powershell
# Chạy bình thường (chu kỳ log mặc định 1.2s rất êm và dễ đọc)
python host_laptop/local_model_tester.py --cam 0

# Tùy chỉnh chu kỳ xuất log chậm hơn (ví dụ 2 giây/lần)
python host_laptop/local_model_tester.py --cam 0 --log-interval 2.0

# Khởi động sẵn ở chế độ COMPACT (in 1 dòng tóm tắt so sánh)
python host_laptop/local_model_tester.py --cam 0 --log-mode COMPACT
```
*(Nếu máy bạn có nhiều camera, bạn có thể thay `--cam 0` bằng `--cam 1` hoặc `--cam 2`)*

> [!IMPORTANT]
> **Đồng bộ laptop = ESP32 (bản 2025):** toàn bộ ngưỡng ADAS (Calib 5s, EAR×0.75,
> MAR×1.60, Slow Blink 0.5s, Microsleep 1.5s, Ngáp 1.5s, Mất tập trung Yaw 30°/Pitch 25°/3.0s)
> và công thức MAR `(h_outer + h_inner) / (2*w)` đã được đồng bộ **1:1** giữa
> `local_model_tester.py` và firmware `adas_controller.cpp`. Bộ test laptop ra số liệu
> thế nào thì ESP32 hành xử y hệt. Crop lúc suy luận cũng dùng đúng mỏ neo Canonical
> như lúc huấn luyện — nếu vẫn thấy landmark lệch, kiểm tra `output/dataset_report.md`.

### 4.2. Chạy với chế độ mô phỏng (nếu không có camera ngoài)
```powershell
python host_laptop/local_model_tester.py --synthetic
```

### 4.3. Quan sát và kiểm thử các tính năng:
- **Giai đoạn 5 giây đầu:** Hệ thống ở trạng thái `CALIBRATING` để học hình dạng mắt và miệng của bạn khi nhìn thẳng.
- **Thử nhắm mắt $\ge 1.5$ giây:** Thanh EAR tụt xuống viền đỏ, dòng chữ `ALARM: MICROSLEEP!` kích hoạt và **loa laptop sẽ phát tiếng còi hú bíp bíp** (`winsound.Beep`).
- **Thử ngáp há to miệng $\ge 1.5$ giây:** Thanh MAR vọt lên màu tím, ghi nhận trạng thái `YAWNING DETECTED`. Nếu ngáp 3 lần trong 3 phút, còi hú báo động mệt mỏi `ALARM: FATIGUE!`.
- **Thử quay mặt sang trái/phải quá $30^\circ$ trong 3 giây:** Trục vector 3D nghiêng đi và kích hoạt báo động mất tập trung `ALARM: DISTRACTED!`.
- **Các phím tắt điều khiển trực tiếp trên màn hình camera:**
  - **Phím `d`**: Chuyển đổi chế độ log Terminal (`FULL` bảng 2 cột $\rightarrow$ `COMPACT` 1 dòng $\rightarrow$ `OFF`).
  - **Phím `p`**: Chụp nhanh (snapshot) và in chi tiết toàn bộ 22 điểm mốc tọa độ ra Terminal ngay lập tức.
  - **Phím `m`**: Đổi Engine giữa `TINYDRIVER` (AI cục bộ) và `MEDIAPIPE` (Ground-Truth chuẩn) để đối chiếu sai số.
  - **Phím `f`**: Bật / Tắt chế độ tự động bám mặt 1:1 (`BAM MAT 1:1` vs `CAT TAM CO DINH`).
  - **Phím `r`**: Hiệu chuẩn lại baseline khuôn mặt của tài xế (`RECALIBRATE`).
  - **Phím `q` hoặc `ESC`**: Thoát kiểm thử.

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
| **`build_clean_dataset.py` báo tải AFLW2000 thất bại** | Mạng nhà bạn chặn server CBSR (Trung Quốc) | Bỏ qua Bước 1.5, chạy thẳng Bước 2 — Colab sẽ **tự động tải AFLW2000 (~83MB) và build dataset**. Hoặc tải thủ công qua Kaggle (`mohamedadlyi/aflw2000-3d`) rồi `--aflw2000-zip <file.zip>`. |
| **`idf.py build` lỗi không tìm thấy `esp_jpeg_dec.h`** | Component `esp_new_jpeg` chưa được tải | Đảm bảo có internet khi build lần đầu (Component Manager tự tải từ `main/idf_component.yml`), hoặc chạy: `idf.py add-dependency "espressif/esp_new_jpeg^1.0.2"`. |
| **Landmark vẫn lệch / tracking trôi sau khi train lại** | Dataset thiếu góc quay lớn hoặc thiếu mẫu | Mở `output/dataset_report.md`: cột Yaw ±40..90° phải có ≥ 200 mẫu, tổng ≥ 5.000 mẫu. Thiếu thì bổ sung 300W-LP/WFLW rồi build + train lại. Kiểm tra NME bằng `python evaluation/eval_nme_holdout.py` (≥ 8% là chưa đạt). |
| **Báo lỗi `No module named cv2` hoặc `numpy`** | Chưa kích hoạt môi trường Conda `projet_13` | Chạy lệnh: `conda activate projet_13` trước khi thực thi bất kỳ lệnh Python nào. |
| **ESP32 không kết nối được Wi-Fi** | Sai tên Wi-Fi, mật khẩu hoặc dùng Wi-Fi 5GHz | ESP32 chỉ hỗ trợ Wi-Fi băng tần 2.4GHz. Hãy bật Hotspot 2.4GHz từ điện thoại hoặc kiểm tra lại `menuconfig`. |
| **ESP32 kết nối Wi-Fi nhưng báo `Connection refused`** | Sai địa chỉ IP của Laptop hoặc tường lửa Windows chặn cổng 8888 | 1. Kiểm tra lại IP bằng `ipconfig`.<br>2. Đảm bảo chạy `host_ip_cam.py` TRƯỚC KHI bật nguồn ESP32.<br>3. Cho phép Python đi qua Windows Firewall (hoặc tạm tắt Public Firewall khi demo). |
| **Hình ảnh camera bị tối hoặc không nhận được mặt** | Ánh sáng ngược hoặc ngồi quá lệch góc | Ngồi chính diện màn hình laptop ở cự ly $45 - 65$ cm, đảm bảo ánh sáng rọi đều khuôn mặt. |
| **Thay đổi tham số ADAS nhưng code không ăn theo** | Tham số chưa được đồng bộ | Mở file [project_config.json](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/project_config.json) sửa tham số, sau đó chạy: `python tools/project_manager.py --sync`. |
