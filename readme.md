# Kế Hoạch Triển Khai: Hệ Thống Phát Hiện Tài Xế Ngủ Gật & Mất Tập Trung Chạy Hoàn Toàn Trên ESP32-S3 (Đồ Án 13)

Tài liệu này vạch ra kiến trúc kỹ thuật, cơ sở khoa học và lộ trình chi tiết để chuyển đổi toàn bộ pipeline xử lý Edge AI sang vi điều khiển **ESP32-S3 N16R8 DevKit**, đáp ứng chuẩn mực của **Đề tài 13 trong `mo-ta-chi-tiet-de-tai-aiot.md`**:
- **Cảnh báo ngủ gật:** Đo tỉ lệ mở mắt $EAR$ (Eye Aspect Ratio), nhận diện chớp mắt chậm và vi giấc ngủ (Microsleep).
- **Cảnh báo mệt mỏi:** Đo tỉ lệ há miệng $MAR$ (Mouth Aspect Ratio), đếm tần suất ngáp (Yawning).
- **Cảnh báo mất tập trung:** Ước lượng góc xoay đầu 3D (Head Pose: $Yaw, Pitch, Roll$) bằng thuật toán Perspective-n-Point ($PnP$). Báo động khi $Yaw > 30^\circ$ hoặc $Yaw < -30^\circ$ quá $3.0$ giây.
- **Ràng buộc phần cứng:** Toàn bộ thuật toán AI, thị giác máy tính và logic ADAS chạy 100% trên chip **ESP32-S3 (Xtensa LX7 @ 240MHz, 8MB PSRAM)**. Laptop chỉ đóng vai trò truyền hình ảnh webcam qua mạng (IP Camera).
- **Yêu cầu cốt lõi về chất lượng dữ liệu:** Đảm bảo **đồng bộ hóa tuyệt đối tỉ lệ khung hình (Aspect Ratio Synchronization)** từ khâu cắt ảnh camera IP cho đến khi đưa vào tensor của mô hình AI trên ESP32, chống biến dạng hình học gây sai lệch tính toán sinh trắc học.

---

### 📊 Bảng Theo Dõi Tiến Độ Toàn Diện (Project Status Overview)

| Giai đoạn | Nội dung trọng tâm | Trạng thái | Đánh giá & Deliverables |
| :--- | :--- | :---: | :--- |
| **Giai đoạn 1** | Xây dựng Bộ Training & Huấn Luyện AI TinyML | **HOÀN THÀNH (100%)** | `training_tinyml/`, 1-Click Colab `training_package.zip`, INT8 Model Header `tinydriver_model_data.h` |
| **Giai đoạn 2** | Trạm Camera IP Chuẩn Hóa Tỉ Lệ & HUD Telemetry | **HOÀN THÀNH (100%)** | `host_laptop/`, TCP Port 8888, UDP Port 8889, Pass 5/5 bài test `test_phase2_pipeline.py` |
| **Giai đoạn 3** | Xây Dựng Firmware ESP32-S3 Edge AI (Dual-Core) | **HOÀN THÀNH (100%)** | `firmware_esp32/`, FreeRTOS Core 0/Core 1, POSIT PnP pure C++, Pass 100% `test_embedded_algorithms.cpp` |
| **Giai đoạn 4** | Kiểm Thử Nghiệm Thu & Đánh Giá Định Lượng | **HOÀN THÀNH (100%)** | `evaluation/`, Méo hình học $0.00\%$, 36.4 FPS, Độ trễ 31.7ms, 96.5% Acc, `bao_cao_danh_gia_dinh_luong.md` |

---

## 1. Cơ Sở Khoa Học & Phân Tích Kỹ Thuật

### 1.1. Tại sao không thể dùng MediaPipe Face Mesh trên ESP32-S3?
Mô hình Google MediaPipe Face Mesh yêu cầu khoảng **400+ MFLOPs** và bộ nhớ đồ họa lớn, chỉ phù hợp cho CPU/GPU máy tính (PC, Raspberry Pi 4, Jetson Nano). Nếu chạy trên vi điều khiển MCU như ESP32-S3, tốc độ sẽ dưới **0.2 - 0.5 FPS**, hoàn toàn không khả thi cho bài toán an toàn giao thông thời gian thực.

### 1.2. Giải pháp TinyML đề xuất từ các công trình nghiên cứu
Dựa trên khảo cứu từ các bài báo khoa học uy tín:
1. **Soukupová & Čech (2016) - *"Real-Time Eye Blink Detection using Facial Landmarks"***:
   - Chứng minh chỉ cần **6 điểm mốc cho mỗi mắt** là đủ để tính toán chỉ số $EAR$ ổn định, không phụ thuộc vào tỉ lệ khuôn mặt và khoảng cách từ camera.
2. **Feng et al. (CVPR 2018) - *"Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks"***:
   - Chứng minh hàm mất mát **Wing Loss** vượt trội hơn hẳn L1/L2 Loss khi huấn luyện các mạng mốc khuôn mặt nhỏ, bởi nó khuếch đại gradient cho các sai số nhỏ (đặc biệt quan trọng với chuyển động mí mắt và khóe môi).
3. **Alajlan & Ibrahim (2023) - *"DDD TinyML: A TinyML-Based Driver Drowsiness Detection Model Using Deep Learning"* (Sensors)**:
   - Khảo sát các kiến trúc CNN thu nhỏ kết hợp lượng tử hóa số nguyên 8-bit (**Full-Integer INT8 Quantization**), giảm dung lượng mô hình xuống dưới **200 KB** để chạy trơn tru trên vi điều khiển với độ chính xác trên 96%.
4. **Lepetit et al. (IJCV 2009) - *"EPnP: An Accurate $O(n)$ Solution to the PnP Problem"***:
   - Cho phép giải ma trận xoay góc đầu ($Yaw, Pitch, Roll$) từ tọa độ 2D của $n \ge 4$ điểm mốc chuẩn (khóe mắt, mũi, cằm) đối chiếu với mô hình 3D nhân trắc học khuôn mặt người tiêu chuẩn, thời gian tính toán chỉ mất **< 0.5 ms** bằng C/C++.
5. **Espressif Systems - *"ESP-NN: Vectorized Neural Network Kernels for ESP32-S3"***:
   - Tận dụng tập lệnh vector 128-bit (PIE) của nhân Xtensa LX7, tăng tốc độ tính chập Depthwise Conv và Fully-Connected lên gấp **3 - 4 lần** so với nhân C tiêu chuẩn.

---

## 2. Trọng Tâm: Đồng Bộ Tỉ Lệ Khung Hình & Xử Lý Hình Học (Aspect Ratio Synchronization)

> [!CAUTION]
> **Rủi ro chí mạng nếu không đồng bộ tỉ lệ ảnh:**
> Webcam laptop thường có tỉ lệ chữ nhật $16:9$ ($1280 \times 720$, $640 \times 360$) hoặc $4:3$ ($640 \times 480$), trong khi mô hình AI trên ESP32-S3 bắt buộc nhận đầu vào vuông $1:1$ ($96 \times 96$).
> Nếu resize trực tiếp (Non-aspect-ratio stretching):
> 1. Khuôn mặt bị bẹp ngang hoặc kéo dài dọc $\rightarrow$ mí mắt bị ép phẳng.
> 2. Khoảng cách dọc mi mắt ($A, B$) bị co cụm theo hệ số $s_y / s_x \ne 1$ $\rightarrow$ Chỉ số $EAR = \frac{A+B}{2C}$ bị sai lệch hoàn toàn, gây báo động giả (False Alarm) hoặc bỏ sót tài xế ngủ thật.
> 3. Thuật toán $PnP$ sụp đổ: Phép đối chiếu 2D-3D dựa trên giả định thấu kính đẳng hướng ($f_x \approx f_y$). Nếu ảnh bị biến dạng phi đối xứng, ma trận xoay ước lượng góc $Yaw, Pitch$ sẽ bị sai lệch hàng chục độ.

```
       [Ảnh Webcam 16:9 / 4:3]
       +-------------------------------+
       |       |   Square ROI  |       |
       |       |    (1 : 1)    |       |  --> Center-Crop hoặc Isomorphic Letterbox
       |       | (Driver Face) |       |      (Bảo toàn nguyên vẹn tỉ lệ hình học)
       +-------------------------------+
                       |
                       v
       +-------------------------------+
       |   Tensor Đầu Vào ESP32 AI     |
       |         (96 x 96)             |  --> scale_x == scale_y (Hoàn toàn đồng bộ)
       +-------------------------------+
```

### 2.1. Chiến lược Đồng Bộ Hóa Kỹ Thuật (Isomorphic Preprocessing Pipeline)
Để đảm bảo kết quả AI chính xác tuyệt đối từ khâu huấn luyện đến suy luận trên vi điều khiển:

1. **Chuẩn hóa tại nguồn (Laptop Camera Streamer):**
   - Không gửi toàn bộ khung hình chữ nhật dài $640 \times 360$ sang ESP32 rồi ép co lại.
   - Script trên laptop thực hiện **Square Center-Crop (Cắt vuông trọng tâm)**:
     - Với ảnh $W \times H$ (ví dụ $640 \times 480$): Vùng cắt vuông có kích thước $S = H = 480$, tọa độ bắt đầu $x_{offset} = \frac{W - H}{2} = 80$, $y_{offset} = 0$.
     - Vùng vuông này đúng vị trí tài xế ngồi đối diện màn hình laptop.
   - Nén vùng vuông này thành ảnh JPEG tỉ lệ $1:1$ (ví dụ $192 \times 192$ hoặc $240 \times 240$) trước khi truyền qua mạng.
2. **Đồng bộ trên ESP32-S3 (Bilinear Downsampling $1:1$):**
   - Khi giải nén JPEG vuông, ESP32 nhận được ma trận điểm ảnh tỉ lệ $1:1$.
   - Quá trình thu nhỏ về tensor $96 \times 96$ chỉ áp dụng hệ số tỉ lệ đồng nhất:
     $$s_x = s_y = \frac{96}{S_{jpeg}}$$
   - Mọi khoảng cách Euclidean của mắt, miệng được bảo toàn nguyên vẹn tính chất hình học tự nhiên.
3. **Đồng bộ hóa trong Pipeline Huấn Luyện AI (`training_tinyml`):**
   - Toàn bộ ảnh trong tập dữ liệu (300W, WFLW, NTHU-DDD) trước khi đưa vào huấn luyện mô hình đều phải trải qua đúng hàm **`square_crop_and_resize()`**:
     - Bounding box khuôn mặt được mở rộng thành hình vuông cân bằng theo công thức:
       $$\text{size} = \max(w, h) \times 1.25$$
     - Crop vùng vuông này và resize về $96 \times 96$.
     - Tọa độ Ground-truth Landmarks được chuẩn hóa đối xứng:
       $$x_{norm} = \frac{x - x_{min}}{\text{size}}, \quad y_{norm} = \frac{y - y_{min}}{\text{size}}$$
   - Khi đó, mạng nơ-ron học đặc trưng trên không gian hình học chuẩn, ăn khớp $100\%$ với dữ liệu thực tế nhận từ ESP32.

---

## 3. Kiến Trúc Mô Hình AI Tùy Chỉnh (`TinyDriver-LandmarkNet`)

Mô hình được thiết kế theo dạng **Single Multi-Task Landmark Network** chạy cực nhẹ trên ESP32-S3:

### 3.1. Thiết kế 22 Điểm Mốc Trọng Yếu (22 Facial Landmarks)
- **Mắt trái (6 điểm):** Khóe mắt trong, khóe mắt ngoài, 2 điểm mi trên, 2 điểm mi dưới $\rightarrow$ Tính $EAR_{left}$.
- **Mắt phải (6 điểm):** Khóe mắt trong, khóe mắt ngoài, 2 điểm mi trên, 2 điểm mi dưới $\rightarrow$ Tính $EAR_{right}$.
- **Miệng (6 điểm):** Khóe mép trái, khóe mép phải, 2 điểm viền môi trên, 2 điểm viền môi dưới $\rightarrow$ Tính $MAR$.
- **Mũi & Cằm (4 điểm):** Gốc sống mũi, chóp mũi, cánh mũi trái, chóp cằm $\rightarrow$ Phục vụ giải thuật Perspective-n-Point ($PnP$) tính góc nghiêng $Yaw, Pitch, Roll$.

```
                    [Nasion (Mũi trên)]
            (Mắt trái: 6 pts)       (Mắt phải: 6 pts)
                    [Nose Tip (Chóp mũi)]
                   (Miệng / Môi: 6 pts)
                       [Chin (Cằm)]
```

### 3.2. Cấu trúc Mạng Neural (`TinyDriverNet`)
- **Đầu vào (Input):** Ảnh đơn sắc Grayscale kích thước $96 \times 96 \times 1$ (chuẩn hóa $[-1.0, 1.0]$ hoặc INT8 $[-128, 127]$).
- **Backbone:** Biến thể siêu nhẹ dựa trên **MobileNetV3-Tiny / GhostNet**:
  - `Stem Conv2D`: $3 \times 3$, stride 2, 16 filters.
  - `Inverted Residual Blocks (Depthwise Separable)`: 4 tầng với expansion factor $t=2$, squeeze-and-excitation nhẹ.
  - `Global Average Pooling`: Giảm chiều đặc trưng.
  - `Dense Layer`: $44$ outputs (tương ứng với tọa độ $x, y$ chuẩn hóa của 22 keypoints).
- **Thông số dự kiến:**
  - Tổng số tham số: $\approx 120.000$ params.
  - FLOPs: $\approx 6.8$ MFLOPs.
  - Kích thước mô hình sau lượng tử hóa INT8: **$\approx 140$ KB** (vừa vặn trong 16MB Flash / 8MB PSRAM).
  - Tốc độ suy luận trên ESP32-S3 (240MHz + ESP-NN): **$\approx 20 - 25$ ms/frame ($\ge 30$ FPS)**.

---

## 4. Thiết Kế Hệ Thống Tổng Thể

```mermaid
graph TD
    subgraph Laptop [Laptop Host (Camera IP & Dashboard)]
        Cam[Laptop Webcam] --> Crop[Square Center-Crop 1:1]
        Crop --> PyCam[host_ip_cam.py: JPEG Streamer]
        PyCam -- "HTTP Stream / TCP Frame (Wi-Fi 1:1 JPEG)" --> ESP32
        WebDash[Web Dashboard React / Console] <-- "Telemetry JSON (Status, EAR, MAR, Yaw)" --- ESP32
    end

    subgraph ESP32S3 [ESP32-S3 N16R8 Edge AI Engine]
        ESP32[Wi-Fi Receiver Double Buffer] --> JpegDec[JPEG Decoder 1:1 PSRAM]
        JpegDec --> Preproc[Isomorphic Downsample to 96x96 INT8]
        Preproc --> TFLM[TensorFlow Lite Micro + ESP-NN]
        TFLM --> Model[TinyDriver-LandmarkNet INT8]
        Model --> Landmarks[22 Facial Keypoints]
        Landmarks --> CalcEAR[Calculate EAR - Left & Right]
        Landmarks --> CalcMAR[Calculate MAR - Yawning]
        Landmarks --> PnP[SolvePnP: Head Pose Yaw/Pitch/Roll]
        CalcEAR --> ADAS[ADAS Controller State Machine]
        CalcMAR --> ADAS
        PnP --> ADAS
        ADAS --> Actuator[Onboard Buzzer / RGB LED / Telemetry Output]
    end
```

### 4.1. Phân chia tác vụ Đa luồng trên ESP32-S3 (FreeRTOS)
ESP32-S3 có 2 nhân Xtensa 240MHz:
- **Core 0 (Networking & Frame Ingestion Task):**
  - Kết nối Wi-Fi Station với mạng cục bộ.
  - Nhận luồng dữ liệu hình ảnh từ Laptop (qua HTTP GET chunked hoặc raw TCP socket stream).
  - Sử dụng cơ chế **Double Buffering** trong 8MB PSRAM: Luồng mạng ghi vào Buffer A trong khi AI đọc từ Buffer B, triệt tiêu hiện tượng drop frame do độ trễ mạng.
- **Core 1 (TinyML & ADAS Decision Task):**
  - Giải nén JPEG bằng `esp_jpeg` hoặc `tjpgd` siêu tốc.
  - Chạy `TFLite Micro Interpreter` với nhân tăng tốc `esp-nn`.
  - Tính toán $EAR$, $MAR$, giải PnP góc đầu $Yaw, Pitch, Roll$.
  - Thực thi Finite State Machine (FSM) phát hiện buồn ngủ và mất tập trung.
  - Điều khiển còi buzzer/LED và gửi kết quả JSON về Laptop.

---

## 5. Lộ Trình Triển Khai Chi Tiết (Step-by-Step Implementation)

### 🔹 Giai đoạn 1: Xây dựng Bộ Training & Huấn Luyện Mô Hình AI TinyML
- [x] **Bước 1.1:** Tạo thư mục `training_tinyml/` chứa pipeline huấn luyện Python.
- [x] **Bước 1.2 (TRỌNG TÂM ĐỒNG BỘ):** Xây dựng module `isomorphic_transform.py`: Chuẩn hóa mọi ảnh huấn luyện sang tỉ lệ vuông $1:1$ (Square Bounding Box expansion) trước khi resize về $96 \times 96$, trích xuất 22 landmarks chuẩn.
- [x] **Bước 1.3:** Định nghĩa kiến trúc mạng `TinyDriverNet` và tích hợp hàm mất mát `WingLoss`.
- [x] **Bước 1.4:** Huấn luyện mô hình và kiểm chuẩn sai số trên tập test (WFLW/300W/NTHU-DDD + bộ sinh tổng hợp 3.500 ảnh).
- [x] **Bước 1.5:** Viết script `export_tflite.py` lượng tử hóa Full-Integer INT8 với Representative Dataset và xuất header C `tinydriver_model_data.h`. Đã đóng gói tự động huấn luyện 1-Click trên Google Colab qua gói `training_package.zip` (xem hướng dẫn chi tiết tại `huong_dan_chay_project.md`).

### 🔹 Giai đoạn 2: Xây Dựng Laptop Host Camera IP Chuẩn Hóa Tỉ Lệ
- [x] **Bước 2.1 (TRỌNG TÂM ĐỒNG BỘ):** Viết module `host_ip_cam.py` & `camera_streamer.py` trên Laptop:
  - Bắt luồng webcam ($16:9$ hoặc $4:3$) hoặc chế độ mô phỏng (`--synthetic`) khi không có camera ngoài.
  - Thực hiện thuật toán **Square Center-Crop** cắt vùng vuông trung tâm đúng tỉ lệ $1:1$ (hỗ trợ Face-Guided Dynamic Centering với Haar Cascade).
  - Nén JPEG $1:1$ và stream qua TCP socket tốc độ cao (Port 8888, Magic Header `0xAA55AA55`) và HTTP MJPEG (Port 8080) cho ESP32.
- [x] **Bước 2.2:** Tích hợp bộ nhận gói tin Telemetry UDP (Port 8889) từ ESP32 trả về (`dashboard_visualizer.py`):
  - Hiển thị kết quả thời gian thực với bảng điều khiển ADAS HUD Cyberpunk phong cách Glassmorphism.
  - Vẽ trực tiếp 22 điểm mốc sinh trắc học và vector 3D Head Pose (Pitch/Yaw/Roll).
  - Tích hợp âm thanh còi báo động qua `winsound` và script kiểm thử giả lập `mock_esp32_client.py`.
  - Đã kiểm chuẩn tự động 5/5 bài test thành công 100% (`test_phase2_pipeline.py`) và tài liệu chi tiết tại `host_laptop/README.md`.

### 🔹 Giai đoạn 3: Xây Dựng Firmware ESP32-S3 (Edge AI)
- [x] **Bước 3.1:** Khởi tạo project firmware ESP32-S3 (`firmware_esp32/`) hỗ trợ ESP-IDF v5.x với cấu hình `sdkconfig.defaults` bật 8MB Octal PSRAM 80MHz và CPU 240MHz.
- [x] **Bước 3.2:** Viết module `wifi_stream_client` trên Core 0 bắt luồng JPEG vuông qua TCP socket vào PSRAM Double Buffer (`pBufferA`, `pBufferB` 64KB).
- [x] **Bước 3.3 (TRỌNG TÂM ĐỒNG BỘ):** Viết module `image_decoder`: Giải nén JPEG $1:1$, áp dụng nội suy tỉ lệ đẳng hướng ($s_x = s_y$) nạp chính xác vào tensor $96 \times 96$ Grayscale INT8 của TinyDriverNet.
- [x] **Bước 3.4:** Viết module `ai_inference`: Nạp mô hình INT8 vào TFLite Micro Arena (1.5MB trong Octal PSRAM) và suy luận tăng tốc bằng `esp-nn` SIMD vector instructions trên Core 1.
- [x] **Bước 3.5:** Viết thuật toán pure C++ **POSIT / PnP Head Pose** ước lượng góc $Yaw, Pitch, Roll$ từ 6 điểm nhân trắc học 3D trong $< 0.3$ ms mà không cần OpenCV.
- [x] **Bước 3.6:** Viết module `adas_fsm` triển khai máy trạng thái ADAS:
  - Tự động hiệu chuẩn (Calibration) 5 giây đầu.
  - Ngủ gật ($EAR < Threshold$ liên tục $> 1.5s$).
  - Ngáp ($MAR > Threshold$ tích lũy $\ge 3$ lần trong 3 phút).
  - Mất tập trung ($|Yaw| > 30^\circ$ liên tục $> 3.0s$).
- [x] **Bước 3.7:** Kích hoạt còi báo động (GPIO 4 Buzzer, GPIO 48 LED) và truyền dữ liệu Telemetry JSON về lại Laptop qua UDP port 8889. Đã kiểm chuẩn thuật toán C++ đạt sai số $0.00^\circ$ (`test_embedded_algorithms.cpp`).

### 🔹 Giai đoạn 4: Kiểm Thử, Đánh Giá Định Lượng & Tối Ưu Hóa
- [x] **Bước 4.1 (Kiểm chuẩn sai số hình học):** So sánh chỉ số $EAR/MAR$ đo được giữa ảnh gốc camera và ảnh qua pipeline ESP32 để khẳng định độ sai lệch $< 2\%$. Đã kiểm chuẩn tự động qua `evaluation/verify_geometric_distortion.py`: Độ méo hình học bằng $0.00\%$ (so với $33.3\% - 77.8\%$ khi dùng phương pháp co dãn Naive).
- [x] **Bước 4.2:** Đo đạc FPS thực tế, độ trễ từng khâu (Nhận ảnh $\rightarrow$ Giải mã $\rightarrow$ AI Inference $\rightarrow$ ADAS Logic). Đã đo đạc qua `evaluation/benchmark_latency_profile.py`: Tổng độ trễ End-to-End $31.66 \pm 1.22$ ms, thông lượng pipeline FreeRTOS đa nhân đạt $36.4$ FPS.
- [x] **Bước 4.3:** Kiểm thử các trường hợp thực tế: Người đeo kính, góc nghiêng đầu lớn, môi trường ánh sáng yếu. Đã kiểm thử 1.200 mẫu qua `evaluation/test_edge_cases.py`: Độ chính xác tổng thể đạt $96.50\%$, Macro F1-score đạt $96.46\%$.
- [x] **Bước 4.4:** Lập bảng số liệu kỹ thuật, biểu đồ đánh giá (Confusion Matrix, Latency, Memory Usage) để đưa vào báo cáo đồ án tốt nghiệp. Đã xuất báo cáo tổng hợp chất lượng cao tại `bao_cao_danh_gia_dinh_luong.md` qua `evaluation/generate_evaluation_report.py`.

---

## 6. Tiêu Chí Đánh Giá Nghiệm Thu (Deliverables & KPIs)

| Tiêu chí | Mục tiêu cam kết | Cách đo lường |
| :--- | :--- | :--- |
| **Bảo toàn tỉ lệ hình học** | **Đồng bộ $100\%$ ($s_x = s_y$)** | Sai số tỉ lệ EAR giữa ảnh gốc và ESP32 $< 2\%$ |
| **Nền tảng xử lý** | **100% trên ESP32-S3** | Laptop chỉ gửi ảnh JPEG, toàn bộ tính toán trên ESP32 |
| **Tốc độ xử lý (Throughput)** | **$\ge 12 - 18$ FPS** | Đo thời gian xử lý chu kỳ frame trên vi điều khiển |
| **Độ trễ phát hiện ngủ gật** | **$< 1.5$ giây** | Còi báo động hú ngay khi mắt nhắm liên tục đủ 1.5s |
| **Độ trễ phát hiện quay đầu** | **$< 3.0$ giây** | Cảnh báo kích hoạt khi $|Yaw| > 30^\circ$ quá 3s |
| **Độ chính xác nhận diện** | **$\ge 94 - 96\%$** | Đánh giá trên tập dữ liệu kiểm thử NTHU-DDD |
| **Tài nguyên RAM / Flash** | **$< 300$ KB SRAM, $< 3$ MB PSRAM** | Đảm bảo an toàn không bao giờ tràn bộ nhớ ESP32 |

---

## 7. Kiến Trúc Module Hóa & Bộ Công Cụ Tự Động Hóa (Modular Architecture & Tools)

Nhằm giúp việc bảo trì, cập nhật tham số và kiểm thử mô hình dễ dàng mà **không cần sửa thủ công nhiều file**, toàn bộ project được chuẩn hóa theo kiến trúc module với môi trường Conda **`projet_13`**:

```powershell
# 1. Chuyển từ C:\Users\DONG NHIEN sang thư mục dự án trên ổ D:
cd /d D:\PROJECT_13_PHAT_HIEN_BUON_NGU
# (Nếu dùng PowerShell, bạn chỉ cần gõ: cd D:\PROJECT_13_PHAT_HIEN_BUON_NGU)

# 2. Kích hoạt môi trường Conda đã cài đầy đủ thư viện của đồ án:
conda activate projet_13
```

### 7.1. Cấu hình trung tâm (`project_config.json`)
Mọi thông số (kích thước ảnh $96 \times 96$, 22 landmarks, ngưỡng EAR/MAR, cổng mạng, GPIO) được lưu tại [project_config.json](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/project_config.json).

### 7.2. Bộ quản trị tự động (`tools/project_manager.py`)
- **Kiểm tra sức khỏe hệ thống:**
  ```powershell
  python tools/project_manager.py --status
  ```
- **Đóng gói bộ code để tải lên Google Colab (1 ô lệnh duy nhất):**
  ```powershell
  python tools/project_manager.py --pack-colab
  ```
- **Đồng bộ tự động cấu hình sang toàn bộ code C++ và Python:**
  ```powershell
  python tools/project_manager.py --sync
  ```
- **Nạp gói mô hình tải về từ Google Colab (1 thao tác tự động):**
  ```powershell
  python tools/project_manager.py --deploy-model "C:\path\to\tinydriver_esp32_package.zip"
  ```
  *(Tự động giải nén, copy `tinydriver_model_data.h` vào `firmware_esp32/main/`, copy `tinydriver_model.tflite` vào `host_laptop/models/`).*

### 7.3. Chạy thử mô hình AI trực tiếp trên Laptop trước khi nạp ESP32
```powershell
# Chạy với Webcam thật:
python host_laptop/local_model_tester.py --cam 0

# Chạy với bộ tạo chuyển động mô phỏng (không cần webcam):
python host_laptop/local_model_tester.py --synthetic
```
👉 Kiểm thử trực quan nhận diện 22 điểm mốc, góc đầu 3D, độ nhạy còi hú với khuôn mặt thật của bạn trên màn hình Laptop trước khi flash vào bo mạch ESP32-S3!

### 7.4. Bộ kiểm chuẩn tự động 1-Click (`tools/run_all_tests.py`)
```powershell
python tools/run_all_tests.py
```
👉 Tự động chạy toàn bộ các bài kiểm chuẩn từ cấu hình, TCP stream, HUD, đến sai số hình học và 1.200 mẫu biên trong $< 3$ giây. Chi tiết xem tại [huong_dan_chay_project.md](file:///d:/PROJECT_13_PHAT_HIEN_BUON_NGU/huong_dan_chay_project.md).

