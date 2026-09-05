# Mô tả chi tiết các nhóm đề tài AI, AIoT & Edge AI (Đồ án 1 & Đồ án 2 - Khoa KTMT)

Tài liệu này được rà soát, tổng hợp và chuẩn hóa từ toàn bộ danh sách đề tài đồ án chính thức của Khoa Kỹ thuật Máy tính (ĐH CNTT - ĐHQG-HCM) qua các học kỳ:
- `KTMT-Danh-sach-de-tai-Do-an-HK2-2025-2026-TB-SV-3.pdf` (83 đề tài)
- `KTMT-Danh-sach-de-tai-do-an-1-2-hoc-ky-1-nam-hoc-2025-2026.pdf` (71 đề tài)
- `PhanDuy-DA nhap ten do an.pdf` (17 đề tài)

Tài liệu phân loại toàn bộ các đề tài liên quan đến **AI (Trí tuệ nhân tạo), AIoT (AI kết hợp IoT), Edge AI (AI tính toán biên) và Hardware AI (Tăng tốc AI trên phần cứng FPGA/SoC)** thành **8 nhóm chuyên sâu (32 đề tài chi tiết)**. Mỗi đề tài đều được làm rõ mục tiêu, phần cứng, phần mềm, kiến trúc xử lý, tiêu chí đánh giá và mức độ phù hợp cho Đồ án 1 hoặc Đồ án 2.

---

## 🔹 NHÓM 1: Edge AI & Thị giác máy tính trên Kit nhúng (Raspberry Pi / Jetson / Kria KV260)

### 1. Nhận diện biển báo giao thông real-time trên Jetson Nano
- **GVHD:** Phan Đình Duy (HK1 #39, HK2 #26, File Duy #1)
- **Mục tiêu:** Phát hiện và phân loại hệ thống biển báo giao thông đường bộ Việt Nam từ camera theo thời gian thực trên kit nhúng biên.
- **Thành phần:**
  - **Phần cứng:** Jetson Nano Developer Kit (hoặc Raspberry Pi 4/5 + Google Coral USB Accelerator), Camera CSI (Sony IMX219 / Pi Cam V2) hoặc Webcam USB Full HD, màn hình HDMI/OLED hiển thị kết quả, nguồn 5V/4A ổn định.
  - **Phần mềm & AI:**
    - Dataset: GTSRB (German Traffic Sign) hoặc bộ dữ liệu biển báo giao thông Việt Nam (tự gán nhãn qua Roboflow / LabelImg).
    - Mô hình: YOLOv8n, YOLOv10n hoặc MobileNetV2-SSD (nhẹ, tối ưu hóa độ trễ).
    - Tối ưu biên: PyTorch → ONNX → TensorRT (FP16 / INT8 quantization) để đạt tốc độ > 25–30 FPS trên Jetson Nano.
    - Ngôn ngữ: Python / C++, OpenCV, JetPack SDK.
  - **Kiến trúc luồng xử lý:** Camera capture → Preprocessing (Resize, Normalization) → Inference engine (TensorRT) → NMS & Postprocessing → Bounding box + Text overlay → Cảnh báo bằng âm thanh (Buzzer/Loa) khi gặp biển báo quan trọng (dừng, cấm, giới hạn tốc độ).
  - **Đánh giá:** FPS (khung hình/giây), Latency (độ trễ ms), mAP@0.5, mức tiêu thụ tài nguyên (RAM, GPU memory, nhiệt độ chip).
  - **Độ khó:** Trung bình. Thời gian: ~2.5–3 tháng. Phù hợp Đồ án 1 hoặc Đồ án 2.

---

### 2. Nghiên cứu áp dụng thuật toán SSD và YOLO trong xử lý ảnh thời gian thực trên kit nhúng
- **GVHD:** Phan Đình Duy (HK1 #40, #41; HK2 #27, #28; File Duy #2, #3)
- **Mục tiêu:** So sánh, đánh giá chuyên sâu và tối ưu hóa hiệu năng thực tế của hai họ kiến trúc One-stage detector kinh điển: Single Shot MultiBox Detector (SSD - MobileNet-SSD) và YOLO (YOLOv5s/v8n/v11n) trên nền tảng máy tính nhúng hạn chế tài nguyên.
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4 (4GB/8GB) / Jetson Nano 4GB, Camera USB / CSI.
  - **Phần mềm & AI:**
    - Framework: TensorFlow Lite, ONNX Runtime, TensorRT, OpenVINO (nếu dùng CPU Intel).
    - Kỹ thuật tối ưu hóa biên: Post-training Quantization (PTQ - FP16, INT8), Layer Fusion, Pruning (cắt tỉa trọng số).
    - Bộ dữ liệu kiểm thử: COCO val2017, Pascal VOC hoặc bài toán phát hiện người/vật cản cụ thể.
  - **Kiến trúc luồng xử lý:** Xây dựng pipeline kiểm chuẩn tự động (Benchmark pipeline) cùng một video input chuẩn: Đo thời gian chạy từng phase (Pre-process, Inference, NMS, Draw) giữa SSD và YOLO.
  - **Đánh giá:** Lập biểu đồ đánh đổi trực quan (Trade-off): mAP vs FPS, Memory footprint, Throughput, FLOPs và công suất tiêu thụ điện (Watt).
  - **Độ khó:** Trung bình. Thời gian: ~2.5 tháng. Rất phù hợp làm báo cáo khoa học hoặc Đồ án 1/2.

---

### 3. Phân đoạn cá thể & Phân đoạn ngữ nghĩa (Instance & Semantic Segmentation) trên hệ thống nhúng
- **GVHD:** Phan Đình Duy (HK1 #42, HK2 #29, File Duy #4); Nguyễn Thanh Thiện (HK1 #54)
- **Mục tiêu:** Phân tách chính xác từng điểm ảnh (pixel-level) của vật thể hoặc vùng không gian (làn đường, vỉa hè, vật cản) phục vụ xe tự hành hoặc robot di động, chạy mượt mà trên kit Raspberry Pi, Jetson Nano hoặc AMD Xilinx Kria KV260.
- **Thành phần:**
  - **Phần cứng:** NVIDIA Jetson Nano / Orin Nano hoặc Kria KV260 Vision AI Starter Kit, Camera góc rộng.
  - **Phần mềm & AI:**
    - Mô hình Semantic Segmentation: U-Net nhẹ (MobileNet backbone), BiSeNetV2, Fast-SCNN, SegFormer-B0.
    - Mô hình Instance Segmentation: Mask R-CNN (MobileNetV3 backbone), YOLOv8-seg-nano.
    - Framework: PyTorch, ONNX, TensorRT, Vitis AI (trên Kria KV260).
  - **Kiến trúc luồng xử lý:** Video stream → Bilinear Resize → TensorRT Engine → Mask output (Sigmoid) → Binary Thresholding → Overlay màu lên ảnh gốc và đo diện tích pixel của đối tượng.
  - **Đánh giá:** mIoU (Mean Intersection over Union), Dice Coefficient, FPS, VRAM usage.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3–3.5 tháng. Phù hợp Đồ án 2.

---

### 4. Nghiên cứu và triển khai mô hình phát hiện đối tượng trong ảnh y khoa trên hệ thống nhúng
- **GVHD:** Nguyễn Thanh Thiện (HK1 #53)
- **Mục tiêu:** Xây dựng thiết bị chẩn đoán biên (Edge Medical AI Assistant) hỗ trợ phát hiện tổn thương hoặc vùng bất thường trong ảnh X-quang phổi (viêm phổi, lao), ảnh nội soi đường tiêu hóa hoặc tổn thương da (melanoma) trên kit Raspberry Pi / Jetson Nano / Kria KV260.
- **Thành phần:**
  - **Phần cứng:** Kit nhúng Jetson Nano hoặc Raspberry Pi 4 + Màn hình cảm ứng LCD hiển thị kết quả cho bác sĩ/kỹ thuật viên, cổng USB đọc ảnh từ thẻ nhớ hoặc máy quét y tế.
  - **Phần mềm & AI:**
    - Dataset công khai: ChestX-ray14, ISIC Melanoma, Kvasir-SEG (ảnh nội soi).
    - Mô hình: YOLOv8-medical fine-tuned, EfficientDet-D0, MobileNet-SSD.
    - Framework: PyTorch, TFLite Micro, ONNX Runtime. Có tích hợp giải thích quyết định AI bằng Grad-CAM (Heatmap vùng tổn thương).
  - **Kiến trúc luồng xử lý:** Nạp file ảnh DICOM/PNG y tế → Tiền xử lý (CLAHE cân bằng độ tương phản, Normalization) → Inference → Xuất tọa độ vùng nghi ngờ tổn thương + Độ tin cậy (Confidence %) + Grad-CAM heatmap overlay.
  - **Đánh giá:** Sensitivity, Specificity, ROC-AUC, F1-Score, thời gian inference trên từng ảnh.
  - **Độ khó:** Khó (đòi hỏi xử lý dữ liệu y khoa cẩn trọng, gán nhãn chuẩn và tối ưu mạng nhẹ không làm giảm độ nhạy chẩn đoán). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 5. Nghiên cứu và triển khai các mô hình thị giác - ngôn ngữ (Vision-Language Model - VLM) trên hệ thống nhúng
- **GVHD:** Nguyễn Thanh Thiện (HK1 #55)
- **Mục tiêu:** Đưa mô hình AI đa phương thức kết hợp mắt nhìn (Vision) và hiểu ngôn ngữ (Language) lên thiết bị biên để trả lời câu hỏi bằng hình ảnh (Visual Question Answering - VQA), mô tả cảnh vật thời gian thực cho người khiếm thị.
- **Thành phần:**
  - **Phần cứng:** Jetson Nano 4GB (hoặc Jetson Orin Nano / Raspberry Pi 5 8GB), Camera USB, Mic & Loa ngoài.
  - **Phần mềm & AI:**
    - Mô hình VLM kích thước nhỏ (Compact VLM / Edge VLM): SmolVLM, Moondream2, NanoLLaVA, MiniCPM-V, CLIP-ViT-Tiny.
    - Kỹ thuật nén: 4-bit / 8-bit Quantization (AWQ, GPTQ, llama.cpp / MLC-LLM), ONNX Runtime GenAI.
    - Ngôn ngữ: Python, C++, Text-to-Speech (eSpeak / Piper TTS offline).
  - **Kiến trúc luồng xử lý:** Camera chụp ảnh → Người dùng bấm nút / hỏi bằng giọng nói "Trước mặt tôi có gì?" → Vision Encoder trích xuất vector đặc trưng ảnh → LLM Decoder sinh câu trả lời văn bản → TTS đọc to phản hồi qua loa.
  - **Đánh giá:** Thời gian từ lúc hỏi đến khi có câu trả lời (Time-to-first-token, End-to-end latency), chất lượng câu trả lời (BLEU / ROUGE score), dung lượng RAM tiêu thụ.
  - **Độ khó:** Khó (đòi hỏi nắm vững kiến thức Transformer, Quantization và triển khai mô hình nén cực đại trên GPU nhúng). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 6. Nghiên cứu và triển khai mô hình phát hiện đối tượng đa phương thức (Multimodal Object Detection) trên hệ thống nhúng
- **GVHD:** Nguyễn Thanh Thiện (HK1 #56)
- **Mục tiêu:** Phát hiện và định vị đối tượng bằng cách kết hợp nhiều luồng dữ liệu (ví dụ: Camera RGB + Camera nhiệt hồng ngoại Thermal, hoặc Camera RGB + Text Prompt mô tả vật thể mở - Open-vocabulary Detection).
- **Thành phần:**
  - **Phần cứng:** Jetson Nano / Kria KV260, Camera RGB + Module cảm biến nhiệt hồng ngoại (FLIR Lepton / MLX90640) hoặc Camera độ sâu (Intel RealSense).
  - **Phần mềm & AI:**
    - Mô hình: YOLO-World (Open-Vocabulary Object Detection), Fast Grounding DINO nhẹ, hoặc mạng Early/Late Fusion kết hợp 2 nhánh RGB-Thermal CNN.
    - Dataset: FLIR Thermal Dataset, LLVIP (cặp ảnh hồng ngoại - ánh sáng khả kiến ban đêm).
  - **Kiến trúc luồng xử lý:** Đồng bộ hóa 2 khung hình RGB và Thermal theo thời gian thực → Đăng ký hình ảnh (Image Registration / Alignment) → Feature Fusion Layer (ghép đặc trưng) → Inference → Định vị người/phương tiện ngay cả trong bóng tối hoàn toàn hoặc sương mù dày đặc.
  - **Đánh giá:** mAP trong các điều kiện ánh sáng khác nhau (ngày, đêm, sương mù), độ trễ đồng bộ cảm biến, FPS.
  - **Độ khó:** Khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 7. Phát triển hệ thống gương thông minh (Smart Mirror)
- **GVHD:** Phan Đình Duy (HK1 #43, HK2 #30, File Duy #5)
- **Mục tiêu:** Xây dựng gương hai chiều tích hợp màn hình thông minh, tự động nhận diện thành viên gia đình qua camera để cá nhân hóa thông tin thời tiết, lịch trình, tin tức và ghi chú.
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4/5, màn hình LCD cũ, kính 2 chiều (Two-way mirror / Kính tráng thủy), khung gỗ/nhôm hoàn thiện thẩm mỹ, Webcam hoặc Pi Camera, cảm biến tiệm cận hồng ngoại (để bật tắt màn hình tiết kiệm điện khi có người đến gần).
  - **Phần mềm & AI:**
    - Giao diện: MagicMirror² (Node.js/Electron) hoặc tự viết Web Dashboard (React.js).
    - AI Face Recognition: InsightFace / FaceNet / DeepFace chạy nhẹ trên CPU Raspberry Pi (OpenCV DNN module).
    - Tích hợp API: OpenWeatherMap, Google Calendar API, News API RSS.
  - **Kiến trúc luồng xử lý:** Cảm biến phát hiện người tiếp cận → Đánh thức màn hình → Camera chụp frame mặt → Face Detect & Align → Face Embedding so sánh cơ sở dữ liệu khuôn mặt → Nếu nhận diện đúng người A: Hiển thị lời chào cá nhân + Lịch Google cá nhân của người A.
  - **Đánh giá:** Tốc độ nhận diện (<1.5s), độ chính xác nhận diện, tính ổn định của hệ thống khi chạy liên tục 24/7.
  - **Độ khó:** Dễ–Trung bình. Thời gian: ~1.5–2 tháng. Rất phù hợp Đồ án 1.

---

### 8. Xây dựng hệ thống điểm danh / chấm công bằng nhận diện khuôn mặt trong phòng học thông minh
- **GVHD:** Phan Đình Duy (HK1 #47, HK2 #34, File Duy #9); Nguyễn Duy Xuân Bách (HK2 #23); Nguyễn Hoài Nhân (HK1 #32)
- **Mục tiêu:** Hệ thống điểm danh tự động cho sinh viên/nhân viên qua camera cố định tại cửa lớp hoặc máy tính bảng nhúng, tự động cập nhật bảng điểm danh vào cơ sở dữ liệu và tích hợp chống gian lận (Anti-spoofing).
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4 (hoặc Jetson Nano), Camera góc rộng Full HD, Màn hình nhỏ cảm ứng hiển thị kết quả, Loa/Buzzer báo "Điểm danh thành công", Relay điều khiển mở chốt cửa điện từ tử (tùy chọn).
  - **Phần mềm & AI:**
    - Face Detection: RetinaFace hoặc MediaPipe Face Detection.
    - Anti-spoofing (Liveness Detection): MiniFASNet để phát hiện giả mạo bằng ảnh in hoặc màn hình điện thoại.
    - Face Recognition: ArcFace / MobileFaceNet (Cosine similarity với vector nhúng lưu sẵn trong SQLite/PostgreSQL).
    - Backend & Web: FastAPI (Python), Web Dashboard ReactJS / HTML5 quản lý danh sách sinh viên, xuất file Excel điểm danh.
  - **Kiến trúc luồng xử lý:** Camera stream → Detect Face → Kiểm tra Liveness (Người thật vs Ảnh chụp) → Trích xuất 512-d embedding → Search vector trong Database → Ghi nhận giờ điểm danh (Timestamp) → Phản hồi giọng nói và hiển thị tên lên màn hình.
  - **Đánh giá:** TAR (True Acceptance Rate), FAR (False Acceptance Rate), FPS, thời gian xử lý điểm danh mỗi người (< 0.8 giây).
  - **Độ khó:** Dễ–Trung bình. Thời gian: ~2 tháng. Phù hợp Đồ án 1 hoặc Đồ án 2.

---

## 🔹 NHÓM 2: Tương tác Người - Máy (HCI) & Nhận dạng cử chỉ / tư thế qua Landmark

### 9. Nghiên cứu thiết kế mô hình hệ thống nhận biết cử chỉ bàn tay thông qua các điểm mốc (Hand Gesture Recognition)
- **GVHD:** Đỗ Trí Nhựt (HK1 #26)
- **Mục tiêu:** Nhận diện và phân loại các cử chỉ bàn tay tĩnh (Static gestures: nắm, xòe, like, chỉ tay...) và cử chỉ động (Dynamic gestures: vuốt trái/phải, xoay tròn, zoom) từ tọa độ 21 điểm mốc (Landmarks) để điều khiển thiết bị không chạm (Touchless Control) hoặc trình chiếu.
- **Thành phần:**
  - **Phần cứng:** Máy tính nhúng (Raspberry Pi 4/5 hoặc laptop), Camera USB / Pi Camera, mạch Arduino/ESP32 kết nối ngoại vi (đèn LED, quạt mini, màn hình để demo điều khiển).
  - **Phần mềm & AI:**
    - Trích xuất đặc trưng hình học: Google MediaPipe Hands (cho ra 21 điểm mốc không gian 3D $x, y, z$).
    - Mô hình phân loại:
      - Cử chỉ tĩnh: Chuẩn hóa tọa độ tương đối + Multi-Layer Perceptron (MLP), SVM hoặc Random Forest.
      - Cử chỉ động: Chuỗi thời gian các điểm mốc + LSTM / 1D-CNN hoặc mô hình Hidden Markov Model (HMM).
    - Bộ điều khiển: PyAutoGUI (điều khiển slide PowerPoint, âm lượng) hoặc Serial/MQTT gửi lệnh điều khiển phần cứng.
  - **Kiến trúc luồng xử lý:** Camera capture → MediaPipe Hands trích xuất 21 landmarks → Chuẩn hóa góc quay & tỷ lệ khoảng cách bàn tay → Bộ phân loại ML → Ánh xạ cử chỉ ra lệnh điều khiển (Action Mapping) → Kích hoạt ngoại vi.
  - **Đánh giá:** Accuracy (>95%), F1-Score từng cử chỉ, độ trễ phản hồi (<50ms), khả năng hoạt động ổn định trong các điều kiện góc nghiêng bàn tay khác nhau.
  - **Độ khó:** Trung bình. Thời gian: ~2.5 tháng. Rất phù hợp Đồ án 1 hoặc Đồ án 2.

---

### 10. Nghiên cứu thiết kế mô hình hệ thống nhận biết dáng bộ, tư thế người thông qua các điểm mốc (Human Pose & Action Recognition)
- **GVHD:** Đỗ Trí Nhựt (HK1 #27)
- **Mục tiêu:** Nhận biết dáng điệu, tư thế cơ thể người (đứng, ngồi, nằm, cúi, té ngã) thông qua 33 điểm mốc giải phẫu học cơ thể (Pose Landmarks) ứng dụng trong cảnh báo té ngã cho người cao tuổi hoặc giám sát tập luyện thể thao (đếm số lần squat, hít đất).
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4 hoặc Jetson Nano, Camera Full HD gắn cố định bao quát toàn thân, Loa cảnh báo / Còi hú, Module gửi SMS (SIM800L) hoặc Telegram Bot cảnh báo người thân khi té ngã.
  - **Phần mềm & AI:**
    - Trích xuất khung xương: MediaPipe Pose hoặc YOLOv8-pose.
    - Phân loại tư thế:
      - Rule-based kết hợp hình học: Tính góc giữa các khớp xương (vai - hông - đầu gối), tỷ lệ khung bao (bounding box aspect ratio) khi đứng so với khi ngã.
      - Machine Learning: ST-GCN (Spatial Temporal Graph Convolutional Network) hoặc Light-LSTM trên chuỗi vector khung xương để phân loại hành động.
  - **Kiến trúc luồng xử lý:** Video stream → Pose Estimation trích xuất 33 keypoints → Tính vận tốc rơi của trọng tâm cơ thể và góc nghiêng thân người → Nếu phát hiện tư thế nằm bất thường quá 5 giây mà không cử động → Kích hoạt cảnh báo khẩn cấp (Còi hú + Bắn tin nhắn Telegram/SMS kèm ảnh hiện trường).
  - **Đánh giá:** Tỷ lệ cảnh báo chính xác (Precision), tỷ lệ bỏ sót tai nạn (False Negative Rate), độ trễ phát hiện té ngã (<1 giây).
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

## 🔹 NHÓM 3: Giao thông thông minh & Hệ thống hỗ trợ lái xe nâng cao (ADAS / ITS)

### 11. Xây dựng hệ thống Hỗ trợ Lái xe Nâng cao (ADAS mini): Cảnh báo lệch làn (LDW) + Hỗ trợ giữ làn (LKA)
- **GVHD:** Phan Đình Duy (HK1 #50, HK2 #40, #41, File Duy #15, #16)
- **Mục tiêu:** Xây dựng hệ thống an toàn chủ động mini trên xe mô hình hoặc thử nghiệm trên video dashcam: Phát hiện vạch kẻ làn đường thời gian thực, đo độ lệch tâm xe và đưa ra cảnh báo âm thanh khi lệch làn (Lane Departure Warning) kèm tín hiệu bẻ lái giữ làn (Lane Keeping Assist).
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4/5 hoặc Jetson Nano, Camera góc rộng gắn hướng trước kính xe, Màn hình HUD nhỏ, Servo motor điều khiển góc lái trên xe mô hình RC 1:10, còi buzzer cảnh báo.
  - **Phần mềm & AI:**
    - Tiền xử lý hình học: Perspective Transform (Bird’s-eye view / Phép chiếu mắt chim).
    - Xử lý ảnh: Bộ lọc màu vàng/trắng HLS + Canny Edge Detection + Sliding Window Polynomial Fit để tìm phương trình parabol của 2 vạch làn.
    - AI nâng cao (tùy chọn): U-Net hoặc Ultra-Fast-Lane-Detection để nhận diện làn đường đứt quãng hoặc bị mờ.
    - Thuật toán điều khiển: Tính bán kính cong của đường và sai số khoảng cách từ tâm xe đến tâm làn ($e_{lateral}$) → Bộ điều khiển PID tính góc bẻ lái servo.
  - **Kiến trúc luồng xử lý:** Camera frame → Perspective Warp → Lane Detection → Fit đường cong bậc 2 → Tính độ lệch tâm → Kích hoạt Buzzer cảnh báo nếu lệch quá ngưỡng cho phép + Xuất tín hiệu PWM ra servo để tự động đưa xe về giữa làn.
  - **Đánh giá:** Tỷ lệ nhận diện làn đường đúng (Lane detection accuracy), thời gian phát hiện lệch làn (<200ms), độ mượt khi điều khiển servo giữ làn.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

### 12. Xây dựng hệ thống Kiểm soát hành trình thích ứng (Adaptive Cruise Control - ACC)
- **GVHD:** Phan Đình Duy (HK2 #42, File Duy #17)
- **Mục tiêu:** Hệ thống tự động duy trì tốc độ cài đặt của xe và tự động giảm tốc/phanh giữ khoảng cách an toàn khi có xe phía trước di chuyển chậm hơn, ứng dụng camera AI kết hợp cảm biến đo khoảng cách.
- **Thành phần:**
  - **Phần cứng:** Xe mô hình Robot/RC tích hợp Raspberry Pi / Jetson Nano, Cảm biến đo khoảng cách phía trước (LiDAR 2D TFmini / RPLIDAR A1 hoặc cảm biến Radar sóng milimet mmWave Radar / Ultrasonic HC-SR04), Camera phía trước, Mạch điều khiển động cơ (Motor Driver H-bridge).
  - **Phần mềm & AI:**
    - Object Detection & Tracking: YOLOv8-nano + ByteTrack để phát hiện xe phía trước và theo dõi liên tục trong vùng quan sát (Region of Interest - ROI).
    - Sensor Fusion: Kết hợp dữ liệu nhận diện bounding box từ Camera với khoảng cách đo trực tiếp từ cảm biến Radar/LiDAR để ước lượng khoảng cách và vận tốc tương đối ($v_{rel}$) của xe phía trước.
    - Thuật toán điều khiển: Mô hình điều khiển ACC đa chế độ (Cruise Mode vs Following Mode vs Emergency Brake) dùng bộ điều khiển PID hoặc Fuzzy Logic.
  - **Kiến trúc luồng xử lý:** Camera + Radar đo liên tục → Tính toán Thời gian va chạm TTC (Time-to-Collision: $TTC = \frac{d}{v_{rel}}$) → Nếu khoảng cách an toàn: Duy trì tốc độ mong muốn; Nếu $TTC < 2s$: Giảm ga điều tiết động cơ; Nếu $TTC < 0.8s$: Kích hoạt phanh khẩn cấp.
  - **Đánh giá:** Độ chính xác ước lượng khoảng cách, thời gian đáp ứng phanh tự động, độ ổn định vận tốc khi bám đuôi xe trước.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

### 13. Hệ thống phát hiện tài xế ngủ gật & mất tập trung (Driver Drowsiness & Distraction Detection)
- **GVHD:** Đỗ Trí Nhựt (HK1 #28); Phan Đình Duy
- **Mục tiêu:** Giám sát liên tục trạng thái khuôn mặt của tài xế xe ô tô trong buồng lái, phát hiện dấu hiệu buồn ngủ (nhắm mắt lâu, ngáp liên tục) hoặc mất tập trung (quay mặt nhìn điện thoại) để cảnh báo kịp thời chống tai nạn giao thông.
- **Thành phần:**
  - **Phần cứng:** Raspberry Pi 4 hoặc Jetson Nano, Camera hồng ngoại (IR Camera có LED hồng ngoại chiếu sáng ban đêm), Loa cảnh báo âm thanh cường độ lớn (hoặc module rung ghế lái).
  - **Phần mềm & AI:**
    - Facial Landmark Detection: MediaPipe Face Mesh (468 điểm mốc) hoặc Dlib 68 points.
    - Chỉ số tính toán sinh trắc học mắt và miệng:
      - $EAR$ (Eye Aspect Ratio): Tính tỷ lệ mở mắt dựa trên khoảng cách giữa mi trên và mi dưới so với khóe mắt.
      - $MAR$ (Mouth Aspect Ratio): Tính độ mở rộng của miệng để phát hiện ngáp.
      - Head Pose Estimation: Tính góc xoay đầu (Pitch, Yaw, Roll) từ các điểm mốc mũi, mắt, cằm qua giải thuật PnP (Perspective-n-Point).
    - Logic cảnh báo: Nếu $EAR < Threshold$ trong $N$ khung hình liên tiếp (~1.5s) → Ngủ gật; Nếu $Yaw > 30^\circ$ quá 3s → Mất tập trung nhìn sang hướng khác.
  - **Đánh giá:** Accuracy phát hiện buồn ngủ (>96%), tốc độ xử lý trên Pi 4 (>20 FPS), độ nhạy trong môi trường thiếu sáng ban đêm.
  - **Độ khó:** Trung bình. Thời gian: ~2–2.5 tháng. Rất phù hợp Đồ án 1 hoặc Đồ án 2.

---

### 14. Hệ thống camera nhận dạng xe đi sai làn đường & Nhận diện biển số (ANPR/ALPR)
- **GVHD:** Phan Đình Duy (HK1 #50, HK2 #37, File Duy #12)
- **Mục tiêu:** Hệ thống phạt nguội giao thông thông minh: Giám sát luồng xe trên các làn đường phân chia, tự động phát hiện phương tiện lấn làn / rẽ sai quy định, đồng thời chụp ảnh trích xuất biển số xe vi phạm.
- **Thành phần:**
  - **Phần cứng:** Máy tính mini (Jetson Nano / mini PC / Laptop có card đồ họa), Camera giao thông Full HD / 2K ghi hình góc nghiêng từ trên cao.
  - **Phần mềm & AI:**
    - Lane Segmentation: Xác định ranh giới làn đường hợp lệ bằng Hough Transform hoặc mạng phân đoạn.
    - Vehicle Detection & Tracking: YOLOv8 phát hiện phương tiện (ô tô, xe máy, xe tải) + DeepSORT / ByteTrack gắn ID duy nhất cho từng xe qua từng khung hình.
    - License Plate Recognition: YOLO phát hiện vùng biển số → Cắt ảnh biển số → Căn chỉnh góc nghiêng (Affine transform) → Nhận dạng ký tự quang học (OCR dùng PaddleOCR hoặc CRNN fine-tune biển số Việt Nam).
    - Violation Logic: Kiểm tra quỹ đạo di chuyển (Trajectory) của tâm xe với đa giác làn đường quy định (Polygon collision detection).
  - **Đánh giá:** Độ chính xác phát hiện xe sai làn (>90%), độ chính xác nhận dạng ký tự biển số OCR (>92%), khả năng xử lý mượt mà video giám sát 25–30 FPS.
  - **Độ khó:** Khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

## 🔹 NHÓM 4: AIoT Giám sát Công nghiệp, Đô thị & An toàn lao động

### 15. Xây dựng hệ thống tự động giám sát xe ra vào trạm cân
- **GVHD:** Trương Văn Cương (HK2 #43)
- **Mục tiêu:** Tự động hóa toàn diện quy trình kiểm soát xe tải tại trạm cân hàng hóa: Tự động nhận diện biển số xe, đọc chỉ số trọng lượng trực tiếp từ màn hình cân điện tử bằng OCR, ước lượng dung tích thùng xe bằng phân tích hình ảnh và phát hiện các hành vi gian lận cân (xe vào nhưng không cân, đỗ xe không đúng vị trí, cân sai quy trình).
- **Thành phần:**
  - **Phần cứng:** Hệ thống gồm 2–3 Camera IP công nghiệp (1 camera chụp biển số trước, 1 camera chụp toàn cảnh thùng xe từ trên cao, 1 camera macro hướng vào đồng hồ số LED của đầu cân điện tử indicator), Máy tính xử lý trạm cân, Bộ điều khiển Barrier tự động và đèn tín hiệu xanh/đỏ (kết nối qua Relay/Modbus).
  - **Phần mềm & AI:**
    - Nhận diện biển số tự động (ANPR): YOLO + OCR chuyên dụng cho biển số xe tải.
    - OCR đọc màn hình cân số: Fine-tune mô hình đọc số 7-đoạn LED (Seven-segment display) hoặc ký tự số LCD từ camera để đối chiếu độc lập với cổng truyền thông RS232 của đầu cân (chống can thiệp phần cứng cân).
    - Ước lượng dung tích và thể tích thùng xe: Thuật toán thị giác phân tích độ võng lốp xe, góc nghiêng nhíp xe hoặc phân đoạn 3D thể tích vật liệu trong thùng hàng.
    - Thuật toán phát hiện gian lận: Kiểm tra vị trí bánh xe có nằm trọn trong bàn cân hay không (thông qua vạch giới hạn); kiểm tra thời gian xe dừng đủ chuẩn; phát hiện xe chạy vượt qua trạm mà không dừng cân.
  - **Kiến trúc luồng xử lý:** Xe tiến vào bàn cân → Camera trước nhận diện biển số → Camera màn hình cân đọc và lưu số kg ổn định → Camera trên cao kiểm tra thể tích thùng hàng → Hệ thống kiểm tra hợp lệ → Mở barrier và in phiếu cân điện tử, lưu log hình ảnh bằng chứng vào Database.
  - **Đánh giá:** Tỷ lệ đọc đúng biển số (>96%), tỷ lệ đọc đúng số cân LED (>99.5%), tỷ lệ phát hiện gian lận vị trí cân, thời gian xử lý mỗi lượt cân (<3 giây).
  - **Độ khó:** Khó (kết hợp cả hệ thống thị giác đa camera, OCR chuyên biệt và nghiệp vụ kiểm soát công nghiệp chống gian lận). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 16. Hệ thống đánh giá hiệu suất & phát hiện hành vi bất thường của công nhân tại nơi làm việc
- **GVHD:** Trương Văn Cương (HK2 #44)
- **Mục tiêu:** Kết nối camera giám sát tại nhà xưởng/dây chuyền sản xuất để phân tích hành vi lao động thực tế của công nhân: Xác định công nhân có thực sự làm việc hay chỉ điểm danh rồi rời đi; theo dõi thời gian hiện diện tại vị trí làm việc; phát hiện tự động các hành vi bất thường (ngồi chơi bấm điện thoại, tụ tập nói chuyện, ngủ gật trong giờ làm việc).
- **Thành phần:**
  - **Phần cứng:** Camera IP giám sát bao quát khu vực chuyền sản xuất, Edge Server / Mini PC trang bị GPU (hoặc Jetson Orin Nano).
  - **Phần mềm & AI:**
    - Face Detection & Re-Identification (ReID): Nhận diện danh tính công nhân và theo dõi liên tục một người qua nhiều camera không bị mất dấu.
    - Pose Estimation & Action Recognition: Dùng YOLOv8-pose kết hợp mạng phân tích hành động (SlowFast hoặc Spatio-Temporal Graph CNN) để phân loại tư thế làm việc chuẩn vs ngồi bấm điện thoại / gục đầu ngủ.
    - Dwell Time & Area of Interest (AOI): Định nghĩa vùng làm việc hợp lệ; tính toán tích lũy thời gian có mặt thực tế tại vị trí so với thời gian ca làm việc.
  - **Kiến trúc luồng xử lý:** Camera stream → Detect công nhân + trích xuất ReID feature → Tracking định danh → Phân tích pose từng người theo chuỗi thời gian → Nếu phát hiện tư thế ngủ gục hoặc vắng mặt khỏi vị trí quá thời gian cho phép → Ghi nhận vi phạm kèm video cắt ngắn (clip bằng chứng) gửi về bảng điều khiển quản lý nhà máy.
  - **Đánh giá:** Độ chính xác phân loại hành vi (F1-score > 88%), độ chính xác ReID qua khung hình, độ trễ cảnh báo.
  - **Độ khó:** Khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 17. Hệ thống phát hiện hành vi bất thường trong quy trình order tại quầy cho chuỗi cafe
- **GVHD:** Trương Văn Cương (HK2 #46)
- **Mục tiêu:** Ứng dụng thị giác máy tính giám sát trực tiếp khu vực quầy thu ngân của quán cafe để phát hiện gian lận doanh thu: Nhân viên thu tiền mặt của khách nhưng không bấm tạo bill, gian lận đổi hủy bill sau khi khách rời đi, hoặc khách trả tiền nhưng nhân viên cố tình không in hóa đơn giao khách.
- **Thành phần:**
  - **Phần cứng:** Camera góc nhìn trên cao (Top-down view) hướng thẳng vào bàn quầy thu ngân (bao quát tay nhân viên, khay tiền và máy POS), Cổng mạng LAN/WiFi đồng bộ với phần mềm POS thu ngân.
  - **Phần mềm & AI:**
    - Hand Action Detection & Object Interaction: Phát hiện và theo dõi bàn tay (Hand tracking) tương tác với các vật thể: tiền mặt, thẻ ngân hàng, máy POS, máy in bill và ly nước.
    - Event Correlation Engine: Kết nối thời gian thực giữa sự kiện thị giác (Visual Event: Khách đưa tiền mặt, nhân viên cầm tiền bỏ vào ngăn kéo) và sự kiện phần mềm POS (POS Event: Lệnh tạo bill được ghi nhận trên hệ thống).
    - Phát hiện bất thường: Nếu camera ghi nhận hành động nhận tiền và mở két tiền nhưng trong khoảng thời gian $\pm 10$ giây không có giao dịch in bill tương ứng trên POS $\rightarrow$ Kích hoạt cờ cảnh báo nghi vấn gian lận.
  - **Đánh giá:** Precision và Recall của thuật toán phát hiện hành vi trao đổi tiền - bill, tỷ lệ báo động giả (False Alarm Rate), khả năng bảo mật dữ liệu khách hàng.
  - **Độ khó:** Khó (bài toán nhận diện tương tác người - vật thể Hand-Object Interaction đòi hỏi độ chi tiết cao). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 18. Giải pháp an toàn cẩu tháp cho công trình xây dựng (Radar công trường + Camera AI)
- **GVHD:** Trương Văn Cương (HK2 #47)
- **Mục tiêu:** Hệ thống an toàn lao động thông minh giám sát vùng hoạt động nguy hiểm của cẩu tháp công trình xây dựng: Sử dụng kết hợp camera AI và cảm biến radar để phát hiện người/xe cơ giới xâm nhập vào bán kính quay nguy hiểm của cẩu tháp, phát hiện công nhân không đội mũ bảo hiểm/đồ bảo hộ và phát hiện các điều kiện vận hành mất an toàn (góc quay vượt giới hạn, tải trọng rung lắc mạnh).
- **Thành phần:**
  - **Phần cứng:** Bộ điều khiển trung tâm trên buồng lái cẩu tháp (Jetson Nano / Bộ điều khiển công nghiệp), Camera PTZ ngoài trời chuẩn IP67 gắn trên thân cẩu tháp, Cảm biến Radar 24GHz/77GHz hoặc Cảm biến đo khoảng cách laser tầm xa, Cảm biến góc nghiêng/gia tốc kế (IMU) gắn trên móc cẩu, Còi báo động công suất lớn và màn hình hiển thị trực quan cho tài xế cẩu tháp.
  - **Phần mềm & AI:**
    - AI Vision: YOLOv8 phát hiện người (Person Detection), phát hiện thiết bị bảo hộ cá nhân PPE (Mũ bảo hộ hardhat, áo phản quang).
    - Radar Processing: Quét và lập bản đồ các chướng ngại vật kim loại trong phạm vi góc quay 360 độ của cẩu tháp.
    - Dynamic Danger Zone Mapping: Tự động tính toán bán kính vùng nguy hiểm rơi vật thể dựa vào độ cao hiện tại của móc cẩu và vận tốc quay của cần cẩu.
  - **Kiến trúc luồng xử lý:** Camera + Radar quét liên tục mặt đất dưới cẩu tháp → Nếu phát hiện người bước vào vùng nguy hiểm bên dưới cẩu tháp đang cẩu hàng $\rightarrow$ Hú còi cảnh báo mặt đất lập tức, đồng thời hiển thị cảnh báo đỏ trên màn hình buồng lái để tài xế dừng hạ tải.
  - **Đánh giá:** Phạm vi quét an toàn (bán kính 20–50m), độ trễ cảnh báo nguy hiểm (<100ms), khả năng kháng chịu thời tiết bụi bẩn ngoài trời.
  - **Độ khó:** Khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 19. Hệ thống kiểm soát chất lượng và môi trường cho bể chứa bùn
- **GVHD:** Trương Văn Cương (HK2 #45)
- **Mục tiêu:** Giám sát tự động bể xử lý bùn thải / bùn vi sinh trong nhà máy xử lý nước thải công nghiệp: Camera AI phân tích quang phổ màu sắc của bùn thành phẩm để nhận diện trạng thái bùn đạt chuẩn hay bị sốc tải vi sinh, phát hiện các vật thể lạ bất thường rơi vào bể, kết hợp mạng cảm biến siêu âm / áp suất để theo dõi mức bùn và mực nước liên tục.
- **Thành phần:**
  - **Phần cứng:** Camera công nghiệp có đèn LED trợ sáng chuẩn chống ăn mòn hóa chất, Cảm biến đo mức bùn/nước siêu âm không tiếp xúc, Cảm biến pH và Oxy hòa tan (DO), Bộ thu thập dữ liệu IoT Gateway (ESP32 / Raspberry Pi công nghiệp), Module truyền thông công nghiệp RS485 / Modbus RTU / LoRa.
  - **Phần mềm & AI:**
    - Color Segmentation & Anomaly Detection: Thuật toán phân tích không gian màu (HSV / Lab) kết hợp mô hình học máy phân loại màu bùn (bùn nâu sẫm tốt, bùn đen yếm khí hư, bùn sáng nổi bọt).
    - CNN Anomaly Detection: Phát hiện rác nổi, màng dầu hoặc vật thể lạ trên bề mặt bể.
    - Time-series Monitoring: Lưu trữ và phân tích xu hướng tăng giảm mức bùn theo thời gian, phát hiện rò rỉ hoặc nguy cơ tràn bể.
    - Dashboard Web SCADA: Hiển thị hình ảnh trực tiếp, đồ thị mức nước/mức bùn và trạng thái vi sinh của bể chứa.
  - **Đánh giá:** Độ chính xác phân loại trạng thái bùn (>90%), sai số đo mức bùn (<1cm), độ bền hoạt động liên tục trong môi trường hơi ẩm hóa chất.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

### 20. Giám sát chất lượng môi trường không khí nhà máy sử dụng công nghệ Thread/Matter kết hợp AI dự đoán cháy nổ
- **GVHD:** Trương Văn Cương (HK2 #48)
- **Mục tiêu:** Xây dựng mạng cảm biến môi trường không dây thế hệ mới trong khuôn viên nhà máy sử dụng chuẩn giao tiếp Thread/Matter (độ trễ thấp, tính liên kết lưới Mesh cao, bảo mật mạnh mẽ), tích hợp camera giám sát và mô hình AI tổng hợp dữ liệu nhiệt độ, nồng độ khí gas/khói để phân tích và cảnh báo sớm nguy cơ cháy nổ trước khi bùng phát.
- **Thành phần:**
  - **Phần cứng:**
    - Các nút cảm biến (Thread End Devices): Vi điều khiển hỗ trợ IEEE 802.15.4 / Thread (ESP32-C6 / Nordic nRF52840), cảm biến khí VOC, CO, CO2 (MQ-2, MQ-7, BME680), cảm biến nhiệt độ - khói.
    - Thread Border Router: Raspberry Pi chạy OpenThread Border Router (OTBR) làm cầu nối sang mạng IP/WiFi nhà máy.
    - Camera AI: Camera quan sát hướng nhiệt hoặc camera quang học nhận diện khói / ngọn lửa sớm.
  - **Phần mềm & AI:**
    - Giao thức: Chuẩn Matter over Thread, bảo mật mã hóa end-to-end.
    - AI Vision: Mô hình YOLOv8n-fire-smoke phát hiện đốm lửa và luồng khói bốc lên.
    - AI Time-series Sensor Fusion: Mô hình LSTM hoặc Random Forest phân tích sự gia tăng bất thường đồng thời của nhiệt độ + nồng độ CO/CO2 theo thời gian để dự đoán xác suất tích nhiệt tự cháy (spontaneous combustion) trước 5–15 phút.
  - **Đánh giá:** Độ tin cậy truyền dữ liệu qua mạng lưới Thread Mesh khi có nút bị mất nguồn, thời gian dự đoán nguy cơ cháy nổ trước khi xảy ra sự cố, tỷ lệ báo động chính xác.
  - **Độ khó:** Khó (yêu cầu nghiên cứu công nghệ mới Thread/Matter kết hợp mạng cảm biến và AI dự đoán). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

## 🔹 NHÓM 5: AIoT Y tế, Chăm sóc sức khỏe & Xử lý tín hiệu sinh học (Healthcare AIoT)

### 21. Xây dựng nền tảng tiên đoán bệnh bằng Deep Learning với dữ liệu ban đầu đa thông số sinh hiệu
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #61)
- **Mục tiêu:** Xây dựng hệ sinh thái IoT y tế tổng thể: Thu thập đồng bộ dữ liệu sinh hiệu ban đầu của bệnh nhân gồm Thân nhiệt, Nồng độ oxy trong máu SpO2, Huyết áp, Điện cơ (EMG) và Điện tâm đồ (ECG), từ đó đưa vào mô hình Deep Learning đa phương thức (Multimodal Biosignal DL) để đánh giá nguy cơ sức khỏe tổng quát và hỗ trợ bác sĩ tiên đoán bệnh lý sớm.
- **Thành phần:**
  - **Phần cứng:** Node thu thập dữ liệu bệnh nhân tích hợp vi điều khiển ESP32 / STM32 kết nối các module cảm biến: MAX30102 (SpO2, nhịp tim), MLX90614 (Thân nhiệt hồng ngoại), Cảm biến huyết áp kỹ thuật số, AD8232 (ECG), Cảm biến điện cơ EMG MyoWare; Mạch cách ly tín hiệu y tế bảo vệ an toàn cho người dùng; Màn hình cảm ứng hiển thị tại chỗ.
  - **Phần mềm & AI:**
    - Thu thập & Tiền xử lý tín hiệu: Bộ lọc số FIR/IIR lọc nhiễu 50Hz, lọc trôi đường đẳng điện (Baseline wander), chuẩn hóa Z-score.
    - Mô hình Deep Learning: Mạng kết hợp 1D-CNN (trích xuất đặc trưng hình thái sóng ECG/EMG) + Bi-LSTM / TabNet (học chuỗi thời gian các chỉ số thân nhiệt, huyết áp, SpO2) để phân loại nguy cơ bệnh lý (Bình thường / Cần theo dõi / Nguy kịch).
    - Nền tảng Cloud & Web: Backend FastAPI, cơ sở dữ liệu InfluxDB lưu trữ dữ liệu chuỗi thời gian, Web Dashboard cho bác sĩ theo dõi biểu đồ sinh hiệu theo thời gian thực (Real-time Streaming qua WebSocket).
  - **Đánh giá:** Độ chính xác phân loại bệnh (F1-score, AUC-ROC), độ trễ truyền dữ liệu từ cảm biến lên dashboard (<500ms), tính an toàn điện đối với cơ thể người.
  - **Độ khó:** Khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 22. Thiết bị đeo thông minh đo ECG / PPG kết hợp TinyML phân loại rối loạn nhịp tim
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #66, HK2 #18); TS. Phạm Hoài Luân (HK1 #17, HK2 #57)
- **Mục tiêu:** Thiết kế thiết bị đeo tay hoặc ngực nhỏ gọn, tiêu thụ năng lượng cực thấp, có khả năng thu nhận tín hiệu điện tâm đồ ECG hoặc quang thể tích PPG và chạy trực tiếp mô hình AI siêu nhẹ (TinyML) trên vi điều khiển/FPGA để phát hiện tức thì các cơn loạn nhịp tim (Arrhythmia) như nhịp nhanh thất, rung nhĩ, ngoại tâm thu.
- **Thành phần:**
  - **Phần cứng:** Module cảm biến ECG AD8232 hoặc cảm biến quang học PPG chuyên dụng, Vi điều khiển công suất thấp (ESP32-S3 / Arduino Nano 33 BLE / Raspberry Pi Pico) hoặc kit FPGA mini (Xilinx Spartan-7 / Lattice iCE40), Pin Li-Po dung lượng 500mAh có mạch sạc bảo vệ, Module BLE truyền dữ liệu đến smartphone.
  - **Phần mềm & AI:**
    - Dataset: MIT-BIH Arrhythmia Database hoặc PTB-XL Dataset.
    - AI siêu nhẹ: Mạng 1D-CNN (3–4 layers) hoặc Depthwise Separable 1D-CNN, huấn luyện trên TensorFlow/PyTorch sau đó lượng tử hóa INT8 bằng TensorFlow Lite for Microcontrollers (TFLM) hoặc Edge Impulse.
    - Kích thước mô hình: < 50KB Flash, tiêu thụ RAM < 20KB.
    - App Mobile: Ứng dụng Flutter / React Native nhận cảnh báo qua Bluetooth Low Energy khi phát hiện nhịp tim bất thường.
  - **Đánh giá:** Tỷ lệ phân loại đúng các dạng loạn nhịp tim (>95%), thời gian thực thi một nhịp tim (<50ms trên vi điều khiển), thời lượng pin hoạt động liên tục (>24 giờ).
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

### 23. Phân loại tín hiệu sóng não (EEG) nhận biết động kinh / trầm cảm và giao diện não - máy (BCI)
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #57, #58, #59, #60; HK2 #14, #15, #16, #17); TS. Phạm Hoài Luân (HK1 #19, HK2 #53)
- **Mục tiêu:** Nghiên cứu và hiện thực chuỗi thu nhận tín hiệu điện não đồ (EEG) từ điện cực đặt trên da đầu, lọc nhiễu và ứng dụng thuật toán học máy / học sâu để giải quyết một trong các bài toán y sinh: (1) Nhận diện cơn động kinh, (2) Nhận biết trạng thái trầm cảm, hoặc (3) Giao diện não - máy tính (BCI - Brain-Computer Interface) phân loại ý định cử động để điều khiển cánh tay robot / xe lăn.
- **Thành phần:**
  - **Phần cứng:** Mũ điện cực EEG hoặc module thu tín hiệu sóng não chuyên dụng (Chip ADS1299 đo điện thế cực nhỏ nV, hoặc chipset TGAT/module TGAM NeuroSky giá rẻ), Kit xử lý trung tâm (Arduino Nano 33 BLE / STM32 / ESP32 hoặc kit FPGA để xử lý phần cứng), Động cơ hoặc mô hình robot nhận lệnh điều khiển BCI.
  - **Phần mềm & AI:**
    - Xử lý tín hiệu: Bộ lọc dải Notch 50Hz, Bandpass filter 0.5–45Hz, biến đổi Wavelet (Continuous Wavelet Transform - CWT) hoặc Biến đổi Fourier nhanh (FFT) để trích xuất mật độ phổ công suất PSD các dải sóng $\delta, \theta, \alpha, \beta, \gamma$.
    - Mô hình AI: SVM / Random Forest (với tập dữ liệu nhỏ) hoặc EEGNet / 1D-CNN (mạng chuyên biệt cho tín hiệu sóng não EEG).
    - Bộ điều khiển robot: Ánh xạ trạng thái não bộ (tập trung / thư giãn / nhấp nháy mắt) thành tín hiệu điều khiển di chuyển tiến/lùi/dừng.
  - **Đánh giá:** Độ chính xác phân loại trạng thái não (>85%), tỷ lệ loại bỏ nhiễu chớp mắt (EOG artifact), độ trễ phản hồi điều khiển BCI.
  - **Độ khó:** Khó (tín hiệu EEG biên độ cực nhỏ, rất dễ nhiễu cơ và nhiễu môi trường, cần nắm vững kỹ thuật xử lý tín hiệu số y sinh). Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

### 24. Hệ thống khám chữa bệnh từ xa (Telehealth mini) và quản lý xét nghiệm kết hợp xử lý ảnh y tế
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #62, #68; HK2 #18)
- **Mục tiêu:** Xây dựng trạm Telehealth mini đặt tại trạm y tế xã phường hoặc gia đình: Bệnh nhân tự đo các chỉ số sinh tồn cơ bản gửi về hồ sơ bệnh án điện tử từ xa cho bác sĩ, kết hợp camera xử lý ảnh tự động phân tích kết quả que thử xét nghiệm y tế nhanh (que thử đường huyết, nước tiểu 10 thông số, que test nhanh).
- **Thành phần:**
  - **Phần cứng:** Gateway trung tâm Raspberry Pi có màn hình cảm ứng, các module đo sinh hiệu (Nhiệt kế, SpO2, Huyết áp), Camera chụp khay que thử xét nghiệm với hộp chống lóa sáng đồng nhất.
  - **Phần mềm & AI:**
    - Image Processing: Cắt vùng phản ứng hóa học trên que test xét nghiệm, chuẩn hóa màu sắc theo thang màu chuẩn (Color Calibration Card), trích xuất giá trị màu RGB/HSV để định lượng nồng độ glucose, protein, pH trong mẫu xét nghiệm.
    - Web / App Telehealth: Nền tảng gọi video WebRTC giữa bác sĩ và bệnh nhân, tự động hiển thị hồ sơ sinh hiệu và kết quả que test đã phân tích tự động.
  - **Đánh giá:** Sai số định lượng màu que test so với đọc thủ công bằng mắt thường (<5%), độ trễ cập nhật dữ liệu, tính trực quan dễ sử dụng cho người già.
  - **Độ khó:** Trung bình. Thời gian: ~2.5–3 tháng. Phù hợp Đồ án 1 hoặc Đồ án 2.

---

## 🔹 NHÓM 6: AIoT Nông nghiệp, Môi trường & Robotics

### 25. Trạm quan trắc môi trường thông minh kết hợp AI dự báo chuỗi thời gian
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #63); Phan Đình Duy
- **Mục tiêu:** Trạm đo đạc đa chỉ tiêu môi trường không khí / thời tiết hoạt động bằng pin năng lượng mặt trời: Thu thập nhiệt độ, độ ẩm, áp suất khí quyển, cường độ ánh sáng, lưu lượng mưa, hướng gió, bụi mịn PM2.5, ứng dụng mô hình AI học chuỗi thời gian để dự báo xu hướng thời tiết và cảnh báo bất thường môi trường cục bộ.
- **Thành phần:**
  - **Phần cứng:** ESP32 làm vi điều khiển thu thập, Cảm biến BME280 (Nhiệt độ, Độ ẩm, Áp suất), Cảm biến bụi mịn PMS7003 / SDS011, Cảm biến ánh sáng BH1750, Cảm biến đo mưa và tốc độ gió, Tấm pin mặt trời 10W + Mạch sạc pin Lithium 18650, Module truyền xa LoRa hoặc 4G LTE Cat-1.
  - **Phần mềm & AI:**
    - Giao thức truyền tin: MQTT nhẹ nhàng qua mạng di động/LoRa về MQTT Broker.
    - Backend & Cơ sở dữ liệu: Python FastAPI + Time-series Database InfluxDB.
    - Mô hình AI: Mạng LSTM / GRU hoặc mô hình Prophet / ARIMA huấn luyện trên chuỗi dữ liệu lịch sử để dự báo xu hướng nhiệt độ, nguy cơ sương giá, chỉ số ô nhiễm không khí AQI trong 6–24 giờ tiếp theo.
    - Frontend: Web Dashboard ReactJS hiển thị bản đồ trực quan các trạm và biểu đồ xu thế.
  - **Đánh giá:** Sai số dự báo nhiệt độ/độ ẩm (RMSE, MAE), mức độ tiêu thụ điện năng ở chế độ ngủ sâu (Deep-sleep current < 15$\mu$A), độ ổn định truyền nhận dữ liệu.
  - **Độ khó:** Trung bình. Thời gian: ~2.5 tháng. Phù hợp Đồ án 1 (phần cứng + hệ thống) và Đồ án 2 (mở rộng thêm AI).

---

### 26. Hệ thống tưới cây tự động thông minh vùng rộng dùng LoRa kết hợp AI thời tiết
- **GVHD:** Phan Đình Duy (HK1 #44, HK2 #31, File Duy #6)
- **Mục tiêu:** Hệ thống nông nghiệp chính xác (Precision Agriculture) cho nông trại diện tích lớn: Mạng cảm biến không dây nhiều nút (Multi-node LoRa) thu thập độ ẩm đất và điều kiện vi khí hậu tại từng luống cây, kết hợp dữ liệu dự báo thời tiết trực tuyến để đưa ra quyết định đóng mở van tưới tự động tối ưu hóa lượng nước.
- **Thành phần:**
  - **Phần cứng:**
    - Các Node cảm biến (Sensor Nodes): Vi điều khiển ESP32 / Arduino Nano + Module LoRa SX1278 (băng tần 433MHz / 433MHz), Cảm biến độ ẩm đất điện dung (Capacitive Soil Moisture Sensor chống ăn mòn).
    - Bộ chấp hành (Actuator Nodes): Node LoRa kết nối mạch relay và van điện từ 12V (Solenoid valve) điều khiển đường ống tưới.
    - Gateway trung tâm: Raspberry Pi tích hợp module LoRa và kết nối Internet (WiFi / 4G).
  - **Phần mềm & AI:**
    - Giao thức mạng LoRa: Cấu hình mạng hình sao (Star topology), mã hóa gói tin, kỹ thuật chống va chạm gói tin (Time-slot / CSMA).
    - Thuật toán quyết định tưới thông minh: Không chỉ dựa trên ngưỡng độ ẩm đất đơn thuần, hệ thống tích hợp API dự báo thời tiết (OpenWeatherMap) và thuật toán Fuzzy Logic / Q-Learning: Nếu độ ẩm đất thấp nhưng dự báo trời sẽ mưa trong vòng 2 giờ tới $\rightarrow$ Hoãn tưới để tiết kiệm nước và tránh ngập úng rễ cây.
  - **Đánh giá:** Khoảng cách truyền thông LoRa trong điều kiện có tán cây che chắn (đạt từ 500m – 2km), lượng nước tiết kiệm được so với tưới theo giờ cố định, tuổi thọ pin của các node cảm biến.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

### 27. Lập trình Drone kết hợp xử lý ảnh ứng dụng cho nông nghiệp
- **GVHD:** Nguyễn Duy Xuân Bách (HK2 #25, #24)
- **Mục tiêu:** Tự động hóa giám sát sức khỏe cây trồng trên diện rộng bằng máy bay không người lái (UAV / Drone): Lập trình bay tự động theo quỹ đạo tuần tra (Mission Planner), chụp ảnh khảo sát từ trên cao, ghép ảnh toàn cảnh và ứng dụng mô hình Deep Learning phân loại vùng cây bị sâu bệnh hoặc thiếu chất dinh dưỡng.
- **Thành phần:**
  - **Phần cứng:** Khung máy bay Drone (F450 quadcopter DIY), Mạch điều khiển bay Pixhawk / ArduPilot tích hợp GPS và la bàn số, Pin Li-Po 4S, Camera hành trình / Camera cảm biến quang phổ (Modified NDVI Camera) gắn trên chống rung Gimbal 2-3 trục, Máy tính mini nhúng onboard Raspberry Pi / Jetson Nano.
  - **Phần mềm & AI:**
    - Lập trình bay: MAVLink protocol, DroneKit-Python ra lệnh cất cánh, bay theo waypoint và hạ cánh an toàn.
    - Xử lý ảnh:
      - Ghép ảnh chỉnh trực giao: Orthomosaic image stitching (sử dụng OpenDroneMap hoặc thuật toán trích xuất điểm đặc trưng SIFT/ORB trong OpenCV).
      - Tính chỉ số thực vật: NDVI (Normalized Difference Vegetation Index) hoặc chỉ số màu xanh VARI từ ảnh RGB.
      - Phân đoạn sâu bệnh: Mạng U-Net / YOLOv8 phát hiện tán lá bị sâu bệnh cắn phá hoặc vàng lá rụng sớm.
  - **Đánh giá:** Sai số định vị tọa độ GPS của vùng sâu bệnh trên bản đồ số, độ phủ ảnh sau ghép, thời gian bay và xử lý ảnh.
  - **Độ khó:** Rất khó (đòi hỏi kỹ năng điều khiển phần cứng bay an toàn kết hợp xử lý ảnh địa không gian nâng cao). Thời gian: ~3.5–4 tháng. Phù hợp Đồ án 2.

---

### 28. Robot di động tự hành tránh chướng ngại vật sử dụng LiDAR + AI SLAM
- **GVHD:** Phan Đình Duy (HK1 #48, HK2 #35, File Duy #10); Tạ Trí Đức (HK2 #71)
- **Mục tiêu:** Thiết kế robot di động hai bánh vi sai (Differential Drive Mobile Robot) có khả năng tự động khám phá môi trường trong nhà không biết trước, dựng bản đồ 2D thời gian thực (SLAM) và tự động lập quỹ đạo tránh vật cản tĩnh lẫn vật cản động di chuyển cắt ngang mặt.
- **Thành phần:**
  - **Phần cứng:** Khung gầm xe robot 2 bánh chủ động + 1 bánh dẫn hướng, Động cơ DC có gắn encoder quang học phản hồi vận tốc, Driver động cơ L298N / TB6612, Cảm biến quét laser 2D LiDAR (RPLIDAR A1 / A2 góc quét 360 độ), Cảm biến quán tính IMU 9-DOF (MPU9250), Máy tính nhúng điều khiển cấp cao Raspberry Pi 4 hoặc Jetson Nano, Mạch vi điều khiển cấp thấp STM32/ESP32 điều khiển PID động cơ.
  - **Phần mềm & AI:**
    - Hệ điều hành robot: ROS 2 (Robot Operating System 2 - Humble / Iron).
    - Thuật toán SLAM: Cartographer hoặc Gmapping kết hợp dữ liệu Odom + IMU + LiDAR Scan (Sensor Fusion qua Extended Kalman Filter).
    - Thuật toán dẫn đường & tránh vật cản: Navigation 2 (Nav2) với Global Planner (A* / Dijkstra) và Local Planner (DWB / TEB Local Planner) tránh vật thể động.
    - AI nâng cao (tùy chọn): Mô hình Deep Reinforcement Learning (DQN / PPO) huấn luyện robot tự tìm đường trong môi trường mô phỏng Gazebo trước khi nạp xuống robot thực.
  - **Đánh giá:** Sai số dựng bản đồ SLAM so với kích thước phòng thực tế (<3%), tỷ lệ né thành công vật cản di động, độ mượt của quỹ đạo di chuyển.
  - **Độ khó:** Khó. Thời gian: ~3–3.5 tháng. Phù hợp Đồ án 2.

---

## 🔹 NHÓM 7: Hardware AI & Tăng tốc xử lý AI trên FPGA / SoC (Edge Hardware AI)

### 29. Thiết kế phần cứng tăng tốc cho mô hình Large Language Model (LLM) / Small Language Model (SLM) trên FPGA cho tính toán biên
- **GVHD:** TS. Phạm Hoài Luân (HK1 #14, #15; HK2 #52, #54)
- **Mục tiêu:** Giải quyết bài toán đưa mô hình ngôn ngữ lớn/nhỏ (SLM như TinyLlama, SmolLM-135M, Qwen-0.5B hoặc kiến trúc Attention cơ bản) xuống chạy trực tiếp trên thiết bị phần cứng biên FPGA: Thiết kế kiến trúc phần cứng chuyên dụng (Dedicated Hardware Architecture) bằng Verilog/VHDL tăng tốc các phép tính nhân ma trận (GEMM) và tính toán hàm Softmax/Attention, tối ưu hóa băng thông bộ nhớ và năng lượng tiêu thụ.
- **Thành phần:**
  - **Phần cứng:** Kit phát triển FPGA Xilinx / AMD (Xilinx Zynq-7000 ZEDBoard, PYNQ-Z2, Kria KV260 hoặc Artix-7), Băng thông bộ nhớ ngoài DDR3/DDR4.
  - **Phần mềm & Công cụ thiết kế:**
    - Ngôn ngữ mô tả phần cứng: Verilog / SystemVerilog hoặc Vitis HLS (High-Level Synthesis - C/C++ sang RTL).
    - Phần mềm EDA: Xilinx Vivado Design Suite, ModelSim / QuestaSim mô phỏng kiểm thử.
    - Framework AI & Quantization: PyTorch trích xuất trọng số $\rightarrow$ Kỹ thuật lượng tử hóa cực thấp (Quantization 1-bit, 2-bit, hoặc INT4 / INT8) để giảm kích thước model và thay phép nhân đắt đỏ bằng các cổng logic XOR/LUT đơn giản (Linear BitNet / XNOR-Net).
  - **Kiến trúc phần cứng (Hardware Architecture):**
    - Mảng xử lý tâm thu (Systolic Array PE - Processing Elements) tính tích vô hướng ma trận trọng số và ma trận kích hoạt.
    - Khối phần cứng xấp xỉ hàm phi tuyến: Khối phần cứng tính hàm $e^x$ và Softmax dùng thuật toán CORDIC hoặc bảng tra LUT (Look-Up Table).
    - Bộ nhớ đệm On-chip Scratchpad Memory (BRAM / UltraRAM) lưu trọng số và KV Cache để hạn chế tối đa truy xuất DDR ngoài gây nghẽn băng thông.
  - **Đánh giá:** Số lượng tài nguyên phần cứng sử dụng trên FPGA (LUTs, FFs, BRAMs, DSP slices), Tần số xung nhịp tối đa ($F_{max}$ MHz), Tốc độ sinh từ (Tokens per second), Công suất tiêu thụ (Watt) so với khi chạy cùng model trên GPU nhúng.
  - **Độ khó:** Rất khó (đỉnh cao chuyên ngành Thiết kế vi mạch & Hệ thống nhúng - đòi hỏi tư duy kiến trúc máy tính và HDL xuất sắc). Thời gian: ~3.5–4 tháng. Phù hợp Đồ án 2.

---

### 30. Thiết kế SoC / IP Core phần cứng tăng tốc mạng nơ-ron (CNN 1D/2D, Autoencoder, Audio AI) trên FPGA
- **GVHD:** TS. Phạm Hoài Luân (HK1 #17, #18, #19, #20; HK2 #50, #51, #53, #55, #57, #58, #59, #60); Trương Văn Cương (HK2 #50, #51); Phan Đình Duy (HK1 #49, HK2 #36)
- **Mục tiêu:** Thiết kế lõi IP Core phần cứng (Hardware Accelerator) tích hợp vào hệ thống trên chip (SoC trên nền tảng vi xử lý RISC-V hoặc ARM Cortex-A9 trên Zynq) để tăng tốc một bài toán mạng nơ-ron chuyên biệt:
  - Tăng tốc 2D-CNN cho nhận diện hình ảnh vật thể (Phan Đình Duy, Phạm Hoài Luân).
  - Tăng tốc 1D-CNN phân loại tín hiệu y sinh thời gian thực (Điện não EEG phát hiện động kinh, Tín hiệu PPG đo oxy nhịp tim) (Phạm Hoài Luân).
  - Tăng tốc Autoencoder khử nhiễu ảnh thời gian thực trên FPGA (Phạm Hoài Luân).
  - WaveNet-Lite hoặc Speech-to-Speech: Tổng hợp và xử lý âm thanh tiếng Việt thời gian thực tiêu thụ điện thấp (Phạm Hoài Luân).
  - Nhận diện phát ngôn không chuẩn mực / âm thanh bất thường trên FPGA (Trương Văn Cương).
- **Thành phần:**
  - **Phần cứng:** Kit FPGA PYNQ-Z2 / Basys 3 / DE10-Lite / Kria KV260, Giao tiếp AXI-Lite / AXI-Stream kết nối giữa vi xử lý chủ và khối tăng tốc IP Core.
  - **Phần mềm & Thiết kế RTL:**
    - Thiết kế kiến trúc luồng dữ liệu (Dataflow): Bộ đệm dòng (Line Buffer) trích xuất cửa sổ trượt $3 \times 3$ hoặc $5 \times 5$ không làm trễ xung nhịp.
    - Cấu trúc tính tích chập song song: Ghép chuỗi các khối DSP48E1 thực hiện phép MAC (Multiply-Accumulate) đồng thời.
    - Khối lượng tử hóa Fixed-point: Chuyển đổi toàn bộ trọng số Float32 về số nguyên cố định 8-bit hoặc 16-bit.
    - Phần mềm điều khiển (Driver): Viết chương trình C/C++ chạy trên RISC-V hoặc ARM khởi động bộ điều khiển DMA (Direct Memory Access) chuyển dữ liệu ảnh/âm thanh sang khối Accelerator và nhận kết quả phân loại.
  - **Đánh giá:** Throughput (GOPS - Giga Operations Per Second), Tốc độ tăng tốc (Speedup factor) so với chạy bằng phần mềm thuần túy (Software-only execution), Mức tiêu thụ tài nguyên FPGA và độ chính xác lượng tử hóa (Drop in accuracy < 1%).
  - **Độ khó:** Rất khó. Thời gian: ~3.5 tháng. Phù hợp Đồ án 2.

---

## 🔹 NHÓM 8: Nền tảng IoT, Bán lẻ thông minh & Quản trị dữ liệu

### 31. Xây dựng nền tảng IoT System / IoT Agent tích hợp AI dự đoán & bảo trì phòng ngừa (Predictive Maintenance)
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #67, #69)
- **Mục tiêu:** Xây dựng nền tảng phần mềm và phần cứng IoT hoàn chỉnh (End-to-End IoT Platform): Cho phép kết nối và quản lý hàng chục node thiết bị IoT từ xa qua các giao thức công nghiệp, tích hợp thành phần thông minh "IoT Agent" có khả năng tự động học hành vi của thiết bị, phát hiện bất thường (Anomaly Detection) và đưa ra dự đoán thời điểm máy móc có nguy cơ hỏng hóc để bảo trì phòng ngừa.
- **Thành phần:**
  - **Phần cứng:** Các node cảm biến độ rung (Gia tốc kế MPU6050 / ADXL345), cảm biến dòng điện không xâm lấn SCT-013, cảm biến nhiệt độ gắn trên động cơ quạt / máy bơm thử nghiệm; Thiết bị biên Gateway đóng vai trò IoT Agent (Raspberry Pi).
  - **Phần mềm:**
    - Thiết bị: Firmware viết bằng C++ (FreeRTOS) trên ESP32 hỗ trợ cập nhật từ xa qua mạng (OTA - Over The Air).
    - Giao thức: MQTT với bảo mật TLS/SSL, CoAP, WebSockets.
    - IoT Platform Server: Microservices bằng Python (FastAPI) hoặc Go, Message Broker EMQX / Mosquitto, Time-series DB InfluxDB kết hợp cơ sở dữ liệu quan hệ PostgreSQL.
    - AI Anomaly Detection: Thuật toán học không giám sát Isolation Forest, Autoencoder hoặc One-Class SVM chạy tại Gateway để phát hiện các tần số rung động lệch khỏi mô hình bình thường.
    - Giao diện: Dashboard Web ReactJS hiện đại theo phong cách Glassmorphism / Dark Mode, hiển thị trực quan cấu trúc liên kết mạng và tình trạng sức khỏe thiết bị.
  - **Đánh giá:** Khả năng mở rộng số lượng node kết nối đồng thời, độ trễ cảnh báo khi động cơ xuất hiện rung lắc bất thường (<1 giây), tỷ lệ phát hiện sớm sự cố hỏng hóc vòng bi/bạc đạn.
  - **Độ khó:** Trung bình–khó. Thời gian: ~3 tháng. Phù hợp Đồ án 1 (xây dựng nền tảng) mở rộng Đồ án 2 (tích hợp AI Agent).

---

### 32. Hệ thống máy bán hàng tự động (Vending Machine) thông minh tích hợp AI nhận diện sản phẩm & khuôn mặt
- **GVHD:** Nguyễn Duy Xuân Bách (HK1 #65); Phan Đình Duy
- **Mục tiêu:** Nâng cấp máy bán hàng tự động truyền thống thành mô hình bán lẻ thông minh (Smart Retail Vending Machine): Người dùng thanh toán bằng quét mã QR động hoặc nhận diện khuôn mặt liên kết ví điện tử, camera bên trong khoang tự động nhận diện sản phẩm khách đã lấy bằng AI thị giác, hệ thống cơ khí điều khiển khay nhả hàng chính xác và tự động đồng bộ tồn kho lên máy chủ đám mây.
- **Thành phần:**
  - **Phần cứng:** Khung mô hình tủ hàng mini (mica / nhôm kính), Bộ cơ cấu đẩy hàng (Động cơ bước kèm trục xoắn hoặc servo), Màn hình cảm ứng LCD hiển thị menu sản phẩm, Camera góc rộng nhận diện người dùng và camera soi khay lấy hàng, Mạch điều khiển cơ cấu nhả hàng (Arduino / ESP32) kết nối với máy tính nhúng trung tâm (Raspberry Pi).
  - **Phần mềm & AI:**
    - AI Face Recognition: Nhận diện khuôn mặt khách hàng thành viên để tích điểm hoặc xác thực thanh toán nhanh.
    - AI Product Identification: YOLOv8-nano nhận dạng loại nước ngọt / bánh kẹo khi qua cửa quét sản phẩm.
    - Hệ thống quản lý: Backend Python quản lý danh mục hàng hóa, doanh thu theo ngày, cảnh báo khi một mặt hàng sắp hết để nhân viên đi tiếp tế.
    - Cổng thanh toán: Sinh mã VietQR động theo từng đơn hàng cụ thể, tự động xác nhận giao dịch qua Webhook ngân hàng.
  - **Đánh giá:** Tỷ lệ nhả hàng chuẩn xác của cơ cấu cơ khí, độ chính xác nhận diện sản phẩm (>97%), thời gian hoàn tất một giao dịch mua hàng (<10 giây).
  - **Độ khó:** Trung bình–khó (do phải tích hợp cả cơ khí chuyển động, mạch điều khiển động cơ và phần mềm AI). Thời gian: ~3 tháng. Phù hợp Đồ án 2.

---

## 📊 Bảng so sánh tổng hợp 32 đề tài (Phân loại & Định hướng chọn đề tài)

| # | Tên đề tài | GVHD chính | Lĩnh vực chuyên môn | Độ khó | Thời gian | Phù hợp Đồ án |
|---|---|---|---|:---:|:---:|:---:|
| **1** | Nhận diện biển báo giao thông real-time (Jetson) | Phan Đình Duy | Computer Vision / Edge AI | Trung bình | 2.5–3 tháng | Cả ĐA 1 & 2 |
| **2** | So sánh & tối ưu SSD vs YOLO trên máy tính nhúng | Phan Đình Duy | Deep Learning / Optimization | Trung bình | 2.5 tháng | Cả ĐA 1 & 2 |
| **3** | Phân đoạn ảnh Mask R-CNN / Semantic Segmentation | P. Đ. Duy / N. T. Thiện | Computer Vision / Edge AI | TB–Khó | 3–3.5 tháng | Đồ án 2 |
| **4** | Phát hiện đối tượng trong ảnh y khoa trên nhúng | Nguyễn Thanh Thiện | Medical AI / Embedded | Khó | 3.5 tháng | Đồ án 2 |
| **5** | Triển khai mô hình Thị giác - Ngôn ngữ (VLM) trên nhúng | Nguyễn Thanh Thiện | Edge VLM / Multimodal | Khó | 3.5 tháng | Đồ án 2 |
| **6** | Phát hiện đối tượng đa phương thức (RGB + Thermal) | Nguyễn Thanh Thiện | Multimodal AI / Edge AI | Khó | 3.5 tháng | Đồ án 2 |
| **7** | Gương thông minh (Smart Mirror) nhận diện khuôn mặt | Phan Đình Duy | Smart Device / OpenCV | Dễ–TB | 1.5–2 tháng | Đồ án 1 |
| **8** | Hệ thống điểm danh khuôn mặt & chống giả mạo | P. Đ. Duy / N. D. X. Bách | Face Recognition / Software | Dễ–TB | 2 tháng | Đồ án 1 |
| **9** | Nhận diện cử chỉ bàn tay (Hand Gesture via Landmark) | Đỗ Trí Nhựt | HCI / MediaPipe / ML | Trung bình | 2.5 tháng | Cả ĐA 1 & 2 |
| **10** | Nhận diện dáng bộ, tư thế người & cảnh báo té ngã | Đỗ Trí Nhựt | Human Pose / Action Recog | TB–Khó | 3 tháng | Đồ án 2 |
| **11** | ADAS mini: Cảnh báo lệch làn (LDW) + Giữ làn (LKA) | Phan Đình Duy | ADAS / Autonomous Car | TB–Khó | 3 tháng | Đồ án 2 |
| **12** | Kiểm soát hành trình thích ứng (ACC) dùng Radar & AI | Phan Đình Duy | ADAS / Sensor Fusion | TB–Khó | 3 tháng | Đồ án 2 |
| **13** | Phát hiện tài xế ngủ gật & mất tập trung | Đ. T. Nhựt / P. Đ. Duy | Computer Vision / Safety | Trung bình | 2–2.5 tháng | Cả ĐA 1 & 2 |
| **14** | Giám sát xe đi sai làn đường & Nhận diện biển số (ANPR) | Phan Đình Duy | ITS / Traffic AI | Khó | 3.5 tháng | Đồ án 2 |
| **15** | Giám sát xe trạm cân tự động (OCR, ANPR, chống gian lận) | Trương Văn Cương | Industrial Vision / OCR | Khó | 3.5 tháng | Đồ án 2 |
| **16** | Đánh giá hiệu suất & giám sát hành vi công nhân | Trương Văn Cương | Behavioral Analysis / ReID | Khó | 3.5 tháng | Đồ án 2 |
| **17** | Phát hiện bất thường quy trình order quầy chuỗi cafe | Trương Văn Cương | Hand-Object Action AI | Khó | 3.5 tháng | Đồ án 2 |
| **18** | Giải pháp an toàn cẩu tháp công trường (Radar + Camera) | Trương Văn Cương | Construction Safety / AIoT | Khó | 3.5 tháng | Đồ án 2 |
| **19** | Giám sát chất lượng & môi trường cho bể chứa bùn | Trương Văn Cương | Environmental AIoT | TB–Khó | 3 tháng | Đồ án 2 |
| **20** | Giám sát môi trường nhà máy qua Thread/Matter & AI cháy nổ | Trương Văn Cương | Next-Gen IoT / Thread Mesh | Khó | 3.5 tháng | Đồ án 2 |
| **21** | Nền tảng tiên đoán bệnh Deep Learning đa thông số | Nguyễn Duy Xuân Bách | Medical AIoT / Deep Learning | Khó | 3.5 tháng | Đồ án 2 |
| **22** | Thiết bị đeo ECG/PPG kết hợp TinyML loạn nhịp tim | N. D. X. Bách / P. H. Luân | Wearable / TinyML | TB–Khó | 3 tháng | Đồ án 2 |
| **23** | Phân loại sóng não EEG (động kinh/trầm cảm/BCI robot) | Nguyễn Duy Xuân Bách | Biosignal / BCI / AI | Khó | 3.5 tháng | Đồ án 2 |
| **24** | Trạm Telehealth mini & phân tích ảnh que xét nghiệm | Nguyễn Duy Xuân Bách | Telehealth / Medical Image | Trung bình | 2.5–3 tháng | Cả ĐA 1 & 2 |
| **25** | Trạm quan trắc môi trường thông minh & AI chuỗi thời gian | N. D. X. Bách / P. Đ. Duy | Environmental IoT / LSTM | Trung bình | 2.5 tháng | Cả ĐA 1 & 2 |
| **26** | Hệ thống quản lý tưới cây diện rộng LoRa & AI thời tiết | Phan Đình Duy | Smart Agriculture / LoRa | TB–Khó | 3 tháng | Đồ án 2 |
| **27** | Drone nông nghiệp tự hành kết hợp xử lý ảnh cây trồng | Nguyễn Duy Xuân Bách | UAV / Robotics / CV | Rất khó | 3.5–4 tháng | Đồ án 2 |
| **28** | Robot di động tự hành né vật cản dùng LiDAR & AI SLAM | P. Đ. Duy / T. T. Đức | Robotics / ROS 2 / SLAM | Khó | 3–3.5 tháng | Đồ án 2 |
| **29** | Thiết kế phần cứng tăng tốc SLM / LLM trên FPGA | TS. Phạm Hoài Luân | Hardware AI / Chip Design | Rất khó | 3.5–4 tháng | Đồ án 2 |
| **30** | Thiết kế SoC / IP Core tăng tốc CNN, Autoencoder trên FPGA | P. H. Luân / T. V. Cương | FPGA RTL / SoC Design | Rất khó | 3.5 tháng | Đồ án 2 |
| **31** | Nền tảng IoT Platform / IoT Agent & bảo trì phòng ngừa | Nguyễn Duy Xuân Bách | Full-stack IoT / Cloud AI | TB–Khó | 3 tháng | Cả ĐA 1 & 2 |
| **32** | Máy bán hàng tự động thông minh AI nhận diện hàng & mặt | N. D. X. Bách / P. Đ. Duy | Smart Retail / Embedded | TB–Khó | 3 tháng | Đồ án 2 |

---

## 🎯 Cẩm nang định hướng chọn đề tài cho sinh viên KTMT

1. **Nếu bạn muốn làm chắc tay, an toàn, hoàn thành đúng tiến độ trong Đồ án 1:**
   - Chọn các đề tài thiên về ứng dụng phần mềm kết hợp phần cứng quen thuộc: **#7 (Smart Mirror)**, **#8 (Điểm danh khuôn mặt)**, **#9 (Cử chỉ tay MediaPipe)**, **#13 (Tài xế ngủ gật)**, **#25 (Trạm quan trắc môi trường cơ bản)**.
2. **Nếu định hướng phát triển chuyên sâu về AI & Thị giác máy tính (AI Engineer / Computer Vision Engineer):**
   - Chọn các đề tài có tính thời sự cao, mô hình hiện đại: **#1 (Biển báo Jetson TensorRT)**, **#3 (Semantic Segmentation Kria KV260)**, **#4 (AI Y khoa trên nhúng)**, **#5 (Edge VLM - Đa phương thức)**, **#10 (Pose Estimation)**, **#14 (Camera phạt nguội ANPR)**.
3. **Nếu định hướng làm IoT, Hệ thống nhúng công nghiệp & Nhà máy thông minh (Firmware / Embedded / IoT Engineer):**
   - Chọn các đề tài tích hợp hệ thống đa thiết bị, chuẩn công nghiệp: **#15 (Trạm cân OCR/ANPR)**, **#18 (An toàn cẩu tháp)**, **#20 (Mạng Thread/Matter dự đoán cháy nổ)**, **#26 (Tưới cây mạng LoRa)**, **#31 (IoT Platform & Agent)**.
4. **Nếu định hướng Y sinh & Xử lý tín hiệu thông minh (Biomedical Engineering / Biosignal AI):**
   - Chọn các đề tài: **#21 (Deep Learning đa dữ liệu sinh hiệu)**, **#22 (Thiết bị đeo TinyML ECG/PPG)**, **#23 (Sóng não EEG & BCI)**.
5. **Nếu đam mê Robotics & Xe tự hành (Robotics Engineer / Autonomous Systems):**
   - Chọn các đề tài: **#11 (ADAS LDW/LKA)**, **#12 (Adaptive Cruise Control ACC)**, **#27 (Drone nông nghiệp)**, **#28 (Robot di động ROS 2 LiDAR SLAM)**.
6. **Nếu định hướng Thiết kế vi mạch & Bán dẫn (IC Design / Hardware Engineer / FPGA):**
   - Chọn các đề tài tăng tốc phần cứng của TS. Phạm Hoài Luân hoặc Thầy Trương Văn Cương: **#29 (Tăng tốc SLM/LLM trên FPGA)**, **#30 (IP Core 1D/2D CNN, Autoencoder, Audio AI trên FPGA/SoC)**. Đây là những đề tài có giá trị portfolio rất cao khi ứng tuyển vào các tập đoàn vi mạch hàng đầu.
