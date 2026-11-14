# Handoff — Project 13: ESP32-S3 driver monitoring

Cập nhật: **13/11/2026**. Board: ESP32-S3 **N16R8** (16 MB flash, 8 MB PSRAM), camera OV5640 QVGA.

## 0. TRẠNG THÁI MỚI NHẤT (14/11 sáng) — đọc mục này trước

**Board đang chạy:** `blaze_accept_app3.bin` (**BẢN NGHIỆM THU ĐẠT**, mục 12 báo cáo) = PIPNet-98 +
camera + BlazeFace INT8 calib ĐA NGƯỜI + FSM ADAS sửa cho 0,12 FPS + **ngưỡng EAR tương đối
0,75×baseline** (sàn 0,085, luật tụt 25 %, cửa sổ 2/3 mẫu) + fusion có cổng hợp lý,
SHA `866DFD9F59200F672D75CBB2A5867A18D39BFC6E4E4BCFCA517FA2BFD12595D2`, console 115200, capture OFF,
0,12 FPS, không WiFi/web. Test flags trong `main.cpp` đều = 0.

**NGHIỆM THU 20:37 — ĐẠT (phương án a):** 25/25 detect (score trung vị 0,930, 541 ms); hiệu chuẩn
baseline EAR 0,20 → ngưỡng 0,15; nhắm 0,124 và 0,101 → **MICROSLEEP ALARM lúc 20:38:46**; mẫu mở
0,171–0,199 không báo nhầm; quay đầu → DISTRACTION; 0 panic/WDT.
Hình: `output/blaze_iq_20261113/acceptance_timeline.png`, `multiid_sheet.png`.
**Giới hạn phải công bố:** cảnh báo mắt cần 2/3 mẫu (~16–24 s), muốn chắc thì nhắm **≥40 s**; cự ly
làm việc box 69–127 px (55–75 cm); MAR/ngáp chưa hiệu chỉnh xong; không dashboard/còi.

**Miền ảnh OV5640 + benchmark đa tài xế (mục 9f báo cáo):** đã đo đặc trưng 40 khung thật
(mean 120,4; std 64,1; nhiễu 5,93; nét p95 128,2) → `tools/ov5640_domain.py` chuyển ảnh dataset về
đúng miền. **ROI k=1,20 (offline tốt hơn) đã bị A/B trên board bác bỏ** (EAR mở thấp nhất 0,145 → 0,131)
⇒ giữ **k=1,08**, muốn nới phải làm theo cỡ mặt. Trong miền OV-like, tách mở/nhắm yếu đi
(nhắm 0,116 → 0,139) ⇒ cần **luật tương đối** (tụt ≥25 % so với baseline), không chỉ ngưỡng tuyệt đối.

**BẰNG CHỨNG ĐA TÀI XẾ TRÊN BOARD (14/11, mục 9e báo cáo) — ĐẠT:** nhúng 8 tài xế YawDD
(4 người **đeo kính**) + 4 khung **mắt nhắm** CEW của 4 người khác, chạy nguyên chuỗi trên ESP32
(`MULTI_ID_TEST 1`): **12/12 phát hiện** (score 0,704–0,970; box 69–124 px),
**EAR mở 0,145–0,193** (trung vị 0,167) và **EAR nhắm 0,092–0,150** (trung vị 0,116)
⇒ tách được mở/nhắm cho **người khác, kể cả đeo kính**. Log `serial_multiid2.log`,
ảnh `output/multiid_stills_20261114/`, công cụ `tools/make_multiid_header.py`,
ảnh nạp `blaze_multiid_app2.bin` (SHA `3343821A…DB2C4F0`).
Chồng lấn duy nhất: 1 khung nhắm 0,150 = khung mở thấp nhất 0,145 ⇒ cần luật "tụt tương đối ≥25 %".

**Trả lời câu hỏi "có chạy được nhiều tài xế không" (đo trên dữ liệu nhiều người — mục 9d báo cáo):**
- **EAR: CÓ.** 100 người mở (trung vị 0,172) vs 100 người nhắm (0,116) — tách 33 %, 96 % ảnh nhắm dưới 0,15.
- **Detector: MỘT PHẦN.** 100 danh tính khác (mặt ~98 px): model calib-1-người chỉ đạt **56 %** ở ngưỡng 0,60;
  model calib-đa-người (đang chạy) đạt **63 %** ở 0,60 và **90 %** ở 0,50 (trước: 56 %/81 %). Không hại
  người cũ: 40 khung thật, trung vị 0,855 → 0,849, **p10 0,500 → 0,704**.
- **Còn 37 % danh tính không phát hiện** (score 0) — phải kiểm bằng **ảnh thật người khác**, chưa kết luận.
- **Đã sửa theo bằng chứng:** sàn ngưỡng EAR **0,15 → 0,10** (14 % ảnh mắt mở của người khác bị báo nhắm giả
  vì sàn cứng chặn hiệu chuẩn theo từng người). Test FSM vẫn **ALL PASS 14/14**.
- **Còn chặn:** ROI k=1,08 cố định (mặt 145 px → landmark văng, EAR 0,06–2,74) và **chưa có bằng chứng
  live trên người thứ 2**.

**Model mới:** `output/port_models_20261110/blaze_iq_mp/blazeface_nchw_full_integer_quant.tflite`
187.416 B, SHA `9594B46F455E5D015A15F45EEBBCC4E0F97D583AA6A2751934748CF161DD33CA`;
calib **319 mẫu / 93 danh tính** (`output/port_models_20261110/blaze_calib_mp/`,
`tools/blaze_calib_multiperson.py`); parity int8-vs-float **PASS** (`blaze_iq_mp/parity_mp.json`).
Header C đã sinh lại: `firmware_esp32/main/blaze_iq_model_data.h`.

**Ảnh nạp:**
| Bản | File | SHA-256 |
|---|---|---|
| **Đang trên board** (model đa người + sàn EAR 0,10) | `blaze_live_app13.bin` | `A8D79837…8486C4C` |
| Model đa người, sàn 0,15 | `blaze_live_app12.bin` | `24796972…715A4E7` |
| Model calib-1-người (dùng cho mọi số trước 14/11) | `blaze_live_app11.bin` | `76057B63…456E09` |
| Bản dump ảnh (921600) | `blaze_capture_app.bin` | `9DA9331D…E6E129C` |
| Hoàn nguyên PIPNet-only | `restore_pipnet_app.bin` | `28330A9E…028A77FD` |

**Công cụ mới cho chu kỳ dữ liệu (dùng lại được mỗi lần đổi model):** 	ools/ov5640_domain.py
(đo/chuyển miền), 	ools/driver_benchmark.py (benchmark YawDD+CEW trong miền OV5640),
	ools/roi_scale_sweep.py (quét k), 	ools/make_multiid_header.py (12 người nhúng vào firmware),
	ools/multiperson_eval.py (100 danh tính 300W_LP + 200 người CEW).

**MỞ RỘNG 32 ẢNH (mục 14):** 3 lô nhúng 12/10/10 (30 người khác nhau + 2 khung nhắm CEW) —
**L2 trung bình 0,39 px, trung vị 0,24 px, p95 0,90 px** trên 3.136 điểm; box IoU 0,936; score lệch
0,000; ΔEAR 0,0036. Xấu nhất 1,06 px (ảnh CEW cắt sát, lệch ở viền mặt). Hình
output/board_vs_laptop_20261114/overlay_sheet_32.png; bảng ll_batches.csv/json.
Công cụ: oard_vs_laptop_stills.py, merge_board_vs_laptop.py, overlay_sheet_batches.py,
dump_multiid_header.py.

**ESP32 vs LAPTOP trên cùng 12 ảnh (mục 13 báo cáo) — model trên board chính xác:** detector score lệch
trung vị **0,000**, box IoU **0,958** (min 0,905); 98 điểm lệch **0,37 px trung vị** (0,42 trung bình,
p95 1,02) trên mặt 69-127 px; **EAR lệch 0,0032** (max 0,011). Sai số còn lại là của model/QVGA, không
phải INT8. Script 	ools/board_vs_laptop_stills.py, 	ools/plot_board_vs_laptop.py;
dữ liệu output/board_vs_laptop_20261114/, hình overlay_04_11.png.
Model đang chạy trên board: **BlazeFace short-range INT8 (TFLM)** + **PIPNet-98 INT8 (ESP-DL, 7 phần)**;
MSR/MNP tắt, det_10g/2d106det/TinyDriver không dùng.

**TRẠNG THÁI: ĐÓNG PROJECT THEO PHƯƠNG ÁN (a)** — hệ kiểm tra định kỳ 0,12 FPS đã nghiệm thu ĐẠT
(mục 12 báo cáo): nhắm mắt → MICROSLEEP ALARM, quay đầu → DISTRACTION, 12/12 người khác chạy trên
thiết bị, 0 panic/WDT. Tài liệu bàn giao: docs/PROJECT13_FINAL_STATUS_20261113.md (định nghĩa hoàn
thành, 8 lỗi đã sửa, giới hạn phải nêu, việc ngoài phạm vi) + docs/DEMO_CHECKLIST_PIPNET_20261113.md.
Các việc KHÔNG chặn bàn giao: real-time ≥2 FPS (cần train hoặc P4), dashboard/web+còi, MAR/ngáp,
ROI theo cỡ mặt khi dùng ở 35-40 cm, người thứ 2 chạy live.

**TỐI ƯU TỐC ĐỘ (13/11 đêm — mục 16 báo cáo): KHÔNG có hướng nào an toàn cho thêm tốc độ.**
- Flash 120 MHz + PSRAM 80 → IDF **không build được** (assertion "FLASH and PSRAM Mode configuration are
  not supported"); PSRAM 120 MHz là **thử nghiệm** (IDF cảnh báo crash khi nhiệt độ đổi ~20 °C) → bỏ.
- Bỏ decode JPEG (OV5640 Y-plane): đo được decode thật chỉ **≈50 ms** (Dec 602→557), 550 ms còn lại là
  **chờ camera (AEC ban đêm)**; ảnh raw-Y **tối 2,7×** → **bỏ**, giữ JPEG (`TD_CAM_RAW_GRAY 0`).
- Detector **đã** chạy song song Core 0 (`TD_LITE_ON_CORE0=y`) → không còn gì để lấy.
- PIPNet 192: chạy được nhưng L2 **12,5 px**, ΔEAR **0,034** → **bỏ** (mất chính xác).
- Compiler -O2 PERF, CPU 240 MHz, cache max → đã tối ưu. Ban đêm 8,2 s là do AEC phơi sáng; **ban ngày
  ≈7,7 s (0,13 FPS)**. Muốn nhanh hơn chỉ còn: train/prune (trái nguyên tắc) hoặc đổi board P4.
- Hồi quy sau thử nghiệm: 12 ảnh nhúng **trùng khít baseline**, FSM replay ALL PASS 14/14.
- **Profiling trên board** (log `[LM split]`): preprocess **10,4 ms** | `model.run()` **7.477 ms** |
  decode **39,2 ms** (lần đầu 3.800 ms cold) ⇒ **99,3 % là model**, không có overhead code để lấy.
- Bản production cuối: `output/blaze_iq_20261113/blaze_prod_final_app.bin`
  SHA `C405729A153DABE76071513C1042249D7C7D36CF3FED921F4E9E15A8AE8B9043` (đang trên board,
  0 panic/WDT, có log `[LM split]` để A/B nhanh sau này).

**A/B TÁCH DETECTOR (14/11 sáng — mục 17 báo cáo): THẤT BẠI, đã rollback.**
- Ý tưởng: detector (541 ms) đang trên đường găng (nằm trong `Dec`); cho chạy Core 0 trên khung trước.
- Bản B (ROI khung trước vô điều kiện): **7850 ms (−5,0 %)** nhưng khi quay/cúi thì ROI trỏ vào chỗ cũ →
  miss 3, **EAR 0,934**, box phình 103–112 px ⇒ **loại**.
- Bản C (lai, chỉ reuse khi Δyaw/Δpitch < 8°): reuse chỉ **32 %**, **8158 ms (−1,2 %)**, vẫn còn 1 khung
  **EAR 1,55** ⇒ **loại**.
- Bẫy: lượt 3 tưởng là bản lai nhưng chạy nhầm **bản B** do **CMake cache** giữ config cũ — kiểm bằng
  dòng `D16: task DECODE+…Core 0` trong log khởi động; muốn chắc thì `Remove-Item -Recurse firmware_esp32\build`.
- Code hybrid giữ lại nhưng **mặc định TẮT** (`TD_ROI_REUSE_STABLE 0`).
- Board đã về `blaze_prod_final_app.bin` SHA `C405729A15…8B9043`, `P4 DIAG` ON, 0 panic/WDT.

**TRẠNG THÁI: ĐÓNG PROJECT (14/11) — báo cáo cuối: `docs/PROJECT13_FINAL_REPORT_20261114.md`.**
Chủ dự án đã quyết định **dừng tối ưu tốc độ**: mọi hướng đã thử và bị loại bằng số (mục 16–17),
8,26 s/khung là sàn của cấu hình hiện tại. Model trên board: **BlazeFace short-range INT8 TFLite**
(187.416 B, SHA `9594B46F…1DD33CA`, TFLM, rodata) + **PIPNet-98 ResNet18 INT8 ESP-DL**
(12.388.576 B, SHA `D402A7D1…633909`, partition `pipnet` @0x300000). Board đang chạy
`blaze_prod_final_app.bin` SHA `C405729A15…8B9043`.

**XEM BOARD QUA WIFI (14/11 — bản `blaze_web_app2.bin`, SHA `564DABF5DFB746D26DE5DAF18BB4D752A02F9338123B15E59E375EB5315B72CE`):**
module mới `firmware_esp32/main/obs_server.cpp` (Wi-Fi STA + esp_http_server), bật bằng `TD_OBS_ENABLE 1`.
- `http://192.168.1.5/` — **trang cảnh báo**: banner TO đổi màu theo `state` (xám hiệu chuẩn / xanh NORMAL /
  vàng YAWN / đỏ MICROSLEEP / tím DISTRACTION), **nhấp nháy + đổi tiêu đề tab + bíp** (phải bấm nút
  🔔 một lần vì chính sách autoplay), ảnh ROI+98 điểm, thanh EAR/MAR so ngưỡng, 6 cảnh báo gần nhất.
- `/status` JSON; `/snapshot.bmp` BMP 77.878 B.
- **Không đổi đường suy luận**: Total vẫn **8.229 ms** (AI 7.624, Dec 601) như bản không WiFi.
- Mật khẩu WiFi: `firmware_esp32/main/wifi_secret.h` — **gitignored**; đã rà soát `git grep` để repo
  không còn chuỗi mật khẩu (Kconfig default, wifi_stream_client fallback, sdkconfig*).
- Bản không WiFi (rollback): `blaze_prod_final_app.bin` SHA `C405729A15…8B9043`.

**MAR ĐÃ SỬA (14/11 — mục 18 báo cáo, bản `blaze_web_app3.bin` SHA `B6DB5615…DFC4765F0`):** người dùng
phát hiện `mar 2,4–2,7` khi **đóng miệng**. Nguyên nhân: map 6 điểm miệng trong `ai_inference.cpp` sai
(`{76,87,80,84,89,92}` — lấy môi dưới làm khoé, "rộng miệng" chỉ 5,2 px). Đã đổi thành
`{76,82,80,85,90,94}` → MAR miệng đóng **0,28–0,32**, ngưỡng ngáp 0,48–0,51; FSM replay ALL PASS.
**Còn phải kiểm:** ngáp thật phải đẩy MAR > 0,5 (nếu không thì hạ sàn ngưỡng hoặc đổi công thức).

**HAI LỖI HIỂN THỊ ĐÃ SỬA (14/11 — mục 18b, bản `blaze_web_app4.bin` SHA `02972E91…EF86BED8`):**
- Thanh EAR: thang cũ = ngưỡng×1,6 nên mắt mở 0,18 luôn ~85 %; đổi **thang đầy 0,40 + vạch trắng
  ngưỡng** (mắt mở ⇒ ~45 %, đúng nghĩa). Thanh MAR thang 1,20 + vạch ngưỡng ngáp.
- yaw: giá trị lúc chưa hiệu chuẩn là **raw** (PnP lệch hệ thống **43–46°** khi nhìn thẳng); sau hiệu
  chuẩn đã trừ bias ⇒ **−4,4…−5,3°**. `/status` thêm `yaw_raw/yaw_bias/pitch_raw/pitch_bias/calibrated/
  ear_baseline`; trang hiện "yaw X° (raw Y°, bias Z°)".
- Thêm nút **↻ Hiệu chuẩn lại** → `POST /recalib` → `adas_controller_request_recalibration()` (đã kiểm).

**Việc đầu tiên phiên sau:**
1. **Đo 3 người thật** (chỉ cách này mới chứng minh đa tài xế): mỗi người 1 vòng 2 phút
   (`tools/live_session_cue.py --rounds 1`), chấm `tools/live_session_analyze.py --latency 7.7`.
   Cổng: score ≥ 0,90, EAR nhắm < ngưỡng, không báo nhắm giả khi quay đầu.
2. **ROI theo cỡ mặt** (box > 120 px → k = max(1,08; 1,35)) — dùng 40 khung trong
   `output/calib_frames_20261113/` để chọn k (`tools/roi_scale_sweep.py`).
3. **Kiểm 37 % danh tính không phát hiện** bằng ảnh thật người khác (không dùng ảnh dán nền tổng hợp).
4. **Quyết định tốc độ** 0,12 FPS (a/b/c) — mục 9.3 báo cáo đã lượng hoá.

**Bẫy build đã gặp (đừng lặp lại):** CMake **cache `SDKCONFIG`**; nếu trước đó build capture với
`-D SDKCONFIG=…\sdkconfig.capture` (921600) thì các lần build sau vẫn dùng bản 921600 → log ra mojibake.
Luôn build với `-D SDKCONFIG=<...>\firmware_esp32\sdkconfig` khi quay lại bản thường.

**Môi trường:** IDF 5.3.5 `C:\Espressif\frameworks\esp-idf-v5.3.5`, python `idf5.3_py3.11`, COM3, app @0x10000.
Test FSM: `g++ -std=c++17 -O2 -I firmware_esp32/main tools\adas_replay_test.cpp
firmware_esp32/main/adas_controller.cpp -o output\adas_replay_20261113\adas_replay_test.exe` → ALL PASS.
**Báo cáo:** `docs/BLAZE_IQ_RESULT_20261113.md` (8 Block B, 9 FSM, 9b arena/fusion, 9c buổi thu ảnh,
**9d đa tài xế**). Mở COM3 làm board reset 1 lần (DTR/RTS) — không phải lỗi firmware.

## 1. Trạng thái hiện hành (đọc trước khi làm)

| Hạng mục | Trạng thái |
|---|---|
| Firmware đang chạy | **PIPNet-98 + BlazeFace INT8 (TFLM)**, camera-only, **0,12 FPS**, không WiFi/web; bản v3/Lite vẫn giữ để hoàn nguyên |
| Ngưỡng EAR | 0,160 (đã sửa: baseline = phân vị 75% lúc hiệu chuẩn, sàn 0,15) |
| Bài học đã sửa | “Nhắm mắt không báo” do **ngưỡng 0,100 quá thấp** (EAR nhắm 0,126) — không phải model hỏng |
| `2d106det` (InsightFace, 106 điểm) | Đã quantize `.espdl` **1,2 MB**, **nạp được trên board** (input 192×192 int8) |
| `det_10g` (InsightFace detector) | `.espdl` 4,1 MB, **không nạp nổi** (thiếu RAM, không phải flash) |
| `human_face_detect` (MSR/MNP) | Component ESP-DL **đã thêm**, reconfigure OK (dự kiến 32,5 + 5,9 ms) |
| TinyDriver-22 | **Bị bác** (điểm không bám bộ phận mặt). Sẽ xoá khi pipeline mới chạy |

- **Build 11/11 (chiều):** đã pin esp-dl ~3.3.14 + human_face_detect ^0.5.2 (bản 0.2.0 lệch API),
  xoá cache component và reconfigure OK. Build hiện dừng ở các file **legacy TFLM**
  (i_inference.cpp, ace_detector.cpp, ace_detector_lite.cpp) vì include
  	ensorflow/lite/schema/schema_generated.h không còn được export.
  ⇒ Quyết định: **xoá luôn TinyDriver/legacy TFLM** (đã bị bác), build lại chỉ với ESP-DL.- **Pipeline ESP-DL 11/11 (chiều muộn):** đã **build thành công** app chỉ dùng ESP-DL (bỏ TFLM):
  i_inference mới chạy 2d106det (192), detector Lite/BlazeFace thay bằng **MSR/MNP** (RGB888),
  map 106→22 điểm, **chạy trên board 2,5 FPS**. Vấn đề còn lại: tensor landmark là **INT8**
  (in exp=0, out exp=-7) nên cách đổ input thủ công cho ra **điểm hằng (160,134)**; cần dùng
  dl::image::ImagePreprocessor hoặc dequant đúng (q-zp)*2^exp để có điểm thật.
  Board đã **hoàn nguyên v3** (4,3 FPS) để giữ bản chạy được.- **Nguyên nhân gốc ESP-DL (11/11):** .espdl tạo từ ONNX **NCHW** trong khi ESP-DL/ImagePreprocessor
  yêu cầu **NHWC** (nó đọc shape[1]=H, shape[2]=W, shape[3]=C). Model ta có shape [1,3,192,192] nên
  constructor assert/abort. Cần xuất lại .espdl với layout NHWC (ESP-PPQ tuỳ chọn input layout hoặc
  chuyển ONNX sang NHWC trước khi lượng tử), rồi truyền ảnh RGB888 cho preprocess.
- Board đang chạy **v3** (4,3 FPS) — an toàn.- **Thử NHWC 11/11:** tạo wrapper ONNX 2d106det_nhwc.onnx (transpose đầu vào, input [1,192,192,3])
  — ONNX chạy đúng, nhưng **ESP-PPQ không lượng tử được** (runtime error tại conv đầu tiên vì
  Transpose làm lệch layout nội bộ). Hướng sửa: dùng espdl_quantize_onnx với ONNX **gốc NCHW** rồi
  để ESP-DL tự xử lý layout, hoặc tìm tham số đổi layout trong ESP-PPQ (chưa có trong chữ ký API).
  Board vẫn chạy **v3** (an toàn).- **ĐÃ CHẠY ĐƯỢC pipeline ESP-DL mới (11/11 tối):** model 2d106det_nhwc_norm.espdl (input NHWC INT8
  exp=-7, output INT8 exp=-7) nạp vào partition lm106; firmware điền input **NHWC q=pixel-128**,
  đọc output int8 × 2^-7. Board báo **EA R 0,27 / MAR 0,50**, **2,6 FPS**, landmark thật (không còn hằng).
  Chìa khoá: wrapper ONNX NHWC + normalize ngoài + sửa rename input khi wrap.
- Board hiện chạy bản ESP-DL mới (không còn TinyDriver trong đường suy luận).- **So sánh model trên CÙNG khung (11/11 tối):** ghi 22 s (mắt mở → nhắm → mở) rồi chạy cả hai model:
  **PIPNet-98**: EAR 0,32 (mở) → **0,106** (nhắm) → 0,32 (mở lại) — tách rõ.
  **2d106det**: EAR gần như không đổi (0,27–0,28) ở mọi tư thế ⇒ **2d106det không đủ để phát hiện
  mí ở QVGA + bbox MSR/MNP**.
  ⇒ Kết luận: dùng **PIPNet-98** cho mắt; cần port lên ESP-DL (nặng ~1,8 GFLOPs, dự kiến 1–2 FPS),
  kèm decode heatmap trong C++.- **So sánh 3 bên trên CÙNG 38 khung (11/11 tối):** MediaPipe (tham chiếu laptop) và **PIPNet-98**
  cùng theo dõi mí: ví dụ lúc nhắm (t≈8,4 s) MP=0,136, PIPNet=0,136; lúc mở MP≈0,22–0,28,
  PIPNet≈0,28–0,38. **2d106det đứng yên 0,435–0,446 ở mọi khung** ⇒ loại 2d106det cho mắt.
  File: 	ools/compare_three_models.py, output/port_models_20261110/live_check/eye_map2/.
- **PIPNet-98 ONNX xuất được:** output/port_models_20261110/pipnet98_resnet18.onnx (49,3 MB float,
  5 đầu ra heatmap). Thách thức khi port: INT8 ~12 MB (flash 16 MB rất sát), decode heatmap phải
  viết bằng C++, FPS dự kiến 1–2.- **PIPNet-98 lượng tử xong (11/11):** output/port_models_20261110/espdl/pipnet98.espdl = **11,8 MB**
  (INT8), input **NHWC [1,256,256,3] exp=-7** ⇒ firmware điền q = pixel-128 như 2d106det.
  Bước tiếp: repartition (app ~3 MB + model 12 MB), load từ partition, **decode 5 head**
  (cls argmax 8×8 + offset x/y + nb), nối EAR/MAR, đo FPS (~1–2), rồi nghiệm thu.- **PIPNet trên board (11/11 đêm):** model 11,8 MB nạp từ partition 12 MB; firmware build OK (app 2,54 MB)
  nhưng **panic/ngừng frame khi nạp hoặc chạy** (RAM cho activation ResNet18@256 vượt khả dụng). Đã thử
  param_copy=false + cap internal 512 KB — vẫn lỗi. Board đã hoàn nguyên **v3**.
  Hướng tiếp: xuất PIPNet input **128** (activation giảm 4×, params vẫn 11,4 MB), hoặc tìm weights
  MobileNetV2 (~3,5 MB) — chưa có công khai.- **Skill project (11/11):** đã cài trong .dsh/skills/: espdl-model-port, drowsiness-acceptance
  (DSH nhận trong catalog).
- **PIPNet theo kích thước (11/11):** @128 decode rác (EAR 7–25, model vốn train 256); @192 EAR
  0,17–0,25, không tụt rõ khi nhắm; **@256 đúng nhưng không nạp được** (activation ResNet18 vượt RAM,
  kể cả param_copy=false + cap 64 KB). Board đã về v3.
  ⇒ Muốn dùng PIPNet cần **giảm RAM** (bỏ stream/1 framebuffer, chấp nhận <1 FPS) hoặc tìm
  **weights MobileNetV2** (nhẹ hơn ~3×, chưa có nguồn công khai).- **Kết luận PIPNet trên ESP32-S3 (11/11 đêm):** B (weights MobileNetV2) **không có nguồn công khai**
  (HF assale02/PIPNet chỉ dataset; torchlm 0.1.6.10 chỉ ResNet). Chuyển A:
  - param_copy=true: crash OOM (copy 12 MB params).
  - param_copy=false: **nạp được**, nhưng **0,1–0,2 FPS (5–10 s/khung)** và output rác.
  ⇒ **PIPNet không khả thi trên S3.** Board đã hoàn nguyên v3.- **Lựa chọn 1 (eye-state classifier nhỏ) — KẾT QUẢ ÂM TÍNH (12/11 sáng):** gom 461 khung từ các
  phiên cũ (seg01–04, eye_map, eye_map2), nhãn MediaPipe EAR (mở ≥0,21, nhắm ≤0,16).
  - Strip 64×64 (381 mẫu, test 22): **AUC 0,55**.
  - Eye-patch 48×48 quanh tâm mắt MP (762 mẫu, test 44): **AUC 0,61**.
  ⇒ Ở QVGA (mắt ~20–25 px) classifier nhỏ **không tách được** mở/nhắm; đúng giới hạn đã gặp.
  Kết luận phần cứng: S3 không đủ cho model ResNet-class (PIPNet) mà chỉ model nhỏ thì thiếu tín hiệu.- **Đo PIPNet nguyên bản trên board (12/11):** AI = **7.618 ms/khung** ⇒ **0,13 FPS** (params 11,8 MB đọc từ
  flash mỗi lượt, cache 32 KB thrash; băng thông hiệu dụng ~1,5 MB/s). Trần lý thuyết nếu streaming tốt
  (~40 MB/s) + compute 1,06 s ≈ **1,4 s ⇒ ~0,7 FPS**. Kết luận: PIPNet nguyên bản **không khả thi realtime**
  trên S3; muốn ≥2 FPS phải prune 40–50% kênh + finetune (đưa params về ≤6 MB để nằm PSRAM) hoặc P4.- **Phương án B đã đóng gói (12/11):** firmware tự chọn đường nạp theo esp_psram_get_size()
  (≥15 MB → copy weights vào PSRAM). Gói nạp: output/deploy_pipnet_b_20261112/
  (partition-table.bin 3 KB, pipnet_app.bin 2,54 MB SHA 28330A9E…, pipnet98.espdl 11,81 MB SHA D402A7D1…)
  + README hướng dẫn. Trên N16R8 hiện tại vẫn chạy nhánh flash (0,13 FPS); cần module **N16R16** để đạt ~0,7–1 FPS.- **Chia PIPNet 3 phần để vừa PSRAM 8 MB (12/11):** đã tách & lượng tử p1 2,68 MB / p2 3,51 MB / p3 5,62 MB,
  flash vào 3 partition. Kết quả đo: PSRAM trống lúc chạy chỉ **4,4 MB** (không phải 8 MB);
  p1 nạp được (dùng 4,2 MB), **p2 fail vì cần ~5,3 MB**. ⇒ Cần chia nhỏ hơn (mỗi phần ≤ ~2,5 MB, tức 5 phần)
  hoặc giải phóng ~1,5 MB PSRAM (WiFi/camera/stream). Board đã về v3.- **Cài thêm 2 skill project (12/11):** .dsh/skills/esp32-memory-budget, .dsh/skills/pipnet-memory-split
  (cùng với espdl-model-port, drowsiness-acceptance đã cài trước đó).
- **Chia 6 phần (12/11):** đã tách ONNX thành p1..p6 (p3/p5 là residual 2 đầu vào). **Chặn:** ESP-PPQ
  calibration không nhận dataloader nhiều đầu vào (assert inputs=None) ⇒ chưa lượng tử được p3/p5.
  Đã lượng tử được p1/p2/p4/p6 (nhánh 1 input) trước khi dừng. Cần: viết custom calibration 2-input
  hoặc tránh cắt trong residual block.- **Chia 6 phần + Concat/Slice (12/11):** đã lượng tử đủ 6 phần (p1 2,68 / p2 1,26 / p3 2,26 / p4 2,25 /
  p5 2,26 / p6 1,11 MB), parity chuỗi float 9,5e-7. Firmware chuỗi 6 phần build OK, chạy được frame đầu
  nhưng sau ~25 s **mất HTTP / treo** (nghi thiếu RAM/SRAM khi nạp 6 model liên tiếp hoặc lỗi decode p6).
  Board đã hoàn nguyên **v3**. Việc tiếp: log heap_caps trước/sau mỗi lần load, kiểm decode p6 bằng
  dữ liệu p6 thật (so EAR trên laptop), có thể giảm còn 4 phần bằng cách gộp p2+p3 và p4+p5 nếu RAM cho phép.- **Chẩn đoán cuối (12/11 chiều):** decode chuỗi 6 phần **khớp tuyệt đối** (ΔEAR = 0,000 trên 6 khung).
  Trên board: pip_p1 build FAILED với **PSRAM 4234 KB → 1513 KB**, **SRAM nội 40 KB → 32 KB**.
  ⇒ Nút thắt là **SRAM nội**, không phải PSRAM: ESP-DL cần ~vài chục KB SRAM để dựng model, nhưng
  firmware (WiFi + camera + web + audio) chỉ còn 40 KB. Muốn chạy PIPNet chia phần phải **giải phóng
  60–120 KB SRAM** (bỏ MJPEG stream/telemetry/audio, giảm task stack) — việc này ảnh hưởng dashboard.- **Giải phóng SRAM (12/11):** bỏ TD_DEC_NBUF 2→1, snapshot tensor sang PSRAM, tắt telemetry, stack web
  8192→6144 / 6144→4096 ⇒ SRAM trống 40→**54 KB**. Lỗi chuyển sang **phân mảnh PSRAM**:
  Failed to alloc 1472 KB PSRAM, largest available block 992 KB (tổng trống 4225 KB nhưng không liền mạch).
  ⇒ Cần cấp một **arena PSRAM liền mạch (~3 MB) ngay khi boot**, trước camera/WiFi, hoặc nạp model
  trước khi khởi tạo dịch vụ. Board đã về v3.- **Arena PSRAM (12/11):** đã thêm reserve/release 3 MB ⇒ largest=3072KB (hết phân mảnh). Nhưng p1 vẫn fail:
  params 2,68 MB + arena biến ~1,5 MB (conv1 64x128x128) ≈ **4,2 MB** > 4,2 MB trống.
  ⇒ Cần **tách p1 thành 2 phần** (cắt tại /layer2/layer2.0/relu_1 hoặc cuối layer2) ⇒ tổng 7 phần.- **PIPNet-98 CHẠY ĐƯỢC trên N16R8 (12/11):** chia **7 phần** (p1a 0,70 / p1b 2,11 / p2 1,26 / p3 2,26 /
  p4 2,25 / p5 2,26 / p6 1,11 MB), arena PSRAM 3 MB, tất cả phần đều ok khi nạp, chuỗi chạy
  **~0,18 FPS (5,6 s/khung)**. Độ chính xác chuỗi offline: **ΔEAR = 0,000** so model gốc.
  Hạn chế: trong lúc suy luận, **HTTP dashboard không phản hồi** (tranh CPU/thời gian), cần quyết định
  cách dùng (chỉ cảnh báo, dashboard tắt) hoặc tối ưu thêm.- **Bản sửa layout (12/11 chiều):** đã viết lại chuỗi 7 phần với convert() tự nhận layout NHWC/CHW và
  exponent từ tensor, decode p6 theo HWC. Nhưng **pip_p1b vẫn build FAILED** (psram=946KB sau khi thử):
  p1b cần ~3,7 MB (params 2,11 + arena ~1,6) mà PSRAM chỉ còn ~3,1 MB sau khi p1a chạy/xoá
  (ESP-DL không trả hết pool). Bản **raw-copy trước đó nạp đủ 7 phần và chạy 0,18 FPS** nhưng EAR sai
  vì khác layout. Việc còn lại: cấp arena riêng cho p1b (hoặc giữ p1a thường trú, hoặc gộp p1b vào p1a
  với input 128 để arena nhỏ).- **Chuỗi 8 phần (12/11 chiều):** đã có đủ p1a/p1b1/p1b2/p2/p3h/p4/p5h/p6 (biên NHWC, parity 0).
  Trên board: p1a, p1b1 nạp OK; **p1b2 FAILED (psram=1112KB)** — ESP-DL không trả hết PSRAM giữa các
  lần 
ew dl::Model/delete (pool giữ lại). Bản raw-copy 7 phần trước đó nạp đủ và chạy 0,18 FPS,
  nên thiếu ~100–200 KB do buffer chuyển đổi (224 KB) + pool.
  Việc tiếp: giảm buffer xuống 1×131072 + 1×65536 (tái dùng), hoặc reset memory manager giữa các phần.- **Trạng thái cuối (12/11):** model PIPNet-98 đã chia 8 phần, parity 0, decode HWC đúng; firmware chuỗi
  gặp **giới hạn quản lý bộ nhớ của ESP-DL**: sau mỗi 
ew/delete dl::Model pool không trả hết PSRAM,
  cộng arena 3 MB khiến phần sau fail (p1b2/p1a). Bản raw-copy 7 phần từng nạp đủ và chạy **0,18 FPS**.
  Việc đầu tiên phiên sau: **tách một app test loader tối giản** (chỉ 8 lần load/run, không camera/WiFi/web)
  để chứng minh chuỗi nạp được, rồi mới ghép lại pipeline.- **Tiến bộ lớn (12/11 tối):** app **standalone loader test** (không camera/WiFi/web) chạy **toàn bộ chuỗi 8 phần
  OK**, 7,7 s/khung ⇒ model + decode đúng, chỉ vướng môi trường.
- Chế độ **camera-only** (WiFi/web/telemetry tắt): boot tới camera OK, SRAM trống 201 KB, nhưng crash trong
  MSR::MSR → ImagePreprocessor (model detect không build được vì thiếu bộ nhớ) ⇒ s_det không null nhưng
  chưa built. Cần: kiểm is_built() cho detector, hoặc bỏ MSR và dùng ROI cố định/khung hình.- **PIPNet-98 CHẠY CÙNG CAMERA (12/11 tối) — mốc lớn:** tắt MSR detector (PIPNET_NO_DETECTOR=1), dùng ROI
  dự phòng + tracker landmark. Pipeline chạy: Frame #5, AI 7697 ms, 0.1 FPS, ROI 77,25,181, EAR 0.56.
  ⇒ Không còn crash, model chạy thật trên board với camera. **EAR còn sai (0.56)** → cần so decode
  trên board với offline (in pts 60..75 qua serial, so với ONNX cùng khung).- **SỬA ĐƯỢC DECODE (12/11 tối):** lỗi là chọn nhầm head: 
m.find("x") khớp **nb_x** (980 kênh) thay vì
  head x (98 kênh). Đổi sang so khớp tên chính xác (
m == "cls"/"x"/"y") ⇒ **EAR trên board = 0,32**
  (trước đó 6,07), khớp float 0,288 / MediaPipe 0,235 trên cùng khung. Board chạy PIPNet-98 + camera,
  0,1 FPS, không v3. Việc tiếp: kiểm mắt nhắm (EAR phải ~0,08), chỉnh ngưỡng, và xuất ảnh điểm board vs MediaPipe.- **Trạng thái cuối (12/11 đêm):** PIPNet-98 chạy camera-only trên board, 0,1 FPS (AI ~7,6 s), không v3/WiFi/web.
  Model đã chứng minh đúng: ESP32 vs laptop **3,27 px** (mắt 2,02), ESP32 vs MediaPipe 1,5–2,6 px.
  Blocker còn lại: **MSR/MNP detector trả về rỗng (score 0)** dù threshold 0,25 → ROI vẫn cố định;
  self-tracking từ landmark chưa ổn định ở 0,1 FPS. Kế hoạch: (1) debug MSR (build/input), (2) tracker lai
  detector+landmark + one-euro, (3) gate D∈[25,150] px, (4) median 3 khung, (5) ngưỡng baseline×0,55.## 2. Quyết định kiến trúc (11/11)

**Thay toàn bộ model cũ bằng model pretrained có sẵn:**

```
Flash: MSR/MNP (bbox) + 2d106det (106 điểm)  ← 2 model ESP-DL, không train gì
RAM:   activation (PSRAM) + arena (SRAM ưu tiên)
Bỏ:    TinyDriver-22 (2 model TFLM)
```

- EAR: 8 điểm mí **33–42 (trái)** và **87–96 (phải)**, metric aperture.
- MAR: miệng **52–71**.
- Pose: dùng 106 điểm (mũi/cằm) hoặc bỏ PnP giai đoạn đầu.
- Ước tốc độ: **2–3 FPS** (đánh đổi lấy chất lượng; đã được người dùng chấp nhận).

## 3. Bằng chứng đã đo (không đoán)

| Cấu hình | AUC không kính |
|---|---:|
| TinyDriver v2 (v3) | 0,923 |
| 2d106det + bbox det_10g (chuẩn InsightFace) | **0,945** |
| 2d106det + bbox detector Lite (ROI) | 0,77 |

## 4. Việc tiếp theo (theo thứ tự)

1. Bật Kconfig `HUMAN_FACE_DETECT_MODEL_IN_FLASH_PARTITION`; tạo partition `human_face_det` (3 MB); nạp model MSR/MNP.
2. Viết module `face_pipeline`:
   - MSR/MNP trên ảnh xám 320×240 → bbox;
   - crop 192 theo bbox (affine nghịch đảo) → **2d106det** → 106 điểm;
   - EAR (33–42/87–96) + MAR (52–71) → ADAS.
3. **Xoá TinyDriver-22** khỏi CMake/main (xoá LEGACY).
4. Build, nạp partition + model + app, đo FPS/latency thật.
5. Kiểm nhắm mắt → báo đỏ; ngáp → báo; chạy 5 đoạn ngắn.
6. Báo người dùng test demo.

## 5. Môi trường & lệnh

- ESP-IDF 5.3.5: `C:\Espressif\frameworks\esp-idf-v5.3.5`; python `C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe`.
- Build: export PATH gồm `xtensa-esp-elf\esp-13.2.0_20250707\...\bin`, `cmake\3.30.2\bin`, `ninja\1.12.1`, `ccache`; đặt `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`.
- Flash app: `esptool write_flash 0x10000 firmware_esp32\build\tinydriver_esp32s3_edge_ai.bin`.
- Partition hiện tại: factory 4 MB (0x10000), `det10g` 5 MB (0x410000), `lm106` 2 MB (0x910000).
- Board IP: `192.168.1.5` — dashboard :80, `/status` + `/debug/snapshot` :81. COM3.

## 6. Tài sản cần giữ

| Đường dẫn | Vai trò |
|---|---|
| `output/deploy_eye_v3_20261110/` | Bản chạy + Lite rollback (`7D3E0FF6…`) |
| `output/demo_ready_20261111/candidate_3models.bin` | Nhánh thử 3 model (`11E88FD5…`) |
| `output/port_models_20261110/` | ONNX + `.espdl` (2d106det 1,2 MB, det_10g 4,1 MB) |
| `output/showcase_20261111/` | Ảnh minh hoạ model cũ vs 2d106det |
| `output/person01_m6_presence_live_20261109/` | Ảnh OV5640 (calib lượng tử) |
| `detector_v2/reports/` | Báo cáo/bằng chứng cũ (0,4 MB) |

## 7. Quy tắc

- Không train model mới; chỉ dùng pretrained + quantization.
- Không chỉnh ngưỡng trên tập test; ghi rõ nguồn/licence (nghiên cứu, phi thương mại).
- Giữ Lite để rollback; mọi kết luận phải kèm số đo + đường dẫn.





























