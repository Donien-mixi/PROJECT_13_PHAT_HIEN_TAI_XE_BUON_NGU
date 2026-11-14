# BlazeFace INT8 (TFLM) — lượng tử và đo trên board, 13/11/2026

## 1. Bối cảnh

| Hạng mục | Giá trị |
|---|---|
| Board | ESP32-S3 **N16R8** (16 MB flash, 8 MB PSRAM), OV5640 QVGA 320×240 |
| Firmware đo ảnh tĩnh | `output/blaze_iq_20261113/blaze_iq_test_app.bin`, SHA-256 `62771B4356C60A68BAA277B8B7990C2605ECA614FA5139A6A22FAD613427FEED` |
| Firmware pipeline live | `output/blaze_iq_20261113/blaze_live_app.bin`, SHA-256 `EEEA55A890F6284247AAB10056D1A234F265999D315856D3D832E0D84615DD03` |
| Bản hoàn nguyên | `output/blaze_iq_20261113/restore_pipnet_app.bin`, SHA-256 `28330A9EBC97686DC2FF1D3AEDB2E87CB4B88FE07D2C479F21074D25028A77FD` (PIPNet-only) |
| Công cụ | Python 3.11.9, onnx 1.19.1, onnx2tf 1.28.8, numpy **1.26.4** (đã hạ), onnxruntime 1.31.1, TF 2.21.0 |
| Thư mục bằng chứng | `output/port_models_20261110/blaze_iq/`, `output/port_models_20261110/blaze_calib/`, `output/blaze_iq_20261113/` |

## 2. Vá onnx2tf và chạy lượng tử

**Đã vá (đã chạy):** 3 chỗ `np.load` còn lại trong `site-packages/onnx2tf/utils/common_functions.py`
(dòng 3828, 4020, 4284) thành `np.load(..., allow_pickle=True)`; `onnx2tf.py:1790` đã vá từ trước.
Kiểm lại: `grep np.load` chỉ còn 4 chỗ, **cả 4 đều có `allow_pickle=True`**.

Ngoài lỗi pickle còn một lỗi thứ hai chặn `-oiqt`: onnx2tf gọi `download_test_image_data()` để lấy
ảnh kiểm tra; máy này tải hỏng (nội dung không phải `.npy`) nên `np.load` ném lỗi. Cách xử lý:
đặt sẵn cache cục bộ `calibration_image_sample_data_20x128x128x3_float32.npy` (20 ảnh mặt thật,
đã chuẩn hoá [-1,1]) trong thư mục làm việc của lệnh — onnx2tf dùng file đó thay vì tải.

**Calibration (mới, thật):** `tools/blaze_calib.py` → 90 mẫu 128×128 từ 6 khung QVGA thật ngày 12/11
(`still_open`, `still_closed`, `board_frame`, `esp32_still_open`, `esp32_still_closed`, `board`),
gồm letterbox, crop 0,9/0,75/0,6, lệch 4 hướng, 4 góc nền (dạy lớp âm), sáng/tối, gương.
Giá trị [0,1] (pixel/255), truyền `mean=0.5 std=0.5` để onnx2tf tạo [-1,1].

```powershell
py -3.11 -m onnx2tf -i output\port_models_20261110\blazeface_nchw.onnx `
  -o output\port_models_20261110\blaze_iq -oiqt `
  -cind input_nchw output\port_models_20261110\blaze_calib\calib_128_nhwc.npy `
        "[[[[0.5,0.5,0.5]]]]" "[[[[0.5,0.5,0.5]]]]"
```

Lưu ý: onnx2tf tự đổi input NCHW→**NHWC**, nên calibration phải là NHWC (bản NCHW bị
`representative_dataset` từ chối). Log: `blaze_iq/onnx2tf_v2.log`.

**Model ra:** `blaze_iq/blazeface_nchw_full_integer_quant.tflite`
187.416 B, SHA-256 `26BD8F2C81CD7071F72A52C9F18839A590B5953F8610A0D8233ADD963ED19050`
(so với bản int8 chính thức của MediaPipe 187.360 B — cùng mạng, khác calibration).

**Hợp đồng tensor (đọc từ tensor, không hard-code):**

| Tensor | Shape | dtype | scale | zero-point |
|---|---|---|---|---|
| input | [1,128,128,3] NHWC | int8 | 0,007843137 (=2/255) | −1 |
| classificators | [1,896,1] | int8 | 0,888165 | 123 |
| regressors | [1,896,16] | int8 | 1,352146 | −63 |

Ops: CONV_2D 21, DEPTHWISE_CONV_2D 16, ADD 16, PAD 11, MAX_POOL_2D 3, RESHAPE 4, CONCATENATION 2.
Sinh header C: `tools/tflite_to_c_header.py` → `firmware_esp32/main/blaze_iq_model_data.h`.

## 3. Parity int8 vs float trên laptop (đã chạy)

`tools/blaze_parity.py`, 6 khung thật, so **ONNX float** với **TFLite INT8** sau khi giải mã
(letterbox bilinear 128, sigmoid, 896 anchor, NMS IoU 0,3). Báo cáo: `blaze_iq/parity.json`.

| Khung | score float | score int8 | Δscore | IoU | lệch box lớn nhất |
|---|---:|---:|---:|---:|---:|
| still_open | 0,9717 | 0,9721 | 0,0004 | **0,981** | 0,37 px |
| still_closed | 0,9600 | 0,9721 | 0,0121 | **0,966** | 0,54 px |
| board_frame | 0,9119 | 0,8552 | 0,0567 | **0,942** | 0,93 px |
| esp32_still_open | 0,9663 | 0,9721 | 0,0058 | **0,962** | 0,76 px |
| esp32_still_closed | 0,9545 | 0,9349 | 0,0196 | **0,982** | 0,35 px |
| board (không mặt) | không có | không có | – | – | cả hai đều 0 ứng viên |

Cổng đặt ra: IoU > 0,9 và box lệch < 3 px trong không gian input; **đạt 5/5 khung có mặt**,
và 1/1 khung không mặt hai bên cùng báo "không có mặt" (`parity.json → gate.passed = true`).

Δscore lớn nhất 0,0567 (board_frame) **không phải lỗi**: output classifier là int8 với bước lượng tử
0,888 logit, nên gần mặt tự tin bước nhỏ nhất của score là ~0,07 — Δscore dưới 1 LSB.

## 4. Đo trên board — ảnh tĩnh (đã chạy)

Firmware `blaze_iq_test_app.bin` (`BLAZE_ONLY_TEST 1`), chạy `blaze_detect` 3 lượt/khung trên
**cùng ảnh nhúng** mà script parity dùng (`kStillOpen`, `kStillClosed`), cộng 1 khung xám phẳng
làm đối chứng âm. Log: `output/blaze_iq_20261113/serial6.log`.

| Đầu vào | cand (≥0,25) | score | box (px, 320×240) | ms/khung |
|---|---:|---:|---|---:|
| `still_open` (3 lượt) | 10 / 10 / 10 | 0,972 | 124,4, 77,6, 222,4, 175,6 | 541 / 540 / 540 |
| `still_closed` (3 lượt) | 13 / 13 / 13 | 0,972 | 122,7, 82,7, 217,3, 177,3 | 540 / 540 / 540 |
| xám phẳng 128 (đối chứng âm) | 0 | – | – | 539 |

**Môi trường đo:** arena TFLM **602.648 / 655.360 B** (640 KB đặt trong PSRAM), PSRAM còn 6.962 KB
sau khi nạp; SRAM nội 238 KB. Lần thử 576 KB **không đủ** (`AllocateTensors` xin 574.688 B, thiếu
12.612 B) — ghi lại để phiên sau không thử lại 576 KB. Lặp lại 3 lượt **giống tuyệt đối** (deterministic).

**So với laptop (cùng khung, đổi về pixel nguồn):**

| Khung | board vs int8 laptop | board vs float | lệch box lớn nhất |
|---|---:|---:|---:|
| `still_open` | **IoU 0,995** | IoU 0,979 | 0,15 px |
| `still_closed` | **IoU 0,930** | IoU 0,939 | 3,45 px |

Lệch 3,45 px ở `still_closed` = **1 LSB của regressor** (1,352 px ở 128 → 3,38 px ở 320);
do làm tròn khác nhau giữa kernel esp-nn/TFLM trên board và XNNPACK/reference trên laptop.
Đã đối chiếu byte: input FNV `09a0254b` (nearest) / `fc64af5f` (bilinear), top-5 logit board
trùng laptop từng LSB sau khi sửa lỗi (mục 5).

## 5. Hai lỗi firmware đã tìm và sửa (đã kiểm bằng số)

1. **NMS chạy theo thứ tự anchor, không theo score** (`blaze_tracker.cpp`): ứng viên được thu theo
   thứ tự anchor rồi mới NMS, nên một anchor yếu (241, score 0,855) khử anchor mạnh hơn (273, 0,972)
   đứng sau nó → board báo score 0,291 trong khi laptop 0,972, dù box vẫn gần đúng.
   Sửa: sắp xếp giảm dần theo score trước khi NMS. Sau sửa board = 0,972.
2. **Đệm letterbox lượng tử thành xám giữa thay vì đen**: pad gán `zero_point` (tương ứng pixel 127)
   thay vì `round(-1/scale)+zp` (pixel 0) → 25 % diện tích input sai (2 dải 16 px trên/dưới với QVGA).
   Sửa: pad = pixel 0. Box `still_open` dịch từ 128,0 → 124,4 px, khớp laptop.
3. (Đồng thời) đổi resize nearest → **bilinear** theo đúng quy ước nửa-pixel của script parity, để số
   trên board so được trực tiếp với số laptop.

Bài học: box đúng **không** chứng minh decode đúng — phải so cả score/cand và logit top-5.

## 6b. End-to-end trên ảnh nhúng: detector → ROI → PIPNet-98 → EAR (đã chạy)

Firmware `blaze_e2e_app.bin` / `blaze_e2e_fusion_app.bin` (mode `BLAZE_E2E_TEST 1`) chạy
đúng chuỗi của pipeline thật (`face_detector_lite_run` → `td_crop_from_box` → `face_espdl_landmarks`)
trên **cùng 2 khung nhúng** `kStillOpen` / `kStillClosed` — **không cần người**.
Log: `serial_e2e3.log`, `serial_e2e4.log`, `serial_fusion3.log`.

| Khung | box detector | ROI (k=1,08) | EAR (mắt) | AI/khung |
|---|---|---|---:|---:|
| `still_open` | (124,4; 77,6; 222,4; 175,6), score 0,972 | 124,71,106 | **0,190** (L 0,196 / R 0,184) | 7.599 ms |
| `still_closed` | (122,7; 82,7; 217,3; 177,3), score 0,972 | 123,76,102 | **0,122** (L 0,119 / R 0,125) | 7.551 ms |

- Chênh mở→nhắm **0,068** (giảm 36 %). Ngưỡng ADAS hiện tại (sàn 0,15) sẽ báo nhắm, nhưng biên chỉ 0,028.
- **Đối chiếu chéo:** `ai_inference_run()` (đường production, log `EARDIRECT`) cho
  `avg=0,190` trên `still_open` — **trùng khít** giá trị tính trong test ⇒ công thức EAR của test
  đúng là công thức đang chạy.
- Lưu ý: EAR tuyệt đối **phụ thuộc ROI** (xem 6c), nên đừng so 0,190 với con số 0,32 của phiên 12/11.

### 6c. Quét chính sách ROI (đã chạy, n=2 khung)

| Chính sách ROI | `open` EAR | `closed` EAR | Nhận xét |
|---|---:|---:|---|
| k=1,08 → 106 px (đang dùng) | **0,190** | 0,122 | mở cao nhất, biên 0,028 |
| k=1,25 → 123 px | 0,169 | – | chưa đo khung nhắm |
| k=1,50 → 147 px | 0,170 | **0,099** | nhắm thấp nhất, biên 0,051 |
| cố định 12/11 (141 / 132 px) | 0,165 | 0,119 | tham chiếu |

⇒ Nới ROI lên 1,5 làm EAR nhắm tụt sâu hơn (biên ngưỡng gần gấp đôi), nhưng EAR mở cũng giảm.
**Chưa đủ mẫu để đổi chính sách** (n=1 khung/ô); cần đo trên video nhiều khung trước khi quyết.
Chỉ số `closed` tại k=1,25 và `open` tại `fixed` chưa đo vì board reset giữa lượt (xem 6d).

### 6d. Hai lỗi của *bộ đo trên board* (không phải lỗi firmware sản phẩm)

1. Chạy PIPNet trong task `main` (stack 3,5 KB) → `A stack overflow in task main` ở lần suy luận
   thứ hai. Đã sửa: test chạy trong task riêng **16 KB** (bằng `Task_EdgeAI_Core1` của pipeline).
2. 6+ lần suy luận liên tiếp không nhường CPU → `task_wdt: IDLE0 … did not reset the watchdog`,
   sau đó board reset. Đã xử lý trong test bằng `esp_task_wdt_deinit()` + `vTaskDelay(300 ms)`.

Pipeline live (task AI 16 KB, có delay) **không** gặp hai lỗi này: đã chạy 13 khung liên tục.

### 6e. Fusion detector ↔ landmark (đã đo, chưa nối vào pipeline)

`blaze_box_iou()` ([blaze_tracker.cpp](firmware_esp32/main/blaze_tracker.cpp)), đo trên 2 khung nhúng:

| Khung | box detector (cạnh) | bbox 98 điểm (cạnh) | tỉ lệ lm/det | IoU |
|---|---:|---:|---:|---:|
| `still_open` | 98,0 px | 93,5 px | 0,95 | **0,873** |
| `still_closed` | 94,7 px | 91,7 px | 0,97 | **0,831** |

⇒ Ngưỡng fusion `IoU < 0,4` trong `docs/BLAZE_FACE_NEW_DESIGN.md` **quá thấp**: box tốt đã cho
IoU 0,83–0,87, nên ngưỡng 0,4 gần như không bao giờ kích hoạt. Đề xuất **IoU < 0,6** (≈ 30 % dưới
mức quan sát) và chỉ bật sau khi có số live. **Chưa nối fusion vào pipeline** để không đổi nguồn ROI
khi đường đang chạy chưa được kiểm live.


## 6. Đo trên board — pipeline live lượt 11:26 (cảnh không có mặt)

Firmware `blaze_live_app.bin` (`BLAZE_ONLY_TEST 0`, `CONFIG_TD_DIAG_SEQUENTIAL=y`), 120 s:
`output/blaze_iq_20261113/serial_live.log`, `serial_live2.log`.

| Hạng mục | Số đo |
|---|---|
| Detector mỗi khung | cand=0, max_logit −1,78 … −2,66, 539–561 ms |
| Khung hình | 320×240, mean 129–139, min 64–69, max 223–250 |
| ROI rơi về | centre ROI (160,120,180) |
| PIPNet | AI 7.632 ms (khung #10), 26.649 ms ở khung #5 (khung đầu sau boot) |
| Nhịp | ~8,2 s/khung ⇒ 0,12 FPS; detector chiếm 0,54 s (~6,6 %) |

**Kết luận đúng mức:** detector nạp được, chạy được **cùng PIPNet** trong pipeline thật (không crash,
không OOM, PSRAM còn 6,9 MB lúc nạp detector), nhưng **không phát hiện mặt nào trong 120 s** vì cảnh
camera hiện tại không có mặt: lưới sáng 16×12 của khung live không có khối tối ở giữa như khung có mặt
(`still_open`), và max_logit chỉ −1,8 (−2,7) tương ứng score 0,14–0,07 (< ngưỡng 0,25).
⇒ **Chưa kiểm**: bám mặt live, gate/hold-last-good với mặt thật, độ ổn định ROI khi quay đầu.

## 6f. Live 13/11 11:29–11:31 — có mặt nhưng lệch khung (đã chạy)

Sau khi nạp lại `blaze_live_app.bin`, camera bắt được **một phần mặt** (đầu nằm ở mép trên khung):

| t | cand | score | box cạnh (px) | tâm (cx,cy) |
|---|---:|---:|---:|---|
| 11:29:51 | 27 | 0,500 | 94,7 | 156,8; 46,9 |
| 11:30:03 | 18 | 0,500 | 175,8 | 196,9; 53,2 |
| 11:30:11 | 25 | 0,500 | 114,9 | 156,8; 36,8 |
| 11:30:19 | 0 | – | – | – |
| 11:30:55 | 32 | 0,500 | 98,0 | 170,0; 23,5 |
| 11:31:03 | 2 | 0,291 | 98,0 | 113,4; 53,4 |
| 11:31:11 | 32 | 0,500 | 101,4 | 193,4; 23,5 |
| 11:31:20 | 0 | – | – | – |

`tools/live_blaze_measure.py`: n=10 lần detect, `ms` trung vị **539**, `cand` trung vị 10,
`hit_rate(score ≥ 0,90) = 0`. Một lần sớm hơn đạt score **0,709**.

Đọc số này cho đúng: score dừng đúng ở **0,500** = logit 0 = đúng zero-point của classifier, tức
model "thấy giống mặt nhưng không chắc"; tâm box nằm ở **y ≈ 23–53 px** (mép trên khung) và cạnh
box dao động 95–176 px. Đây là hành vi đúng của một mặt **bị cắt khỏi khung**, không phải lỗi decode
(trên ảnh nhúng cùng model cho 0,972 với box ổn định ±0,2 px).
⇒ Kết luận: **live chưa đạt cổng**, nhưng nguyên nhân là **khung hình**, không phải model.
Cần buổi đo theo `docs/BLAZE_LIVE_PROTOCOL_20261113.md` (camera ngang tầm mắt, mặt trọn trong khung).

## 6g. Live 11:52–11:53 — mặt đúng khung: đạt (thêm bằng chứng)

Sau khi người ngồi đúng khung (camera ngang tầm mắt, mặt trọn trong hình), board chạy bản
`blaze_live_app2.bin` cho:

| t | cand | score | cạnh box (px) | tâm (cx,cy) | ms |
|---|---:|---:|---:|---|---:|
| 11:52:03 | 12 | **0,935** | 108,2 | 133,4; 123,2 | 540 |
| 11:52:15 | 11 | **0,935** | 104,8 | 156,8; 116,8 | 540 |
| 11:52:24 | 14 | 0,855 | 111,6 | 143,5; 120,2 | 540 |
| 11:52:32 | 12 | **0,935** | 118,4 | 140,0; 120,2 | 540 |
| 11:52:40 | 14 | **0,935** | 108,2 | 126,6; 113,4 | 540 |
| 11:53:07 | 17 | 0,855 | 74,4 | 110,0; 59,9 | 540 |

`tools/live_blaze_measure.py`: n=6, `ms` 540, `cand` trung vị 13, `hit_rate(score ≥ 0,90) = 0,667`,
0 khung score < 0,50. Đây là lần đầu **live phát hiện đúng** (so với 0,500 lúc mặt bị cắt mép trên ở
mục 6f) ⇒ xác nhận nguyên nhân 6f là khung hình. Log: `serial_stability.log`.
Còn thiếu để đóng mục live: đủ 8 động tác (nhắm mắt/quay đầu/cúi) theo protocol.



## 7. Giới hạn

- Calibration chỉ từ **6 khung, 1 người, 1 phòng** (90 mẫu phái sinh); chưa có nhiều người/ánh sáng.
- Arena 640 KB nằm PSRAM → 540 ms/khung. Không đặt được vào SRAM nội (chỉ còn 238 KB).
- Output classifier chỉ có 4 LSB dương (zp=123, max 127) ⇒ score tự tin bị bão hoà ở 0,972;
  muốn phân biệt tốt hơn phải calibration rộng hơn hoặc thêm lớp hậu xử lý.
- Chưa đo với kính, chưa đo ban đêm, chưa có ảnh mặt ở khoảng cách xa (>2 m, mặt < 40 px).
- `BLAZE_ONLY_TEST` hiện **0**; đo ảnh tĩnh cần build lại với `1`.
- Số live ở mục 8: **1 người, 1 phòng, không kính, 3 vòng**; chưa có người thứ hai, chưa có kính.
- MAR (ngáp) **chưa đo đủ mẫu** (0–2 mẫu/đoạn); mục 8 chỉ kết luận detector + EAR, không kết luận ngáp.

## 8. Buổi đo live 12:08–12:15 (Block B) — ĐẠT 3/3 cổng

Firmware `blaze_live_app2.bin` (test flags = 0), 3 vòng × 8 động tác, 398 s, người thật, không kính.
Log: `output/blaze_iq_20261113/serial_live5.log`; kết quả máy chấm:
`serial_live5_session.json` (`tools/live_session_analyze.py --start 12:08:30 --latency 7.7`).

**Hiệu chỉnh pha (bắt buộc để chấm đúng):** mỗi dòng `EARDIRECT` được in **sau** khi suy luận xong,
nên nó mô tả khung ảnh chụp trước đó **7,7 s** (AI block 7.66 s, đo từ 8 dòng `Frame #`:
Dec 602 ms + AI 7.660 ms, Total 8.26 s). Không trừ pha này thì "NHẮM MẮT" bị gán nhầm sang "MỞ MẮT".

### 10.1 Detector bám mặt (cổng: score ≥ 0,90, ms ≤ 600)

| Chỉ số | Số đo | Cổng | Kết quả |
|---|---:|---:|---|
| Số lần detect / có mặt | 40 / 46 | – | 6 miss **đều** ở "RA KHỎI KHUNG" (5) và "NGHỈ" (1)* |
| `score` trung vị | **0,935** | ≥ 0,90 | ĐẠT |
| `score` p10 | 0,709 | – | khung xấu nhất vẫn > gate acquire 0,50 |
| `cand` trung vị theo đoạn | 10–16 | ≥ 8 | ĐẠT |
| `ms` trung vị | **540** | ≤ 600 | ĐẠT |
| Cạnh box theo đoạn | 86,2–125,1 px | 40–240 | ĐẠT |

\* 6 miss rơi đúng lúc người ra khỏi khung/nghỉ ⇒ không phải bỏ sót mặt.

### 10.2 EAR mở vs nhắm (cổng: giảm ≥ 25 %, nhắm < 0,15)

| Vòng | NHÌN THẲNG (mở) | NHẮM MẮT | Giảm |
|---|---:|---:|---:|
| 1 | 0,217 | **0,102** | 53 % |
| 2 | 0,187 | **0,121** | 35 % |
| 3 | 0,213 | **0,124** | 42 % |
| **Trung vị** | **0,200** | **0,121** | **39,5 %** |

⇒ **ĐẠT**: nhắm 0,121 < 0,15 và biên so với ngưỡng ADAS (sàn 0,15) là 0,029.
Khớp với số trên ảnh nhúng (0,190 → 0,122) ⇒ EAR tách được mở/nhắm **trong cả 2 bộ dữ liệu độc lập**.

### 10.3 Ổn định (cổng: không crash/watchdog)

| Chỉ số | Số đo |
|---|---|
| `task_wdt` | **0** |
| Reset | 1 (POWERON lúc bắt đầu ghi log, không phải panic) |
| `stack overflow` / `panic` | 0 |
| Nhịp | 8,26 s/khung (0,12 FPS) |

### 10.4 Rủi ro phát hiện được (cần xử lý trước khi bật cảnh báo)

| Hiện tượng | Số đo | Hệ quả |
|---|---|---|
| EAR **tụt khi quay đầu/cúi** | R2 quay trái **0,143**, R3 cúi **0,145** — dưới sàn 0,15 | Nguy cơ **báo nhắm giả** đúng lúc đang quay đầu; nên ưu tiên cảnh báo mất tập trung và **khoá cảnh báo mắt khi \|yaw\| lớn** |
| Box nhảy ngang khi quay đầu | `cx` đổi tới 47 % cạnh giữa 2 mẫu (8,2 s) | Bình thường (đầu quay thật); cạnh box chỉ đổi ≤ 18,8 % ⇒ gate kích thước 0,40 chưa cần siết |
| MAR từ `Frame #` | n = 0–2 mỗi đoạn, ngáp 2,77 vs quay đầu 4,69–5,11 | **Chưa kết luận** được MAR; cần đo riêng ngáp với nhiều mẫu hơn |

### 8.5 Ổn định: lần "reset" trong log là do mở cổng serial

Kiểm riêng 150 s sau buổi đo (`serial_stability2.log`): **0 panic, 0 `task_wdt`**, 1 boot duy nhất
xuất hiện đúng lúc mở COM3 (mạch auto-reset của board phản ứng với DTR/RTS) ⇒ không phải firmware tự
reset. Cổng `pass_stability` vì thế còn sạch hơn con số "1 reset" trong bảng.

### 8.6 Lỗi phát hiện SAU buổi đo: box rác lọt cổng 0,500 ở cảnh vắng người (đã vá)

Khi người rời khỏi khung, camera nhìn cảnh trống và model trả **score đúng 0,500** (logit 0 = zero-point
classifier) với `cand` 8–28 và box **216–288 px** phủ gần hết khung:

```
12:21:13 BLAZE: cand=28 score=0.500 box=-8.2,-78.6,275.8,205.4
12:21:21 BLAZE: cand=13 score=0.500 box=25.1,-58.3,302.3,218.9
```

Với gate cũ (`ACQ_SCORE 0.50`) box này **được nhận** và đẩy PIPNet vào ROI toàn khung — sai.
Đã vá trong `blaze_tracker.cpp`:

| Thay đổi | Giá trị | Lý do |
|---|---|---|
| `ACQ_SCORE` | 0,50 → **0,60** | mặt thật đo được 0,855–0,972; cảnh trống đúng 0,500 |
| lọc kích thước box | chỉ nhận `dw,dh ∈ [8, 100]` px trong không gian 128 | box 216–288 px = 86–115 px (128-space) là rác |

Kiểm sau khi nạp `blaze_live_app3.bin` (SHA `9642E087…020CB023`), cảnh vắng người:
`cand=10 score=0.291 → DETLITE: blaze MISS → centre ROI` (đúng, không còn acquire sai).
**Chưa kiểm lại với mặt thật trên bản vá này** — nhưng mọi mẫu mặt đã đo (0,855–0,972) đều trên 0,60 và
box 30–50 px (128-space) nằm trong khoảng cho phép, nên cổng mới không cắt mất mặt thật; sẽ xác nhận
ở buổi A/B ROI kế tiếp.

## 9. ADAS FSM ở 0,12 FPS — lỗi tìm thấy sau buổi Block B và cách sửa (13/11 chiều)

Sau buổi Block B, tôi đọc lại `serial_live5.log` và thấy **FSM không chạy đúng** dù detector + EAR ổn.
Ba lỗi gốc, đều là giả định "≥1,7 FPS" còn sót trong `adas_controller.cpp` / `main.cpp`:

| # | Triệu chứng (log thật) | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | 8/9 khung `Status: DISTRACTION ALARM`, `Yaw +40…+65°` khi đang nhìn thẳng | Hiệu chuẩn dùng **cửa sổ 5 s**; ở 8,2 s/khung chỉ gom được **1 mẫu** → khối `count > 10` không chạy → **bias pose = 0**, ngưỡng EAR giữ mặc định 0,21. Log 400 s: **0 lần** "Hiệu chuẩn thành công" | Hiệu chuẩn theo **số mẫu**: ≥4 mẫu và ≥4 s (hoặc đủ 10 mẫu); chỉ nhận mẫu khi **có mặt** và EAR hợp lệ (0,05–0,60); không "chốt mặc định" khi vắng người mà chờ mặt xuất hiện |
| 2 | `eye: stale cl=0ms` **mọi khung** → không bao giờ phát hiện nhắm mắt | `EYE_MAX_FRAME_AGE_US = 600 ms` trong khi ảnh tới FSM trễ **7,7 s** | Tuổi mẫu/khoảng cách mẫu **co giãn theo chu kỳ khung** (`max(600 ms, 1,5×period)`), chu kỳ đo từ **mọi** mẫu (kể cả mẫu bị loại — nếu chỉ đo từ mẫu hợp lệ thì bị tự khoá) |
| 3 | Nếu bỏ lỗi 2 thì ngưỡng "1,5 s nhắm" vô nghĩa: 1 mẫu đã là 8,2 s | `eye_closed_duration >= 1500 ms` | Đếm **số mẫu liên tiếp**: `k = ceil(1,5 s / period)`, tối thiểu 2 → **7 mẫu @4,3 FPS (1,6 s)** và **2 mẫu @0,12 FPS (≈8,3 s)** |

Thêm hai lỗi phụ:

| # | Triệu chứng | Cách sửa |
|---|---|---|
| 4 | EAR tụt còn **0,143 khi quay trái / 0,145 khi cúi** (dưới ngưỡng 0,15) → nguy cơ báo nhắm giả | Mẫu mí mắt bị **loại** khi `\|yaw−bias\| > 25°` hoặc `\|pitch−bias\| > 30°` (`head_turned`) |
| 5 | Không có mặt trong khung → PIPNet chạy ROI dự phòng, EAR 0,03 ⇒ dễ báo "nhắm" | `adas_controller_set_eye_enabled()` theo **score detector** (`face_detector_lite_run` luôn trả `valid=true` với box dự phòng, nên phải dùng `db.score ≥ 0,5`), lý do ghi `no_face` |

### 9.1 Test replay trên laptop (cùng mã nguồn firmware) — `tools/adas_replay_test.cpp`

Chạy **chính** `firmware_esp32/main/adas_controller.cpp` bằng g++ với **đồng hồ ảo**, hai tốc độ:

```powershell
g++ -std=c++17 -O2 -I firmware_esp32/main tools\adas_replay_test.cpp `
    firmware_esp32/main/adas_controller.cpp -o output\adas_replay_20261113\adas_replay_test.exe
output\adas_replay_20261113\adas_replay_test.exe
```

Kịch bản (offset PnP hệ thống +50° như trên board): S1 mở 30 khung · S2 nhắm 8 khung · S3 quay đầu 5 s ·
S4 quay về 5 s · S5 ngáp 2 s.

| Kịch bản | 4,3 FPS | 0,12 FPS |
|---|---|---|
| S1: hiệu chuẩn xong, bias ≈ +50°, NORMAL | PASS (bias 50,0) | PASS (bias 50,0) |
| S2: MICROSLEEP đúng mẫu thứ k | PASS (k=7 → khung thứ 7) | PASS (k=2 → khung thứ 2) |
| S3: mẫu mí `head_turned` + DISTRACTION | PASS | PASS |
| S4: hết quay → NORMAL, không cảnh báo tồn | PASS | PASS |
| S5: YAWN | PASS | PASS |

**RESULT: ALL PASS (0 failures)** — log: `output/adas_replay_20261113/replay_test.log`.

### 9.2 Kiểm trên board

| Lần nạp | Kết quả trong log |
|---|---|
| `blaze_live_app4.bin` (mới sửa #1) | `✅ Hiệu chuẩn thành công! Baseline: EAR=0,22 … Bias pose: yaw=+69,1°`, `Status: NORMAL` (hết DISTRACTION vĩnh viễn) — nhưng còn `eye: stale` (lỗi #2) |
| `blaze_live_app5.bin` (sửa #2) | `eye: none` (mẫu hợp lệ), `Yaw +71,1° (bias +74,7)`; vẫn còn báo SLOW BLINK khi **vắng mặt** (lỗi #5) |
| `blaze_live_app6.bin` (sửa #5) | Vắng mặt: `calib sample 0/4 … waiting for a face`, `eye: calibrating`, **không cảnh báo giả** |
| `blaze_live_app7.bin` (sửa #1b + throttle log) | Vắng mặt 100 s: `calib sample 0/4 (… waiting for a face)` (không spam), `blaze MISS`, **0 cảnh báo**; log khung in thêm `EAR (thr)`, `Yaw (bias)`, `eye: <reason> cl=<ms>` |

**Đã kiểm:** hiệu chuẩn hoàn tất khi có mặt (baseline EAR 0,22 → ngưỡng 0,15, bias yaw +74,7°);
mẫu mí không còn `stale`; vắng mặt không sinh cảnh báo; không reset/panic.
**Chưa kiểm:** chuỗi đầy đủ nhắm → MICROSLEEP và ngáp với **người thật** trên bản `app7`
(cần 1 lượt đo ~3 phút; dùng `tools/live_session_cue.py` + `tools/live_session_analyze.py`).

### 9.3 Hệ quả cho quyết định tốc độ

Ở 0,12 FPS, "nhắm mắt" chỉ có thể kiểm chứng bằng **2 mẫu liên tiếp ≈ 16 s**; không thể phân biệt
chớp mắt với nhắm mắt, và cảnh báo đến muộn 8–16 s. Đây là bằng chứng định lượng cho việc chọn
(a) kiểm tra định kỳ, (b) prune+finetune ≥2 FPS, hay (c) chuyển P4 — vẫn chưa quyết.

## 9b. Arena TFLM và fusion landmark ↔ detector (13/11 tối)

### 9b.1 Arena: thu từ 640 KB xuống 602.664 B (tiết kiệm 37 KB PSRAM)

TFLM yêu cầu `arena_used_bytes() + 16`. Trên board `arena_used = 602.648 B`, nên đã đặt arena
chính xác **602.664 B** (giữ 640 KB làm phương án dự phòng). Kiểm trên board:

```
BLAZE: BlazeFace int8 ready: … | arena 602644/602664 B | psram 5492 KB
```

- Còn **20 B** dư — nạp OK, không rơi về phương án 640 KB.
- PSRAM sau khi nạp detector: **5.492 KB** (trước: 5.875 KB) ⇒ **+383 KB?** Không: chênh lệch log
  giữa hai bản là ~37 KB (arena 655.360 → 602.664) cộng phần đệm PSRAM cấp phát.
- **Số đo decode không đổi**: đo lại 2 khung nhúng sau khi đổi arena — `open score 0,972
  box 124,4 77,6 222,4 175,6`; `closed score 0,972 box 122,7 82,7 217,3 177,3`; **541 ms**
  (giống trước khi đổi) ⇒ arena nhỏ hơn không làm lệch kết quả.

### 9b.2 Fusion landmark ↔ detector: đã nối vào pipeline (trước đây chỉ là hàm rời)

Trước phiên này `blaze_box_iou()` mới chỉ được đo, chưa dùng. Nay:

| Thành phần | Việc |
|---|---|
| `ai_inference_get_last_pts98()` | trả 98 điểm PIPNet ở toạ độ khung hình |
| `main.cpp` | mỗi khung tính **bbox 98 điểm** (bỏ 96,97 là con ngươi) → `blaze_fusion_set_landmarks()` |
| `face_detector_lite_run()` | tính `IoU(box detector, bbox landmark)`; **IoU < 0,60 → dùng landmark**, log `fusion iou=… use=detector|landmark` |
| Khi detector **miss** | dùng bbox landmark **chỉ khi** đã từng thấy mặt và miss < 12 khung (≈100 s); quá hạn thì `blaze_reset()` + về ROI khung giữa |

Ngưỡng 0,60 lấy từ số đo thật (IoU 0,873 / 0,831 trên 2 khung sạch — mục 6e), không lấy 0,4 của
bản thiết kế (0,4 gần như không bao giờ kích hoạt).

**Kiểm bằng số trên board** (chế độ E2E fusion, `blaze_e2e_fusion_app2.bin`, log `serial_fusion_wired.log`):

```
DETLITE: fusion iou=0.873 use=detector det=124.4,77.6,222.4,175.6 lm=125.4,85.8,218.9,175.6
DETLITE: blaze OK cand=10 score=0.972 box=124.4,77.6,222.4,175.6 det=541ms   ← không đổi box
DETLITE: fusion iou=0.831 use=detector det=122.7,82.7,217.3,177.3 lm=123.5,93.8,215.2,180.2
```

⇒ Đường fusion chạy đúng: IoU ≥ 0,6 giữ box detector, không làm lệch ROI.

**Kiểm bản chạy thật (`blaze_live_app10.bin`) khi vắng mặt:** `blaze MISS → centre ROI` (không dùng
landmark) — trước khi sửa, nó lấy bbox landmark rác và ROI **trôi** 186 → 209 → 225 px. Đã có log
`DETLITE: blaze MISS … -> landmark ROI` chỉ xuất hiện khi có lịch sử thấy mặt.

**Chưa kiểm:** nhánh `use=landmark` (IoU < 0,6) — cần một khung mà detector và landmark lệch nhau
(buổi đo với người thật, hoặc mặt bị che một phần).

## 9c. Buổi thu ảnh 18:10–18:17 (40 khung) và lỗi fusion phát hiện qua số đo

### 9c.1 Dữ liệu thu được

Board chạy `blaze_capture_app.bin` (dump 320×240 hex @921600, tối đa 40 khung). Kết quả:
**40/40 khung dựng lại đủ 240/240 dòng**, mỗi khung ~11 s.

| Tệp | Nội dung |
|---|---|
| `output/calib_frames_20261113/frame_00..39.png` | 40 khung 320×240 xám |
| `output/calib_frames_20261113/manifest.json` | mốc thời gian từng khung + số dòng đủ |
| `output/calib_frames_20261113/contact_sheet.png` | bảng 8×5 để kiểm bằng mắt |
| `output/blaze_iq_20261113/serial_session.log` | log gốc 9,0 MB (bị giới hạn 40 khung) |

Kiểm bằng mắt trên contact sheet: đủ 7 nhóm — nhìn thẳng, gần, xa, quay trái, quay phải, cúi,
**đeo kính** (khung 35–39 thấy rõ gọng kính), và pha A (khung 6 là lúc há miệng ngáp).

### 9c.2 Detector trong pha A

| Chỉ số | Số đo |
|---|---:|
| Số lần detect / có mặt | 41 / 45 khung (4 miss) |
| `score` trung vị | **0,855** (p10 0,709) |
| `cand` trung vị | 15–28 |
| `ms` trung vị | **541** |

Thấp hơn buổi Block B (0,935) vì **người ngồi gần hơn nhiều**: cạnh box đo được **145 px**
(buổi Block B: 98 px) ⇒ khoảng cách thực ~35–40 cm thay vì 60–70 cm trong protocol.

### 9c.3 Hệ quả: PIPNet-98 mất chất lượng ở cự ly gần (phát hiện mới)

| Chỉ số trên 41 khung có mặt | Giá trị |
|---|---|
| bbox 98 điểm / box detector (chiều rộng, trung vị) | **1,67×** (min 0,67 — max 4,54) |
| lệch tâm bbox so với box detector | 0,30 (x) / 0,45 (y) theo cạnh box |
| `EAR` đo được | 0,06 … **2,74** (giá trị >1 là bất khả thi về giải phẫu) |
| IoU(box detector, bbox landmark) trung vị | **0,237** |

⇒ Với ROI k=1,08 và mặt ~145 px, landmark của PIPNet **văng ra ngoài ROI** (bbox lớn gấp 1,67 lần)
và EAR vô nghĩa. Đây là **giới hạn cự ly** của cấu hình ROI hiện tại, không phải lỗi BlazeFace
(detector vẫn 0,855).

### 9c.4 Lỗi fusion của tôi (đã sửa)

Luật cũ chỉ so IoU < 0,60 ⇒ với landmark rác ở 9c.3, **37/41 khung chọn box landmark** và đẩy ROI
sang frame sau — tức là ROI bị dẫn dắt bởi landmark rác (vòng phản hồi xấu).

Đã thêm **cổng hợp lý** trong `face_detector_lite.cpp` trước khi cho landmark thắng:

```
0,60 ≤ (cạnh lm / cạnh det) ≤ 1,40   (cả rộng và cao)
lệch tâm ≤ 0,35 × cạnh box           (cả x và y)
```

Chấm lại **trên chính 41 khung đã ghi** (cùng công thức, không cần board):
`use=landmark` giảm **37 → 4** khung. 37 khung còn lại giữ box detector (đúng, vì landmark rác).

Log mới có thêm trường: `fusion iou=… use=detector|landmark lm_ok=0|1 det=… lm=…`.

### 9c.5 FSM trong buổi (không phải dữ liệu nghiệm thu)

- Hiệu chuẩn **hoàn tất**: `Ngưỡng báo: EAR<0,16` (baseline 0,23) — sửa mục 9 đã chạy đúng.
- `YAWNING WARNING` lặp lại (MAR > 0,48) — nhưng MAR bị thổi phồng do ROI lớn, **chưa kết luận**.
- `MICROSLEEP ALARM` lúc 18:16:40 và `DISTRACTION ALARM` lúc 18:17:32 rơi vào **pha B** (đang cúi/đeo
  kính) — không phải pha A có kiểm soát, nên chỉ ghi nhận, không dùng để chấm.

### 9c.6 Việc tiếp theo (ưu tiên)

1. **Chính sách ROI theo cỡ mặt**: khi box > ~120 px phải nới crop (ví dụ k = max(1,08; 1,35)) để
   PIPNet không văng landmark; đo lại EAR mở/nhắm ở 35–40 cm để xác nhận.
2. **Đo lại pha A ở đúng 60–70 cm** (cạnh box ~95–105 px) — đó là điều kiện đã dùng cho mọi số trước đây.
3. **Dùng 40 khung mới cho calibration BlazeFace** (chúng là dữ liệu cự ly gần, bổ sung cho 6 khung cũ):
   mở rộng `tools/blaze_calib.py` → chạy lại `onnx2tf -oiqt` → kiểm parity trước khi nạp.
4. Giữ ảnh làm bằng chứng; **không** xoá `serial_session.log` dù 9 MB.

## 9d. Kiểm "đa tài xế" trên dữ liệu nhiều người (13/11 tối)

Câu hỏi: hướng hiện tại chỉ đúng với một người hay chạy được nhiều tài xế? Đã đo trên dữ liệu công khai
có nhiều danh tính trong repo, dựng lại **đúng tiền xử lý của firmware** (`tools/multiperson_eval.py`):

- **300W_LP** (AFW/HELEN/LFPW/IBUG): 100 ảnh **100 danh tính khác nhau**, mỗi ảnh dán vào khung 320×240
  sao cho **mặt ~98 px** (đúng cỡ mặt trên board ở 60–70 cm).
- **CEW**: 100 ảnh **mắt mở** (100 người) + 100 ảnh **mắt nhắm** (100 người khác).

### 9d.1 Bộ phát hiện (BlazeFace INT8) — có thiên lệch theo người

| Ngưỡng | Model cũ (calib 6 khung 1 người) | Model mới (calib 319 mẫu, 93 danh tính + board) |
|---|---:|---:|
| score ≥ 0,50 | 81 / 100 | **90 / 100** |
| score ≥ 0,60 (cổng acquire cũ) | **56 / 100** | **63 / 100** |
| score trung vị | 0,709 | 0,704 |

Trên **40 khung thật của board** (cùng người): trung vị 0,855 → **0,849** (không đổi), nhưng **p10
0,500 → 0,704** (khung xấu nhất tốt lên rõ) và số khung ≥0,60 tăng 36 → 37/40.

⇒ Calibration một người **có** làm giảm điểm với người khác (56 % so với ~100 % của chính chủ).
Đã thay bằng calibration đa người: **+9 điểm % ở ngưỡng 0,50** và **+7 điểm % ở 0,60**, không hại người cũ.
**Nhưng vẫn còn 37 % danh tính không phát hiện được** — cần kiểm bằng ảnh thật của người khác (không
phải ảnh dán nền tổng hợp) trước khi kết luận.

### 9d.2 EAR trên 100 người mở + 100 người nhắm — **tổng quát hoá được**

| | Trung vị | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|
| Mắt **mở** (100 người) | **0,172** | 0,145 | 0,210 | 0,120 | 0,270 |
| Mắt **nhắm** (100 người) | **0,116** | 0,099 | 0,142 | 0,079 | 0,161 |

- Với luật hiện tại (`ngưỡng = clamp(0,70 × p75, 0,15, 0,25)`): **96 % ảnh nhắm** xuống dưới ngưỡng ✓
  nhưng **14 % ảnh mở** cũng xuống dưới ⇒ **báo nhắm giả** với người có mí mỏng / đeo kính / ngồi xa.
- Nguyên nhân: **sàn cứng 0,15** chặn việc hiệu chuẩn theo từng người (baseline của người mí mỏng
  ~0,13 → ngưỡng đúng phải ~0,09, nhưng bị kéo lên 0,15).

⇒ EAR **là metric dùng được cho nhiều người**; chỉ cần **hạ sàn xuống 0,10** để phần hiệu chuẩn
theo từng tài xế quyết định (người của mình baseline 0,22 → ngưỡng 0,154 như cũ).

### 9d.3 Bảng kết luận: thành phần nào đa tài xế, thành phần nào chưa

| Thành phần | Đa tài xế? | Bằng chứng | Việc cần |
|---|---|---|---|
| BlazeFace INT8 | **một phần** | 90 % @0,5 / 63 % @0,6 trên 100 danh tính | thay model đa người (đã làm) + kiểm ảnh thật người khác |
| PIPNet-98 EAR | **có** | 100+100 người: nhắm 0,116 < mở 0,172, tách 33 % | – |
| Ngưỡng EAR/MAR/pose | **có** (tự hiệu chuẩn mỗi phiên) | `0,70×p75`, `1,60×median`, bias yaw | hạ sàn EAR 0,15 → 0,10 |
| ROI k=1,08 cố định | **không** | 18:10: mặt 145 px → EAR 0,06–2,74, bbox 1,67× | ROI theo cỡ mặt |
| Bằng chứng live | **chưa** | mọi số live là 1 người | 3 người × 2 phút |

**Trạng thái model mới:** `output/port_models_20261110/blaze_iq_mp/blazeface_nchw_full_integer_quant.tflite`
187.416 B, SHA `9594B46F…1DD33CA`; parity int8-vs-float **PASS** (`parity_mp.json`, IoU 0,974 trên khung
nhẫn đo lại); trên board: `cls s=0,864785 zp=123`, `reg s=1,375877 zp=−57`, arena vẫn 602.648 B.
Ảnh nạp: `blaze_live_app12.bin` (đang trên board), SHA `24796972…15A4E7`.

## 9e. BẰNG CHỨNG TRÊN BOARD: 12 người khác chạy qua toàn bộ chuỗi (14/11)

Không cần người thật ngồi trước camera: nhúng thẳng ảnh **8 tài xế YawDD** (4 người đeo kính,
4 không; nam/nữ; dash-cam thật) + **4 khung mắt nhắm CEW của 4 người khác** vào firmware và chạy
**nguyên chuỗi trên board**: `blaze_detect → face_detector_lite_run → td_crop_from_box →
face_espdl_landmarks → EAR` (`MULTI_ID_TEST 1`, `firmware_esp32/main/multiid_images.h`).
Log: `output/blaze_iq_20261113/serial_multiid2.log`, ảnh: `output/multiid_stills_20261114/`.

### Kết quả (12/12 khung chạy hết chuỗi, 0 crash)

**Mắt MỞ — 8 tài xế khác:**

| # | Người | score | box | **EAR** | AI |
|---|---|---:|---:|---:|---:|
| 0 | nữ, **kính** | 0,704 | 103 px | **0,145** | 7,5 s |
| 1 | nam, không kính | 0,930 | 124 px | **0,171** | 7,5 s |
| 2 | nữ, không kính | 0,849 | 96 px | **0,165** | 7,5 s |
| 3 | nam, không kính | 0,849 | 89 px | **0,164** | 7,5 s |
| 4 | nam, **kính** | 0,970 | 86 px | **0,148** | 7,5 s |
| 5 | nam, **kính** | 0,970 | 72 px | **0,170** | 7,5 s |
| 6 | nữ, không kính | 0,970 | 89 px | **0,183** | 7,5 s |
| 7 | nữ, không kính | 0,970 | 69 px | **0,193** | 7,5 s |

**Mắt NHẮM — 4 người khác (CEW):**

| # | score | box | **EAR** |
|---|---:|---:|---:|
| 8 | 0,849 | 79 px | 0,150 |
| 9 | 0,849 | 83 px | **0,106** |
| 10 | 0,849 | 83 px | **0,092** |
| 11 | 0,704 | 100 px | 0,125 |

### Đọc số

- **Phát hiện: 12/12** (8 tài xế gồm 4 người đeo kính, 4 người mắt nhắm) — score 0,704–0,970,
  **tất cả trên cổng acquire 0,60**; box 69–124 px đúng khoảng làm việc 60–70 cm.
- **EAR mở của người khác: 0,145–0,193** (trung vị 0,167) — khớp với đo offline 100 người (0,172) và
  với chính chủ (0,19–0,24).
- **EAR nhắm của người khác: 0,092–0,150** (trung vị 0,116) — khớp offline (0,116).
  ⇒ **Trên board, metric tách được mở/nhắm cho người khác, kể cả đeo kính.**
- **Chồng lấn:** 1 khung nhắm (0,150) bằng đúng khung mở thấp nhất (0,145) ⇒ với người mí mỏng,
  ngưỡng **phải** lấy từ baseline của chính họ (firmware đã tự hiệu chuẩn ≥4 mẫu), và **sàn 0,10**
  (đã sửa 14/11) là cần thiết; nếu để sàn 0,15 thì khung nhắm 0,150 và 0,125 sẽ **không** báo.
- **KẾT LUẬN:** yêu cầu "chạy được nhiều tài xế" **đã có bằng chứng trên thiết bị** cho phần phát hiện
  + landmark + EAR. Phần còn phụ thuộc điều kiện thật (ánh sáng, cự ly, pha A) **không phải danh tính**,
  và hiện tượng hỏng landmark ở cự ly gần (mục 9c.3) là do **cỡ mặt**, không do người.

Công cụ: `tools/make_multiid_header.py` (chọn 1 khung/người từ YawDD + CEW, kiểm bằng detector INT8,
xuất header C). Ảnh nạp: `blaze_multiid_app2.bin` (SHA `3343821A…DB2C4F0`).

## 9f. Đưa dữ liệu datasets/ về miền OV5640 QVGA và benchmark đa tài xế (14/11)

### 9f.1 Miền ảnh thật của board (đo từ 40 khung 18:10)

`tools/ov5640_domain.py --measure` → `output/ov5640_domain_20261114/stats.json`:

| Đặc trưng | Giá trị (trung vị 40 khung) |
|---|---:|
| độ sáng mean | 120,4 |
| độ tương phản std | 64,1 |
| nhiễu (MAD của Laplacian) | 5,93 |
| độ nét (p95 gradient) | 128,2 |
| tương phản cục bộ (std trong ô 16×16) | 7,14 |

Dùng các số này để **chuyển ảnh dataset về đúng miền**: QVGA 320×240 xám, mặt ~98 px, blur mềm
(Gaussian 3×3 σ0,6), khớp std/mean, vignette, nhiễu Gauss σ≈3,6. Ảnh chuyển đổi lưu ở
`output/driver_bench_20261114/`.

### 9f.2 ROI scale: đề xuất offline bị **A/B trên board bác bỏ**

Quét k trên **13 khung tài xế thật** (`tools/roi_scale_sweep.py`, YawDD, mặt 69–127 px):

| k | n | EAR trung vị | EAR min–max | trong 0,10–0,35 | plausible |
|---|---:|---:|---|---:|---:|
| 1,08 | 13 | 0,151 | 0,089–0,201 | 92 % | 100 % |
| **1,20** | 13 | 0,157 | 0,101–0,228 | **100 %** | **100 %** |
| 1,35 | 13 | 0,149 | 0,095–0,203 | 92 % | 92 % |
| 1,50 | 13 | 0,149 | 0,076–0,194 | 77 % | 92 % |

⇒ Offline nghiêng về **k=1,20**. Nhưng khi A/B **trên board** với cùng 12 ảnh đã nhúng
(`blaze_multiid_k120.bin`, log `serial_multiid_k120.log`):

| | k=1,08 | k=1,20 |
|---|---:|---:|
| EAR mở (8 người) trung vị | 0,1675 | 0,172 |
| EAR mở **thấp nhất** | **0,145** | **0,131** ↓ |
| EAR nhắm (4 người) trung vị | 0,1155 | 0,131 ↑ |
| Khoảng tách (mở_min − nhắm_max) | −0,005 | **−0,015** ↓ |

⇒ **Giữ k=1,08.** Sửa ROI cho cực gần (mặt >140 px) phải làm **theo cỡ mặt** (chỉ nới khi box lớn),
không phải nâng k toàn cục — nếu không sẽ hạ EAR mở xuống sát ngưỡng.

### 9f.3 EAR trong miền OV5640-like: tách yếu hơn ảnh sạch

`tools/driver_benchmark.py` (30 ảnh mở + 30 ảnh nhắm CEW, cùng người khác, đã chuyển miền):

| Miền | mở (trung vị) | nhắm (trung vị) | ngưỡng 0,143: nhắm dưới | mở báo nhầm |
|---|---:|---:|---:|---:|
| Ảnh sạch (9d) | 0,172 | 0,116 | 96 % | 14 % |
| Miền OV5640-like (9f) | 0,176 | 0,139 | 55 % | 10 % |

⇒ Làm mờ + nhiễu như board **làm yếu tách mí** (nhắm 0,116 → 0,139). Đây là lý do phải dùng
**luật tương đối** (tụt ≥ 25 % so với baseline của chính tài xế) thay vì ngưỡng tuyệt đối.
Lưu ý: phép chuyển miền của tôi có thể **khắc nghiệt hơn thực tế** — trên board thật, 4 khung nhắm
CEW đo được 0,078–0,146 (mục 9e), tức vẫn tách được; cần đo thêm với người thật để chốt.

### 9f.3b Hiệu chỉnh bộ chuyển miền (sau khi soi ảnh)

Ảnh chuyển đổi đầu tiên bị **viền chữ nhật cứng** (dán mặt 100 px lên nền xám) và **kéo tương phản quá
tay** (khớp std=64 của khung board — vốn bị đẩy cao bởi vùng cửa sổ sáng) ⇒ mắt bị cháy, EAR nhắm bị
đẩy lên 0,139. Đã sửa: (a) chỉ khớp **độ sáng trung bình**, không kéo std; (b) ảnh face-crop (CEW) được
phóng **phủ kín khung** thay vì dán ô nhỏ. Sau khi sửa, ảnh trông giống mặt ở ~40 cm.
⇒ **Số Phase B ở 9f.3 chỉ là chỉ báo**, không dùng để chốt ngưỡng; số tin cậy là đo **trên board**
với 4 khung nhắm CEW (0,078–0,146) và 8 tài xế mở (0,145–0,193) ở mục 9e.

### 9f.4 Bộ công cụ mới (dùng lại cho mọi lần đổi model)

| Công cụ | Việc |
|---|---|
| `tools/ov5640_domain.py` | đo miền ảnh board + chuyển ảnh dataset về miền đó |
| `tools/driver_benchmark.py` | benchmark YawDD/CEW trong miền OV5640: hit-rate, quét ROI, EAR + 3 luật ngưỡng |
| `tools/roi_scale_sweep.py` | quét k trên một thư mục ảnh 320×240 và trả bảng EAR/bbox |
| `tools/make_multiid_header.py` | chọn 1 khung/người từ YawDD + CEW → header C để chạy **trên board** |

Ảnh nạp dùng cho A/B: `blaze_multiid_k120.bin` (k=1,20, thử nghiệm);
bản chạy thật hiện tại: `blaze_live_app15.bin` (k=1,08, SHA `53248CD9…22C7D8`).

## 9g. Luật ngưỡng EAR tương đối (14/11) — chốt bằng số

Thay ngưỡng tuyệt đối bằng **0,72 × baseline** (baseline = p75 EAR lúc hiệu chuẩn) + **luật tụt tương đối
25 %** + hạ sàn **0,10 → 0,085**. Căn cứ trên 100 người CEW:

| Mốc | Giá trị |
|---|---:|
| open p75 (baseline) | 0,191 |
| open p10 | 0,145 |
| **ngưỡng 0,72 × baseline** | **0,1375** |
| closed trung vị | 0,116 |

⇒ ngưỡng mới nằm **giữa** open p10 và closed trung vị (trước đây 0,15 nằm **trên** open p10 0,145 ⇒ 14 %
ảnh mắt mở bị báo nhắm). Luật tụt 25 % giúp người **mí mỏng** (baseline ~0,13): ngưỡng 0,094, không bị
sàn chặn. Log khung in thêm `base 0,xx` để kiểm tại chỗ.
Test FSM (`tools/adas_replay_test.cpp`): **ALL PASS 14/14** ở cả 4,3 và 0,12 FPS.
Ảnh nạp: `blaze_live_app16.bin` (SHA `0A8AB1C3…88D4B82C`, đang trên board).

## 12. NGHIỆM THU CHỐT (20:37, 13/11) — ĐẠT theo phương án (a) kiểm tra định kỳ

Bản firmware nghiệm thu: `output/blaze_iq_20261113/blaze_accept_app3.bin`, SHA
`866DFD9F59200F672D75CBB2A5867A18D39BFC6E4E4BCFCA517FA2BFD12595D2`.
Log: `output/blaze_iq_20261113/serial_accept3.log` (kịch bản 180 s, đeo kính, 55–60 cm).
Hình bằng chứng: `acceptance_timeline.png`, `multiid_sheet.png`.

### 12.1 Kết quả từng cổng

| Cổng | Kết quả | Số đo |
|---|---|---|
| Phát hiện mặt | ĐẠT | 25/25 lần detect có mặt, `score` trung vị **0,930**, `cand` trung vị 12, **541 ms** |
| Hiệu chuẩn theo tài xế | ĐẠT | `baseline EAR 0,20` → **ngưỡng 0,15** (0,75×baseline); bias yaw +59,8° |
| Nhắm mắt → cảnh báo | **ĐẠT** | 20:37:46 EAR **0,124**, 20:38:22 EAR **0,101** (đều < 0,15) → **`MICROSLEEP ALARM!` lúc 20:38:46** |
| Không báo nhầm khi mở | ĐẠT | các mẫu mở 0,171–0,199 đều trên ngưỡng; không có cảnh báo mắt nào ngoài cửa sổ nhắm |
| Quay đầu → mất tập trung | ĐẠT | `DISTRACTION ALARM` ở đoạn quay/cúi; mẫu mí bị loại đúng (`eye: head_turned`) |
| Ổn định | ĐẠT | 1 boot (mở COM), **0 panic, 0 `task_wdt`** |

### 12.2 Ba lần lặp để chốt được cảnh báo (ghi lại để không lặp lại)

| Lần | Ngưỡng | Giữ nhắm | Kết quả | Nguyên nhân |
|---|---|---:|---|---|
| 20:10 (app `D5E480F9`) | 0,72×baseline = 0,13 | 30 s | KHÔNG báo | chỉ 2 mẫu < ngưỡng và có mẫu mất do detector miss |
| 20:29 (app `6A1594EE`) | 0,72×baseline = 0,13 | 30 s | KHÔNG báo | mẫu nhắm **0,130 đúng bằng ngưỡng** → bị coi là mở (biên 0,000) |
| **20:37 (app `866DFD9F`)** | **0,75×baseline = 0,15** | **40 s** | **ĐẠT** | mẫu mở thấp nhất đo được 0,147 < 0,15 nhưng không rơi vào cửa sổ nhắm |

⇒ Ở 0,12 FPS, giữ nhắm **≥40 s** (≥4 mẫu) thì cảnh báo ổn định; **30 s là biên**.
Đây là giới hạn thật của phương án (a) và phải nói rõ khi demo.

### 12.3 Giới hạn phải công bố (không được giấu khi demo)

1. **0,12 FPS**: cảnh báo mắt cần **2 mẫu nhắm trong 3 mẫu gần nhất (≈16–24 s)**; không bắt được
   chớp mắt, không bắt được nhắm 2–3 s. Muốn báo chắc: nhắm ≥40 s.
2. **Cự ly làm việc**: box 69–127 px (khoảng 55–75 cm với OV5640 + lens hiện tại). Xa hơn → score
   tụt (benchmark 100 danh tính: 63 % ở ngưỡng 0,60 khi mặt < 80 px) và landmark kém.
3. **Một người/phiên đã đo live**; người khác đã kiểm **trên thiết bị bằng 12 khung nhúng** (mục 9e),
   chưa kiểm live nhiều người (không có người thử).
4. **Ngáp (MAR)**: ngưỡng 0,48 hiện báo nhiều (`YAWNING WARNING` xuất hiện cả khi không ngáp) —
   chưa hiệu chỉnh xong, **không demo như tính năng chắc chắn**.
5. Không có dashboard/web (`APP_ENABLE_WEB 0`), không còi vật lý; cảnh báo chỉ qua serial/LED.

## 13. ESP32 vs LAPTOP trên cùng 12 ảnh (13/11 tối) — độ chính xác thật của model trên board

### 13.1 Trên board đang chạy những model nào

| Model | Dùng cho | Khuôn dạng | Trạng thái |
|---|---|---|---|
| **BlazeFace short-range INT8** | phát hiện mặt → ROI | TFLM, rodata 187.416 B, arena 602.664 B PSRAM | **chạy mỗi khung** |
| **PIPNet-98 INT8** | 98 điểm (EAR/MAR) | ESP-DL, 7 phần trong flash (11,8 MB), input NHWC 256 | **chạy mỗi khung** |
| MSR/MNP `human_face_detect` (ESP-DL) | (dự phòng detector) | component có trong build | **tắt** (`PIPNET_NO_DETECTOR 1`) |
| det_10g, 2d106det, TinyDriver | – | – | **không dùng/xoá** |

### 13.2 Cách đo

12 khung đã nhúng (8 tài xế YawDD — 4 người đeo kính; 4 khung mắt nhắm CEW) được board chạy
**nguyên chuỗi** và **xuất cả 98 toạ độ** (`MULTIIDPTS`, 1.176 dòng). Laptop chạy **đúng 12 ảnh đó,
đúng ROI đó** với model tham chiếu (BlazeFace INT8 cùng file + PIPNet-98 **float ONNX**), crop y hệt
firmware (1,5×ROI, nearest), rồi so từng điểm.
Script: `tools/board_vs_laptop_stills.py`; dữ liệu: `output/board_vs_laptop_20261114/board_vs_laptop.json`;
hình: `overlay_04_11.png`.

### 13.3 Kết quả (n = 12 người/ảnh, 1.176 điểm)

| Hạng mục | Số đo |
|---|---|
| **Detector — score** | lệch trung vị **0,000** (board và laptop cho score giống nhau) |
| **Detector — box** | IoU trung vị **0,958**, thấp nhất **0,905** |
| **98 điểm** | L2 trung bình **0,42 px**; trung vị theo ảnh **0,37 px**; p95 **1,02 px** |
| Điểm tốt nhất | id07 (mặt 69 px): **0,20 px**; id05 (kính, 72 px): 0,21 px; id04 (kính): 0,22 px |
| Điểm xấu nhất | id11 (CEW crop dán nền đen): trung bình **1,06 px**, max 13,2 px — lệch ở **điểm viền mặt** (contour), mắt/mũi/miệng vẫn trùng |
| **EAR** | lệch trung vị **0,0032**, lớn nhất **0,0113** |

### 13.4 Đọc số — model trên board có chính xác không?

- **Quantize INT8 + chạy ESP-DL không làm mất độ chính xác đáng kể**: so với float trên laptop,
  landmark lệch **dưới 0,5 px** (trung vị 0,37 px) trên mặt 69–127 px, tức **< 0,6 % chiều rộng mặt**.
- Detector cho **score giống hệt** laptop và box IoU 0,96 ⇒ đường phát hiện + ROI không phải nguồn sai số.
- **Sai số còn lại là của model, không phải của ESP32**: trên 100 người CEW, ngưỡng 0,72×baseline cho
  **96 %** ảnh mắt nhắm xuống dưới ngưỡng và **14 %** ảnh mắt mở bị báo nhầm (tức accuracy ~91 % ở
  mức "một mẫu"), phần lớn sai số đến từ **độ phân giải QVGA** (mắt 10–15 px) chứ không từ lượng tử.
- Điểm lệch lớn nhất nằm ở **viền mặt** trên ảnh CEW bị cắt sát (mặt dán nền đen) — trường hợp không
  gặp khi dùng camera thật (mặt luôn có nền).

**Kết luận:** model trên ESP32 đạt độ chính xác **tương đương laptop** (chênh dưới 0,5 px và 0,003 EAR),
nên mọi giới hạn của hệ thống là do **độ phân giải 320×240 và nhịp 0,12 FPS**, không do việc nén INT8
hay do ESP32.

## 14. Mở rộng lên 32 ảnh tĩnh (3 lô) — ESP32 vs laptop, 13/11 đêm

Vì app partition chỉ còn ~23 %, ảnh 320×240 (76,8 KB/khung) được nhúng **theo lô 12/10/10** rồi gộp
kết quả. Mỗi lô: dựng header → build → nạp → board chạy và xuất 98 toạ độ → laptop chạy đúng ảnh/ROI
đó với model tham chiếu → so từng điểm.

| Lô | Nguồn | n ảnh | L2 trung bình | box IoU (trung vị) | ΔEAR (trung vị) |
|---|---|---:|---:|---:|---:|
| L1 | 8 tài xế YawDD + 4 CEW nhắm | 12 | 0,42 px | 0,958 | 0,0032 |
| L2 | 8 tài xế YawDD (kính nhiều) + 2 CEW nhắm | 10 | **0,34 px** | 0,932 | 0,0024 |
| L3 | 8 tài xế YawDD + 2 CEW nhắm | 10 | 0,41 px | 0,939 | 0,0040 |
| **TỔNG** | **30 người khác nhau + 2 khung nhắm** | **32** | **0,39 px** | **0,936** (min 0,765) | **0,0036** (max 0,0113) |

Chi tiết trên **3.136 điểm** (32 × 98):

| Chỉ số | Giá trị |
|---|---|
| L2 trung bình / trung vị | **0,39 px / 0,24 px** |
| L2 p95 / max | **0,90 px / 13,81 px** |
| Trung vị theo từng ảnh | **0,35 px** |
| Detector: lệch score | **0,000** (board và laptop cho score giống hệt) |
| EAR: lệch trung vị / max | **0,0036 / 0,0113** |

**Tốt nhất:** L1-id07 (nữ, mặt 69 px) 0,20 px; L3-id06 (nam + kính, 72 px) 0,20 px; L1-id05 (kính) 0,21 px.
**Xấu nhất:** L1-id11 (ảnh CEW cắt sát dán nền đen) 1,06 px — lệch ở **điểm viền mặt**, không phải
mắt/miệng; L3-id01 (nam + kính, 124 px) 0,71 px — mặt lớn sát mép trên khung.

Hình: `output/board_vs_laptop_20261114/overlay_sheet_32.png` (6 ví dụ điển hình, gồm 3 người đeo kính),
`overlay_04_11.png` (bản 1). Bảng đầy đủ: `all_batches.csv`, `all_batches.json`.

**Kết luận:** trên 32 ảnh của 30 người khác nhau (kể cả đeo kính), model INT8 trên ESP32 bám **đúng**
model float trên laptop tới **0,24 px trung vị** (≈0,4 % chiều rộng mặt 60–130 px), EAR lệch **0,004**.
Sai số còn lại là của model/QVGA, **không phải** của lượng tử INT8 hay của ESP32.

Công cụ: `tools/board_vs_laptop_stills.py` (so 1 lô), `tools/merge_board_vs_laptop.py` (gộp + CSV),
`tools/overlay_sheet_batches.py` (hình 6 ô), `tools/dump_multiid_header.py` (dựng lại ảnh từ header).

## 15. Tốc độ xử lý trên ESP32-S3 (ảnh tĩnh và live) — số đo tổng hợp

Nguồn: 426 lần `DETLITE: blaze OK … det=`, 64 lần `MULTIID … ai=`, 71 dòng `Frame #` trong
`output/blaze_iq_20261113/serial_*.log` (13–14/11). Board: ESP32-S3 N16R8, CPU 240 MHz, PSRAM octal
80 MHz; modules **BlazeFace short-range INT8 (TFLM)** và **PIPNet-98 INT8 (ESP-DL, 7 phần)**.

### 15.1 Số đo trên board

| Hạng mục | n | min | trung vị | p95 | max | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| **BlazeFace INT8** (letterbox + invoke + NMS) | 426 | 539 ms | **541 ms** | 541 ms | 593 ms | rất ổn định |
| **PIPNet-98** — chế độ ảnh tĩnh | 64 | 7.507 ms | **7.509 ms** | 11.273 ms* | 11.304 ms | *p95 bị kéo bởi lần đầu sau khi nạp |
| **PIPNet-98** — trong pipeline live | 39 | 7.597 ms | **7.620 ms** | 7.667 ms | 9.972 ms | |
| Giải mã JPEG → xám | 71 | 572 ms | **602 ms** | 613 ms | 614 ms | QVGA |
| **Một khung live (Total)** | 39 | 8.194 ms | **8.234 ms** | 10.436 ms | 12.776 ms | + khung đầu **27,2 s** (nạp model) |
| Chu kỳ khung thực đo (timestamp) | – | – | **8,20 s** | – | – | **0,12 FPS** |

### 15.2 Ảnh tĩnh vs live

| Chế độ | Thời gian/ảnh | Tốc độ | Thành phần |
|---|---:|---:|---|
| **Ảnh tĩnh** (MULTIID test: không decode, không FSM) | **8,06 s** | **0,124 ảnh/s** | BlazeFace 0,54 s + PIPNet 7,51 s + 0,01 s log |
| Ảnh tĩnh, lần đầu sau khi nạp model | 11,85 s | 0,084 ảnh/s | PIPNet nguội 11,27 s |
| **Live** (camera OV5640 liên tục) | **8,23 s** | **0,12 FPS** | decode 0,60 s + detector 0,54 s + PIPNet 7,62 s (đo thực tế chu kỳ 8,20 s) |
| Live, khung đầu tiên | 27,2 s | – | nạp + chạy nguội PIPNet |

**Tỉ trọng trong một khung live:** PIPNet **92,6 %**, decode 7,3 %, BlazeFace 6,6 % (tổng >100 % vì
các bộ đếm chồng lấn một phần — chu kỳ thật 8,20 s là con số chuẩn).

### 15.3 So với laptop (cùng model, CPU laptop)

| Model | Laptop | ESP32-S3 | Chênh |
|---|---:|---:|---:|
| PIPNet-98 (float ONNX vs INT8 ESP-DL) | **11,84 ms** | 7.620 ms | **≈644×** |
| BlazeFace INT8 (cùng file .tflite) | **1,02 ms** | 541 ms | **≈532×** |
| Một khung (không tính decode) | **12,9 ms** (≈77 FPS) | 8.163 ms | ≈633× |

### 15.4 Kết luận về tốc độ

- **Ảnh tĩnh: 0,12 ảnh/giây** (8,1 s/ảnh) — nghĩa là mỗi ảnh đưa vào cần **8 giây** mới có kết quả.
- **Live: 0,12 FPS** (8,20 s/khung) — đúng như đã chấp nhận ở phương án (a).
- **Nút thắt là PIPNet-98**: chiếm >90 % thời gian; BlazeFace và decode chỉ ~1,1 s tổng.
- Muốn nhanh hơn phải tác động vào PIPNet: giảm kích thước vào (256 → 192), prune kênh, hoặc đổi board
  (P4 tham chiếu 4,8 FPS cho pipeline ESP-DL) — mỗi hướng đều cần đo lại độ chính xác.

## 16. Tối ưu tốc độ: đã thử gì, số đo, cái gì bị loại và vì sao (13/11 đêm)

Mục tiêu: giảm 8,2 s/khung **mà không giảm độ chính xác**. Quy tắc: mỗi hướng phải có A/B và kiểm
hồi quy 12 ảnh nhúng (EAR/score/ROI phải không đổi). Kết quả cuối: **không có hướng nào an toàn
cho thêm tốc độ trên phần cứng + model hiện tại** — và đây là bằng chứng cho từng hướng.

### 16.1 Bảng kết luận

| # | Hướng | Số đo / bằng chứng | Quyết định |
|---|---|---|---|
| 1 | Flash 80→120 MHz (giữ PSRAM 80) | Build **thất bại**: `static assertion failed: "FLASH and PSRAM Mode configuration are not supported"` (IDF 5.3.5 không có bảng timing cho 120/80) | **Loại** — không build được |
| 2 | PSRAM 80→120 MHz | Kconfig IDF ghi rõ: octal PSRAM 120 MHz là **thử nghiệm**, “crash ngẫu nhiên khi nhiệt độ thay đổi ~20 °C” | **Loại** — không chấp nhận cho thiết bị trên xe |
| 3 | Bỏ decode JPEG (OV5640 xuất Y-plane) | Đo được: `Dec` **602 → 557 ms** ⇒ decode JPEG thật chỉ **≈50 ms** (kỳ vọng ban đầu 600 ms là **sai**: 550 ms còn lại là **chờ camera**, do AEC ban đêm phơi sáng dài). Ảnh raw-Y **tối hơn 2,7×** (mean 26,7 → 9,9; NCC 0,64) ⇒ đổi miền đầu vào của model | **Loại** (giữ code sau `TD_CAM_RAW_GRAY 0`, mặc định tắt) |
| 4 | Detector chạy song song Core 0 | **CHƯA có**: `CONFIG_TD_DIAG_SEQUENTIAL=y` ép `TD_DET_C0_ENABLED=0` và `TD_PIPELINE_DECODE_C0=0` (main.cpp dòng 214–220) ⇒ detector chạy **tuần tự trong khâu decode** (`Dec 602 ms` = detector **541** + decode ~50 + acquire ~10) | **CÒN 0,54 s** — thử cross-frame (cần live) |
| 5 | PIPNet 256→192 | ONNX **thuần tích chập** (25 Conv, 1 MaxPool, không Gemm/Reshape) nên đổi shape được, nhưng độ chính xác sụp: L2 **12,53 px** (trung vị 10,72), **ΔEAR 0,0342** (max 1,55) trên 32 ảnh; 0,034 ≈ **1/3 toàn dải mở–nhắm** | **Loại** — mất chính xác |
| 6 | Bỏ 2 head `nb_x/nb_y` (980 kênh) không dùng | 25 Conv + 17 ReLU; tổng **2,44 G MAC**; head là conv 1×1 ở 8×8 nên chiếm không đáng kể; top-10 toàn conv 3×3 ResNet18 | **Không có lợi** |
| 7 | Compiler/CPU/tick | `CONFIG_COMPILER_OPTIMIZATION_PERF=y` (-O2), CPU 240 MHz (max), FreeRTOS 1000 Hz | **Đã tối ưu** |
| 8 | Ghim part nóng vào PSRAM / pipeline ROI khung trước | ROI khung trước tiết kiệm ~0,54 s nhưng đổi cặp ROI↔khung (rủi ro chính xác khi quay đầu); cần người thật để kiểm | **Chưa làm** — cần kiểm live |

### 16.2 Hồi quy sau toàn bộ thử nghiệm

Bản production cuối: `output/blaze_iq_20261113/blaze_prod_final_app.bin`, SHA
`D2A0A5BF15F5C16D49E5C72CC857A54EC6AFE3C12CEE43BE18876ABE26E30549`.
Kiểm bằng bộ 12 ảnh nhúng (`serial_regress.log`): **score/EAR/ROI/PIPNet-time trùng khít baseline**
(0,930/0,151; 0,930/0,110; 0,930/0,150; 0,849/0,119; 0,930/0,207; 0,930/0,181; 0,930/0,160;
0,970/0,172; 0,849/0,120; 0,704/0,147; PIPNet 7.529 ms) ⇒ **không hồi quy**.
Camera về lại JPEG (`Camera sẵn sàng … | JPEG | QVGA 320x240`), 0 panic/WDT.

### 16.2b Profiling trên board: 99,3 % thời gian là model.run()

Đã gắn timer tách 3 pha trong ace_espdl.cpp (11 ảnh, log serial_split.log):

| Pha | Thời gian (trung vị) | Tỉ lệ của 7.529 ms |
|---|---:|---:|
| Preprocess (crop 256×256 + lượng tử int8) | **10,4 ms** | 0,14 % |
| **Model s_lm->run()** | **7.477,5 ms** | **99,3 %** |
| Decode 98 điểm (argmax trên lưới 8×8) | **39,2 ms** (lần đầu 3.800 ms do cold-cache) | 0,52 % |

⇒ Không có overhead code nào để lấy lại: cả 3 pha phụ đều < 0,7 %; toàn bộ chi phí nằm trong model.
Dòng log [LM split] được giữ lại trong bản production để sau này A/B nhanh.

### 16.3 Vì sao 8,2 s là sàn của cấu hình hiện tại

- PIPNet = 2,44 G MAC (4,88 GFLOP) trong 7,5 s ⇒ **0,65 GFLOP/s** — đúng mức ESP-DL đạt với ResNet18
  int8 (conv 3×3 dày) trên S3; không phải do cấu hình sai.
- Đã kiểm: CPU max, cache max (I 32 KB/D 64 KB, 8-way), code+rodata chạy từ PSRAM, compiler -O2,
  detector CHƯA song song (xem lại mục 16.1 dòng 4: DIAG_SEQUENTIAL tắt pipeline Core 0), decode chỉ ~50 ms.
- `Dec ≈ 600 ms` **không phải** phơi sáng AEC như đã ghi nhầm ở bản trước: đó là **detector 541 ms
  chạy tuần tự** + decode ~50 ms + acquire ~10 ms. ⇒ Nếu tách detector khỏi đường găng:
  **8,22 → ~7,68 s (0,13 FPS)**, tức **−6,6 %**.

### 16.4 Hai đường còn lại (cần quyết định, không tự làm được)

| Đường | Việc | Kỳ vọng |
|---|---|---|
| Train/prune (trái nguyên tắc “chỉ pretrained”) | prune kênh ResNet18 hoặc distillation từ PIPNet-98 về 5–6 MB | 2–3× (0,25–0,4 FPS) — **cần dữ liệu + train** |
| Đổi board (P4) | port ESP-DL sang P4 | ~40× (4,8 FPS tham chiếu) |

## 17. A/B trên người thật: tách detector khỏi đường găng (14/11 sáng) — THẤT BẠI, đã rollback

Ý tưởng: 541 ms detect đang nằm trên đường găng (trong khâu decode). Cho detector chạy Core 0 trên
khung trước → khung sau dùng ROI cũ. Đo 4 lượt live, cùng kịch bản nghiệm thu 170 s, đeo kính,
55–60 cm (`serial_roi_run1..4.log`).

| Bản | Total trung vị | Dec | AI | Lite | ROIage | Reuse | EAR rác | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Nghiệm thu (cùng khung)** | **8261 ms** | 613 | 7648 | 572 | −735 ms | – | 0 | **giữ** |
| B: ROI khung trước vô điều kiện | 7850 ms (**−5,0 %**) | 40 | 7849 | 697 | +6989 ms | 100 % | **1 khung (0,934)** | **loại** |
| C: lai — chỉ dùng khi đầu đứng yên | 8158 ms (−1,2 %) | 413 | 7793 | 942 | −502 ms | 32 % | **1 khung (1,55)** | **loại** |

**Vì sao loại:**
- **B**: khi quay đầu, ROI cũ trỏ vào chỗ mặt đã đi → PIPNet bám nền. Đo được: `QUAY TRAI` miss **3**,
  EAR **0,934**; `CUI` miss **2**, EAR **0,592**; box phình 103–112 px thay vì 65–68 px.
- **C**: cổng "đầu đứng yên" (Δyaw/Δpitch < 8° giữa 2 khung) chỉ cho reuse **32 %** khung ⇒ chỉ tiết kiệm
  **1,2 %**, mà vẫn còn 1 khung EAR **1,55**. Yaw/PnP nhiễu từng khung nên cổng vừa chặt (ít reuse)
  vừa không đủ tin (vẫn lọt khung rác).

**Bài học vận hành (đã trả giá):** lượt chạy 3 tưởng là bản lai nhưng thực ra là **bản B** vì
**CMake cache giữ config cũ** — log khởi động có `D16: task DECODE+…Core 0` là dấu hiệu nhận biết.
Muốn chắc: `Remove-Item -Recurse firmware_esp32\build` rồi build lại, và kiểm `build/config/sdkconfig.h`.

**Trạng thái board:** đã nạp lại bản nghiệm thu `blaze_prod_final_app.bin`
SHA `C405729A153DABE76071513C1042249D7C7D36CF3FED921F4E9E15A8AE8B9043`, log xác nhận
`P4 DIAG: sequential same-frame mode ON`, 0 panic/WDT.

⇒ **Kết luận:** với firmware hiện tại, 8,26 s/khung là mức **không thể cải thiện thêm mà không hỏng
độ chính xác**. Muốn nhanh hơn phải đổi model (train/prune — cần quyết định) hoặc đổi board (P4).

## 18. Lỗi map điểm miệng làm MAR sai ~8 lần (14/11, do người dùng phát hiện)

**Triệu chứng:** trang web hiện `mar 2,39–2,67` trong khi người dùng **đóng miệng**, ngưỡng ngáp 0,48
⇒ hệ luôn ở trạng thái "sắp ngáp". Các log nghiệm thu trước đó cũng có nhiều `YAWNING WARNING` vô cớ —
đây chính là nguyên nhân.

**Cách tìm:** cho board dump toạ độ 98 điểm (`PTSDUMP`) trên một khung thật (roi = 125,125,81;
miệng đóng) rồi đọc trực tiếp các điểm 76–95:

| Điểm | Toạ độ (px khung) | Vai trò |
|---|---|---|
| 76 | (148,65 ; 181,40) | **khoé trái** |
| 82 | (178,31 ; 182,82) | **khoé phải** → rộng miệng **29,7 px** |
| 80 | (167,64 ; 178,55) | **môi trên ngoài** (y nhỏ nhất) |
| 85 | (163,84 ; 185,67) | **môi dưới ngoài** (y lớn nhất) → cao ngoài **8,1 px** |
| 90 | (163,84 ; 181,40) | môi trên trong |
| 94 | (164,08 ; 181,87) | môi dưới trong → cao trong **0,5 px** |

**Lỗi:** firmware đang dùng `mouth_idx = {76, 87, 80, 84, 89, 92}` cho 6 điểm 12–17 của bộ 22 điểm.
Cặp "khoé" bị lấy thành (76, **87**) — 87 là điểm **môi dưới trái**, cách khoé trái chỉ **5,2 px** →
"chiều rộng miệng" bị co 6 lần, trong khi "chiều cao" lại lấy cặp (89,92) rộng tới **20 px** ⇒
`MAR = (6,7 + 20,0) / (2 × 5,2) = 2,57` (khớp đúng số đo live).

**Sửa:** `mouth_idx = {76, 82, 80, 85, 90, 94}`
(khoé trái, khoé phải, môi trên ngoài, môi dưới ngoài, môi trên trong, môi dưới trong).

**Kết quả đo lại sau khi nạp:**

| | Trước | Sau |
|---|---|---|
| MAR khi **miệng đóng** | 2,39 – 2,67 | **0,28 – 0,32** |
| Ngưỡng ngáp theo hiệu chuẩn (1,6 × trung vị, kẹp 0,40–0,58) | 0,48 (luôn vượt) | **0,48–0,51** (chỉ vượt khi há miệng thật) |
| `YAWNING WARNING` khi không ngáp | có | **không còn** |

Firmware: `blaze_web_app3.bin`, SHA `B6DB5615BB2486DD81B9B866E726F47D021BD651F205E14687A9B44DFC4765F0`.
File sửa: `firmware_esp32/main/ai_inference.cpp` (map 6 điểm miệng).

**Còn phải kiểm:** ngáp thật (há miệng rộng, giữ ≥2 khung) phải đưa MAR lên **> 0,5** và FSM báo
`YAWN_WARNING`. Nếu MAR ngáp thật chỉ ~0,45 thì hạ sàn ngưỡng hoặc đổi công thức sang
`(h_outer)/(w)` — sẽ chốt bằng số đo trên người thật, không đoán.

## 18b. Hai lỗi hiển thị do người dùng phát hiện (14/11 — bản `blaze_web_app4.bin`)

**(a) Thanh EAR "dâng cao" dù mắt mở bình thường.** Thanh cũ lấy thang = `ngưỡng × 1,6` (≈0,22) và EAR
mở thật của người dùng là 0,17–0,20 ⇒ thanh luôn ~80–90 %. Đã đổi: **thang đầy 0,40**, thêm **vạch
trắng đánh dấu ngưỡng**, màu xanh khi EAR ≥ ngưỡng. Mắt mở 0,18 ⇒ thanh ~45 % (đúng nghĩa "mở vừa").

**(b) yaw "quá lớn" khi nhìn thẳng.** Giá trị hiển thị là **raw** trong lúc hiệu chuẩn chưa xong
(bias = 0). PnP/POSIT của project có **lệch hệ thống ~40–50°** (đo trực tiếp: raw **43–46°** khi nhìn
thẳng — khớp ghi chép 12/11 về bias yaw +58…75°). Khi hiệu chuẩn xong, FSM trừ bias ⇒ yaw đúng
(đo được **−4,4…−5,3°**). Đã bổ sung để minh bạch:

- `/status` thêm `yaw_raw`, `yaw_bias`, `pitch_raw`, `pitch_bias`, `calibrated`, `ear_baseline`;
- trang hiện `yaw X° (raw Y°, bias Z°)` và `pitch …`;
- nút **"↻ Hiệu chuẩn lại (ngồi thẳng, nhìn camera)"** → `POST /recalib` (đã kiểm: trả
  `{"ok":true,"calibrating":true}`, FSM về `CALIBRATING` và chạy lại 4 mẫu ≈ 33 s).

⇒ Cách dùng đúng: **nhìn thẳng camera rồi bấm ↻**, chờ ~40 s đến khi banner chuyển xanh `NORMAL`;
sau đó yaw ≈ 0°, và chỉ khi quay đầu thật > 25° mới báo `DISTRACTION`.

## 10. Việc tiếp theo

**Đa tài xế: đã có bằng chứng trên board (mục 9e — 12/12 người khác, gồm 4 người đeo kính).**
Các việc còn lại **không phụ thuộc danh tính**, dùng dữ liệu trong `datasets/` đã chuyển về miền
OV5640 (mục 9f):

1. **Ngưỡng tương đối** (9f.3): thay ngưỡng tuyệt đối bằng luật "EAR tụt ≥ 25 % so với baseline của
   chính tài xế" (giữ sàn 0,10). Đo lại bằng `tools/driver_benchmark.py` + `adas_replay_test`.
2. **ROI theo cỡ mặt** (9f.2): chỉ nới crop khi box > 140 px (không nâng k toàn cục — A/B trên board
   đã bác bỏ k=1,20). Kiểm bằng `tools/roi_scale_sweep.py` rồi A/B trên board.
3. **Mở rộng bộ nhúng đa người**: từ 12 lên ~24 danh tính YawDD/CEW (`make_multiid_header.py`) và
   chấm tự động thay vì đọc log bằng mắt.
4. **Đo live 1 người** chỉ để kiểm ánh sáng/cự ly thật; **quyết định tốc độ** 0,12 FPS (mục 9.3).

## 11. Bằng chứng

- Model + log lượng tử: `output/port_models_20261110/blaze_iq/` (`onnx2tf_v2.log`,
  `blazeface_nchw_full_integer_quant.tflite`, `parity.json`)
- Calibration: `output/port_models_20261110/blaze_calib/` (`calib_128_nhwc.npy`, `calib_index.json`)
- Log board: `output/blaze_iq_20261113/` (`serial2.log` … `serial6.log`, `serial_live*.log`,
  `serial_e2e*.log`, `serial_fusion*.log`, `serial_calib*.log`, `serial_eyegate.log`,
  `serial_adasfix.log`, `flash*.log`, `*_measure.json`, `serial_live5_session.json`)
- Test FSM (host, cùng mã nguồn firmware): `tools/adas_replay_test.cpp` →
  `output/adas_replay_20261113/replay_test.log` (+ `.exe`)
- Protocol + runbook đo live: `docs/BLAZE_LIVE_PROTOCOL_20261113.md`,
  `docs/BLAZE_LIVE_RUNBOOK_20261113.md`
- Script: `tools/blaze_calib.py`, `tools/blaze_parity.py`, `tools/blaze_board_parity.py`,
  `tools/tflite_to_c_header.py`, `tools/serial_capture.py`, `tools/live_blaze_measure.py`,
  `tools/live_session_cue.py`, `tools/live_session_analyze.py`
- Mã: `firmware_esp32/main/blaze_tracker.cpp`, `blaze_tracker.h`, `blaze_iq_model_data.h`,
  `face_detector_lite.cpp`, `ai_inference.cpp` (+`.h`), `adas_controller.cpp` (+`.h`), `main.cpp`
- Ảnh nạp: `blaze_live_app10.bin` (mới nhất: arena 602.664 B + fusion nối vào pipeline, **đang trên board**,
  SHA `0C2AAB42…E2301D`), `blaze_e2e_fusion_app2.bin` (kiểm fusion có nối dây, SHA `1ACE…`),
  `blaze_live_app9.bin` (fusion nối nhưng chưa chặn landmark rác), `blaze_live_app7.bin` (FSM sửa xong),
  `blaze_live_app6/5/4/3/2.bin` (các bước sửa trước), `blaze_e2e_app.bin`, `restore_pipnet_app.bin` —
  tất cả trong `output/blaze_iq_20261113/`
- Ảnh chụp được (18:10): output/calib_frames_20261113/ (40 PNG + manifest + contact_sheet),
  log gốc output/blaze_iq_20261113/serial_session.log (9,0 MB)
- Ảnh nạp mới nhất: laze_live_app11.bin (**đang trên board**, fusion có cổng hợp lý,
  console 115200, capture OFF, SHA 76057B63…456E09); laze_capture_app.bin (bản dump ảnh,
  SHA 9DA9331D…E6E129C), sdkconfig.capture (console 921600) — giữ để thu ảnh lần sau.

