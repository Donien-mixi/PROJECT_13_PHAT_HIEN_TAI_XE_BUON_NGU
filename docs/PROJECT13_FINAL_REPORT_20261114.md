# PROJECT 13 — BÁO CÁO CUỐI (14/11/2026)

**Phạm vi đóng:** hệ **kiểm tra định kỳ** buồn ngủ / mất tập trung cho tài xế trên ESP32-S3 + OV5640,
chạy **pretrained + lượng tử** (không train), đã nghiệm thu trên người thật và trên 30 người khác nhau
từ `datasets/`. **Tối ưu tốc độ đã dừng theo quyết định của chủ dự án** (xem §7).

---

## 1. Mô hình dùng trên ESP32 (phiên bản chính xác)

### 1.1 Bảng mô hình

| # | Mô hình | Nguồn gốc | Định dạng trên board | Kích thước | SHA-256 | Nằm ở |
|---|---|---|---|---|---|---|
| 1 | **BlazeFace short-range** (MediaPipe face detector) | `output/port_models_20261110/blazeface_nchw.onnx` (423.154 B) → `onnx2tf 1.28.8 -oiqt` lượng tử int8, hiệu chuẩn **319 mẫu / 93 danh tính** (300W_LP + CEW + 40 khung board) | **TFLite int8**, chạy bằng **TFLM** (`esp-tflite-micro`) | **187.416 B** | `9594B46F455E5D015A15F45EEBBCC4E0F97D583AA6A2751934748CF161DD33CA` | rodata của app (`firmware_esp32/main/blaze_iq_model_data.h`) |
| 2 | **PIPNet-98** (ResNet18 backbone, 98 điểm, train trên WFLW) | `output/port_models_20261110/pipnet98_resnet18.onnx` (49.322.794 B) → **ESP-PPQ** → `.espdl` int8 | **ESP-DL** (`dl::Model`), fbs **7 phần**, đọc trực tiếp từ flash partition (XIP qua cache) | **12.388.576 B** (11,8 MB) | `D402A7D1C3B274338C90D8BD8D94F1A62DEEA692DD15E2B616CA805902633909` | flash partition **`pipnet`** @ `0x300000` (12 MB); đọc lại từ board: `output/blaze_iq_20261113/board_pipnet_partition.bin` |

### 1.2 Hợp đồng lượng tử (đọc từ log khởi động thật)

**BlazeFace (TFLite int8)**
```
input  : 1x128x128x3 int8   scale 0.007843137718737125   zero_point -1
outputs: classificators 1x896x1  scale 0.864785  zp 123
         regressors     1x896x16 scale 1.375877  zp -57
anchors: 896 (16x16x2 + 8x8x6); NMS IoU 0.3; score acquire 0.60; box 8..100 px (thang 128)
arena  : 602.664 B PSRAM (dùng 602.644)
```
**PIPNet-98 (ESP-DL int8)**
```
input  : 1x256x256x3 int8, q = pixel − 128 (scale 1, zp −128), xám 3 kênh
output : cls  1x8x8x98  (exp −7)   — logits trên lưới 8x8
         x,y  1x8x8x98  (exp −6)   — offset trong ô lưới
         nb_x,nb_y 1x8x8x980 (exp −5) — KHÔNG dùng trong decode hiện tại
decode : lấy ô có cls lớn nhất cho từng điểm → (x,y) ∈ [0,1) trong ô → toạ độ khung
```

### 1.3 Mô hình **không** dùng (đã tắt/bỏ)

| Mô hình | Trạng thái |
|---|---|
| MSR/MNP `human_face_detect` (ESP-DL) | component có trong build nhưng **tắt** bằng `PIPNET_NO_DETECTOR 1` |
| `det_10g` (+ `2d106det`) | **không dùng** (đã thử 12/11, bị bác vì chia 106 điểm không ổn định) |
| TinyDriver-22 | **đã xoá** khỏi đường chạy (bị bác vì điểm không bám bộ phận mặt) |
| Haar / MediaPipe host | chỉ dùng trên laptop để **so sánh**, không có trên board |

### 1.4 Tổng tài nguyên model trên board

| Vùng | Chứa | Dùng |
|---|---|---|
| app partition (3,08 MB) | code + BlazeFace 187 KB | **1,79 MB (58 %)** |
| `pipnet` (12 MB) | PIPNet-98 .espdl | **11,8 MB (96 %)** |
| `blaze` (1 MB) | trống (BlazeFace đã ở rodata) | 0 % |

---

## 2. Kiến trúc chạy

```
OV5640 320x240 (JPEG) → decode 0,05 s → xám
  → BlazeFace int8 (TFLM) 0,54 s → face box → ROI 1,08×
  → PIPNet-98 (ESP-DL) 7,48 s → 98 điểm → EAR/MAR/pose (PnP, 22 điểm)
  → FSM ADAS: hiệu chuẩn 4 mẫu / p75 → ngưỡng EAR = 0,75×baseline (sàn 0,085)
             cửa sổ 2 mẫu trong 3 (0,12 FPS), head-turn invalidation
  → cảnh báo: MICROSLEEP / DISTRACTION / YAWN (qua serial)
```

---

## 3. Độ chính xác (đã kiểm trên thiết bị)

| Phép đo | Kết quả |
|---|---|
| **ESP32 vs laptop**, 32 ảnh / 30 người (`output/board_vs_laptop_20261114/`) | 98 điểm lệch **0,24 px trung vị / 0,39 px trung bình / p95 0,90**; box IoU **0,936**; score lệch **0,000**; ΔEAR **0,0036** |
| **12 người khác nhúng trong firmware** (`serial_multiid2.log`) | 12/12 phát hiện, score 0,704–0,970; EAR mở 0,145–0,193; EAR nhắm 0,092–0,150 |
| **100 người mở + 100 người nhắm (CEW)** | EAR mở trung vị 0,172 / nhắm 0,116; 96 % nhắm dưới ngưỡng |
| **Nghiệm thu người thật 20:37** | baseline EAR 0,20 → ngưỡng 0,15; mẫu nhắm 0,124 & 0,101 → **MICROSLEEP ALARM** (20:38:46); mẫu mở 0,171–0,199 không báo nhầm |

---

## 4. Tốc độ (đo trên board)

| Hạng mục | Thời gian |
|---|---:|
| BlazeFace int8 (letterbox + invoke + NMS) | **541 ms** (n=426, p95 541) |
| PIPNet-98 — preprocess | 10,4 ms |
| PIPNet-98 — `model.run()` | **7.477 ms** |
| PIPNet-98 — decode 98 điểm | 39,2 ms (lần đầu 3,8 s cold) |
| Một khung live | **8,26 s** → **0,12 FPS** |
| Ảnh tĩnh (bộ test nhúng) | 8,06 s/ảnh (0,124 ảnh/s) |

---

## 5. Bộ nhớ (đo trên board)

| Vùng | Trống lúc chạy |
|---|---|
| SRAM nội 512 KB | 252–293 KB |
| PSRAM 8 MB (octal 80 MHz) | 3,7–5,5 MB (sau khi nạp BlazeFace arena 0,60 MB + PIPNet ~1,1 MB) |

---

## 6. Bản firmware

| Bản | File | SHA-256 |
|---|---|---|
| **Đang trên board (bản nghiệm thu)** | `output/blaze_iq_20261113/blaze_prod_final_app.bin` | `C405729A153DABE76071513C1042249D7C7D36CF3FED921F4E9E15A8AE8B9043` |
| Bản nghiệm thu (trước log `[LM split]`) | `blaze_accept_app3.bin` | `866DFD9F…1D2595D2` |
| Model calib 1 người (dùng cho mọi số trước 14/11) | `blaze_live_app11.bin` | `76057B63…456E09` |
| Snapshot ảnh qua serial | `blaze_snapshot_app.bin` / `blaze_snapshot_multiid.bin` | – |
| Ảnh test đa người (12 danh tính) | `blaze_multiid_app2.bin` | `3343821A…DB2C4F0` |
| Hoàn nguyên PIPNet-only | `restore_pipnet_app.bin` | `28330A9E…028A77FD` |

---

## 7. Tối ưu tốc độ: đã dừng — mọi hướng đều bị loại bằng số

| Hướng | Kết quả | Lý do loại |
|---|---|---|
| Flash 120 MHz (PSRAM 80) | build fail | IDF: “FLASH and PSRAM Mode configuration are not supported” |
| PSRAM 120 MHz | – | IDF: octal 120 MHz là thử nghiệm, crash khi nhiệt độ đổi ~20 °C |
| Bỏ decode JPEG (Y-plane) | −50 ms | decode thật chỉ ~50 ms; ảnh raw-Y tối 2,7× → đổi miền model |
| Tách detector (ROI khung trước) | −5,0 % (7,85 s) | **hỏng khi quay/cúi**: EAR 0,934; miss 3; box 103–112 px |
| Lai (chỉ khi đầu đứng yên) | −1,2 % | reuse chỉ 32 %, vẫn lọt 1 khung EAR 1,55 |
| PIPNet 192 | – | L2 12,5 px; ΔEAR 0,034 (32 ảnh) |
| Bỏ head `nb_x/nb_y` | ~0 | head 1×1 ở 8×8, không đáng kể MAC |
| Compiler/cache/CPU | đã max | `-O2 PERF`, I 32 KB / D 64 KB, 240 MHz |

**Kết luận:** 8,26 s/khung là **sàn** của cấu hình hiện tại; 99,3 % thời gian là `PIPNet.run()`
(2,44 G MAC ≈ 0,65 GFLOP/s). Muốn nhanh hơn phải **train/prune** (trái nguyên tắc pretrained) hoặc
**đổi board P4**.

---

## 8. Giới hạn phải nêu

1. **0,12 FPS** — cảnh báo mắt cần 2 mẫu nhắm trong 3 mẫu gần nhất (~16–24 s); muốn chắc: nhắm **≥40 s**.
   Không bắt được chớp mắt / nhắm 2–3 s.
2. **Cự ly 55–75 cm** (box 69–127 px). Ngoài khoảng đó score và landmark giảm; gần hơn ~40 cm
   (box >140 px) landmark văng (mục 9c).
3. **Ngáp/MAR chưa hiệu chỉnh** — ngưỡng 0,48 báo cả khi không ngáp, không demo như tính năng chắc chắn.
4. **1 người chạy live** (chủ dự án); 30 người khác chỉ kiểm bằng ảnh (trên thiết bị + trên laptop).
5. Không dashboard/web/còi; cảnh báo qua serial. Không WiFi (tắt để dành CPU/RAM).

---

## 8b. Xem board qua WiFi (bản laze_web_app.bin, SHA 2CFB053B6F709C5568EB20EE20D7C11AD432C2555C337FD264A614ABD1AB737E)

Wi-Fi STA + HTTP server (obs_server.cpp), **không đổi đường suy luận** (Total 8.229 ms như bản không WiFi).

| Endpoint | Nội dung |
|---|---|
| http://192.168.1.5/ | **trang cảnh báo**: banner to đổi màu (xám/xanh/vàng/đỏ/tím) + nhấp nháy + bíp + đổi tiêu đề tab khi có cảnh báo; ảnh ROI+98 điểm; thanh EAR/MAR so ngưỡng; 6 cảnh báo gần nhất |
| /status | JSON: frame, fps, EAR/ngưỡng, MAR, yaw/pitch, detector score/ms, state, status |
| /snapshot.bmp | ảnh 320×240 xám BMP 77.878 B, có ROI + 98 điểm |

Mật khẩu WiFi nằm trong irmware_esp32/main/wifi_secret.h (**gitignored**); bản thân repo đã được
rà soát để không còn chuỗi mật khẩu nào.

## 9. Cách chạy / hoàn nguyên

Xem `docs/DEMO_CHECKLIST_PIPNET_20261113.md` (7 bước demo, lệnh nạp, xử lý sự cố) và
`docs/SERIAL_SNAPSHOT_VIEWER_20261114.md` (chụp ảnh ROI + 98 điểm qua USB serial).

---

## 10. Chỉ mục bằng chứng

| Nội dung | Đường dẫn |
|---|---|
| Báo cáo kỹ thuật đầy đủ (mục 1–17) | `docs/BLAZE_IQ_RESULT_20261113.md` |
| Bản đồ bộ nhớ + hướng tăng tốc | `docs/MEMORY_AND_SPEED_OPTIMIZATION_20261113.md` |
| Nghiệm thu trên người thật | `output/blaze_iq_20261113/serial_accept3.log`, `acceptance_timeline.png` |
| Đa người trên thiết bị | `serial_multiid2.log`, `multiid_sheet.png`, `output/multiid_from_header_20261114/` |
| ESP32 vs laptop (32 ảnh) | `output/board_vs_laptop_20261114/all_batches.csv` + `overlay_sheet_32.png` |
| PIPNet 192 bị loại | `output/pipnet192_20261114/feasibility.json` |
| A/B tách detector (bị loại) | `serial_roi_run1..4.log`, mục 17 |
| Model đang nạp | `output/blaze_iq_20261113/board_pipnet_partition.bin` (12 MB, đọc từ board) |
| Test FSM trên laptop | `output/adas_replay_20261113/replay_test.log` (ALL PASS 14/14) |

