# Agent — Project 13 (ESP32-S3 driver monitoring)

Đọc [handoff.md](handoff.md) trước. Kiến trúc hiện tại: **chỉ dùng model pretrained**, không train.

## Bối cảnh

- Board: ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM, 512 KB SRAM), OV5640 QVGA 320×240.
- Mục tiêu: phát hiện buồn ngủ (mắt nhắm, ngáp) + mất tập trung (quay đầu) cho vài người,
  **không đeo kính là điều kiện test chính**.
- Firmware đang chạy: v3 (TinyDriver-22 + detector Lite). **TinyDriver bị bác** vì điểm không bám
  bộ phận mặt; đang thay bằng **MSR/MNP → 2d106det** (ESP-DL, pretrained).

## Nguyên tắc

1. **Không train model.** Chỉ: tải pretrained → lượng tử (ESP-PPQ/ESP-DL) → tích hợp → sửa lỗi.
2. Model nằm trong **flash** (rodata hoặc partition); activation/arena trong **RAM/PSRAM**.
   Flash 16 MB không phải giới hạn — **RAM và tốc độ** mới là.
3. Mọi thay đổi phải có: số đo trước/sau, đường dẫn bằng chứng, hash firmware.
4. Giữ bản chạy được (v3) và Lite để rollback trước khi thử nghiệm.
5. Không chỉnh ngưỡng trên tập test; không dùng test `person01_test_20261107_04`, reserved 83,
   `p4_real_01` để tune.
6. Kết luận phải trung thực: nêu rõ “đã kiểm” vs “chưa kiểm”, kèm giới hạn.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `firmware_esp32/main/` | nguồn firmware: `main.cpp` (pipeline), `ai_inference.cpp` (TFLM, LEGACY), `face_espdl.cpp` (ESP-DL 2d106det), `adas_controller.cpp` (FSM), `web_server.cpp` |
| `firmware_esp32/partitions.csv` | factory 4 MB, `det10g` 5 MB, `lm106` 2 MB |
| `output/port_models_20261110/` | ONNX gốc + `.espdl` (2d106det 1,2 MB, det_10g 4,1 MB) + `tflite/` |
| `output/deploy_eye_v3_20261110/` | bản v3 + Lite rollback |
| `output/demo_ready_20261111/` | checklist demo + nhánh 3 model |
| `output/showcase_20261111/` | ảnh minh hoạ đầu ra model |
| `detector_v2/reports/` | báo cáo lịch sử (bằng chứng) |

## Lệnh thường dùng

- Build: export PATH (xtensa 13.2.0, cmake 3.30.2, ninja, ccache) + `PYTHONIOENCODING=utf-8`;
  chạy `python.exe <IDF>\tools\idf.py -C firmware_esp32 build`.
- Flash app: `esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x10000 <app.bin>`.
- Kiểm: `http://192.168.1.5:81/status`, `/debug/snapshot` (cùng frame, có ROI + 22 điểm).

## Việc đang làm

Thay TinyDriver-22 bằng pipeline **MSR/MNP → 2d106det → EAR/MAR**:
1. Bật model MSR/MNP trong flash partition `human_face_det`.
2. Module `face_pipeline`: detect → crop 192 → 106 điểm → EAR (33–42, 87–96) + MAR (52–71).
3. Xoá LEGACY (TinyDriver + ai_inference) khi pipeline mới chạy.
4. Đo FPS/latency, chạy 5 đoạn nghiệm thu, rồi mời người dùng test demo.
