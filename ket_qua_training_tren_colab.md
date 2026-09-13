   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.9/37.9 MB 61.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 137.4/137.4 kB 14.5 MB/s eta 0:00:00
📥 Hãy chọn file 'training_package.zip' từ máy tính của bạn:
training_package.zip
training_package.zip(application/x-zip-compressed) - 59212339 bytes, last modified: 12/9/2026 - 100% done
Saving training_package.zip to training_package.zip
🚀 Đang giải nén và bắt đầu huấn luyện...
2026-09-12 14:54:39.813555: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
===========================================================================
🚀 BẮT ĐẦU HUẤN LUYỆN TINYDRIVER PFLD-EDGE (PHIÊN BẢN CẢI TIẾN)
   Kiến trúc: MobileNetV2 MBConv + Multi-Scale Fusion + Auxiliary 3D Pose Head
   Hàm mất mát: Adaptive Biometric Wing Loss + Geometric EAR/MAR Constraint Loss
   Tăng cường: Bounding Box Translation & Scale Jitter (Triệt tiêu Mean Face)
   Mục tiêu:   Mixed-Precision INT8 (Convs INT8 + Head Float32) cho ESP32-S3 N16R8
===========================================================================
✅ Đã kích hoạt phần cứng GPU: /physical_device:GPU:0 (Khuyên dùng Tesla T4)

[Bước 1/5] Khởi tạo Mô hình Thầy MediaPipe Face Mesh...
[Teacher] Dang dong bo mo hinh MediaPipe Tasks Face Landmarker (~3.7MB)...
WARNING: Logging before InitGoogle() is written to STDERR
W0000 00:00:1789224885.267511    4035 face_landmarker_graph.cc:180] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1789224885.282779    4039 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1789224885.312818    4040 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
[Teacher] MediaPipe Tasks FaceLandmarker Teacher da san sang (478 -> 22 Landmarks)!
  ✓ Mô hình Thầy MediaPipe Face Mesh đã sẵn sàng dán nhãn sinh học!

[Bước 2/5] Chuẩn bị dữ liệu Full Mặt (Preprocessed Real Dataset) + In-Cabin Augmentation...
  ✓ Đã phát hiện tập dữ liệu tiền xử lý chuẩn: preprocessed_driver_dataset.npz
[DatasetLoader] [SUCCESS] Đã nạp 7651 mẫu thật từ: /content/training_tinyml/preprocessed_driver_dataset.npz
  • Train: 6894 | Val giữ-out: 757
[DatasetLoader] [REAL-DATA] Kích hoạt luồng huấn luyện Thực Tế (3-Way Balanced Sampling): 6894 mẫu train!
  • Mẫu nhắm mắt/microsleep thật (EAR < 0.20): 2940 mẫu
  • Mẫu ngáp/há miệng thật      (MAR >= 0.40): 1138 mẫu
  • Mẫu tỉnh táo/bình thường    (Attentive)   : 2816 mẫu
    [STATIC-EXPAND] Đang tiền-tính 45906 bản augment cân bằng 3 trạng thái (13771 nhắm mắt + 13771 ngáp + 18364 tỉnh táo, expand_factor=6)...
    [STATIC-EXPAND] Xong trong 66.4s (45906 mẫu: 13771 nhắm mắt / 13771 ngáp / 18364 tỉnh táo, RAM ~423MB, ĐÃ TRỘN TOÀN CỤC)
2026-09-12 14:55:53.277295: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1789224953.278775    3978 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
  ✓ [STATIC-EXPAND] Tiền-tính augment x6 xong -> epoch chỉ chạy GPU thuần (~15-25s)
  ✓ Mẫu huấn luyện (Augmented): 39021 | Mẫu kiểm chuẩn: 6885
  ✓ Kích hoạt Bounding Box Translation Jitter: triệt tiêu học vẹt tọa độ cố định.
  ✓ [REAL-VAL] Val giữ-out THẬT: 757 mẫu (NME mỗi epoch trên bản JITTER x2 = 1514 mẫu, tối đa 600)

[Bước 3/5] Xây dựng kiến trúc PFLD-Edge với Auxiliary 3D Pose Head...
  ✓ Kiến trúc PFLD-Edge: 266,135 tham số
  ✓ Tích hợp Inverted Residual Blocks (MBConv) + Nhánh ước lượng góc đầu 3D
  ✓ Hàm mất mát v2.3.0 [EAR-MẠNH / MAR-YẾU]: Wing(toạ độ) + EAR x20.0 + MAR x1.5 + LipGap x4.0 + MouthWidth x5.0 (mẫu số clamp, bỏ focal/detach)

[Bước 4/5] Bắt đầu huấn luyện qua 60 epochs...
2026-09-12 14:56:16.486479: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91900
  [Epoch 01/60] Step 609/609 (100%) | Loss:  51.12 (LM:  51.06)2026-09-12 14:57:18.597142: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2026-09-12 14:57:23.066372: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  1/60 [86.9s] - Train:  65.33 (LM:  65.27) | Val:  32.02 | NME: overall 14.98% (mắt 15.00% / miệng 15.15% / mũi-cằm 14.71%) | EAR_MAE 9.89% (nhắm->0.236) ⭐ (Best)
  [Epoch 02/60] Step 609/609 (100%) | Loss:  47.05 (LM:  47.01)2026-09-12 14:58:25.345274: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  2/60 [61.1s] - Train:  48.97 (LM:  48.92) | Val:  30.34 | NME: overall 14.19% (mắt 13.48% / miệng 15.38% / mũi-cằm 14.55%) | EAR_MAE 7.96% (nhắm->0.136) ⭐ (Best)
Epoch  3/60 [61.4s] - Train:  45.92 (LM:  45.88) | Val:  26.95 | NME: overall 12.77% (mắt 12.19% / miệng 13.63% / mũi-cằm 13.22%) | EAR_MAE 7.17% (nhắm->0.140) ⭐ (Best)
  [Epoch 04/60] Step 609/609 (100%) | Loss:  42.09 (LM:  42.06)2026-09-12 15:00:31.797936: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  4/60 [63.0s] - Train:  43.64 (LM:  43.60) | Val:  25.40 | NME: overall 11.84% (mắt 11.27% / miệng 12.57% / mũi-cằm 12.45%) | EAR_MAE 7.22% (nhắm->0.125) ⭐ (Best)
Epoch  5/60 [62.8s] - Train:  41.99 (LM:  41.95) | Val:  24.90 | NME: overall 11.50% (mắt 10.82% / miệng 12.25% / mũi-cằm 12.40%) | EAR_MAE 6.76% (nhắm->0.137) ⭐ (Best)
Epoch  6/60 [62.9s] - Train:  40.85 (LM:  40.81) | Val:  24.04 | NME: overall 11.02% (mắt 10.31% / miệng 11.77% / mũi-cằm 12.01%) | EAR_MAE 6.49% (nhắm->0.122) ⭐ (Best)
Epoch  7/60 [62.7s] - Train:  39.92 (LM:  39.88) | Val:  22.47 | NME: overall 10.66% (mắt 9.74% / miệng 11.70% / mũi-cằm 11.86%) | EAR_MAE 6.81% (nhắm->0.106) 
  [Epoch 08/60] Step 609/609 (100%) | Loss:  39.34 (LM:  39.32)2026-09-12 15:04:46.602503: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  8/60 [62.4s] - Train:  39.29 (LM:  39.25) | Val:  21.09 | NME: overall 10.48% (mắt 9.76% / miệng 11.09% / mũi-cằm 11.68%) | EAR_MAE 6.22% (nhắm->0.111) ⭐ (Best)
Epoch  9/60 [62.4s] - Train:  38.94 (LM:  38.92) | Val:  21.22 | NME: overall 10.18% (mắt 9.35% / miệng 10.93% / mũi-cằm 11.56%) | EAR_MAE 6.12% (nhắm->0.113) ⭐ (Best)
Epoch 10/60 [62.6s] - Train:  38.61 (LM:  38.59) | Val:  22.75 | NME: overall 10.71% (mắt 9.94% / miệng 11.42% / mũi-cằm 11.94%) | EAR_MAE 7.40% (nhắm->0.086) 
Epoch 11/60 [62.6s] - Train:  38.39 (LM:  38.36) | Val:  20.04 | NME: overall 10.16% (mắt 9.16% / miệng 11.17% / mũi-cằm 11.65%) | EAR_MAE 5.65% (nhắm->0.128) ⭐ (Best)
Epoch 12/60 [62.5s] - Train:  38.20 (LM:  38.18) | Val:  21.24 | NME: overall 10.34% (mắt 9.37% / miệng 11.41% / mũi-cằm 11.69%) | EAR_MAE 5.33% (nhắm->0.114) ⭐ (Best)
Epoch 13/60 [62.6s] - Train:  38.02 (LM:  37.99) | Val:  21.96 | NME: overall 10.26% (mắt 9.47% / miệng 10.99% / mũi-cằm 11.56%) | EAR_MAE 5.78% (nhắm->0.107) 
Epoch 14/60 [62.6s] - Train:  37.90 (LM:  37.87) | Val:  20.13 | NME: overall 9.98% (mắt 9.05% / miệng 10.87% / mũi-cằm 11.44%) | EAR_MAE 5.89% (nhắm->0.094) 
Epoch 15/60 [62.8s] - Train:  37.70 (LM:  37.68) | Val:  19.48 | NME: overall 10.08% (mắt 9.07% / miệng 11.12% / mũi-cằm 11.52%) | EAR_MAE 5.09% (nhắm->0.096) ⭐ (Best)
  [Epoch 16/60] Step 609/609 (100%) | Loss:  36.59 (LM:  36.58)2026-09-12 15:13:16.155408: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 16/60 [62.9s] - Train:  37.61 (LM:  37.59) | Val:  21.06 | NME: overall 10.00% (mắt 8.96% / miệng 11.02% / mũi-cằm 11.62%) | EAR_MAE 5.07% (nhắm->0.100) ⭐ (Best)
Epoch 17/60 [62.8s] - Train:  37.44 (LM:  37.42) | Val:  19.84 | NME: overall 9.82% (mắt 8.93% / miệng 10.62% / mũi-cằm 11.28%) | EAR_MAE 4.88% (nhắm->0.103) ⭐ (Best)
Epoch 18/60 [62.9s] - Train:  37.34 (LM:  37.31) | Val:  21.47 | NME: overall 9.88% (mắt 9.00% / miệng 10.69% / mũi-cằm 11.33%) | EAR_MAE 6.43% (nhắm->0.080) 
Epoch 19/60 [62.8s] - Train:  37.21 (LM:  37.19) | Val:  19.60 | NME: overall 10.11% (mắt 9.27% / miệng 10.90% / mũi-cằm 11.46%) | EAR_MAE 5.61% (nhắm->0.087) 
Epoch 20/60 [62.7s] - Train:  37.08 (LM:  37.06) | Val:  19.85 | NME: overall 9.63% (mắt 8.76% / miệng 10.35% / mũi-cằm 11.16%) | EAR_MAE 5.39% (nhắm->0.088) 
Epoch 21/60 [62.7s] - Train:  37.03 (LM:  37.00) | Val:  20.77 | NME: overall 9.93% (mắt 9.11% / miệng 10.63% / mũi-cằm 11.32%) | EAR_MAE 4.99% (nhắm->0.099) 
Epoch 22/60 [62.5s] - Train:  36.92 (LM:  36.90) | Val:  19.83 | NME: overall 9.62% (mắt 8.68% / miệng 10.48% / mũi-cằm 11.14%) | EAR_MAE 4.78% (nhắm->0.105) ⭐ (Best)
Epoch 23/60 [62.5s] - Train:  36.81 (LM:  36.79) | Val:  21.26 | NME: overall 9.81% (mắt 8.94% / miệng 10.65% / mũi-cằm 11.14%) | EAR_MAE 5.86% (nhắm->0.078) 
Epoch 24/60 [62.3s] - Train:  36.76 (LM:  36.73) | Val:  20.20 | NME: overall 9.67% (mắt 8.58% / miệng 10.77% / mũi-cằm 11.31%) | EAR_MAE 4.67% (nhắm->0.099) 
Epoch 25/60 [85.4s] - Train:  36.57 (LM:  36.54) | Val:  20.99 | NME: overall 9.85% (mắt 8.99% / miệng 10.67% / mũi-cằm 11.20%) | EAR_MAE 4.84% (nhắm->0.087) 
Epoch 26/60 [62.5s] - Train:  36.50 (LM:  36.48) | Val:  19.79 | NME: overall 9.77% (mắt 8.85% / miệng 10.66% / mũi-cằm 11.19%) | EAR_MAE 4.33% (nhắm->0.096) ⭐ (Best)
Epoch 27/60 [62.6s] - Train:  36.35 (LM:  36.33) | Val:  20.72 | NME: overall 9.88% (mắt 9.09% / miệng 10.59% / mũi-cằm 11.21%) | EAR_MAE 5.12% (nhắm->0.080) 
Epoch 28/60 [62.7s] - Train:  36.26 (LM:  36.24) | Val:  20.59 | NME: overall 9.82% (mắt 8.97% / miệng 10.62% / mũi-cằm 11.17%) | EAR_MAE 5.09% (nhắm->0.086) 
Epoch 29/60 [62.6s] - Train:  36.18 (LM:  36.16) | Val:  20.66 | NME: overall 9.83% (mắt 9.05% / miệng 10.50% / mũi-cằm 11.15%) | EAR_MAE 5.07% (nhắm->0.089) 
Epoch 30/60 [62.7s] - Train:  36.03 (LM:  36.01) | Val:  20.10 | NME: overall 9.68% (mắt 8.74% / miệng 10.58% / mũi-cằm 11.17%) | EAR_MAE 4.70% (nhắm->0.095) 
Epoch 31/60 [62.6s] - Train:  35.97 (LM:  35.95) | Val:  21.17 | NME: overall 9.63% (mắt 8.72% / miệng 10.49% / mũi-cằm 11.07%) | EAR_MAE 4.58% (nhắm->0.087) 
  [Epoch 32/60] Step 609/609 (100%) | Loss:  36.35 (LM:  36.33)2026-09-12 15:30:35.511689: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 32/60 [62.7s] - Train:  35.85 (LM:  35.82) | Val:  20.71 | NME: overall 9.59% (mắt 8.65% / miệng 10.48% / mũi-cằm 11.07%) | EAR_MAE 4.81% (nhắm->0.104) 
Epoch 33/60 [62.6s] - Train:  35.65 (LM:  35.63) | Val:  21.40 | NME: overall 9.85% (mắt 9.08% / miệng 10.51% / mũi-cằm 11.16%) | EAR_MAE 4.65% (nhắm->0.102) 
Epoch 34/60 [63.0s] - Train:  35.54 (LM:  35.52) | Val:  21.83 | NME: overall 9.55% (mắt 8.74% / miệng 10.28% / mũi-cằm 10.88%) | EAR_MAE 4.87% (nhắm->0.087) 
Epoch 35/60 [63.3s] - Train:  35.42 (LM:  35.39) | Val:  21.82 | NME: overall 9.75% (mắt 9.01% / miệng 10.37% / mũi-cằm 11.04%) | EAR_MAE 5.23% (nhắm->0.086) 
Epoch 36/60 [63.6s] - Train:  35.26 (LM:  35.24) | Val:  21.02 | NME: overall 9.80% (mắt 9.04% / miệng 10.46% / mũi-cằm 11.08%) | EAR_MAE 4.70% (nhắm->0.100) 
Epoch 37/60 [64.3s] - Train:  35.14 (LM:  35.12) | Val:  22.10 | NME: overall 9.66% (mắt 8.88% / miệng 10.34% / mũi-cằm 11.00%) | EAR_MAE 4.94% (nhắm->0.087) 
Epoch 38/60 [62.6s] - Train:  35.03 (LM:  35.01) | Val:  22.28 | NME: overall 9.78% (mắt 9.10% / miệng 10.36% / mũi-cằm 10.95%) | EAR_MAE 4.78% (nhắm->0.098) 
Epoch 39/60 [62.5s] - Train:  34.84 (LM:  34.82) | Val:  22.33 | NME: overall 9.79% (mắt 9.10% / miệng 10.38% / mũi-cằm 10.96%) | EAR_MAE 4.76% (nhắm->0.092) 
Epoch 40/60 [62.5s] - Train:  34.77 (LM:  34.75) | Val:  22.12 | NME: overall 9.67% (mắt 8.98% / miệng 10.25% / mũi-cằm 10.86%) | EAR_MAE 4.72% (nhắm->0.096) ⭐ (Best)
Epoch 41/60 [62.2s] - Train:  34.58 (LM:  34.56) | Val:  22.81 | NME: overall 9.94% (mắt 9.34% / miệng 10.40% / mũi-cằm 11.06%) | EAR_MAE 5.02% (nhắm->0.102) 
Epoch 42/60 [62.5s] - Train:  34.45 (LM:  34.43) | Val:  22.93 | NME: overall 9.96% (mắt 9.31% / miệng 10.50% / mũi-cằm 11.08%) | EAR_MAE 4.81% (nhắm->0.107) 
Epoch 43/60 [62.2s] - Train:  34.32 (LM:  34.30) | Val:  22.85 | NME: overall 9.73% (mắt 8.99% / miệng 10.43% / mũi-cằm 10.91%) | EAR_MAE 4.79% (nhắm->0.099) 
Epoch 44/60 [62.1s] - Train:  34.22 (LM:  34.20) | Val:  23.17 | NME: overall 9.98% (mắt 9.38% / miệng 10.47% / mũi-cằm 11.07%) | EAR_MAE 4.85% (nhắm->0.097) 
Epoch 45/60 [62.4s] - Train:  34.10 (LM:  34.08) | Val:  23.28 | NME: overall 9.99% (mắt 9.28% / miệng 10.63% / mũi-cằm 11.13%) | EAR_MAE 4.74% (nhắm->0.099) 
Epoch 46/60 [62.2s] - Train:  34.03 (LM:  34.01) | Val:  23.44 | NME: overall 10.15% (mắt 9.53% / miệng 10.72% / mũi-cằm 11.18%) | EAR_MAE 4.78% (nhắm->0.097) 
Epoch 47/60 [63.1s] - Train:  33.86 (LM:  33.84) | Val:  23.40 | NME: overall 10.10% (mắt 9.47% / miệng 10.66% / mũi-cằm 11.18%) | EAR_MAE 4.84% (nhắm->0.104) 
Epoch 48/60 [63.0s] - Train:  33.74 (LM:  33.72) | Val:  24.08 | NME: overall 10.20% (mắt 9.54% / miệng 10.75% / mũi-cằm 11.33%) | EAR_MAE 4.89% (nhắm->0.094) 
Epoch 49/60 [63.5s] - Train:  33.67 (LM:  33.65) | Val:  23.92 | NME: overall 10.15% (mắt 9.57% / miệng 10.61% / mũi-cằm 11.20%) | EAR_MAE 4.91% (nhắm->0.105) 
Epoch 50/60 [64.1s] - Train:  33.52 (LM:  33.50) | Val:  24.05 | NME: overall 10.15% (mắt 9.51% / miệng 10.68% / mũi-cằm 11.25%) | EAR_MAE 4.86% (nhắm->0.099) 
Epoch 51/60 [63.3s] - Train:  33.46 (LM:  33.44) | Val:  23.74 | NME: overall 10.08% (mắt 9.44% / miệng 10.61% / mũi-cằm 11.21%) | EAR_MAE 4.85% (nhắm->0.096) 
Epoch 52/60 [63.1s] - Train:  33.37 (LM:  33.35) | Val:  24.07 | NME: overall 10.12% (mắt 9.51% / miệng 10.62% / mũi-cằm 11.24%) | EAR_MAE 4.89% (nhắm->0.097) 
Epoch 53/60 [63.0s] - Train:  33.29 (LM:  33.27) | Val:  24.28 | NME: overall 10.20% (mắt 9.60% / miệng 10.69% / mũi-cằm 11.28%) | EAR_MAE 4.93% (nhắm->0.097) 
Epoch 54/60 [63.4s] - Train:  33.24 (LM:  33.22) | Val:  24.28 | NME: overall 10.17% (mắt 9.53% / miệng 10.69% / mũi-cằm 11.31%) | EAR_MAE 4.88% (nhắm->0.094) 
Epoch 55/60 [63.0s] - Train:  33.18 (LM:  33.16) | Val:  24.37 | NME: overall 10.15% (mắt 9.52% / miệng 10.66% / mũi-cằm 11.30%) | EAR_MAE 4.83% (nhắm->0.102) 
Epoch 56/60 [63.2s] - Train:  33.14 (LM:  33.12) | Val:  24.57 | NME: overall 10.26% (mắt 9.63% / miệng 10.77% / mũi-cằm 11.39%) | EAR_MAE 4.85% (nhắm->0.098) 
Epoch 57/60 [62.9s] - Train:  33.07 (LM:  33.05) | Val:  24.74 | NME: overall 10.26% (mắt 9.64% / miệng 10.76% / mũi-cằm 11.38%) | EAR_MAE 4.92% (nhắm->0.102) 
Epoch 58/60 [62.9s] - Train:  33.05 (LM:  33.04) | Val:  24.74 | NME: overall 10.26% (mắt 9.62% / miệng 10.78% / mũi-cằm 11.39%) | EAR_MAE 4.86% (nhắm->0.101) 
Epoch 59/60 [63.2s] - Train:  33.02 (LM:  33.00) | Val:  24.80 | NME: overall 10.30% (mắt 9.69% / miệng 10.79% / mũi-cằm 11.43%) | EAR_MAE 4.87% (nhắm->0.101) 
Epoch 60/60 [62.9s] - Train:  32.97 (LM:  32.95) | Val:  24.86 | NME: overall 10.27% (mắt 9.65% / miệng 10.78% / mũi-cằm 11.39%) | EAR_MAE 4.90% (nhắm->0.100) 

✅ Huấn luyện hoàn tất sau: 3868.4 giây!, Best Composite Score (max(NME, Miệng)+EAR_MAE): 14.97%
  ✓ Đã nạp lại trọng số tốt nhất (Best Weights).

===========================================================================
📊 BÁO CÁO NME & SAI SỐ ĐỊNH VỊ CUỐI CÙNG (Val giữ-out người thật 100%):
===========================================================================
1. NME CANONICAL (Đo trên khuôn mặt chuẩn, không rung lắc biến dạng):
   • NME Tổng thể             : 6.65% (✅ ĐẠT CHUẨN TỐT (<8%))
   • Sai số pixel trung bình : 2.67px / 96px (Toàn bộ 22 điểm mốc)
   • Sai số nhóm Mắt         : 6.07% (2.43px)
   • Sai số nhóm Miệng       : 7.35% (2.96px)
   • Sai số nhóm Mũi / Cằm   : 7.37% (2.96px)

2. NME JITTER (Khả năng định vị pixel khi bị rung lắc dịch chuyển/xoay):
   • NME Jitter Tổng thể      : 9.55% (✅ ĐẠT CHUẨN KHÁ (<12%))
   • Sai số pixel trung bình : 3.86px / 96px
   • Sai số nhóm Mắt         : 8.60% (3.47px)
   • Sai số nhóm Miệng       : 10.49% (4.25px)
   • Sai số nhóm Mũi / Cằm   : 10.97% (4.44px)
   • Mẫu biên ngoại lệ tệ nhất: #254 (P21 lệch tối đa 39.7px - chỉ là 1 mẫu cá biệt)

3. ĐÁNH GIÁ ĐỊNH VỊ (Tỉ lệ Jitter/Canonical):
   • Tỉ lệ Jitter / Canon     : 1.43x (✅ ĐẠT CHUẨN (<2.0x): Mô hình định vị pixel thật, không học vẹt!)

4. CỔNG NHÓM MIỆNG (chống sập khóe miệng P12/P13):
   • NME Miệng Jitter         : 10.49% (✅ ĐẠT (<12%))

5. CỔNG PHẢN HỒI MẮT (chống nén dải EAR):
   • EAR MAE (jitter)          : 4.73% (ngưỡng <6%)
   • EAR đoán khi mắt NHẮM thật: 0.093 (ngưỡng <0.16, càng thấp càng nhạy)
   • Đánh giá                  : ✅ ĐẠT — mắt phản hồi nhắm/mở tốt
===========================================================================
   👉 Kiểm tra lại sau khi tải về bằng: python evaluation/eval_nme_holdout.py

  ✓ Đã lưu biểu đồ tiến trình: training_loss.png

[Bước 5/5] Cắt bỏ Auxiliary Head và Lượng tử hóa Mixed-Precision cho ESP32-S3...
  ✓ Mô hình xuất xưởng (Landmark-Only): 262,804 tham số
WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
I0000 00:00:1789228828.644048    3978 devices.cc:67] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 1
I0000 00:00:1789228828.644249    3978 single_machine.cc:376] Starting new session
I0000 00:00:1789228828.652127    3978 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/convert.py:863: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn(
W0000 00:00:1789228829.152843    3978 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1789228829.152903    3978 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
2026-09-12 16:00:29.349201: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:4150] Estimated count of arithmetic ops: 50.649 M  ops, equivalently 25.324 M  MACs
fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: FLOAT32
2026-09-12 16:00:35.103345: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:4150] Estimated count of arithmetic ops: 50.649 M  ops, equivalently 25.324 M  MACs
2026-09-12 16:00:35.103418: W tensorflow/compiler/mlir/lite/flatbuffer_export.cc:3705] Skipping runtime version metadata in the model. This will be generated by the exporter.
  ✓ Kích thước mô hình Mixed-Precision: 341,736 bytes (333.7 KB)
/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
[Export] Đã xuất file C Header thành công tại: /content/colab_output/tinydriver_model_data.h
  ✓ Đã tạo file C Header: tinydriver_model_data.h (16-byte aligned, Output Float32: True)
  ✓ Đã đóng gói val_holdout.npz (757 mẫu giữ-out) vào gói

===========================================================================
🎉 ĐÓNG GÓI THÀNH CÔNG: tinydriver_esp32_package.zip (6423.6 KB)
===========================================================================
📥 Đang gửi lệnh tự động tải file tinydriver_esp32_package.zip về máy tính của bạn...
ℹ️ File kết quả đã sẵn sàng tại: /content/tinydriver_esp32_package.zip

👉 BƯỚC TIẾP THEO TRÊN MÁY TÍNH CỦA BẠN:
  1. Chạy lệnh nạp model tự động:
     python tools/project_manager.py --deploy-model tinydriver_esp32_package.zip
  2. Chạy thử AI với Webcam trên Laptop:
     python host_laptop/local_model_tester.py --cam 0
===========================================================================
I0000 00:00:1789228839.594500   22417 migration_state_tracking.cc:24] Migration not enabled - not starting notification watcher.
