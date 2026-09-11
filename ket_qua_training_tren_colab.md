   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.9/37.9 MB 60.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 137.4/137.4 kB 10.7 MB/s eta 0:00:00
training_package.zip
training_package.zip(application/x-zip-compressed) - 87940860 bytes, last modified: 11/9/2026 - 100% done
Saving training_package.zip to training_package.zip
2026-09-11 10:09:28.690047: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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
W0000 00:00:1789121374.022597    2285 face_landmarker_graph.cc:180] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1789121374.036844    2289 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1789121374.064687    2290 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
[Teacher] MediaPipe Tasks FaceLandmarker Teacher da san sang (478 -> 22 Landmarks)!
  ✓ Mô hình Thầy MediaPipe Face Mesh đã sẵn sàng dán nhãn sinh học!

[Bước 2/5] Chuẩn bị dữ liệu Full Mặt (Preprocessed Real Dataset) + In-Cabin Augmentation...
  ✓ Đã phát hiện tập dữ liệu tiền xử lý chuẩn: preprocessed_driver_dataset.npz
[DatasetLoader] [SUCCESS] Đã nạp 11174 mẫu thật từ: /content/training_tinyml/preprocessed_driver_dataset.npz
  • Train: 10305 | Val giữ-out: 869
[DatasetLoader] [REAL-DATA] Kích hoạt luồng huấn luyện Thực Tế (3-Way Balanced Sampling): 10305 mẫu train!
  • Mẫu nhắm mắt/microsleep thật (EAR < 0.20): 3202 mẫu
  • Mẫu ngáp/há miệng thật      (MAR >= 0.40): 1437 mẫu
  • Mẫu tỉnh táo/bình thường    (Attentive)   : 5666 mẫu
    [STATIC-EXPAND] Đang tiền-tính 67044 bản augment cân bằng 3 trạng thái (20113 nhắm mắt + 20113 ngáp + 26818 tỉnh táo, expand_factor=6)...
    [STATIC-EXPAND] Xong trong 90.6s (67044 mẫu: 20113 nhắm mắt / 20113 ngáp / 26818 tỉnh táo, RAM ~618MB, ĐÃ TRỘN TOÀN CỤC)
2026-09-11 10:11:06.808513: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1789121466.809997    2236 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
2026-09-11 10:11:06.815587: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 617877504 exceeds 10% of free system memory.
2026-09-11 10:11:07.379330: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 617877504 exceeds 10% of free system memory.
  ✓ [STATIC-EXPAND] Tiền-tính augment x6 xong -> epoch chỉ chạy GPU thuần (~15-25s)
  ✓ Mẫu huấn luyện (Augmented): 56988 | Mẫu kiểm chuẩn: 10056
  ✓ Kích hoạt Bounding Box Translation Jitter: triệt tiêu học vẹt tọa độ cố định.
  ✓ [REAL-VAL] Val giữ-out THẬT: 869 mẫu (NME mỗi epoch trên bản JITTER x2 = 1738 mẫu, tối đa 600)

[Bước 3/5] Xây dựng kiến trúc PFLD-Edge với Auxiliary 3D Pose Head...
  ✓ Kiến trúc PFLD-Edge: 266,135 tham số
  ✓ Tích hợp Inverted Residual Blocks (MBConv) + Nhánh ước lượng góc đầu 3D
  ✓ Hàm mất mát: Detached Adaptive Biometric Wing Loss (EAR x25.0, MAR x20.0 [Detached w_m], LipGap x8.0, MouthWidth x2.0, Pose x1.5)

[Bước 4/5] Bắt đầu huấn luyện qua 60 epochs...
2026-09-11 10:11:10.014632: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 617877504 exceeds 10% of free system memory.
2026-09-11 10:11:29.197551: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91900
  [Epoch 01/60] Step 890/890 (100%) | Loss: 149.29 (LM: 149.25)2026-09-11 10:12:53.990656: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
2026-09-11 10:13:00.346494: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  1/60 [110.3s] - Train: 178.82 (LM: 178.75) | Val: 145.83 | Real-Val NME: 86.06% (miệng 126.27%) ⭐ (Best)
2026-09-11 10:13:01.422482: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 617877504 exceeds 10% of free system memory.
  [Epoch 02/60] Step 890/890 (100%) | Loss: 132.00 (LM: 131.95)2026-09-11 10:14:27.507592: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  2/60 [86.1s] - Train: 142.71 (LM: 142.65) | Val: 124.72 | Real-Val NME: 72.69% (miệng 104.62%) ⭐ (Best)
2026-09-11 10:14:28.529215: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 617877504 exceeds 10% of free system memory.
Epoch  3/60 [86.0s] - Train: 120.60 (LM: 120.54) | Val:  99.04 | Real-Val NME: 55.60% (miệng 69.37%) ⭐ (Best)
  [Epoch 04/60] Step 890/890 (100%) | Loss:  87.68 (LM:  87.64)2026-09-11 10:17:23.595288: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  4/60 [88.0s] - Train:  97.95 (LM:  97.90) | Val:  78.05 | Real-Val NME: 40.73% (miệng 26.01%) ⭐ (Best)
Epoch  5/60 [88.1s] - Train:  84.20 (LM:  84.15) | Val:  68.62 | Real-Val NME: 34.95% (miệng 16.08%) ⭐ (Best)
Epoch  6/60 [88.3s] - Train:  77.71 (LM:  77.65) | Val:  61.84 | Real-Val NME: 30.50% (miệng 16.11%) ⭐ (Best)
Epoch  7/60 [87.9s] - Train:  70.06 (LM:  70.01) | Val:  53.76 | Real-Val NME: 24.78% (miệng 16.09%) ⭐ (Best)
  [Epoch 08/60] Step 890/890 (100%) | Loss:  58.76 (LM:  58.72)2026-09-11 10:24:18.936672: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch  8/60 [147.1s] - Train:  63.11 (LM:  63.06) | Val:  45.86 | Real-Val NME: 17.60% (miệng 16.12%) ⭐ (Best)
Epoch  9/60 [87.8s] - Train:  57.01 (LM:  56.97) | Val:  41.56 | Real-Val NME: 15.11% (miệng 16.10%) ⭐ (Best)
Epoch 10/60 [88.1s] - Train:  56.03 (LM:  55.99) | Val:  41.73 | Real-Val NME: 15.14% (miệng 16.15%) 
Epoch 11/60 [87.7s] - Train:  56.01 (LM:  55.97) | Val:  41.48 | Real-Val NME: 15.11% (miệng 16.13%) ⭐ (Best)
Epoch 12/60 [87.9s] - Train:  56.03 (LM:  55.99) | Val:  41.53 | Real-Val NME: 15.10% (miệng 16.11%) ⭐ (Best)
Epoch 13/60 [146.8s] - Train:  56.04 (LM:  56.00) | Val:  41.90 | Real-Val NME: 15.10% (miệng 16.10%) 
Epoch 14/60 [87.6s] - Train:  56.05 (LM:  56.01) | Val:  41.79 | Real-Val NME: 15.18% (miệng 16.18%) 
Epoch 15/60 [87.5s] - Train:  56.03 (LM:  55.99) | Val:  41.66 | Real-Val NME: 15.16% (miệng 16.11%) 
  [Epoch 16/60] Step 890/890 (100%) | Loss:  57.38 (LM:  57.34)2026-09-11 10:37:07.641671: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 16/60 [87.7s] - Train:  56.06 (LM:  56.03) | Val:  40.80 | Real-Val NME: 15.12% (miệng 16.12%) 
Epoch 17/60 [87.2s] - Train:  56.03 (LM:  56.00) | Val:  41.49 | Real-Val NME: 15.09% (miệng 16.12%) ⭐ (Best)
Epoch 18/60 [87.2s] - Train:  56.04 (LM:  56.01) | Val:  41.25 | Real-Val NME: 15.07% (miệng 16.05%) ⭐ (Best)
Epoch 19/60 [89.2s] - Train:  56.03 (LM:  56.00) | Val:  41.60 | Real-Val NME: 15.13% (miệng 16.13%) 
Epoch 20/60 [89.9s] - Train:  55.99 (LM:  55.96) | Val:  41.53 | Real-Val NME: 15.11% (miệng 16.12%) 
Epoch 21/60 [89.5s] - Train:  55.99 (LM:  55.96) | Val:  41.51 | Real-Val NME: 15.15% (miệng 16.13%) 
Epoch 22/60 [87.5s] - Train:  55.99 (LM:  55.97) | Val:  41.02 | Real-Val NME: 15.18% (miệng 16.12%) 
Epoch 23/60 [87.6s] - Train:  55.97 (LM:  55.95) | Val:  41.53 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 24/60 [87.4s] - Train:  55.98 (LM:  55.96) | Val:  41.99 | Real-Val NME: 15.09% (miệng 16.11%) 
Epoch 25/60 [87.4s] - Train:  56.02 (LM:  56.00) | Val:  41.65 | Real-Val NME: 15.18% (miệng 16.13%) 
Epoch 26/60 [87.5s] - Train:  56.01 (LM:  55.99) | Val:  41.61 | Real-Val NME: 15.14% (miệng 16.11%) 
Epoch 27/60 [87.3s] - Train:  56.00 (LM:  55.98) | Val:  41.63 | Real-Val NME: 15.21% (miệng 16.14%) 
Epoch 28/60 [87.8s] - Train:  56.04 (LM:  56.02) | Val:  41.43 | Real-Val NME: 15.09% (miệng 16.10%) 
Epoch 29/60 [87.8s] - Train:  55.98 (LM:  55.96) | Val:  41.52 | Real-Val NME: 15.13% (miệng 16.10%) 
Epoch 30/60 [147.3s] - Train:  55.99 (LM:  55.98) | Val:  42.15 | Real-Val NME: 15.11% (miệng 16.11%) 
Epoch 31/60 [88.8s] - Train:  55.99 (LM:  55.98) | Val:  41.55 | Real-Val NME: 15.09% (miệng 16.10%) 
  [Epoch 32/60] Step 890/890 (100%) | Loss:  55.72 (LM:  55.70)2026-09-11 11:01:50.829419: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
Epoch 32/60 [89.5s] - Train:  55.97 (LM:  55.96) | Val:  42.28 | Real-Val NME: 15.18% (miệng 16.13%) 
Epoch 33/60 [90.0s] - Train:  56.03 (LM:  56.02) | Val:  40.99 | Real-Val NME: 15.16% (miệng 16.13%) 
Epoch 34/60 [89.3s] - Train:  56.00 (LM:  55.98) | Val:  41.62 | Real-Val NME: 15.16% (miệng 16.12%) 
Epoch 35/60 [88.0s] - Train:  55.98 (LM:  55.95) | Val:  41.47 | Real-Val NME: 15.15% (miệng 16.13%) 
Epoch 36/60 [146.8s] - Train:  55.97 (LM:  55.95) | Val:  41.53 | Real-Val NME: 15.15% (miệng 16.15%) 
Epoch 37/60 [88.6s] - Train:  55.97 (LM:  55.96) | Val:  41.53 | Real-Val NME: 15.13% (miệng 16.11%) 
Epoch 38/60 [87.8s] - Train:  56.04 (LM:  56.02) | Val:  41.58 | Real-Val NME: 15.15% (miệng 16.14%) 
Epoch 39/60 [87.9s] - Train:  55.99 (LM:  55.97) | Val:  41.39 | Real-Val NME: 15.11% (miệng 16.10%) 
Epoch 40/60 [87.9s] - Train:  55.99 (LM:  55.97) | Val:  41.58 | Real-Val NME: 15.13% (miệng 16.11%) 
Epoch 41/60 [88.2s] - Train:  55.98 (LM:  55.97) | Val:  41.31 | Real-Val NME: 15.09% (miệng 16.08%) 
Epoch 42/60 [88.0s] - Train:  56.01 (LM:  56.00) | Val:  41.51 | Real-Val NME: 15.14% (miệng 16.11%) 
Epoch 43/60 [88.1s] - Train:  56.02 (LM:  56.01) | Val:  41.44 | Real-Val NME: 15.11% (miệng 16.09%) 
Epoch 44/60 [146.8s] - Train:  55.98 (LM:  55.97) | Val:  41.40 | Real-Val NME: 15.12% (miệng 16.09%) 
Epoch 45/60 [88.5s] - Train:  56.00 (LM:  55.99) | Val:  41.42 | Real-Val NME: 15.14% (miệng 16.11%) 
Epoch 46/60 [89.6s] - Train:  56.00 (LM:  55.99) | Val:  41.97 | Real-Val NME: 15.09% (miệng 16.08%) 
Epoch 47/60 [88.2s] - Train:  55.98 (LM:  55.97) | Val:  41.46 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 48/60 [88.5s] - Train:  55.96 (LM:  55.95) | Val:  40.31 | Real-Val NME: 15.12% (miệng 16.09%) 
Epoch 49/60 [89.4s] - Train:  56.01 (LM:  55.99) | Val:  41.44 | Real-Val NME: 15.13% (miệng 16.10%) 
Epoch 50/60 [89.2s] - Train:  56.00 (LM:  55.99) | Val:  42.12 | Real-Val NME: 15.14% (miệng 16.10%) 
Epoch 51/60 [90.2s] - Train:  56.02 (LM:  56.01) | Val:  41.41 | Real-Val NME: 15.12% (miệng 16.09%) 
Epoch 52/60 [89.0s] - Train:  56.01 (LM:  56.00) | Val:  41.55 | Real-Val NME: 15.13% (miệng 16.08%) 
Epoch 53/60 [146.7s] - Train:  55.98 (LM:  55.97) | Val:  41.42 | Real-Val NME: 15.14% (miệng 16.09%) 
Epoch 54/60 [88.3s] - Train:  56.00 (LM:  55.99) | Val:  40.86 | Real-Val NME: 15.12% (miệng 16.10%) 
Epoch 55/60 [87.4s] - Train:  55.99 (LM:  55.98) | Val:  41.39 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 56/60 [87.2s] - Train:  55.99 (LM:  55.98) | Val:  41.39 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 57/60 [86.8s] - Train:  56.01 (LM:  56.00) | Val:  41.40 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 58/60 [94.0s] - Train:  56.02 (LM:  56.01) | Val:  42.05 | Real-Val NME: 15.13% (miệng 16.08%) 
Epoch 59/60 [87.7s] - Train:  56.03 (LM:  56.03) | Val:  41.40 | Real-Val NME: 15.13% (miệng 16.09%) 
Epoch 60/60 [86.9s] - Train:  56.01 (LM:  56.00) | Val:  41.41 | Real-Val NME: 15.13% (miệng 16.09%) 

✅ Huấn luyện hoàn tất sau: 5720.4 giây!, Best Real-Val NME: 15.07%
  ✓ Đã nạp lại trọng số tốt nhất (Best Weights).

===========================================================================
📊 BÁO CÁO NME & SAI SỐ ĐỊNH VỊ CUỐI CÙNG (Val giữ-out người thật 100%):
===========================================================================
1. NME CANONICAL (Đo trên khuôn mặt chuẩn, không rung lắc biến dạng):
   • NME Tổng thể             : 10.68% (⚠️ CẦN CẢI THIỆN (>=8%))
   • Sai số pixel trung bình : 4.38px / 96px (Toàn bộ 22 điểm mốc)
   • Sai số nhóm Mắt         : 9.90% (4.01px)
   • Sai số nhóm Miệng       : 12.32% (5.11px)
   • Sai số nhóm Mũi / Cằm   : 10.57% (4.38px)

2. NME JITTER (Khả năng định vị pixel khi bị rung lắc dịch chuyển/xoay):
   • NME Jitter Tổng thể      : 14.62% (⚠️ CẦN THÊM DỮ LIỆU (>=12%))
   • Sai số pixel trung bình : 6.06px / 96px
   • Sai số nhóm Mắt         : 13.99% (5.76px)
   • Sai số nhóm Miệng       : 15.79% (6.59px)
   • Sai số nhóm Mũi / Cằm   : 14.76% (6.15px)
   • Mẫu biên ngoại lệ tệ nhất: #1562 (P21 lệch tối đa 37.3px - chỉ là 1 mẫu cá biệt)

3. ĐÁNH GIÁ ĐỊNH VỊ (Tỉ lệ Jitter/Canonical):
   • Tỉ lệ Jitter / Canon     : 1.37x (✅ ĐẠT CHUẨN (<2.0x): Mô hình định vị pixel thật, không học vẹt!)
===========================================================================
   👉 Kiểm tra lại sau khi tải về bằng: python evaluation/eval_nme_holdout.py

  ✓ Đã lưu biểu đồ tiến trình: training_loss.png

[Bước 5/5] Cắt bỏ Auxiliary Head và Lượng tử hóa Mixed-Precision cho ESP32-S3...
  ✓ Mô hình xuất xưởng (Landmark-Only): 262,804 tham số
WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
I0000 00:00:1789127195.301557    2236 devices.cc:67] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 1
I0000 00:00:1789127195.301739    2236 single_machine.cc:376] Starting new session
I0000 00:00:1789127195.309637    2236 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13757 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5
/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/convert.py:863: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn(
W0000 00:00:1789127195.771489    2236 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1789127195.771534    2236 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
2026-09-11 11:46:35.981995: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:4150] Estimated count of arithmetic ops: 50.649 M  ops, equivalently 25.324 M  MACs
fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: FLOAT32
2026-09-11 11:46:40.715357: I tensorflow/compiler/mlir/lite/flatbuffer_export.cc:4150] Estimated count of arithmetic ops: 50.649 M  ops, equivalently 25.324 M  MACs
2026-09-11 11:46:40.715410: W tensorflow/compiler/mlir/lite/flatbuffer_export.cc:3705] Skipping runtime version metadata in the model. This will be generated by the exporter.
  ✓ Kích thước mô hình Mixed-Precision: 341,736 bytes (333.7 KB)
/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
[Export] Đã xuất file C Header thành công tại: /content/colab_output/tinydriver_model_data.h
  ✓ Đã tạo file C Header: tinydriver_model_data.h (16-byte aligned, Output Float32: True)
  ✓ Đã đóng gói val_holdout.npz (869 mẫu giữ-out) vào gói

===========================================================================
🎉 ĐÓNG GÓI THÀNH CÔNG: tinydriver_esp32_package.zip (7276.7 KB)
===========================================================================
📥 Đang gửi lệnh tự động tải file tinydriver_esp32_package.zip về máy tính của bạn...
ℹ️ File kết quả đã sẵn sàng tại: /content/tinydriver_esp32_package.zip

👉 BƯỚC TIẾP THEO TRÊN MÁY TÍNH CỦA BẠN:
  1. Chạy lệnh nạp model tự động:
     python tools/project_manager.py --deploy-model tinydriver_esp32_package.zip
  2. Chạy thử AI với Webcam trên Laptop:
     python host_laptop/local_model_tester.py --cam 0
===========================================================================
I0000 00:00:1789127205.025282    2287 migration_state_tracking.cc:24] Migration not enabled - not starting notification watcher.
