"""
Configuration parameters for TinyDriver-LandmarkNet training and export.
Standardized for ESP32-S3 N16R8 Edge AI ADAS System.
"""

# ==============================================================================
# 1. Image & Input Specifications (Isomorphic 1:1)
# ==============================================================================
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_CHANNELS = 1          # Grayscale (saves memory & bandwidth on ESP32-S3)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Bounding box expansion ratio to ensure whole face + context is captured (Standardized with FaceTracker & ESP32)
BBOX_EXPANSION_RATIO = 1.35


# ==============================================================================
# 2. Facial Landmark Selection (22 Points selected from 68 standard iBUG points)
# ==============================================================================
NUM_LANDMARKS = 22
OUTPUT_DIMS = NUM_LANDMARKS * 2  # (x, y) normalized to [0.0, 1.0]

# Mapping from standard 68-point facial landmarks (0-indexed):
# Left eye (6 points): outer corner, 2 upper lid points, inner corner, 2 lower lid points
LEFT_EYE_68 = [36, 37, 38, 39, 40, 41]

# Right eye (6 points): inner corner, 2 upper lid points, outer corner, 2 lower lid points
RIGHT_EYE_68 = [42, 43, 44, 45, 46, 47]

# Mouth (6 points): left corner, right corner, top outer lip, bottom outer lip, top inner, bottom inner
MOUTH_68 = [48, 54, 51, 57, 62, 66]

# Nose and Chin (4 points): Nasion (bridge), Nose tip, Left/Right subnasale, Chin bottom
NOSE_CHIN_68 = [27, 30, 33, 8]

# Master index list of 22 points extracted from 68 points:
IBUG_68_TO_22_INDICES = LEFT_EYE_68 + RIGHT_EYE_68 + MOUTH_68 + NOSE_CHIN_68

# Semantic groups inside the 22-point array:
# Index 0..5   : Left Eye
# Index 6..11  : Right Eye
# Index 12..17 : Mouth
# Index 18..21 : Nose & Chin
LEFT_EYE_22 = list(range(0, 6))
RIGHT_EYE_22 = list(range(6, 12))
MOUTH_22 = list(range(12, 18))
NOSE_CHIN_22 = list(range(18, 22))

# ==============================================================================
# 3. 3D Model Reference Points for Perspective-n-Point (PnP) Head Pose
# (Standard Anthropometric 3D Human Face Model in millimeters)
# ==============================================================================
# Point correspondence with 22 landmarks:
# Nose tip: Index 19
# Chin: Index 21
# Left eye outer corner: Index 0
# Right eye outer corner: Index 9 (Note: right outer corner is index 9 in our 22-pt list)
# Left mouth corner: Index 12
# Right mouth corner: Index 13
PNP_LANDMARK_INDICES = [19, 21, 0, 9, 12, 13]

# [v2.4.1 - RESTORE] 6 điểm PnP gốc (chóp mũi, cằm, 2 khóe mắt, 2 khóe miệng).
FACE_3D_MODEL_POINTS = [
    [0.0,    0.0,   0.0],     # P19 Chóp mũi
    [0.0,   65.0, -35.0],     # P21 Cằm
    [-43.0, -32.0, -30.0],    # P0  Khóe mắt trái ngoài
    [43.0,  -32.0, -30.0],    # P9  Khóe mắt phải ngoài
    [-30.0,  30.0, -20.0],    # P12 Khóe miệng trái
    [30.0,   30.0, -20.0]     # P13 Khóe miệng phải
]

# ==============================================================================
# 4. Training Hyperparameters (Optimized for Colab T4 GPU)
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Wing Loss & Biometric Focal Loss Parameters
WING_W = 10.0
WING_EPSILON = 2.0

# [v2.3.0 - CÂN BẰNG LẠI] Toạ độ Wing Loss là nền tảng, nhưng MẮT cần loss tỉ lệ EAR
# MẠNH để phản hồi nhắm/mở (bản v2.2.0 để EAR=1.5 làm mắt bị nén dải, không nhắm được).
#   - EAR: mạnh + detach width (bảo vệ khóe mắt bằng coordinate loss).
#   - MAR: yếu + clamp + neo cứng bề rộng (chống sập khóe miệng như bản cũ).
EAR_LOSS_WEIGHT = 20.0
MAR_LOSS_WEIGHT = 1.5
LIP_GAP_WEIGHT = 4.0
MOUTH_WIDTH_WEIGHT = 5.0    # Neo cứng P12-P13, chống sập khóe miệng
FOCAL_EAR_GAMMA = 1.0       # Giữ lại để tương thích cấu hình cũ (không còn dùng)
FOCAL_MAR_GAMMA = 1.0

# ==============================================================================
# 5. ADAS Thresholds (Aligned with firmware_esp32)
# ==============================================================================
DEFAULT_EAR_THRESHOLD = 0.22
MICROSLEEP_TIME_SEC = 1.5
SLOW_BLINK_TIME_SEC = 0.5

# [SYNC 2025] MAR dùng công thức (h_outer + h_inner) / (2*w) - đồng bộ wing_loss.py,
# local_model_tester.py và firmware adas_controller.cpp. Ngưỡng khớp project_config.json.
DEFAULT_MAR_THRESHOLD = 0.45
YAWN_DURATION_SEC = 1.5
YAWN_FATIGUE_COUNT = 3
YAWN_WINDOW_SEC = 180.0

HEAD_YAW_THRESHOLD_DEG = 30.0
DISTRACTION_TIME_SEC = 3.0
