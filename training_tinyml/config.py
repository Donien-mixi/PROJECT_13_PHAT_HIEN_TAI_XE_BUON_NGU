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

FACE_3D_MODEL_POINTS = [
    [0.0,    0.0,   0.0],     # Nose tip
    [0.0,  -65.0, -35.0],     # Chin
    [-43.0, 32.0, -30.0],     # Left eye outer corner
    [43.0,  32.0, -30.0],     # Right eye outer corner
    [-30.0,-30.0, -20.0],     # Left mouth corner
    [30.0, -30.0, -20.0]      # Right mouth corner
]

# ==============================================================================
# 4. Training Hyperparameters (Optimized for Colab T4 GPU)
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Wing Loss Parameters (Feng et al., CVPR 2018)
WING_W = 10.0
WING_EPSILON = 2.0

# ==============================================================================
# 5. ADAS Thresholds (Aligned with firmware_esp32)
# ==============================================================================
DEFAULT_EAR_THRESHOLD = 0.22
MICROSLEEP_TIME_SEC = 1.5
SLOW_BLINK_TIME_SEC = 0.5

DEFAULT_MAR_THRESHOLD = 0.50
YAWN_DURATION_SEC = 1.5
YAWN_FATIGUE_COUNT = 3
YAWN_WINDOW_SEC = 180.0

HEAD_YAW_THRESHOLD_DEG = 30.0
DISTRACTION_TIME_SEC = 3.0
