"""
Isomorphic Preprocessing & Landmark Transformation.
Guarantees 100% aspect ratio preservation (scale_x == scale_y)
from training dataset to real-time ESP32-S3 inference.
"""

import cv2
import numpy as np
try:
    from config import IMAGE_WIDTH, IMAGE_HEIGHT, BBOX_EXPANSION_RATIO
except (ImportError, ValueError):
    from .config import IMAGE_WIDTH, IMAGE_HEIGHT, BBOX_EXPANSION_RATIO

def square_crop_and_resize(image, bbox, landmarks=None):
    """
    Expands a bounding box into a symmetric square (1:1), crops the region with padding
    if needed, and resizes to (IMAGE_WIDTH, IMAGE_HEIGHT) Grayscale.

    Args:
        image: np.ndarray (H, W, 3) or (H, W)
        bbox: tuple or list (xmin, ymin, xmax, ymax) in absolute pixel coordinates
        landmarks: np.ndarray of shape (N, 2) in absolute pixel coordinates (optional)

    Returns:
        crop_resized: np.ndarray of shape (IMAGE_HEIGHT, IMAGE_WIDTH, 1), normalized to [0, 255] uint8
        norm_landmarks: np.ndarray of shape (N, 2), normalized to [0.0, 1.0] (if landmarks is provided)
        transform_meta: dict containing (x0, y0, square_size) for inverse mapping
    """
    h_img, w_img = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # 1. Compute center and bounding box dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin
    cx = xmin + box_w / 2.0
    cy = ymin + box_h / 2.0

    # 2. Expand into a perfect square
    square_size = int(max(box_w, box_h) * BBOX_EXPANSION_RATIO)
    # Ensure even size
    if square_size % 2 != 0:
        square_size += 1

    x0 = int(round(cx - square_size / 2.0))
    y0 = int(round(cy - square_size / 2.0))
    x1 = x0 + square_size
    y1 = y0 + square_size

    # 3. Handle borders with zero-padding (reflection or black padding)
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w_img)
    pad_bottom = max(0, y1 - h_img)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(image.shape) == 3:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            padded_img = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=0
            )
        # Shift coordinate system to padded image
        crop_x0 = x0 + pad_left
        crop_y0 = y0 + pad_top
        crop = padded_img[crop_y0:crop_y0 + square_size, crop_x0:crop_x0 + square_size]
    else:
        crop = image[y0:y1, x0:x1]

    # 4. Convert to Grayscale if image is BGR
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop

    # 5. Isomorphic Bilinear Resize to Target Dimensions
    crop_resized = cv2.resize(crop_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    crop_resized = np.expand_dims(crop_resized, axis=-1)

    # 6. Normalize landmarks to [0.0, 1.0] if provided
    norm_landmarks = None
    if landmarks is not None:
        landmarks = np.array(landmarks, dtype=np.float32)
        norm_landmarks = np.zeros_like(landmarks)
        # Formula: x_norm = (x - x0) / square_size
        norm_landmarks[:, 0] = (landmarks[:, 0] - x0) / float(square_size)
        norm_landmarks[:, 1] = (landmarks[:, 1] - y0) / float(square_size)

    transform_meta = {
        'x0': x0,
        'y0': y0,
        'square_size': square_size
    }

    return crop_resized, norm_landmarks, transform_meta


def denormalize_landmarks(norm_landmarks, transform_meta):
    """
    Inverse mapping: Converts normalized landmarks [0.0, 1.0] back to original frame coordinates.

    Args:
        norm_landmarks: np.ndarray of shape (N, 2) in [0.0, 1.0]
        transform_meta: dict with 'x0', 'y0', 'square_size'

    Returns:
        abs_landmarks: np.ndarray of shape (N, 2) in original pixel coordinates
    """
    x0 = transform_meta['x0']
    y0 = transform_meta['y0']
    square_size = transform_meta['square_size']

    abs_landmarks = np.zeros_like(norm_landmarks, dtype=np.float32)
    abs_landmarks[:, 0] = norm_landmarks[:, 0] * square_size + x0
    abs_landmarks[:, 1] = norm_landmarks[:, 1] * square_size + y0

    return abs_landmarks


def center_crop_webcam_frame(frame, target_size=240):
    """
    Center-crops a webcam frame (e.g. 640x480 or 1280x720) to a square (1:1)
    and resizes it to target_size for low-bandwidth streaming.

    Args:
        frame: np.ndarray (H, W, 3) from cv2.VideoCapture
        target_size: int (e.g. 240)

    Returns:
        square_frame: np.ndarray (target_size, target_size, 3)
        crop_meta: dict (x0, y0, S)
    """
    h, w = frame.shape[:2]
    S = min(w, h)
    x0 = int((w - S) // 2)
    y0 = int((h - S) // 2)

    cropped = frame[y0:y0 + S, x0:x0 + S]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    crop_meta = {
        'x0': x0,
        'y0': y0,
        'square_size': S,
        'stream_size': target_size
    }

    return resized, crop_meta
