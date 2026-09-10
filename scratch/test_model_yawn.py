import numpy as np
import tensorflow as tf
import cv2

interp = tf.lite.Interpreter(model_path='host_laptop/models/tinydriver_model.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print('Input Details:', inp['shape'], inp['dtype'], inp['quantization'])
print('Output Details:', out['shape'], out['dtype'], out['quantization'])

# Test on synthetic images with open mouth vs closed mouth
import sys
sys.path.append('training_tinyml')
from dataset_loader import generate_synthetic_driver_sample
from config import MOUTH_22, LEFT_EYE_22, RIGHT_EYE_22

# Normal sample
img_n, lm_n, _ = generate_synthetic_driver_sample(1, apply_aug=False, force_state='normal')
# Yawn sample
img_y, lm_y, _ = generate_synthetic_driver_sample(1, apply_aug=False, force_state='yawn')
# Microsleep sample
img_c, lm_c, _ = generate_synthetic_driver_sample(1, apply_aug=False, force_state='microsleep')

def run_tflite(img):
    scale, zp = inp['quantization']
    norm = (img.astype(np.float32) - 128.0) / 128.0
    if inp['dtype'] in [np.int8, np.uint8]:
        quant = np.clip(np.round(norm / scale) + zp, -128, 127).astype(inp['dtype'])
    else:
        quant = norm.astype(np.float32)
    interp.set_tensor(inp['index'], quant[np.newaxis, ...])
    interp.invoke()
    res = interp.get_tensor(out['index'])[0]
    o_scale, o_zp = out['quantization']
    if out['dtype'] in [np.int8, np.uint8]:
        res = (res.astype(np.float32) - o_zp) * o_scale
    return res.reshape(22, 2)

def compute_mar(pts):
    # P12, 13 (corners), P14, 15 (outer top/bot), P16, 17 (inner top/bot)
    w = np.linalg.norm(pts[12] - pts[13])
    h_out = np.linalg.norm(pts[14] - pts[15])
    h_in = np.linalg.norm(pts[16] - pts[17])
    return (h_out + h_in) / (2.0 * w)

def compute_ear(pts):
    # Left eye: 0, 1, 2, 3, 4, 5
    wl = np.linalg.norm(pts[0] - pts[3])
    hl1 = np.linalg.norm(pts[1] - pts[5])
    hl2 = np.linalg.norm(pts[2] - pts[4])
    ear_l = (hl1 + hl2) / (2.0 * wl)
    return ear_l

pred_n = run_tflite(img_n)
pred_y = run_tflite(img_y)
pred_c = run_tflite(img_c)

gt_n = lm_n.reshape(22, 2)
gt_y = lm_y.reshape(22, 2)
gt_c = lm_c.reshape(22, 2)

print('\n--- NORMAL SAMPLE ---')
print(f'GT:   EAR={compute_ear(gt_n):.3f}, MAR={compute_mar(gt_n):.3f}, P14_y={gt_n[14,1]:.3f}, P15_y={gt_n[15,1]:.3f}')
print(f'PRED: EAR={compute_ear(pred_n):.3f}, MAR={compute_mar(pred_n):.3f}, P14_y={pred_n[14,1]:.3f}, P15_y={pred_n[15,1]:.3f}')

print('\n--- YAWN SAMPLE ---')
print(f'GT:   EAR={compute_ear(gt_y):.3f}, MAR={compute_mar(gt_y):.3f}, P14_y={gt_y[14,1]:.3f}, P15_y={gt_y[15,1]:.3f}')
print(f'PRED: EAR={compute_ear(pred_y):.3f}, MAR={compute_mar(pred_y):.3f}, P14_y={pred_y[14,1]:.3f}, P15_y={pred_y[15,1]:.3f}')

print('\n--- MICROSLEEP SAMPLE ---')
print(f'GT:   EAR={compute_ear(gt_c):.3f}, MAR={compute_mar(gt_c):.3f}, P1_y={gt_c[1,1]:.3f}, P5_y={gt_c[5,1]:.3f}')
print(f'PRED: EAR={compute_ear(pred_c):.3f}, MAR={compute_mar(pred_c):.3f}, P1_y={pred_c[1,1]:.3f}, P5_y={pred_c[5,1]:.3f}')
