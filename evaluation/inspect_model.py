from ai_edge_litert.interpreter import Interpreter
import numpy as np
import cv2

interp = Interpreter('host_laptop/models/tinydriver_model.tflite')
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

scale, zero_point = in_det['quantization']
print(f"Input scale: {scale}, zero_point: {zero_point}")

# Create a high-contrast pattern
img = np.zeros((96, 96), dtype=np.uint8)
img[20:76, 20:76] = 200 # white square in center
cv2.circle(img, (35, 40), 8, 30, -1)
cv2.circle(img, (61, 40), 8, 30, -1)
cv2.rectangle(img, (40, 60), (56, 70), 50, -1)

norm = (img.astype(np.float32) - 128.0) / 128.0
quant = np.clip(np.round(norm / scale) + zero_point, -128, 127).astype(np.int8)
input_tensor = np.expand_dims(np.expand_dims(quant, axis=0), axis=-1)

interp.set_tensor(in_det['index'], input_tensor)
interp.invoke()

print("\nIntermediate Layer Activations after invoke:")
for d in interp.get_tensor_details():
    idx = d['index']
    t = interp.get_tensor(idx)
    name = d['name']
    # If not a weight (pseudo_qconst), print it
    if 'pseudo_qconst' not in name:
        print(f"Index {idx:3d}: {name[:45]:45s} shape={str(t.shape):18s} dtype={str(t.dtype):10s} min={t.min():.2f} max={t.max():.2f} std={t.std():.3f}")
