import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_tinyml.isomorphic_transform import canonical_face_crop


FACE_POINTS = np.array([
    [10, 20], [14, 17], [18, 17], [20, 20], [18, 23], [14, 23],
    [20, 20], [24, 17], [28, 17], [30, 20], [28, 23], [24, 23],
    [15, 55], [30, 55], [17, 50], [28, 62], [19, 53], [26, 58],
    [22, 32], [25, 40], [24, 46], [22, 70],
], dtype=np.float32)


class TestCanonicalTransform(unittest.TestCase):
    def test_padding_and_roundtrip(self):
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        crop, norm, meta = canonical_face_crop(image, FACE_POINTS, jitter=False)

        self.assertEqual(crop.shape, (96, 96, 1))
        self.assertLess(meta["x0"], 0)
        self.assertLess(meta["y0"], 0)
        self.assertEqual(int(crop[0, 0, 0]), 0)
        self.assertEqual(meta["square_size"] % 2, 0)

        mapped = np.column_stack([
            meta["x0"] + norm[:, 0] * meta["square_size"],
            meta["y0"] + norm[:, 1] * meta["square_size"],
        ])
        np.testing.assert_allclose(mapped, FACE_POINTS, atol=1e-3)

    def test_right_bottom_padding_keeps_meta(self):
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        pts = FACE_POINTS + np.array([275, 165], dtype=np.float32)
        crop, norm, meta = canonical_face_crop(image, pts, jitter=False)
        self.assertEqual(crop.shape, (96, 96, 1))
        self.assertGreater(meta["x0"] + meta["square_size"], 320)
        self.assertGreater(meta["y0"] + meta["square_size"], 240)
        mapped = np.column_stack([
            meta["x0"] + norm[:, 0] * meta["square_size"],
            meta["y0"] + norm[:, 1] * meta["square_size"],
        ])
        inside = (norm[:, 0] >= 0) & (norm[:, 0] <= 1) & (norm[:, 1] >= 0) & (norm[:, 1] <= 1)
        np.testing.assert_allclose(mapped[inside], pts[inside], atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
