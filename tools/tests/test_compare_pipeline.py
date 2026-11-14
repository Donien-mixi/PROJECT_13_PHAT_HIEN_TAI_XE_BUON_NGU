import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import compare_pipeline as cp


FACE_22 = np.array([
    [0.30, 0.40], [0.32, 0.38], [0.34, 0.39], [0.36, 0.40], [0.34, 0.42], [0.32, 0.41],
    [0.60, 0.40], [0.62, 0.38], [0.64, 0.39], [0.66, 0.40], [0.64, 0.42], [0.62, 0.41],
    [0.40, 0.65], [0.60, 0.65], [0.45, 0.62], [0.55, 0.72], [0.47, 0.66], [0.53, 0.68],
    [0.48, 0.50], [0.50, 0.58], [0.52, 0.60], [0.50, 0.75],
], dtype=np.float64)


class TestNormalizeLandmarks(unittest.TestCase):
    def test_flat_and_nested_formats_are_identical(self):
        flat = FACE_22.reshape(44).tolist()
        nested = FACE_22.tolist()

        from_flat = cp.normalize_landmarks(flat)
        from_nested = cp.normalize_landmarks(nested)

        self.assertEqual(from_flat.shape, (22, 2))
        np.testing.assert_array_equal(from_flat, from_nested)
        np.testing.assert_array_equal(from_flat, FACE_22)
        np.testing.assert_allclose(cp.ear_mar(from_flat * 96.0), cp.ear_mar(from_nested * 96.0))

    def test_accepts_numpy_flat_and_numpy_2d(self):
        np.testing.assert_array_equal(
            cp.normalize_landmarks(FACE_22.reshape(44)),
            cp.normalize_landmarks(FACE_22),
        )

    def test_missing_landmarks(self):
        for value in (None, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(cp.LandmarkDataError) as ctx:
                    cp.normalize_landmarks(value)
                self.assertEqual(ctx.exception.reason, "missing")

    def test_non_numeric_landmarks(self):
        with self.assertRaises(cp.LandmarkDataError) as ctx:
            cp.normalize_landmarks(["x"] * 44)
        self.assertEqual(ctx.exception.reason, "invalid")

    def test_unsupported_shapes(self):
        for value in (np.zeros((21, 2)), np.zeros((45,)), np.zeros((22, 3)), np.zeros((2, 22))):
            with self.subTest(shape=value.shape):
                with self.assertRaises(cp.LandmarkDataError) as ctx:
                    cp.normalize_landmarks(value)
                self.assertEqual(ctx.exception.reason, "shape")
                self.assertIn(str(value.shape), str(ctx.exception))

    def test_non_finite_landmarks(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad_value):
                bad = FACE_22.copy()
                bad[4, 1] = bad_value
                with self.assertRaises(cp.LandmarkDataError) as ctx:
                    cp.normalize_landmarks(bad.reshape(44).tolist())
                self.assertEqual(ctx.exception.reason, "nonfinite")
                self.assertIn("1 gia tri NaN/Inf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
