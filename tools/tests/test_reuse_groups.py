import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.prepare_reuse_groups import Components, photo_key, video_record


class ReuseGroups(unittest.TestCase):
    def test_afw_different_targets_render_flip_stay_in_one_photo(self):
        a = photo_key('300W_LP', 'AFW/AFW_123_1_0.jpg')
        b = photo_key('300W_LP', 'AFW_Flip/AFW_123_2_19.jpg')
        self.assertEqual(a, b)
        self.assertNotEqual(a, photo_key('300W_LP', 'AFW/AFW_124_1_0.jpg'))

    def test_cew_augmentation_and_face_crops_stay_together(self):
        a = photo_key('CEW', 'closed/aug_14_closed_eye_0649.jpg_face_2.jpg')
        self.assertEqual(a, photo_key('CEW', 'closed/closed_eye_0649.jpg_face_1.jpg'))

    def test_duplicate_suffix_not_a_new_photo(self):
        self.assertEqual(photo_key('CEW', 'open/A_0001(1).jpg'), photo_key('CEW', 'open/A_0001.jpg'))

    def test_exact_duplicate_bridge_merges_transitively(self):
        c = Components()
        c.union('a', 'b')
        c.union('c', 'b')
        self.assertEqual(c.find('a'), c.find('c'))
        c.find('d')
        self.assertNotEqual(c.find('a'), c.find('d'))

    def test_cross_view_identity_not_assumed(self):
        evidence = {'rows': [dict(view=v, table_section='Male', participant_number=1, page=1)
                             for v in ['Dash', 'Mirror']]}
        a = video_record('YawDD/Dash/Dash/Male/1-MaleGlasses.avi', evidence)
        b = video_record('YawDD/Mirror/Mirror/Male/1-MaleGlasses-Normal.avi', evidence)
        self.assertNotEqual(a['dataset_scoped_participant_id'], b['dataset_scoped_participant_id'])
        self.assertEqual(a['leakage_component'], b['leakage_component'])
        self.assertFalse(a['cross_view_identity_verified'])
        self.assertIsNone(a['split_assignment'])

    def test_unknown_participant_fails_closed(self):
        with self.assertRaises(ValueError):
            video_record('YawDD/Mirror/Mirror/Male/999-MaleGlasses-Normal.avi', {'rows': []})


if __name__ == '__main__':
    unittest.main()
