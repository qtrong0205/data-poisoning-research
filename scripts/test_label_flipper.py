import unittest
import numpy as np
import os
import json
from scripts.label_flipper import LabelFlipper

class TestLabelFlipper(unittest.TestCase):
    def setUp(self):
        self.num_classes = 4
        self.flipper = LabelFlipper(num_classes=self.num_classes)
        # 10 mẫu với các nhãn 0, 1, 2, 3
        self.labels = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]

    def test_random_flip_rate(self):
        flip_rate = 0.3 
        poisoned_labels, manifest = self.flipper.random_flip(self.labels, flip_rate=flip_rate, random_seed=42)
        
        self.assertEqual(len(manifest), 3)
        for idx, info in manifest.items():
            self.assertNotEqual(info["original_label"], info["poisoned_label"])
            self.assertEqual(poisoned_labels[idx], info["poisoned_label"])
            self.assertTrue(0 <= info["poisoned_label"] < self.num_classes)

    def test_targeted_flip_logic(self):
        # Chuyển nhãn 1 thành 3 với tỷ lệ 66% 
        source_class = 1
        target_class = 3
        flip_rate = 0.66 
        
        poisoned_labels, manifest = self.flipper.targeted_flip(
            self.labels, source_class, target_class, flip_rate=flip_rate, random_seed=123
        )
        
        self.assertEqual(len(manifest), 1)  # Chỉ có 1 mẫu nhãn 1 được flip
        for idx, info in manifest.items():
            self.assertEqual(info["original_label"], source_class)
            self.assertEqual(info["poisoned_label"], target_class)
            self.assertEqual(poisoned_labels[idx], target_class)

    def test_save_manifest(self):
        _, manifest = self.flipper.random_flip(self.labels, flip_rate=0.2)
        test_filepath = "test_manifest.json"
        self.flipper.save_manifest(manifest, test_filepath)
        
        self.assertTrue(os.path.exists(test_filepath))
        with open(test_filepath, 'r') as f:
            loaded_manifest = json.load(f)
            
        self.assertEqual(len(loaded_manifest), 2)
        os.remove(test_filepath)

if __name__ == "__main__":
    unittest.main()