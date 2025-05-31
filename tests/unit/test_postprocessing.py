# tests/unit/test_postprocessing.py
import unittest
import numpy as np
from src.postprocessing.heatmap_processor import TileProcessor
from src.utils.config_loader import ConfigLoader


class TestPostprocessing(unittest.TestCase):
    def setUp(self):
        config = ConfigLoader().get_config("postprocessing")
        self.processor = TileProcessor(config)

    def test_thresholding(self):
        # Create synthetic heatmap
        char_heatmap = np.zeros((512, 512), dtype=np.float32)
        char_heatmap[100:120, 100:200] = 0.8  # Valid text
        char_heatmap[300:310, 300:310] = 0.3  # Low confidence

        # Process
        detections = self.processor.process(char_heatmap, None, {})
        self.assertEqual(len(detections), 1)
        self.assertGreater(detections[0]["confidence"], 0.7)