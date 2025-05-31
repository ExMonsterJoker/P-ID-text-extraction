from data_loader.sahi_slicer import Slicer
from text_detection.inference import TextDetector
from postprocessing.heatmap_processor import TileProcessor
from grouping.graph_clustering import GroupingEngine
from ocr.recognizer import OCRRecognizer
from utils.config_loader import ConfigLoader
from utils.visualization import ResultVisualizer


class PipelineRunner:
    def __init__(self):
        self.config = ConfigLoader().load_configs()

    def run(self, image_path):
        # 1. Image slicing
        slicer = Slicer(self.config["base"])
        tiles, metadata = slicer.slice(image_path)

        # 2. Text detection
        detector = TextDetector(self.config["text_detection"])
        heatmaps = detector.detect_batch(tiles)

        # 3. Per-tile processing
        processor = TileProcessor(self.config["postprocessing"])
        detections = []
        for i, tile in enumerate(tiles):
            tile_dets = processor.process(tile, heatmaps[i], metadata[i])
            detections.extend(tile_dets)

        # 4. Global grouping
        grouper = GroupingEngine(self.config["grouping"])
        groups = grouper.process(detections)

        # 5. OCR recognition
        ocr = OCRRecognizer(self.config["ocr"])
        results = ocr.recognize_groups(image_path, groups)

        # 6. Output and visualization
        if self.config["base"]["debug_visualization"]:
            visualizer = ResultVisualizer()
            visualizer.save(image_path, results,
                            self.config["base"]["output_dir"])

        return results