import json
import os
from pathlib import Path
from typing import List, Dict
from .bbox_grouper import BBoxGrouper
import yaml


class GroupProcessor:
    def __init__(self, config_path: str = "configs/grouping.yaml"):
        self.load_config(config_path)
        self.grouper = BBoxGrouper(config_path)

    def load_config(self, config_path: str):
        """Load project configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create absolute paths
        self.tile_metadata_dir = Path(config['tile_metadata_dir'])
        self.detection_metadata_dir = Path(config['detection_metadata_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_image_names(self) -> List[str]:
        """Get all unique image names from tile metadata"""
        image_names = set()
        for file in self.tile_metadata_dir.glob("*_tiles.json"):
            # Extract base name: DURI-OTPF03GS000-PRO-PID-IBU-0008-00
            image_name = file.name.split('_tiles.json')[0]
            image_names.add(image_name)
        return sorted(image_names)

    def load_tile_data(self, image_name: str) -> Dict:
        """Load tile metadata for an image"""
        tile_file = self.tile_metadata_dir / f"{image_name}_tiles.json"
        with open(tile_file, 'r') as f:
            return json.load(f)

    def load_detections(self, image_name: str) -> List[Dict]:
        """Load all OCR detections for an image across all tiles"""
        all_detections = []

        # Get list of detection files for this image
        pattern = f"{image_name}_*_ocr.json"
        for file in self.detection_metadata_dir.glob(pattern):
            with open(file, 'r') as f:
                tile_detections = json.load(f)
                if isinstance(tile_detections, list):
                    all_detections.extend(tile_detections)

        return all_detections

    def process_image(self, image_name: str):
        """Process all detections for a single image"""
        print(f"\nProcessing image: {image_name}")

        # Load detections from all tiles
        detections = self.load_detections(image_name)
        print(f"Loaded {len(detections)} detections from tiles")

        # Process detections with grouper
        results = self.grouper.process_detections(detections, image_name)

        # Save results
        output_file = self.output_dir / f"{image_name}_groups.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved grouping results to {output_file}")

        return results

    def process_all_images(self):
        """Process all images in the project"""
        image_names = self.get_image_names()
        print(f"Found {len(image_names)} images to process")

        for image_name in image_names:
            self.process_image(image_name)


if __name__ == "__main__":
    print("Starting grouping process...")
    processor = GroupProcessor()
    processor.process_all_images()
    print("Grouping process completed!")