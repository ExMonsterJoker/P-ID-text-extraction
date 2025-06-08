import json
import numpy as np
from typing import List, Dict, Tuple
import yaml
from pathlib import Path


class BBoxGrouper:
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.validate_parameters()

    def load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Grouping parameters
        self.height_tolerance = config['height_tolerance']
        self.width_tolerance = config['width_tolerance']
        self.proximity_factor = config['proximity_factor']
        self.confidence_threshold = config['confidence_threshold']

    def validate_parameters(self):
        assert 0 <= self.height_tolerance <= 1, "height_tolerance must be between 0-1"
        assert 0 <= self.width_tolerance <= 1, "width_tolerance must be between 0-1"
        assert self.proximity_factor > 0, "proximity_factor must be positive"
        assert 0 <= self.confidence_threshold <= 1, "confidence_threshold must be between 0-1"

    def convert_to_global_coordinates(self, detections: List[Dict]) -> List[Dict]:
        global_detections = []
        for detection in detections:
            if detection['confidence'] < self.confidence_threshold:
                continue

            tile_coords = detection['tile_coordinates']
            bbox = detection['bbox']
            global_bbox = []
            for point in bbox:
                global_x = point[0] + tile_coords[0]
                global_y = point[1] + tile_coords[1]
                global_bbox.append([global_x, global_y])

            detection_copy = detection.copy()
            detection_copy['global_bbox'] = global_bbox
            global_detections.append(detection_copy)

        return global_detections

    def get_bbox_properties(self, bbox: List[List[int]], rotation_angle: int) -> Dict:

    # ... (same as original implementation) ...

    def are_boxes_compatible(self, box1: Dict, box2: Dict) -> bool:

    # ... (same as original implementation) ...

    def are_boxes_adjacent(self, box1: Dict, box2: Dict) -> bool:

    # ... (same as original implementation) ...

    def group_boxes(self, detections: List[Dict]) -> List[List[Dict]]:

    # ... (same as original implementation) ...

    def merge_group_bbox(self, group: List[Dict]) -> Tuple[List[List[int]], Dict]:
        all_points = []
        texts = []
        confidences = []
        rotation_angles = set()

        for box in group:
            all_points.extend(box['detection']['global_bbox'])
            texts.append(box['detection']['text'])
            confidences.append(box['detection']['confidence'])
            rotation_angles.add(box['detection']['rotation_angle'])

        all_points = np.array(all_points)
        min_x = int(np.min(all_points[:, 0]))
        min_y = int(np.min(all_points[:, 1]))
        max_x = int(np.max(all_points[:, 0]))
        max_y = int(np.max(all_points[:, 1]))

        merged_bbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]

        metadata = {
            "text_parts": texts,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(min(confidences)),
            "max_confidence": float(max(confidences)),
            "rotation_angle": rotation_angles.pop() if len(rotation_angles) == 1 else -1,
            "num_parts": len(group)
        }

        return merged_bbox, metadata

    def process_detections(self, detections: List[Dict], image_name: str) -> Dict:
        print(f"Processing {len(detections)} detections")

        # Convert to global coordinates
        global_detections = self.convert_to_global_coordinates(detections)
        print(f"Converted to global coordinates: {len(global_detections)} valid detections")

        # Group boxes
        groups = self.group_boxes(global_detections)
        print(f"Grouped into {len(groups)} text regions")

        # Generate results
        results = {
            "image_name": image_name,
            "groups": [],
            "stats": {
                "total_groups": len(groups),
                "single_part_groups": 0,
                "multi_part_groups": 0
            }
        }

        # Process each group
        for group_idx, group in enumerate(groups):
            merged_bbox, metadata = self.merge_group_bbox(group)

            group_data = {
                "group_id": group_idx,
                "merged_bbox": merged_bbox,
                **metadata
            }

            results["groups"].append(group_data)

            # Update statistics
            if metadata["num_parts"] == 1:
                results["stats"]["single_part_groups"] += 1
            else:
                results["stats"]["multi_part_groups"] += 1

        print(f"Single detections: {results['stats']['single_part_groups']}")
        print(f"Merged groups: {results['stats']['multi_part_groups']}")

        return results