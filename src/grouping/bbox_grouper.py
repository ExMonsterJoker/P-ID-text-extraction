# In src/grouping/bbox_grouper.py

import json
import numpy as np
from typing import List, Dict, Tuple
import yaml
from pathlib import Path
import logging



class BBoxGrouper:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['grouping']
        self.height_tolerance = config['height_tolerance']
        self.proximity_factor = config['proximity_factor']
        self.confidence_threshold = config['confidence_threshold']

    def load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Grouping parameters
        self.height_tolerance = config['height_tolerance']
        self.width_tolerance = config['width_tolerance']
        self.proximity_factor = config['proximity_factor']
        self.confidence_threshold = config['confidence_threshold']
        self.max_dimension_expansion = config.get('max_dimension_expansion', 1.5)  # New parameter

        # New filtering parameters - note the underscore prefix
        self._filter_irregular_boxes = config.get('filter_irregular_boxes', True)
        self.horizontal_aspect_ratio_threshold = config.get('horizontal_aspect_ratio_threshold',
                                                            0.5)  # width/height for 0° text
        self.vertical_aspect_ratio_threshold = config.get('vertical_aspect_ratio_threshold',
                                                          2.0)  # height/width for 90° text

    def validate_parameters(self):
        # Existing validations...
        assert 0 <= self.height_tolerance <= 1, "height_tolerance must be between 0-1"
        assert 0 <= self.width_tolerance <= 1, "width_tolerance must be between 0-1"
        assert self.proximity_factor > 0, "proximity_factor must be positive"
        assert 0 <= self.confidence_threshold <= 1, "confidence_threshold must be between 0-1"
        assert self.max_dimension_expansion >= 1, "max_dimension_expansion must be >= 1"

        # New validations
        assert self.horizontal_aspect_ratio_threshold > 0, "horizontal_aspect_ratio_threshold must be positive"
        assert self.vertical_aspect_ratio_threshold > 0, "vertical_aspect_ratio_threshold must be positive"

    def is_bbox_regular(self, bbox_props: Dict, rotation_angle: int) -> bool:
        """Check if bbox dimensions are regular for its orientation"""
        if not self._filter_irregular_boxes:  # Updated reference
            return True

        width = bbox_props['width']
        height = bbox_props['height']

        # Avoid division by zero
        if width == 0 or height == 0:
            logging.warning(f"Zero dimension detected: width={width}, height={height}")
            return False

        if rotation_angle == 0:
            # Horizontal text: width should be greater than height
            aspect_ratio = width / height
            is_regular = aspect_ratio >= self.horizontal_aspect_ratio_threshold
            logging.debug(
                f"Horizontal text: aspect_ratio={aspect_ratio:.2f}, threshold={self.horizontal_aspect_ratio_threshold}, regular={is_regular}")
            return is_regular

        elif rotation_angle == 90:
            # Vertical text: height should be greater than width
            aspect_ratio = height / width
            is_regular = aspect_ratio >= self.vertical_aspect_ratio_threshold
            logging.debug(
                f"Vertical text: aspect_ratio={aspect_ratio:.2f}, threshold={self.vertical_aspect_ratio_threshold}, regular={is_regular}")
            return is_regular

        else:
            # Unknown rotation, keep the box
            logging.warning(f"Unknown rotation angle: {rotation_angle}")
            return True

    def filter_irregular_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Filter out boxes with irregular dimensions for their orientation"""
        if not self._filter_irregular_boxes:  # Updated reference
            return boxes

        regular_boxes = []
        filtered_count = 0

        for box in boxes:
            props = box['properties']
            rotation = box['rotation_angle']

            if self.is_bbox_regular(props, rotation):
                regular_boxes.append(box)
            else:
                filtered_count += 1
                logging.info(f"Filtered irregular box: rotation={rotation}°, "
                             f"width={props['width']:.1f}, height={props['height']:.1f}, "
                             f"aspect_ratio={props['width'] / props['height']:.2f}")

        logging.info(f"Irregular box filtering: {len(boxes)} -> {len(regular_boxes)} boxes "
                     f"({filtered_count} filtered)")

        return regular_boxes

    def is_bbox_inside(self, inner_box: Dict, outer_box: Dict) -> bool:
        """Check if inner_box is completely inside outer_box"""
        inner_props = inner_box['properties']
        outer_props = outer_box['properties']

        return (inner_props['min_x'] >= outer_props['min_x'] and
                inner_props['max_x'] <= outer_props['max_x'] and
                inner_props['min_y'] >= outer_props['min_y'] and
                inner_props['max_y'] <= outer_props['max_y'])

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

    def remove_inner_bboxes(self, boxes: List[Dict]) -> List[Dict]:
        """Remove smaller bounding boxes that are completely inside larger bounding boxes"""
        filtered_boxes = []

        for i, box1 in enumerate(boxes):
            is_inner = False
            props1 = box1['properties']

            for j, box2 in enumerate(boxes):
                if i != j:
                    props2 = box2['properties']
                    # Check if box1 is completely inside box2
                    if (props1['min_x'] >= props2['min_x'] and
                            props1['max_x'] <= props2['max_x'] and
                            props1['min_y'] >= props2['min_y'] and
                            props1['max_y'] <= props2['max_y']):
                        is_inner = True
                        break

            if not is_inner:
                filtered_boxes.append(box1)

        return filtered_boxes

    def get_bbox_properties(self, detection: Dict) -> Dict:
        """Calculates geometric properties of a single detection bbox."""
        bbox = np.array(detection['bbox_original'])

        # Calculate width, height, and center
        min_x, min_y = np.min(bbox, axis=0)
        max_x, max_y = np.max(bbox, axis=0)

        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        return {
            'w': width, 'h': height, 'cx': center_x, 'cy': center_y,
            'min_y': min_y, 'max_y': max_y
        }

    def are_boxes_compatible(self, det1: Dict, props1: Dict, det2: Dict, props2: Dict) -> bool:
        """
        Checks if two text boxes are compatible to be on the same line.
        """
        # 1. Must have the same rotation angle
        if det1['rotation_angle'] != det2['rotation_angle']:
            return False

        # 2. Height similarity check
        if abs(props1['h'] - props2['h']) > self.height_tolerance * max(props1['h'], props2['h']):
            return False

        # 3. Vertical alignment check (significant y-overlap)
        vertical_overlap = max(0, min(props1['max_y'], props2['max_y']) - max(props1['min_y'], props2['min_y']))
        if vertical_overlap < 0.5 * min(props1['h'], props2['h']):
            return False

        # 4. Proximity check
        # Max allowed horizontal distance is proportional to the average height of the two boxes
        max_allowed_dist = self.proximity_factor * ((props1['h'] + props2['h']) / 2)
        dist = abs(props1['cx'] - props2['cx']) - ((props1['w'] + props2['w']) / 2)

        return dist < max_allowed_dist

    def are_boxes_adjacent(self, box1: Dict, box2: Dict) -> bool:
        """Check if two boxes are adjacent based on their orientation"""
        if not self.are_boxes_compatible(box1, box2):
            return False

        props1 = box1['properties']
        props2 = box2['properties']
        rotation = box1['rotation_angle']

        # Calculate search distance based on primary dimension
        search_distance = max(props1['primary_dim'], props2['primary_dim']) * self.proximity_factor

        if rotation == 0:
            # Horizontal text - check left/right adjacency
            y_overlap = not (props1['max_y'] < props2['min_y'] or props2['max_y'] < props1['min_y'])
            if not y_overlap:
                return False

            horizontal_gap = min(
                abs(props1['min_x'] - props2['max_x']),
                abs(props2['min_x'] - props1['max_x'])
            )
            return horizontal_gap <= search_distance

        else:  # rotation == 90
            # Vertical text - check up/down adjacency
            x_overlap = not (props1['max_x'] < props2['min_x'] or props2['max_x'] < props1['min_x'])
            if not x_overlap:
                return False

            vertical_gap = min(
                abs(props1['min_y'] - props2['max_y']),
                abs(props2['min_y'] - props1['max_y'])
            )
            return vertical_gap <= search_distance

    def group_boxes(self, detections: List[Dict]) -> List[List[Dict]]:
        """Groups detections into text lines using a graph-based approach."""
        if not detections:
            return []

        # Calculate properties for all boxes first
        props = [self.get_bbox_properties(d) for d in detections]

        # Build adjacency list for the graph
        adj = {i: [] for i in range(len(detections))}
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if self.are_boxes_compatible(detections[i], props[i], detections[j], props[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        # Find connected components (groups) using BFS/DFS
        groups = []
        visited = set()
        for i in range(len(detections)):
            if i not in visited:
                group = []
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    group.append(detections[u])
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                groups.append(group)

        return groups

    def is_valid_merge(self, group: List[Dict], merged_bbox: List[List[int]]) -> bool:
        """Check if merged bbox expansion is within acceptable limits"""
        if len(group) == 1:
            return True

        # Calculate merged dimensions
        min_x = min(p[0] for p in merged_bbox)
        max_x = max(p[0] for p in merged_bbox)
        min_y = min(p[1] for p in merged_bbox)
        max_y = max(p[1] for p in merged_bbox)
        merged_width = max_x - min_x
        merged_height = max_y - min_y

        # Get rotation angle (all should be same in group)
        rotation = group[0]['detection']['rotation_angle']

        # Calculate average part dimensions
        widths = []
        heights = []
        for box in group:
            props = box['properties']
            widths.append(props['width'])
            heights.append(props['height'])

        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        # Check expansion limits based on orientation
        if rotation == 0:  # Horizontal text
            # Should expand horizontally, not vertically
            height_ratio = merged_height / avg_height
            if height_ratio > self.max_dimension_expansion:
                return False

        elif rotation == 90:  # Vertical text
            # Should expand vertically, not horizontally
            width_ratio = merged_width / avg_width
            if width_ratio > self.max_dimension_expansion:
                return False

        return True

    import logging

    def filter_by_centroid_containment(self, boxes: List[Dict]) -> List[List[Dict]]:
        """Merge bounding boxes if:
        1. Smaller box's centroid falls within larger box, or
        2. One box is completely inside another (same orientation)
        """
        logging.info(f"Starting centroid/containment filtering with {len(boxes)} boxes")

        if len(boxes) <= 1:
            logging.info("Only 1 or fewer boxes, returning as-is")
            return [boxes]

        merged_boxes = []
        used_indices = set()

        for i, box1 in enumerate(boxes):
            if i in used_indices:
                continue

            if not isinstance(box1, dict) or 'properties' not in box1:
                print(f"ERROR: Invalid box structure at index {i}: {box1}")
                continue

            current_group = [box1]
            used_indices.add(i)
            props1 = box1['properties']

            for j, box2 in enumerate(boxes):
                if j in used_indices or i == j:
                    continue

                if not isinstance(box2, dict) or 'properties' not in box2:
                    print(f"ERROR: Invalid box structure at index {j}: {box2}")
                    continue

                # Check same orientation
                if box1['rotation_angle'] != box2['rotation_angle']:
                    continue

                props2 = box2['properties']

                # Check centroid containment
                centroid1 = (props1['center_x'], props1['center_y'])
                centroid2 = (props2['center_x'], props2['center_y'])

                centroid1_in_box2 = self.is_centroid_inside_bbox(centroid1, props2)
                centroid2_in_box1 = self.is_centroid_inside_bbox(centroid2, props1)

                # Check box containment
                box1_in_box2 = self.is_bbox_inside(box1, box2)
                box2_in_box1 = self.is_bbox_inside(box2, box1)

                if centroid1_in_box2 or centroid2_in_box1 or box1_in_box2 or box2_in_box1:
                    logging.info(f"  MERGING: Box {i} and Box {j}")
                    current_group.append(box2)
                    used_indices.add(j)

            merged_boxes.append(current_group)

        logging.info(f"Centroid/containment filtering complete: {len(boxes)} -> {len(merged_boxes)} groups")
        return merged_boxes

    def is_centroid_inside_bbox(self, centroid: tuple, bbox_props: Dict) -> bool:
        """Check if centroid falls within bounding box"""
        cx, cy = centroid
        inside = (bbox_props['min_x'] <= cx <= bbox_props['max_x'] and
                  bbox_props['min_y'] <= cy <= bbox_props['max_y'])

        logging.debug(f"      Checking centroid ({cx:.1f}, {cy:.1f}) in bbox "
                      f"[{bbox_props['min_x']}, {bbox_props['min_y']}, {bbox_props['max_x']}, {bbox_props['max_y']}]: {inside}")
        return inside

    def merge_group(self, group: List[Dict]) -> Dict:
        """Merges a group of detections into a single text line."""
        # Sort by horizontal position
        group.sort(key=lambda d: self.get_bbox_properties(d)['cx'])

        full_text = " ".join([d['text'] for d in group])
        avg_confidence = float(np.mean([d['confidence'] for d in group]))

        # Create a bounding box that encloses all boxes in the group
        all_points = []
        for d in group:
            all_points.extend(d['bbox_original'])

        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        merged_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "bbox": merged_bbox,
            "component_detections": group
        }

    def process(self, all_detections: List[Dict]) -> List[Dict]:
        """Main processing function to filter and group detections."""

        # 1. Filter by confidence
        valid_detections = [d for d in all_detections if d['confidence'] >= self.confidence_threshold]

        # 2. Group boxes into lines
        groups = self.group_boxes(valid_detections)

        # 3. Merge each group into a single entity
        merged_lines = [self.merge_group(g) for g in groups]

        return merged_lines

    def process_detections(self, detections: List[Dict], image_name: str) -> Dict:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        print(f"Processing {len(detections)} detections")

        # Convert to global coordinates
        global_detections = self.convert_to_global_coordinates(detections)
        print(f"Converted to global coordinates: {len(global_detections)} valid detections")

        # Group boxes
        groups = self.group_boxes(global_detections)
        print(f"Grouped into {len(groups)} text regions")

        # Filter irregular boxes in each group
        filtered_groups = []
        for group_idx, group in enumerate(groups):
            # Filter irregular boxes first
            regular_boxes = self.filter_irregular_boxes(group)

            if regular_boxes:  # Only process non-empty groups
                # Apply centroid-based filtering
                centroid_filtered = self.filter_by_centroid_containment(regular_boxes)
                filtered_groups.extend(centroid_filtered)
            else:
                logging.info(f"Group {group_idx}: All boxes filtered as irregular")

        print(f"After filtering: {len(filtered_groups)} groups")

        # Generate results
        results = {
            "image_name": image_name,
            "groups": [],
            "stats": {
                "total_groups": len(groups),
                "single_part_groups": 0,
                "multi_part_groups": 0,
                "valid_merges": 0,
                "invalid_merges": 0
            }
        }

        # Process each filtered group
        for group_idx, group in enumerate(filtered_groups):
            # Skip empty groups
            if not group:
                continue

            merged_bbox, metadata = self.merge_group_bbox(group)

            # For invalid merges, split into individual boxes
            if metadata["num_parts"] > 1 and not metadata["is_valid_merge"]:
                results["stats"]["invalid_merges"] += 1
                # Split group into individual detections
                for box in group:
                    single_bbox, single_meta = self.merge_group_bbox([box])
                    group_data = {
                        "group_id": f"{group_idx}-{box['detection']['tile_id']}",
                        "merged_bbox": single_bbox,
                        **single_meta
                    }
                    results["groups"].append(group_data)
                    results["stats"]["single_part_groups"] += 1
                continue

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
                if metadata["is_valid_merge"]:
                    results["stats"]["valid_merges"] += 1

        print(f"Single detections: {results['stats']['single_part_groups']}")
        print(f"Multi-part groups: {results['stats']['multi_part_groups']}")
        print(f"Valid merges: {results['stats']['valid_merges']}")
        print(f"Invalid merges split: {results['stats']['invalid_merges']}")

        return results