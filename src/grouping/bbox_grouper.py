# src/grouping/bbox_grouper.py
import yaml
import numpy as np
from typing import List, Dict
import logging

# Import the new modular filter function
from .post_processing_filters import apply_aspect_ratio_filter


class BBoxGrouper:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        grouping_config = config.get('grouping', {})
        post_group_config = config.get('post_group_filtering', {}).get('aspect_ratio_filter', {})

        # Grouping parameters
        self.h_height_tolerance = grouping_config.get('h_height_tolerance', 0.3)
        self.h_proximity_factor = grouping_config.get('h_proximity_factor', 2.5)
        self.h_min_vertical_overlap = grouping_config.get('h_min_vertical_overlap', 0.4)
        self.v_width_tolerance = grouping_config.get('v_width_tolerance', 0.3)
        self.v_proximity_factor = grouping_config.get('v_proximity_factor', 1.5)
        self.v_min_horizontal_overlap = grouping_config.get('v_min_horizontal_overlap', 0.4)
        # REMOVED: confidence_threshold - no longer filtering by confidence

        # Post-grouping filter parameters
        self.max_hw_ratio_horizontal = post_group_config.get('max_hw_ratio_horizontal', 0.8)
        self.max_wh_ratio_vertical = post_group_config.get('max_wh_ratio_vertical', 0.8)

    def _get_bbox_properties(self, detection: Dict) -> Dict:
        """Calculates geometric properties of a bounding box."""
        bbox = np.array(detection['bbox_original'])
        min_x, min_y = np.min(bbox, axis=0)
        max_x, max_y = np.max(bbox, axis=0)
        props = {
            'h': max_y - min_y, 'w': max_x - min_x,
            'cy': (min_y + max_y) / 2, 'cx': (min_x + max_x) / 2,
            'min_y': min_y, 'max_y': max_y, 'min_x': min_x, 'max_x': max_x
        }
        logging.debug(
            f"Props for '{detection.get('text', 'N/A')}': h={props.get('h', 0):.1f}, w={props.get('w', 0):.1f}, cx={props.get('cx', 0):.1f}, cy={props.get('cy', 0):.1f}")
        return props

    def _are_boxes_compatible(self, det1: Dict, props1: Dict, det2: Dict, props2: Dict) -> bool:
        """Determines if two bounding boxes can be grouped, with detailed logging."""
        logging.debug(
            f"\n--- Checking compatibility: ['{det1.get('text', 'N/A')}'] vs ['{det2.get('text', 'N/A')}'] ---")

        if det1.get('rotation_angle') != det2.get('rotation_angle'):
            logging.debug("-> FAIL: Different rotation angles.")
            return False

        orientation = det1.get('rotation_angle', 0)
        if orientation == 0:  # Horizontal Grouping Logic
            # Height similarity check
            h_diff = abs(props1['h'] - props2['h'])
            h_thresh = self.h_height_tolerance * max(props1['h'], props2['h'])
            if h_diff > h_thresh:
                logging.debug(f"-> H-FAIL: Height difference {h_diff:.1f} > tolerance {h_thresh:.1f}")
                return False
            logging.debug(f"-> H-PASS: Height difference OK.")

            # Vertical overlap check
            vertical_overlap = max(0, min(props1['max_y'], props2['max_y']) - max(props1['min_y'], props2['min_y']))
            v_overlap_thresh = self.h_min_vertical_overlap * min(props1['h'], props2['h'])
            if vertical_overlap < v_overlap_thresh:
                logging.debug(f"-> H-FAIL: Vertical overlap {vertical_overlap:.1f} < threshold {v_overlap_thresh:.1f}")
                return False
            logging.debug(f"-> H-PASS: Vertical overlap OK.")

            # Proximity check
            avg_h = (props1['h'] + props2['h']) / 2
            max_allowed_dist = self.h_proximity_factor * avg_h
            horizontal_dist = abs(props1['cx'] - props2['cx']) - ((props1['w'] + props2['w']) / 2)
            if horizontal_dist >= max_allowed_dist:
                logging.debug(
                    f"-> H-FAIL: Horizontal distance {horizontal_dist:.1f} >= max allowed {max_allowed_dist:.1f}")
                return False
            logging.debug(f"-> H-PASS: Horizontal distance OK.")

            logging.debug("===> SUCCESS: Boxes are compatible.")
            return True

        elif orientation == 90:  # Vertical Grouping Logic
            if abs(props1['w'] - props2['w']) > self.v_width_tolerance * max(props1['w'], props2['w']):
                logging.debug("-> V-FAIL: Width difference too large.")
                return False
            horizontal_overlap = max(0, min(props1['max_x'], props2['max_x']) - max(props1['min_x'], props2['min_x']))
            if horizontal_overlap < self.v_min_horizontal_overlap * min(props1['w'], props2['w']):
                logging.debug("-> V-FAIL: Horizontal overlap too small.")
                return False
            max_allowed_dist = self.v_proximity_factor * ((props1['w'] + props2['w']) / 2)
            vertical_dist = abs(props1['cy'] - props2['cy']) - ((props1['h'] + props2['h']) / 2)
            if vertical_dist >= max_allowed_dist:
                logging.debug("-> V-FAIL: Vertical distance too large.")
                return False
            logging.debug("===> SUCCESS: Vertical boxes are compatible.")
            return True

        return False

    def _group_boxes(self, detections: List[Dict]) -> List[List[Dict]]:
        """Groups detections into lists of connected components."""
        if not detections: return []
        props = [self._get_bbox_properties(d) for d in detections]
        adj = {i: [] for i in range(len(detections))}
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if self._are_boxes_compatible(detections[i], props[i], detections[j], props[j]):
                    adj[i].append(j)
                    adj[j].append(i)
        groups, visited = [], set()
        for i in range(len(detections)):
            if i not in visited:
                group, q = [], [i]
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

    def _merge_group(self, group: List[Dict]) -> Dict:
        """Merges a list of grouped detections into a single text line."""
        if not group: return {}
        orientation = group[0].get('rotation_angle', 0)
        if orientation == 90:
            group.sort(key=lambda d: self._get_bbox_properties(d)['cy'])
        else:
            group.sort(key=lambda d: self._get_bbox_properties(d)['cx'])
        full_text = " ".join([d['text'] for d in group])
        avg_confidence = float(np.mean([d['confidence'] for d in group]))
        all_points = np.vstack([d['bbox_original'] for d in group])
        min_x, min_y = map(int, np.min(all_points, axis=0))
        max_x, max_y = map(int, np.max(all_points, axis=0))
        merged_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        return {
            "text": full_text, "confidence": avg_confidence, "bbox": merged_bbox,
            "orientation": orientation, "component_detections": group
        }

    def process(self, all_detections: List[Dict]) -> List[Dict]:
        """The main entry point for processing a list of detections."""
        # REMOVED: confidence threshold filtering - process all detections
        logging.info(f"Processing {len(all_detections)} detections without pre-filtering")

        groups = self._group_boxes(all_detections)
        merged_lines = [self._merge_group(g) for g in groups if g]

        # Apply ONLY the aspect ratio filter after grouping
        final_lines = apply_aspect_ratio_filter(
            merged_lines,
            self.max_hw_ratio_horizontal,
            self.max_wh_ratio_vertical
        )

        return final_lines
