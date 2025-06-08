# In src/grouping/bbox_grouper.py

import yaml
import numpy as np
from typing import List, Dict


class BBoxGrouper:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('grouping', {})

        self.height_tolerance = config.get('height_tolerance', 0.2)
        self.proximity_factor = config.get('proximity_factor', 2.5)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.min_vertical_overlap = config.get('min_vertical_overlap', 0.5)
        self._validate_parameters()

    def _validate_parameters(self):
        assert 0 <= self.height_tolerance <= 1, "height_tolerance must be between 0-1"
        assert self.proximity_factor > 0, "proximity_factor must be a positive number"
        assert 0 <= self.confidence_threshold <= 1, "confidence_threshold must be between 0-1"
        assert 0 <= self.min_vertical_overlap <= 1, "min_vertical_overlap must be between 0-1"

    def _get_bbox_properties(self, detection: Dict) -> Dict:
        bbox = np.array(detection['bbox_original'])
        min_x, min_y = np.min(bbox, axis=0)
        max_x, max_y = np.max(bbox, axis=0)
        return {
            'h': max_y - min_y, 'w': max_x - min_x,
            'cy': (min_y + max_y) / 2, 'cx': (min_x + max_x) / 2,
            'min_y': min_y, 'max_y': max_y
        }

    def _are_boxes_compatible(self, det1: Dict, props1: Dict, det2: Dict, props2: Dict) -> bool:
        if det1.get('rotation_angle') != det2.get('rotation_angle'):
            return False

        if abs(props1['h'] - props2['h']) > self.height_tolerance * max(props1['h'], props2['h']):
            return False

        vertical_overlap = max(0, min(props1['max_y'], props2['max_y']) - max(props1['min_y'], props2['min_y']))
        if vertical_overlap < self.min_vertical_overlap * min(props1['h'], props2['h']):
            return False

        max_allowed_dist = self.proximity_factor * ((props1['h'] + props2['h']) / 2)
        horizontal_dist = abs(props1['cx'] - props2['cx']) - ((props1['w'] + props2['w']) / 2)
        return horizontal_dist < max_allowed_dist

    def _group_boxes(self, detections: List[Dict]) -> List[List[Dict]]:
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
        group.sort(key=lambda d: self._get_bbox_properties(d)['cx'])

        full_text = " ".join([d['text'] for d in group])
        avg_confidence = float(np.mean([d['confidence'] for d in group]))

        # FIX: Get orientation from the first component in the group
        orientation = group[0].get('rotation_angle', 0) if group else 0

        all_points = []
        for d in group:
            all_points.extend(d['bbox_original'])

        all_points_np = np.array(all_points)
        min_x, min_y = map(int, np.min(all_points_np, axis=0))
        max_x, max_y = map(int, np.max(all_points_np, axis=0))

        merged_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

        return {
            "text": full_text,
            "confidence": avg_confidence,
            "bbox": merged_bbox,
            "orientation": orientation,  # NEW: Add orientation to the output
            "component_detections": group
        }

    def process(self, all_detections: List[Dict]) -> List[Dict]:
        valid_detections = [d for d in all_detections if d['confidence'] >= self.confidence_threshold]
        groups = self._group_boxes(valid_detections)
        return [self._merge_group(g) for g in groups if g]
