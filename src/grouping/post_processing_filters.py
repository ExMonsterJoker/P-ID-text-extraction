# src/grouping/post_processing_filters.py
import numpy as np
import logging
from typing import List, Dict


def calculate_box_area(bbox: List[List[int]]) -> float:
    """Calculate the area of a bounding box from its vertices."""
    try:
        points = np.array(bbox)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        return float(width * height)
    except Exception:
        return 0.0


def calculate_complete_iou(box1: List[List[int]], box2: List[List[int]]) -> float:
    """Robust IoU calculation that handles any bbox format."""

    def normalize_bbox(bbox):
        points = np.array(bbox)
        return [np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])]

    try:
        box1_norm = normalize_bbox(box1)
        box2_norm = normalize_bbox(box2)
        x1_min, y1_min, x1_max, y1_max = box1_norm
        x2_min, y2_min, x2_max, y2_max = box2_norm

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        logging.warning(f"Error calculating IoU: {e}")
        return 0.0


def apply_soft_nms_filter(
        detections: List[Dict],
        iou_threshold: float,
        sigma: float,
        min_confidence: float
) -> List[Dict]:
    """
    Applies Soft-NMS to a list of detections.
    This version is robust and works on both raw detections (pre-grouping)
    and merged lines (post-grouping).
    """
    if not detections:
        return []

    logging.info(f"Applying Soft-NMS to {len(detections)} items...")

    # Work with a copy to avoid modifying the original list in place
    sorted_dets = [d.copy() for d in detections]

    # Sort by confidence
    sorted_dets.sort(key=lambda x: x['confidence'], reverse=True)

    for i in range(len(sorted_dets)):
        for j in range(i + 1, len(sorted_dets)):
            # MODIFICATION: Use 'bbox' from grouped lines, or fall back to 'bbox_original' for raw detections.
            box_i = sorted_dets[i].get('bbox') or sorted_dets[i].get('bbox_original')
            box_j = sorted_dets[j].get('bbox') or sorted_dets[j].get('bbox_original')

            # Skip if for some reason a box is missing
            if not box_i or not box_j:
                continue

            iou = calculate_complete_iou(box_i, box_j)

            if iou > iou_threshold:
                weight = np.exp(-(iou * iou) / sigma)
                sorted_dets[j]['confidence'] *= weight

    final_detections = [d for d in sorted_dets if d['confidence'] >= min_confidence]
    logging.info(f"Soft-NMS reduced items from {len(detections)} to {len(final_detections)}.")
    return final_detections


def apply_smart_aspect_ratio_filter(
        merged_lines: List[Dict],
        base_ratio: float,
        length_factor: float
) -> List[Dict]:
    """P&ID-specific aspect ratio filtering based on text length."""
    if not merged_lines:
        return []

    filtered = []
    for line in merged_lines:
        bbox = np.array(line['bbox'])
        w = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
        h = np.max(bbox[:, 1]) - np.min(bbox[:, 1])

        if w <= 0 or h <= 0: continue

        orientation = line.get('orientation', 0)
        text_length = len(line.get('text', ''))

        # Adaptive thresholds based on text length
        max_ratio = base_ratio + (text_length * length_factor)

        keep = False
        if orientation == 0:  # Horizontal
            if (h / w) <= max_ratio:
                keep = True
        else:  # Vertical
            if (w / h) <= max_ratio:
                keep = True

        if keep:
            filtered.append(line)
        else:
            logging.debug(f"Dropped by smart aspect ratio filter: '{line['text']}'")

    logging.info(f"Smart aspect ratio filter reduced lines from {len(merged_lines)} to {len(filtered)}.")
    return filtered