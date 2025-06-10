# src/grouping/post_processing_filters.py
import numpy as np
import logging
from typing import List, Dict


def calculate_iou(box1: List[List[int]], box2: List[List[int]]) -> float:
    """
    Calculates Intersection over Union (IoU) for two rectangular bounding boxes.
    Assumes box format: [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    """
    x1_min, y1_min = box1[0]
    x1_max, y1_max = box1[2]

    x2_min, y2_min = box2[0]
    x2_max, y2_max = box2[2]

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def apply_soft_nms_filter(
    lines: List[Dict],
    iou_threshold: float,
    sigma: float,
    min_confidence: float
) -> List[Dict]:
    """
    Applies Soft-NMS to a list of detected lines. Instead of removing overlapping
    boxes, it decays their confidence scores using a Gaussian penalty.
    """
    if not lines:
        return []

    logging.info(f"Applying Soft-NMS with IoU threshold={iou_threshold}, sigma={sigma}, min_confidence={min_confidence}")

    # Work on a copy of the lines list
    all_lines = [line.copy() for line in lines]

    # Sort detections by confidence score in descending order
    all_lines.sort(key=lambda x: x['confidence'], reverse=True)

    for i in range(len(all_lines)):
        # Iterate through the rest of the boxes to apply suppression
        for j in range(i + 1, len(all_lines)):
            iou = calculate_iou(all_lines[i]['bbox'], all_lines[j]['bbox'])

            # If overlap is above the threshold, apply the Gaussian penalty
            if iou > iou_threshold:
                weight = np.exp(-(iou * iou) / sigma)
                all_lines[j]['confidence'] *= weight

    # Filter out lines that have fallen below the minimum confidence threshold
    final_lines = [line for line in all_lines if line['confidence'] >= min_confidence]

    logging.info(f"Soft-NMS reduced lines from {len(lines)} to {len(final_lines)}.")
    return final_lines


def apply_aspect_ratio_filter(
        merged_lines: List[Dict],
        max_hw_ratio_horizontal: float,
        max_wh_ratio_vertical: float
) -> List[Dict]:
    """
    Filters merged text lines based on their aspect ratio, which can remove
    many common false positives.
    (This function remains unchanged)
    """
    if not merged_lines:
        return []

    filtered_results = []
    for line in merged_lines:
        bbox = np.array(line['bbox'])
        w = np.max(bbox[:, 0]) - np.min(bbox[:, 0])
        h = np.max(bbox[:, 1]) - np.min(bbox[:, 1])

        if w == 0 or h == 0:
            continue  # Avoid division by zero for invalid boxes

        orientation = line.get('orientation', 0)
        keep = True

        # Filter horizontal lines that are "too tall" for their width
        if orientation == 0 and max_hw_ratio_horizontal > 0:
            if (h / w) > max_hw_ratio_horizontal:
                keep = False
                logging.debug(
                    f"Dropped horizontal line (h/w={(h / w):.2f} > {max_hw_ratio_horizontal}): '{line['text']}'")

        # Filter vertical lines that are "too wide" for their height
        elif orientation == 90 and max_wh_ratio_vertical > 0:
            if (w / h) > max_wh_ratio_vertical:
                keep = False
                logging.debug(f"Dropped vertical line (w/h={(w / h):.2f} > {max_wh_ratio_vertical}): '{line['text']}'")

        if keep:
            filtered_results.append(line)

    if len(merged_lines) > len(filtered_results):
        logging.info(f"Aspect ratio filter reduced lines from {len(merged_lines)} to {len(filtered_results)}.")

    return filtered_results