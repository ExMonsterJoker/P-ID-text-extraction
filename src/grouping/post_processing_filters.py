# src/grouping/post_processing_filters.py
import numpy as np
import logging
from typing import List, Dict


def apply_aspect_ratio_filter(
        merged_lines: List[Dict],
        max_hw_ratio_horizontal: float,
        max_wh_ratio_vertical: float
) -> List[Dict]:
    """
    Filters merged text lines based on their aspect ratio, which can remove
    many common false positives.

    Args:
        merged_lines: A list of merged text line dictionaries.
        max_hw_ratio_horizontal: For horizontal text (0°), the maximum allowed
                                 height-to-width ratio.
        max_wh_ratio_vertical: For vertical text (90°), the maximum allowed
                               width-to-height ratio.

    Returns:
        A list of text lines that pass the filter.
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
