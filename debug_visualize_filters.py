# debug_visualize_filters.py

import os
import json
import cv2
import logging
from glob import glob
import numpy as np

# --- Configuration ---
GROUPED_TEXT_DIR = "data/processed/metadata/final_grouped_text"
RAW_IMG_DIR = "data/raw"
DEBUG_VIS_DIR = "src/grouping/data/outputs/debug_filter_visualizations"  # New output directory

# --- Visualization Settings ---
COLOR_KEPT = (0, 255, 0)  # Green for kept boxes
COLOR_FILTERED = (0, 0, 255)  # Red for filtered boxes
THICKNESS_KEPT = 2
THICKNESS_FILTERED = 1

# --- Setup ---
os.makedirs(DEBUG_VIS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def draw_bounding_box(img, bbox, color, thickness, is_dashed=False):
    """Draws a bounding box, either solid or dashed."""
    pts = np.array(bbox, dtype=np.int32)
    if is_dashed:
        # Draw a dashed polyline for the bounding box
        num_points = len(pts)
        for i in range(num_points):
            start_point = tuple(pts[i])
            end_point = tuple(pts[(i + 1) % num_points])
            cv2.line(img, start_point, end_point, color, thickness, lineType=cv2.LINE_AA, shift=0)
    else:
        # Draw a solid polyline
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def get_box_identifier(line_dict):
    """Creates a unique, hashable identifier for a detection."""
    bbox = line_dict['bbox']
    # Flatten the bbox coordinates and join with text to create a unique enough ID
    return f"{line_dict['text']}_{'_'.join(map(str, [c for p in bbox for c in p]))}"


def visualize_filter_effects():
    """
    Loads both unfiltered and filtered results and visualizes the difference.
    - Kept boxes are drawn in solid green.
    - Filtered-out boxes are drawn in dashed red.
    """
    unfiltered_files = glob(os.path.join(GROUPED_TEXT_DIR, '*_unfiltered.json'))

    if not unfiltered_files:
        logging.warning(f"No unfiltered detection files found in '{GROUPED_TEXT_DIR}'. "
                        f"Please run the grouping pipeline first.")
        return

    logging.info(f"Found {len(unfiltered_files)} file sets to visualize.")

    for unfiltered_path in unfiltered_files:
        try:
            base_name = os.path.basename(unfiltered_path).replace('_grouped_text_unfiltered.json', '')
            filtered_path = os.path.join(GROUPED_TEXT_DIR, f"{base_name}_grouped_text.json")

            if not os.path.exists(filtered_path):
                logging.warning(f"Missing filtered file for {base_name}. Skipping.")
                continue

            # Load the corresponding raw image
            image_path = None
            for ext in ["*.jpg", "*.png", "*.tiff"]:
                candidates = glob(os.path.join(RAW_IMG_DIR, f"{base_name}{ext[-4:]}"))
                if candidates:
                    image_path = candidates[0]
                    break

            if not image_path:
                logging.warning(f"Could not find a matching raw image for '{base_name}' in '{RAW_IMG_DIR}'")
                continue

            image = cv2.imread(image_path)
            logging.info(f"Visualizing filter effects for: {base_name}")

            # Load both sets of detections
            with open(unfiltered_path, 'r') as f:
                unfiltered_lines = json.load(f)
            with open(filtered_path, 'r') as f:
                filtered_lines = json.load(f)

            # Create a set of identifiers for the lines that were kept
            kept_line_ids = {get_box_identifier(line) for line in filtered_lines}

            # Iterate through all unfiltered lines and draw them
            for line in unfiltered_lines:
                identifier = get_box_identifier(line)
                bbox = line.get('bbox')
                if not bbox: continue

                if identifier in kept_line_ids:
                    # This box was kept: draw solid green
                    draw_bounding_box(image, bbox, color=COLOR_KEPT, thickness=THICKNESS_KEPT, is_dashed=False)
                else:
                    # This box was filtered out: draw dashed red
                    draw_bounding_box(image, bbox, color=COLOR_FILTERED, thickness=THICKNESS_FILTERED, is_dashed=True)

            out_path = os.path.join(DEBUG_VIS_DIR, f"{base_name}_filter_debug.jpg")
            cv2.imwrite(out_path, image)
            logging.info(f"Saved filter debug visualization to: {out_path}\n")

        except Exception as e:
            logging.error(f"Error processing {unfiltered_path}: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_filter_effects()