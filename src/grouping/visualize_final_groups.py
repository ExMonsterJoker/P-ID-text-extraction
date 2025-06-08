# visualize_final_groups.py

import os
import json
import cv2
import logging
import yaml
from glob import glob
import numpy as np

# --- Configuration ---
FINAL_GROUPED_DIR = "data/processed/metadata/final_grouped_text"
RAW_IMG_DIR = "data/raw"
TILES_BASE_DIR = "data/processed/tiles"
OUT_IMG_DIR = "data/outputs/final_visualizations"

# --- Visualization Settings ---
SHOW_TILE_GRID = True

# --- Setup ---
os.makedirs(OUT_IMG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_tile_boundary(img, tile_coords, color=(255, 255, 0), thickness=1):
    x1, y1, x2, y2 = map(int, tile_coords)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_label(img, text, bbox, font_scale=0.7, color=(255, 0, 0), thickness=2):
    top_left_point = min(bbox, key=lambda p: p[1])
    x, y = int(top_left_point[0]), int(top_left_point[1])
    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def visualize_final_results(show_grid=False):
    """
    Loads final grouped text, visualizes it, and indicates text orientation.
    """
    detection_files = glob(os.path.join(FINAL_GROUPED_DIR, '*_grouped_text.json'))

    if not detection_files:
        logging.warning(f"No grouped detection files found in '{FINAL_GROUPED_DIR}'.")
        return

    logging.info(f"Found {len(detection_files)} grouped detection files to visualize.")

    for det_path in detection_files:
        try:
            base_name = os.path.basename(det_path).replace('_grouped_text.json', '')

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
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue

            logging.info(f"Visualizing results for: {base_name}")

            if show_grid:
                tile_meta_path = os.path.join(TILES_BASE_DIR, base_name, "tiles_metadata.yaml")
                if os.path.exists(tile_meta_path):
                    with open(tile_meta_path, 'r') as f:
                        tile_data = yaml.safe_load(f)
                    for tile in tile_data.get('tiles', []):
                        draw_tile_boundary(image, tile['coordinates'])
                    logging.info("Drew tile grid on the image.")
                else:
                    logging.warning(f"Could not find tile metadata at {tile_meta_path} to draw grid.")

            with open(det_path, 'r') as f:
                grouped_lines = json.load(f)

            for line in grouped_lines:
                bbox = line.get('bbox')
                if not bbox: continue

                # NEW: Get orientation and set color accordingly
                # Default to 0 if the 'orientation' key is missing
                orientation = line.get('orientation', 0)
                # Use Green for horizontal (0d) and Orange for vertical (90d)
                box_color = (0, 165, 255) if orientation == 90 else (0, 255, 0)

                draw_bounding_box(image, bbox, color=box_color)

                # NEW: Add orientation to the display label
                label_parts = []
                if 'text' in line:
                    label_parts.append(line['text'])
                if 'confidence' in line:
                    label_parts.append(f"conf: {line['confidence']:.2f}")
                label_parts.append(f"rot: {orientation}d")  # Add rotation degree

                label = ' | '.join(label_parts)
                draw_label(image, label, bbox)

            out_path = os.path.join(OUT_IMG_DIR, f"{base_name}_final_overlay.jpg")
            cv2.imwrite(out_path, image)
            logging.info(f"Saved final overlay to: {out_path}\n")

        except Exception as e:
            logging.error(f"Error processing {det_path}: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_final_results(show_grid=SHOW_TILE_GRID)
