# debug_visualize_tiles.py
import os
import json
import cv2
import logging
import argparse
from glob import glob
import numpy as np

# --- Configuration ---
# Directory containing the sliced tile images
TILES_BASE_DIR = "data/processed/tiles"
# Directory containing the raw OCR metadata for each tile
OCR_METADATA_DIR = "data/processed/metadata/detection_metadata"
# Directory where the debug images will be saved
DEBUG_OUTPUT_DIR = "data/outputs/debug_tile_visualizations"

# --- Setup ---
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image from a list of points.
    """
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_label(img, text, bbox, font_scale=0.5, color=(255, 0, 0), thickness=1):
    """
    Draws a text label at the top-left corner of the bounding box.
    """
    top_left_point = min(bbox, key=lambda p: (p[0], p[1]))
    x, y = int(top_left_point[0]), int(top_left_point[1])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def visualize_tile_results(image_name=None):
    """
    Loads raw OCR detections for each tile and overlays them on the tile images
    to help diagnose issues at the source.

    Args:
        image_name (str, optional): The specific P&ID base name to process.
                                    If None, all images will be processed.
    """
    if image_name:
        image_dirs = [os.path.join(TILES_BASE_DIR, image_name)]
        if not os.path.exists(image_dirs[0]):
            logging.error(f"Directory not found for image: {image_name}")
            return
    else:
        # Get all P&ID directories from the tiles folder
        image_dirs = [d for d in glob(os.path.join(TILES_BASE_DIR, "*")) if os.path.isdir(d)]

    if not image_dirs:
        logging.warning(f"No sliced tile directories found in '{TILES_BASE_DIR}'.")
        return

    logging.info(f"Found {len(image_dirs)} P&ID image(s) to visualize.")

    for image_dir in image_dirs:
        base_name = os.path.basename(image_dir)
        logging.info(f"\n--- Processing tiles for: {base_name} ---")

        specific_output_dir = os.path.join(DEBUG_OUTPUT_DIR, base_name)
        os.makedirs(specific_output_dir, exist_ok=True)

        tile_image_paths = glob(os.path.join(image_dir, "tile_*.png")) + glob(os.path.join(image_dir, "tile_*.jpg"))

        if not tile_image_paths:
            logging.warning(f"No tile images found in {image_dir}")
            continue

        for tile_path in tile_image_paths:
            try:
                tile_filename = os.path.basename(tile_path)
                image = cv2.imread(tile_path)

                tile_id = os.path.splitext(tile_filename.replace("tile_", ""))[0]
                ocr_json_path = os.path.join(OCR_METADATA_DIR, f"{tile_id}_ocr.json")

                if os.path.exists(ocr_json_path):
                    # If JSON exists, load it and draw detections
                    with open(ocr_json_path, 'r') as f:
                        detections = json.load(f)

                    for det in detections:
                        bbox = det.get('bbox')
                        if not bbox: continue
                        draw_bounding_box(image, bbox)

                        label_parts = [det.get('text', '')]
                        if 'confidence' in det:
                            label_parts.append(f"conf: {det['confidence']:.2f}")
                        label = " | ".join(label_parts)
                        draw_label(image, label, bbox)
                else:
                    # FIX: If JSON does not exist, label the image accordingly
                    logging.warning(f"Missing OCR JSON for tile: {tile_filename}")
                    cv2.putText(image, "No OCR Data", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Save the visualized tile regardless of whether it had detections
                output_tile_path = os.path.join(specific_output_dir, tile_filename)
                cv2.imwrite(output_tile_path, image)

            except Exception as e:
                logging.error(f"Error processing tile {tile_path}: {e}", exc_info=True)

        logging.info(f"Saved {len(tile_image_paths)} debug tiles to: {specific_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug visualizer for raw OCR results on individual tiles.")
    parser.add_argument("-i", "--image-name", type=str,
                        help="Optional: The base name of a single P&ID image to process (e.g., DURI-OTPF0...).")
    args = parser.parse_args()

    visualize_tile_results(image_name=args.image_name)
