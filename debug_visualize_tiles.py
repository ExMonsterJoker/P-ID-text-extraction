# debug_visualize_tiles.py
import os
import json
import cv2
import logging
import argparse
from glob import glob
import numpy as np
from pathlib import Path

# --- Configuration ---
# Directory containing the sliced tile images
TILES_BASE_DIR = "data/processed/tiles"
# Directory containing the raw OCR metadata for each tile
OCR_METADATA_DIR = "data/processed/metadata/detection_metadata"
# Directory containing the original, full-size source images
FULL_IMAGE_INPUT_DIR = "data/raw"
# Directory where the debug images will be saved
DEBUG_OUTPUT_DIR = "data/outputs/debug_tile_visualizations"
# Directory for the new full-image visualizations
FULL_IMAGE_OUTPUT_DIR = "data/outputs/debug_full_image_visualizations"

# --- Setup ---
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(FULL_IMAGE_OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image from a list of points.
    """
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_label(img, text, bbox, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """
    Draws a text label inside the bounding box with a background for better visibility.
    Adjusts the text position dynamically based on the box size.
    """
    # Calculate the bounding box dimensions
    min_x = min(p[0] for p in bbox)
    min_y = min(p[1] for p in bbox)
    max_x = max(p[0] for p in bbox)
    max_y = max(p[1] for p in bbox)

    # Create a background for the label
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)[0]
    box_x = min_x
    box_y = min_y - text_size[1] - 5  # Adjust above the box
    if box_y < 0:  # Adjust if it overflows beyond the image
        box_y = min_y + 5

    cv2.rectangle(img, (box_x, box_y), (box_x + text_size[0], box_y + text_size[1]), color, -1)  # Fill background
    cv2.putText(img, text, (box_x, box_y + text_size[1]), cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 0), thickness,
                cv2.LINE_AA)  # Draw text


def format_detection_label(detection, mode="detailed"):
    """
    Formats the detection information into a readable label.

    Args:
        detection: Dictionary containing detection information
        mode: "detailed" for full info, "compact" for shorter labels

    Returns:
        Formatted string for the label
    """
    text = detection.get('text', 'N/A')
    confidence = detection.get('confidence', 0)
    rotation_angle = detection.get('rotation_angle', 0)

    # Format confidence as percentage
    conf_percent = confidence * 100

    # Format rotation angle with degree symbol
    rotation_text = f"{rotation_angle}°"

    if mode == "detailed":
        return f"'{text}' | {conf_percent:.1f}% | {rotation_text}"
    elif mode == "compact":
        return f"'{text}' ({conf_percent:.0f}%, {rotation_text})"
    else:
        return f"'{text}'"


def visualize_tile_results(image_name):
    """
    (Tile Mode) Loads raw OCR detections for each tile and overlays them on the tile images
    to help diagnose issues at the source.
    """
    image_dir = os.path.join(TILES_BASE_DIR, image_name)
    if not os.path.exists(image_dir):
        logging.error(f"Directory not found for image: {image_name} in {TILES_BASE_DIR}")
        return

    logging.info(f"\n--- Processing tiles for: {image_name} ---")

    specific_output_dir = os.path.join(DEBUG_OUTPUT_DIR, image_name)
    os.makedirs(specific_output_dir, exist_ok=True)

    tile_image_paths = glob(os.path.join(image_dir, "tile_*.png")) + glob(os.path.join(image_dir, "tile_*.jpg"))
    if not tile_image_paths:
        logging.warning(f"No tile images found in {image_dir}")
        return

    for tile_path in tile_image_paths:
        try:
            tile_filename = os.path.basename(tile_path)
            image = cv2.imread(tile_path)
            tile_id = Path(tile_filename).stem.replace("tile_", "")
            ocr_json_path = os.path.join(OCR_METADATA_DIR, image_name, f"{tile_id}_ocr.json")

            if os.path.exists(ocr_json_path):
                with open(ocr_json_path, 'r') as f:
                    detections = json.load(f)
                for det in detections:
                    if not (bbox := det.get('bbox')): continue

                    # Get orientation for color coding
                    orientation = det.get('rotation_angle', 0)
                    box_color = (0, 165, 255) if orientation == 90 else (0, 255, 0)  # Orange for vertical

                    draw_bounding_box(image, bbox, color=box_color)

                    # Create formatted label with confidence and orientation
                    label = format_detection_label(det, mode="detailed")
                    draw_label(image, label, bbox)
            else:
                logging.warning(f"Missing OCR JSON for tile: {tile_filename}")
                cv2.putText(image, "No OCR Data", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            output_tile_path = os.path.join(specific_output_dir, tile_filename)
            cv2.imwrite(output_tile_path, image)
        except Exception as e:
            logging.error(f"Error processing tile {tile_path}: {e}", exc_info=True)
    logging.info(f"Saved {len(tile_image_paths)} debug tiles to: {specific_output_dir}")


def visualize_full_image_results(image_name):
    """
    (Full Mode) Loads raw OCR detections from all tile JSONs, calculates their
    global positions, and draws them on the original full-size source image.
    """
    logging.info(f"--- Processing full image visualization for: {image_name} ---")

    # Find the original source image
    source_image_path = None
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        found = glob(os.path.join(FULL_IMAGE_INPUT_DIR, f"{image_name}{ext[-4:]}"))
        if found:
            source_image_path = found[0]
            break

    if not source_image_path:
        logging.error(f"Could not find source image for '{image_name}' in '{FULL_IMAGE_INPUT_DIR}'")
        return

    # Path to the directory containing all tile JSONs for this image
    image_meta_dir = os.path.join(OCR_METADATA_DIR, image_name)
    if not os.path.isdir(image_meta_dir):
        logging.error(f"Metadata directory not found at: {image_meta_dir}")
        return

    tile_json_paths = glob(os.path.join(image_meta_dir, "*_ocr.json"))
    if not tile_json_paths:
        logging.warning(f"No OCR JSON files found in {image_meta_dir}")
        return

    try:
        logging.info(f"Loading source image: {source_image_path}")
        image = cv2.imread(source_image_path)
        if image is None:
            logging.error(f"Failed to load image at {source_image_path}")
            return

        detection_count = 0
        logging.info(f"Processing {len(tile_json_paths)} tile JSON files...")
        for json_path in tile_json_paths:
            with open(json_path, 'r') as f:
                detections = json.load(f)

            for det in detections:
                tile_bbox = det.get('bbox')
                tile_coords = det.get('tile_coordinates')

                if not tile_bbox or not tile_coords:
                    continue

                # Calculate global coordinates by adding the tile's origin
                tile_x_origin, tile_y_origin = tile_coords[0], tile_coords[1]
                global_bbox = [[pt[0] + tile_x_origin, pt[1] + tile_y_origin] for pt in tile_bbox]

                # Green for horizontal (0°), Orange for vertical (90°)
                orientation = det.get('rotation_angle', 0)
                box_color = (0, 165, 255) if orientation == 90 else (0, 255, 0)

                # Calculate thickness based on image size
                thickness = max(1, image.shape[1] // 10000)

                draw_bounding_box(image, global_bbox, color=box_color, thickness=thickness)

                # Use compact mode for full image to avoid overcrowding
                label = format_detection_label(det, mode="compact")
                draw_label(
                    image, label, global_bbox,
                    font_scale=max(0.1, image.shape[1] / 10000),
                    color=(255, 0, 0),
                    thickness=thickness
                )
                detection_count += 1

        output_path = os.path.join(FULL_IMAGE_OUTPUT_DIR, f"{image_name}_raw_ocr_visualization.jpg")
        logging.info(f"Saving visualization with {detection_count} raw detections to: {output_path}")
        cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    except Exception as e:
        logging.error(f"An error occurred during full image visualization: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Debug visualizer for OCR results. Can operate on individual tiles or a full reconstructed image.")
    parser.add_argument(
        "-i", "--image-name", type=str, required=True,
        help="The base name of the P&ID image to process (e.g., DURI-WTPF03GS000-PRO-PID-IBU-0057-00). Required for both modes."
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="tile", choices=["tile", "full"],
        help="Visualization mode: 'tile' for individual raw tiles, 'full' for all raw tile detections on the full image."
    )
    args = parser.parse_args()

    if args.mode == 'tile':
        visualize_tile_results(image_name=args.image_name)
    elif args.mode == 'full':
        visualize_full_image_results(image_name=args.image_name)
