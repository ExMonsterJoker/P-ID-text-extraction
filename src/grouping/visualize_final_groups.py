# src/grouping/visualize_final_groups.py

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
CORE_TILES_DIR = "data/processed/metadata/core_tile_metadata"  # NEW: Core tiles directory
OUT_IMG_DIR = "data/outputs/final_visualizations"

# --- Visualization Settings ---
SHOW_TILE_GRID = True
SHOW_CORE_TILES = True  # NEW: Option to show core tiles
SHOW_ORIGINAL_TILES = False  # NEW: Option to show original tiles

# --- Setup ---
os.makedirs(OUT_IMG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def find_matching_image(base_name, image_dir):
    """
    Find matching image file for the given base name
    """
    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

    for ext in extensions:
        # Try exact match first
        image_path = os.path.join(image_dir, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path

        # Try case-insensitive match
        image_path = os.path.join(image_dir, f"{base_name}{ext.upper()}")
        if os.path.exists(image_path):
            return image_path

    # If no exact match, try glob pattern matching
    for ext in extensions:
        pattern = os.path.join(image_dir, f"{base_name}{ext}")
        candidates = glob(pattern)
        if candidates:
            return candidates[0]

        # Try case-insensitive glob
        pattern = os.path.join(image_dir, f"{base_name}{ext.upper()}")
        candidates = glob(pattern)
        if candidates:
            return candidates[0]

    return None


def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_tile_boundary(img, tile_coords, color=(255, 255, 0), thickness=1, label=None):
    """
    Draw tile boundary with optional label
    """
    x1, y1, x2, y2 = map(int, tile_coords)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Add label if provided
    if label:
        cv2.putText(img, label, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_core_tiles(img, core_tiles_data, show_original=False, show_core=True):
    """
    Draw core tiles and optionally original tiles
    """
    core_tiles = core_tiles_data.get('core_tiles', [])

    for tile in core_tiles:
        tile_id = tile.get('tile_id', 'unknown')

        # Draw original tile boundary (in yellow, thinner line)
        if show_original:
            original_coords = tile.get('original_coordinates')
            if original_coords:
                draw_tile_boundary(img, original_coords,
                                   color=(0, 255, 255), thickness=1,
                                   label=f"orig_{tile_id}")

        # Draw core tile boundary (in cyan, thicker line)
        if show_core:
            core_coords = tile.get('core_coordinates')
            if core_coords:
                draw_tile_boundary(img, core_coords,
                                   color=(255, 255, 0), thickness=2,
                                   label=f"core_{tile_id}")

        # Add overlap information as text
        overlap_width = tile.get('overlap_width', 0)
        overlap_height = tile.get('overlap_height', 0)
        if overlap_width > 0 or overlap_height > 0:
            core_coords = tile.get('core_coordinates', [0, 0, 0, 0])
            x1, y1 = int(core_coords[0]), int(core_coords[1])
            overlap_text = f"OW:{overlap_width} OH:{overlap_height}"
            cv2.putText(img, overlap_text, (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def load_core_tiles_metadata(base_name):
    """
    Load core tiles metadata for the given base name
    """
    core_tiles_path = os.path.join(CORE_TILES_DIR, f"{base_name}_core_tiles.json")

    if not os.path.exists(core_tiles_path):
        logging.warning(f"Core tiles metadata not found: {core_tiles_path}")
        return None

    try:
        with open(core_tiles_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading core tiles metadata: {e}")
        return None


def draw_label(img, text, bbox, font_scale=0.7, color=(255, 0, 0), thickness=2):
    top_left_point = min(bbox, key=lambda p: p[1])
    x, y = int(top_left_point[0]), int(top_left_point[1])
    cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def list_available_files():
    """
    Debug function to list available files
    """
    logging.info("=== Available Files Debug ===")

    # List detection files
    detection_files = glob(os.path.join(FINAL_GROUPED_DIR, '*_grouped_text.json'))
    logging.info(f"Detection files found: {len(detection_files)}")
    for f in detection_files[:5]:  # Show first 5
        base_name = os.path.basename(f).replace('_grouped_text.json', '')
        logging.info(f"  - {base_name}")

    # List core tiles files
    core_tiles_files = glob(os.path.join(CORE_TILES_DIR, '*_core_tiles.json'))
    logging.info(f"Core tiles files found: {len(core_tiles_files)}")
    for f in core_tiles_files[:5]:  # Show first 5
        base_name = os.path.basename(f).replace('_core_tiles.json', '')
        logging.info(f"  - {base_name}")

    # List image files
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob(os.path.join(RAW_IMG_DIR, pattern)))
        all_images.extend(glob(os.path.join(RAW_IMG_DIR, pattern.upper())))

    logging.info(f"Image files found: {len(all_images)}")
    for f in all_images[:5]:  # Show first 5
        logging.info(f"  - {os.path.basename(f)}")

    logging.info("=== End Debug ===\n")


def visualize_final_results(show_grid=False, show_core_tiles=True, show_original_tiles=False):
    """
    Loads final grouped text, visualizes it, and indicates text orientation.
    Now uses core tiles instead of tiles_metadata.yaml
    """
    # Debug: List available files
    list_available_files()

    detection_files = glob(os.path.join(FINAL_GROUPED_DIR, '*_grouped_text.json'))

    if not detection_files:
        logging.warning(f"No grouped detection files found in '{FINAL_GROUPED_DIR}'.")
        return

    logging.info(f"Found {len(detection_files)} grouped detection files to visualize.")

    successful_matches = 0
    failed_matches = 0

    for det_path in detection_files:
        try:
            base_name = os.path.basename(det_path).replace('_grouped_text.json', '')
            logging.info(f"Processing: {base_name}")

            # Use improved image finding function
            image_path = find_matching_image(base_name, RAW_IMG_DIR)

            if not image_path:
                logging.warning(f"Could not find matching image for '{base_name}' in '{RAW_IMG_DIR}'")
                failed_matches += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                failed_matches += 1
                continue

            logging.info(f"Successfully matched: {base_name} -> {os.path.basename(image_path)}")

            # NEW: Draw core tiles instead of old tile grid
            if show_grid:
                core_tiles_data = load_core_tiles_metadata(base_name)
                if core_tiles_data:
                    draw_core_tiles(image, core_tiles_data,
                                    show_original=show_original_tiles,
                                    show_core=show_core_tiles)
                    logging.info("Drew core tiles grid on the image.")
                else:
                    logging.warning(f"Could not find core tiles metadata for {base_name}")

            # Load and draw grouped text detections
            with open(det_path, 'r') as f:
                grouped_lines = json.load(f)

            detection_count = 0
            for line in grouped_lines:
                bbox = line.get('bbox')
                if not bbox:
                    continue

                # Get orientation and set color accordingly
                orientation = line.get('orientation', 0)
                # Use Orange for vertical (90°) and Green for horizontal (0°)
                box_color = (0, 165, 255) if orientation == 90 else (0, 255, 0)

                draw_bounding_box(image, bbox, color=box_color)

                # Add orientation to the display label
                label_parts = []
                if 'text' in line:
                    # Truncate long text for display
                    text = line['text'][:50] + "..." if len(line['text']) > 50 else line['text']
                    label_parts.append(text)
                if 'confidence' in line:
                    label_parts.append(f"conf: {line['confidence']:.2f}")
                label_parts.append(f"rot: {orientation}°")

                label = ' | '.join(label_parts)
                draw_label(image, label, bbox)
                detection_count += 1

            # Add legend
            legend_y = 30
            cv2.putText(image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Green: Horizontal text", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(image, "Orange: Vertical text", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255), 2)
            if show_grid:
                cv2.putText(image, "Cyan: Core tiles", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 2)
                if show_original_tiles:
                    cv2.putText(image, "Yellow: Original tiles", (10, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)

            out_path = os.path.join(OUT_IMG_DIR, f"{base_name}_final_overlay.jpg")
            cv2.imwrite(out_path, image)
            logging.info(f"Saved final overlay with {detection_count} detections to: {out_path}")
            successful_matches += 1

        except Exception as e:
            logging.error(f"Error processing {det_path}: {e}", exc_info=True)
            failed_matches += 1

    logging.info(f"\n=== Summary ===")
    logging.info(f"Successful matches: {successful_matches}")
    logging.info(f"Failed matches: {failed_matches}")
    logging.info(f"Total files processed: {len(detection_files)}")


if __name__ == '__main__':
    visualize_final_results(
        show_grid=SHOW_TILE_GRID,
        show_core_tiles=SHOW_CORE_TILES,
        show_original_tiles=SHOW_ORIGINAL_TILES
    )
