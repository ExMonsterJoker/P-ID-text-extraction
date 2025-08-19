# D:/Stuff/P-ID-text-extraction/src/grouping/visualize_grouping.py

import os
import sys
import json
import logging
import argparse
import cv2
import numpy as np
import glob

# --- Solution for the import problem ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from configs import get_config, get_config_value
except ImportError:
    print("Warning: 'configs' module not found. Using default values.")

    # Set project_root for the fallback case
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


    # Dummy functions to avoid crashing
    def get_config_value(key, default):
        return default


    def get_config(key):
        return {}

# --- Constants ---
VISUALIZATION_OUTPUT_DIR = "data/outputs/grouping_visualizations"


def setup_logging(is_debug: bool):
    """Configures logging for the script."""
    log_level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        force=True,
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_raw_detections(image_name: str, detection_metadata_dir: str) -> list:
    """Loads all individual raw detections for a given image name."""
    subfolder_path = os.path.join(detection_metadata_dir, image_name)
    if not os.path.isdir(subfolder_path):
        logging.warning(f"Raw detection subfolder not found for '{image_name}': {subfolder_path}")
        return []

    all_boxes = []
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith('.json')]
    logging.debug(f"Found {len(json_files)} raw detection files in {subfolder_path}.")

    for filename in json_files:
        file_path = os.path.join(subfolder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_boxes.extend(data)
                elif isinstance(data, dict) and data:
                    all_boxes.append(data)
        except Exception as e:
            logging.error(f"Error reading raw detection file {file_path}: {e}")

    logging.info(f"Loaded a total of {len(all_boxes)} raw boxes for '{image_name}'.")
    return all_boxes


def load_grouped_detections(image_name: str, grouped_metadata_dir: str) -> list:
    """Loads the final grouped detections for a given image name."""
    grouped_file_path = os.path.join(grouped_metadata_dir, f"{image_name}_grouped.json")
    if not os.path.exists(grouped_file_path):
        logging.warning(f"Grouped detection file not found for '{image_name}': {grouped_file_path}")
        return []

    try:
        with open(grouped_file_path, 'r', encoding='utf-8') as f:
            grouped_boxes = json.load(f)
        logging.info(f"Loaded {len(grouped_boxes)} grouped boxes for '{image_name}'.")
        return grouped_boxes
    except Exception as e:
        logging.error(f"Error reading grouped detection file {grouped_file_path}: {e}")
        return []


def get_box_orientation_and_color(bbox_points: list) -> tuple:
    """
    Determine if a bounding box is horizontal or vertical and return appropriate color.
    Returns (orientation, color) where orientation is 'horizontal' or 'vertical'
    and color is a BGR tuple.
    """
    if not bbox_points or len(bbox_points) < 4:
        return 'unknown', (128, 128, 128)  # Gray for unknown

    # Calculate width and height from bbox points
    x_coords = [point[0] for point in bbox_points]
    y_coords = [point[1] for point in bbox_points]

    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    if width > height:
        return 'horizontal', (0, 255, 0)  # Green for horizontal
    else:
        return 'vertical', (0, 0, 255)   # Red for vertical


def draw_boxes_on_image_with_orientation(image: np.ndarray, boxes: list, thickness: int = 2) -> np.ndarray:
    """Draws a list of bounding boxes onto an image with orientation-based colors."""
    vis_image = image.copy()
    img_height, img_width = image.shape[:2]

    logging.debug(f"Image dimensions: {img_width}x{img_height}")

    boxes_drawn = 0
    boxes_skipped = 0
    orientation_counts = {'horizontal': 0, 'vertical': 0, 'unknown': 0}

    for box_data in boxes:
        if 'bbox' not in box_data:
            logging.debug(f"Skipping item without 'bbox' key: {box_data.get('text', 'N/A')}")
            boxes_skipped += 1
            continue

        bbox_points = box_data['bbox']
        if not bbox_points or len(bbox_points) < 4:
            logging.debug("Skipping box with insufficient bbox points")
            boxes_skipped += 1
            continue

        # Get orientation and color before coordinate transformation
        orientation, color = get_box_orientation_and_color(bbox_points)
        orientation_counts[orientation] += 1

        # Transform coordinates from tile space to original image space
        if 'tile_coordinates' in box_data:
            # tile_coordinates format: [x_start, y_start, x_end, y_end]
            tile_coords = box_data['tile_coordinates']
            tile_x_start, tile_y_start = tile_coords[0], tile_coords[1]

            logging.debug(f"Tile coordinates: {tile_coords}")
            logging.debug(f"Original bbox points: {bbox_points}")

            # Transform each point from tile coordinates to original image coordinates
            transformed_points = []
            for point in bbox_points:
                orig_x = int(tile_x_start + point[0])
                orig_y = int(tile_y_start + point[1])

                # Clamp coordinates to image boundaries
                orig_x = max(0, min(orig_x, img_width - 1))
                orig_y = max(0, min(orig_y, img_height - 1))

                transformed_points.append([orig_x, orig_y])

            logging.debug(f"Transformed points: {transformed_points}")
            points = np.array(transformed_points, dtype=np.int32)
        else:
            # If no tile_coordinates, assume bbox is already in original image space
            # But still clamp to image boundaries
            logging.debug("No tile_coordinates found, using bbox as-is with clamping")
            clamped_points = []
            for point in bbox_points:
                x = max(0, min(int(point[0]), img_width - 1))
                y = max(0, min(int(point[1]), img_height - 1))
                clamped_points.append([x, y])

            points = np.array(clamped_points, dtype=np.int32)

        # Only draw if we have valid points
        if len(points) >= 3:  # Need at least 3 points for a polygon
            cv2.polylines(vis_image, [points], isClosed=True, color=color, thickness=thickness)
            boxes_drawn += 1
        else:
            logging.debug(f"Skipping box with insufficient transformed points: {len(points)}")
            boxes_skipped += 1

    logging.info(f"Drew {boxes_drawn} boxes, skipped {boxes_skipped} boxes")
    logging.info(f"Orientation breakdown: {orientation_counts['horizontal']} horizontal (green), "
                f"{orientation_counts['vertical']} vertical (red), {orientation_counts['unknown']} unknown (gray)")
    return vis_image


def draw_boxes_on_image(image: np.ndarray, boxes: list, color: tuple, thickness: int) -> np.ndarray:
    """Draws a list of bounding boxes onto an image."""
    # Keep the old function for backward compatibility, but redirect to orientation-based drawing
    return draw_boxes_on_image_with_orientation(image, boxes, thickness)


def main():
    """Main function to run the visualization script for all images."""
    parser = argparse.ArgumentParser(description="Visualize bounding boxes for all images before and after grouping.")
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging for more detailed output."
    )
    parser.add_argument(
        '--image',
        type=str,
        help="Process only a specific image by name (without extension)"
    )
    args = parser.parse_args()

    setup_logging(args.debug)

    # --- Load Configuration ---
    raw_image_dir = os.path.join(project_root, "data", "raw")

    # Use fallback directories if config fails
    try:
        detection_metadata_dir = os.path.join(project_root, get_config_value("data_loader.detection_metadata_dir",
                                                                             "data/processed/detection_metadata"))
        grouped_metadata_dir = os.path.join(project_root, get_config_value("data_loader.group_detection_metadata_dir",
                                                                           "data/processed/grouped_metadata"))
    except:
        detection_metadata_dir = os.path.join(project_root, "data/processed/detection_metadata")
        grouped_metadata_dir = os.path.join(project_root, "data/processed/grouped_metadata")

    output_dir = os.path.join(project_root, VISUALIZATION_OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Visualizations will be saved to: {output_dir}")
    logging.info(f"Raw image directory: {raw_image_dir}")
    logging.info(f"Detection metadata directory: {detection_metadata_dir}")
    logging.info(f"Grouped metadata directory: {grouped_metadata_dir}")

    # --- Find and Process All Images ---
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(raw_image_dir, f"*{ext}")))
        # Also try uppercase extensions
        image_paths.extend(glob.glob(os.path.join(raw_image_dir, f"*{ext.upper()}")))

    if not image_paths:
        logging.error(f"No images found in the raw data directory: {raw_image_dir}")
        return

    # Filter for specific image if requested
    if args.image:
        image_paths = [path for path in image_paths if args.image in os.path.basename(path)]
        if not image_paths:
            logging.error(f"No image found matching '{args.image}'")
            return

    logging.info(f"Found {len(image_paths)} images to process.")

    for i, image_path in enumerate(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        logging.info(f"\n--- [{i + 1}/{len(image_paths)}] Processing image: {image_name} ---")

        base_image = cv2.imread(image_path)
        if base_image is None:
            logging.error(f"Failed to read image file, skipping: {image_path}")
            continue

        logging.info(f"Image loaded successfully. Dimensions: {base_image.shape}")

        # --- Visualize Pre-Grouping Boxes ---
        raw_boxes = load_raw_detections(image_name, detection_metadata_dir)
        if raw_boxes:
            logging.info("Drawing pre-grouping boxes with orientation-based colors...")
            pre_group_image = draw_boxes_on_image_with_orientation(base_image, raw_boxes, thickness=2)
            output_path_pre = os.path.join(output_dir, f"{image_name}_pre_grouping.jpg")
            success = cv2.imwrite(output_path_pre, pre_group_image)
            if success:
                logging.info(f"Saved pre-grouping visualization to: {output_path_pre}")
            else:
                logging.error(f"Failed to save pre-grouping visualization to: {output_path_pre}")
        else:
            logging.warning("No raw boxes found to visualize for pre-grouping.")

        # --- Visualize Post-Grouping Boxes ---
        grouped_boxes = load_grouped_detections(image_name, grouped_metadata_dir)
        if grouped_boxes:
            logging.info("Drawing post-grouping boxes with orientation-based colors...")
            post_group_image = draw_boxes_on_image_with_orientation(base_image, grouped_boxes, thickness=3)
            output_path_post = os.path.join(output_dir, f"{image_name}_post_grouping.jpg")
            success = cv2.imwrite(output_path_post, post_group_image)
            if success:
                logging.info(f"Saved post-grouping visualization to: {output_path_post}")
            else:
                logging.error(f"Failed to save post-grouping visualization to: {output_path_post}")
        else:
            logging.warning("No grouped boxes found to visualize for post-grouping.")

    logging.info("\nVisualization script finished.")
    logging.info("Color legend: Green = Horizontal boxes, Red = Vertical boxes, Gray = Unknown orientation")


if __name__ == "__main__":
    main()