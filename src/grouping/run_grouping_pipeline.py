# src/grouping/run_grouping_pipeline.py
import os
import json
import logging
import argparse
from glob import glob
import yaml

# Import the grouper class and the filtering functions
from .bbox_grouper import BBoxGrouper
from .post_processing_filters import (
    calculate_box_area,
    apply_soft_nms_filter,
    apply_smart_aspect_ratio_filter
)

# --- Configuration ---
TILES_BASE_DIR = "data/processed/tiles"
DET_META_DIR = "data/processed/metadata/detection_metadata"
GROUPING_CONFIG_PATH = "configs/grouping.yaml"
OUTPUT_DIR = "data/processed/metadata/final_grouped_text"


def setup_logging(debug=False):
    """Configures logging for the pipeline."""
    level = logging.DEBUG if debug else logging.INFO
    log_format = "[%(asctime)s] [%(levelname)s] - %(message)s"
    # Clear existing handlers to avoid duplicate logs in interactive environments
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(level=level, format=log_format, datefmt="%H:%M:%S")
    logging.info(f"Logging configured. Debug mode: {'ON' if debug else 'OFF'}")


def load_all_detections_for_image(image_tile_dir: str, detection_base_dir: str) -> list:
    """
    Loads all OCR detections for a given image by iterating through all its tiles
    using the tiles_metadata.json file as a guide.
    """
    base_name = os.path.basename(image_tile_dir)
    # MODIFIED: Look for .json instead of .yaml
    metadata_path = os.path.join(image_tile_dir, "tiles_metadata.json")

    if not os.path.exists(metadata_path):
        logging.warning(f"Could not find 'tiles_metadata.json' in {image_tile_dir}. Skipping image.")
        return []

    # MODIFIED: Load with json.load
    with open(metadata_path, 'r') as f:
        tile_metadata = json.load(f)

    # ... (The rest of the function remains the same) ...
    all_detections = []
    for tile_info in tile_metadata.get('tiles', []):
        tile_id = tile_info['tile_id']
        tile_coords = tile_info['coordinates']

        detection_file_path = os.path.join(detection_base_dir, base_name, f"{tile_id}_ocr.json")
        if not os.path.exists(detection_file_path):
            continue

        try:
            with open(detection_file_path, 'r') as f:
                detections_in_tile = json.load(f)

            for det in detections_in_tile:
                local_bbox = det.get('bbox', [])
                if len(local_bbox) < 3:
                    continue

                global_bbox = [[p[0] + tile_coords[0], p[1] + tile_coords[1]] for p in local_bbox]
                det['bbox_original'] = global_bbox
                all_detections.append(det)
        except Exception as e:
            logging.error(f"Failed to process detection file {detection_file_path}: {e}")

    return all_detections


def main(args):
    """
    Main function to run the complete text grouping and filtering pipeline.
    """
    setup_logging(args.debug)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(GROUPING_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    pre_filter_config = config.get('pre_group_filtering', {})
    soft_nms_config = config.get('soft_nms', {})
    post_filter_config = config.get('post_group_filtering', {})
    smart_ar_config = post_filter_config.get('smart_aspect_ratio', {})

    grouper = BBoxGrouper(config_path=GROUPING_CONFIG_PATH, debug=args.debug)

    image_tile_dirs = [d for d in glob(os.path.join(TILES_BASE_DIR, '*')) if os.path.isdir(d)]

    if not image_tile_dirs:
        logging.warning(f"No processed tile directories found in '{TILES_BASE_DIR}'. Please run the slicing step.")
        return

    for image_dir in image_tile_dirs:
        base_name = os.path.basename(image_dir)
        logging.info(f"\n{'=' * 50}\nProcessing image: {base_name}\n{'=' * 50}")

        # Step 0: Load ALL detections from ALL tiles
        all_detections_from_tiles = load_all_detections_for_image(image_dir, DET_META_DIR)
        logging.info(f"Loaded {len(all_detections_from_tiles)} total detections for '{base_name}'.")

        if not all_detections_from_tiles:
            logging.warning(f"No detections loaded for {base_name}. Skipping.")
            continue

        # Step 1: Basic Quality Filtering
        min_conf = pre_filter_config.get('min_confidence', 0.4)
        min_area = pre_filter_config.get('min_area', 10)
        logging.info(f"Applying basic quality filters (min_conf > {min_conf}, min_area > {min_area})...")

        quality_filtered_dets = []
        if args.debug:
            logging.debug("--- Debug Mode: Basic Quality Filtering Details ---")

        for d in all_detections_from_tiles:
            conf = d.get('confidence', 0.0)
            area = calculate_box_area(d.get('bbox_original', []))
            passes_conf = conf >= min_conf
            passes_area = area >= min_area

            if passes_conf and passes_area:
                quality_filtered_dets.append(d)
            elif args.debug:
                reasons = []
                if not passes_conf: reasons.append(f"confidence {conf:.2f} < {min_conf}")
                if not passes_area: reasons.append(f"area {area:.0f} < {min_area}")
                logging.debug(f"DROPPED '{d.get('text', 'N/A')}': Reason(s): {', '.join(reasons)}")

        logging.info(f"After quality filtering: {len(quality_filtered_dets)} detections remain.")

        # Step 2: Grouping
        grouped_lines = grouper.process(quality_filtered_dets)

        # Step 3: Soft-NMS (Post-Grouping)
        if soft_nms_config.get('enabled', False):
            nms_filtered_lines = apply_soft_nms_filter(
                grouped_lines,
                soft_nms_config.get('iou_threshold', 0.5),
                soft_nms_config.get('sigma', 0.5),
                soft_nms_config.get('min_confidence', 0.3)
            )
        else:
            nms_filtered_lines = grouped_lines

        # Step 4: Final Validation Filtering
        if smart_ar_config.get('enabled', False):
            final_lines = apply_smart_aspect_ratio_filter(
                nms_filtered_lines,
                smart_ar_config.get('base_ratio', 0.5),
                smart_ar_config.get('length_factor', 0.02)
            )
        else:
            final_lines = nms_filtered_lines

        logging.info(f"Pipeline finished for {base_name}. Produced {len(final_lines)} final text lines.")

        # --- Save the final result ---
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_grouped_text.json")
        with open(output_path, 'w') as f:
            json.dump(final_lines, f, indent=4)
        logging.info(f"Saved final grouped text to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the text grouping and filtering pipeline.")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging.')
    args = parser.parse_args()
    main(args)