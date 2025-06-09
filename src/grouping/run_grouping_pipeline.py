import os
import json
import logging
import argparse
from glob import glob
from shapely.geometry import Polygon
import yaml

# Import the grouper class and the filtering functions
from .bbox_grouper import BBoxGrouper
from .filter_by_core import load_core_metadata, filter_tile_detections

# --- Configuration ---
CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
DET_META_DIR = "data/processed/metadata/detection_metadata"
GROUPING_CONFIG_PATH = "configs/grouping.yaml"
OUTPUT_DIR = "data/processed/metadata/final_grouped_text"


def pre_group_filter(detections, iou_threshold=0.5):
    """
    Filters overlapping detections before grouping (Non-Maximum Suppression).
    This version prioritizes longer text strings to avoid being suppressed by
    high-confidence partial detections.
    """
    if not detections:
        return []

    # NEW: Sort by text length (descending), then confidence (descending).
    # This ensures longer detections are considered first.
    detections.sort(key=lambda x: (len(x.get('text', '')), x.get('confidence', 0)), reverse=True)

    final_detections = []
    for current_det in detections:
        is_duplicate = False
        # Ensure bbox_original and its area are valid
        if not current_det.get('bbox_original'): continue
        current_poly = Polygon(current_det['bbox_original'])
        if current_poly.area == 0: continue

        for final_det in final_detections:
            final_poly = Polygon(final_det['bbox_original'])

            # Use intersection over the area of the SMALLER box.
            # This is a better metric for one box containing another.
            intersection = current_poly.intersection(final_poly).area
            min_area = min(current_poly.area, final_poly.area)

            if min_area == 0: continue

            # If the overlap ratio is high, it's a duplicate.
            overlap_metric = intersection / min_area
            if overlap_metric > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            final_detections.append(current_det)

    logging.info(f"Pre-grouping NMS filter reduced detections from {len(detections)} to {len(final_detections)}")
    return final_detections


def setup_logging(debug_mode=False):
    """Configures logging based on whether debug mode is active."""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    handlers = [logging.StreamHandler()]  # Always log to console

    if debug_mode:
        log_file_path = 'grouping_debug.log'
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        # Add file handler only in debug mode
        handlers.append(logging.FileHandler(log_file_path, mode='w'))

    # Get the root logger
    root_logger = logging.getLogger()
    # Reset any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        handlers=handlers
    )
    logging.info(f"Logging configured. Debug mode: {'ON' if debug_mode else 'OFF'}")


def main(args):
    """
    Main function to run the full filtering and grouping pipeline.
    """
    setup_logging(args.debug)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GROUPING_CONFIG_PATH):
        logging.error(f"Grouping config file not found at: {GROUPING_CONFIG_PATH}")
        return

    # Load filter configurations
    with open(GROUPING_CONFIG_PATH, 'r') as f:
        filter_config = yaml.safe_load(f).get('filtering', {})

    min_core_ratio = filter_config.get('min_core_area_ratio', 0.1)
    iou_thresh = filter_config.get('pre_group_iou_threshold', 0.4)

    grouper = BBoxGrouper(config_path=GROUPING_CONFIG_PATH)
    logging.info("BBoxGrouper initialized.")

    core_map = load_core_metadata(CORE_META_DIR)
    if not core_map:
        logging.warning("No core metadata found. Please run previous pipeline steps.")
        return

    logging.info(f"Found core metadata for {len(core_map)} source images.")

    for base_name, tiles in core_map.items():
        logging.info(f"\n{'=' * 50}\nProcessing image: {base_name}\n{'=' * 50}")

        # --- Core Area Filtering ---
        # If in debug mode, set ratio to 0 to disable filtering.
        current_min_core_ratio = 0.0 if args.debug else min_core_ratio
        if args.debug:
            logging.info("DEBUG MODE: Core area filter is disabled (min_area_ratio=0.0).")

        all_detections_from_tiles = []
        for tile_id, core_coords in tiles.items():
            kept_detections = filter_tile_detections(
                base_name, tile_id, core_coords, DET_META_DIR,
                min_area_ratio=current_min_core_ratio
            )
            all_detections_from_tiles.extend(kept_detections)

        logging.info(f"Found {len(all_detections_from_tiles)} text detections after core area filter.")

        # --- Pre-Grouping NMS Filter ---
        if args.debug:
            logging.info("DEBUG MODE: Pre-grouping NMS filter has been REMOVED.")
            final_clean_detections = all_detections_from_tiles
        else:
            logging.info(f"Applying pre-grouping NMS filter with IoU threshold: {iou_thresh}")
            final_clean_detections = pre_group_filter(all_detections_from_tiles, iou_threshold=iou_thresh)

        if not final_clean_detections:
            logging.warning(f"No raw detections left to process for {base_name}. Skipping.")
            continue

        logging.info("Starting grouping process...")
        final_grouped_lines = grouper.process(final_clean_detections)
        logging.info(f"Grouped detections into {len(final_grouped_lines)} final text lines.")

        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_grouped_text.json")
        try:
            with open(output_path, 'w') as f:
                output_data = [
                    {k: v for k, v in line.items() if k != 'component_detections'}
                    for line in final_grouped_lines
                ]
                json.dump(output_data, f, indent=2)
            logging.info(f"Successfully saved final output to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to write output for {base_name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the text grouping pipeline with an optional debug mode.")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode. This provides detailed logs and disables pre-grouping filters.'
    )
    args = parser.parse_args()
    main(args)
