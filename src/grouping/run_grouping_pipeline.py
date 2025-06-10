import os
import json
import logging
import argparse
from glob import glob
import yaml

# Import the grouper class and the filtering functions
from .bbox_grouper import BBoxGrouper
from .filter_by_core import load_core_metadata, filter_tile_detections

# --- Configuration ---
CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
DET_META_DIR = "data/processed/metadata/detection_metadata"
GROUPING_CONFIG_PATH = "configs/grouping.yaml"
OUTPUT_DIR = "data/processed/metadata/final_grouped_text"


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
    Main function to run the grouping pipeline with only post-grouping aspect ratio filtering.
    """
    setup_logging(args.debug)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GROUPING_CONFIG_PATH):
        logging.error(f"Grouping config file not found at: {GROUPING_CONFIG_PATH}")
        return

    # REMOVED: All pre-grouping filter configurations
    # No more min_core_ratio or iou_thresh loading

    grouper = BBoxGrouper(config_path=GROUPING_CONFIG_PATH)
    logging.info("BBoxGrouper initialized.")

    core_map = load_core_metadata(CORE_META_DIR)
    if not core_map:
        logging.warning("No core metadata found. Please run previous pipeline steps.")
        return

    logging.info(f"Found core metadata for {len(core_map)} source images.")

    for base_name, tiles in core_map.items():
        logging.info(f"\n{'=' * 50}\nProcessing image: {base_name}\n{'=' * 50}")

        # --- Load ALL detections without any filtering ---
        all_detections_from_tiles = []
        for tile_id, core_coords in tiles.items():
            # REMOVED: Core area filtering - load all detections with min_area_ratio=0.0
            kept_detections = filter_tile_detections(
                base_name, tile_id, core_coords, DET_META_DIR,
                min_area_ratio=0.0  # Accept all detections regardless of core overlap
            )
            all_detections_from_tiles.extend(kept_detections)

        logging.info(f"Loaded {len(all_detections_from_tiles)} text detections (no pre-filtering applied).")

        # REMOVED: All pre-grouping filtering steps
        # - No confidence threshold filtering
        # - No core area filtering
        # - No pre-grouping NMS filtering

        if not all_detections_from_tiles:
            logging.warning(f"No detections found for {base_name}. Skipping.")
            continue

        logging.info("Starting grouping process...")
        final_grouped_lines = grouper.process(all_detections_from_tiles)
        logging.info(f"Grouped detections into {len(final_grouped_lines)} final text lines.")
        logging.info("Only post-grouping aspect ratio filtering was applied.")

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
    parser = argparse.ArgumentParser(description="Run the text grouping pipeline with only post-grouping aspect ratio filtering.")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for detailed logging.'
    )
    args = parser.parse_args()
    main(args)
