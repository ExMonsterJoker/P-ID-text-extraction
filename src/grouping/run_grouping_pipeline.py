import os
import json
import logging
from glob import glob
from shapely.geometry import Polygon
import yaml

# Import the grouper class and the filtering functions
from .bbox_grouper import BBoxGrouper
from .filter_by_core import load_core_metadata, filter_tile_detections

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
DET_META_DIR = "data/processed/metadata/detection_metadata"
GROUPING_CONFIG_PATH = "configs/grouping.yaml"
OUTPUT_DIR = "data/processed/metadata/final_grouped_text"


def load_filter_config(config_path):
    """Loads only the 'filtering' section of the config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f).get('filtering', {})
    return {
        'min_core_area_ratio': config.get('min_core_area_ratio', 0.5),
        'pre_group_iou_threshold': config.get('pre_group_iou_threshold', 0.5)
    }


def pre_group_filter(detections, iou_threshold=0.5):
    """
    Filters overlapping detections before grouping (Non-Maximum Suppression).
    """
    if not detections:
        return []
    detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    final_detections = []
    for current_det in detections:
        is_duplicate = False
        current_poly = Polygon(current_det['bbox_original'])
        for final_det in final_detections:
            final_poly = Polygon(final_det['bbox_original'])
            intersection = current_poly.intersection(final_poly).area
            union = current_poly.union(final_poly).area
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append(current_det)
    logging.info(f"Pre-grouping filter reduced detections from {len(detections)} to {len(final_detections)}")
    return final_detections


def main():
    """
    Main function to run the full filtering and grouping pipeline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GROUPING_CONFIG_PATH):
        logging.error(f"Grouping config file not found at: {GROUPING_CONFIG_PATH}")
        return

    grouper = BBoxGrouper(config_path=GROUPING_CONFIG_PATH)
    logging.info("BBoxGrouper initialized.")

    core_map = load_core_metadata(CORE_META_DIR)
    if not core_map:
        logging.warning("No core metadata found. Please run previous pipeline steps.")
        return

    logging.info(f"Found core metadata for {len(core_map)} source images.")

    for base_name, tiles in core_map.items():
        logging.info(f"\n{'=' * 50}\nProcessing image: {base_name}\n{'=' * 50}")

        # --- DEBUGGING STEP 1: Effectively disable core filtering ---
        # We pass min_area_ratio=0.0 to keep ALL detections from all tiles,
        # even those on the very edge. This helps us see if this filter was too aggressive.
        logging.info("DEBUG MODE: Core area filter is disabled (min_area_ratio=0.0).")
        all_detections_from_tiles = []
        for tile_id, core_coords in tiles.items():
            kept_detections = filter_tile_detections(
                base_name, tile_id, core_coords, DET_META_DIR,
                min_area_ratio=0.0  # Setting to 0 keeps everything
            )
            all_detections_from_tiles.extend(kept_detections)

        logging.info(f"Found {len(all_detections_from_tiles)} text detections, keeping all for debugging.")

        # --- DEBUGGING STEP 2: Disable the pre-grouping filter ---
        # We bypass this step to see the raw output of the grouper
        # without removing any nested/overlapping detections first.
        logging.info("DEBUG MODE: Pre-grouping NMS filter is disabled.")
        # final_clean_detections = pre_group_filter(all_detections_from_tiles) # The filter is bypassed
        final_clean_detections = all_detections_from_tiles  # Use the unfiltered list directly

        if not final_clean_detections:
            logging.warning(f"No raw detections found for {base_name}. Skipping.")
            continue

        logging.info("Starting grouping process on raw, unfiltered data...")
        final_grouped_lines = grouper.process(final_clean_detections)
        logging.info(f"Grouped raw detections into {len(final_grouped_lines)} final text lines.")

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
    main()
