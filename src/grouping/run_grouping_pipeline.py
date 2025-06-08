import os
import json
import logging
from glob import glob

# Import the grouper class and the filtering functions
from bbox_grouper import BBoxGrouper
from filter_by_core import load_core_metadata, filter_tile_detections

# --- Configuration ---
# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Define paths based on your project structure
CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
DET_META_DIR = "data/processed/metadata/detection_metadata"
GROUPING_CONFIG_PATH = "configs/grouping.yaml"
OUTPUT_DIR = "data/processed/metadata/final_grouped_text"


def main():
    """
    Main function to run the full filtering and grouping pipeline.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize the BBoxGrouper with your configuration
    if not os.path.exists(GROUPING_CONFIG_PATH):
        logging.error(f"Grouping config file not found at: {GROUPING_CONFIG_PATH}")
        return
    grouper = BBoxGrouper(config_path=GROUPING_CONFIG_PATH)
    logging.info("BBoxGrouper initialized.")

    # 2. Load the map of all core tile metadata
    # This gives us a dictionary like: {'image_base_name': {tile_id: core_coords}}
    core_map = load_core_metadata(CORE_META_DIR)
    if not core_map:
        logging.warning("No core metadata found. Please run previous pipeline steps.")
        return

    logging.info(f"Found core metadata for {len(core_map)} source images.")

    # 3. Process each source image
    for base_name, tiles in core_map.items():
        logging.info(f"\n{'=' * 50}\nProcessing image: {base_name}\n{'=' * 50}")

        # --- FILTERING STAGE ---
        # Consolidate all filtered detections for this single image
        all_filtered_detections = []
        for tile_id, core_coords in tiles.items():
            # Use the improved area-based filtering function
            kept_detections = filter_tile_detections(base_name, tile_id, core_coords, DET_META_DIR, min_area_ratio=0.5)
            all_filtered_detections.extend(kept_detections)

        logging.info(f"Found {len(all_filtered_detections)} de-duplicated text detections after filtering.")

        if not all_filtered_detections:
            logging.warning(f"No detections left for {base_name} after filtering. Skipping.")
            continue

        # --- GROUPING STAGE ---
        # Pass the clean, consolidated list of detections to the grouper
        logging.info("Starting grouping process...")
        final_grouped_lines = grouper.process(all_filtered_detections)
        logging.info(f"Grouped detections into {len(final_grouped_lines)} final text lines.")

        # 4. Save the final output
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_grouped_text.json")
        try:
            with open(output_path, 'w') as f:
                # We can't directly JSON-serialize the component_detections if they are dataclasses
                # So we can either convert them to dicts or just save the main info.
                # Here's a safe way to save it:
                output_data = []
                for line in final_grouped_lines:
                    # Create a copy and remove the complex objects before saving
                    line_copy = line.copy()
                    del line_copy['component_detections']
                    output_data.append(line_copy)

                json.dump(output_data, f, indent=2)
            logging.info(f"Successfully saved final output to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to write output for {base_name}: {e}")


if __name__ == '__main__':
    main()