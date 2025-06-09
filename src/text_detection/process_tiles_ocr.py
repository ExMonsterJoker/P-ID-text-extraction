import os
import sys
import glob
import yaml
import logging
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.text_detection.ocr_detector import EasyOCRDetector
from src.data_loader.metadata_manager import MetadataManager
from src.data_loader.sahi_slicer import SahiSlicer

def get_tile_id_from_path(tile_path: str) -> str:
    """Extracts tile ID like 'DURI..._T0000' from a path like '.../tile_DURI..._T0000.png'."""
    filename = os.path.basename(tile_path)
    # Remove 'tile_' prefix and file extension
    return os.path.splitext(filename[5:])[0] if filename.startswith('tile_') else os.path.splitext(filename)[0]


def process_tiles_with_ocr(config: Dict[str, Any]):
    """
    Process all tiles with OCR individually using parameters from the config file.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Get paths and parameters from the main config dictionary
    data_loader_config = config.get('data_loader', {})
    ocr_config = config.get('ocr', {})

    tiles_base_dir = data_loader_config.get('sahi_slicer_output_dir', "data/processed/tiles")
    metadata_output_dir = data_loader_config.get('metadata_output_dir', "data/processed/metadata")
    filter_overlaps = ocr_config.get('filter_overlaps', True)

    # Make paths absolute
    if not os.path.isabs(tiles_base_dir):
        tiles_base_dir = os.path.join(project_root, tiles_base_dir)
    if not os.path.isabs(metadata_output_dir):
        metadata_output_dir = os.path.join(project_root, metadata_output_dir)

    logging.info(f"Looking for tiles in: {tiles_base_dir}")
    if not os.path.exists(tiles_base_dir):
        logging.error(f"Tiles directory does not exist: {tiles_base_dir}")
        return

    logging.info("Initializing EasyOCR detector with config...")
    # Pass the ocr_config subsection to the detector
    ocr_detector = EasyOCRDetector(ocr_config)
    metadata_manager = MetadataManager(metadata_output_dir, pipeline_version="1.0")

    image_dirs = [d for d in glob.glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning(f"No image directories found in {tiles_base_dir}")
        return

    for image_dir in image_dirs:
        image_name = os.path.basename(image_dir)
        logging.info(f"\n{'=' * 60}\nProcessing OCR for image: {image_name}")

        yaml_metadata_path = os.path.join(image_dir, "tiles_metadata.yaml")
        if not os.path.exists(yaml_metadata_path):
            logging.error(f"Metadata file not found: {yaml_metadata_path}")
            continue

        try:
            tile_metadata_list = SahiSlicer.load_metadata(yaml_metadata_path)
            tile_metadata_map = {tm.tile_id: tm for tm in tile_metadata_list}

            tile_files = glob.glob(os.path.join(image_dir, "tile_*.png")) + glob.glob(
                os.path.join(image_dir, "tile_*.jpg"))
            logging.info(f"Found {len(tile_files)} tiles to process for {image_name}")

            for i, tile_path in enumerate(tile_files):
                logging.info(f"Processing tile {i + 1}/{len(tile_files)}: {os.path.basename(tile_path)}")
                try:
                    ocr_detections = ocr_detector.detect_text_with_rotation(tile_path)

                    if filter_overlaps and len(ocr_detections) > 1:
                        ocr_detections = ocr_detector.filter_overlapping_detections(ocr_detections)

                    tile_id = get_tile_id_from_path(tile_path)
                    if tile_id in tile_metadata_map:
                        tile_metadata = tile_metadata_map[tile_id]
                        metadata_manager.save_ocr_metadata(ocr_detections, tile_path, tile_metadata)
                    else:
                        logging.warning(f"No metadata found for tile ID: {tile_id}")

                except Exception as e:
                    logging.error(f"Failed to process tile {tile_path}: {e}", exc_info=True)

            logging.info(f"Completed processing for {image_name}.")
        except Exception as e:
            logging.error(f"Failed to process image directory {image_name}: {e}", exc_info=True)

    logging.info(f"\n{'=' * 60}\nOCR processing completed!")

def load_full_config(config_dir="configs"):
    """Loads all YAML config files from a directory into one dictionary."""
    config = {}
    for config_file in glob.glob(os.path.join(config_dir, "*.yaml")):
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))
    return config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    # Load the entire configuration from the 'configs' directory
    full_config = load_full_config()
    if not full_config:
        logging.error("No config files found. Aborting.")
    else:
        # Pass the full config to the processing function
        process_tiles_with_ocr(full_config)
