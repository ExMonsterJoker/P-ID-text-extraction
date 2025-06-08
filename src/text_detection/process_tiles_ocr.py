import os
import sys
import glob
import yaml
import logging
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.text_detection.ocr_detector import EasyOCRDetector
from src.data_loader.metadata_manager import MetadataManager
from src.data_loader.sahi_slicer import SahiSlicer


def get_tile_id_from_path(tile_path: str) -> str:
    """Extracts tile ID like 'DURI..._T0000' from a path like '.../tile_DURI..._T0000.png'."""
    filename = os.path.basename(tile_path)
    if filename.startswith('tile_'):
        filename = filename[5:]
    return os.path.splitext(filename)[0]


def process_tiles_with_ocr(
        tiles_base_dir: str = "data/processed/tiles",
        metadata_output_dir: str = "data/processed/metadata",
        languages: List[str] = ['en'],
        confidence_threshold: float = 0.3,
        gpu: bool = True,
        use_rotation: bool = True,
        filter_overlaps: bool = True,
        rotation_angles: List[int] = [0, 90]
):
    """
    Process all tiles with OCR individually and save results as metadata.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isabs(tiles_base_dir):
        tiles_base_dir = os.path.join(project_root, tiles_base_dir)
    if not os.path.isabs(metadata_output_dir):
        metadata_output_dir = os.path.join(project_root, metadata_output_dir)

    logging.info(f"Looking for tiles in: {tiles_base_dir}")

    if not os.path.exists(tiles_base_dir):
        logging.error(f"Tiles directory does not exist: {tiles_base_dir}")
        return

    logging.info("Initializing EasyOCR detector...")
    ocr_detector = EasyOCRDetector(languages=languages, gpu=gpu)
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
            with open(yaml_metadata_path, 'r') as f:
                source_image = yaml.safe_load(f)["source_image"]

            tile_metadata_list = SahiSlicer.load_metadata(yaml_metadata_path)
            tile_metadata_map = {tm.tile_id: tm for tm in tile_metadata_list}

            tile_files = glob.glob(os.path.join(image_dir, "tile_*.png")) + glob.glob(
                os.path.join(image_dir, "tile_*.jpg"))
            logging.info(f"Found {len(tile_files)} tiles to process for {image_name}")

            for i, tile_path in enumerate(tile_files):
                logging.info(f"Processing tile {i + 1}/{len(tile_files)}: {os.path.basename(tile_path)}")

                try:
                    # Perform OCR on the single tile
                    ocr_detections = ocr_detector.detect_text_with_rotation(
                        tile_path,
                        confidence_threshold=confidence_threshold,
                        rotation_angles=rotation_angles
                    )

                    # Filter overlapping detections from different rotations on the same tile
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
            continue

    logging.info(f"\n{'=' * 60}\nOCR processing completed!")


if __name__ == "__main__":
    config = {
        "tiles_base_dir": "data/processed/tiles",
        "metadata_output_dir": "data/processed/metadata",
        "languages": ['en'],
        "confidence_threshold": 0.3,
        "gpu": True,
        "use_rotation": True,
        "filter_overlaps": True,
        "rotation_angles": [0, 90],
    }
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    process_tiles_with_ocr(**config)
