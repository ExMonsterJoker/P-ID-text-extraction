# run_pipeline.py
import os
import sys
import yaml
import logging
import argparse
import datetime
import easyocr
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from glob import glob


# --- Add project root to system path for module imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import Core Logic from Project Modules ---
from src.convert_coord import main as run_coordinate_conversion, validate_json_with_pdf, test_dimension_validation
from src.data_loader.sahi_slicer import SahiSlicer
from src.data_loader.metadata_manager import MetadataManager
# Updated import for the new text detection logic
from src.text_detection.text_detection import text_detection
from src.text_detection.text_recognition import run_text_recognition_step
from src.grouping.grouping_logic import BoundingBoxGrouper
from src.cropping.cropping_Images import crop_image
from src.visualization.visualizer import visualize_annotations
import json

# From root (if you have PDF conversion)
try:
    from PDF_to_image import pdf_to_image_high_quality
    PDF_CONVERTER_AVAILABLE = True
except ImportError:
    PDF_CONVERTER_AVAILABLE = False


def setup_logging(log_level="INFO"):
    """Sets up a centralized logger for the pipeline."""
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info("Logging configured.")

def load_config(config_dir: str) -> Dict:
    """
    Loads YAML configuration files from a directory, skipping empty or invalid files.
    """
    config = {}
    config_files = glob(os.path.join(config_dir, '*.yaml'))

    if not config_files:
        logging.warning(f"No YAML config files found in directory: {config_dir}")
        return config

    logging.info(f"Loading config files from: {config_dir}")
    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                single_config = yaml.safe_load(f)
                if single_config:
                    config.update(single_config)
                    logging.info(f"Successfully loaded and merged config from: {config_path}")
                else:
                    logging.warning(f"Config file is empty or invalid, skipping: {config_path}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")

    return config


def run_pdf_conversion_step(input_dir: str, config: dict):
    """Runs the PDF to image conversion step."""
    if not PDF_CONVERTER_AVAILABLE:
        logging.warning("PDF_to_image.py not found. Skipping PDF conversion.")
        return

    logging.info("--- Starting Step 0: PDF to Image Conversion ---")
    pdf_input_dir = os.path.join(input_dir, "pdfs")
    if not os.path.exists(pdf_input_dir):
        logging.info("No 'pdfs' subdirectory found. Skipping PDF conversion.")
        return

    pdf_config = config.get('pdf_conversion', {})
    dpi = pdf_config.get('dpi', 600)
    image_format = pdf_config.get('format', 'PNG')

    pdf_to_image_high_quality(pdf_input_dir, input_dir, dpi=dpi, image_format=image_format)
    logging.info("--- Finished PDF to Image Conversion ---")


def run_slicing_step(input_dir: str, config: dict):
    """Runs the image slicing step for all images in the input directory."""
    logging.info("--- Starting Step 1: Image Slicing ---")
    slicer_config = config.get("sahi_slicer", {})
    slicer = SahiSlicer(slicer_config)

    output_dir = config.get('data_loader', {}).get('sahi_slicer_output_dir', 'data/processed/tiles')
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob(os.path.join(input_dir, "*.png")) + \
                  glob(os.path.join(input_dir, "*.jpg")) + \
                  glob(os.path.join(input_dir, "*.tiff"))

    if not image_files:
        logging.warning(f"No image files found in '{input_dir}'.")
        return

    for image_path in image_files:
        try:
            image_filename = Path(image_path).stem
            logging.info(f"Slicing image: {image_filename}")

            tiles, metadata = slicer.slice(image_path)
            if not tiles:
                logging.warning(f"No tiles were generated for {image_filename}.")
                continue

            image_output_dir = os.path.join(output_dir, image_filename)
            os.makedirs(image_output_dir, exist_ok=True)

            metadata_path = os.path.join(image_output_dir, "tiles_metadata.json")
            slicer.save_metadata(metadata, metadata_path)
            logging.info(f"Saved metadata for {len(metadata)} tiles to {metadata_path}")

            for i, tile_img_np in enumerate(tiles):
                tile_img = Image.fromarray(tile_img_np)
                tile_path = os.path.join(image_output_dir, f"tile_{metadata[i].tile_id}.png")
                tile_img.save(tile_path)
            logging.info(f"Saved {len(tiles)} tiles to {image_output_dir}")

        except Exception as e:
            logging.error(f"Failed to slice image {image_path}: {e}", exc_info=True)


def run_metadata_step(config: dict):
    """Generates metadata for each set of tiles (simplified - no core tiles computation)."""
    logging.info("--- Starting Step 2: Metadata Generation ---")

    data_loader_config = config.get('data_loader', {})
    metadata_base_dir = data_loader_config.get('metadata_output_dir', "data/processed/metadata")
    tiles_base_dir = data_loader_config.get('sahi_slicer_output_dir', "data/processed/tiles")
    manager = MetadataManager(metadata_base_dir)

    image_dirs = [d for d in glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning("No sliced image directories found to generate metadata from.")
        return

    for image_dir in image_dirs:
        try:
            # Look for tiles_metadata.json
            json_path = os.path.join(image_dir, "tiles_metadata.json")
            if not os.path.exists(json_path):
                logging.warning(f"No 'tiles_metadata.json' in {image_dir}. Skipping.")
                continue

            with open(json_path, 'r') as f:
                metadata_file = json.load(f)
                source_image = metadata_file["source_image"]

            logging.info(f"Processing metadata for: {Path(source_image).name}")
            # Simply load and save the tile metadata (no core tiles computation)
            tile_metadata = SahiSlicer.load_metadata(json_path)
            manager.save_tile_metadata(tile_metadata, source_image)
            logging.info(f"Saved tile metadata for {len(tile_metadata)} tiles.")

        except Exception as e:
            logging.error(f"Failed to process metadata for {image_dir}: {e}", exc_info=True)

    logging.info("--- Finished Metadata Generation ---")


def run_text_detection_step(config: dict):
    """Runs the new text detection step using CRAFT detector on all sliced tiles."""
    logging.info("--- Starting Step 3: Text Detection (CRAFT/DBNet + EasyOCR) ---")

    try:
        # Get configuration
        ocr_config = config.get('ocr', {})
        data_loader_config = config.get('data_loader', {})

        # Get directories
        tiles_base_dir = data_loader_config.get('sahi_slicer_output_dir', 'data/processed/tiles')
        detection_metadata_dir = data_loader_config.get('detection_metadata_dir', 'data/processed/metadata/detection_metadata')

        # Create output directory
        os.makedirs(detection_metadata_dir, exist_ok=True)

        # Initialize text detector
        detector = text_detection(ocr_config)
        logging.info("Text detector initialized successfully")

        # Get all image directories that contain tiles
        image_dirs = [d for d in glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]

        if not image_dirs:
            logging.warning("No sliced image directories found for text detection.")
            return

        total_tiles_processed = 0
        total_detections_found = 0

        for image_dir in image_dirs:
            try:
                image_name = os.path.basename(image_dir)
                logging.info(f"Processing text detection for image: {image_name}")

                # Create output directory for this image's detections
                image_detection_dir = os.path.join(detection_metadata_dir, image_name)
                os.makedirs(image_detection_dir, exist_ok=True)

                # Load tile metadata to get tiles info
                tiles_metadata_path = os.path.join(image_dir, "tiles_metadata.json")
                if not os.path.exists(tiles_metadata_path):
                    logging.warning(f"No tiles metadata found for {image_name}. Skipping.")
                    continue

                with open(tiles_metadata_path, 'r') as f:
                    tiles_metadata = json.load(f)

                source_image = tiles_metadata.get("source_image", "")
                original_image_size = tiles_metadata.get("original_size", [])  # Fixed: use "original_size" instead of "original_image_size"

                # Get all tile image files
                tile_files = glob(os.path.join(image_dir, "tile_*.png"))
                logging.info(f"Found {len(tile_files)} tile files for {image_name}")

                image_detections_count = 0

                for tile_file in tile_files:
                    try:
                        tile_filename = os.path.basename(tile_file)
                        tile_id = tile_filename.replace("tile_", "").replace(".png", "")

                        # Run text detection on this tile
                        detections = detector.detect_text(tile_file)

                        if not detections:
                            continue

                        # Find corresponding tile metadata
                        tile_metadata = None
                        for tile_info in tiles_metadata.get('tiles', []):
                            if tile_info.get('tile_id') == tile_id:
                                tile_metadata = tile_info
                                break

                        if not tile_metadata:
                            logging.warning(f"No metadata found for tile {tile_id}")
                            continue

                        # Convert detections to JSON format with metadata
                        tile_detections = []
                        for detection in detections:
                            detection_data = {
                                'bbox': detection.bbox,
                                'bbox_normalized': detection.bbox_normalized,
                                'rotation_angle': detection.rotation_angle,
                                'tile_id': tile_id,
                                'tile_path': tile_file,
                                'tile_coordinates': tile_metadata.get('coordinates', []),
                                'grid_position': tile_metadata.get('grid_position', []),
                                'source_image': source_image,
                                'original_image_size': original_image_size,  # Now correctly populated from "original_size"
                                'detection_type': 'craft_detection',
                                'tile_size': tile_metadata.get('tile_size', [])
                            }
                            tile_detections.append(detection_data)

                        # Save detections for this tile
                        if tile_detections:
                            output_file = os.path.join(image_detection_dir, f"{tile_id}_ocr.json")
                            with open(output_file, 'w') as f:
                                json.dump(tile_detections, f, indent=2)

                            image_detections_count += len(tile_detections)
                            total_tiles_processed += 1

                    except Exception as e:
                        logging.error(f"Error processing tile {tile_file}: {e}")
                        continue

                total_detections_found += image_detections_count
                logging.info(f"Completed {image_name}: {image_detections_count} detections from {len(tile_files)} tiles")

            except Exception as e:
                logging.error(f"Error processing image directory {image_dir}: {e}", exc_info=True)
                continue

        logging.info(f"Text detection completed:")
        logging.info(f"  Total tiles processed: {total_tiles_processed}")
        logging.info(f"  Total detections found: {total_detections_found}")
        logging.info(f"  Average detections per tile: {total_detections_found/total_tiles_processed:.2f}" if total_tiles_processed > 0 else "  No tiles processed")

    except Exception as e:
        logging.error(f"Error in text detection step: {e}", exc_info=True)
        raise

    logging.info("--- Finished Text Detection ---")


def run_grouping_and_filtering_step(config: dict, grouping_args: argparse.Namespace):
    """Runs the entire text grouping and filtering pipeline using the new BoundingBoxGrouper."""
    logging.info("--- Starting Step 4: Text Grouping and Filtering ---")

    try:
        # Initialize the new BoundingBoxGrouper
        grouper = BoundingBoxGrouper()

        # Run the processing for all images
        grouper.process_all_images()

        logging.info("--- Finished Text Grouping and Filtering ---")

    except Exception as e:
        logging.error(f"Error in grouping step: {e}", exc_info=True)
        raise

def run_cropping_step(config: Dict):
    """
    Runs the cropping process and creates a manifest for each set of crops.
    Updated to work with the new grouping logic output and use config parameters.
    """
    logging.info("--- Starting Step 5: Cropping Images ---")

    try:
        cropping_config = config.get('cropping', {})
        data_loader_config = config.get('data_loader', {})
        pipeline_config = config.get('pipeline', {})

        # Get paths from configuration - using group_detection_metadata_dir instead of final_grouped_text_dir
        image_source_dir = pipeline_config.get('input_dir', 'data/raw')
        grouped_text_dir = data_loader_config.get('group_detection_metadata_dir', 'data/processed/metadata/group_detection_metadata')
        output_dir = cropping_config.get('output_dir', 'data/processed/cropping')
        padding = cropping_config.get('padding', 10)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Check if grouped text directory exists
        if not os.path.exists(grouped_text_dir):
            logging.error(f"Grouped text directory not found: {grouped_text_dir}")
            logging.error("Make sure the grouping step has been completed successfully.")
            return

        # First, let's see what files are actually in the directory
        all_files = os.listdir(grouped_text_dir)
        logging.info(f"Files found in {grouped_text_dir}: {all_files}")

        # Try multiple patterns to find the grouped text JSON files
        json_patterns = [
            '*_grouped_text.json',
            '*_grouped.json',
            '*.json',
            '*grouped*.json'
        ]

        json_files = []
        for pattern in json_patterns:
            potential_files = glob(os.path.join(grouped_text_dir, pattern))
            if potential_files:
                json_files = potential_files
                logging.info(f"Found JSON files using pattern '{pattern}': {[os.path.basename(f) for f in json_files]}")
                break

        if not json_files:
            logging.warning(f"No JSON files found in: {grouped_text_dir}")
            logging.warning("Available files in directory:")
            for file in all_files:
                logging.warning(f"  - {file}")
            return

        logging.info(f"Found {len(json_files)} JSON files to process for cropping.")

        total_crops_created = 0
        total_images_processed = 0

        for json_file in json_files:
            try:
                # Extract base name from different possible patterns
                base_name = os.path.basename(json_file)
                for suffix in ['_grouped_text.json', '_grouped.json', '.json']:
                    if base_name.endswith(suffix):
                        base_name = base_name.replace(suffix, '')
                        break

                logging.info(f"Processing cropping for: {base_name}")

                # Find the corresponding source image
                possible_extensions = ['png', 'jpg', 'jpeg', 'tiff', 'tif']
                image_path = None
                for ext in possible_extensions:
                    test_path = os.path.join(image_source_dir, f"{base_name}.{ext}")
                    if os.path.exists(test_path):
                        image_path = test_path
                        break

                if not image_path:
                    logging.warning(f"  - Could not find source image for '{base_name}'. Skipping.")
                    continue

                # Load the grouped detections
                with open(json_file, 'r') as f:
                    detections = json.load(f)

                if not detections:
                    logging.info(f"  - No detections found in {base_name}. Skipping.")
                    continue

                # Create output directory for this image's crops
                image_output_dir = os.path.join(output_dir, base_name)
                os.makedirs(image_output_dir, exist_ok=True)

                # Process all detections without confidence filtering
                logging.info(f"  - Processing {len(detections)} detections (no confidence filtering)")

                # Run the cropping
                manifest_entries = crop_image(
                    image_path=image_path,
                    detections=detections,
                    output_dir=image_output_dir,
                    padding=padding
                )

                if manifest_entries:
                    # Save the manifest
                    manifest_path = os.path.join(image_output_dir, "manifest.json")
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest_entries, f, indent=2)

                    total_crops_created += len(manifest_entries)
                    total_images_processed += 1

                    logging.info(f"  - Created {len(manifest_entries)} crops and saved manifest to {manifest_path}")
                else:
                    logging.warning(f"  - No crops were created for {base_name}")

            except Exception as e:
                logging.error(f"Error processing {json_file}: {e}", exc_info=True)
                continue

        logging.info(f"Cropping completed:")
        logging.info(f"  Total images processed: {total_images_processed}")
        logging.info(f"  Total crops created: {total_crops_created}")
        logging.info(f"  Average crops per image: {total_crops_created/total_images_processed:.2f}" if total_images_processed > 0 else "  No images processed")

    except Exception as e:
        logging.error(f"Error in cropping step: {e}", exc_info=True)
        raise

    logging.info("--- Finished Step 5: Cropping Images ---")


def run_re_ocr_step(config: Dict):
    """
    Runs re-OCR on cropped images, rotating vertical crops for better accuracy,
    and then creates final annotation JSON files.
    """
    logging.info("--- Starting Step: Re-OCR on Cropped Images ---")
    re_ocr_config = config.get('re_ocr', {})
    input_dir = re_ocr_config.get('input_dir')
    output_dir = re_ocr_config.get('output_dir')
    languages = re_ocr_config.get('languages', ['en'])
    gpu = re_ocr_config.get('gpu', True)
    min_conf = re_ocr_config.get('min_confidence', 0.1)

    if not all([input_dir, output_dir]):
        logging.error("Re-OCR config missing 'input_dir' or 'output_dir'. Skipping step.")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        reader = easyocr.Reader(languages, gpu=gpu)
    except Exception as e:
        logging.critical(f"Failed to initialize EasyOCR reader: {e}")
        return

    crop_folders = [d for d in glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]

    for folder in crop_folders:
        base_name = os.path.basename(folder)
        logging.info(f"Re-OCRing crops for: {base_name}")
        manifest_path = os.path.join(folder, "manifest.json")

        if not os.path.exists(manifest_path):
            logging.warning(f"  - Manifest not found in {folder}. Skipping.")
            continue

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        final_annotations = []
        for item in manifest:
            crop_path = os.path.join(folder, item['crop_filename'])
            if not os.path.exists(crop_path):
                continue

            try:
                # Load the cropped image
                image_crop = cv2.imread(crop_path)
                if image_crop is None:
                    logging.warning(f"  - Could not read crop image: {crop_path}. Skipping.")
                    continue

                # Check orientation and rotate if necessary for better OCR
                orientation = item.get('orientation', 0)
                if orientation == 90:
                    # Rotate vertical text to be horizontal
                    image_crop = cv2.rotate(image_crop, cv2.ROTATE_90_CLOCKWISE)

                # Pass the image data (NumPy array) directly to EasyOCR
                results = reader.readtext(image_crop, detail=1)

                if not results: continue

                re_ocrd_text = " ".join([res[1] for res in results])
                avg_confidence = sum([res[2] for res in results]) / len(results)

                if avg_confidence >= min_conf:
                    final_annotations.append({
                        "text": re_ocrd_text,
                        "confidence": float(avg_confidence),
                        "bbox": item['original_bbox'],
                        "orientation": item.get('orientation', 0)
                    })
            except Exception as e:
                logging.error(f"  - Error processing crop {crop_path}: {e}")

        if final_annotations:
            output_json_path = os.path.join(output_dir, f"{base_name}_final.json")
            with open(output_json_path, 'w') as f:
                json.dump(final_annotations, f, indent=2)
            logging.info(f"  - Saved {len(final_annotations)} final annotations to {output_json_path}")

    logging.info("--- Finished Step: Re-OCR ---")


def run_coordinate_conversion_step(config: Dict):
    """
    Runs the coordinate conversion step to convert image coordinates to PDF points.
    """
    logging.info("--- Starting Step: Coordinate Conversion ---")

    coord_config = config.get('coordinate_conversion', {})
    image_dpi = coord_config.get('image_dpi', 600)
    # The source directory for JSONs is the output of the re_ocr step.
    input_dir_json_dir = config.get('re_ocr', {}).get('output_dir', "data/processed/metadata/final_annotations")
    image_perspective_dir = coord_config.get('image_perspective_dir', "data/outputs/image_perspective")
    pdf_perspective_dir = coord_config.get('pdf_perspective_dir', "data/outputs/json_pdf_perspective")

    if not os.path.exists(input_dir_json_dir):
        logging.warning(
            f"Input JSON directory not found: {input_dir_json_dir}. Skipping coordinate conversion.")
        return

    try:
        # The main coordinate conversion function is called.
        # Validation is off by default and only runs if convert_coord.py is executed directly.
        run_coordinate_conversion(
            input_dir_json_dir=input_dir_json_dir,
            image_perspective_dir=image_perspective_dir,
            pdf_perspective_dir=pdf_perspective_dir,
            dpi=image_dpi
        )

        logging.info(f"Coordinate conversion completed. Output saved to: {pdf_perspective_dir}")


    except Exception as e:
        logging.error(f"Failed to run coordinate conversion: {e}", exc_info=True)

    logging.info("--- Finished Step: Coordinate Conversion ---")

def run_visualization_step(config: Dict):
    """
    Visualizes the final text detections from the re-OCR step.
    """
    logging.info("--- Starting Step: Final Visualization ---")
    viz_config = config.get('visualization', {})
    re_ocr_config = config.get('re_ocr', {})

    image_source_dir = config.get('pipeline', {}).get('input_dir')
    annotation_dir = re_ocr_config.get('output_dir')
    output_dir = viz_config.get('output_dir')

    if not all([image_source_dir, annotation_dir, output_dir]):
        logging.error(
            "Visualization config is missing required paths. Check pipeline:input_dir, re_ocr:output_dir, and visualization:output_dir. Skipping.")
        return

    visualize_annotations(
        image_dir=image_source_dir,
        json_dir=annotation_dir,
        output_dir=output_dir
    )
    logging.info("--- Finished Step: Final Visualization ---")


def run_text_recognition_step_pipeline(config: Dict):
    """
    Wrapper function to run the TrOCR text recognition step in the pipeline.
    """
    run_text_recognition_step(config)

def main():
    """Main function to orchestrate the entire P&ID text extraction pipeline."""
    parser = argparse.ArgumentParser(description="P&ID Text Extraction Master Pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing raw P&ID images.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing YAML configuration files.")
    parser.add_argument("--start-at", type=str, default="pdf",
                        choices=['pdf', 'slice', 'meta', 'detect', 'group', 'crop', 'recognize', 'coord', 'viz'],
                        help="The pipeline step to start from.")
    parser.add_argument("--stop-at", type=str, default="viz",
                        choices=['pdf', 'slice', 'meta', 'detect', 'group', 'crop', 'recognize', 'coord', 'viz'],
                        help="The pipeline step to stop after.")
    parser.add_argument("--no-pdf", action="store_true", help="Explicitly skip the PDF conversion step.")
    parser.add_argument('--debug-grouping', action='store_true', help='Enable debug mode for the grouping step.')
    args = parser.parse_args()

    setup_logging()

    logging.info("=============================================")
    logging.info("= P&ID Text Extraction Pipeline Started =")
    logging.info("=============================================")
    logging.info(f"Start Time: {datetime.datetime.now()}")
    logging.info(f"Arguments: {args}")

    config = load_config(args.config_dir)
    grouping_args = argparse.Namespace(debug=args.debug_grouping)

    pipeline_steps = {
        'pdf': lambda: run_pdf_conversion_step(args.input_dir, config) if not args.no_pdf else logging.info(
            "Skipping PDF conversion."),
        'slice': lambda: run_slicing_step(args.input_dir, config),
        'meta': lambda: run_metadata_step(config),
        'detect': lambda: run_text_detection_step(config),
        'group': lambda: run_grouping_and_filtering_step(config, grouping_args),
        'crop': lambda: run_cropping_step(config),
        'recognize': lambda: run_text_recognition_step_pipeline(config),  # Updated to use TrOCR
        'coord': lambda: run_coordinate_conversion_step(config),
        'viz': lambda: run_visualization_step(config)
    }

    step_order = ['pdf', 'slice', 'meta', 'detect', 'group', 'crop', 'recognize', 'coord', 'viz']  # Updated step order

    try:
        start_index = step_order.index(args.start_at)
        stop_index = step_order.index(args.stop_at)
    except ValueError:
        logging.error("Invalid step name provided.")
        return

    if start_index > stop_index:
        logging.error("Start step cannot be after stop step.")
        return

    for i in range(start_index, stop_index + 1):
        step_name = step_order[i]
        try:
            logging.info(f"\n{'='*20} EXECUTING STEP: {step_name.upper()} {'='*20}")
            pipeline_steps[step_name]()
            logging.info(f"--- Finished Step: {step_name.upper()} ---")
        except Exception as e:
            logging.critical(f"Pipeline failed at step '{step_name}'. Error: {e}", exc_info=True)
            break

    logging.info("\n=============================================")
    logging.info("= Pipeline Finished =")
    logging.info("=============================================")


if __name__ == "__main__":
    main()
