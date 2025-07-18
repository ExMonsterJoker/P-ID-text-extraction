# run_pipeline.py
import os
import sys
import glob
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
from src.convert_coord import main as run_coordinate_conversion
from src.data_loader.sahi_slicer import SahiSlicer
from src.data_loader.metadata_manager import MetadataManager
from src.text_detection.process_tiles_ocr import process_tiles_with_ocr
# The grouping main now handles all filtering internally
from src.grouping.run_grouping_pipeline import main as run_grouping_main
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
    """Generates core (non-overlapping) metadata for each set of tiles."""
    logging.info("--- Starting Step 2: Metadata Generation (Core Tiles) ---")

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
            # MODIFIED: Look for tiles_metadata.json
            json_path = os.path.join(image_dir, "tiles_metadata.json")
            if not os.path.exists(json_path):
                logging.warning(f"No 'tiles_metadata.json' in {image_dir}. Skipping.")
                continue

            with open(json_path, 'r') as f:
                source_image = json.load(f)["source_image"]

            logging.info(f"Generating core metadata for: {Path(source_image).name}")
            # The load_metadata function is now correctly loading JSON
            tile_metadata = SahiSlicer.load_metadata(json_path)
            core_tiles = manager.compute_core_tiles(tile_metadata)
            manager.save_core_tile_metadata(core_tiles, source_image)
            logging.info(f"Saved core metadata for {len(core_tiles)} tiles.")

        except Exception as e:
            logging.error(f"Failed to generate metadata for {image_dir}: {e}", exc_info=True)


def run_ocr_step(config: dict):
    """Runs the OCR step on all sliced tiles."""
    logging.info("--- Starting Step 3: OCR Processing ---")
    process_tiles_with_ocr(config)
    logging.info("--- Finished OCR Processing ---")


def run_grouping_and_filtering_step(config: dict, grouping_args: argparse.Namespace):
    """Runs the entire text grouping and filtering pipeline."""
    logging.info("--- Starting Step 4: Text Grouping and Filtering ---")
    run_grouping_main(grouping_args)
    logging.info("--- Finished Text Grouping and Filtering ---")


def run_cropping_step(config: Dict):
    """
    Runs the cropping process and creates a manifest for each set of crops.
    """
    logging.info("--- Starting Step: Cropping Images ---")

    cropping_config = config.get('cropping', {})
    image_source_dir = cropping_config.get('image_source_dir')
    # The directory for grouped text is now produced by the single grouping step
    grouped_text_dir = "data/processed/metadata/final_grouped_text"
    output_dir = cropping_config.get('output_dir')
    padding = cropping_config.get('padding', 10)
    min_confidence = cropping_config.get('min_confidence', 0.5)

    if not all([grouped_text_dir, output_dir, image_source_dir]):
        logging.error(
            "Cropping config missing required keys ('image_source_dir', 'output_dir'). Check config. Skipping step.")
        return

    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(grouped_text_dir, '*_grouped_text.json'))

    logging.info(f"Found {len(json_files)} JSON files to process for cropping.")

    for json_file in json_files:
        base_name = os.path.basename(json_file).replace('_grouped_text.json', '')
        logging.info(f"Processing: {base_name}")

        image_path = next(iter(glob(os.path.join(image_source_dir, f"{base_name}.*"))), None)
        if not image_path:
            logging.warning(f"  - Could not find matching image for '{base_name}'. Skipping.")
            continue

        with open(json_file, 'r') as f:
            detections = json.load(f)

        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        filtered_detections = [det for det in detections if det.get('confidence', 0) >= min_confidence]

        if not filtered_detections:
            logging.info(f"  - No detections above confidence threshold {min_confidence} for {base_name}.")
            continue

        manifest_entries = crop_image(
            image_path=image_path,
            detections=filtered_detections,
            output_dir=image_output_dir,
            padding=padding
        )

        if manifest_entries:
            manifest_path = os.path.join(image_output_dir, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest_entries, f, indent=2)
            logging.info(f"  - Saved crop manifest to {manifest_path}")

    logging.info("--- Finished Step: Cropping Images ---")


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
    image_perspective_dir = coord_config.get('image_perspective_dir', "data/processed/metadata/final_annotations")
    pdf_perspective_dir = coord_config.get('pdf_perspective_dir', "data/outputs/json_pdf_perspective")

    if not os.path.exists(image_perspective_dir):
        logging.warning(
            f"Image perspective directory not found: {image_perspective_dir}. Skipping coordinate conversion.")
        return

    try:
        run_coordinate_conversion(
            input_dir=image_perspective_dir,
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


def main():
    """Main function to orchestrate the entire P&ID text extraction pipeline."""
    parser = argparse.ArgumentParser(description="P&ID Text Extraction Master Pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing raw P&ID images.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing YAML configuration files.")
    parser.add_argument("--start-at", type=str, default="pdf",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group', 'crop', 're_ocr', 'coord', 'viz'],
                        help="The pipeline step to start from.")
    parser.add_argument("--stop-at", type=str, default="viz",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group', 'crop', 're_ocr', 'coord', 'viz'],
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
    # The debug argument is now passed to the consolidated grouping step
    grouping_args = argparse.Namespace(debug=args.debug_grouping)

    pipeline_steps = {
        'pdf': lambda: run_pdf_conversion_step(args.input_dir, config) if not args.no_pdf else logging.info(
            "Skipping PDF conversion."),
        'slice': lambda: run_slicing_step(args.input_dir, config),
        'meta': lambda: run_metadata_step(config),
        'ocr': lambda: run_ocr_step(config),
        'group': lambda: run_grouping_and_filtering_step(config, grouping_args),
        'crop': lambda: run_cropping_step(config),
        're_ocr': lambda: run_re_ocr_step(config),
        'coord': lambda: run_coordinate_conversion_step(config),
        'viz': lambda: run_visualization_step(config)
    }

    step_order = ['pdf', 'slice', 'meta', 'ocr', 'group', 'crop', 're_ocr', 'coord', 'viz']

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