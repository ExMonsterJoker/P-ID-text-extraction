# run_pipeline.py
import os
import sys
import glob
import yaml
import logging
import argparse
import datetime
from PIL import Image
from pathlib import Path

# --- Add project root to system path for module imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import Core Logic from Project Modules ---
from src.data_loader.sahi_slicer import SahiSlicer
from src.data_loader.metadata_manager import MetadataManager
from src.text_detection.process_tiles_ocr import process_tiles_with_ocr
from src.grouping.run_grouping_pipeline import main as run_grouping_main
from src.grouping.visualize_final_groups import visualize_final_results

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


def load_config(config_dir: str) -> dict:
    """Loads all YAML configuration files from a directory and merges them."""
    logging.info(f"Loading configuration from: {config_dir}")
    config = {}
    for config_file in glob.glob(os.path.join(config_dir, "*.yaml")):
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))
    logging.info("Configuration loaded successfully.")
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

    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.tiff"))

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

            metadata_path = os.path.join(image_output_dir, "tiles_metadata.yaml")
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

    image_dirs = [d for d in glob.glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning("No sliced image directories found to generate metadata from.")
        return

    for image_dir in image_dirs:
        try:
            yaml_path = os.path.join(image_dir, "tiles_metadata.yaml")
            if not os.path.exists(yaml_path):
                logging.warning(f"No 'tiles_metadata.yaml' in {image_dir}. Skipping.")
                continue

            with open(yaml_path, 'r') as f:
                source_image = yaml.safe_load(f)["source_image"]

            logging.info(f"Generating core metadata for: {Path(source_image).name}")
            tile_metadata = SahiSlicer.load_metadata(yaml_path)
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


def run_grouping_step(config: dict):
    """Runs the final text detection grouping and de-duplication step."""
    logging.info("--- Starting Step 4: Text Grouping ---")
    run_grouping_main()
    logging.info("--- Finished Text Grouping ---")


def run_visualization_step(config: dict):
    """Runs the final visualization step."""
    logging.info("--- Starting Step 5: Final Visualization ---")
    visualize_final_results(show_grid=True)
    logging.info("--- Finished Final Visualization ---")


def main():
    """Main function to orchestrate the entire P&ID text extraction pipeline."""
    parser = argparse.ArgumentParser(description="P&ID Text Extraction Master Pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Directory containing raw P&ID images (or a 'pdfs' subdirectory).")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML configuration files.")
    parser.add_argument("--start-at", type=str, default="pdf",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group', 'viz'],
                        help="The pipeline step to start from.")
    parser.add_argument("--stop-at", type=str, default="viz",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group', 'viz'],
                        help="The pipeline step to stop after.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Explicitly skip the PDF conversion step.")

    args = parser.parse_args()

    setup_logging()

    logging.info("=============================================")
    logging.info("= P&ID Text Extraction Pipeline Started =")
    logging.info("=============================================")
    logging.info(f"Start Time: {datetime.datetime.now()}")
    logging.info(f"Arguments: {args}")

    config = load_config(args.config_dir)

    pipeline_steps = {
        'pdf': lambda: run_pdf_conversion_step(args.input_dir, config) if not args.no_pdf else logging.info(
            "Skipping PDF conversion as requested."),
        'slice': lambda: run_slicing_step(args.input_dir, config),
        'meta': lambda: run_metadata_step(config),
        'ocr': lambda: run_ocr_step(config),
        'group': lambda: run_grouping_step(config),
        'viz': lambda: run_visualization_step(config)
    }

    step_order = ['pdf', 'slice', 'meta', 'ocr', 'group', 'viz']

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
            pipeline_steps[step_name]()
        except Exception as e:
            logging.critical(f"Pipeline failed at step '{step_name}'. Error: {e}", exc_info=True)
            break

    logging.info("=============================================")
    logging.info("= Pipeline Finished =")
    logging.info("=============================================")


if __name__ == "__main__":
    main()
