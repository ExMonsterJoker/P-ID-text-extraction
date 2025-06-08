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
# This allows the script to be run from anywhere and still find the 'src' modules.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import Core Logic from Project Modules ---
# It's a best practice to encapsulate the core logic of each script into functions
# that can be imported, rather than running the whole file.
# You might need to slightly refactor your existing scripts to ensure the main logic
# is within a function that can be called from here.

# From src.data_loader
from src.data_loader.sahi_slicer import SahiSlicer
from src.data_loader.metadata_manager import MetadataManager

# From src.text_detection
from src.text_detection.process_tiles_ocr import process_tiles_with_ocr

# From src.grouping
from src.grouping.run_grouping_pipeline import main as run_grouping_main

# From root (if you have PDF conversion)
try:
    from PDF_to_image import pdf_to_image_high_quality

    PDF_CONVERTER_AVAILABLE = True
except ImportError:
    PDF_CONVERTER_AVAILABLE = False


def setup_logging(log_level="INFO"):
    """
    Sets up a centralized logger for the pipeline.
    """
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
            # You can also add a FileHandler here to log to a file
            # logging.FileHandler("pipeline.log")
        ]
    )
    # Suppress verbose messages from PIL
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info("Logging configured.")


def load_config(config_dir: str) -> dict:
    """
    Loads all YAML configuration files from a directory and merges them.

    Args:
        config_dir: Path to the directory containing config files.

    Returns:
        A single dictionary containing all configuration parameters.
    """
    logging.info(f"Loading configuration from: {config_dir}")
    config = {}
    for config_file in glob.glob(os.path.join(config_dir, "*.yaml")):
        with open(config_file, 'r') as f:
            # The key for the config dict will be the filename without extension
            config_key = Path(config_file).stem
            config.update(yaml.safe_load(f))
    logging.info("Configuration loaded successfully.")
    return config


def run_pdf_conversion_step(input_dir: str, output_dir: str, config: dict):
    """
    Runs the PDF to image conversion step.
    """
    if not PDF_CONVERTER_AVAILABLE:
        logging.warning("PDF_to_image.py not found or has errors. Skipping PDF conversion.")
        return

    logging.info("--- Starting Step 0: PDF to Image Conversion ---")
    pdf_input_dir = os.path.join(input_dir, "pdfs")
    if not os.path.exists(pdf_input_dir):
        logging.info("No 'pdfs' subdirectory found in input directory. Skipping PDF conversion.")
        return

    # Use parameters from a potential 'pdf_conversion' section in config, or defaults
    pdf_config = config.get('pdf_conversion', {})
    dpi = pdf_config.get('dpi', 600)
    image_format = pdf_config.get('format', 'PNG')

    # The converter saves images to the main input_dir, ready for slicing.
    pdf_to_image_high_quality(pdf_input_dir, input_dir, dpi=dpi, image_format=image_format)
    logging.info("--- Finished PDF to Image Conversion ---")


def run_slicing_step(input_dir: str, output_dir: str, config: dict):
    """
    Runs the image slicing step for all images in the input directory.
    """
    logging.info("--- Starting Step 1: Image Slicing ---")
    slicer_config = config.get("sahi_slicer", {
        "tile_size": 1080, "overlap_ratio": 0.5, "min_area_ratio": 0.1, "verbose": False
    })
    slicer = SahiSlicer(slicer_config)

    tiles_base_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_base_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.tiff"))

    if not image_files:
        logging.warning(f"No image files found in '{input_dir}'.")
        return

    for image_path in image_files:
        try:
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            logging.info(f"Slicing image: {image_filename}")

            tiles, metadata = slicer.slice(image_path)

            image_output_dir = os.path.join(tiles_base_dir, image_filename)
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
            continue
    logging.info("--- Finished Image Slicing ---")


def run_metadata_step(output_dir: str, config: dict):
    """
    Runs the metadata generation step (e.g., core tiles).
    Note: This step is now often integrated, but can be run explicitly if needed.
    The primary purpose here is to create the core_tile_metadata.
    """
    logging.info("--- Starting Step 2: Metadata Generation (Core Tiles) ---")
    metadata_base_dir = os.path.join(output_dir, "metadata")
    manager = MetadataManager(metadata_base_dir)

    tiles_base_dir = os.path.join(output_dir, "tiles")
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
            continue
    logging.info("--- Finished Metadata Generation ---")


def run_ocr_step(output_dir: str, config: dict):
    """
    Runs the OCR step on all sliced tiles.
    """
    logging.info("--- Starting Step 3: OCR Processing ---")
    # REVERTED: Removed 'batch_size' from the config dictionary
    ocr_config = config.get("ocr", {
        "languages": ['en'], "confidence_threshold": 0.3, "gpu": True,
        "use_rotation": True, "filter_overlaps": True, "rotation_angles": [0, 90]
    })

    tiles_base_dir = os.path.join(output_dir, "tiles")
    metadata_output_dir = os.path.join(output_dir, "metadata")

    # The imported function handles the entire OCR process
    process_tiles_with_ocr(
        tiles_base_dir=tiles_base_dir,
        metadata_output_dir=metadata_output_dir,
        **ocr_config
    )
    logging.info("--- Finished OCR Processing ---")


def run_grouping_step(config: dict):
    """
    Runs the final text detection grouping and de-duplication step.
    """
    logging.info("--- Starting Step 4: Text Grouping ---")
    # The imported `run_grouping_main` function should handle its own setup
    # based on the config files it reads.
    # Note: Ensure `run_grouping_pipeline.py` is configured to read from the correct paths.
    run_grouping_main()
    logging.info("--- Finished Text Grouping ---")


def main():
    """
    Main function to orchestrate the entire P&ID text extraction pipeline.
    """
    parser = argparse.ArgumentParser(description="P&ID Text Extraction Master Pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Directory containing raw P&ID images (or a 'pdfs' subdirectory).")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Base directory to save all processed outputs (tiles, metadata).")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing YAML configuration files.")
    parser.add_argument("--start-at", type=str, default="pdf",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group'],
                        help="The pipeline step to start from.")
    parser.add_argument("--stop-at", type=str, default="group",
                        choices=['pdf', 'slice', 'meta', 'ocr', 'group'],
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

    # Define the sequence of steps
    pipeline_steps = {
        'pdf': lambda: run_pdf_conversion_step(args.input_dir, args.output_dir,
                                               config) if not args.no_pdf else logging.info(
            "Skipping PDF conversion as requested."),
        'slice': lambda: run_slicing_step(args.input_dir, args.output_dir, config),
        'meta': lambda: run_metadata_step(args.output_dir, config),
        'ocr': lambda: run_ocr_step(args.output_dir, config),
        'group': lambda: run_grouping_step(config),
    }

    step_order = ['pdf', 'slice', 'meta', 'ocr', 'group']

    # Determine which steps to run
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
            # Stop the pipeline on critical failure
            break

    logging.info("=============================================")
    logging.info("= Pipeline Finished =")
    logging.info("=============================================")


if __name__ == "__main__":
    main()

