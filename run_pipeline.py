# run_pipeline.py
import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
from glob import glob
import json
import cv2

# --- Add project root to system path for module imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- Import Core Logic from Project Modules ---
from src.convert_coord import main as run_coordinate_conversion
from src.data_loader.sahi_slicer import SahiSlicer
from src.data_loader.metadata_manager import MetadataManager
from src.text_detection.text_detection import text_detection
from src.text_detection.text_recognition import run_text_recognition_step
from src.grouping.grouping_logic import BoundingBoxGrouper
from src.cropping.cropping_Images import crop_image, crop_segmented_lines
from src.segmentation.hpp_segmenter import HPPSegmenter
from src.visualization.visualizer import visualize_annotations
from configs import get_config

# --- Import Optional PDF Converter ---
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
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info("Logging configured.")


def run_pdf_conversion_step(input_dir: str):
    """Runs the PDF to image conversion step."""
    if not PDF_CONVERTER_AVAILABLE:
        logging.warning("PDF_to_image.py not found. Skipping PDF conversion.")
        return

    logging.info("--- Starting Step 0: PDF to Image Conversion ---")
    pdf_input_dir = os.path.join(input_dir, "pdfs")
    if not os.path.exists(pdf_input_dir):
        logging.info("No 'pdfs' subdirectory found. Skipping PDF conversion.")
        return

    pdf_config = get_config('pdf_conversion')
    dpi = pdf_config.get('dpi', 600)
    image_format = pdf_config.get('format', 'PNG')

    pdf_to_image_high_quality(pdf_input_dir, input_dir, dpi=dpi, image_format=image_format)
    logging.info("--- Finished PDF to Image Conversion ---")


def run_slicing_step(input_dir: str):
    """Runs the image slicing step for all images in the input directory."""
    logging.info("--- Starting Step 1: Image Slicing ---")
    slicer = SahiSlicer()

    data_loader_config = get_config('data_loader')
    output_dir = data_loader_config.get('sahi_slicer_output_dir', 'data/processed/tiles')
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
                from PIL import Image
                tile_img = Image.fromarray(tile_img_np)
                tile_path = os.path.join(image_output_dir, f"tile_{metadata[i].tile_id}.png")
                tile_img.save(tile_path)
            logging.info(f"Saved {len(tiles)} tiles to {image_output_dir}")

        except Exception as e:
            logging.error(f"Failed to slice image {image_path}: {e}", exc_info=True)


def run_metadata_step():
    """Generates metadata for each set of tiles."""
    logging.info("--- Starting Step 2: Metadata Generation ---")
    data_loader_config = get_config('data_loader')
    metadata_base_dir = data_loader_config.get('metadata_output_dir', "data/processed/metadata")
    tiles_base_dir = data_loader_config.get('sahi_slicer_output_dir', "data/processed/tiles")
    manager = MetadataManager(metadata_base_dir)

    image_dirs = [d for d in glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning("No sliced image directories found to generate metadata from.")
        return

    for image_dir in image_dirs:
        try:
            json_path = os.path.join(image_dir, "tiles_metadata.json")
            if not os.path.exists(json_path):
                logging.warning(f"No 'tiles_metadata.json' in {image_dir}. Skipping.")
                continue

            with open(json_path, 'r') as f:
                metadata_file = json.load(f)
                source_image = metadata_file["source_image"]

            logging.info(f"Processing metadata for: {Path(source_image).name}")
            tile_metadata = SahiSlicer.load_metadata(json_path)
            manager.save_tile_metadata(tile_metadata, source_image)
            logging.info(f"Saved tile metadata for {len(tile_metadata)} tiles.")

        except Exception as e:
            logging.error(f"Failed to process metadata for {image_dir}: {e}", exc_info=True)

    logging.info("--- Finished Metadata Generation ---")


def run_text_detection_step():
    """Runs text detection on all sliced tiles."""
    logging.info("--- Starting Step 3: Text Detection ---")
    data_loader_config = get_config('data_loader')
    tiles_base_dir = data_loader_config.get('sahi_slicer_output_dir', 'data/processed/tiles')
    detection_metadata_dir = data_loader_config.get('detection_metadata_dir', 'data/processed/metadata/detection_metadata')
    os.makedirs(detection_metadata_dir, exist_ok=True)

    detector = text_detection()
    logging.info("Text detector initialized successfully")

    image_dirs = [d for d in glob(os.path.join(tiles_base_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning("No sliced image directories found for text detection.")
        return

    total_tiles_processed, total_detections_found = 0, 0
    for image_dir in image_dirs:
        try:
            image_name = os.path.basename(image_dir)
            logging.info(f"Processing text detection for image: {image_name}")
            image_detection_dir = os.path.join(detection_metadata_dir, image_name)
            os.makedirs(image_detection_dir, exist_ok=True)

            tiles_metadata_path = os.path.join(image_dir, "tiles_metadata.json")
            if not os.path.exists(tiles_metadata_path):
                logging.warning(f"No tiles metadata found for {image_name}. Skipping.")
                continue

            with open(tiles_metadata_path, 'r') as f:
                tiles_metadata = json.load(f)

            source_image = tiles_metadata.get("source_image", "")
            original_image_size = tiles_metadata.get("original_size", [])

            tile_files = glob(os.path.join(image_dir, "tile_*.png"))
            image_detections_count = 0
            for tile_file in tile_files:
                try:
                    tile_id = Path(tile_file).stem.replace("tile_", "")
                    detections = detector.detect_text(tile_file)
                    if not detections: continue

                    tile_meta = next((t for t in tiles_metadata.get('tiles', []) if t.get('tile_id') == tile_id), None)
                    if not tile_meta:
                        logging.warning(f"No metadata found for tile {tile_id}")
                        continue

                    tile_detections = [
                        {
                            'bbox': d.bbox, 'bbox_normalized': d.bbox_normalized, 'rotation_angle': d.rotation_angle,
                            'tile_id': tile_id, 'tile_path': tile_file, 'tile_coordinates': tile_meta.get('coordinates', []),
                            'grid_position': tile_meta.get('grid_position', []), 'source_image': source_image,
                            'original_image_size': original_image_size, 'detection_type': 'craft_detection',
                            'tile_size': tile_meta.get('tile_size', [])
                        } for d in detections
                    ]

                    if tile_detections:
                        with open(os.path.join(image_detection_dir, f"{tile_id}_ocr.json"), 'w') as f:
                            json.dump(tile_detections, f, indent=2)
                        image_detections_count += len(tile_detections)
                        total_tiles_processed += 1

                except Exception as e:
                    logging.error(f"Error processing tile {tile_file}: {e}")

            total_detections_found += image_detections_count
            logging.info(f"Completed {image_name}: {image_detections_count} detections from {len(tile_files)} tiles")

        except Exception as e:
            logging.error(f"Error processing image directory {image_dir}: {e}", exc_info=True)

    logging.info(f"Text detection completed. Total detections: {total_detections_found}")


def run_grouping_and_filtering_step(grouping_args: argparse.Namespace):
    """Runs text grouping and filtering."""
    logging.info("--- Starting Step 4: Text Grouping and Filtering ---")
    grouper = BoundingBoxGrouper()
    grouper.process_all_images()
    logging.info("--- Finished Text Grouping and Filtering ---")


def run_cropping_step():
    """Runs cropping process."""
    logging.info("--- Starting Step 5: Cropping Images ---")
    cropping_config = get_config('cropping')
    data_loader_config = get_config('data_loader')
    pipeline_config = get_config('pipeline')

    image_source_dir = pipeline_config.get('input_dir', 'data/raw')
    grouped_text_dir = data_loader_config.get('group_detection_metadata_dir', 'data/processed/metadata/group_detection_metadata')
    output_dir = cropping_config.get('output_dir', 'data/processed/cropping')
    padding = cropping_config.get('padding', 10)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(grouped_text_dir):
        logging.error(f"Grouped text directory not found: {grouped_text_dir}")
        return

    json_files = glob(os.path.join(grouped_text_dir, '*.json'))
    if not json_files:
        logging.warning(f"No JSON files found in: {grouped_text_dir}")
        return

    total_crops, total_images = 0, 0
    for json_file in json_files:
        try:
            base_name = Path(json_file).stem.replace('_grouped_text', '').replace('_grouped', '')
            logging.info(f"Processing cropping for: {base_name}")

            image_path = next((p for ext in ['png', 'jpg', 'jpeg', 'tiff'] for p in glob(os.path.join(image_source_dir, f"{base_name}.{ext}"))), None)
            if not image_path:
                logging.warning(f"  - Could not find source image for '{base_name}'. Skipping.")
                continue

            with open(json_file, 'r') as f:
                detections = json.load(f)
            if not detections: continue

            image_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(image_output_dir, exist_ok=True)

            manifest_entries = crop_image(image_path, detections, image_output_dir, padding)
            if manifest_entries:
                with open(os.path.join(image_output_dir, "manifest.json"), 'w') as f:
                    json.dump(manifest_entries, f, indent=2)
                total_crops += len(manifest_entries)
                total_images += 1
                logging.info(f"  - Created {len(manifest_entries)} crops.")
        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}", exc_info=True)

    logging.info(f"Cropping completed. Total crops: {total_crops} from {total_images} images.")


def run_hpp_segmentation_step():
    """Runs HPP segmentation on cropped images to find individual text lines."""
    logging.info("--- Starting Step 6: HPP Segmentation ---")
    hpp_config = get_config('hpp_segmentation')
    data_loader_config = get_config('data_loader')

    cropping_output_dir = data_loader_config.get('cropping_output_dir')
    segmentation_output_dir = hpp_config.get('segmented_crops_dir')
    os.makedirs(segmentation_output_dir, exist_ok=True)

    segmenter = HPPSegmenter(hpp_config)

    image_dirs = [d for d in glob(os.path.join(cropping_output_dir, "*")) if os.path.isdir(d)]
    if not image_dirs:
        logging.warning("No cropped image directories found for segmentation.")
        return

    total_lines_found = 0
    for image_dir in image_dirs:
        try:
            base_name = os.path.basename(image_dir)
            logging.info(f"Processing HPP segmentation for: {base_name}")

            manifest_path = os.path.join(image_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                logging.warning(f"  - No manifest.json found in {image_dir}. Skipping.")
                continue

            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)

            logging.info(f"  - Found {len(manifest_data)} crops to process.")

            all_line_boxes = []
            for crop_info in manifest_data:
                try:
                    crop_path = os.path.join(image_dir, crop_info['crop_filename'])
                    crop_image_data = cv2.imread(crop_path)

                    if crop_image_data is None:
                        logging.warning(f"    - Could not read crop image {crop_path}. Skipping.")
                        continue

                    line_boxes = segmenter.segment_multiline_crop(crop_image_data, crop_info['original_bbox'])

                    for i, line_box in enumerate(line_boxes):
                        all_line_boxes.append({
                            'original_crop_filename': crop_info['crop_filename'],
                            'line_number': i, 'line_bbox': line_box,
                            'original_text': crop_info.get('original_text', 'unknown'),
                            'orientation': crop_info.get('orientation', 0),
                            'original_image_size': crop_info.get('original_image_size')
                        })
                except Exception as e:
                    logging.error(f"    - Error processing crop {crop_info.get('crop_filename', 'N/A')}: {e}", exc_info=True)

            total_lines_found += len(all_line_boxes)
            output_json_path = os.path.join(segmentation_output_dir, f"{base_name}_segmented_lines.json")
            with open(output_json_path, 'w') as f:
                json.dump(all_line_boxes, f, indent=2)
            logging.info(f"  - Saved {len(all_line_boxes)} segmented line boxes to {output_json_path}")

        except Exception as e:
            logging.error(f"Failed to process HPP segmentation for directory {image_dir}: {e}", exc_info=True)

    logging.info(f"HPP segmentation completed. Total lines found: {total_lines_found}")


def run_line_cropping_step():
    """Crops the final text lines from the original images."""
    logging.info("--- Starting Step 7: Final Line Cropping ---")
    hpp_config = get_config('hpp_segmentation')
    cropping_config = get_config('cropping')
    pipeline_config = get_config('pipeline')

    image_source_dir = pipeline_config.get('input_dir')
    segmentation_input_dir = hpp_config.get('segmented_crops_dir')
    padding = cropping_config.get('padding', 10)

    json_files = glob(os.path.join(segmentation_input_dir, '*_segmented_lines.json'))
    if not json_files:
        logging.warning(f"No segmented line JSON files found in: {segmentation_input_dir}")
        return

    total_line_crops = 0
    for json_file in json_files:
        try:
            base_name = Path(json_file).stem.replace('_segmented_lines', '')
            logging.info(f"Processing line cropping for: {base_name}")

            image_path = next((p for ext in ['png', 'jpg'] for p in glob(os.path.join(image_source_dir, f"{base_name}.{ext}"))), None)
            if not image_path:
                logging.warning(f"  - Source image for '{base_name}' not found. Skipping.")
                continue

            with open(json_file, 'r') as f:
                line_detections = json.load(f)

            if not line_detections:
                logging.info(f"  - No line detections found for {base_name}. Skipping.")
                continue

            logging.info(f"  - Found {len(line_detections)} lines to crop.")
            image_output_dir = os.path.join(segmentation_input_dir, base_name)
            os.makedirs(image_output_dir, exist_ok=True)

            manifest_entries = crop_segmented_lines(image_path, line_detections, image_output_dir, padding)

            if manifest_entries:
                manifest_path = os.path.join(image_output_dir, "segmented_manifest.json")
                with open(manifest_path, 'w') as f:
                    json.dump(manifest_entries, f, indent=2)
                logging.info(f"  - Created {len(manifest_entries)} line crops and manifest for {base_name}.")
                total_line_crops += len(manifest_entries)

        except Exception as e:
            logging.error(f"Failed to process line cropping for file {json_file}: {e}", exc_info=True)

    logging.info(f"Line cropping completed. Total line crops created: {total_line_crops}")


def run_re_ocr_step():
    """Runs re-OCR on cropped images."""
    # This step is now part of the 'recognize' step using TrOCR + EasyOCR fallback.
    # Kept for compatibility if needed, but the main logic is in run_text_recognition_step.
    logging.warning("Skipping deprecated re-OCR step. Use 'recognize' step instead.")


def run_coordinate_conversion_step():
    """Converts image coordinates to PDF points."""
    logging.info("--- Starting Step: Coordinate Conversion ---")
    coord_config = get_config('coordinate_conversion')
    data_loader_config = get_config('data_loader')

    input_dir_json = data_loader_config.get('text_recognition_output_dir', 'data/processed/metadata/final_annotations')
    if not os.path.exists(input_dir_json):
        logging.warning(f"Input JSON directory not found: {input_dir_json}. Skipping coordinate conversion.")
        return

    run_coordinate_conversion(
        input_dir_json_dir=input_dir_json,
        image_perspective_dir=coord_config.get('image_perspective_dir'),
        pdf_perspective_dir=coord_config.get('pdf_perspective_dir'),
        dpi=coord_config.get('image_dpi', 600)
    )
    logging.info("--- Finished Step: Coordinate Conversion ---")


def run_visualization_step():
    """Visualizes final text detections."""
    logging.info("--- Starting Step: Final Visualization ---")
    viz_config = get_config('visualization')
    pipeline_config = get_config('pipeline')
    text_recognition_config = get_config('data_loader')

    visualize_annotations(
        image_dir=pipeline_config.get('input_dir'),
        json_dir=text_recognition_config.get('text_recognition_output_dir'),
        output_dir=viz_config.get('output_dir')
    )
    logging.info("--- Finished Step: Final Visualization ---")


def run_text_recognition_step_pipeline():
    """Wrapper for the text recognition step."""
    hpp_config = get_config('hpp_segmentation')
    input_dir = hpp_config.get('segmented_crops_dir')
    run_text_recognition_step(specific_dir=input_dir)


def main():
    """Main function to orchestrate the entire P&ID text extraction pipeline."""
    parser = argparse.ArgumentParser(description="P&ID Text Extraction Master Pipeline")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing raw P&ID images.")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory for config files (used by manager).")
    parser.add_argument("--start-at", type=str, default="pdf",
                        choices=['pdf', 'slice', 'meta', 'detect', 'group', 'crop', 'segment', 'recrop', 'recognize', 'coord', 'viz'],
                        help="The pipeline step to start from.")
    parser.add_argument("--stop-at", type=str, default="viz",
                        choices=['pdf', 'slice', 'meta', 'detect', 'group', 'crop', 'segment', 'recrop', 'recognize', 'coord', 'viz'],
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

    grouping_args = argparse.Namespace(debug=args.debug_grouping)

    pipeline_steps = {
        'pdf': lambda: run_pdf_conversion_step(args.input_dir) if not args.no_pdf else logging.info("Skipping PDF conversion."),
        'slice': lambda: run_slicing_step(args.input_dir),
        'meta': run_metadata_step,
        'detect': run_text_detection_step,
        'group': lambda: run_grouping_and_filtering_step(grouping_args),
        'crop': run_cropping_step,
        'segment': run_hpp_segmentation_step,
        'recrop': run_line_cropping_step,
        'recognize': run_text_recognition_step_pipeline,
        'coord': run_coordinate_conversion_step,
        'viz': run_visualization_step
    }
    step_order = list(pipeline_steps.keys())

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
