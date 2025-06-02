import os
import sys
import glob
import yaml
from typing import List, Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.text_detection.ocr_detector import EasyOCRDetector, OCRDetection
from src.data_loader.metadata_manager import MetadataManager
from src.data_loader.sahi_slicer import SahiSlicer


def extract_tile_id_from_filename(filename: str) -> str:
    """
    Extract tile ID from filename
    Example: tile_DURI-OTPF0_590966_8c433f88_T0000.png -> DURI-OTPF0_590966_8c433f88_T0000
    """
    # Remove 'tile_' prefix and file extension
    if filename.startswith('tile_'):
        filename = filename[5:]  # Remove 'tile_' prefix

    # Remove file extension
    tile_id = os.path.splitext(filename)[0]
    return tile_id


def process_tiles_with_ocr(
        tiles_base_dir: str = "data/processed/tiles",
        metadata_output_dir: str = "data/processed/metadata",
        languages: List[str] = ['en'],
        confidence_threshold: float = 0.5,
        gpu: bool = True,
        use_rotation: bool = True,  # New parameter
        filter_overlaps: bool = True,  # New parameter
        rotation_angles: List[int] = [0, 90, 180, 270]  # New parameter
):
    """
    Process all tiles with OCR and save results as metadata

    Args:
        tiles_base_dir: Base directory containing tile folders
        metadata_output_dir: Directory to save metadata
        languages: List of language codes for OCR
        confidence_threshold: Minimum confidence for OCR detections
        gpu: Whether to use GPU acceleration
        use_rotation: Whether to try multiple rotations for better text detection
        filter_overlaps: Whether to filter overlapping detections from different rotations
        rotation_angles: List of rotation angles to try (in degrees)
    """

    # Get absolute path from project root
    if not os.path.isabs(tiles_base_dir):
        # Get project root (go up from src/text_detection to project root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tiles_base_dir = os.path.join(project_root, tiles_base_dir)
        metadata_output_dir = os.path.join(project_root, metadata_output_dir)

    print(f"Looking for tiles in: {tiles_base_dir}")
    print(f"Metadata output directory: {metadata_output_dir}")
    print(f"Rotation enabled: {use_rotation}")
    if use_rotation:
        print(f"Rotation angles: {rotation_angles}")
        print(f"Overlap filtering: {filter_overlaps}")

    # Check if tiles directory exists
    if not os.path.exists(tiles_base_dir):
        print(f"Error: Tiles directory does not exist: {tiles_base_dir}")
        return

    # Initialize OCR detector
    print("Initializing EasyOCR detector...")
    ocr_detector = EasyOCRDetector(languages=languages, gpu=gpu)

    # Initialize metadata manager
    metadata_manager = MetadataManager(metadata_output_dir, pipeline_version="1.0")

    # Find all image directories in the tiles folder
    image_dirs = []
    for item in os.listdir(tiles_base_dir):
        item_path = os.path.join(tiles_base_dir, item)
        if os.path.isdir(item_path):
            image_dirs.append(item_path)

    print(f"Contents of {tiles_base_dir}:")
    for item in os.listdir(tiles_base_dir):
        item_path = os.path.join(tiles_base_dir, item)
        item_type = "DIR" if os.path.isdir(item_path) else "FILE"
        print(f"  {item_type}: {item}")

    if not image_dirs:
        print(f"No image directories found in {tiles_base_dir}")
        return

    print(f"\nFound {len(image_dirs)} image directories to process:")
    for img_dir in image_dirs:
        print(f"  - {os.path.basename(img_dir)}")

    for image_dir in image_dirs:
        image_name = os.path.basename(image_dir)
        print(f"\n{'=' * 60}")
        print(f"Processing OCR for image: {image_name}")
        print(f"Directory: {image_dir}")
        print(f"{'=' * 60}")

        # Check directory contents
        dir_contents = os.listdir(image_dir)
        print(f"Directory contains {len(dir_contents)} items:")

        tile_files = []
        yaml_file = None

        for item in dir_contents:
            item_path = os.path.join(image_dir, item)
            if item.endswith('.yaml'):
                yaml_file = item_path
                print(f"  YAML: {item}")
            elif item.startswith('tile_') and (item.endswith('.png') or item.endswith('.jpg')):
                tile_files.append(item_path)
                if len(tile_files) <= 3:  # Show first 3 tiles
                    print(f"  TILE: {item}")

        if len(tile_files) > 3:
            print(f"  ... and {len(tile_files) - 3} more tile files")

        # Load tile metadata
        yaml_metadata_path = os.path.join(image_dir, "tiles_metadata.yaml")
        if not os.path.exists(yaml_metadata_path):
            print(f"Error: Metadata file not found: {yaml_metadata_path}")
            continue

        try:
            # Load YAML metadata to get source image info
            with open(yaml_metadata_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            source_image = yaml_data["source_image"]

            # Load tile metadata
            tile_metadata_list = SahiSlicer.load_metadata(yaml_metadata_path)
            print(f"Loaded metadata for {len(tile_metadata_list)} tiles")

            print(f"Found {len(tile_files)} tile images")

            # Create mapping from tile_id to metadata
            tile_metadata_map = {}
            for metadata in tile_metadata_list:
                tile_id = metadata.tile_id
                tile_metadata_map[tile_id] = metadata

            print(f"Created metadata mapping for {len(tile_metadata_map)} tiles")
            print("Sample tile IDs from metadata:")
            for i, tile_id in enumerate(list(tile_metadata_map.keys())[:3]):
                print(f"  {i + 1}. {tile_id}")

            # Process each tile with OCR
            total_detections = 0
            processed_tiles = 0
            rotation_summary = {angle: 0 for angle in rotation_angles}

            for i, tile_file in enumerate(tile_files):
                tile_filename = os.path.basename(tile_file)

                # Extract tile ID from filename
                tile_id = extract_tile_id_from_filename(tile_filename)

                print(f"Processing tile {i + 1}/{len(tile_files)}: {tile_filename}")
                print(f"  Extracted tile ID: {tile_id}")

                # Find corresponding metadata
                if tile_id not in tile_metadata_map:
                    print(f"  Warning: No metadata found for tile ID: {tile_id}")
                    print(f"  Available tile IDs: {list(tile_metadata_map.keys())[:5]}...")
                    continue

                tile_metadata = tile_metadata_map[tile_id]

                try:
                    # Perform OCR on the tile with rotation support
                    if use_rotation:
                        ocr_detections = ocr_detector.detect_text_with_rotation(
                            tile_file,
                            confidence_threshold=confidence_threshold,
                            rotation_angles=rotation_angles
                        )

                        # Filter overlapping detections if requested
                        if filter_overlaps and len(ocr_detections) > 1:
                            original_count = len(ocr_detections)
                            ocr_detections = ocr_detector.filter_overlapping_detections(ocr_detections)
                            if original_count != len(ocr_detections):
                                print(f"  Filtered {original_count - len(ocr_detections)} overlapping detections")
                    else:
                        ocr_detections = ocr_detector.detect_text(
                            tile_file,
                            confidence_threshold=confidence_threshold,
                            use_rotation=False
                        )

                    print(f"  Found {len(ocr_detections)} text detections")

                    if len(ocr_detections) > 0:
                        # Show rotation statistics
                        rotation_stats = {}
                        for det in ocr_detections:
                            angle = det.rotation_angle
                            rotation_stats[angle] = rotation_stats.get(angle, 0) + 1
                            rotation_summary[angle] = rotation_summary.get(angle, 0) + 1

                        for j, detection in enumerate(ocr_detections[:3]):  # Show first 3 detections
                            print(
                                f"    {j + 1}. '{detection.text}' (conf: {detection.confidence:.3f}, rot: {detection.rotation_angle}°)")
                        if len(ocr_detections) > 3:
                            print(f"    ... and {len(ocr_detections) - 3} more detections")

                        if rotation_stats and use_rotation:
                            rotation_info = ", ".join([f"{angle}°: {count}" for angle, count in rotation_stats.items()])
                            print(f"    Rotation breakdown: {rotation_info}")

                    total_detections += len(ocr_detections)

                    # Save OCR metadata
                    metadata_manager.save_ocr_metadata(
                        ocr_detections,
                        tile_file,
                        tile_metadata
                    )

                    processed_tiles += 1

                except Exception as e:
                    print(f"  Error processing {tile_filename}: {e}")
                    continue

            print(f"\nCompleted processing {image_name}:")
            print(f"  - Processed tiles: {processed_tiles}/{len(tile_files)}")
            print(f"  - Total text detections: {total_detections}")

            if use_rotation and total_detections > 0:
                print(f"  - Rotation summary:")
                for angle, count in rotation_summary.items():
                    if count > 0:
                        percentage = (count / total_detections) * 100
                        print(f"    {angle}°: {count} detections ({percentage:.1f}%)")

            # Generate consolidated metadata report
            try:
                consolidated = metadata_manager.consolidate_metadata_with_ocr(source_image)

                # Save consolidated report
                base_name = os.path.splitext(os.path.basename(source_image))[0]
                report_path = os.path.join(metadata_output_dir, f"{base_name}_consolidated_report.json")

                import json
                with open(report_path, 'w') as f:
                    json.dump(consolidated, f, indent=2)

                print(f"  - Saved consolidated report: {report_path}")

                # Print summary
                total_ocr_detections = sum(len(dets) for dets in consolidated.get("ocr_detections", {}).values())
                print(f"  - Total OCR detections in consolidated report: {total_ocr_detections}")

            except Exception as e:
                print(f"  Error creating consolidated report: {e}")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print("OCR processing completed!")
    print(f"Results saved in: {metadata_output_dir}/detection_metadata/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Configuration
    config = {
        "tiles_base_dir": "data/processed/tiles",
        "metadata_output_dir": "data/processed/metadata",
        "languages": ['en'],  # Add more languages as needed: ['en', 'ch_sim', 'th', 'ja', 'ko']
        "confidence_threshold": 0.1,
        "gpu": True,
        "use_rotation": True,  # Enable rotation detection
        "filter_overlaps": True,  # Filter overlapping detections from different rotations
        "rotation_angles": [0, 90]  # Try these rotation angles
    }

    process_tiles_with_ocr(**config)
