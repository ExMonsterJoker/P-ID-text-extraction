#src/data_loader/metadata_manager.py
import os
import json
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional, Tuple

# Handle imports for both package and direct script execution
try:
    from .sahi_slicer import TileMetadata
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.data_loader.sahi_slicer import TileMetadata

@dataclass
class CoreTileMetadata:
    """Core tile metadata with non-overlapping regions"""
    tile_id: str
    original_coordinates: Tuple[int, int, int, int]  # [x0, y0, x1, y1]
    core_coordinates: Tuple[int, int, int, int]      # [x0, y0, core_x1, core_y1]
    grid_position: Tuple[int, int]                   # [row, col]
    has_right_neighbor: bool
    has_bottom_neighbor: bool
    overlap_width: int
    overlap_height: int

@dataclass
class OCRDetection:
    """OCR detection result"""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    bbox_normalized: List[List[float]]  # Normalized coordinates (0-1)

@dataclass
class GlobalMetadata:
    """Global metadata for entire pipeline processing"""
    source_image: str
    processing_time: str
    pipeline_version: str
    config_hash: str

class MetadataManager:
    def __init__(self, output_dir: str, pipeline_version: str = "1.0"):
        """
        Initialize metadata manager

        Args:
            output_dir: Base directory for saving metadata
            pipeline_version: Version identifier for the pipeline
        """
        self.output_dir = output_dir
        self.pipeline_version = pipeline_version
        os.makedirs(output_dir, exist_ok=True)

        # Create metadata storage directories
        self.tile_metadata_dir = os.path.join(output_dir, "tile_metadata")
        self.core_tile_metadata_dir = os.path.join(output_dir, "core_tile_metadata")
        self.global_metadata_dir = os.path.join(output_dir, "global_metadata")
        self.detection_metadata_dir = os.path.join(output_dir, "detection_metadata")

        for dir_path in [self.tile_metadata_dir, self.global_metadata_dir, self.detection_metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def compute_core_tiles(self, tile_metadata_list: List[TileMetadata]) -> List[CoreTileMetadata]:
        """
        Compute core (non-overlapping) regions for each tile

        Args:
            tile_metadata_list: List of TileMetadata objects

        Returns:
            List of CoreTileMetadata objects with core coordinates
        """
        # Create a lookup dictionary for tiles by grid position
        tile_grid = {}
        for tile in tile_metadata_list:
            row, col = tile.grid_position
            tile_grid[(row, col)] = tile

        core_tiles = []

        for tile in tile_metadata_list:
            row, col = tile.grid_position
            x0, y0, x1, y1 = tile.coordinates

            # Initialize core coordinates (left and top edges never change)
            core_x0, core_y0 = x0, y0
            core_x1, core_y1 = x1, y1

            # Check for right neighbor
            right_neighbor = tile_grid.get((row, col + 1))
            has_right_neighbor = right_neighbor is not None
            overlap_width = 0

            if has_right_neighbor:
                x0_r = right_neighbor.coordinates[0]  # Right neighbor's x_min
                overlap_width = x1 - x0_r
                core_x1 = x0_r  # Remove overlap by setting core_x1 = right_neighbor.x_min

            # Check for bottom neighbor
            bottom_neighbor = tile_grid.get((row + 1, col))
            has_bottom_neighbor = bottom_neighbor is not None
            overlap_height = 0

            if has_bottom_neighbor:
                y0_b = bottom_neighbor.coordinates[1]  # Bottom neighbor's y_min
                overlap_height = y1 - y0_b
                core_y1 = y0_b  # Remove overlap by setting core_y1 = bottom_neighbor.y_min

            # Create core tile metadata
            core_tile = CoreTileMetadata(
                tile_id=tile.tile_id,
                original_coordinates=tile.coordinates,
                core_coordinates=(core_x0, core_y0, core_x1, core_y1),
                grid_position=tile.grid_position,
                has_right_neighbor=has_right_neighbor,
                has_bottom_neighbor=has_bottom_neighbor,
                overlap_width=overlap_width,
                overlap_height=overlap_height
            )

            core_tiles.append(core_tile)

        return core_tiles

    def save_core_tile_metadata(self, core_tiles: List[CoreTileMetadata], source_image: str):
        """
        Save core tile metadata to JSON file

        Args:
            core_tiles: List of CoreTileMetadata objects
            source_image: Path to source image
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        output_path = os.path.join(self.core_tile_metadata_dir, f"{base_name}_core_tiles.json")

        # Convert to serializable format
        core_data = []
        for core_tile in core_tiles:
            core_dict = asdict(core_tile)
            core_dict["source_image"] = source_image
            core_data.append(core_dict)

        with open(output_path, 'w') as f:
            json.dump({
                "source_image": source_image,
                "core_tiles": core_data
            }, f, indent=2)

    def load_core_tile_metadata(self, source_image: str) -> List[CoreTileMetadata]:
        """
        Load core tile metadata from JSON file

        Args:
            source_image: Path to source image

        Returns:
            List of CoreTileMetadata objects
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        input_path = os.path.join(self.tile_metadata_dir, f"{base_name}_core_tiles.json")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No core tile metadata found for {source_image}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        return [CoreTileMetadata(**core_data) for core_data in data["core_tiles"]]

    def save_ocr_metadata(self, ocr_detections: List['OCRDetection'], tile_path: str, tile_metadata: 'TileMetadata'):
        """
        Save OCR detection metadata for a tile

        Args:
            ocr_detections: List of OCRDetection objects
            tile_path: Path to the tile image
            tile_metadata: TileMetadata for this tile
        """
        base_name = os.path.splitext(os.path.basename(tile_metadata.source_image))[0]
        output_path = os.path.join(
            self.detection_metadata_dir,
            f"{base_name}_{tile_metadata.tile_id}_ocr.json"
        )

        # Convert OCR detections to serializable format
        ocr_data = []
        for detection in ocr_detections:
            det_dict = asdict(detection)
            det_dict.update({
                "tile_id": tile_metadata.tile_id,
                "tile_path": tile_path,
                "source_image": tile_metadata.source_image,
                "detection_type": "ocr",
                "tile_coordinates": tile_metadata.coordinates,  # [x1, y1, x2, y2]
                "tile_size": tile_metadata.tile_size,
                "grid_position": tile_metadata.grid_position,
                "original_image_size": tile_metadata.original_image_size
            })
            ocr_data.append(det_dict)

        with open(output_path, 'w') as f:
            json.dump(ocr_data, f, indent=2)

    def load_ocr_metadata(self, source_image: str) -> Dict[str, List[Dict]]:
        """
        Load all OCR metadata for an image

        Args:
            source_image: Path to source image

        Returns:
            Dictionary of OCR detections keyed by tile ID
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        ocr_detections = {}

        for file_name in os.listdir(self.detection_metadata_dir):
            if file_name.startswith(base_name) and file_name.endswith("_ocr.json"):
                tile_id = file_name.split('_')[-2]  # Extract tile ID
                with open(os.path.join(self.detection_metadata_dir, file_name), 'r') as f:
                    ocr_detections[tile_id] = json.load(f)

        return ocr_detections

    def consolidate_metadata_with_ocr(self, source_image: str) -> Dict[str, Any]:
        """
        Consolidate all metadata including OCR for an image

        Args:
            source_image: Path to source image

        Returns:
            Consolidated metadata dictionary with OCR results
        """
        # Get existing consolidated metadata
        consolidated = self.consolidate_metadata(source_image)

        # Add OCR detections
        ocr_detections = self.load_ocr_metadata(source_image)
        consolidated["ocr_detections"] = ocr_detections

        return consolidated

    def save_tile_metadata(self, metadata_list: List[TileMetadata], source_image: str):
        """
        Save tile metadata to JSON file

        Args:
            metadata_list: List of TileMetadata objects
            source_image: Path to source image
        """
        # Create filename based on source image
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        output_path = os.path.join(self.tile_metadata_dir, f"{base_name}_tiles.json")

        # Convert to serializable format
        metadata_dicts = [asdict(m) for m in metadata_list]

        with open(output_path, 'w') as f:
            json.dump({
                "source_image": source_image,
                "tiles": metadata_dicts
            }, f, indent=2)

    def load_tile_metadata(self, source_image: str) -> List[TileMetadata]:
        """
        Load tile metadata from JSON file

        Args:
            source_image: Path to source image

        Returns:
            List of TileMetadata objects
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        input_path = os.path.join(self.tile_metadata_dir, f"{base_name}_tiles.json")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No tile metadata found for {source_image}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        return [TileMetadata(**t) for t in data["tiles"]]

    def init_global_metadata(self, source_image: str, config: Dict[str, Any]) -> GlobalMetadata:
        """
        Initialize global metadata for a processing job

        Args:
            source_image: Path to source image
            config: Pipeline configuration dictionary

        Returns:
            GlobalMetadata object
        """
        import datetime
        from hashlib import md5

        # Create config hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = md5(config_str.encode()).hexdigest()[:8]

        return GlobalMetadata(
            source_image=source_image,
            processing_time=datetime.datetime.now().isoformat(),
            pipeline_version=self.pipeline_version,
            config_hash=config_hash
        )

    def save_global_metadata(self, metadata: GlobalMetadata):
        """
        Save global metadata to JSON file

        Args:
            metadata: GlobalMetadata object
        """
        base_name = os.path.splitext(os.path.basename(metadata.source_image))[0]
        output_path = os.path.join(self.global_metadata_dir, f"{base_name}_global.json")

        with open(output_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

    def save_detection_metadata(self, detections: List[Dict], tile_metadata: TileMetadata):
        """
        Save detection metadata for a tile

        Args:
            detections: List of detection dictionaries
            tile_metadata: TileMetadata for this tile
        """
        base_name = os.path.splitext(os.path.basename(tile_metadata.source_image))[0]
        output_path = os.path.join(
            self.detection_metadata_dir,
            f"{base_name}_{tile_metadata.tile_id}_detections.json"
        )

        # Add tile metadata reference
        for det in detections:
            det["tile_id"] = tile_metadata.tile_id
            det["source_image"] = tile_metadata.source_image

        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)

    def load_detection_metadata(self, source_image: str) -> Dict[str, List[Dict]]:
        """
        Load all detection metadata for an image

        Args:
            source_image: Path to source image

        Returns:
            Dictionary of detections keyed by tile ID
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        detections = {}

        for file_name in os.listdir(self.detection_metadata_dir):
            if file_name.startswith(base_name) and file_name.endswith("_detections.json"):
                tile_id = file_name.split('_')[-2]
                with open(os.path.join(self.detection_metadata_dir, file_name), 'r') as f:
                    detections[tile_id] = json.load(f)

        return detections

    def consolidate_metadata(self, source_image: str) -> Dict[str, Any]:
        """
        Consolidate all metadata for an image into a single structure

        Args:
            source_image: Path to source image

        Returns:
            Consolidated metadata dictionary
        """
        base_name = os.path.splitext(os.path.basename(source_image))[0]

        # Load global metadata
        global_path = os.path.join(self.global_metadata_dir, f"{base_name}_global.json")
        with open(global_path, 'r') as f:
            global_meta = json.load(f)

        # Load tile metadata
        tile_meta = self.load_tile_metadata(source_image)

        # Load detections
        detections = self.load_detection_metadata(source_image)

        return {
            "global": global_meta,
            "tiles": [asdict(t) for t in tile_meta],
            "detections": detections
        }

if __name__ == "__main__":
    import glob
    import yaml
    import sys
    import os
    from typing import List  # Import List for type hinting

    # Add the project root to the Python path to enable absolute imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.data_loader.sahi_slicer import SahiSlicer
    from src.data_loader.metadata_manager import MetadataManager  # Ensure MetadataManager is imported
    import datetime

    # Define the base directory where sahi_slicer.py saved the tiles
    base_processed_dir = os.path.join("data", "processed", "tiles")

    # Find all image directories in the processed folder
    image_dirs = [d for d in glob.glob(os.path.join(base_processed_dir, "*")) if os.path.isdir(d)]

    if not image_dirs:
        print(f"Error: No image directories found in {base_processed_dir}")
    else:
        print(f"Found {len(image_dirs)} image directories to process")

        # Create a metadata manager for our output
        output_dir = os.path.join("data", "processed", "metadata")
        metadata_manager = MetadataManager(output_dir, pipeline_version="1.0")

        for image_dir in image_dirs:
            image_name = os.path.basename(image_dir)
            print(f"\nProcessing metadata for image: {image_name}")

            # Path to the YAML metadata file created by sahi_slicer.py
            yaml_metadata_path = os.path.join(image_dir, "tiles_metadata.yaml")

            if not os.path.exists(yaml_metadata_path):
                print(f"Error: Metadata file not found at {yaml_metadata_path}")
                continue

            try:
                # Load the YAML metadata
                print(f"Loading metadata from {yaml_metadata_path}")
                with open(yaml_metadata_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)

                # Get the source image path
                source_image = yaml_data["source_image"]
                print(f"Source image: {source_image}")

                # Load the tile metadata using SahiSlicer
                tile_metadata = SahiSlicer.load_metadata(yaml_metadata_path)
                print(f"Loaded {len(tile_metadata)} tile metadata entries")

                # Save the tile metadata using MetadataManager
                metadata_manager.save_tile_metadata(tile_metadata, source_image)
                print(f"Saved tile metadata to {metadata_manager.core_tile_metadata_dir}")

                # Create and save global metadata
                config = {
                    "tile_size": yaml_data["tile_size"],
                    "overlap_ratio": yaml_data["overlap_ratio"],
                    "processing_time": datetime.datetime.now().isoformat()
                }

                global_metadata = metadata_manager.init_global_metadata(source_image, config)
                metadata_manager.save_global_metadata(global_metadata)
                print(f"Saved global metadata to {metadata_manager.global_metadata_dir}")

                # Compute core tiles
                core_tiles = metadata_manager.compute_core_tiles(tile_metadata)
                print(f"Computed core regions for {len(core_tiles)} tiles")

                # Save core tile metadata
                metadata_manager.save_core_tile_metadata(core_tiles, source_image)
                print(f"Saved core tile metadata to {metadata_manager.core_tile_metadata_dir}")

                # Demonstrate consolidate_metadata
                try:
                    consolidated = metadata_manager.consolidate_metadata(source_image)
                    print("\nConsolidated metadata summary:")
                    print(f"- Global metadata: {consolidated['global']['processing_time']}")
                    print(f"- Number of tiles: {len(consolidated['tiles'])}")
                    print(f"- Number of detections: {sum(len(dets) for dets in consolidated['detections'].values())}")
                except Exception as e:
                    print(f"Error consolidating metadata: {str(e)}")

            except Exception as e:
                print(f"Error processing metadata for {image_name}: {str(e)}")
                continue

        print("\nMetadata processing complete!")

