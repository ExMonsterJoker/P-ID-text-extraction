# src/data_loader/metadata_manager.py
import os
import json
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

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
    core_coordinates: Tuple[int, int, int, int]  # [x0, y0, core_x1, core_y1]
    grid_position: Tuple[int, int]  # [row, col]
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

        for dir_path in [self.tile_metadata_dir, self.global_metadata_dir, self.detection_metadata_dir,
                         self.core_tile_metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def compute_core_tiles(self, tile_metadata_list: List[TileMetadata]) -> List[CoreTileMetadata]:
        """
        Compute core (non-overlapping) regions for each tile
        """
        tile_grid = {}
        for tile in tile_metadata_list:
            row, col = tile.grid_position
            tile_grid[(row, col)] = tile

        core_tiles = []
        for tile in tile_metadata_list:
            row, col = tile.grid_position
            x0, y0, x1, y1 = tile.coordinates
            core_x0, core_y0, core_x1, core_y1 = x0, y0, x1, y1

            right_neighbor = tile_grid.get((row, col + 1))
            overlap_width = x1 - right_neighbor.coordinates[0] if right_neighbor else 0
            if right_neighbor: core_x1 = right_neighbor.coordinates[0]

            bottom_neighbor = tile_grid.get((row + 1, col))
            overlap_height = y1 - bottom_neighbor.coordinates[1] if bottom_neighbor else 0
            if bottom_neighbor: core_y1 = bottom_neighbor.coordinates[1]

            core_tiles.append(CoreTileMetadata(
                tile_id=tile.tile_id,
                original_coordinates=tile.coordinates,
                core_coordinates=(core_x0, core_y0, core_x1, core_y1),
                grid_position=tile.grid_position,
                has_right_neighbor=right_neighbor is not None,
                has_bottom_neighbor=bottom_neighbor is not None,
                overlap_width=overlap_width,
                overlap_height=overlap_height
            ))
        return core_tiles

    def save_core_tile_metadata(self, core_tiles: List[CoreTileMetadata], source_image: str):
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        output_path = os.path.join(self.core_tile_metadata_dir, f"{base_name}_core_tiles.json")
        core_data = [asdict(ct) for ct in core_tiles]
        with open(output_path, 'w') as f:
            json.dump({"source_image": source_image, "core_tiles": core_data}, f, indent=2)

    def save_ocr_metadata(self, ocr_detections: List['OCRDetection'], tile_path: str, tile_metadata: 'TileMetadata'):
        """
        Save OCR detection metadata for a tile.
        """
        # FIX: The tile_id is already globally unique (e.g., 'DURI..._T0000').
        # We should not prepend the base_name again.
        #
        # Old line: f"{base_name}_{tile_metadata.tile_id}_ocr.json"
        #
        output_path = os.path.join(
            self.detection_metadata_dir,
            f"{tile_metadata.tile_id}_ocr.json"
        )

        ocr_data = []
        for detection in ocr_detections:
            det_dict = asdict(detection)
            det_dict.update({
                "tile_id": tile_metadata.tile_id,
                "tile_path": tile_path,
                "source_image": tile_metadata.source_image,
                "detection_type": "ocr",
                "tile_coordinates": tile_metadata.coordinates,
                "tile_size": tile_metadata.tile_size,
                "grid_position": tile_metadata.grid_position,
                "original_image_size": tile_metadata.original_image_size
            })
            ocr_data.append(det_dict)

        with open(output_path, 'w') as f:
            json.dump(ocr_data, f, indent=2)

    # --- Other methods remain unchanged ---

    def load_ocr_metadata(self, source_image: str) -> Dict[str, List[Dict]]:
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        ocr_detections = {}
        for file_name in os.listdir(self.detection_metadata_dir):
            if file_name.startswith(base_name) and file_name.endswith("_ocr.json"):
                # This logic should now correctly match the new filenames
                tile_id = file_name.replace("_ocr.json", "")
                with open(os.path.join(self.detection_metadata_dir, file_name), 'r') as f:
                    ocr_detections[tile_id] = json.load(f)
        return ocr_detections

    def save_tile_metadata(self, metadata_list: List[TileMetadata], source_image: str):
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        output_path = os.path.join(self.tile_metadata_dir, f"{base_name}_tiles.json")
        metadata_dicts = [asdict(m) for m in metadata_list]
        with open(output_path, 'w') as f:
            json.dump({"source_image": source_image, "tiles": metadata_dicts}, f, indent=2)

    def load_tile_metadata(self, source_image: str) -> List[TileMetadata]:
        base_name = os.path.splitext(os.path.basename(source_image))[0]
        input_path = os.path.join(self.tile_metadata_dir, f"{base_name}_tiles.json")
        if not os.path.exists(input_path): raise FileNotFoundError(f"No tile metadata for {source_image}")
        with open(input_path, 'r') as f: data = json.load(f)
        return [TileMetadata(**t) for t in data["tiles"]]

    def init_global_metadata(self, source_image: str, config: Dict[str, Any]) -> GlobalMetadata:
        import datetime
        from hashlib import md5
        config_str = json.dumps(config, sort_keys=True)
        config_hash = md5(config_str.encode()).hexdigest()[:8]
        return GlobalMetadata(source_image=source_image, processing_time=datetime.datetime.now().isoformat(),
                              pipeline_version=self.pipeline_version, config_hash=config_hash)

    def save_global_metadata(self, metadata: GlobalMetadata):
        base_name = os.path.splitext(os.path.basename(metadata.source_image))[0]
        output_path = os.path.join(self.global_metadata_dir, f"{base_name}_global.json")
        with open(output_path, 'w') as f: json.dump(asdict(metadata), f, indent=2)

