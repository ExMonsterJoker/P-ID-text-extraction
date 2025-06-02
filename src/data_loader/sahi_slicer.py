# src/data_loader/sahi_slicer.py

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, cast, Iterable
from dataclasses import dataclass
from sahi.slicing import slice_image
import yaml


@dataclass
class TileMetadata:
    """Metadata container for individual tiles"""
    tile_id: str
    source_image: str
    coordinates: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    tile_size: Tuple[int, int]
    grid_position: Tuple[int, int]  # (row, col)
    overlap: float
    original_image_size: Tuple[int, int]


class SahiSlicer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SAHI-based image slicer

        Args:
            config: Configuration dictionary with slicing parameters
        """
        self.tile_size = config.get("tile_size", 512)
        self.overlap_ratio = config.get("overlap_ratio", 0.5)
        self.min_area_ratio = config.get("min_area_ratio", 0.2)
        self.verbose = config.get("verbose", False)

        # Validate parameters
        if not 0 < self.overlap_ratio < 1:
            raise ValueError("Overlap ratio must be between 0 and 1")
        if self.tile_size < 64:
            raise ValueError("Tile size too small (min 64px)")

    def slice(self, image_path: str) -> Tuple[List[np.ndarray], List[TileMetadata]]:
        """
        Slice input image into overlapping tiles

        Args:
            image_path: Path to input image file

        Returns:
            tiles: List of tile images as numpy arrays
            metadata: List of TileMetadata objects
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate unique identifier for this slicing operation
        # source_id = self._generate_source_id(image_path)

        # Open and validate image
        with Image.open(image_path) as img:
            width, height = img.size
            if width < self.tile_size or height < self.tile_size:
                raise ValueError(
                    f"Image size ({width}x{height}) smaller than tile size ({self.tile_size})"
                )

        # Perform slicing with SAHI
        slice_image_result = slice_image(
            image=image_path,
            output_file_name=os.path.basename(image_path),
            output_dir=None,  # We'll handle output ourselves
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            min_area_ratio=self.min_area_ratio,
            verbose=self.verbose
        )

        # Process results
        tiles = []
        metadata_list = []

        # Debug: Print the first slice_dict to see its structure
        if self.verbose and len(slice_image_result) > 0:
            print("Debug - First slice_dict keys:", slice_image_result[0].keys())

        # Fix for line 86: Cast slice_image_result to Iterable
        for idx, slice_dict in enumerate(cast(Iterable, slice_image_result)):
            # Extract tile image
            tile_img = np.array(slice_dict["image"])

            # Create metadata
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            tile_id = f"{base_name}_T{idx:04d}"

            # Get the width and height of the tile from numpy array shape
            # For numpy arrays, shape is (height, width, channels)
            tile_height, tile_width = slice_dict["image"].shape[:2]

            if self.verbose and idx == 0:
                print(f"Debug - First tile shape: {slice_dict['image'].shape}")

            # Fix for line 164: Cast to Tuple[int, int, int, int]
            coordinates = cast(Tuple[int, int, int, int], (
                slice_dict["starting_pixel"][0],
                slice_dict["starting_pixel"][1],
                slice_dict["starting_pixel"][0] + tile_width,
                slice_dict["starting_pixel"][1] + tile_height
            ))

            # Calculate grid position based on starting_pixel and tile size
            # Effective tile size considering overlap
            effective_width = int(tile_width * (1 - self.overlap_ratio))
            effective_height = int(tile_height * (1 - self.overlap_ratio))

            if effective_width == 0 or effective_height == 0:
                # Fallback to avoid division by zero
                effective_width = max(1, tile_width // 2)
                effective_height = max(1, tile_height // 2)

            # Calculate row and column indices
            row_index = slice_dict["starting_pixel"][1] // effective_height
            col_index = slice_dict["starting_pixel"][0] // effective_width

            # Fix for line 165-166: Cast to Tuple[int, int]
            grid_pos = cast(Tuple[int, int], (row_index, col_index))

            metadata = TileMetadata(
                tile_id=tile_id,
                source_image=image_path,
                coordinates=coordinates,
                # Fix for line 168: Cast to Tuple[int, int]
                tile_size=cast(Tuple[int, int], (tile_width, tile_height)),
                grid_position=grid_pos,
                overlap=self.overlap_ratio,
                original_image_size=cast(Tuple[int, int], (width, height)))

            tiles.append(tile_img)
            metadata_list.append(metadata)

            if self.verbose:
                print(f"Created tile {tile_id} at {coordinates} (size: {metadata.tile_size})")

        return tiles, metadata_list

    """
        def _generate_source_id(self, image_path: str) -> str:
        # Generate unique ID based on image content and slicing parameters
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        config_hash = hashlib.md5(
            f"{self.tile_size}{self.overlap_ratio}".encode()
        ).hexdigest()[:8]

        return f"{os.path.basename(image_path)[:10]}_{file_hash[:6]}_{config_hash}"
    """


    def save_metadata(self, metadata_list: List[TileMetadata], output_path: str):
        """Save metadata to YAML file"""
        metadata_dict = {
            "source_image": metadata_list[0].source_image,
            "original_size": metadata_list[0].original_image_size,
            "tile_size": self.tile_size,
            "overlap_ratio": self.overlap_ratio,
            "num_tiles": len(metadata_list),
            "tiles": [
                {
                    "tile_id": m.tile_id,
                    "coordinates": m.coordinates,
                    "grid_position": m.grid_position,
                    "tile_size": m.tile_size
                }
                for m in metadata_list
            ]
        }

        with open(output_path, 'w') as f:
            yaml.safe_dump(metadata_dict, f, sort_keys=False)

    @staticmethod
    def load_metadata(metadata_path: str) -> List[TileMetadata]:
        """Load metadata from YAML file"""
        with open(metadata_path, 'r') as f:
            data = yaml.safe_load(f)

        metadata_list = []
        for tile_data in data["tiles"]:
            metadata = TileMetadata(
                tile_id=tile_data["tile_id"],
                source_image=data["source_image"],
                coordinates=cast(Tuple[int, int, int, int], tuple(tile_data["coordinates"])),
                tile_size=cast(Tuple[int, int], tuple(tile_data["tile_size"])),
                grid_position=cast(Tuple[int, int], tuple(tile_data["grid_position"])),
                overlap=data["overlap_ratio"],
                original_image_size=cast(Tuple[int, int], tuple(data["original_size"]))
            )
            metadata_list.append(metadata)

        return metadata_list

if __name__ == "__main__":
    import glob

    # Define configuration for the slicer
    config = {
        "tile_size": 1024,
        "overlap_ratio": 0.3,
        "min_area_ratio": 0.1,
        "verbose": True
    }

    # Initialize the slicer
    slicer = SahiSlicer(config)

    # Get all image files from data/raw folder
    raw_dir = os.path.join("data", "raw")
    image_files = glob.glob(os.path.join(raw_dir, "*.jpg")) + glob.glob(os.path.join(raw_dir, "*.png")) + glob.glob(os.path.join(raw_dir, "*.tiff"))

    if not image_files:
        print(f"Error: No image files found in {raw_dir}")
    else:
        print(f"Found {len(image_files)} image(s) to process")

        for image_path in image_files:
            # Extract image filename without extension for creating output directory
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\nProcessing image: {image_path}")

            try:
                # Slice the image
                tiles, metadata = slicer.slice(image_path)

                # Print information about the slicing
                print(f"Image sliced into {len(tiles)} tiles")
                print(f"Original image size: {metadata[0].original_image_size}")
                print(f"Tile size: {metadata[0].tile_size}")

                # Create an output directory specific to this image
                output_dir = os.path.join("data", "processed", "tiles", image_filename)
                os.makedirs(output_dir, exist_ok=True)

                # Save metadata
                metadata_path = os.path.join(output_dir, "tiles_metadata.yaml")
                slicer.save_metadata(metadata, metadata_path)
                print(f"Metadata saved to {metadata_path}")

                # Save ALL tiles
                total_tiles = len(tiles)
                for i in range(total_tiles):
                    tile_img = Image.fromarray(tiles[i])
                    tile_path = os.path.join(output_dir, f"tile_{metadata[i].tile_id}.png")
                    tile_img.save(tile_path)
                    print(f"Saved tile {i + 1}/{total_tiles} to {tile_path}")

            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
