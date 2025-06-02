import os
import tempfile
import yaml
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from glob import glob
from src.data_loader.sahi_slicer import SahiSlicer, TileMetadata

@pytest.fixture(scope="module")
def real_image_path():
    # Find the first image in data/raw (jpg, png, etc.)
    candidates = glob(os.path.join("data", "raw", "*.jpg")) + \
                 glob(os.path.join("data", "raw", "*.png"))
    if not candidates:
        pytest.skip("No image found in data/raw for testing.")
    return os.path.abspath(candidates[0])

def test_tile_metadata_instantiation():
    meta = TileMetadata(
        tile_id="T1",
        source_image="img.png",
        coordinates=(0, 0, 10, 10),
        tile_size=(10, 10),
        grid_position=(0, 0),
        overlap=0.5,
        original_image_size=(100, 100)
    )
    assert meta.tile_id == "T1"
    assert meta.coordinates == (0, 0, 10, 10)

def test_sahi_slicer_init_validation():
    # Valid config
    SahiSlicer({"tile_size": 128, "overlap_ratio": 0.5})
    # Invalid overlap
    with pytest.raises(ValueError):
        SahiSlicer({"tile_size": 128, "overlap_ratio": 1.5})
    # Invalid tile size
    with pytest.raises(ValueError):
        SahiSlicer({"tile_size": 32, "overlap_ratio": 0.5})

def test_save_and_load_metadata_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = TileMetadata(
            tile_id="T1",
            source_image="img.png",
            coordinates=(0, 0, 10, 10),
            tile_size=(10, 10),
            grid_position=(0, 0),
            overlap=0.5,
            original_image_size=(100, 100)
        )
        slicer = SahiSlicer({"tile_size": 10, "overlap_ratio": 0.5})
        out_path = os.path.join(tmpdir, "meta.yaml")
        slicer.save_metadata([meta], out_path)
        loaded = SahiSlicer.load_metadata(out_path)
        assert len(loaded) == 1
        assert loaded[0].tile_id == "T1"
        assert loaded[0].coordinates == (0, 0, 10, 10)

def test_slice_with_real_image(real_image_path):
    slicer = SahiSlicer({"tile_size": 256, "overlap_ratio": 0.2})
    tiles, metadata = slicer.slice(real_image_path)
    assert len(tiles) == len(metadata)
    assert len(tiles) > 0
    assert isinstance(tiles[0], np.ndarray)
    assert hasattr(metadata[0], "tile_id")
    assert metadata[0].source_image == real_image_path
