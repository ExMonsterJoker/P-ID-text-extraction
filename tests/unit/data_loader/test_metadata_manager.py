import os
import tempfile
import json
import pytest
from unittest.mock import patch, MagicMock
from src.data_loader.metadata_manager import MetadataManager, GlobalMetadata
from src.data_loader.sahi_slicer import TileMetadata
from glob import glob

import pytest

@pytest.fixture(scope="module")
def real_image_path():
    candidates = glob(os.path.join("data", "raw", "*.jpg")) + \
                 glob(os.path.join("data", "raw", "*.png"))
    if not candidates:
        pytest.skip("No image found in data/raw for testing.")
    return os.path.abspath(candidates[0])

def make_tile_metadata(tile_id="T0001", source_image=None):
    if source_image is None:
        source_image = real_image_path()
    return TileMetadata(
        tile_id=tile_id,
        source_image=source_image,
        coordinates=(0, 0, 256, 256),
        tile_size=(256, 256),
        grid_position=(0, 0),
        overlap=0.2,
        original_image_size=(2048, 2048)
    )

def test_metadata_manager_init_creates_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MetadataManager(tmpdir, pipeline_version="2.0")
        assert os.path.isdir(mgr.tile_metadata_dir)
        assert os.path.isdir(mgr.global_metadata_dir)
        assert os.path.isdir(mgr.detection_metadata_dir)
        assert mgr.pipeline_version == "2.0"

def test_save_and_load_tile_metadata(real_image_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MetadataManager(tmpdir)
        tiles = [make_tile_metadata(tile_id="T0001", source_image=real_image_path)]
        mgr.save_tile_metadata(tiles, real_image_path)
        loaded = mgr.load_tile_metadata(real_image_path)
        assert len(loaded) == 1
        assert loaded[0].tile_id == "T0001"
        assert loaded[0].source_image == real_image_path

def test_init_and_save_global_metadata(real_image_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MetadataManager(tmpdir, pipeline_version="3.1")
        config = {"a": 1, "b": 2}
        meta = mgr.init_global_metadata(real_image_path, config)
        assert isinstance(meta, GlobalMetadata)
        assert meta.source_image == real_image_path
        assert meta.pipeline_version == "3.1"
        mgr.save_global_metadata(meta)
        # Check file exists and content
        base = os.path.splitext(os.path.basename(real_image_path))[0]
        path = os.path.join(mgr.global_metadata_dir, f"{base}_global.json")
        with open(path) as f:
            data = json.load(f)
        assert data["source_image"] == real_image_path
        assert data["pipeline_version"] == "3.1"

def test_save_and_load_detection_metadata(real_image_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MetadataManager(tmpdir)
        tile = make_tile_metadata(tile_id="T0002", source_image=real_image_path)
        detections = [{"label": "foo", "score": 0.9}]
        mgr.save_detection_metadata(detections, tile)
        loaded = mgr.load_detection_metadata(real_image_path)
        assert "T0002" in loaded
        assert loaded["T0002"][0]["label"] == "foo"
        assert loaded["T0002"][0]["tile_id"] == "T0002"
        assert loaded["T0002"][0]["source_image"] == real_image_path

def test_consolidate_metadata(real_image_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = MetadataManager(tmpdir)
        # Save global
        config = {"x": 1}
        global_meta = mgr.init_global_metadata(real_image_path, config)
        mgr.save_global_metadata(global_meta)
        # Save tile
        tiles = [make_tile_metadata(tile_id="T0003", source_image=real_image_path)]
        mgr.save_tile_metadata(tiles, real_image_path)
        # Save detection
        mgr.save_detection_metadata([{"label": "bar"}], tiles[0])
        # Consolidate
        consolidated = mgr.consolidate_metadata(real_image_path)
        assert "global" in consolidated
        assert "tiles" in consolidated
        assert "detections" in consolidated
        assert consolidated["tiles"][0]["tile_id"] == "T0003"
        assert "T0003" in consolidated["detections"]
