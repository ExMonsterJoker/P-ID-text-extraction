# In src/grouping/filter_by_core.py

import os
import json
import logging
from glob import glob
import numpy as np
from shapely.geometry import Polygon, box

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_core_metadata(core_meta_dir):
    """
    Load all core_tiles.json files from core_meta_dir.
    Returns a dict mapping base_image_name -> { tile_id: core_coords }
    """
    core_map = {}
    pattern = os.path.join(core_meta_dir, '*_core_tiles.json')
    for fn in glob(pattern):
        try:
            with open(fn, 'r') as f:
                meta = json.load(f)
            base = os.path.splitext(os.path.basename(meta['source_image']))[0]
            tile_map = {tile['tile_id']: tile['core_coordinates'] for tile in meta['core_tiles']}
            core_map[base] = tile_map
        except Exception as e:
            logging.error(f"Failed to process {fn}: {e}")
    return core_map


def get_intersection_area_shapely(detection_polygon: Polygon, core_coords: tuple) -> float:
    """
    Calculates the precise area of intersection between a detection polygon and a core region using Shapely.
    """
    if not detection_polygon.is_valid:
        return 0.0
    core_box = box(core_coords[0], core_coords[1], core_coords[2], core_coords[3])
    intersection = detection_polygon.intersection(core_box)
    return intersection.area


def filter_tile_detections(base, tile_id, core_coords, det_meta_dir, min_area_ratio=0.1):
    """
    Load OCR detections for a tile and filter them based on their overlap with the core region.
    """
    fname = f"{tile_id}_ocr.json"
    path = os.path.join(det_meta_dir, base, fname) #MODIFIED

    if not os.path.exists(path):
        logging.warning(f"Missing detection file: {path}")
        return []
    try:
        with open(path, 'r') as f:
            detections = json.load(f)
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return []

    kept = []
    for det in detections:
        try:
            tile_coords = det.get('tile_coordinates', [0, 0, 0, 0])
            bbox_tile = det.get('bbox', [])
            if len(bbox_tile) < 3:
                continue

            bbox_global = [[p[0] + tile_coords[0], p[1] + tile_coords[1]] for p in bbox_tile]

            detection_polygon = Polygon(bbox_global)
            detection_area = detection_polygon.area

            if detection_area == 0:
                continue

            intersection_area = get_intersection_area_shapely(detection_polygon, core_coords)

            if (intersection_area / detection_area) > min_area_ratio:
                det['bbox_original'] = bbox_global
                kept.append(det)
        except Exception as e:
            logging.warning(f"Error processing detection in {fname}: {e}")
            continue

    return kept


def main():
    CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
    DET_META_DIR = "data/processed/metadata/detection_metadata"
    OUT_DIR = "data/processed/metadata/core_detection_metadata"
    os.makedirs(OUT_DIR, exist_ok=True)

    core_map = load_core_metadata(CORE_META_DIR)

    for base, tiles in core_map.items():
        all_kept = []
        for tile_id, core_coords in tiles.items():
            kept = filter_tile_detections(base, tile_id, core_coords, DET_META_DIR)
            all_kept.extend(kept)
        out_path = os.path.join(OUT_DIR, f"{base}_ocr_core_detections.json")
        try:
            with open(out_path, 'w') as f:
                json.dump(all_kept, f, indent=2)
            logging.info(f"Written consolidated OCR detections for {base}: {len(all_kept)} total")
        except Exception as e:
            logging.error(f"Failed to write {out_path}: {e}")


if __name__ == '__main__':
    main()