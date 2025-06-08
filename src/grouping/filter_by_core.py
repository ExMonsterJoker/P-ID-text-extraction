# In src/grouping/filter_by_core.py

import os
import json
import logging
from glob import glob
import numpy as np # Make sure to import numpy

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Fixed paths based on project structure
CORE_META_DIR = "data/processed/metadata/core_tile_metadata"
DET_META_DIR = "data/processed/metadata/detection_metadata"
OUT_DIR = "data/processed/metadata/core_detection_metadata"

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
            tile_map = { tile['tile_id']: tile['core_coordinates'] for tile in meta['core_tiles'] }
            core_map[base] = tile_map
        except Exception as e:
            logging.error(f"Failed to process {fn}: {e}")
    return core_map


def filter_tile_detections(base, tile_id, core_coords, det_meta_dir, min_area_ratio=0.5):
    """
    Load OCR detections for a tile and filter them based on their overlap with the core region.
    A detection is kept if its intersection with the core region is >= min_area_ratio of its total area.
    """
    fname = f"{base}_{tile_id}_ocr.json"
    path = os.path.join(det_meta_dir, fname)
    if not os.path.exists(path):
        logging.warning(f"Missing detection file: {fname}")
        return []
    try:
        with open(path, 'r') as f:
            detections = json.load(f)
    except Exception as e:
        logging.error(f"Error reading {fname}: {e}")
        return []

    kept = []
    for det in detections:
        try:
            tile_coords = det.get('tile_coordinates', [0, 0, 0, 0])
            bbox_tile = det.get('bbox', [])  # Bbox relative to the tile
            if len(bbox_tile) != 4:
                continue

            # Bbox in global image coordinates
            bbox_global = [[p[0] + tile_coords[0], p[1] + tile_coords[1]] for p in bbox_tile]

            # Calculate detection area (using shoelace formula for polygon area)
            x, y = zip(*bbox_global)
            detection_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            if detection_area == 0:
                continue

            # Calculate intersection area with the core tile region
            intersection_area = get_intersection_area(bbox_global, core_coords)

            # Keep the detection if it has sufficient overlap with the core region
            if (intersection_area / detection_area) > min_area_ratio:
                det['bbox_original'] = bbox_global  # Add the global coordinates for the next step
                kept.append(det)
        except Exception as e:
            logging.warning(f"Error processing detection in {fname}: {e}")
            continue

    return kept


def get_intersection_area(box, core_coords):
    """Calculates the area of intersection between a detection box and a core region."""
    # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # core_coords is (x0, y0, x1, y1)

    # Convert box to a simple min/max rectangle for intersection calculation
    min_x = min(p[0] for p in box)
    max_x = max(p[0] for p in box)
    min_y = min(p[1] for p in box)
    max_y = max(p[1] for p in box)

    core_x0, core_y0, core_x1, core_y1 = core_coords

    # Find the intersection rectangle
    inter_x0 = max(min_x, core_x0)
    inter_y0 = max(min_y, core_y0)
    inter_x1 = min(max_x, core_x1)
    inter_y1 = min(max_y, core_y1)

    # Calculate intersection area
    inter_width = max(0, inter_x1 - inter_x0)
    inter_height = max(0, inter_y1 - inter_y0)

    return inter_width * inter_height


def main():
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
