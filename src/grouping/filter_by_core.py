import os
import json
import logging
from glob import glob

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


def filter_tile_detections(base, tile_id, core_coords, det_meta_dir):
    """
    Load OCR detections for a tile, filter by core, return kept detections list.
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

    x0, y0, x1, y1 = core_coords
    kept = []
    for det in detections:
        try:
            tx0, ty0, _, _ = det.get('tile_coordinates', [0, 0, 0, 0])
            quad = det.get('bbox', [])
            if len(quad) != 4:
                continue
            quad_orig = [[p[0] + tx0, p[1] + ty0] for p in quad]
            cx = sum(p[0] for p in quad_orig) / 4.0
            cy = sum(p[1] for p in quad_orig) / 4.0
            if x0 <= cx < x1 and y0 <= cy < y1:
                det['bbox_original'] = quad_orig
                kept.append(det)
        except Exception as e:
            logging.warning(f"Error processing detection in {fname}: {e}")
            continue
    return kept


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
