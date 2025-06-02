import os
import json
import cv2
import logging
from glob import glob
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Paths
CORE_DET_DIR = "data/processed/metadata/core_detection_metadata"
RAW_IMG_DIR = "data/raw"
OUT_IMG_DIR = "data/outputs/visualizations"

# Create output directory if it doesn't exist
os.makedirs(OUT_IMG_DIR, exist_ok=True)

def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    pts = [(int(x), int(y)) for x, y in points]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

def draw_label(img, text, position, font_scale=0.4, color=(255, 0, 0), thickness=1):
    x, y = position
    cv2.putText(img, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def visualize_detections(
    show_text=True,
    show_confidence=True,
    show_tile_id=True
):
    detection_files = glob(os.path.join(CORE_DET_DIR, '*_ocr_core_detections.json'))
    for det_path in detection_files:
        try:
            base_name = os.path.basename(det_path).replace('_ocr_core_detections.json', '')
            image_path = os.path.join(RAW_IMG_DIR, f"{base_name}.jpg")
            if not os.path.exists(image_path):
                logging.warning(f"Missing image for {base_name}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue

            with open(det_path, 'r') as f:
                detections = json.load(f)

            for det in detections:
                quad = det.get('bbox_original', [])
                if len(quad) == 4:
                    draw_polygon(image, quad)

                    label_parts = []
                    if show_text and 'text' in det:
                        label_parts.append(det['text'])
                    if show_confidence and 'confidence' in det:
                        label_parts.append(f"{det['confidence']:.2f}")
                    if show_tile_id and 'tile_id' in det:
                        tile_id_short = det['tile_id'].split('_')[-1]  # Get TXXXX only
                        label_parts.append(tile_id_short)

                    if label_parts:
                        label = ' | '.join(label_parts)
                        draw_label(image, label, quad[0])

            out_path = os.path.join(OUT_IMG_DIR, f"{base_name}_ocr_overlay.jpg")
            cv2.imwrite(out_path, image)
            logging.info(f"Saved overlay: {out_path}")

        except Exception as e:
            logging.error(f"Error processing {det_path}: {e}")

if __name__ == '__main__':
    visualize_detections(
        show_text=True,
        show_confidence=True,
        show_tile_id=True
    )
