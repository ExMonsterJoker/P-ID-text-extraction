import os
import json
import cv2
import logging
from glob import glob
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Configuration ---
# Point to the NEW output directory from the combined grouping script
FINAL_GROUPED_DIR = "data/processed/metadata/final_grouped_text"
# The original raw images are still our background
RAW_IMG_DIR = "data/raw"
# Save the new visualizations to a different output folder
OUT_IMG_DIR = "data/outputs/final_visualizations"

# Create output directory if it doesn't exist
os.makedirs(OUT_IMG_DIR, exist_ok=True)


def draw_polygon(img, points, color=(0, 255, 0), thickness=2):
    """Draws a polygon on the image."""
    pts = [(int(x), int(y)) for x, y in points]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)


def draw_label(img, text, position, font_scale=0.5, color=(255, 0, 0), thickness=1):
    """Draws a text label above the given position."""
    x, y = position
    # Put text slightly above the top-left corner of the bounding box
    cv2.putText(img, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def visualize_grouped_detections(show_text=True, show_confidence=True):
    """
    Loads the final grouped text data and overlays it on the original images.
    """
    # Look for the new JSON files created by the grouping pipeline
    detection_files = glob(os.path.join(FINAL_GROUPED_DIR, '*_grouped_text.json'))

    if not detection_files:
        logging.warning(f"No grouped detection files found in '{FINAL_GROUPED_DIR}'.")
        logging.warning("Please run the 'run_grouping_pipeline.py' script first.")
        return

    logging.info(f"Found {len(detection_files)} grouped detection files to visualize.")

    for det_path in detection_files:
        try:
            # Extract the base name to find the corresponding raw image
            # e.g., "DURI-OTPF0..._grouped_text.json" -> "DURI-OTPF0..."
            base_name = os.path.basename(det_path).replace('_grouped_text.json', '')

            # Find the matching .jpg or .png image
            image_path_jpg = os.path.join(RAW_IMG_DIR, f"{base_name}.jpg")
            image_path_png = os.path.join(RAW_IMG_DIR, f"{base_name}.png")

            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
            elif os.path.exists(image_path_png):
                image_path = image_path_png
            else:
                logging.warning(f"Missing image for '{base_name}' in '{RAW_IMG_DIR}'")
                continue

            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue

            logging.info(f"Visualizing detections for: {base_name}")

            with open(det_path, 'r') as f:
                grouped_lines = json.load(f)

            # Iterate through the final, merged text lines
            for line in grouped_lines:
                # The bounding box is now the merged 'bbox' for the whole line
                bbox = line.get('bbox', [])
                if len(bbox) == 4:
                    draw_polygon(image, bbox)

                    # Prepare the label with the merged text and average confidence
                    label_parts = []
                    if show_text and 'text' in line:
                        label_parts.append(line['text'])
                    if show_confidence and 'confidence' in line:
                        # Display the average confidence of the group
                        label_parts.append(f"conf: {line['confidence']:.2f}")

                    if label_parts:
                        label = ' | '.join(label_parts)
                        # Use the top-left point of the bbox for the label position
                        draw_label(image, label, bbox[0])

            out_path = os.path.join(OUT_IMG_DIR, f"{base_name}_final_overlay.jpg")
            cv2.imwrite(out_path, image)
            logging.info(f"Saved final overlay to: {out_path}\n")

        except Exception as e:
            logging.error(f"Error processing {det_path}: {e}")


if __name__ == '__main__':
    visualize_grouped_detections(
        show_text=True,
        show_confidence=True
    )