import cv2
import os
import logging
from typing import List, Dict


def crop_image(image_path: str, detections: List[Dict], output_dir: str, padding: int) -> List[Dict]:
    """
    Crops text regions from an image based on bounding box detections and
    returns a manifest list containing metadata about each created crop,
    including the text orientation.

    Args:
        image_path: Path to the source image.
        detections: A list of detection dictionaries from the grouping step.
        output_dir: The directory to save the cropped images.
        padding: The number of pixels to add as padding around the crop.

    Returns:
        A list of manifest entries, where each entry maps a crop filename
        to its original bounding box, text, and orientation.
    """
    manifest_data = []
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return []
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return []

    h, w, _ = image.shape

    for i, det in enumerate(detections):
        try:
            bbox_original_global = det['bbox']
            original_text = det.get('text', 'unknown')

            # Get bounding box coordinates and apply padding
            x_min, y_min = bbox_original_global[0]
            x_max, y_max = bbox_original_global[2]

            x_min_pad = max(0, x_min - padding)
            y_min_pad = max(0, y_min - padding)
            x_max_pad = min(w, x_max + padding)
            y_max_pad = min(h, y_max + padding)

            cropped_img = image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

            if cropped_img.size == 0:
                logging.warning(f"  - Skipping zero-size crop for text: '{original_text}'")
                continue

            output_filename = f"crop_{i:04d}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_img)

            # --- THIS IS THE CRITICAL PART ---
            # It adds the 'orientation' key to each entry in the manifest.
            manifest_data.append({
                "crop_filename": output_filename,
                "original_bbox": bbox_original_global,
                "original_text": original_text,
                "original_confidence": det.get('confidence'),
                "orientation": det.get('orientation', 0)  # Default to 0 if missing
            })

        except Exception as e:
            logging.error(f"  - Failed to process detection #{i} ('{det.get('text', 'N/A')}'): {e}")

    logging.info(f"  - Successfully created {len(manifest_data)} crops in: {output_dir}")
    return manifest_data
