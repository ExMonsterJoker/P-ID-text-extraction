import cv2
import os
import json
import logging
from glob import glob


def visualize_annotations(image_dir: str, json_dir: str, output_dir: str):
    """
    Draws final annotations from JSON files onto the original images,
    placing text labels intelligently based on text orientation.

    Args:
        image_dir: Directory containing the original source images.
        json_dir: Directory containing the final annotation JSON files.
        output_dir: Directory where the visualized images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(json_dir, '*_final.json'))

    if not json_files:
        logging.warning(f"No final annotation JSON files found in {json_dir}. Skipping visualization.")
        return

    logging.info(f"Found {len(json_files)} annotation files for visualization.")

    for json_file in json_files:
        base_name = os.path.basename(json_file).replace('_final.json', '')

        image_path = next(iter(glob(os.path.join(image_dir, f"{base_name}.*"))), None)
        if not image_path:
            logging.warning(f"Cannot find source image for '{base_name}' in {image_dir}. Skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not load image: {image_path}. Skipping.")
            continue

        img_height, img_width = image.shape[:2]

        with open(json_file, 'r') as f:
            annotations = json.load(f)

        for ann in annotations:
            bbox = ann['bbox']
            text = ann['text']
            orientation = ann.get('orientation', 0)

            # Validate bbox format and extract coordinates
            if len(bbox) != 4 or not all(len(point) == 2 for point in bbox):
                logging.warning(f"Invalid bbox format: {bbox}. Skipping annotation.")
                continue

            # Extract min/max coordinates from the 4 points
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            pt1 = (int(x_min), int(y_min))  # top-left
            pt2 = (int(x_max), int(y_max))  # bottom-right

            # Draw green rectangle for the bounding box
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

            # Text styling
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            padding = 5

            (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

            # Place text label based on orientation
            if orientation == 90:  # Vertical text - place on the side
                # Try to place on the right first, then left if not enough space
                if x_max + text_w + padding * 3 <= img_width:
                    # Place on the right side
                    text_x = x_max + padding * 2
                    text_y = max(y_min + text_h, text_h + padding)

                    bg_pt1 = (text_x - padding, text_y - text_h - padding)
                    bg_pt2 = (text_x + text_w + padding, text_y + baseline + padding)
                else:
                    # Place on the left side
                    text_x = max(x_min - text_w - padding * 2, padding)
                    text_y = max(y_min + text_h, text_h + padding)

                    bg_pt1 = (text_x - padding, text_y - text_h - padding)
                    bg_pt2 = (text_x + text_w + padding, text_y + baseline + padding)

            else:  # Horizontal text - place on top or bottom
                # Try to place above first, then below if not enough space
                if y_min - text_h - padding * 2 >= 0:
                    # Place above the box
                    text_x = x_min
                    text_y = y_min - padding * 2

                    bg_pt1 = (text_x - padding, text_y - text_h - padding)
                    bg_pt2 = (min(text_x + text_w + padding, img_width), text_y + baseline + padding)
                else:
                    # Place below the box
                    text_x = x_min
                    text_y = y_max + text_h + padding * 2

                    bg_pt1 = (text_x - padding, text_y - text_h - padding)
                    bg_pt2 = (min(text_x + text_w + padding, img_width), text_y + baseline + padding)

            # Ensure coordinates are within image boundaries
            text_x = max(0, min(text_x, img_width - text_w))
            text_y = max(text_h, min(text_y, img_height))

            bg_pt1 = (max(0, bg_pt1[0]), max(0, bg_pt1[1]))
            bg_pt2 = (min(img_width, bg_pt2[0]), min(img_height, bg_pt2[1]))

            text_pos = (int(text_x), int(text_y))

            # Draw background rectangle and text
            cv2.rectangle(image, bg_pt1, bg_pt2, (0, 0, 255), -1)
            cv2.putText(image, text, text_pos, font_face, font_scale, (255, 255, 255), thickness)

        output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
        cv2.imwrite(output_path, image)
        logging.info(f"Saved final visualization to: {output_path}")
