import os
import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any
from dataclasses import dataclass
import yaml
import logging
import matplotlib.pyplot as plt
from configs import get_config, get_config_value

@dataclass
class text_detection_data_class:
    """Stores data for a single text detection result (detection only, no recognition)."""
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    bbox_normalized: List[List[float]]  # Normalized coordinates (0-1)
    rotation_angle: int = 0  # Angle at which text was detected


class text_detection:
    def __init__(self):
        """
        Initialize the EasyOCR detector.
        Configuration is fetched automatically from the global config manager.
        """
        self.config = get_config('ocr')
        self.languages = self.config.get('languages', ['en'])
        self.gpu = self.config.get('gpu', True)
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """Initialize EasyOCR reader with a fallback to CPU."""
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logging.info(f"EasyOCR initialized with languages: {self.languages}, GPU: {self.gpu}")
        except Exception as e:
            logging.error(f"Error initializing EasyOCR on GPU: {e}")
            if self.gpu:
                logging.info("Falling back to CPU...")
                self.gpu = False
                try:
                    self.reader = easyocr.Reader(self.languages, gpu=False)
                    logging.info("EasyOCR initialized on CPU")
                except Exception as cpu_error:
                    logging.error(f"Failed to initialize EasyOCR on CPU: {cpu_error}")
                    raise

    def detect_text(self, image_path: str) -> List[text_detection_data_class]:
        """
        Detect text regions and return their coordinates and orientation.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        text_threshold = self.config.get('text_threshold', 0.7)
        link_threshold = self.config.get('link_threshold', 0.4)
        low_text = self.config.get('low_text', 0.4)
        height_ths = self.config.get('height_ths', 0.5)
        width_ths = self.config.get('width_ths', 0.5)
        slope_ths = self.config.get('slope_ths', 0.1)
        ycenter_ths = self.config.get('ycenter_ths', 0.5)

        horizontal_lists, free_lists = self.reader.detect(
            image,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            height_ths=height_ths,
            width_ths=width_ths,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths
        )

        all_detections_data = []

        # horizontal_lists is a list of detected items. Process each one.
        for i, horizontal_list in enumerate(horizontal_lists[0]):
            try:
                # Process each bounding box returned by the detector
                x_min, x_max, y_min, y_max = map(int, horizontal_list)

                orientation = self.get_orientation_from_bbox([x_min, x_max, y_min, y_max])
                corner_bbox = self.convert_to_corners([x_min, x_max, y_min, y_max])
                bbox_normalized = [[p[0] / width, p[1] / height] for p in corner_bbox]

                all_detections_data.append(text_detection_data_class(
                    bbox=corner_bbox,
                    bbox_normalized=bbox_normalized,
                    rotation_angle=orientation
                ))
            except Exception as e:
                logging.warning(f"Error processing box {i}: {e}")
                continue

        return all_detections_data

    def get_orientation_from_bbox(self, horizontal_list):
        """
        Determines orientation for an axis-aligned bounding box.
        Returns 90 if taller than wide (vertical), else 0 (horizontal).
        input is list of this format [x_min, x_max, y_min, y_max].
        """
        x_min, x_max, y_min, y_max = horizontal_list
        width = x_max - x_min
        height = y_max - y_min
        return 90 if height > width else 0

    def convert_to_corners(self, horizontal_list):
        """
        Convert a bounding box from [x_min, x_max, y_min, y_max] / horizontal_listformat
        to [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] corner format.

        The output is in clockwise order starting from the top-left corner.
        """
        x_min, x_max, y_min, y_max = horizontal_list

        return [
            [x_min, y_min],  # top-left
            [x_max, y_min],  # top-right
            [x_max, y_max],  # bottom-right
            [x_min, y_max],  # bottom-left
        ]

    def visualize_detections(self, image_path: str, detections: List[text_detection_data_class]) -> np.ndarray:
        """
        Draws bounding boxes on the image to visualize detections.

        Args:
            image_path: Path to the source image.
            detections: A list of detection data objects.

        Returns:
            The image with bounding boxes drawn on it.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image for visualization: {image_path}")

        for det in detections:
            # Bbox is a list of points, so we need to convert it to a NumPy array
            points = np.array(det.bbox, dtype=np.int32)
            # Use cv2.polylines to draw the bounding box
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Config is now loaded automatically by the text_detection class.
    # 1. Initialize the detector
    try:
        detector = text_detection()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # 3. Define the path to the image you want to process
        sample_image_path = os.path.join(
            project_root, "data", "processed", "tiles",
            "DURI-GENF05GS000-PRO-PID-IBU-0011-00",
            "tile_DURI-GENF05GS000-PRO-PID-IBU-0011-00_T0320.png"
        )

        if os.path.exists(sample_image_path):
            logging.info(f"Running CRAFT detection on: {sample_image_path}")

            # 4. Get the list of detection objects
            detections = detector.detect_text(sample_image_path)
            logging.info(f"Found {len(detections)} text regions using CRAFT.")

            # Print detection details
            for i, det in enumerate(detections):
                orientation_str = "vertical" if det.rotation_angle == 90 else "horizontal"
                logging.info(f"Detection {i + 1}: {orientation_str} text region")

            # 5. Visualize results
            if detections:
                output_image = detector.visualize_detections(sample_image_path, detections)

                # Display the image
                plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                plt.title("CRAFT Text Detection")
                plt.axis("off")
                plt.show()
            else:
                logging.info("No text regions detected by CRAFT.")

        else:
            logging.warning(f"Test image not found at: {sample_image_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()