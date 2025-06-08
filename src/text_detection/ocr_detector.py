import os
import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class OCRDetection:
    """OCR detection result"""
    text: str
    confidence: float
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    bbox_normalized: List[List[float]]  # Normalized coordinates (0-1)
    rotation_angle: int = 0  # Angle at which text was detected


class EasyOCRDetector:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize EasyOCR detector

        Args:
            languages: List of language codes (e.g., ['en', 'ch_sim', 'th'])
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages
        self.gpu = gpu
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            print(f"EasyOCR initialized with languages: {self.languages}, GPU: {self.gpu}")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            if self.gpu:
                print("Falling back to CPU...")
                self.gpu = False
                self.reader = easyocr.Reader(self.languages, gpu=False)

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle (0, 90, 180, 270)."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _transform_bbox_after_rotation(self, bbox: List[List[int]], angle: int,
                                       original_shape: Tuple[int, int]) -> List[List[int]]:
        """Transform bounding box coordinates back to original image coordinates after rotation."""
        if angle == 0:
            return bbox

        orig_h, orig_w = original_shape
        transformed_bbox = []

        for point in bbox:
            x, y = point
            if angle == 90:
                new_x, new_y = orig_h - y, x
            elif angle == 180:
                new_x, new_y = orig_w - x, orig_h - y
            elif angle == 270:
                new_x, new_y = y, orig_w - x
            else:
                new_x, new_y = x, y
            transformed_bbox.append([int(new_x), int(new_y)])
        return transformed_bbox

    def detect_text_with_rotation(self, image_path: str, confidence_threshold: float = 0.5,
                                  rotation_angles: List[int] = [0, 90]) -> List[OCRDetection]:
        """Detect text in a single image, trying multiple rotations."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_shape = image.shape[:2]  # (height, width)
        all_detections = []
        height, width = original_shape

        for angle in rotation_angles:
            rotated_image = self._rotate_image(image, angle)
            results = self.reader.readtext(rotated_image)

            for (bbox, text, confidence) in results:
                if confidence < confidence_threshold:
                    continue

                bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
                original_bbox = self._transform_bbox_after_rotation(bbox_int, angle, original_shape)
                bbox_normalized = [[p[0] / width, p[1] / height] for p in original_bbox]

                all_detections.append(OCRDetection(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=original_bbox,
                    bbox_normalized=bbox_normalized,
                    rotation_angle=angle
                ))
        return all_detections

    def filter_overlapping_detections(self, detections: List[OCRDetection],
                                      iou_threshold: float = 0.5) -> List[OCRDetection]:
        """Filter out overlapping detections from different rotations on the same tile."""
        if not detections:
            return []

        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        # Using shapely is more precise for rotated polygons.
        try:
            from shapely.geometry import Polygon
        except ImportError:
            print("Shapely not installed. Using a less accurate IoU calculation.")

            # Fallback to a simple rectangular IoU if shapely is not available
            def calculate_iou(box1, box2):
                r1 = Polygon(box1).minimum_rotated_rectangle
                r2 = Polygon(box2).minimum_rotated_rectangle
                return r1.intersection(r2).area / r1.union(r2).area

        def calculate_iou_shapely(box1, box2):
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            if not poly1.is_valid or not poly2.is_valid: return 0.0
            return poly1.intersection(poly2).area / poly1.union(poly2).area

        kept_detections = []
        for det in sorted_detections:
            is_overlapping = any(
                calculate_iou_shapely(det.bbox, kept_det.bbox) > iou_threshold for kept_det in kept_detections)
            if not is_overlapping:
                kept_detections.append(det)

        return kept_detections
