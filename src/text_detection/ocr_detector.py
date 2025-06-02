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
            # Fallback to CPU if GPU fails
            if self.gpu:
                print("Falling back to CPU...")
                self.gpu = False
                self.reader = easyocr.Reader(self.languages, gpu=False)

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate image by specified angle

        Args:
            image: Input image as numpy array
            angle: Rotation angle (0, 90, 180, 270)

        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # For arbitrary angles, use affine transformation
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height))

    def _transform_bbox_after_rotation(self, bbox: List[List[int]], angle: int,
                                       original_shape: Tuple[int, int],
                                       rotated_shape: Tuple[int, int]) -> List[List[int]]:
        """
        Transform bounding box coordinates back to original image coordinates after rotation

        Args:
            bbox: Bounding box from rotated image
            angle: Rotation angle used
            original_shape: (height, width) of original image
            rotated_shape: (height, width) of rotated image

        Returns:
            Transformed bounding box coordinates
        """
        if angle == 0:
            return bbox

        orig_h, orig_w = original_shape
        rot_h, rot_w = rotated_shape

        transformed_bbox = []

        for point in bbox:
            x, y = point

            if angle == 90:
                # 90° clockwise: (x,y) -> (y, orig_w - x)
                new_x = y
                new_y = orig_w - x
            elif angle == 180:
                # 180°: (x,y) -> (orig_w - x, orig_h - y)
                new_x = orig_w - x
                new_y = orig_h - y
            elif angle == 270:
                # 270° clockwise: (x,y) -> (orig_h - y, x)
                new_x = orig_h - y
                new_y = x
            else:
                # For arbitrary angles, use inverse transformation
                center_x, center_y = orig_w // 2, orig_h // 2
                cos_a = np.cos(np.radians(-angle))
                sin_a = np.sin(np.radians(-angle))

                # Translate to origin
                x_centered = x - center_x
                y_centered = y - center_y

                # Rotate
                new_x = x_centered * cos_a - y_centered * sin_a + center_x
                new_y = x_centered * sin_a + y_centered * cos_a + center_y

            transformed_bbox.append([int(new_x), int(new_y)])

        return transformed_bbox

    def detect_text_with_rotation(self, image_path: str, confidence_threshold: float = 0.5,
                                  rotation_angles: List[int] = [0, 90, 180, 270]) -> List[OCRDetection]:
        """
        Detect text in an image with multiple rotation angles

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence threshold for detections
            rotation_angles: List of rotation angles to try

        Returns:
            List of OCRDetection objects from all rotations
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_shape = image.shape[:2]  # (height, width)
        all_detections = []

        for angle in rotation_angles:
            try:
                # Rotate image
                rotated_image = self._rotate_image(image, angle)
                rotated_shape = rotated_image.shape[:2]

                # Perform OCR on rotated image
                results = self.reader.readtext(rotated_image)

                for result in results:
                    bbox, text, confidence = result

                    # Filter by confidence
                    if confidence < confidence_threshold:
                        continue

                    # Convert bbox to integer coordinates
                    bbox_int = [[int(point[0]), int(point[1])] for point in bbox]

                    # Transform bbox back to original image coordinates
                    original_bbox = self._transform_bbox_after_rotation(
                        bbox_int, angle, original_shape, rotated_shape
                    )

                    # Normalize bbox coordinates
                    height, width = original_shape
                    bbox_normalized = [
                        [point[0] / width, point[1] / height] for point in original_bbox
                    ]

                    detection = OCRDetection(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=original_bbox,
                        bbox_normalized=bbox_normalized,
                        rotation_angle=angle
                    )
                    all_detections.append(detection)

            except Exception as e:
                print(f"Error processing rotation {angle}° for {image_path}: {e}")
                continue

        return all_detections

    def detect_text(self, image_path: str, confidence_threshold: float = 0.5,
                    use_rotation: bool = True) -> List[OCRDetection]:
        """
        Detect text in an image

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence threshold for detections
            use_rotation: Whether to try multiple rotations

        Returns:
            List of OCRDetection objects
        """
        if use_rotation:
            return self.detect_text_with_rotation(image_path, confidence_threshold)
        else:
            # Original single-rotation detection
            return self.detect_text_with_rotation(image_path, confidence_threshold, [0])

    def filter_overlapping_detections(self, detections: List[OCRDetection],
                                      iou_threshold: float = 0.5) -> List[OCRDetection]:
        """
        Filter out overlapping detections, keeping the one with highest confidence

        Args:
            detections: List of OCRDetection objects
            iou_threshold: IoU threshold for considering detections as overlapping

        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        def calculate_iou(bbox1: List[List[int]], bbox2: List[List[int]]) -> float:
            """Calculate IoU between two bounding boxes"""

            # Convert to [x1, y1, x2, y2] format
            def bbox_to_rect(bbox):
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]
                return [min(xs), min(ys), max(xs), max(ys)]

            rect1 = bbox_to_rect(bbox1)
            rect2 = bbox_to_rect(bbox2)

            # Calculate intersection
            x1 = max(rect1[0], rect2[0])
            y1 = max(rect1[1], rect2[1])
            x2 = min(rect1[2], rect2[2])
            y2 = min(rect1[3], rect2[3])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)

            # Calculate union
            area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
            area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        filtered_detections = []

        for detection in sorted_detections:
            # Check if this detection overlaps significantly with any already selected detection
            is_duplicate = False
            for selected in filtered_detections:
                iou = calculate_iou(detection.bbox, selected.bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(detection)

        return filtered_detections

    def detect_batch(self, image_paths: List[str], confidence_threshold: float = 0.5,
                     use_rotation: bool = True, filter_overlaps: bool = True) -> Dict[str, List[OCRDetection]]:
        """
        Detect text in multiple images

        Args:
            image_paths: List of image file paths
            confidence_threshold: Minimum confidence threshold for detections
            use_rotation: Whether to try multiple rotations
            filter_overlaps: Whether to filter overlapping detections

        Returns:
            Dictionary mapping image paths to detection lists
        """
        results = {}
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            print(f"Processing {i + 1}/{total_images}: {os.path.basename(image_path)}")

            try:
                detections = self.detect_text(image_path, confidence_threshold, use_rotation)

                if filter_overlaps and len(detections) > 1:
                    detections = self.filter_overlapping_detections(detections)

                results[image_path] = detections

                # Show rotation statistics
                rotation_stats = {}
                for det in detections:
                    angle = det.rotation_angle
                    rotation_stats[angle] = rotation_stats.get(angle, 0) + 1

                print(f"  Found {len(detections)} text detections")
                if rotation_stats:
                    rotation_info = ", ".join([f"{angle}°: {count}" for angle, count in rotation_stats.items()])
                    print(f"  Rotation breakdown: {rotation_info}")

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                results[image_path] = []

        return results
