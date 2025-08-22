import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
from scipy.signal import find_peaks

from src.segmentation.preprocessing import preprocess_for_hpp

class HPPSegmenter:
    """
    A class to segment multi-line text crops into single-line crops using
    Horizontal and Vertical Projection Profiling (HPP).
    """

    def __init__(self, config: Dict):
        """Initializes the HPPSegmenter with configuration parameters."""
        self.config = config

    def is_multiline_candidate(self, image: np.ndarray, orientation: str) -> bool:
        """Determines if a crop is a candidate for multi-line segmentation."""
        height, width = image.shape[:2]
        if orientation == 'horizontal':
            return height > self.config.get('multiline_height_threshold', 40)
        elif orientation == 'vertical':
            return width > self.config.get('multiline_width_threshold', 40)
        return False

    def _find_valleys(self, profile: np.ndarray) -> np.ndarray:
        """
        Finds valleys in a projection profile using SciPy's find_peaks for better accuracy.
        """
        # Finding valleys is equivalent to finding peaks in the inverted profile.
        inverted_profile = -profile

        # Use prominence to avoid shallow valleys and distance to enforce separation.
        min_line_height = self.config.get('min_line_height', 5)

        # Prominence measures how much a peak stands out from the surrounding baseline.
        # A threshold relative to the profile's dynamic range makes it more robust.
        prominence_threshold = (np.max(profile) - np.min(profile)) * self.config.get('valley_threshold', 0.1)

        valleys, _ = find_peaks(
            inverted_profile,
            distance=min_line_height,
            prominence=prominence_threshold
        )
        return valleys

    def _create_line_boxes(self, original_bbox: List[int], split_points: List[int], orientation: str) -> List[List[int]]:
        """Creates new bounding boxes for each segmented line from split points."""
        line_boxes = []
        x_min, y_min, x_max, y_max = original_bbox

        # The dimension to split along
        max_dim = (y_max - y_min) if orientation == 'horizontal' else (x_max - x_min)

        # Add the start and end of the crop to the split points to create segments
        all_splits = sorted(list(set([0] + split_points + [max_dim])))

        for i in range(len(all_splits) - 1):
            start_split = all_splits[i]
            end_split = all_splits[i+1]

            # Filter out segments that are too small
            if (end_split - start_split) < self.config.get('min_line_height', 5):
                continue

            if orientation == 'horizontal':
                new_box = [x_min, y_min + start_split, x_max, y_min + end_split]
            else:  # vertical
                new_box = [x_min + start_split, y_min, x_min + end_split, y_max]

            line_boxes.append(new_box)

        return line_boxes

    def _segment_lines(self, image: np.ndarray, orientation: str) -> Optional[np.ndarray]:
        """Calculates the projection profile and finds valleys to determine split points."""
        profile = np.sum(image, axis=1) if orientation == 'horizontal' else np.sum(image, axis=0)

        if np.sum(profile) == 0:  # Handle empty or all-black images
            logging.warning("Cannot segment an empty preprocessed image.")
            return None

        valleys = self._find_valleys(profile)

        return valleys if valleys.any() else None

    def segment_multiline_crop(self, crop_image: np.ndarray, original_bbox: List[int]) -> List[List[int]]:
        """Main function to segment a multi-line crop into single lines."""
        if crop_image is None or crop_image.size == 0:
            logging.warning(f"Cannot segment null or empty image for bbox {original_bbox}.")
            return [original_bbox]

        height, width = crop_image.shape[:2]
        orientation = 'horizontal' if width >= height else 'vertical'

        if not self.is_multiline_candidate(crop_image, orientation):
            return [original_bbox]

        try:
            preprocessed_image = preprocess_for_hpp(
                crop_image,
                light_denoising=self.config.get('light_denoising', True),
                enhance_contrast=self.config.get('enhance_contrast', True),
                noise_reduction=self.config.get('noise_reduction', False),
                use_adaptive_threshold=self.config.get('use_adaptive_threshold', False),
                deskew=True
            )

            split_points = self._segment_lines(preprocessed_image, orientation)

            if split_points is None:
                logging.info(f"No valid split points found for bbox {original_bbox}. Returning original box.")
                return [original_bbox]

            line_boxes = self._create_line_boxes(original_bbox, split_points.tolist(), orientation)

            if not line_boxes:
                logging.warning(f"Segmentation of bbox {original_bbox} resulted in no valid line boxes. Returning original.")
                return [original_bbox]

            logging.info(f"Segmented bbox {original_bbox} into {len(line_boxes)} lines.")
            return line_boxes

        except Exception as e:
            logging.error(f"Error during HPP segmentation for bbox {original_bbox}: {e}", exc_info=True)
            return [original_bbox]  # Fallback to original box on any error
