# src/utils/visualization.py
import cv2
import numpy as np
from pathlib import Path

class ResultVisualizer:
    def save(self, image_path, results, output_dir):
        image = cv2.imread(image_path)
        for result in results:
            # Draw bounding boxes
            box = np.array(result["bbox"], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(image, [box], True, (0, 255, 0), 2)

            # Calculate the position for the text (slightly above the bounding box)
            x, y = box[0][0], box[0][1] - 10  # Move the text 10 pixels above the top-left corner

            # Ensure (x, y) is a tuple of integers
            cv2.putText(image, f"{result['text']} ({result['confidence']:.2f})",
                        (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        output_path = f"{output_dir}/visualizations/{Path(image_path).stem}_annotated.png"
        cv2.imwrite(output_path, image)