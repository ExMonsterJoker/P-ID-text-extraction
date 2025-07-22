import os
import json
import logging
import torch
from typing import List, Dict, Optional
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from glob import glob


class TrOCRTextRecognition:
    """
    Text recognition using TrOCR (Transformer-based Optical Character Recognition) model.
    Specifically uses the small printed text model for P&ID text recognition.
    """

    def __init__(self, config: Dict):
        """
        Initialize the TrOCR text recognition model.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('gpu', True) else "cpu")
        self.min_confidence = config.get('confidence_threshold', 0.1)

        # Initialize TrOCR model and processor
        self.model_name = "microsoft/trocr-small-printed"
        self.processor = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the TrOCR model and processor."""
        try:
            logging.info(f"Loading TrOCR model: {self.model_name}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"TrOCR model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load TrOCR model: {e}")
            raise

    def recognize_text(self, image_path: str) -> Optional[Dict]:
        """
        Recognize text from a single image using TrOCR.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing recognized text and confidence, or None if recognition fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')

            # Process image with TrOCR
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # TrOCR doesn't provide confidence scores directly, so we'll use a default high confidence
            # You could implement a confidence estimation method if needed
            confidence = 0.95

            return {
                "text": generated_text.strip(),
                "confidence": confidence
            }

        except Exception as e:
            logging.error(f"Error recognizing text from {image_path}: {e}")
            return None

    def process_cropped_images(self, input_dir: str, output_dir: str) -> None:
        """
        Process all cropped images in the input directory and save recognition results.

        Args:
            input_dir: Directory containing cropped images (from cropping step)
            output_dir: Directory to save text recognition results
        """
        logging.info("Starting TrOCR text recognition on cropped images")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all crop folders
        crop_folders = [d for d in glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]

        if not crop_folders:
            logging.warning(f"No crop folders found in {input_dir}")
            return

        total_images_processed = 0
        total_crops_recognized = 0

        for folder in crop_folders:
            try:
                base_name = os.path.basename(folder)
                logging.info(f"Processing text recognition for: {base_name}")

                # Load manifest to get crop information
                manifest_path = os.path.join(folder, "manifest.json")
                if not os.path.exists(manifest_path):
                    logging.warning(f"  - Manifest not found in {folder}. Skipping.")
                    continue

                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                final_annotations = []
                crops_processed = 0

                for item in manifest:
                    crop_path = os.path.join(folder, item['crop_filename'])
                    if not os.path.exists(crop_path):
                        logging.warning(f"  - Crop file not found: {crop_path}")
                        continue

                    try:
                        # Recognize text using TrOCR
                        recognition_result = self.recognize_text(crop_path)

                        if recognition_result and recognition_result['text']:
                            recognized_text = recognition_result['text']
                            confidence = recognition_result['confidence']

                            # Apply confidence filtering if needed
                            if confidence >= self.min_confidence:
                                final_annotations.append({
                                    "text": recognized_text,
                                    "confidence": float(confidence),
                                    "bbox": item['original_bbox'],
                                    "crop_filename": item['crop_filename']
                                })
                                crops_processed += 1
                            else:
                                logging.debug(f"  - Skipping low confidence recognition: {confidence:.3f} < {self.min_confidence}")
                        else:
                            logging.debug(f"  - No text recognized from {crop_path}")

                    except Exception as e:
                        logging.error(f"  - Error processing crop {crop_path}: {e}")
                        continue

                # Save recognition results
                if final_annotations:
                    output_json_path = os.path.join(output_dir, f"{base_name}_final.json")
                    with open(output_json_path, 'w') as f:
                        json.dump(final_annotations, f, indent=2)

                    total_crops_recognized += len(final_annotations)
                    total_images_processed += 1

                    logging.info(f"  - Recognized text from {crops_processed} crops, saved {len(final_annotations)} final annotations to {output_json_path}")
                else:
                    logging.info(f"  - No valid text recognition results for {base_name}")

            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}", exc_info=True)
                continue

        logging.info(f"TrOCR text recognition completed:")
        logging.info(f"  Total images processed: {total_images_processed}")
        logging.info(f"  Total crops with recognized text: {total_crops_recognized}")
        logging.info(f"  Average recognized crops per image: {total_crops_recognized/total_images_processed:.2f}" if total_images_processed > 0 else "  No images processed")


def run_text_recognition_step(config: Dict) -> None:
    """
    Main function to run the TrOCR text recognition step.

    Args:
        config: Configuration dictionary from pipeline
    """
    logging.info("--- Starting Step: TrOCR Text Recognition ---")

    try:
        # Get configuration
        ocr_config = config.get('ocr', {})
        data_loader_config = config.get('data_loader', {})
        cropping_config = config.get('cropping', {})

        # Get directories
        input_dir = cropping_config.get('output_dir', 'data/processed/cropping')
        output_dir = data_loader_config.get('text_recognition_output_dir', 'data/processed/metadata/final_annotations')

        # Check if input directory exists
        if not os.path.exists(input_dir):
            logging.error(f"Input directory not found: {input_dir}")
            logging.error("Make sure the cropping step has been completed successfully.")
            return

        # Initialize text recognition
        recognizer = TrOCRTextRecognition(ocr_config)

        # Process all cropped images
        recognizer.process_cropped_images(input_dir, output_dir)

    except Exception as e:
        logging.error(f"Error in TrOCR text recognition step: {e}", exc_info=True)
        raise

    logging.info("--- Finished Step: TrOCR Text Recognition ---")


if __name__ == "__main__":
    # Example usage for testing
    import yaml

    # Load config
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run text recognition
    run_text_recognition_step(config)

