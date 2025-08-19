import os
import json
import logging
import torch
from typing import Dict, Optional
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import numpy as np
from glob import glob
from configs import get_config
from torch.nn import functional as F

class TrOCRTextRecognition:
    def __init__(self):
        """
        Initializes the TrOCR text recognizer.
        Configuration is fetched automatically.
        """
        self.config = get_config('ocr')
        logging.info(f"Loaded OCR config: {self.config}")
        if not self.config:
            logging.error("OCR configuration could not be loaded. Exiting.")
            # Added a return to stop initialization if config is missing
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get('gpu', True) else "cpu")
        self.min_confidence = self.config.get('confidence_threshold', 0.95)
        self.model_name = self.config.get('model_name', "microsoft/trocr-small-printed")

        self.processor = None
        self.model = None
        self.easyocr_reader = None

        self._load_models()

    def _load_models(self):
        try:
            logging.info(f"Loading TrOCR model: {self.model_name}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"TrOCR model loaded successfully on {self.device}")

            logging.info("Initializing EasyOCR reader")
            self.easyocr_reader = easyocr.Reader(['en'])
            logging.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def _rotate_image_90_clockwise(self, image: Image.Image) -> Image.Image:
        return image.rotate(-90, expand=True)

    def _recognize_with_trocr(self, image: Image.Image) -> Optional[Dict]:
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                tokens = generated_ids[0]
                scores_per_step = outputs.scores
                token_confidences = []
                for token_id, step_logits in zip(tokens[1:], scores_per_step):
                    probs = F.softmax(step_logits, dim=-1)
                    token_confidences.append(probs[0, token_id].item())
                confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0

            return {
                "text": generated_text.strip(),
                "confidence": confidence,
                "method": "TrOCR"
            }
        except Exception as e:
            logging.error(f"Error in TrOCR recognition: {e}")
            return None

    def _recognize_with_easyocr(self, image: Image.Image) -> Optional[Dict]:
        try:
            image_np = np.array(image)
            results = self.easyocr_reader.readtext(image_np)
            if results:
                combined_text = " ".join([result[1] for result in results])
                avg_confidence = sum([result[2] for result in results]) / len(results)
                return {
                    "text": combined_text.strip(),
                    "confidence": avg_confidence,
                    "method": "EasyOCR",
                    "easyocr_detections": len(results)
                }
            else:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "method": "EasyOCR",
                    "easyocr_detections": 0
                }
        except Exception as e:
            logging.error(f"Error in EasyOCR recognition: {e}")
            return None

    def recognize_text(self, image_path: str, manifest_item: Dict = None) -> Optional[Dict]:
        try:
            image = Image.open(image_path).convert('RGB')
            trocr_result = self._recognize_with_trocr(image)

            if trocr_result and trocr_result['confidence'] >= self.min_confidence:
                return trocr_result

            image_for_easyocr = image
            if manifest_item and manifest_item.get('orientation') == 90:
                image_for_easyocr = self._rotate_image_90_clockwise(image)

            easyocr_result = self._recognize_with_easyocr(image_for_easyocr)

            if easyocr_result and easyocr_result['text']:
                return easyocr_result
            else:
                return trocr_result if trocr_result else None
        except Exception as e:
            logging.error(f"Error recognizing text from {image_path}: {e}")
            return None

    def process_cropped_images(self, input_dir: str, output_dir: str) -> None:
        logging.info("Starting text recognition on cropped images (TrOCR + EasyOCR fallback)")
        os.makedirs(output_dir, exist_ok=True)
        crop_folders = [d for d in glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]

        total_images_processed = 0
        total_crops_recognized = 0
        trocr_success_count = 0
        easyocr_success_count = 0

        for folder in crop_folders:
            try:
                base_name = os.path.basename(folder)
                manifest_path = os.path.join(folder, "manifest.json")
                if not os.path.exists(manifest_path):
                    logging.warning(f"Manifest not found in {folder}. Skipping.")
                    continue

                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                original_image_size = manifest[0].get('original_image_size') if manifest else None
                final_annotations = []

                for item in manifest:
                    crop_path = os.path.join(folder, item['crop_filename'])
                    if not os.path.exists(crop_path):
                        continue

                    recognition_result = self.recognize_text(crop_path, item)

                    if recognition_result and recognition_result['text']:
                        annotation = {
                            "text": recognition_result['text'],
                            "confidence": float(recognition_result['confidence']),
                            "bbox": item['original_bbox'],
                            "crop_filename": item['crop_filename'],
                            "ocr_method": recognition_result['method']
                        }
                        if recognition_result['method'] == "EasyOCR" and 'easyocr_detections' in recognition_result:
                            annotation["easyocr_detections"] = recognition_result['easyocr_detections']
                        if original_image_size:
                            annotation["original_image_size"] = original_image_size
                        final_annotations.append(annotation)

                        if recognition_result['method'] == "TrOCR":
                            trocr_success_count += 1
                        elif recognition_result['method'] == "EasyOCR":
                            easyocr_success_count += 1

                if final_annotations:
                    output_json_path = os.path.join(output_dir, f"{base_name}_final.json")
                    with open(output_json_path, 'w') as f:
                        json.dump(final_annotations, f, indent=2)

                    total_crops_recognized += len(final_annotations)
                    total_images_processed += 1

            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}", exc_info=True)

        logging.info(f"Total images processed: {total_images_processed}")
        logging.info(f"Total crops with recognized text: {total_crops_recognized}")
        logging.info(f"Successful TrOCR recognitions: {trocr_success_count}")
        logging.info(f"Successful EasyOCR fallback recognitions: {easyocr_success_count}")


def run_text_recognition_step() -> None:
    """
    Runs the full text recognition step, including TrOCR and EasyOCR fallback.
    Fetches all required configuration internally.
    """
    logging.info("--- Starting Step: Text Recognition (TrOCR + EasyOCR Fallback) ---")
    try:
        # Get required configuration sections
        data_loader_config = get_config('data_loader')
        cropping_config = get_config('cropping')

        # Determine input/output directories from config
        input_dir = cropping_config.get('output_dir', 'data/processed/cropping')
        output_dir = data_loader_config.get('text_recognition_output_dir', 'data/processed/metadata/final_annotations')

        if not os.path.exists(input_dir):
            logging.error(f"Input directory for text recognition not found: {input_dir}")
            return

        # Initialize the recognizer (it will fetch its own 'ocr' config)
        recognizer = TrOCRTextRecognition()
        recognizer.process_cropped_images(input_dir, output_dir)

    except Exception as e:
        logging.error(f"Error in text recognition step: {e}", exc_info=True)
        raise
    logging.info("--- Finished Step: Text Recognition (TrOCR + EasyOCR Fallback) ---")


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # No need to get config here, the functions will do it.
    logging.info("Running text recognition step for testing...")
    run_text_recognition_step()
    logging.info("Text recognition step finished.")
