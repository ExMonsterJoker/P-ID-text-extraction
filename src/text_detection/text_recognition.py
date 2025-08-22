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
        """Initializes the TrOCR text recognizer."""
        self.config = get_config('ocr')
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get('gpu', True) else "cpu")
        self.min_confidence = self.config.get('confidence_threshold', 0.95)
        self.model_name = self.config.get('model_name', "microsoft/trocr-small-printed")
        self.processor = None
        self.model = None
        self.easyocr_reader = None
        self._load_models()

    def _load_models(self):
        try:
            logging.info(f"Loading TrOCR model: {self.model_name} to {self.device}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logging.info("Initializing EasyOCR reader")
            self.easyocr_reader = easyocr.Reader(['en'])
        except Exception as e:
            logging.error(f"Failed to load models: {e}", exc_info=True)
            raise

    def _rotate_image_90_clockwise(self, image: Image.Image) -> Image.Image:
        return image.rotate(-90, expand=True)

    def _recognize_with_trocr(self, image: Image.Image) -> Optional[Dict]:
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
                generated_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

                # Simplified confidence score calculation
                probs = torch.log_softmax(outputs.scores[0], dim=-1)
                confidence = probs.max().exp().item()

            return {"text": generated_text.strip(), "confidence": confidence, "method": "TrOCR"}
        except Exception as e:
            logging.error(f"Error in TrOCR recognition: {e}", exc_info=True)
            return None

    def _recognize_with_easyocr(self, image: Image.Image) -> Optional[Dict]:
        try:
            image_np = np.array(image)
            results = self.easyocr_reader.readtext(image_np)
            if results:
                combined_text = " ".join([res[1] for res in results])
                avg_confidence = sum([res[2] for res in results]) / len(results)
                return {"text": combined_text.strip(), "confidence": avg_confidence, "method": "EasyOCR"}
            return {"text": "", "confidence": 0.0, "method": "EasyOCR"}
        except Exception as e:
            logging.error(f"Error in EasyOCR recognition: {e}", exc_info=True)
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
            return easyocr_result if easyocr_result and easyocr_result['text'] else trocr_result
        except Exception as e:
            logging.error(f"Error recognizing text from {image_path}: {e}", exc_info=True)
            return None

    def process_cropped_images(self, input_dir: str, output_dir: str, manifest_filename: str):
        logging.info(f"Starting text recognition on images in '{input_dir}' using '{manifest_filename}'")
        os.makedirs(output_dir, exist_ok=True)

        for folder in glob(os.path.join(input_dir, '*')):
            if not os.path.isdir(folder): continue

            base_name = os.path.basename(folder)
            logging.info(f"Processing folder: {base_name}")
            manifest_path = os.path.join(folder, manifest_filename)

            if not os.path.exists(manifest_path):
                logging.warning(f"Manifest '{manifest_filename}' not found in {folder}. Skipping.")
                continue

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            final_annotations = []
            for item in manifest:
                crop_path = os.path.join(folder, item['crop_filename'])
                if not os.path.exists(crop_path): continue

                result = self.recognize_text(crop_path, item)
                if result and result['text']:
                    final_annotations.append({
                        "text": result['text'],
                        "confidence": float(result['confidence']),
                        "bbox": item['original_bbox'],
                        "crop_filename": item['crop_filename'],
                        "ocr_method": result['method'],
                        "original_image_size": item.get('original_image_size')
                    })

            if final_annotations:
                output_json_path = os.path.join(output_dir, f"{base_name}_final.json")
                with open(output_json_path, 'w') as f:
                    json.dump(final_annotations, f, indent=2)
                logging.info(f"Recognized {len(final_annotations)} crops in {base_name}.")

def run_text_recognition_step(specific_dir: Optional[str] = None):
    """
    Runs the text recognition step. If a specific_dir is provided, it processes
    that directory with 'segmented_manifest.json'. Otherwise, it uses the default
    cropping directory with 'manifest.json'.
    """
    logging.info("--- Starting Step: Text Recognition ---")
    try:
        data_loader_config = get_config('data_loader')
        output_dir = data_loader_config.get('text_recognition_output_dir')

        if specific_dir:
            input_dir = specific_dir
            manifest_filename = "segmented_manifest.json"
        else:
            cropping_config = get_config('cropping')
            input_dir = cropping_config.get('output_dir')
            manifest_filename = "manifest.json"

        logging.info(f"Input directory: {input_dir}, Manifest: {manifest_filename}")
        if not os.path.exists(input_dir):
            logging.error(f"Input directory not found: {input_dir}")
            return

        recognizer = TrOCRTextRecognition()
        recognizer.process_cropped_images(input_dir, output_dir, manifest_filename)

    except Exception as e:
        logging.error(f"Error in text recognition step: {e}", exc_info=True)
        raise
    logging.info("--- Finished Step: Text Recognition ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_text_recognition_step()
