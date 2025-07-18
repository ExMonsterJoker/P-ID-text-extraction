import json
import glob
import os
import copy

def convert_bbox_to_pdf_points(image_bbox, image_dpi=600):
    """
    Converts a bounding box from image pixel coordinates to PDF point coordinates.
    """
    scale_factor = 72.0 / image_dpi
    pdf_points = [[round(px * scale_factor, 4), round(py * scale_factor, 4)] for px, py in image_bbox]
    return pdf_points

def process_and_convert_json(json_path, dpi):
    """
    Loads a JSON file and returns two versions: the original data and the converted data.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"  -> Error reading file {json_path}: {e}. Skipping.")
        return None, None

    converted_data = copy.deepcopy(original_data)

    if not isinstance(converted_data, list):
        print(f"  -> Warning: JSON in {json_path} is not a list. Skipping.")
        return None, None

    for detection_object in converted_data:
        if isinstance(detection_object, dict) and 'bbox' in detection_object:
            original_bbox = detection_object['bbox']
            detection_object['bbox'] = convert_bbox_to_pdf_points(original_bbox, image_dpi=dpi)

    return original_data, converted_data

def main(input_dir, image_perspective_dir, pdf_perspective_dir, dpi):
    """
    Processes JSON files to create two sets of outputs: one for the image perspective
    and one for the PDF perspective.
    """
    os.makedirs(image_perspective_dir, exist_ok=True)
    os.makedirs(pdf_perspective_dir, exist_ok=True)

    search_path = os.path.join(input_dir, '*.json')
    for json_path in glob.glob(search_path):
        base_name = os.path.basename(json_path)
        print(f"Processing {base_name}...")

        original_data, converted_data = process_and_convert_json(json_path, dpi)

        if original_data and converted_data:
            image_output_path = os.path.join(image_perspective_dir, base_name)
            try:
                with open(image_output_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, indent=4)
                print(f"  -> Saved image perspective to {image_output_path}")
            except Exception as e:
                print(f"  -> Error writing to {image_output_path}: {e}")

            pdf_output_path = os.path.join(pdf_perspective_dir, base_name)
            try:
                with open(pdf_output_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=4)
                print(f"  -> Saved PDF perspective to {pdf_output_path}")
            except Exception as e:
                print(f"  -> Error writing to {pdf_output_path}: {e}")