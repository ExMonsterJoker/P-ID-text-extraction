import json
import os
from pathlib import Path
import time

# --- Configuration based on your folder tree ---
# These are default paths that will be suggested to the user.
DEFAULT_IMAGE_DIR = "data/raw"
MY_OWN_JSON_DIR = "data/processed/metadata/final_annotations"
DEFAULT_OUTPUT_DIR = "data/outputs/converted_annotations"


# --- Helper Functions (No changes from previous version) ---

def get_image_info(base_filename, image_dir):
    """Finds a matching image and returns its full path and filename."""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    for ext in supported_extensions:
        image_path = Path(image_dir) / f"{base_filename}{ext}"
        if image_path.exists():
            return str(image_path), image_path.name
    return None, None


def to_coco_bbox(points):
    """Converts a 4-point polygon to a COCO bbox [x, y, w, h]."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def from_coco_bbox(bbox):
    """Converts a COCO bbox [x, y, w, h] to a 4-point polygon."""
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


# --- Loading Functions (No changes from previous version) ---

def load_my_own_format(json_path):
    """Load annotations from custom format."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations = []
        for item in data:
            if "text" in item and "bbox" in item:
                annotations.append({
                    "transcription": item["text"],
                    "points": item["bbox"]
                })
        return annotations
    except Exception as e:
        print(f"Error loading custom format: {e}")
        return []


def load_ppocr_format(label_file_path, image_filename):
    annotations = []
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2 and Path(parts[0]).name == image_filename:
                return json.loads(parts[1])
    return []


def load_via_format(json_path):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    key = list(data.keys())[0]
    image_filename = data[key]['filename']
    annotations = []
    for region in data[key]['regions']:
        points_x = region['shape_attributes']['all_points_x']
        points_y = region['shape_attributes']['all_points_y']
        annotations.append({"transcription": region['region_attributes']['text'],
                            "points": [[x, y] for x, y in zip(points_x, points_y)]})
    return annotations, image_filename


def load_coco_format(json_path):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    image_filename = data['images'][0]['file_name']
    annotations = []
    for anno in data['annotations']:
        text = anno.get('attributes', {}).get('text', '')
        annotations.append({"transcription": text, "points": from_coco_bbox(anno['bbox'])})
    return annotations, image_filename


# --- Saving Functions (No changes from previous version) ---

def save_to_ppocr(annotations, image_path_str, output_label_file):
    # PPOCR format: image_path\t[{"transcription": "text", "points": [[x,y]...]}, ...]
    ppocr_annotations = []
    for item in annotations:
        ppocr_annotations.append({
            "transcription": item["transcription"],
            "points": item["points"]
        })

    json_string = json.dumps(ppocr_annotations, ensure_ascii=False)
    line = f"{image_path_str}\t{json_string}\n"
    with open(output_label_file, "a", encoding="utf-8") as f:
        f.write(line)


def save_to_via(annotations, image_filename, output_path):
    # Get actual file size if possible
    file_size = 0
    try:
        file_size = os.path.getsize(output_path.parent / image_filename)
    except:
        file_size = -1

    via_regions = []
    for item in annotations:
        points_x = [p[0] for p in item['points']]
        points_y = [p[1] for p in item['points']]
        via_regions.append({
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": points_x,
                "all_points_y": points_y
            },
            "region_attributes": {"text": item["transcription"]}
        })

    # VIA key format: filename + filesize
    via_key = f"{image_filename}{file_size}"
    via_data = {
        via_key: {
            "filename": image_filename,
            "size": file_size,
            "regions": via_regions,
            "file_attributes": {}
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(via_data, f, indent=2)


def save_to_coco(annotations, image_filename, output_path, image_dir=None):
    # Try to get actual image dimensions
    width, height = -1, -1
    if image_dir:
        from PIL import Image
        try:
            img_path = Path(image_dir) / image_filename
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            print(f"Warning: Could not get dimensions for {image_filename}")

    coco_output = {
        "images": [{"id": 1, "file_name": image_filename, "width": width, "height": height}],
        "annotations": [],
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}]
    }

    for i, item in enumerate(annotations):
        bbox = to_coco_bbox(item['points'])
        coco_output["annotations"].append({
            "id": i + 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "attributes": {"text": item["transcription"]}
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=2)


# --- Main Interactive Execution Logic ---

def get_user_choice(prompt, options):
    """Helper function to get a valid user choice from a dictionary of options."""
    print(prompt)
    for key, value in options.items():
        print(f"  {key}) {value['name']}")

    while True:
        choice = input("Enter the number of your choice: ")
        if choice in options:
            return options[choice]['id']
        else:
            print("❌ Invalid selection. Please try again.")


def get_user_path(prompt, default_path):
    """Helper function to get a valid path from the user."""
    while True:
        path_str = input(f"{prompt} (press Enter for default: '{default_path}'): ") or default_path
        path = Path(path_str)
        if path.exists():
            return str(path)
        else:
            print(f"❌ Error: Path not found: '{path}'. Please enter a valid path.")


def interactive_main():
    """Guides the user through the conversion process with interactive prompts."""
    print("--- Universal Annotation Converter ---")

    # Define format choices
    formats = {
        '1': {'id': 'my_own', 'name': 'My Own Custom Format (*_final.json)'},
        '2': {'id': 'ppocr', 'name': 'PPOCR Format (label.txt)'},
        '3': {'id': 'via', 'name': 'VGG Image Annotator (VIA) (*_via.json)'},
        '4': {'id': 'coco', 'name': 'MS COCO Format (*_coco.json)'}
    }

    # 1. Get FROM format
    from_format = get_user_choice("\n➡️ Step 1: Select the SOURCE format (the format you are converting from):",
                                  formats)

    # 2. Get TO format
    # Create a copy of formats and remove the selected from_format
    to_formats_options = {k: v for k, v in formats.items() if v['id'] != from_format}
    to_format = get_user_choice("\n➡️ Step 2: Select the TARGET format (the format you want to convert to):",
                                to_formats_options)

    # 3. Get paths
    default_input = MY_OWN_JSON_DIR if from_format == 'my_own' else DEFAULT_OUTPUT_DIR
    input_path = get_user_path("\n➡️ Step 3: Enter the path to the input file or directory:", default_input)
    output_path = get_user_path("➡️ Step 4: Enter the path to the output directory:", DEFAULT_OUTPUT_DIR)
    image_dir = get_user_path("➡️ Step 5: Enter the path to the image directory:", DEFAULT_IMAGE_DIR)

    print("\n--- Starting Conversion ---")
    print(f"From: {from_format} | To: {to_format}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("---------------------------\n")
    time.sleep(1)  # Pause for user to read

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    input_files = list(Path(input_path).rglob('*.*')) if Path(input_path).is_dir() else [Path(input_path)]
    if not input_files:
        print(f"❌ Error: No files found in the input path: {input_path}")
        return

    if to_format == 'ppocr':
        output_label_file = Path(output_path) / 'ppocr_label.txt'
        output_label_file.unlink(missing_ok=True)
        print(f"Will generate a single PPOCR file: {output_label_file}")

    for file_path in input_files:
        if file_path.is_dir(): continue

        annotations, image_filename = [], None
        try:
            if from_format == 'my_own':
                if not file_path.name.endswith('_final.json'): continue
                base_filename = file_path.stem.replace('_final', '')
                _, image_filename = get_image_info(base_filename, image_dir)
                if image_filename: annotations = load_my_own_format(file_path)
            elif from_format == 'ppocr':
                # This requires iterating images to find lines in the ppocr file
                print("Note: Converting FROM PPOCR processes all images in the image dir.")
                # This part is simplified for interactive use. A more robust implementation would be needed.
                # For now, we assume we want to convert the *entire* ppocr file for all relevant images.
                # This part of the logic is complex for an interactive script, we'll skip its full implementation for now.
                print("Error: Interactive conversion FROM PPOCR is complex and not fully supported in this version.")
                break  # exit the loop
            elif from_format == 'via':
                if not file_path.name.endswith('_via.json'): continue
                annotations, image_filename = load_via_format(file_path)
            elif from_format == 'coco':
                if not file_path.name.endswith('_coco.json'): continue
                annotations, image_filename = load_coco_format(file_path)

            if not image_filename or not annotations:
                continue

            print(f"  ✔ Loaded {len(annotations)} annotations for '{image_filename}'")
            image_path_str, _ = get_image_info(Path(image_filename).stem, image_dir)

            if to_format == 'ppocr':
                save_to_ppocr(annotations, image_path_str, output_label_file)
            elif to_format == 'via':
                output_file = Path(output_path) / f"{Path(image_filename).stem}_via.json"
                save_to_via(annotations, image_filename, output_file)
            elif to_format == 'coco':
                output_file = Path(output_path) / f"{Path(image_filename).stem}_coco.json"
                save_to_coco(annotations, image_filename, output_file)

            if to_format != 'ppocr':
                print(f"    ✔ Saved converted file to {output_file}")

        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}. Skipping.")

    print("\n✅ Conversion process complete!")


if __name__ == "__main__":
    interactive_main()