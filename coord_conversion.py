import json
import glob
import os


def convert_bbox_to_pdf_points(image_bbox, image_dpi=600):
    """
    Converts a bounding box from image pixel coordinates to PDF point coordinates.

    A PDF uses a standard of 72 points per inch. The conversion factor is
    calculated by dividing the PDF points-per-inch by the image's dots-per-inch (DPI).

    Args:
      image_bbox (list): A list of [x, y] corner points in pixels.
      image_dpi (int): The DPI of the source image (e.g., 600).

    Returns:
      list: A new list containing the converted [x, y] coordinates in PDF points.
    """
    # The scaling factor is derived from (points per inch) / (pixels per inch)
    scale_factor = 72.0 / image_dpi

    # Use a list comprehension for a clean and efficient conversion
    pdf_points = [[round(px * scale_factor, 4), round(py * scale_factor, 4)] for px, py in image_bbox]

    return pdf_points


def process_json_file(json_path, dpi):
    """
    Loads a single JSON file, converts all bounding boxes, and returns the modified data.

    Args:
        json_path (str): The path to the input JSON file.
        dpi (int): The DPI of the source image.

    Returns:
        list or None: The modified data with converted coordinates, or None on error.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"  -> Error: Could not decode JSON from {json_path}. Skipping.")
        return None
    except Exception as e:
        print(f"  -> Error reading file {json_path}: {e}. Skipping.")
        return None

    # Check if the data is a list of detection objects
    if not isinstance(data, list):
        print(f"  -> Warning: JSON in {json_path} is not a list. Skipping.")
        return None

    # Iterate through each detection object in the list
    for detection_object in data:
        # Ensure the object is a dictionary and has a 'bbox' key
        if isinstance(detection_object, dict) and 'bbox' in detection_object:
            original_bbox = detection_object['bbox']
            # Convert the bbox and update the dictionary in-place
            detection_object['bbox'] = convert_bbox_to_pdf_points(original_bbox, image_dpi=dpi)

    return data


def main(input_dir, output_dir, dpi):
    """
    Finds all JSON files in the input directory, converts their bounding box
    coordinates, and saves the output to the output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all files ending with .json in the input directory
    search_path = os.path.join(input_dir, '*.json')
    for json_path in glob.glob(search_path):
        print(f"Processing {os.path.basename(json_path)}...")

        # Process the file to get the converted data
        converted_data = process_json_file(json_path, dpi)

        if converted_data:
            # Create the new filename for the output directory
            base_name = os.path.basename(json_path)
            output_path = os.path.join(output_dir, base_name)

            # Save the new data to the output file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=4)
                print(f"  -> Successfully converted and saved to {output_path}")
            except Exception as e:
                print(f"  -> Error writing to {output_path}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Set your default input and output folders here.
    # The script will read all .json files from the input_folder
    # and save the converted files to the output_folder.
    input_folder = "data/processed/metadata/final_annotations"  # <<< CHANGE THIS to your input folder name
    output_folder = "data/outputs/final_visualizations"  # <<< CHANGE THIS to your desired output folder name
    image_dpi = 600  # The DPI of the source images

    # Call the main function with the configured paths and DPI
    print(f"Starting conversion...")
    print(f"Input folder: '{os.path.abspath(input_folder)}'")
    print(f"Output folder: '{os.path.abspath(output_folder)}'")
    print("-" * 40)

    main(input_dir=input_folder, output_dir=output_folder, dpi=image_dpi)

    print("-" * 40)
    print("Conversion complete.")