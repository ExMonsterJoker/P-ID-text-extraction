import json
import glob
import os
import copy
import math
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: Neither PyPDF2 nor PyMuPDF is available. PDF dimension checking will be disabled.")

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

def get_pdf_dimensions(pdf_path):
    """
    Get the dimensions of the first page of a PDF file in points.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        tuple: (width, height) in points, or None if unable to read
    """
    if not PDF_AVAILABLE:
        print(f"  -> Warning: PDF libraries not available. Cannot check dimensions for {pdf_path}")
        return None

    try:
        # Try PyPDF2 first
        if 'PdfReader' in globals():
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                page = reader.pages[0]
                # Get mediabox (page dimensions)
                mediabox = page.mediabox
                width = float(mediabox.width)
                height = float(mediabox.height)
                return (width, height)

        # Try PyMuPDF if PyPDF2 not available
        elif 'fitz' in globals():
            doc = fitz.open(pdf_path)
            page = doc[0]
            rect = page.rect
            width = rect.width
            height = rect.height
            doc.close()
            return (width, height)

    except Exception as e:
        print(f"  -> Error reading PDF dimensions from {pdf_path}: {e}")
        return None

    return None

def check_dimension_compatibility(image_size, pdf_dimensions, dpi=600, tolerance_percent=5.0):
    """
    Check if image dimensions are compatible with PDF dimensions using DPI conversion.

    Args:
        image_size (tuple): (width, height) of image in pixels
        pdf_dimensions (tuple): (width, height) of PDF in points
        dpi (int): DPI used for conversion (default 600)
        tolerance_percent (float): Allowed percentage difference (default 5%)

    Returns:
        dict: Contains compatibility info with keys:
            - compatible (bool): Whether dimensions are compatible
            - image_size (tuple): Original image size
            - pdf_dimensions (tuple): PDF dimensions in points
            - expected_image_size (tuple): Expected image size based on PDF
            - actual_scale_factor (float): Actual scale factor from comparison
            - expected_scale_factor (float): Expected scale factor (72/DPI)
            - difference_percent (tuple): Percentage difference for width and height
    """
    if not image_size or not pdf_dimensions:
        return {
            'compatible': False,
            'error': 'Missing image size or PDF dimensions'
        }

    image_width, image_height = image_size
    pdf_width, pdf_height = pdf_dimensions

    # Expected scale factor: points to pixels
    expected_scale_factor = dpi / 72.0

    # Expected image size based on PDF dimensions
    expected_image_width = pdf_width * expected_scale_factor
    expected_image_height = pdf_height * expected_scale_factor

    # Calculate actual scale factors
    actual_scale_width = image_width / pdf_width if pdf_width > 0 else 0
    actual_scale_height = image_height / pdf_height if pdf_height > 0 else 0

    # Calculate percentage differences
    width_diff_percent = abs(expected_image_width - image_width) / expected_image_width * 100 if expected_image_width > 0 else 100
    height_diff_percent = abs(expected_image_height - image_height) / expected_image_height * 100 if expected_image_height > 0 else 100

    # Check if within tolerance
    compatible = (width_diff_percent <= tolerance_percent and
                 height_diff_percent <= tolerance_percent)

    return {
        'compatible': compatible,
        'image_size': image_size,
        'pdf_dimensions': pdf_dimensions,
        'expected_image_size': (round(expected_image_width), round(expected_image_height)),
        'actual_scale_factor': (actual_scale_width, actual_scale_height),
        'expected_scale_factor': expected_scale_factor,
        'difference_percent': (round(width_diff_percent, 2), round(height_diff_percent, 2)),
        'dpi_used': dpi,
        'tolerance_percent': tolerance_percent
    }


def validate_json_with_pdf(json_path, pdf_path=None, dpi=600, tolerance_percent=5.0, pdf_base_dir=None):
    """
    Validate that the JSON file's image dimensions are compatible with the corresponding PDF.
    This version constructs the PDF path using a configured base directory.

    Args:
        json_path (str): Path to the JSON file containing text recognition results.
        pdf_path (str, optional): Direct path to the PDF file. If provided, this is used instead of searching.
        dpi (int): DPI used for conversion (default 600).
        tolerance_percent (float): Allowed percentage difference (default 5%).
        pdf_base_dir (str, optional): The base directory where PDF files are stored.
                                      This should be passed from a configuration file.

    Returns:
        dict: Validation result with compatibility information.
    """
    try:
        # Load JSON data to get the original image size
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {'compatible': False, 'error': f'Error reading JSON file: {e}'}

    # Extract image size from the first item in the JSON data
    image_size = None
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        image_size = data[0].get('original_image_size')

    if not image_size:
        return {'compatible': False, 'error': 'Could not find "original_image_size" in JSON data'}

    # --- MODIFIED PDF PATH LOGIC ---
    # If a direct PDF path is not provided, construct it from the pdf_base_dir
    if pdf_path is None:
        if not pdf_base_dir:
            return {
                'compatible': False,
                'error': 'PDF path was not provided and pdf_base_dir is not configured.'
            }

        # Get the base name of the JSON file (e.g., "document_A_final" -> "document_A")
        json_base_name = os.path.splitext(os.path.basename(json_path))[0]
        if json_base_name.endswith('_final'):
            json_base_name = json_base_name.replace('_final', '')

        # Construct the expected PDF path
        pdf_path = os.path.join(pdf_base_dir, f"{json_base_name}.pdf")

    # Check if the final PDF path exists
    if not os.path.exists(pdf_path):
        return {
            'compatible': False,
            'error': f'PDF file not found at the expected path: {pdf_path}'
        }
    # --- END OF MODIFICATION ---

    # Get PDF dimensions
    pdf_dimensions = get_pdf_dimensions(pdf_path)
    if not pdf_dimensions:
        return {'compatible': False, 'error': 'Could not read dimensions from the PDF file'}

    # Perform the final compatibility check
    result = check_dimension_compatibility(image_size, pdf_dimensions, dpi, tolerance_percent)
    result['json_path'] = json_path
    result['pdf_path'] = pdf_path

    return result

def main(input_dir_json_dir, image_perspective_dir, pdf_perspective_dir, dpi, validate_dimensions=False):
    """
    Processes JSON files to create two sets of outputs: one for the image perspective
    and one for the PDF perspective.
    """
    os.makedirs(image_perspective_dir, exist_ok=True)
    os.makedirs(pdf_perspective_dir, exist_ok=True)

    search_path = os.path.join(input_dir_json_dir, '*_final.json')
    validation_results = []

    for json_path in glob.glob(search_path):
        base_name = os.path.basename(json_path)
        print(f"Processing {base_name}...")

        # Validate dimensions if requested
        if validate_dimensions:
            validation_result = validate_json_with_pdf(json_path, dpi=dpi)
            validation_results.append(validation_result)

            if validation_result.get('compatible', False):
                print(f"  -> ✓ Dimension validation passed")
                diff_w, diff_h = validation_result.get('difference_percent', (0, 0))
                print(f"     Width difference: {diff_w}%, Height difference: {diff_h}%")
            else:
                print(f"  -> ⚠ Dimension validation failed: {validation_result.get('error', 'Unknown error')}")
                if 'difference_percent' in validation_result:
                    diff_w, diff_h = validation_result['difference_percent']
                    print(f"     Width difference: {diff_w}%, Height difference: {diff_h}%")

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

    # Print validation summary
    if validate_dimensions and validation_results:
        print(f"\n=== Dimension Validation Summary ===")
        compatible_count = sum(1 for r in validation_results if r.get('compatible', False))
        total_count = len(validation_results)
        print(f"Compatible files: {compatible_count}/{total_count}")

        if compatible_count < total_count:
            print(f"\nFiles with dimension issues:")
            for result in validation_results:
                if not result.get('compatible', False):
                    json_file = os.path.basename(result.get('json_path', 'Unknown'))
                    error = result.get('error', 'Unknown error')
                    print(f"  - {json_file}: {error}")

def test_dimension_validation(json_path, pdf_path=None, dpi=600, pdf_base_dir='data/GS6'):
    """
    Test function to validate a single JSON file's dimensions against its PDF.

    Args:
        json_path (str): Path to the JSON file
        pdf_path (str, optional): Path to PDF file
        dpi (int): DPI used for conversion (default 600)
        pdf_base_dir (str): Base directory to search for PDF files (default 'data/GS6')
    """
    print(f"Testing dimension validation for: {os.path.basename(json_path)}")
    print(f"Using DPI: {dpi}")
    print(f"PDF base directory: {pdf_base_dir}")

    result = validate_json_with_pdf(json_path, pdf_path, dpi, pdf_base_dir=pdf_base_dir)

    print(f"\nValidation Result:")
    print(f"  Compatible: {result.get('compatible', False)}")

    if 'error' in result:
        print(f"  Error: {result['error']}")
        return result

    print(f"  JSON file: {result.get('json_path', 'Unknown')}")
    print(f"  PDF file: {result.get('pdf_path', 'Unknown')}")
    print(f"  Image size: {result.get('image_size', 'Unknown')}")
    print(f"  PDF dimensions: {result.get('pdf_dimensions', 'Unknown')}")
    print(f"  Expected image size: {result.get('expected_image_size', 'Unknown')}")
    print(f"  Difference percentage (W, H): {result.get('difference_percent', 'Unknown')}")
    print(f"  Expected scale factor: {result.get('expected_scale_factor', 'Unknown'):.2f}")
    print(f"  Actual scale factor: {result.get('actual_scale_factor', 'Unknown')}")
    print(f"  Tolerance: {result.get('tolerance_percent', 'Unknown')}%")

    return result

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test" and len(sys.argv) > 2:
            # Test mode: python convert_coord.py test <json_path> [pdf_path] [dpi]
            json_path = sys.argv[2]
            pdf_path = sys.argv[3] if len(sys.argv) > 3 else None
            dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 600
            test_dimension_validation(json_path, pdf_path, dpi)
        else:
            print("Usage:")
            print("  python convert_coord.py test <json_path> [pdf_path] [dpi]")
            print("  python convert_coord.py [input_dir] [image_output_dir] [pdf_output_dir] [dpi]")
    else:
        # Default behavior - process all files
        input_dir = "data/raw"  # Default input directory
        image_perspective_dir = "data/outputs/image_perspective"
        pdf_perspective_dir = "data/outputs/pdf_perspective"
        dpi = 600

        print("Using default directories:")
        print(f"  Input: {input_dir}")
        print(f"  Image output: {image_perspective_dir}")
        print(f"  PDF output: {pdf_perspective_dir}")
        print(f"  DPI: {dpi}")

        main(input_dir, image_perspective_dir, pdf_perspective_dir, dpi, validate_dimensions=True)
