import os
from pdf2image import convert_from_path
from pathlib import Path
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disables the safety limit

def pdf_to_image_high_quality(input_folder, output_folder, dpi=600, image_format='PNG',
                              thread_count=1, use_pdftocairo=True):
    """
    Convert PDF files to high-quality images with customizable parameters

    Args:
        input_folder (str): Path to folder containing PDF files
        output_folder (str): Path to folder where images will be saved
        dpi (int): Resolution (300=good, 600=excellent, 1200=ultra-high for large P&IDs)
        image_format (str): Output format ('PNG' for lossless, 'TIFF' for professional)
        thread_count (int): Number of threads for faster processing
        use_pdftocairo (bool): Use pdftocairo backend for better quality
    """

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the input folder.")
        return

    print(f"Found {len(pdf_files)} PDF files to convert...")
    print(f"Settings: DPI={dpi}, Format={image_format}, Threads={thread_count}")
    print(f"Backend: {'pdftocairo' if use_pdftocairo else 'pdftoppm'}")
    print("-" * 50)

    total_start_time = time.time()

    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]

            print(f"[{idx}/{len(pdf_files)}] Converting: {pdf_file}")
            start_time = time.time()

            # Convert PDF to images with high quality settings
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                thread_count=thread_count,
                use_pdftocairo=use_pdftocairo,  # Better quality for technical drawings
                fmt=image_format.lower()
            )

            # Save each page as an image
            for i, page in enumerate(pages):
                if len(pages) == 1:
                    # Single page PDF
                    image_name = f"{pdf_name}.{image_format.lower()}"
                else:
                    # Multi-page PDF
                    image_name = f"{pdf_name}_page_{i + 1:02d}.{image_format.lower()}"

                image_path = os.path.join(output_folder, image_name)

                # Save with optimal quality settings
                if image_format.upper() == 'PNG':
                    page.save(image_path, 'PNG', optimize=False)  # No compression for PNG
                elif image_format.upper() == 'JPEG':
                    page.save(image_path, 'JPEG', quality=95, optimize=True)
                elif image_format.upper() == 'TIFF':
                    page.save(image_path, 'TIFF', compression='lzw')  # Lossless compression
                else:
                    page.save(image_path, image_format.upper())

                # Get file size for info
                file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
                print(f"  ✓ Saved: {image_name} ({file_size:.1f} MB)")

            elapsed_time = time.time() - start_time
            print(f"  Time: {elapsed_time:.1f}s")
            print()

        except Exception as e:
            print(f"  ✗ Error converting {pdf_file}: {str(e)}")
            print()

    total_time = time.time() - total_start_time
    print("-" * 50)
    print(f"Conversion completed! Total time: {total_time:.1f}s")


# Preset configurations for different quality needs
def convert_with_preset(input_folder, output_folder, preset='high_quality'):
    """
    Convert with predefined quality presets

    Presets:
    - 'draft': Fast conversion, lower quality (200 DPI)
    - 'standard': Good quality for viewing (300 DPI)
    - 'high_quality': High quality for detailed work (600 DPI)
    - 'ultra_high': Maximum quality for large P&IDs (1200 DPI)
    - 'print_ready': Professional print quality (600 DPI, TIFF)
    """

    presets = {
        'draft': {
            'dpi': 200,
            'image_format': 'JPEG',
            'thread_count': 4,
            'use_pdftocairo': False
        },
        'standard': {
            'dpi': 300,
            'image_format': 'PNG',
            'thread_count': 2,
            'use_pdftocairo': True
        },
        'high_quality': {
            'dpi': 600,
            'image_format': 'PNG',
            'thread_count': 1,
            'use_pdftocairo': True
        },
        'ultra_high': {
            'dpi': 1200,
            'image_format': 'PNG',
            'thread_count': 1,
            'use_pdftocairo': True
        },
        'print_ready': {
            'dpi': 600,
            'image_format': 'TIFF',
            'thread_count': 1,
            'use_pdftocairo': True
        }
    }

    if preset not in presets:
        print(f"Unknown preset: {preset}")
        print(f"Available presets: {list(presets.keys())}")
        return

    settings = presets[preset]
    print(f"Using preset: {preset}")

    pdf_to_image_high_quality(input_folder, output_folder, **settings)


# Interactive version with user input
def interactive_conversion():
    """
    Interactive version where user can specify all parameters
    """
    print("=== PDF to Image Converter for P&ID ===")
    print()

    # Get folder paths
    input_folder = input("Enter input folder path: ").strip().strip('"')
    output_folder = input("Enter output folder path: ").strip().strip('"')

    # Get DPI
    print("\nDPI Options:")
    print("200 - Draft quality (fast)")
    print("300 - Standard quality")
    print("600 - High quality (recommended for P&ID)")
    print("1200 - Ultra high quality (large files)")

    while True:
        try:
            dpi = int(input("Enter DPI (default 600): ") or "600")
            if dpi < 72:
                print("DPI too low, minimum is 72")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    # Get image format
    print("\nImage Format Options:")
    print("PNG - Lossless, larger files (recommended)")
    print("JPEG - Smaller files, slight quality loss")
    print("TIFF - Professional, lossless compression")

    format_choice = input("Enter format (PNG/JPEG/TIFF, default PNG): ").upper() or "PNG"
    if format_choice not in ['PNG', 'JPEG', 'TIFF']:
        format_choice = 'PNG'

    # Get thread count
    thread_count = int(input("Enter thread count (1-4, default 1): ") or "1")
    thread_count = max(1, min(4, thread_count))

    print(f"\nStarting conversion with:")
    print(f"DPI: {dpi}")
    print(f"Format: {format_choice}")
    print(f"Threads: {thread_count}")
    print()

    pdf_to_image_high_quality(
        input_folder,
        output_folder,
        dpi=dpi,
        image_format=format_choice,
        thread_count=thread_count
    )


# Usage examples
if __name__ == "__main__":
    # Method 1: Direct call with custom parameters
    # pdf_to_image_high_quality(
    #     input_folder="path/to/pdf/folder",
    #     output_folder="path/to/output/folder",
    #     dpi=600,  # Adjust this for quality vs file size
    #     image_format='PNG'
    # )

    # Method 2: Use presets
    # convert_with_preset(
    #     input_folder="path/to/pdf/folder",
    #     output_folder="path/to/output/folder",
    #     preset='high_quality'  # or 'ultra_high' for maximum quality
    # )

    # Method 3: Interactive mode
    interactive_conversion()
