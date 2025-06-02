Sliced images
python -m src.data_loader.sahi_slicer

metadata maker
python -m src.data_loader.metadata_manager

OCR tiles
python -m src.text_detection.process_tiles_ocr

Filter OCR result based core tiles only
python -m src.grouping.filter_core_detections
