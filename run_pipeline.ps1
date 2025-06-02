Write-Host "▶ Slicing images..."
python -m src.data_loader.sahi_slicer
if ($LASTEXITCODE -ne 0) { Write-Error "❌ sahi_slicer failed"; exit }

Write-Host "▶ Making metadata..."
python -m src.data_loader.metadata_manager
if ($LASTEXITCODE -ne 0) { Write-Error "❌ metadata_manager failed"; exit }

Write-Host "▶ Running OCR on tiles..."
python -m src.text_detection.process_tiles_ocr
if ($LASTEXITCODE -ne 0) { Write-Error "❌ process_tiles_ocr failed"; exit }

Write-Host "▶ Filtering OCR results by core tiles..."
python -m src.grouping.filter_by_core
if ($LASTEXITCODE -ne 0) { Write-Error "❌ filter_by_core failed"; exit }

Write-Host "▶ Visualizing filtered results..."
python -m src.grouping.visualize_core_detections
if ($LASTEXITCODE -ne 0) { Write-Error "❌ visualize_core_detections failed"; exit }

Write-Host "✅ All steps completed successfully."
