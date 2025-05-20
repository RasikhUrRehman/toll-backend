# Toll System Vehicle and License Plate Tracking

This project provides a complete pipeline for detecting, tracking, and analyzing vehicles and license plates in video footage using YOLOv8 and the SORT tracking algorithm. It is designed for toll booth or traffic analysis scenarios, supporting vehicle counting, toll calculation, and metadata export.

## Features

- **Vehicle Detection:** Uses YOLOv8 to detect cars, motorbikes, buses, and trucks.
- **License Plate Detection:** Detects license plates within vehicle bounding boxes.
- **Tracking:** SORT algorithm assigns unique IDs to vehicles across frames.
- **Toll Calculation:** Calculates tolls based on vehicle type and tracks total toll collected.
- **CSV Export:** Saves detection and tracking metadata to CSV files.
- **Video Output:** Annotates and saves processed video with bounding boxes and stats overlays.
- **Filtering:** Supports filtering vehicles by bounding box area (e.g., "close to camera" vehicles).

## Requirements

- Python 3.8+
- CUDA-enabled GPU (for best performance)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- NumPy
- pandas
- [SORT tracker](https://github.com/abewley/sort) (as a Python module)
- Pre-trained YOLOv8 weights (`yolov8n.pt`) and a license plate detector model (`license_plate_detector.pt`)

Install dependencies:
```bash
pip install ultralytics opencv-python numpy pandas
# Add SORT tracker to your project or install as a module
```

## Usage

### 1. Prepare Models

- Download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics).
- Place your license plate detector weights as `license_plate_detector.pt` in the project directory.

### 2. Run Processing

You can run the main notebook or scripts to process a video:

- **From Notebook:** Open `create_metadata.ipynb` and run the desired cells.
- **From Command Line:** If using a script, run:
  ```bash
  python create_metadata.py --source path/to/video.mp4 --output output_video.mp4
  ```

### 3. Output

- **CSV:** Metadata for each frame and vehicle is saved (see `vehicle_detections.csv` or similar).
- **Video:** Annotated video is saved with bounding boxes, IDs, and toll info.

### 4. Customization

- Adjust `MIN_BBOX_AREA` to filter vehicles by size (e.g., for "close vehicles").
- Change `TOLL_AMOUNT` dictionary to set toll rates per vehicle type.

## File Structure

- `create_metadata.ipynb` — Main notebook for detection, tracking, and metadata export.
- `vehicle_detections.csv` — Output CSV with detection and tracking data.
- `*_tracked.mp4` — Output video with overlays.
- `README.md` — Project documentation.

## Notes

- For best results, use high-quality video and ensure the models are properly trained for your region.
- License plate recognition (OCR) is not included; detected plates are marked as "DETECTED".

## License

This project is for educational and research purposes.
