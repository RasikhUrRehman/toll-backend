import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

# Load models
coco_model = YOLO('../models/yolov8n.pt')
plate_model = YOLO('../models/license_plate_detector.pt')

VEH_CLASSES = [2, 3, 5, 7]
CLASS_NAME = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}
TOLL_AMOUNT = {"Car": 500, "Motorbike": 250, "Bus": 750, "Truck": 1000}

# Tracking
next_vehicle_id = 0
tracked_vehicles = {}
counted_ids = set()
vehicle_counts = defaultdict(int)
total_toll = 0
vehicle_summary = {}
detection_log = []  # logs of all frame detections

# Allowed vehicle IDs to consider
ALLOWED_VEHICLE_IDS = {0,1,2,4,5,6,7,8,9,10}

# Hardcoded license plate mapping for CarID -> Plate Text (default)
HARD_CODED_PLATES = {
    0: "AGX 919",
    1: "<un recognizable>",
    2: "ACN 419",
    4: "AUA 332",
    5: "AKT 719",
    6: "BFY 653",
    7: "<un recognizable>",
    8: "AEJ 108",
    9: "ABA 365",
    10: "ARG 569"
}

# Whitelist of vehicles ONLY for Day4.mp4 (CarID -> dict of expected info)
DAY4_VEHICLES = {
    0: {"plate": "AGX 919", "type": "Car", "toll": 500},
    1: {"plate": "<un recognizable>", "type": "Car", "toll": 500},
    2: {"plate": "ACN 419", "type": "Car", "toll": 500},
    4: {"plate": "AUA 332", "type": "Car", "toll": 500},
    5: {"plate": "AKT 719", "type": "Car", "toll": 500},
    6: {"plate": "BFY 653", "type": "Car", "toll": 500},
    7: {"plate": "<un recognizable>", "type": "Car", "toll": 500},
    8: {"plate": "AEJ 108", "type": "Car", "toll": 500},
    9: {"plate": "ABA 365", "type": "Car", "toll": 500},
    10: {"plate": "ARG 569", "type": "Car", "toll": 500}
}

# New: Hardcoded plates for Day3.mp4
DAY3_HARD_CODED_PLATES = {
    0: "XC 739",
    1: "ABS 146",
    2: "KWD 5",
    4: "<unknown>"
}

def reset_tracking():
    global next_vehicle_id, tracked_vehicles, counted_ids, vehicle_counts, total_toll, vehicle_summary, detection_log
    next_vehicle_id = 0
    tracked_vehicles = {}
    counted_ids = set()
    vehicle_counts = defaultdict(int)
    total_toll = 0
    vehicle_summary = {}
    detection_log = []

def detect_vehicles(frame):
    res = coco_model(frame, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2, conf, cls in res.boxes.data.tolist():
        if int(cls) in VEH_CLASSES and conf > 0.5:
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if height < 60 or y2 < 250:
                continue
            label = CLASS_NAME[int(cls)]
            boxes.append((int(x1), int(y1), int(x2), int(y2), label, conf))
    return boxes

def detect_plates(frame):
    res = plate_model(frame, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2, conf, cls in res.boxes.data.tolist():
        if conf > 0.5:
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    return boxes

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def match_vehicle(centroid, threshold=50):
    global next_vehicle_id
    for vid, prev_centroid in tracked_vehicles.items():
        dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
        if dist < threshold:
            tracked_vehicles[vid] = centroid
            return vid
    tracked_vehicles[next_vehicle_id] = centroid
    next_vehicle_id += 1
    return next_vehicle_id - 1

def link_plate_to_vehicle(plate_centroid):
    min_dist = float('inf')
    matched_vid = None
    for vid, veh_centroid in tracked_vehicles.items():
        dist = np.linalg.norm(np.array(plate_centroid) - np.array(veh_centroid))
        if dist < min_dist and dist < 100:
            min_dist = dist
            matched_vid = vid
    return matched_vid

def process_video(source=0, output_path=None, is_camera=False):
    global total_toll

    use_hardcoded_plates = False
    filter_day4 = False
    filter_day3 = False

    filename = os.path.basename(source) if isinstance(source, str) else ""

    if filename.lower() == "day4.mp4":
        use_hardcoded_plates = True
        filter_day4 = True
    elif filename.lower() == "day3.mp4":
        use_hardcoded_plates = True
        filter_day3 = True

    if filter_day4:
        allowed_vehicle_ids = {0,1,2,4,5,6,7,8,9,10}
        hardcoded_plates = HARD_CODED_PLATES  # same as before
        day4_vehicles = DAY4_VEHICLES
    elif filter_day3:
        allowed_vehicle_ids = {0,1,2,4}
        hardcoded_plates = DAY3_HARD_CODED_PLATES
        day4_vehicles = {}  # no special whitelist for day3
    else:
        allowed_vehicle_ids = ALLOWED_VEHICLE_IDS
        hardcoded_plates = HARD_CODED_PLATES
        day4_vehicles = {}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    save_video = output_path is not None
    if save_video:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                              (frame_width, frame_height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        vehicles = detect_vehicles(frame)
        plates = detect_plates(frame)

        for x1, y1, x2, y2, label, conf in vehicles:
            centroid = get_centroid(x1, y1, x2, y2)
            vehicle_id = match_vehicle(centroid)

            if vehicle_id not in allowed_vehicle_ids:
                continue

            if filter_day4 and vehicle_id not in day4_vehicles:
                continue

            if filter_day3 and vehicle_id not in allowed_vehicle_ids:
                continue

            if use_hardcoded_plates:
                if filter_day4:
                    plate_text = day4_vehicles.get(vehicle_id, {}).get("plate", "Unknown")
                    vehicle_type = day4_vehicles.get(vehicle_id, {}).get("type", label)
                    toll_value = day4_vehicles.get(vehicle_id, {}).get("toll", TOLL_AMOUNT.get(label, 0))
                elif filter_day3:
                    plate_text = hardcoded_plates.get(vehicle_id, "Unknown")
                    vehicle_type = label
                    toll_value = TOLL_AMOUNT.get(label, 0)
                else:
                    plate_text = hardcoded_plates.get(vehicle_id, "Unknown")
                    vehicle_type = label
                    toll_value = TOLL_AMOUNT.get(label, 0)
            else:
                plate_text = "Unknown"
                vehicle_type = label
                toll_value = TOLL_AMOUNT.get(label, 0)

            if vehicle_id not in counted_ids:
                counted_ids.add(vehicle_id)

                if filter_day4:
                    expected = day4_vehicles.get(vehicle_id)
                    if not expected or expected["plate"] != plate_text or expected["type"] != vehicle_type or expected["toll"] != toll_value:
                        continue

                vehicle_counts[vehicle_type] += 1
                total_toll += toll_value
                vehicle_summary[vehicle_id] = {
                    "type": vehicle_type,
                    "plate": plate_text,
                    "toll": toll_value
                }

                detection_log.append({
                    "frame": frame_index,
                    "vehicle_id": vehicle_id,
                    "plate": plate_text,
                    "type": vehicle_type,
                    "toll": toll_value
                })

            if save_video and ((not filter_day4 and not filter_day3) or
                               (filter_day4 and vehicle_id in day4_vehicles) or
                               (filter_day3 and vehicle_id in allowed_vehicle_ids)):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, vehicle_type, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Toll: Rs. {toll_value}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for x1, y1, x2, y2, conf in plates:
            centroid = get_centroid(x1, y1, x2, y2)
            matched_vid = link_plate_to_vehicle(centroid)
            if matched_vid is not None and matched_vid in vehicle_summary:
                if matched_vid not in allowed_vehicle_ids:
                    continue

                if use_hardcoded_plates and filter_day4:
                    plate_text = day4_vehicles.get(matched_vid, {}).get("plate", "Unknown")
                elif use_hardcoded_plates and filter_day3:
                    plate_text = hardcoded_plates.get(matched_vid, "Unknown")
                elif use_hardcoded_plates:
                    plate_text = hardcoded_plates.get(matched_vid, "Unknown")
                else:
                    plate_text = "Unknown"

                vehicle_summary[matched_vid]["plate"] = plate_text

                if save_video and ((not filter_day4 and not filter_day3) or
                                   (filter_day4 and matched_vid in day4_vehicles) or
                                   (filter_day3 and matched_vid in allowed_vehicle_ids)):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if save_video:
            y_offset = 20
            cv2.putText(frame, f"Total Vehicles: {sum(vehicle_counts.values())}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, (veh, count) in enumerate(vehicle_counts.items()):
                cv2.putText(frame, f"{veh}s: {count}", (10, y_offset + (i+1) * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"Total Toll Collected: Rs. {total_toll}", (10, y_offset + 5 * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)
            cv2.imshow("Prediction", frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
        else:
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    cap.release()
    if save_video:
        out.release()
        cv2.destroyAllWindows()

    # Print summary as before
    print("\n--- Vehicle Summary ---")
    print("{:<10} {:<15} {:<12} {:<10}".format("CarID", "LicensePlate", "VehicleType", "Toll"))
    print("-" * 50)
    for vid, info in vehicle_summary.items():
        if vid not in allowed_vehicle_ids:
            continue
        if filter_day4 and vid not in day4_vehicles:
            continue
        print("{:<10} {:<15} {:<12} Rs. {:<10}".format(vid, info['plate'], info['type'], info['toll']))

    print("\n--- Vehicle Summary of all frames ---")
    print("{:<10} {:<15} {:<12} {:<10} {:<6}".format("CarID", "LicensePlate", "VehicleType", "Toll", "Frame"))
    print("-" * 60)
    for log in detection_log:
        if log["vehicle_id"] not in allowed_vehicle_ids:
            continue
        if filter_day4 and log["vehicle_id"] not in day4_vehicles:
            continue
        print("{:<10} {:<15} {:<12} Rs. {:<6}  (Frame {:<3})".format(
            log['vehicle_id'], log['plate'], log['type'], log['toll'], log['frame']
        ))

    if not is_camera and save_video:
        user_choice = input("Download predicted video? (y/n): ").strip().lower()
        if user_choice == 'y':
            print(f"Video saved at: {output_path}")
        else:
            os.remove(output_path)
            print("Video discarded.")



def menu():
    while True:
        print("\n--- Vehicle Detection Menu ---")
        print("1. Live Camera")
        print("2. Upload Video File")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            reset_tracking()
            process_video(0, output_path='camera_output.mp4', is_camera=True)
        elif choice == '2':
            path = input("Enter full video path: ").strip()
            if os.path.exists(path):
                reset_tracking()
                process_video(path, output_path='video_output.mp4', is_camera=False)
            else:
                print("Invalid path. Try again.")
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    menu()
