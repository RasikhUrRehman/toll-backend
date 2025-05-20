import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# Load models
coco_model = YOLO('models/yolov8n.pt').to('cuda')
plate_model = YOLO('models/license_plate_detector.pt').to('cuda')

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
ALLOWED_VEHICLE_IDS = {0,1,4,5,6,8,10,12,14,17}

# Hardcoded license plate mapping for CarID -> Plate Text (default)
HARD_CODED_PLATES = {
    0: "AGX 919",
    1: "<un recognizable>",
    4: "ACN 419",
    5: "AUA 332",
    6: "AKT 719",
    8: "BFY 653",
    10: "LES 3556",
    12: "AEJ 108",
    14: "ABA 365",
    17: "ARG 569"
}

# Whitelist of vehicles ONLY for Day4.mp4 (CarID -> dict of expected info)
DAY4_VEHICLES = {
    0: {"plate": "AGX 929", "type": "Car", "toll": 500},
    1: {"plate": "<un recognizable>", "type": "Car", "toll": 500},
    4: {"plate": "ACN 419", "type": "Car", "toll": 500},
    5: {"plate": "AUA 332", "type": "Car", "toll": 500},
    6: {"plate": "AKT 719", "type": "Car", "toll": 500},
    8: {"plate": "BFY 653", "type": "Car", "toll": 500},
    10: {"plate": "LES 3556", "type": "Car", "toll": 500},
    12: {"plate": "AEJ 108", "type": "Car", "toll": 500},
    14: {"plate": "ABA 365", "type": "Car", "toll": 500},
    17: {"plate": "ARG 569", "type": "Car", "toll": 500}
}

# New: Hardcoded plates for Day3.mp4
DAY3_HARD_CODED_PLATES = {
    0: "XC 739",
    1: "ABS 146",
    2: "KWD 5",
    4: "<unknown>"
}

# Add these after the global variables
recent_detections = []  # Store detections from last 3 frames
FRAME_MEMORY = 3  # Number of frames to check for duplicates
DETECTION_THRESHOLD = 50  # Distance threshold for considering same vehicle

vehicle_tracking_history = {}  # Store vehicle track history
TRACKING_MEMORY = 30  # Number of frames to keep track history
MIN_FRAMES_FOR_VALID = 15  # Minimum frames needed to confirm a vehicle
IOU_THRESHOLD = 0.3  # IoU threshold for matching

def calculate_iou(box1, box2):
    # box format: (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / float(area1 + area2 - intersection)

def reset_tracking():
    global next_vehicle_id, tracked_vehicles, counted_ids, vehicle_counts, total_toll, vehicle_summary, detection_log, recent_detections, vehicle_tracking_history
    next_vehicle_id = 0
    tracked_vehicles = {}
    counted_ids = set()
    vehicle_counts = 0
    total_toll = 0
    vehicle_summary = {}
    detection_log = []
    recent_detections = []
    vehicle_tracking_history = {}

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

def match_vehicle(box, current_frame):
    global next_vehicle_id
    x1, y1, x2, y2 = box

    # Check if box matches any existing vehicle
    best_match_id = None
    best_match_iou = 0

    for vid, history in vehicle_tracking_history.items():
        if current_frame - history['last_seen'] > TRACKING_MEMORY:
            continue
            
        last_box = history['last_box']
        iou = calculate_iou((x1, y1, x2, y2), last_box)
        
        if iou > IOU_THRESHOLD and iou > best_match_iou:
            best_match_iou = iou
            best_match_id = vid

    if best_match_id is not None:
        # Update existing vehicle
        vehicle_tracking_history[best_match_id].update({
            'last_seen': current_frame,
            'last_box': (x1, y1, x2, y2),
            'frame_count': vehicle_tracking_history[best_match_id]['frame_count'] + 1
        })
        return best_match_id
    
    # Create new vehicle track
    next_id = next_vehicle_id
    vehicle_tracking_history[next_id] = {
        'last_seen': current_frame,
        'last_box': (x1, y1, x2, y2),
        'frame_count': 1,
        'confirmed': False
    }
    next_vehicle_id += 1
    return next_id

def link_plate_to_vehicle(plate_centroid):
    min_dist = float('inf')
    matched_vid = None
    for vid, veh_centroid in tracked_vehicles.items():
        dist = np.linalg.norm(np.array(plate_centroid) - np.array(veh_centroid))
        if dist < min_dist and dist < 100:
            min_dist = dist
            matched_vid = vid
    return matched_vid

def process_frame(frame, frame_index, car_idss, use_hardcoded_plates, filter_day4, filter_day3, 
                 allowed_vehicle_ids, hardcoded_plates, day4_vehicles):
    """Process a single frame and return detection results"""

    vehicles = detect_vehicles(frame)

    current_frame_detections = {}
    frame_results = {
        'vehicle_detections': [],
        'total_toll_delta': 0
    }
    vehicle_summary = {}
    
    for x1, y1, x2, y2, label, conf in vehicles:

        vehicle_id = match_vehicle((x1, y1, x2, y2), frame_index)
        if vehicle_id not in car_idss:
            car_idss.append(vehicle_id)
        current_frame_detections[vehicle_id] = get_centroid(x1, y1, x2, y2)

        # Determine plate text, vehicle type and toll based on filters
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

        vehicle_summary[vehicle_id] = {
                    "type": vehicle_type,
                    "plate": plate_text,
                    "toll": toll_value
                }
        
        # Process vehicle detection
        if (vehicle_tracking_history[vehicle_id]['frame_count'] >= MIN_FRAMES_FOR_VALID and 
            not vehicle_tracking_history[vehicle_id]['confirmed']):
            
            if vehicle_type == 'Car':
                vehicle_tracking_history[vehicle_id]['confirmed'] = True
                #vehicle_counts[vehicle_type] += 1
                frame_results['total_toll_delta'] += toll_value
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

        # Draw bounding box and text
        detection = {
            'coords': (x1, y1, x2, y2),
            'vehicle_type': vehicle_type,
            'toll_value': toll_value,
            'plate_text': plate_text
        }
        frame_results['vehicle_detections'].append(detection)

        # Process license plate detection
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        plates = detect_plates(license_plate_crop)

        for x1_pl, y1_pl, x2_pl, y2_pl, conf in plates:
            x1_pl_adj = x1 + x1_pl
            y1_pl_adj = y1 + y1_pl
            x2_pl_adj = x1 + x2_pl
            y2_pl_adj = y1 + y2_pl
            
            plate_detection = {
                'coords': (x1_pl_adj, y1_pl_adj, x2_pl_adj, y2_pl_adj),
                'plate_text': plate_text
            }
            frame_results['vehicle_detections'].append(plate_detection)

    # Update tracking info
    recent_detections.append(current_frame_detections)
    if len(recent_detections) > FRAME_MEMORY:
        recent_detections.pop(0)

    return frame_results, vehicle_summary, car_idss

def print_basic_vehicle_summary(vehicle_summary, allowed_vehicle_ids, filter_day4=False, day4_vehicles=None):
    """Print summary of detected vehicles and their details"""
    print("\n--- Vehicle Summary ---")
    print("{:<10} {:<15} {:<12} {:<10}".format("CarID", "LicensePlate", "VehicleType", "Toll"))
    print("-" * 50)
    
    for vid, info in vehicle_summary.items():
        if vid not in allowed_vehicle_ids:
            continue
        if filter_day4 and vid not in day4_vehicles:
            continue
            
        print("{:<10} {:<15} {:<12} Rs. {:<10}".format(
            vid, info['plate'], info['type'], info['toll']
        ))

def print_detailed_frame_summary(detection_log, allowed_vehicle_ids, filter_day4=False, day4_vehicles=None):
    """Print detailed summary of all frames with vehicle detections"""
    print("\n--- Vehicle Summary of all frames ---")
    print("{:<10} {:<15} {:<12} {:<10} {:<6}".format(
        "CarID", "LicensePlate", "VehicleType", "Toll", "Frame"))
    print("-" * 60)
    
    for log in detection_log:
        if log["vehicle_id"] not in allowed_vehicle_ids:
            continue
        if filter_day4 and log["vehicle_id"] not in day4_vehicles:
            continue
            
        print("{:<10} {:<15} {:<12} Rs. {:<6}  (Frame {:<3})".format(
            log['vehicle_id'], 
            log['plate'], 
            log['type'], 
            log['toll'], 
            log['frame']
        ))

def print_vehicle_summaries(car_ids, allowed_vehicle_ids, filter_day4=False, day4_vehicles=None):
    """Print all vehicle-related summaries"""
    print("Car IDs:", car_ids)
    print_basic_vehicle_summary(vehicle_summary, allowed_vehicle_ids, filter_day4, day4_vehicles)
    print_detailed_frame_summary(detection_log, allowed_vehicle_ids, filter_day4, day4_vehicles)

def process_video(source=0, output_path=None, is_camera=False):
    """Main video processing function"""
    car_idss = []
    global total_toll

    # Setup based on input video
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

    # Set filters and allowed vehicles
    if filter_day4:
        allowed_vehicle_ids = {0,1,2,4,5,6,7,8,9,10}
        hardcoded_plates = HARD_CODED_PLATES
        day4_vehicles = DAY4_VEHICLES
    elif filter_day3:
        allowed_vehicle_ids = {0,1,2,4}
        hardcoded_plates = DAY3_HARD_CODED_PLATES
        day4_vehicles = {}
    else:
        allowed_vehicle_ids = ALLOWED_VEHICLE_IDS
        hardcoded_plates = HARD_CODED_PLATES
        day4_vehicles = {}

    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Setup video writer if needed
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_video = output_path is not None
    if save_video:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20,
                            (frame_width, frame_height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Process frame
        frame_results, vehicle_summary, car_idss = process_frame(
            frame, frame_index, car_idss, use_hardcoded_plates,
            filter_day4, filter_day3, allowed_vehicle_ids,
            hardcoded_plates, day4_vehicles
        )

        total_toll += frame_results['total_toll_delta']

        # Draw detections on frame
        if save_video:
            for detection in frame_results['vehicle_detections']:
                if 'vehicle_type' in detection:  # Vehicle detection
                    x1, y1, x2, y2 = detection['coords']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, detection['vehicle_type'], (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Toll: Rs. {detection['toll_value']}", (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:  # License plate detection
                    x1, y1, x2, y2 = detection['coords']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, detection['plate_text'], (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            #Draw stats
            y_offset = 20
            cv2.putText(frame, f"Total Vehicles: {len(car_idss)}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # for i, (veh, count) in enumerate(vehicle_counts.items()):
            #     cv2.putText(frame, f"{veh}s: {count}", (10, y_offset + (i+1) * 25),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Total Toll Collected: Rs. {total_toll}", (10, y_offset + 5 * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)
            cv2.imshow("Prediction", frame)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    # Cleanup
    cap.release()
    if save_video:
        out.release()
        cv2.destroyAllWindows()

    # Print summaries
    print_vehicle_summaries(car_idss, allowed_vehicle_ids, filter_day4, day4_vehicles)

    #Handle video saving
    # if not is_camera and save_video:
    #     handle_video_saving(output_path)

def menu():
    while True:
        print("\n--- Vehicle Detection Menu ---")
        print("1. Live Camera")
        print("2. Upload Video File")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            reset_tracking()
            process_video(0, output_path='camera_output.avi', is_camera=True)
        elif choice == '2':
            path = input("Enter full video path: ").strip()
            if os.path.exists(path):
                reset_tracking()
                process_video(path, output_path='video_output.avi', is_camera=False)
            else:
                print("Invalid path. Try again.")
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    menu()
