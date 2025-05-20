import cv2
import csv
import argparse
import os

def load_csv_data(csv_file):
    """
    Load CSV data into a dictionary indexed by frame number
    """
    frame_data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['frame'])
            if frame_num not in frame_data:
                frame_data[frame_num] = []
            frame_data[frame_num].append(row)
    return frame_data

def process_single_frame(frame, frame_number, frame_data_for_frame, previous_total_toll=0, previous_vehicle_counts=None):
    """
    Process a single frame of video
    
    Args:
        frame: The video frame to process
        frame_number: The current frame number
        frame_data_for_frame: The CSV data for this specific frame
        previous_total_toll: The total toll from the previous frame
        previous_vehicle_counts: Dictionary of vehicle counts from previous frame
        
    Returns:
        processed_frame: The frame with visualizations added
        logs: A list of log messages generated during processing
        frame_summary: A dictionary containing summary information about the frame
        current_total_toll: The total toll from the current frame
        current_vehicle_counts: Dictionary of vehicle counts for this frame
    """
    logs = []
    
    # Create a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Add frame number to the frame
    cv2.putText(processed_frame, f"Frame: {frame_number}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initialize vehicle counts dictionary
    if previous_vehicle_counts is None:
        previous_vehicle_counts = {
            "Car": 0,
            "Bus": 0,
            "Bike": 0,
            "Truck": 0
        }
    
    # Copy previous vehicle counts for current frame
    vehicle_type_counts = previous_vehicle_counts.copy()
    
    # Prepare the summary data structure
    vehicle_ids = []
    vehicle_summary = {}
    
    # Get current total toll from CSV if available
    current_total_toll = 0
    if frame_data_for_frame and 'total_toll' in frame_data_for_frame[0]:
        current_total_toll = float(frame_data_for_frame[0]['total_toll'])
    
    # Check if toll decreased (a vehicle went through)
    toll_decreased = previous_total_toll > current_total_toll and current_total_toll > 0
    
    # If toll decreased, find which vehicle type passed and increment its count
    if toll_decreased and frame_data_for_frame:
        # Get the vehicle type of the most recently detected vehicle
        for vehicle_info in frame_data_for_frame:
            if vehicle_info['detected'] == '1':
                vehicle_type = vehicle_info['vehicle_type']
                if vehicle_type in vehicle_type_counts:
                    vehicle_type_counts[vehicle_type] += 1
                else:
                    # Default to Car if unknown type
                    vehicle_type_counts["Car"] += 1
                logs.append(f"Frame {frame_number}: Detected vehicle of type {vehicle_type} passed through")
                break
    
    # If no data for this frame, return early
    if not frame_data_for_frame:
        frame_summary = {
            "total_vehicles": sum(vehicle_type_counts.values()),
            "total_toll": current_total_toll,
            "vehicle_details": [],
            "vehicle_detections": vehicle_type_counts
        }
        return processed_frame, logs, frame_summary, current_total_toll, vehicle_type_counts
    
    # Process each vehicle in this frame
    for vehicle_info in frame_data_for_frame:
        vehicle_id = vehicle_info['vehicle_id']
        vehicle_type = vehicle_info['vehicle_type']
        
        # Track vehicle IDs and populate summary dictionary
        if vehicle_id not in vehicle_ids and vehicle_info['detected'] == '1':
            vehicle_ids.append(vehicle_id)
            
            # Get toll from CSV
            individual_toll = float(vehicle_info['individual_toll']) if 'individual_toll' in vehicle_info else 0
            
            # Store vehicle details for summary
            vehicle_summary[vehicle_id] = {
                "type": vehicle_type,
                "plate": vehicle_info.get('plate_text', ''),
                "toll": individual_toll
            }
        
        # Draw vehicle bounding box
        if vehicle_info['detected'] == '1':
            try:
                v_x1 = int(float(vehicle_info['v_x1']))
                v_y1 = int(float(vehicle_info['v_y1']))
                v_x2 = int(float(vehicle_info['v_x2']))
                v_y2 = int(float(vehicle_info['v_y2']))
                cv2.rectangle(processed_frame, (v_x1, v_y1), (v_x2, v_y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Vehicle: {vehicle_info['vehicle_type']}", (v_x1, v_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                logs.append(f"Frame {frame_number}: Vehicle {vehicle_info['vehicle_id']} of type {vehicle_info['vehicle_type']} detected")
            except (ValueError, KeyError) as e:
                error_msg = f"Error with vehicle bounding box in frame {frame_number}: {e}"
                logs.append(error_msg)
        
        # Draw license plate bounding box
        if vehicle_info['plate_detected'] == '1':
            try:
                p_x1 = int(float(vehicle_info['p_x1']))
                p_y1 = int(float(vehicle_info['p_y1']))
                p_x2 = int(float(vehicle_info['p_x2']))
                p_y2 = int(float(vehicle_info['p_y2']))
                cv2.rectangle(processed_frame, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Plate: {vehicle_info['plate_text']}", (p_x1, p_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                logs.append(f"Frame {frame_number}: License plate {vehicle_info['plate_text']} detected")
            except (ValueError, KeyError) as e:
                error_msg = f"Error with license plate bounding box in frame {frame_number}: {e}"
                logs.append(error_msg)
        
        # Display toll information
        try:
            toll_info = f"Toll: ${vehicle_info['individual_toll']} (Total: ${vehicle_info['total_toll']})"
            # Use vehicle coords for toll info if available, otherwise use default position
            if 'detected' in vehicle_info and vehicle_info['detected'] == '1' and 'v_x1' in vehicle_info and 'v_y2' in vehicle_info:
                try:
                    x_pos = int(float(vehicle_info['v_x1']))
                    y_pos = int(float(vehicle_info['v_y2'])) + 20
                except (ValueError, KeyError):
                    x_pos, y_pos = 10, 60
            else:
                x_pos, y_pos = 10, 60
            
            cv2.putText(processed_frame, toll_info, (x_pos, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            logs.append(f"Frame {frame_number}: Toll applied - Individual: ${vehicle_info['individual_toll']}, Total: ${vehicle_info['total_toll']}")
        except (ValueError, KeyError) as e:
            error_msg = f"Error displaying toll information in frame {frame_number}: {e}"
            logs.append(error_msg)
    
    # Define ALLOWED_VEHICLE_IDS - here we allow all vehicles
    ALLOWED_VEHICLE_IDS = vehicle_ids
    
    # Create the frame summary dictionary with the requested structure
    frame_summary = {
        "total_vehicles": sum(vehicle_type_counts.values()),
        "total_toll": current_total_toll,
        "vehicle_details": [
            {
                "id": vid,
                "type": info["type"],
                "plate": info["plate"],
                "toll": info["toll"]
            }
            for vid, info in vehicle_summary.items()
            if vid in ALLOWED_VEHICLE_IDS
        ],
        "vehicle_detections": vehicle_type_counts
    }
    
    return processed_frame, logs, frame_summary, current_total_toll, vehicle_type_counts

def process_video(video_path, csv_path, output_path=None, log_path=None):
    """
    Process a video file using data from a CSV file
    """
    # Load CSV data
    frame_data = load_csv_data(csv_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Setup log file if log path is provided
    log_file = None
    if log_path:
        log_file = open(log_path, 'w')
        log_file.write("Frame,Log\n")  # CSV header
    
    # Collect all logs and frame summaries
    all_logs = []
    all_frame_summaries = []
    
    frame_number = 0
    previous_total_toll = 0
    previous_vehicle_counts = {
        "Car": 0,
        "Bus": 0,
        "Bike": 0,
        "Truck": 0
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the data for the current frame, if any
        frame_data_for_frame = frame_data.get(frame_number, [])
        
        # Process the single frame
        processed_frame, frame_logs, frame_summary, current_total_toll, current_vehicle_counts = process_single_frame(
            frame, frame_number, frame_data_for_frame, previous_total_toll, previous_vehicle_counts
        )
        
        # Update previous values for next frame
        previous_total_toll = current_total_toll
        previous_vehicle_counts = current_vehicle_counts
        
        # Collect logs and summary
        all_logs.extend(frame_logs)
        all_frame_summaries.append(frame_summary)
        
        # Print logs to console
        for log in frame_logs:
            print(log)
        
        # Write logs to file if needed
        if log_file:
            for log in frame_logs:
                log_file.write(f"{frame_number},{log}\n")
        
        # Show the processed frame
        cv2.imshow('Toll System Video', processed_frame)
        
        # Write the frame if we're saving the output
        if writer:
            writer.write(processed_frame)
        
        # Increment frame counter
        frame_number += 1
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    if log_file:
        log_file.close()
    cv2.destroyAllWindows()
    
    return all_logs, all_frame_summaries

def main():
    parser = argparse.ArgumentParser(description='Process toll system video with CSV data')
    parser.add_argument('video_path', help='Path to the MP4 video file')
    parser.add_argument('csv_path', help='Path to the CSV data file')
    parser.add_argument('--output', '-o', help='Path to save the processed video (optional)')
    parser.add_argument('--log', '-l', help='Path to save processing logs (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return
    
    logs, frame_summaries = process_video(args.video_path, args.csv_path, args.output, args.log)
    
    print(f"Processing complete. Processed {len(logs)} events across {len(frame_summaries)} frames.")
    
    # Additional summary information
    total_vehicles = sum(summary["total_vehicles"] for summary in frame_summaries)
    total_toll = sum(summary["total_toll"] for summary in frame_summaries)
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Total toll collected: ${total_toll:.2f}")
    
    if args.output:
        print(f"Processed video saved to {args.output}")
    if args.log:
        print(f"Processing logs saved to {args.log}")

if __name__ == "__main__":
    main()
