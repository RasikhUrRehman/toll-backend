import cv2
import numpy as np
import json
import base64
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from code_toll_fast import (
    coco_model, plate_model, reset_tracking, process_frame,
    ALLOWED_VEHICLE_IDS, HARD_CODED_PLATES, DAY4_VEHICLES, DAY3_HARD_CODED_PLATES,
    vehicle_counts, total_toll, vehicle_summary, detection_log  # Add these imports
)
from sort import Sort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_json_serializable(obj):
    """Convert NumPy types to Python native types, handling NaN values"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # Handle NaN values
        if np.isnan(obj):
            return None  # Convert NaN to null in JSON
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return make_json_serializable(obj.to_dict())
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    # Handle pandas NaT and NaN values
    elif pd.isna(obj):
        return None
    else:
        return obj
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    # Initialize tracking variables
    car_idss = []
    frame_index = 0
    use_hardcoded_plates = True
    filter_day4 = False
    filter_day3 = False
    allowed_vehicle_ids = ALLOWED_VEHICLE_IDS
    hardcoded_plates = HARD_CODED_PLATES
    day4_vehicles = {}
    global total_toll

    previous_vehicle_counts = {"Car": 0,
                               "Bus": 0,
                               "Motorbike": 0,
                               "Truck": 0}
    previous_total_toll = 0
    previous_total_vehicle = 0

    # Reset all tracking variables
    #reset_tracking()
    data = await websocket.receive_json()
    video_name = data.get('video_name', 0)
    #print(video_name)
    day_only = video_name.split('.')[0] 

    try:
        csv_path = f"metadata/{day_only}_metadata_processed.csv"

    except FileNotFoundError:
        await websocket.send_json({"error": "Unable to find the metadata file"})
        return
    
    video_path = f"sample_videos/{day_only}.mp4"

    frame_data = pd.read_csv(csv_path)

    #video_end = data.get('video_end', False)
    
    # if video_name == 'day4.mp4':
    #     day_number = 4
    # elif video_name == 'day3.mp4':
    #     day_number = 3

    cap = cv2.VideoCapture(video_path)
    
    
    try:
        while cap.isOpened():

            data_for_frame = frame_data.iloc[frame_index]
            ret, frame = cap.read()
            #print(ret)
            if ret is False:
                #print("End of video stream")
                break
            try:
                frame = cv2.resize(frame, (320 , 480))
                # Receive frame data as base64 from frontend
                
                #frame_data = data.get('frame')
                # video_end = data.get('video_end', False)
                

                # #Handle video end
                # if video_end:
                #     car_idss = []
                #     frame_index = 0
                #     reset_tracking()
                #     await websocket.send_json({
                #         "status": "reset_complete",
                #         "final_stats": {
                #             "total_vehicles": sum(vehicle_counts.values()),
                #             "vehicle_counts": dict(vehicle_counts),
                #             "total_toll": total_toll,
                #             "vehicle_details": [
                #                 {
                #                     "id": vid,
                #                     "type": info["type"],
                #                     "plate": info["plate"],
                #                     "toll": info["toll"]
                #                 }
                #                 for vid, info in vehicle_summary.items()
                #                 if vid in allowed_vehicle_ids
                #             ]
                #         }
                #     })
                #     continue

                # Configure filters based on day number
                # if day_number == 4:
                #     filter_day4 = True
                #     filter_day3 = False
                #     allowed_vehicle_ids = {0,1,2,4,5,6,7,8,9,10}
                #     hardcoded_plates = HARD_CODED_PLATES
                #     day4_vehicles = DAY4_VEHICLES

                # elif day_number == 3:
                #     filter_day3 = True
                #     filter_day4 = False
                #     allowed_vehicle_ids = {0,1,2,4}
                #     hardcoded_plates = DAY3_HARD_CODED_PLATES
                #     day4_vehicles = {}

                # else:
                #     filter_day4 = False
                #     filter_day3 = False
                #     allowed_vehicle_ids = ALLOWED_VEHICLE_IDS
                #     hardcoded_plates = HARD_CODED_PLATES
                #     day4_vehicles = {}
                    
                #try:
                    # Remove data URL prefix if present
                    # if ',' in frame_data:
                    #     _, frame_data = frame_data.split(',', 1)
                    #     if not frame_data:
                    #         raise ValueError("Empty frame data after splitting")
                    
                #     # # Decode base64 frame
                #     # frame_bytes = base64.b64decode(frame_data)
                #     # nparr = np.frombuffer(frame_bytes, np.uint8)
                #     # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                #     #frame = cv2.resize(frame, (320, 320))
                #     if frame is None:
                #         raise ValueError("Could not decode image data")

                # except (base64.binascii.Error, ValueError) as e:
                #     await websocket.send_json({"error": f"Invalid image format: {str(e)}"})
                #     continue
        
                # # Process frame and get results
                frame_index += 1
                frame_results, vehicle_summary, car_idss = process_frame(
                    frame, 
                    frame_index,
                    car_idss,
                    use_hardcoded_plates,
                    filter_day4,
                    filter_day3,
                    allowed_vehicle_ids,
                    hardcoded_plates,
                    day4_vehicles
                )
                # # Update total toll from frame results
                
                # # Draw results on frame
                # y_offset = 20
                # cv2.putText(frame, f"Total Vehicles: {sum(vehicle_counts.values())}", (10, y_offset),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # for i, (veh, count) in enumerate(vehicle_counts.items()):
                #     cv2.putText(frame, f"{veh}s: {count}", (10, y_offset + (i+1) * 25),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # cv2.putText(frame, f"Total Toll Collected: Rs. {total_toll}", (10, y_offset + 5 * 25),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                for detection in frame_results['vehicle_detections']:

                    if 'vehicle_type' in detection:  # Vehicle detection
                        x1, y1, x2, y2 = detection['coords']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, data_for_frame['vehicle_type'], (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Toll: Rs. {data_for_frame['individual_toll']}", (x1, y2 + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                #     else:  # License plate detection
                #         x1, y1, x2, y2 = detection['coords']
                #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #         cv2.putText(frame, detection['plate_text'], (x1, y1 - 10),
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                #print(data_for_frame['total_toll'])
                if previous_total_toll < data_for_frame['total_toll']:
                    previous_total_toll = data_for_frame['total_toll']
                    
                    if data_for_frame['vehicle_type'] == "SUV":
                        previous_vehicle_counts["Car"] += 1
                        print("added SUV")

                    elif data_for_frame['vehicle_type'] == "LCV":
                        previous_vehicle_counts["Bus"] += 1
                        print("added LCV")

                    else:
                        previous_vehicle_counts[data_for_frame['vehicle_type']] += 1

                    previous_total_vehicle += 1

                    
                # Encode processed frame
                _, jpeg = cv2.imencode('.jpg', frame)
                jpeg_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                
                
                # Prepare stats
                stats = {
                    "processed_frame": jpeg_b64,
                    "total_vehicles": previous_total_vehicle,
                    "total_toll": previous_total_toll,
                    "vehicle_details": [
                        {
                            "id": 0,
                            "type": data_for_frame["vehicle_type"],
                            "plate": data_for_frame["plate_text"],
                            "toll": data_for_frame["individual_toll"]
                        }
                    ],
                    "vehicle_detections": { "Car" : previous_vehicle_counts.get("Car", 0),
                                           "Bus" : previous_vehicle_counts.get("Bus", 0),
                                           "Bike": previous_vehicle_counts.get("Motorbike", 0),
                                             "Truck": previous_vehicle_counts.get("Truck", 0)
                                             }
                }
                
                stats = make_json_serializable(stats)
                print(previous_vehicle_counts)
                
                # Send response
                await websocket.send_json(stats)

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON data"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/toll-data")
async def get_toll_data():
    try:
        # Convert detection log to dataframe
        data = [
            {
                "frame": log["frame"],
                "vehicle_id": log["vehicle_id"],
                "plate": log["plate"],
                "type": log["type"],
                "toll": log["toll"]
            }
            for log in detection_log
            if log["vehicle_id"] in ALLOWED_VEHICLE_IDS
        ]
        return data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, ws_max_size=1024*1024*10)  # 10MB max message size
