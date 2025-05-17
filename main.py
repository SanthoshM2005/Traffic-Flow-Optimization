import cv2
import torch
import numpy as np
import pandas as pd
from sort import Sort

# Load YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4
model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Load your video
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

# SORT Tracker
tracker = Sort()

# Virtual line parameters
line_position = 250  # y-coordinate
offset = 6  # tolerance for counting
counted_ids = set()
vehicle_count = 0
frame_number = 0

# Output CSV data
count_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Detect vehicles
    results = model(frame)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append([x1, y1, x2, y2, float(conf)])

    detections_np = np.array(detections)
    if len(detections_np) == 0:
        detections_np = np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections_np)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Line crossing check
        if line_position - offset < cy < line_position + offset:
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1
                count_data.append({"Frame": frame_number, "Vehicle_ID": int(track_id)})
                print(f"Vehicle {int(track_id)} counted at frame {frame_number}")

    # Draw virtual line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)
    cv2.putText(frame, f'Total Vehicles: {vehicle_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Vehicle Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Export CSV
df = pd.DataFrame(count_data)
df.to_csv("vehicle_count.csv", index=False)
print("Saved vehicle_count.csv")

cap.release()
cv2.destroyAllWindows()
