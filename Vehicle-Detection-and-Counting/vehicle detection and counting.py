import cv2
from sort import *
import numpy as np
from ultralytics import YOLO
import cvzone
import os
import json
from datetime import timedelta
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))

# Prefer the large MP4 if available, otherwise fall back to the smaller sample video
video_file = os.path.join(script_dir, 'video1.mp4')
if not os.path.exists(video_file):
    video_file = os.path.join(script_dir, 'video.mp4')

print(f"Opening video: {video_file}")
print(f"Exists: {os.path.exists(video_file)}")

cap = cv2.VideoCapture(video_file, cv2.CAP_MSMF)
print('Using capture backend: CAP_MSMF')
if not cap.isOpened():
    print('CAP_MSMF failed, falling back to default backend')
    cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    raise RuntimeError(f"Unable to open video file: {video_file}")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video opened. FPS={fps}, Frame count={cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

model = YOLO(os.path.join(script_dir, 'yolov8n.pt'))

classnames = []
with open(os.path.join(script_dir, 'classes.txt'), 'r') as f:
    classnames = f.read().splitlines()

tracker = Sort(max_age=20)

crossed_ids = set()
counts = {}  # Dynamically track all vehicle types
CONFIDENCE_THRESHOLD = 0.25
output_json = os.path.join(script_dir, 'vehicle_counts.json')
output_details_json = os.path.join(script_dir, 'vehicle_details.json')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# Store detailed vehicle information
vehicle_details = []
tracked_vehicles = {}

def get_dominant_color(frame, x1, y1, x2, y2):
    """Extract dominant color from center of bounding box region"""
    try:
        # Only sample from the center 60% of the bounding box to avoid edges/background
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        center_x1 = max(0, x1 + margin_x)
        center_y1 = max(0, y1 + margin_y)
        center_x2 = min(frame.shape[1], x2 - margin_x)
        center_y2 = min(frame.shape[0], y2 - margin_y)
        
        roi = frame[center_y1:center_y2, center_x1:center_x2]
        if roi.size == 0:
            return "Unknown"
        
        # Get average color from center region
        avg_color = np.mean(roi, axis=(0, 1)).astype(int)
        b, g, r = avg_color
        
        # Better color classification
        if r > 180 and g < 100 and b < 100:
            return "Red"
        elif g > 180 and r < 100 and b < 100:
            return "Green"
        elif b > 180 and r < 100 and g < 100:
            return "Blue"
        elif r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 80 and g < 80 and b < 80:
            return "Black"
        elif r > 200 and g > 150 and b < 100:
            return "Orange"
        elif r > 180 and g > 180 and b < 100:
            return "Yellow"
        elif r > 150 and b > 150 and g < 100:
            return "Purple"
        elif r < 100 and g < 100 and b > 150:
            return "Dark Blue"
        elif 100 < r < 180 and 100 < g < 180 and 100 < b < 180:
            return "Gray"
        else:
            return "Other"
    except:
        return "Unknown"

def get_time_string(frame_number, fps):
    """Convert frame number to time string"""
    if fps > 0:
        seconds = frame_number / fps
        return str(timedelta(seconds=int(seconds)))
    return "N/A"

def classify_vehicle_type(bbox, detected_class):
    """Map misclassified vehicles to correct types based on bounding box dimensions."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / (height + 1)

    if detected_class == 'truck':
        # Small trucks with high aspect ratio are likely autos (three-wheelers)
        if width < 350 and height < 220 and aspect_ratio > 1.3:
            return 'auto'
        # Very small trucks are autos
        if width < 280 and height < 160:
            return 'auto'

    if detected_class == 'car':
        # Small cars with specific dimensions might be autos
        if width < 280 and height < 170 and aspect_ratio > 1.2:
            return 'auto'

    # Ensure bikes are classified as 'motorbike' or 'bicycle'
    if detected_class == 'motorbike' or detected_class == 'bicycle':
        return detected_class

    # Buses should remain as bus
    if detected_class == 'bus':
        return 'bus'

    return detected_class

cv2.namedWindow('Vehicle Counter', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width = frame.shape[:2]
    # Resize large frames for faster display and smoother playback
    if width > 1280:
        scale = 1280 / width
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        height, width = frame.shape[:2]

    # move the counting line much lower, closer to the road surface
    line_y = int(height * 0.92)
    line = [int(width * 0.05), line_y, int(width * 0.95), line_y]

    detections = np.zeros((0, 5), dtype=float)
    detection_info = []  # Store detection info with class and bbox

    result = model(frame, stream=True, verbose=False)
    for info in result:
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            classindex = int(box.cls[0])
            objectdetect = classnames[classindex]

            if conf > CONFIDENCE_THRESHOLD:  # Lowered threshold to detect more vehicles and people
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf * 100])))
                detection_info.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': objectdetect,
                    'confidence': conf,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2
                })

    track_result = tracker.update(detections)

    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 7)

    for track in track_result:
        x1, y1, x2, y2, track_id = map(int, track)
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2  # use center y for crossing detection
        track_id = int(track_id)

        # Draw bounding box around the vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Get color and time for this detection
        color = get_dominant_color(frame, x1, y1, x2, y2)
        current_time = get_time_string(frame_count, fps)
        
        # Find best matching detection for this track by center distance
        matched_class = "unknown"
        min_distance = float('inf')
        best_det = None
        
        for det in detection_info:
            center_dist = abs(cx - det['center_x']) + abs(cy - det['center_y'])
            if center_dist < min_distance:
                min_distance = center_dist
                matched_class = det['class']
                best_det = det
        
        # Classify vehicle subtype based on bounding box dimensions
        if best_det:
            matched_class = classify_vehicle_type(best_det['bbox'], matched_class)

        # Draw vehicle class on the bounding box
        cv2.putText(frame, f'{matched_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Store or update vehicle information
        if track_id not in tracked_vehicles:
            tracked_vehicles[track_id] = {
                'id': track_id,
                'class': matched_class,
                'original_class': best_det['class'] if best_det else 'unknown',
                'color': color,
                'first_detected_time': current_time,
                'first_detected_frame': frame_count,
                'positions': [(cx, cy, current_time)],
                'crossed': False,
                'bbox': best_det['bbox'] if best_det else None,
                'width': best_det['bbox'][2] - best_det['bbox'][0] if best_det else 0,
                'height': best_det['bbox'][3] - best_det['bbox'][1] if best_det else 0
            }
        else:
            tracked_vehicles[track_id]['positions'].append((cx, cy, current_time))
            tracked_vehicles[track_id]['color'] = color  # Update color

        # Check if vehicle crossed the line
        if line[0] < cx < line[2] and abs(cy - line[1]) < 15:
            if track_id not in crossed_ids:
                crossed_ids.add(track_id)
                
                if matched_class != 'person':
                    # Dynamically add to counts if not present
                    if matched_class not in counts:
                        counts[matched_class] = 0
                    counts[matched_class] += 1
                    tracked_vehicles[track_id]['crossed'] = True
                    tracked_vehicles[track_id]['crossed_time'] = current_time
                    tracked_vehicles[track_id]['crossed_frame'] = frame_count
                    
                    with open(output_json, 'w') as f:
                        json.dump(counts, f, indent=2)
                else:
                    tracked_vehicles[track_id]['crossed'] = False

    display_text = "Vehicles: " + " | ".join([f"{k}={v}" for k, v in sorted(counts.items())])
    cvzone.putTextRect(frame, display_text, [10, 30], scale=1.3, thickness=2, border=2)

    cv2.imshow('Vehicle Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(output_json, 'w') as f:
    json.dump(counts, f, indent=2)

# Prepare detailed vehicle information for JSON
detailed_info = {
    'total_vehicles_crossed': len(crossed_ids),
    'vehicle_counts': counts,
    'fps': fps,
    'total_frames': frame_count,
    'vehicles': []
}

for track_id, info in tracked_vehicles.items():
    # Only include vehicles that actually crossed the line and are not people
    if info['crossed'] and info['class'] != 'person':
        aspect_ratio = info['width'] / (info['height'] + 1) if info['height'] > 0 else 0
        vehicle_info = {
            'id': track_id,
            'type': info['class'],
            'color': info['color'],
            'detected_time': info['first_detected_time'],
            'crossed_time': info.get('crossed_time', 'N/A')
        }
        
        detailed_info['vehicles'].append(vehicle_info)

# Save detailed information to JSON
with open(output_details_json, 'w') as f:
    json.dump(detailed_info, f, indent=2)

print(f"Vehicle counting completed!")
print(f"Summary saved to: {output_json}")
print(f"Detailed information saved to: {output_details_json}")
print(f"Total vehicles tracked: {len(tracked_vehicles)}")
print(f"Total vehicles crossed line: {len(crossed_ids)}")

cap.release()
cv2.destroyAllWindows()
