import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import math
import time

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = "input.mp4"          # your video
OUTPUT_PATH = "output.mp4"
MODEL_PATH = "yolov8n.pt"

CONF_THRESH = 0.4
NEAR_MISS_DIST_PX = 60            # pixel distance threshold
TTC_THRESHOLD = 2.0               # seconds (proxy)
FPS_ASSUMED = 30

# ===============================
# LOAD MODEL
# ===============================
model = YOLO(MODEL_PATH)

# ===============================
# DATA STRUCTURES
# ===============================
trajectories = defaultdict(list)
last_positions = {}
last_times = {}

near_miss_events = []

# ===============================
# HELPER FUNCTIONS
# ===============================
def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def estimate_speed(p1, p2, dt):
    if dt == 0:
        return 0
    return euclidean(p1, p2) / dt

def draw_trajectory(frame, points):
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)

# ===============================
# VIDEO IO
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ASSUMED

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_idx = 0

# ===============================
# MAIN LOOP
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESH,
        classes=[2, 3, 5, 7]  # car, motorcycle, bus, truck
    )

    detections = []

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, clss):
            cx, cy = center_of_box(box)
            detections.append((int(obj_id), (cx, cy), box))

            # store trajectory
            trajectories[obj_id].append((cx, cy))

            # draw box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"id:{obj_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # draw trajectory
            draw_trajectory(frame, trajectories[obj_id])

    # ===============================
    # NEAR MISS CHECK
    # ===============================
    for i in range(len(detections)):
        id1, p1, _ = detections[i]
        for j in range(i + 1, len(detections)):
            id2, p2, _ = detections[j]

            dist = euclidean(p1, p2)

            if dist < NEAR_MISS_DIST_PX:
                ttc = dist / max(1, fps)  # simple proxy

                if ttc < TTC_THRESHOLD:
                    near_miss_events.append({
                        "id_1": id1,
                        "id_2": id2,
                        "safety_indicator": round(ttc, 2),
                        "class_1": "car",
                        "class_2": "car",
                        "timestamp": round(timestamp, 2)
                    })

                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "NEAR MISS",
                        ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# ===============================
# SAVE EVENTS TABLE
# ===============================
df = pd.DataFrame(near_miss_events)
df.to_csv("near_miss_events.csv", index=False)

print("Processing complete.")
print(f"Near-miss events detected: {len(df)}")
print("Saved: output.mp4, near_miss_events.csv")
