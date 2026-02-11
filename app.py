import sys
import cv2
import math
import time
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# =============================
# CONFIG
# =============================
CONF_THRESH = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck
NEAR_MISS_DIST = 60              # pixels
TTC_THRESHOLD = 2.0              # seconds
MAX_TABLE_ROWS = 5

OUTPUT_VIDEO = "output.mp4"
OUTPUT_CSV = "near_miss_events.csv"

# =============================
# DRAW NEAR-MISS TABLE (SCREENSHOT FEATURE)
# =============================
def draw_near_miss_panel(frame, events):
    x, y = 20, 40
    row_h = 22

    cv2.rectangle(frame, (10, 10), (700, 200), (40, 40, 40), -1)
    cv2.putText(frame, "Near miss - Near miss events",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    headers = ["ID", "Safety", "Class 1", "Class 2", "Time"]
    for i, h in enumerate(headers):
        cv2.putText(frame, h,
                    (x + i * 130, y + row_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

    recent = events[-MAX_TABLE_ROWS:]
    for r, e in enumerate(recent):
        yy = y + (r + 2) * row_h
        cv2.putText(frame, str(e["id"]), (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, str(e["safety_indicator"]), (x + 130, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, e["traj_1"], (x + 260, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, e["traj_2"], (x + 390, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, e["time"], (x + 520, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

# =============================
# MAIN VIDEO PIPELINE
# =============================
def main(video_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    trajectories = defaultdict(list)
    near_miss_events = []
    event_id = 1
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESH,
            classes=VEHICLE_CLASSES
        )

        detections = []

        # =============================
        # TRACKING + DRAWING
        # =============================
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, oid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                trajectories[oid].append((cx, cy))
                detections.append((oid, (cx, cy)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"id:{int(oid)}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

                for i in range(1, len(trajectories[oid])):
                    cv2.line(frame,
                             trajectories[oid][i - 1],
                             trajectories[oid][i],
                             (0, 255, 0), 2)

        # =============================
        # NEAR-MISS DETECTION
        # =============================
        for i in range(len(detections)):
            id1, p1 = detections[i]
            for j in range(i + 1, len(detections)):
                id2, p2 = detections[j]
                dist = math.dist(p1, p2)

                if dist < NEAR_MISS_DIST:
                    ttc = dist / fps
                    if ttc < TTC_THRESHOLD:
                        event = {
                            "id": event_id,
                            "safety_indicator": round(ttc, 2),
                            "traj_1": "car",
                            "traj_2": "car",
                            "time": time.strftime("%H:%M:%S")
                        }
                        near_miss_events.append(event)
                        event_id += 1

                        cv2.line(frame, p1, p2, (0, 0, 255), 2)
                        cv2.putText(frame, "NEAR MISS",
                                    ((p1[0] + p2[0]) // 2,
                                     (p1[1] + p2[1]) // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

        # =============================
        # DRAW TABLE PANEL
        # =============================
        draw_near_miss_panel(frame, near_miss_events)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    pd.DataFrame(near_miss_events).to_csv(OUTPUT_CSV, index=False)

    print("DONE")
    print(f"Video saved: {OUTPUT_VIDEO}")
    print(f"Events saved: {OUTPUT_CSV}")
    print(f"Total near-miss events: {len(near_miss_events)}")

# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py input_video.mp4")
        sys.exit(1)

    main(sys.argv[1])
