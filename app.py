import streamlit as st
import cv2
import math
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, deque

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(layout="wide")
st.title("Traffic Near-Miss Detection (TTC + PET)")

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.header("Calibration & Thresholds")

meters_per_pixel = st.sidebar.number_input(
    "Meters per pixel (camera calibration)",
    min_value=0.001,
    max_value=1.0,
    value=20/600,
    step=0.001
)

ttc_threshold = st.sidebar.slider("TTC threshold (s)", 0.5, 5.0, 2.0)
pet_threshold = st.sidebar.slider("PET threshold (s)", 0.5, 5.0, 1.5)

uploaded_video = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
run_button = st.button("Run Analysis")

# ======================================================
# GEOMETRY / SAFETY METRICS
# ======================================================
def velocity(p_prev, p_curr, dt, mpp):
    if dt == 0:
        return np.array([0.0, 0.0])
    return (np.array(p_curr) - np.array(p_prev)) * mpp / dt

def time_to_collision(p1, v1, p2, v2, mpp):
    dp = (np.array(p2) - np.array(p1)) * mpp
    dv = v2 - v1
    dv_norm = np.dot(dv, dv)
    if dv_norm < 1e-6:
        return np.inf
    ttc = -np.dot(dp, dv) / dv_norm
    return ttc if ttc > 0 else np.inf

def post_encroachment_time(path1, path2, fps):
    if len(path1) < 2 or len(path2) < 2:
        return np.inf
    min_dist = np.inf
    i_min = j_min = 0
    for i, p1 in enumerate(path1):
        for j, p2 in enumerate(path2):
            d = math.dist(p1, p2)
            if d < min_dist:
                min_dist = d
                i_min, j_min = i, j
    return abs(i_min - j_min) / fps

# ======================================================
# VIDEO PROCESSING
# ======================================================
def process_video(video_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "output.mp4"
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    trajectories = defaultdict(lambda: deque(maxlen=30))
    near_miss_events = []
    event_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7]  # vehicles
        )

        ids = []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, oid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                trajectories[oid].append((cx, cy))
                ids.append(oid)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"id:{int(oid)}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                for i in range(1, len(trajectories[oid])):
                    cv2.line(frame,
                             trajectories[oid][i-1],
                             trajectories[oid][i],
                             (0,255,0), 2)

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1, id2 = ids[i], ids[j]
                if len(trajectories[id1]) < 2 or len(trajectories[id2]) < 2:
                    continue

                p1_now, p1_prev = trajectories[id1][-1], trajectories[id1][-2]
                p2_now, p2_prev = trajectories[id2][-1], trajectories[id2][-2]

                v1 = velocity(p1_prev, p1_now, 1/fps, meters_per_pixel)
                v2 = velocity(p2_prev, p2_now, 1/fps, meters_per_pixel)

                ttc = time_to_collision(p1_now, v1, p2_now, v2, meters_per_pixel)
                pet = post_encroachment_time(trajectories[id1], trajectories[id2], fps)

                if ttc < ttc_threshold or pet < pet_threshold:
                    near_miss_events.append({
                        "ID": event_id,
                        "TTC (s)": round(ttc, 2),
                        "PET (s)": round(pet, 2),
                        "Vehicle 1": "car",
                        "Vehicle 2": "car",
                        "Time": time.strftime("%H:%M:%S")
                    })
                    event_id += 1

                    cv2.line(frame, p1_now, p2_now, (0,0,255), 2)
                    cv2.putText(frame, "NEAR MISS",
                                ((p1_now[0]+p2_now[0])//2,
                                 (p1_now[1]+p2_now[1])//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return out_path, pd.DataFrame(near_miss_events)

# ======================================================
# RUN PIPELINE
# ======================================================
if uploaded_video and run_button:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    with st.spinner("Processing video…"):
        out_video, df = process_video(video_path)

    st.success("Processing complete")

    st.header("Processed Video")
    st.video(out_video)

    st.header("Near-Miss Events")
    if len(df) > 0:
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Near Misses", len(df))
        col2.metric("Avg TTC (s)", round(df["TTC (s)"].mean(), 2))
    else:
        st.info("No near-miss events detected.")
