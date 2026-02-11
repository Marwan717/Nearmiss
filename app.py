import streamlit as st
import sys
import cv2
import math
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, deque

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("Traffic Near-Miss Detection (TTC + PET)")

st.caption(f"Python version: {sys.version}")

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("Calibration & Thresholds")

meters_per_pixel = st.sidebar.number_input(
    "Meters per pixel",
    min_value=0.001,
    max_value=1.0,
    value=20/600,
    step=0.001
)

ttc_threshold = st.sidebar.slider("TTC threshold (s)", 0.5, 5.0, 2.0)
pet_threshold = st.sidebar.slider("PET threshold (s)", 0.5, 5.0, 1.5)

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
run_button = st.button("Run analysis")

# ===============================
# METRICS
# ===============================
def velocity(p_prev, p_curr, dt, mpp):
    if dt == 0:
        return np.array([0.0, 0.0])
    return (np.array(p_curr) - np.array(p_prev)) * mpp / dt

def time_to_collision(p1, v1, p2, v2, mpp):
    dp = (np.array(p2) - np.array(p1)) * mpp
    dv = v2 - v1
    dv2 = np.dot(dv, dv)
    if dv2 < 1e-6:
        return np.inf
    ttc = -np.dot(dp, dv) / dv2
    return ttc if ttc > 0 else np.inf

def post_encroachment_time(path1, path2, fps):
    if len(path1) < 2 or len(path2) < 2:
        return np.inf
    min_d = np.inf
    i_min = j_min = 0
    for i, p1 in enumerate(path1):
        for j, p2 in enumerate(path2):
            d = math.dist(p1, p2)
            if d < min_d:
                min_d = d
                i_min, j_min = i, j
    return abs(i_min - j_min) / fps

# ===============================
# VIDEO PROCESSING
# ===============================
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
    events = []
    event_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7]
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
                cv2.putText(frame, f"id:{int(oid)}",
                            (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1, id2 = ids[i], ids[j]
                if len(trajectories[id1]) < 2 or len(trajectories[id2]) < 2:
                    continue

                p1n, p1p = trajectories[id1][-1], trajectories[id1][-2]
                p2n, p2p = trajectories[id2][-1], trajectories[id2][-2]

                v1 = velocity(p1p, p1n, 1/fps, meters_per_pixel)
                v2 = velocity(p2p, p2n, 1/fps, meters_per_pixel)

                ttc = time_to_collision(p1n, v1, p2n, v2, meters_per_pixel)
                pet = post_encroachment_time(trajectories[id1], trajectories[id2], fps)

                if ttc < ttc_threshold or pet < pet_threshold:
                    events.append({
                        "ID": event_id,
                        "TTC (s)": round(ttc, 2),
                        "PET (s)": round(pet, 2),
                        "Vehicle 1": "car",
                        "Vehicle 2": "car",
                        "Time": time.strftime("%H:%M:%S")
                    })
                    event_id += 1

                    cv2.line(frame, p1n, p2n, (0,0,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return out_path, pd.DataFrame(events)

# ===============================
# RUN
# ===============================
if uploaded_video and run_button:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_video.read())
        path = tmp.name

    with st.spinner("Processing video…"):
        out_video, df = process_video(path)

    st.success("Done")

    st.video(out_video)

    st.subheader("Near-Miss Events")
    if len(df):
        st.dataframe(df, use_container_width=True)
        st.metric("Total Near Misses", len(df))
    else:
        st.info("No near-miss events detected.")
