import sys
import os
import tempfile
from collections import deque
from itertools import combinations
import math

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CAR_CLASS_ID = 2
MATCH_DIST_PX = 60
MAX_TRACK_AGE = 20
NEAR_MISS_PET_S = 4.0
FPS_FALLBACK = 30.0

# =========================
# UTILS
# =========================
def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2, (y1 + y2) / 2

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def speed_px_per_s(hist, fps):
    if len(hist) < 2:
        return 0.0
    f0, x0, y0 = hist[-2]
    f1, x1, y1 = hist[-1]
    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0
    return math.hypot(x1 - x0, y1 - y0) / dt

# =========================
# TRACKER
# =========================
class Tracker:
    def __init__(self, fps):
        self.fps = fps
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        updated = {}

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            matched_id = None

            for tid, tr in self.tracks.items():
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist < MATCH_DIST_PX:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                updated[matched_id] = {
                    "id": matched_id,
                    "cx": cx,
                    "cy": cy,
                    "bbox": det["bbox"],
                    "last_frame": frame_idx,
                    "hist": deque(maxlen=30),
                }
            else:
                tr = self.tracks[matched_id]
                tr["cx"], tr["cy"] = cx, cy
                tr["bbox"] = det["bbox"]
                tr["last_frame"] = frame_idx
                updated[matched_id] = tr

            updated[matched_id]["hist"].append((frame_idx, cx, cy))

        self.tracks = {
            tid: tr for tid, tr in updated.items()
            if frame_idx - tr["last_frame"] <= MAX_TRACK_AGE
        }

        return self.tracks

# =========================
# SAFETY METRICS
# =========================
def compute_ttc(p1, v1, p2, v2):
    rel_pos = np.array(p2) - np.array(p1)
    rel_vel = np.array(v2) - np.array(v1)
    denom = np.dot(rel_vel, rel_vel)
    if denom <= 1e-6:
        return None
    ttc = -np.dot(rel_pos, rel_vel) / denom
    return ttc if ttc > 0 else None

def severity_score(pet, ttc, v1, v2):
    score = 0
    if pet is not None:
        score += max(0, (4 - pet)) * 2
    if ttc is not None:
        score += max(0, (3 - ttc)) * 3
    score += (v1 + v2) / 50
    return round(score, 2)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Vehicle Near-Miss and Safety Analysis")

st.write("Python:", sys.version)

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "mov", "avi"])

st.subheader("Conflict Zone (pixels)")
zx1 = st.number_input("xmin", 0)
zx2 = st.number_input("xmax", 300)
zy1 = st.number_input("ymin", 0)
zy2 = st.number_input("ymax", 300)

run = st.button("Run Analysis", type="primary")

# =========================
# MAIN
# =========================
if run and uploaded:
    with st.spinner("Running analysis this may take a minute"):
        progress = st.progress(0.0)
        frame_slot = st.empty()

        tmp = tempfile.mkdtemp()
        video_path = os.path.join(tmp, uploaded.name)
        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = FPS_FALLBACK

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(tmp, "annotated.mp4")
        writer = make_writer(out_path, fps, w, h)

        st.write("Loading YOLO model")
        model = YOLO("yolov8n.pt")
        st.write("Model loaded")

        tracker = Tracker(fps)
        events = []
        logged_pairs = set()

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = model.predict(frame, conf=0.25, verbose=False)[0]
            detections = []

            if res.boxes is not None:
                for bb, cls in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.cls.cpu().numpy().astype(int),
                ):
                    if cls == CAR_CLASS_ID:
                        cx, cy = bb_centroid(bb)
                        detections.append({"cx": cx, "cy": cy, "bbox": bb})

            tracks = tracker.update(detections, frame_idx)

            active = []
            for tr in tracks.values():
                if zx1 <= tr["cx"] <= zx2 and zy1 <= tr["cy"] <= zy2:
                    active.append(tr)

            for a, b in combinations(active, 2):
                pair = tuple(sorted([a["id"], b["id"]]))
                if pair in logged_pairs:
                    continue

                pet = abs(a["hist"][-1][0] - b["hist"][-1][0]) / fps
                if pet <= NEAR_MISS_PET_S:
                    v1 = speed_px_per_s(a["hist"], fps)
                    v2 = speed_px_per_s(b["hist"], fps)

                    p1 = a["hist"][-1][1:]
                    p2 = b["hist"][-1][1:]

                    ttc = compute_ttc(
                        p1, (v1, v1),
                        p2, (v2, v2),
                    )

                    sev = severity_score(pet, ttc, v1, v2)

                    logged_pairs.add(pair)
                    events.append({
                        "time_s": round(frame_idx / fps, 2),
                        "car_1": a["id"],
                        "car_2": b["id"],
                        "speed_1_px_s": round(v1, 2),
                        "speed_2_px_s": round(v2, 2),
                        "PET_s": round(pet, 2),
                        "TTC_s": None if ttc is None else round(ttc, 2),
                        "severity": sev,
                    })

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)

            for tr in tracks.values():
                x1, y1, x2, y2 = map(int, tr["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {tr['id']}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if frame_idx % 10 == 0:
                frame_slot.image(frame, channels="BGR", use_container_width=True)

            writer.write(frame)
            progress.progress(min(frame_idx / total_frames, 1.0))
            frame_idx += 1

        cap.release()
        writer.release()

    df = pd.DataFrame(events)

    # =========================
    # TEXT ANALYSIS SUMMARY
    # =========================
    st.subheader("Safety Analysis Summary")

    if len(df) == 0:
        st.write("No near-miss events detected in the selected conflict zone.")
    else:
        st.write(f"Total near-miss events detected: {len(df)}")
        st.write(f"Average severity score: {round(df['severity'].mean(), 2)}")
        st.write(f"Minimum PET: {df['PET_s'].min()} seconds")
        if df['TTC_s'].notna().any():
            st.write(f"Minimum TTC: {df['TTC_s'].min()} seconds")

    st.subheader("Detailed Events")
    st.dataframe(df, use_container_width=True)

    csv_path = os.path.join(tmp, "safety_events.csv")
    df.to_csv(csv_path, index=False)

    st.video(out_path)
    st.download_button(
        "Download CSV",
        open(csv_path, "rb"),
        "safety_events.csv",
    )
