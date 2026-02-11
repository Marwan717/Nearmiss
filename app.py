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
CLOSE_DIST_PX = 80
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

def speed_mph(hist, fps, meters_per_pixel):
    if len(hist) < 2:
        return 0.0
    f0, x0, y0 = hist[-2]
    f1, x1, y1 = hist[-1]
    dt = (f1 - f0) / fps
    if dt <= 0:
        return 0.0
    dist_m = math.hypot(x1 - x0, y1 - y0) * meters_per_pixel
    return dist_m / dt * 2.23694

# =========================
# TRACKER
# =========================
class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        updated = {}

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            matched = None

            for tid, tr in self.tracks.items():
                if math.hypot(cx - tr["cx"], cy - tr["cy"]) < MATCH_DIST_PX:
                    matched = tid
                    break

            if matched is None:
                matched = self.next_id
                self.next_id += 1
                updated[matched] = {
                    "id": matched,
                    "cx": cx,
                    "cy": cy,
                    "bbox": det["bbox"],
                    "last": frame_idx,
                    "hist": deque(maxlen=30),
                }
            else:
                tr = self.tracks[matched]
                tr["cx"], tr["cy"] = cx, cy
                tr["bbox"] = det["bbox"]
                tr["last"] = frame_idx
                updated[matched] = tr

            updated[matched]["hist"].append((frame_idx, cx, cy))

        self.tracks = {
            k: v for k, v in updated.items()
            if frame_idx - v["last"] <= MAX_TRACK_AGE
        }
        return self.tracks

# =========================
# TTC + SEVERITY
# =========================
def compute_ttc(p1, v1, p2, v2):
    rp = np.array(p2) - np.array(p1)
    rv = np.array(v2) - np.array(v1)
    d = np.dot(rv, rv)
    if d < 1e-6:
        return None
    t = -np.dot(rp, rv) / d
    return t if t > 0 else None

def severity(pet, ttc, v1, v2):
    score = max(0, 4 - pet) * 2
    if ttc is not None:
        score += max(0, 3 - ttc) * 3
    score += (v1 + v2) / 40
    return round(score, 2)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Calibrated Near-Miss Safety Analysis")

st.write("Python:", sys.version)

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "mov", "avi"])

st.subheader("Calibration")
meters_per_pixel = st.number_input(
    "Meters per pixel (estimate from lane width)",
    value=0.05,
    step=0.01
)

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
    with st.spinner("Running analysis"):
        progress = st.progress(0.0)
        frame_slot = st.empty()
        side_panel = st.empty()

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = FPS_FALLBACK

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(tmp, "annotated.mp4")
        writer = make_writer(out_path, fps, w, h)

        model = YOLO("yolov8n.pt")
        tracker = Tracker()

        events = []
        shown = set()

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = model.predict(frame, conf=0.25, verbose=False)[0]
            dets = []

            if res.boxes is not None:
                for bb, cls in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.cls.cpu().numpy().astype(int),
                ):
                    if cls == CAR_CLASS_ID:
                        cx, cy = bb_centroid(bb)
                        dets.append({"cx": cx, "cy": cy, "bbox": bb})

            tracks = tracker.update(dets, frame_idx)

            active = [
                tr for tr in tracks.values()
                if zx1 <= tr["cx"] <= zx2 and zy1 <= tr["cy"] <= zy2
            ]

            danger_ids = set()
            side_text = []

            for a, b in combinations(active, 2):
                d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
                pet = abs(a["hist"][-1][0] - b["hist"][-1][0]) / fps

                if d < CLOSE_DIST_PX and pet <= NEAR_MISS_PET_S:
                    danger_ids.update([a["id"], b["id"]])

                    v1 = speed_mph(a["hist"], fps, meters_per_pixel)
                    v2 = speed_mph(b["hist"], fps, meters_per_pixel)

                    ttc = compute_ttc(
                        a["hist"][-1][1:], (v1, v1),
                        b["hist"][-1][1:], (v2, v2),
                    )

                    sev = severity(pet, ttc, v1, v2)

                    key = tuple(sorted([a["id"], b["id"]]))
                    if key not in shown:
                        shown.add(key)
                        events.append({
                            "time_s": round(frame_idx / fps, 2),
                            "car_1": a["id"],
                            "car_2": b["id"],
                            "avg_speed_mph": round((v1 + v2) / 2, 2),
                            "PET_s": round(pet, 2),
                            "TTC_s": None if ttc is None else round(ttc, 2),
                            "severity": sev,
                        })

                    side_text.append(
                        f"⚠️ Cars {a['id']} & {b['id']} | "
                        f"Avg Speed {round((v1+v2)/2,1)} mph | "
                        f"Time {round(frame_idx/fps,1)} s"
                    )

            side_panel.info("\n".join(side_text) if side_text else "No active conflicts")

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)

            for tr in tracks.values():
                x1, y1, x2, y2 = map(int, tr["bbox"])
                color = (0, 0, 255) if tr["id"] in danger_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {tr['id']}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if frame_idx % 10 == 0:
                frame_slot.image(frame, channels="BGR", use_container_width=True)

            writer.write(frame)
            progress.progress(min(frame_idx / total, 1.0))
            frame_idx += 1

        cap.release()
        writer.release()

    df = pd.DataFrame(events)

    st.subheader("Analysis Summary")
    if df.empty:
        st.write("No near-miss events detected.")
    else:
        st.write(f"Total near-misses: {len(df)}")
        st.write(f"Average severity: {round(df['severity'].mean(),2)}")
        st.write(f"Min PET: {df['PET_s'].min()} s")

    st.subheader("Detected Near-Miss Events")
    st.dataframe(df, use_container_width=True)

    csv_path = os.path.join(tmp, "near_miss_events.csv")
    df.to_csv(csv_path, index=False)

    st.video(out_path)
    st.download_button(
        "Download CSV",
        open(csv_path, "rb"),
        "near_miss_events.csv",
    )
