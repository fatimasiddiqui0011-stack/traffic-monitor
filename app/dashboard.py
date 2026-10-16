import time
import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="Traffic Monitor", layout="wide")
st.title("AI-Based Traffic Monitoring System")

with st.sidebar:
    st.header("Input")
    kind = st.radio("Source", ["Video file", "Camera"])
    if kind == "Video file":
        source = st.selectbox("Video", ["data/traffic1.mp4", "data/traffic2.mp4"])
    else:
        source = st.text_input("Camera index (0) or stream URL", "0")
    orientation = st.radio("Counting line", ["horizontal", "vertical"])
    position = st.slider("Line position (0 = top/left, 1 = bottom/right)",
                         0.05, 0.95, 0.5, 0.05)
    if st.button("Start"):
        requests.post(f"{API}/start", params={
            "source": source, "orientation": orientation, "position": position})

    
    if st.button("Stop"):
        requests.post(f"{API}/stop")

left, right = st.columns([3, 1])
video_box = left.empty()
chart_box = left.empty()
info_box = right.empty()

while True:
    try:
        res = requests.get(f"{API}/results", timeout=2).json()
        img = requests.get(f"{API}/frame", timeout=2).content
    except Exception:
        st.error("Backend not running. Start it with uvicorn (see instructions).")
        break

    if img:
        video_box.image(img, use_container_width=True)
    with info_box.container():
        st.metric("Camera status", res["status"])
        st.write(res["timestamp"])
        st.metric("Total count", res["total"])
        st.metric("Traffic density", res["density"].upper())
        st.write(f"Vehicles in frame: {res['in_frame']}  |  FPS: {res['fps']}")
        for name, n in res["counts"].items():
            st.write(f"**{name}**: {n}")
    if res["history"]:
        df = pd.DataFrame(res["history"]).set_index("time")
        chart_box.line_chart(df["total"])
    time.sleep(0.2)