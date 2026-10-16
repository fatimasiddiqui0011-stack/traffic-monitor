# AI-Based Traffic Monitoring System

Detects and tracks vehicles in road video, counts them by class when they cross a line, estimates traffic density (low / medium / high), and shows live results on a web dashboard.

## Features
- Video file or camera input, selected in the dashboard
- YOLOv8n detection (car, bus, truck, motorcycle, bicycle) with ByteTrack tracking
- Each vehicle counted once when it crosses a configurable counting line
- Density rule based on vehicles in the current frame
- FastAPI backend (`/start`, `/stop`, `/status`, `/results`, `/frame`)
- Summaries saved to `outputs/summary.csv` every 2 seconds
- Streamlit dashboard: annotated video, total and class-wise counts, status, timestamp, FPS, count-over-time chart

## Project structure
```
traffic-monitor/
  app/
    backend.py      FastAPI + detection/tracking/counting
    dashboard.py    Streamlit dashboard
    count.py        standalone test script (OpenCV window)
  data/             input videos (not committed)
  outputs/          summary.csv
  docs/             report and presentation
  requirements.txt
```

## Setup (Windows)
1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Put your test videos in `data/` (for example `traffic1.mp4`, `traffic2.mp4`). The model file `yolov8n.pt` downloads automatically on first run.

## Run
Open two terminals, activate the venv in each, and run from the project folder:

Terminal 1 (backend):
```
uvicorn app.backend:app --port 8000
```
Terminal 2 (dashboard):
```
streamlit run app/dashboard.py
```
In the dashboard: choose the video, set the counting line orientation and position, then click **Start**. Click **Stop** before changing the line and starting again.

## Density rule
| Vehicles in frame | Level |
|---|---|
| fewer than 8 | Low |
| 8 to 15 | Medium |
| 16 or more | High |

Change the thresholds in `density_level()` in `app/backend.py`.

## Evaluation summary
A 30-second sample of each clip was checked against an approximate manual count.

| | Clip 1 | Clip 2 |
|---|---|---|
| Duration / resolution | 44 s, 3840x2160, 50 fps | 60 s, 3840x2160, 30 fps |
| Manual count (approx.) | 29 | 54 |
| System count | 38 | 60 |
| Missed / duplicates noticed | 0 / 2 | 1 / 0 |
| Accuracy | 69.0% | 88.9% |
| Speed (CPU) | about 6.8 FPS | about 6.5 FPS |

The system tends to overcount (extra counts from ID switches near the line) rather than miss vehicles.

See `docs/` for the full report.

## Known limitations
Occlusion and ID switches can cause missed or duplicate counts (the main error in testing was overcounting); low light and small vehicles reduce detection quality; the counting line and density thresholds depend on the camera view; processing is slower than real time on a CPU.

## Tech stack
Python, Ultralytics YOLOv8, OpenCV, FastAPI, Uvicorn, Streamlit, pandas.

## Credits
Test videos from Pexels (free license).
