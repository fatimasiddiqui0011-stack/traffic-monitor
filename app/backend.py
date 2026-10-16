import csv, os, threading, time
from datetime import datetime
import cv2
from fastapi import FastAPI, Response
from ultralytics import YOLO

app = FastAPI()

CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CSV_FILE = "outputs/summary.csv"

state = {"running": False, "status": "idle", "source": "", "counts": {},
         "total": 0, "in_frame": 0, "density": "low", "fps": 0.0,
         "timestamp": "", "history": [], "frame": b""}
stop_flag = threading.Event()
worker = {"thread": None}


def density_level(n):
    # Rule: number of vehicles visible in the current frame
    if n < 8:
        return "low"
    if n < 16:
        return "medium"
    return "high"


def process(source, orientation, position):
    model = YOLO("yolov8n.pt")  # fresh model = fresh tracker
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        state.update(running=False, status="error: cannot open source")
        return

    counts = {n: 0 for n in CLASS_NAMES.values()}
    counted, last_side, history = set(), {}, []
    state.update(running=True, status="running", source=source, history=history)
    t0 = last_log = time.time()
    frames = 0

    while not stop_flag.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
        h, w = frame.shape[:2]
        line_pos = int((h if orientation == "horizontal" else w) * position)

        r = model.track(frame, persist=True, classes=list(CLASS_NAMES),
                        tracker="bytetrack.yaml", verbose=False)[0]
        in_frame = 0
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            classes = r.boxes.cls.int().cpu().tolist()
            in_frame = len(ids)
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, classes):
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                pos = cy if orientation == "horizontal" else cx
                side = pos > line_pos
                if tid in last_side and last_side[tid] != side and tid not in counted:
                    counted.add(tid)
                    counts[CLASS_NAMES[c]] += 1
                last_side[tid] = side

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{CLASS_NAMES[c]} #{tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if orientation == "horizontal":
            cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 0), 2)
        else:
            cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 255, 0), 2)

        frames += 1
        now = time.time()
        total = sum(counts.values())
        density = density_level(in_frame)
        state.update(counts=dict(counts), total=total, in_frame=in_frame,
                     density=density, fps=round(frames / (now - t0), 1),
                     timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     frame=cv2.imencode(".jpg", frame)[1].tobytes())

        # save a summary row every 2 seconds
        if now - last_log >= 2:
            last_log = now
            row = {"time": datetime.now().strftime("%H:%M:%S"), "total": total,
                   "density": density, **counts}
            history.append(row)
            os.makedirs("outputs", exist_ok=True)
            new_file = not os.path.exists(CSV_FILE)
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

    cap.release()
    state.update(running=False, status="finished")


@app.post("/start")
def start(source: str, orientation: str = "horizontal", position: float = 0.5):
    if state["running"]:
        return {"message": "already running, click Stop first"}
    stop_flag.clear()
    t = threading.Thread(target=process, args=(source, orientation, position),
                         daemon=True)
    worker["thread"] = t
    t.start()
    return {"message": f"started {source}"}


@app.post("/stop")
def stop():
    stop_flag.set()
    if worker["thread"]:
        worker["thread"].join(timeout=10)  # wait until the old run has fully stopped
    return {"message": "stopped"}


@app.get("/status")
def status():
    return {"status": state["status"], "running": state["running"],
            "source": state["source"], "fps": state["fps"],
            "timestamp": state["timestamp"]}


@app.get("/results")
def results():
    return {k: state[k] for k in ["counts", "total", "in_frame", "density",
                                  "fps", "timestamp", "status", "history"]}


@app.get("/frame")
def frame():
    return Response(content=state["frame"], media_type="image/jpeg")