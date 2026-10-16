from ultralytics import YOLO
import cv2

VIDEO = "data/traffic1.mp4"
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LINE_POSITION = 0.6   # counting line at 60% of frame height (0=top, 1=bottom)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO)

counts = {name: 0 for name in CLASS_NAMES.values()}
counted_ids = set()   # IDs already counted (prevents duplicates)
prev_y = {}           # last vertical position of each ID

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # shrink frame for speed
    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    line_y = int(frame.shape[0] * LINE_POSITION)

    results = model.track(frame, persist=True, classes=list(CLASS_NAMES),
                          tracker="bytetrack.yaml", verbose=False)
    r = results[0]

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        classes = r.boxes.cls.int().cpu().tolist()

        for (x1, y1, x2, y2), tid, c in zip(boxes, ids, classes):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # crossed the line between previous and current frame?
            if tid in prev_y and tid not in counted_ids:
                if (prev_y[tid] < line_y <= cy) or (prev_y[tid] > line_y >= cy):
                    counted_ids.add(tid)
                    counts[CLASS_NAMES[c]] += 1
            prev_y[tid] = cy

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{CLASS_NAMES[c]} #{tid}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # draw line and counts
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
    cv2.putText(frame, f"Total: {sum(counts.values())}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    for i, (name, n) in enumerate(counts.items()):
        cv2.putText(frame, f"{name}: {n}", (10, 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    cv2.imshow("Counting", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Counting", cv2.WND_PROP_VISIBLE) < 1:
        break
    

cap.release()
cv2.destroyAllWindows()
print("Final counts:", counts)