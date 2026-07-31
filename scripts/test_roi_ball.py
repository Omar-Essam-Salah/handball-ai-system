"""
Measure whether ROI (crop-around-last-ball) recovers balls that full-frame YOLO
misses. Grounds the decision to add ROI ball detection.

For each frame:
  1. full-frame ball detect (current behaviour).
  2. if it found a ball -> remember position.
  3. if it MISSED but we have a recent position -> crop a window around it and
     detect on the (high-res) crop. Count how many misses the ROI recovers.
"""
import sys, argparse
sys.path.insert(0, "src")
import cv2
from ultralytics import YOLO
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max-frames", type=int, default=1500)
ap.add_argument("--ball-conf", type=float, default=0.03)
ap.add_argument("--roi", type=int, default=416)
args = ap.parse_args()

AREA_MIN, AREA_MAX, ASPECT = 20, 9000, 4.0
root = Path(__file__).resolve().parent.parent
wts = root / "handball_ball.pt"
model = YOLO(str(wts) if wts.exists() else "yolov8n.pt", task="detect")
cls = [0] if wts.exists() else [32]
print("ball model:", "handball_ball.pt" if wts.exists() else "yolov8n.pt(COCO)")

def detect(img, imgsz):
    r = model(img, classes=cls, conf=args.ball_conf, imgsz=imgsz,
              verbose=False, half=True, device="cuda:0")[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        w, h = x2-x1, y2-y1; a = w*h
        if a < AREA_MIN or a > AREA_MAX: continue
        if h > 0 and not (1/ASPECT <= w/h <= ASPECT): continue
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out

cap = cv2.VideoCapture(args.source)
n = full_hit = roi_recovered = miss_total = 0
last = None
R = args.roi
while n < args.max_frames:
    ret, frame = cap.read()
    if not ret: break
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0]*(1280/frame.shape[1]))))
    n += 1
    full = detect(frame, 960)
    if full:
        full_hit += 1
        bx = (full[0][0]+full[0][2])/2; by = (full[0][1]+full[0][3])/2
        last = (bx, by)
        continue
    # full-frame missed
    miss_total += 1
    if last is not None:
        cx, cy = last
        x1 = max(0, int(cx-R/2)); y1 = max(0, int(cy-R/2))
        x2 = min(frame.shape[1], x1+R); y2 = min(frame.shape[0], y1+R)
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            roi = detect(crop, 640)   # crop is small -> ball is relatively large
            if roi:
                roi_recovered += 1
                last = (x1+(roi[0][0]+roi[0][2])/2, y1+(roi[0][1]+roi[0][3])/2)

cap.release()
print(f"\nframes={n}")
print(f"full-frame hit   : {full_hit}  ({100*full_hit/n:.1f}%)")
print(f"full-frame missed: {miss_total}  ({100*miss_total/n:.1f}%)")
print(f"ROI recovered    : {roi_recovered}  ({100*roi_recovered/max(miss_total,1):.1f}% of misses, "
      f"+{100*roi_recovered/n:.1f}% absolute)")
print(f"=> combined recall: {100*(full_hit+roi_recovered)/n:.1f}%  (was {100*full_hit/n:.1f}%)")
