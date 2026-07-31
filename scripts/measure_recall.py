"""Measure ball-detection recall of a given model on a video (handball.mp4)."""
import sys, argparse
sys.path.insert(0, "src")
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max", type=int, default=500)
a = ap.parse_args()

m = YOLO(a.model, task="detect")
cap = cv2.VideoCapture(a.source)
n = hit = 0
while n < a.max:
    ok, f = cap.read()
    if not ok:
        break
    if f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
    n += 1
    g = False
    for b in m(f, classes=[0], conf=0.03, imgsz=960, verbose=False,
               half=True, device="cuda:0")[0].boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        if h > 0 and 20 <= w * h <= 9000 and 0.25 <= w / h <= 4:
            g = True
            break
    hit += g
cap.release()
print("%.1f" % (100 * hit / max(n, 1)))
