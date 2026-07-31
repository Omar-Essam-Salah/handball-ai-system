"""Does sliced (SAHI-style) inference recover balls the full frame misses?
Measures ball-detection recall: full-frame @960  vs  tiled slices @640.
Tested on handball.mp4 (representative), NOT hand.mp4."""
import sys, time
sys.path.insert(0, "src")
import cv2
from ultralytics import YOLO
from pathlib import Path

root = Path(__file__).resolve().parent.parent
wts = root / "handball_ball.pt"
model = YOLO(str(wts) if wts.exists() else "yolov8n.pt", task="detect")
cls = [0] if wts.exists() else [32]
CONF = 0.03
AREA_MIN, AREA_MAX, ASP = 20, 9000, 4.0

def valid(b):
    x1,y1,x2,y2 = b; w,h = x2-x1, y2-y1; a = w*h
    if a < AREA_MIN or a > AREA_MAX: return False
    return h > 0 and (1/ASP <= w/h <= ASP)

def det(img, imgsz):
    r = model(img, classes=cls, conf=CONF, imgsz=imgsz, verbose=False, half=True, device="cuda:0")[0]
    return any(valid(b.xyxy[0].tolist()) for b in r.boxes)

def det_tiled(frame, tile=512, ov=0.25):
    H, W = frame.shape[:2]
    step = int(tile * (1 - ov))
    xs = list(range(0, max(1, W - tile + 1), step)) + [W - tile]
    ys = list(range(0, max(1, H - tile + 1), step)) + [H - tile]
    for y in sorted(set(max(0, v) for v in ys)):
        for x in sorted(set(max(0, v) for v in xs)):
            crop = frame[y:y+tile, x:x+tile]
            if crop.size and det(crop, 512):
                return True
    return False

cap = cv2.VideoCapture("handball.mp4")
n = full = tiled = 0
t_full = t_tile = 0.0
while n < 500:
    ok, f = cap.read()
    if not ok: break
    if f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0]*(1280/f.shape[1]))))
    n += 1
    t0 = time.time(); full += det(f, 960);      t_full += time.time()-t0
    t0 = time.time(); tiled += det_tiled(f);     t_tile += time.time()-t0
cap.release()
print(f"frames={n}")
print(f"FULL-FRAME @960 : recall {100*full/n:5.1f}%   ({n/t_full:5.1f} fps)")
print(f"TILED (SAHI)    : recall {100*tiled/n:5.1f}%   ({n/t_tile:5.1f} fps)")
print(f"=> SAHI gain: +{100*(tiled-full)/n:.1f} pts, at {t_tile/t_full:.1f}x the cost")
