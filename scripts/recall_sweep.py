"""
Ball DETECTION recall sweep. Measures, per config, the % of frames where the
ball model produces at least one valid ball candidate (same area/aspect gates
the live detector uses). This tells us how to make the ball actually visible to
the tracker. No tracking here — pure detection recall.
"""
import sys, argparse, time
sys.path.insert(0, "src")
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max-frames", type=int, default=600)
args = ap.parse_args()

AREA_MIN, AREA_MAX, ASPECT = 20, 9000, 4.0

def valid(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a = w * h
    if a < AREA_MIN or a > AREA_MAX:
        return False
    if h > 0:
        r = w / h
        if r > ASPECT or r < 1.0 / ASPECT:
            return False
    return True

ft = YOLO("handball_ball.pt", task="detect")    # fine-tuned, class 0
coco = YOLO("yolov8n.pt", task="detect")          # COCO, class 32

# Pre-load frames (downscaled to 1280 like the pipeline) for fair reuse
frames = []
cap = cv2.VideoCapture(args.source)
while len(frames) < args.max_frames:
    ret, f = cap.read()
    if not ret:
        break
    if f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0] * (1280 / f.shape[1]))))
    frames.append(f)
cap.release()
N = len(frames)
print(f"loaded {N} frames from {args.source}\n")

def recall(model, classes, conf, imgsz):
    hit = 0
    t0 = time.time()
    for f in frames:
        r = model(f, classes=classes, conf=conf, imgsz=imgsz,
                  verbose=False, half=True, device="cuda:0")[0]
        ok = False
        for b in r.boxes:
            if valid(b.xyxy[0].tolist()):
                ok = True
                break
        if ok:
            hit += 1
    dt = time.time() - t0
    return 100 * hit / max(N, 1), N / dt

print(f"{'config':42s} {'recall%':>8s} {'fps':>7s}")
print("-" * 60)
for imgsz in (640, 960, 1280):
    for conf in (0.04, 0.02):
        rc, fps = recall(ft, [0], conf, imgsz)
        print(f"{'fine-tuned imgsz=%d conf=%.2f' % (imgsz, conf):42s} {rc:8.1f} {fps:7.1f}")
# COCO at best imgsz
for imgsz in (640, 1280):
    rc, fps = recall(coco, [32], 0.10, imgsz)
    print(f"{'COCO(yolov8n) imgsz=%d conf=0.10' % imgsz:42s} {rc:8.1f} {fps:7.1f}")

# UNION: fine-tuned + COCO at imgsz=1280 (does combining help recall?)
hit = 0
t0 = time.time()
for f in frames:
    ok = False
    for model, cls, cf in ((ft, [0], 0.03), (coco, [32], 0.10)):
        r = model(f, classes=cls, conf=cf, imgsz=1280, verbose=False, half=True, device="cuda:0")[0]
        for b in r.boxes:
            if valid(b.xyxy[0].tolist()):
                ok = True
                break
        if ok:
            break
    if ok:
        hit += 1
dt = time.time() - t0
print(f"{'UNION ft+COCO imgsz=1280':42s} {100*hit/max(N,1):8.1f} {N/dt:7.1f}")
