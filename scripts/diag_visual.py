"""
Save annotated frames from a gameplay section so we can SEE what the ball
tracker actually does on a given video: every raw YOLO ball candidate (with its
skin%) and the tracker's final pick.

    .venv\\Scripts\\python.exe scripts\\diag_visual.py --source 22.mp4 --frac 0.5 --n 6
"""
import sys, argparse
sys.path.insert(0, "src")
import cv2
from detector import Detector
from tracker import Tracker
from ball_appearance import skin_ratio

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="22.mp4")
ap.add_argument("--frac", type=float, default=0.5)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--every", type=int, default=25)
ap.add_argument("--out", default="reports/diag")
a = ap.parse_args()

from pathlib import Path
outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
det = Detector(device="cuda:0", model="yolov8s-pose.pt", conf=0.40, imgsz=640, ball_conf=0.05)
trk = Tracker(device="cuda:0", backend="botsort")

cap = cv2.VideoCapture(a.source)
tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.frac * tot))
saved = 0
i = 0
stem = Path(a.source).stem
while saved < a.n:
    ok, f = cap.read()
    if not ok:
        break
    if f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
    i += 1
    dets = det.detect(f)
    tracks = trk.update(dets, f)
    if i % a.every != 0:
        continue
    vis = f.copy()
    # raw ball candidates from YOLO (yellow boxes) + skin%
    for d in dets:
        if getattr(d, "class_id", 0) == 32:
            sr = skin_ratio(f, d.x1, d.y1, d.x2, d.y2)
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 255), 2)
            cv2.putText(vis, f"cand {d.confidence:.2f} skin{int(sr*100)}%",
                        (int(d.x1), int(d.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # players (thin) for context
    for t in tracks:
        if getattr(t, "class_id", 0) != 32:
            cv2.rectangle(vis, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), (90, 90, 90), 1)
    # tracker's chosen ball (big red circle)
    ball = next((t for t in tracks if getattr(t, "class_id", 0) == 32), None)
    hall = bool(getattr(trk, "ball_was_hallucinated", False))
    if ball is not None:
        col = (0, 0, 255) if not hall else (255, 0, 255)
        cv2.circle(vis, (int(ball.cx), int(ball.cy)), 12, col, 2)
        cv2.putText(vis, "BALL" + (" (guess)" if hall else ""), (int(ball.cx) + 14, int(ball.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    p = outdir / f"{stem}_{a.frac:.2f}_{saved:02d}.jpg"
    cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("saved", p)
    saved += 1
cap.release()
