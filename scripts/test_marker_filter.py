"""
A/B the StaticMarkerFilter on a broadcast clip: default 10px cells (camera
jitter smears the goal/7m marker across cells so it's never caught) vs the
video-mode wider cells. Reports blacklisted cells + ball candidates dropped.

    .venv\\Scripts\\python.exe scripts\\test_marker_filter.py --max 800
"""
import sys, argparse
sys.path.insert(0, "src")
import cv2
from detector import Detector
from static_marker_filter import StaticMarkerFilter

BALL = 32

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max", type=int, default=800)
a = ap.parse_args()

det = Detector(device="cuda:0", model="yolov8s-pose.pt", conf=0.40,
               imgsz=640, ball_conf=0.03)

def make(default: bool):
    if default:
        return StaticMarkerFilter(frame_w=1280, frame_h=720)
    return StaticMarkerFilter(frame_w=1280, frame_h=720,
                              cell_size=22, neighbor_ring=0, hit_window=60,
                              hit_threshold=0.60, warmup_frames=25,
                              exclusion_ttl=200, force_blacklist_ratio=0.75)

filters = {"DEFAULT(10px)": make(True), "VIDEO(22px)": make(False)}
dropped = {k: 0 for k in filters}
cand_total = 0
max_bl = {k: 0 for k in filters}

cap = cv2.VideoCapture(a.source)
n = 0
while n < a.max:
    ok, f = cap.read()
    if not ok:
        break
    if f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
    n += 1
    dets = det.detect(f)
    ball_cands = [d for d in dets if getattr(d, "class_id", 0) == BALL]
    players = [((d.x1 + d.x2) * .5, (d.y1 + d.y2) * .5)
               for d in dets if getattr(d, "class_id", 0) != BALL]
    cand_total += len(ball_cands)
    for name, filt in filters.items():
        filt.observe(ball_cands, n, players)
        kept = filt.filter(ball_cands)
        dropped[name] += len(ball_cands) - len(kept)
        max_bl[name] = max(max_bl[name], filt.stats()["blacklist"])

cap.release()
print(f"\n=== StaticMarkerFilter A/B  ({a.source}, {n} frames, "
      f"{cand_total} raw ball candidates) ===")
for name in filters:
    print(f"  {name:14s}: peak blacklist cells={max_bl[name]:3d}   "
          f"marker candidates dropped={dropped[name]:4d}")
print("\n  (more blacklist cells + more dropped = the persistent floor marker "
      "is being caught instead of picked as the ball)")
