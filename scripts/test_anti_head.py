"""
A/B the anti-head appearance filter: how often does the EMITTED ball land on a
skin/head region, and is continuity preserved?  Runs Detector+Tracker twice.

    .venv\\Scripts\\python.exe scripts\\test_anti_head.py --max 1200
"""
import sys, argparse, math
sys.path.insert(0, "src")
import cv2
from detector import Detector
import tracker as T
from ball_appearance import skin_ratio

BALL = 32
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max", type=int, default=1200)
a = ap.parse_args()

det = Detector(device="cuda:0", model="yolov8s-pose.pt", conf=0.40,
               imgsz=640, ball_conf=0.05)


def run(skin_on: bool):
    trk = T.Tracker(device="cuda:0", backend="botsort")
    if not skin_on:                       # disable the new filter for the baseline
        trk.SKIN_HARD_REJECT = 9.9
        trk.SKIN_SOFT = 9.9
        trk.APPEARANCE_W = 0.0
    cap = cv2.VideoCapture(a.source)
    n = shown = on_skin = jumps = 0
    last = None
    while n < a.max:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
        n += 1
        tracks = trk.update(det.detect(f), f)
        ball = next((t for t in tracks if getattr(t, "class_id", 0) == BALL), None)
        hall = bool(getattr(trk, "ball_was_hallucinated", False))
        if ball is not None and not hall:                 # a REAL emitted ball
            shown += 1
            sr = skin_ratio(f, ball.x1, ball.y1, ball.x2, ball.y2)
            if sr >= 0.45:
                on_skin += 1
            if last is not None and math.hypot(ball.cx - last[0], ball.cy - last[1]) > 150:
                jumps += 1
            last = (ball.cx, ball.cy)
    cap.release()
    return n, shown, on_skin, jumps


for label, on in (("BASELINE (filter OFF)", False), ("ANTI-HEAD (filter ON)", True)):
    n, shown, on_skin, jumps = run(on)
    pct = 100 * on_skin / max(shown, 1)
    print(f"  {label:24s}: real_shown={shown:4d}  on_head={on_skin:4d} "
          f"({pct:4.1f}% of shown)  big_jumps={jumps}")
