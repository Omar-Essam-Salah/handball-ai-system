"""
Ball-trace diagnostic. Runs Detector + Tracker on a clip and reports, per frame:
  - how many RAW ball candidates YOLO produced (detection recall)
  - whether the tracker emitted a real ball / phantom(hand) / coast(kalman) / NOTHING
  - lost-streak lengths and big position jumps

This isolates the ball problem: if YOLO rarely produces candidates, it's a
DETECTION problem (model/conf), not a tracking problem.
"""
import sys, argparse, math
from collections import Counter
sys.path.insert(0, "src")

import cv2
import numpy as np
from detector import Detector
from tracker import Tracker

BALL = 32

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="handball.mp4")
ap.add_argument("--max-frames", type=int, default=1200)
ap.add_argument("--ball-conf", type=float, default=0.05)
ap.add_argument("--infer-every", type=int, default=1)
ap.add_argument("--ball-coast", type=int, default=10)
ap.add_argument("--reacquire-after", type=int, default=3)
ap.add_argument("--motion-ball", action="store_true")
ap.add_argument("--split", action="store_true",
                help="every-frame ball detection + ROI (the new path)")
args = ap.parse_args()

det = Detector(device="cuda:0", model="yolov8s-pose.pt", conf=0.40,
               imgsz=640, ball_conf=args.ball_conf)
trk = Tracker(device="cuda:0", backend="botsort", ball_coast_max=args.ball_coast,
              reacquire_after=args.reacquire_after)
mball = None
if args.motion_ball:
    from motion_ball import MotionBall
    mball = MotionBall()

cap = cv2.VideoCapture(args.source)
n = 0
states = Counter()
raw_cand_hist = Counter()
last_pos = None
lost_streak = 0
lost_streaks = []
jumps = 0
cached = []
last_infer = 0

while n < args.max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))))
    n += 1
    do_infer = (n % args.infer_every == 0) or (n == 1)

    SPLIT = getattr(args, "split", False)
    if do_infer:
        if SPLIT:
            players = det.detect_players(frame)
            ballc   = det.detect_ball(frame, roi_center=getattr(trk, "last_ball_pos", None))
            dets = list(players) + list(ballc)
        else:
            dets = det.detect(frame)
        if mball is not None:
            players = [(d.x1, d.y1, d.x2, d.y2) for d in dets
                       if getattr(d, "class_id", 0) != BALL]
            dets = list(dets) + list(mball.find(frame, players))
        n_raw_ball = sum(1 for d in dets if getattr(d, "class_id", 0) == BALL)
        raw_cand_hist[min(n_raw_ball, 5)] += 1
        tracks = trk.update(dets, frame)
        cached = tracks
        last_infer = n
    elif SPLIT:
        # every-frame ball: re-detect ball, extrapolated players unchanged
        players = [t for t in cached if getattr(t, "class_id", 0) != BALL]
        ballc   = det.detect_ball(frame, roi_center=getattr(trk, "last_ball_pos", None))
        n_raw_ball = len(ballc); raw_cand_hist[min(n_raw_ball, 5)] += 1
        bt = trk.update_ball(ballc, players, frame)
        tracks = players + ([bt] if bt is not None else [])
    else:
        tracks = cached

    ball = next((t for t in tracks if getattr(t, "class_id", 0) == BALL), None)
    hall = bool(getattr(trk._tracker if hasattr(trk, "_tracker") else trk,
                        "ball_was_hallucinated", False))
    coast = getattr(trk, "_ball_coast", 0)

    if ball is None:
        states["lost"] += 1
        # Was there a raw candidate this frame that the tracker rejected?
        if do_infer and n_raw_ball > 0:
            states["lost_BUT_had_candidate"] += 1
        else:
            states["lost_no_candidate"] += 1
        lost_streak += 1
    else:
        if not hall:
            states["real"] += 1
        elif coast > 0:
            states["coast"] += 1
        else:
            states["hand_phantom"] += 1
        if lost_streak > 0:
            lost_streaks.append(lost_streak)
            lost_streak = 0
        bx, by = ball.cx, ball.cy
        if last_pos is not None:
            if math.hypot(bx - last_pos[0], by - last_pos[1]) > 150:
                jumps += 1
        last_pos = (bx, by)

if lost_streak > 0:
    lost_streaks.append(lost_streak)

cap.release()
total = states["real"] + states["coast"] + states["hand_phantom"] + states["lost"]
print(f"\n=== BALL TRACE DIAGNOSTIC  ({args.source}, {total} frames, "
      f"infer_every={args.infer_every}, ball_conf={args.ball_conf}) ===")
for k in ("real", "coast", "hand_phantom", "lost"):
    v = states.get(k, 0)
    print(f"  {k:12s}: {v:5d}  ({100*v/max(total,1):5.1f}%)")
shown = total - states.get("lost", 0)
print(f"  {'BALL SHOWN':12s}: {shown:5d}  ({100*shown/max(total,1):5.1f}%)   <-- continuity")
lbc = states.get("lost_BUT_had_candidate", 0)
lnc = states.get("lost_no_candidate", 0)
print(f"    of lost: {lbc} had a candidate the tracker REJECTED (recoverable), "
      f"{lnc} had NO candidate (model miss)")
print(f"\n  RAW YOLO ball candidates per inference frame:")
inf_total = sum(raw_cand_hist.values())
for k in sorted(raw_cand_hist):
    label = f"{k}+" if k == 5 else str(k)
    v = raw_cand_hist[k]
    print(f"    {label} candidates: {v:5d} frames ({100*v/max(inf_total,1):5.1f}%)")
zero = raw_cand_hist.get(0, 0)
print(f"\n  >> YOLO saw NO ball at all in {100*zero/max(inf_total,1):.1f}% of inference frames")
if lost_streaks:
    print(f"  >> lost streaks: count={len(lost_streaks)}  "
          f"longest={max(lost_streaks)}  mean={sum(lost_streaks)/len(lost_streaks):.1f} frames")
print(f"  >> big position jumps (>150px): {jumps}")
