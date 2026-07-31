"""
Quick diagnostic: take ONE frame from the video, push it through the full
pipeline stages, and report what each stage produced.

Run from project root:
    .venv\Scripts\python.exe diagnose.py handball.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector  import Detector
from tracker   import Tracker
from player_reid import PlayerReID
from team_color  import AutoColorDetector

BALL_CLASS_ID = 32

def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "handball.mp4"
    if not Path(video).exists():
        print(f"[ERR] video not found: {video}")
        return 1

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERR] cv2 cannot open: {video}")
        return 1

    # Skip ahead 5 seconds so we land on a meaningful frame, not a logo title screen
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = int(fps * 5)
    for _ in range(skip):
        cap.grab()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERR] could not read frame")
        return 1

    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))))

    print(f"\n[1] Frame loaded: {frame.shape[1]}x{frame.shape[0]}")

    # ── Stage A: Detector ─────────────────────────────────────────
    print("\n[2] Loading Detector (yolov8s-pose + yolov8n)…")
    det = Detector(device="cuda:0", model="yolov8s-pose.pt",
                   conf=0.40, ball_conf=0.15, imgsz=640)
    raw = det.detect(frame)
    persons = [d for d in raw if d.class_id == 0]
    balls   = [d for d in raw if d.class_id == BALL_CLASS_ID]
    print(f"   raw detections: {len(raw)}")
    print(f"   persons: {len(persons)}")
    print(f"   balls:   {len(balls)}")

    if persons:
        confs = [round(p.confidence, 2) for p in persons]
        areas = [int((p.x2-p.x1)*(p.y2-p.y1)) for p in persons]
        print(f"   person confs: {confs}")
        print(f"   person areas: {areas}")
    if balls:
        confs = [round(b.confidence, 2) for b in balls]
        coords = [(int(b.x1+b.x2)//2, int(b.y1+b.y2)//2) for b in balls]
        print(f"   ball confs: {confs}")
        print(f"   ball centroids: {coords}")

    # ── Stage B: y2 filter (the cut applied in pipeline.py) ──────
    H = frame.shape[0]
    after_y_filter = [d for d in raw if d.class_id == BALL_CLASS_ID
                                          or d.y2 >= H * 0.30]
    print(f"\n[3] After y2>={H*0.30:.0f} filter: "
          f"{sum(1 for d in after_y_filter if d.class_id==0)} persons, "
          f"{sum(1 for d in after_y_filter if d.class_id==BALL_CLASS_ID)} balls")

    # ── Stage C: Tracker ─────────────────────────────────────────
    print("\n[4] Pushing through BoT-SORT tracker…")
    base_tracker = Tracker(device="cuda:0")
    tracks_a = base_tracker.update(after_y_filter, frame)
    print(f"   tracks after frame 1: {len(tracks_a)} "
          f"(persons={sum(1 for t in tracks_a if t.class_id==0)}, "
          f"balls={sum(1 for t in tracks_a if t.class_id==BALL_CLASS_ID)})")

    # BoT-SORT often needs a 2nd frame for persons; push the same frame again
    tracks_b = base_tracker.update(after_y_filter, frame)
    print(f"   tracks after frame 2: {len(tracks_b)} "
          f"(persons={sum(1 for t in tracks_b if t.class_id==0)}, "
          f"balls={sum(1 for t in tracks_b if t.class_id==BALL_CLASS_ID)})")

    if tracks_b:
        for t in tracks_b[:5]:
            print(f"     tid={t.track_id} cls={t.class_id} "
                  f"box=({t.x1:.0f},{t.y1:.0f},{t.x2:.0f},{t.y2:.0f}) "
                  f"conf={t.confidence:.2f}")

    # ── Stage D: PlayerReID wrap ─────────────────────────────────
    print("\n[5] Pushing through PlayerReID wrapper…")
    rid_tracker = PlayerReID(base_tracker, device="cuda:0")
    tracks_c = rid_tracker.update(after_y_filter, frame)
    print(f"   tracks after PlayerReID: {len(tracks_c)} "
          f"(persons={sum(1 for t in tracks_c if t.class_id==0)}, "
          f"balls={sum(1 for t in tracks_c if t.class_id==BALL_CLASS_ID)})")

    # ── Save annotated frame so user can see what was detected ──
    out = frame.copy()
    for d in raw:
        col = (0, 255, 255) if d.class_id == BALL_CLASS_ID else (50, 200, 50)
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
        cv2.putText(out, f"{d.class_id}:{d.confidence:.2f}",
                    (int(d.x1), int(d.y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    out_path = Path("diagnose_output.png")
    cv2.imwrite(str(out_path), out)
    print(f"\n[6] Annotated frame written to: {out_path.resolve()}")
    print("\n=== SUMMARY ===")
    if not persons:
        print("  Detector returned 0 persons — pose model is the problem.")
    elif not tracks_b:
        print("  Detector found persons but BoT-SORT returned empty — tracker is the problem.")
    elif sum(1 for t in tracks_b if t.class_id==0) == 0:
        print("  Tracker dropped all persons — investigate tracker exception path.")
    else:
        print("  Pipeline stages all working — issue must be inside rules engine or visualization.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
