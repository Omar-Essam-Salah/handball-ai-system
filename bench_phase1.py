"""
Phase-1 micro-benchmark.
Measures:
  - Detector full-frame vs ROI ball-detection latency
  - Tracker latency
  - Velocity-interp latency on skipped frames

Run from project root:
    .venv\Scripts\python.exe bench_phase1.py handball.mp4
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector  import Detector
from tracker   import Tracker
from player_reid import PlayerReID

BALL_CLASS_ID = 32
N_FRAMES = 60   # benchmark over 60 frames


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "handball.mp4"
    if not Path(video).exists():
        print(f"video not found: {video}")
        return 1

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Skip ~10s in to land in the middle of the action
    for _ in range(int(fps * 10)):
        cap.grab()

    frames = []
    for _ in range(N_FRAMES):
        ret, f = cap.read()
        if not ret:
            break
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(f.shape[0] * (1280 / f.shape[1]))))
        frames.append(f)
    cap.release()

    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")

    print("\n=== Detector ===")
    det = Detector(device="cuda:0", model="yolov8s-pose.pt",
                   conf=0.40, ball_conf=0.05, imgsz=640)
    base_tracker = Tracker(device="cuda:0")
    rid = PlayerReID(base_tracker, device="cuda:0")

    # warmup
    _ = det.detect(frames[0])
    _ = rid.update([], frames[0])

    det_times = []
    trk_times = []
    person_counts = []
    ball_counts = []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        dets = det.detect(f)
        t1 = time.perf_counter()
        # mimic pipeline filtering
        clean = [d for d in dets if d.class_id == BALL_CLASS_ID
                                      or d.y2 >= f.shape[0] * 0.30]
        tracks = rid.update(clean, f)
        t2 = time.perf_counter()
        det_times.append((t1 - t0) * 1000)
        trk_times.append((t2 - t1) * 1000)
        person_counts.append(sum(1 for t in tracks if t.class_id == 0))
        ball_counts.append(sum(1 for t in tracks if t.class_id == BALL_CLASS_ID))

    det_arr = np.array(det_times)
    trk_arr = np.array(trk_times)
    print(f"\n--- Latency (ms over {N_FRAMES} frames) ---")
    print(f"Detector:  mean={det_arr.mean():.1f}  median={np.median(det_arr):.1f}  "
          f"p90={np.percentile(det_arr,90):.1f}  max={det_arr.max():.1f}")
    print(f"Tracker:   mean={trk_arr.mean():.1f}  median={np.median(trk_arr):.1f}  "
          f"p90={np.percentile(trk_arr,90):.1f}  max={trk_arr.max():.1f}")
    print(f"Combined:  mean={(det_arr+trk_arr).mean():.1f} ms/frame  "
          f"=> ~{1000.0 / (det_arr+trk_arr).mean():.1f} FPS upper bound")

    print(f"\n--- Detection counts ---")
    print(f"persons/frame: mean={np.mean(person_counts):.1f} max={max(person_counts)}")
    print(f"balls/frame:   mean={np.mean(ball_counts):.1f}   max={max(ball_counts)}")
    print(f"frames with ball detected: {sum(1 for c in ball_counts if c>0)}/{len(ball_counts)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
