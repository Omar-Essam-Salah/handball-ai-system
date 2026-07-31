"""
A/B benchmark: base Detector vs SAHIDetector on the same frames.
Reports per-frame:
  * persons detected
  * smallest person bbox (proxy for distant-player recall)
  * latency
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector       import Detector
from sahi_detector  import SAHIDetector


N_FRAMES = 30
SKIP_S   = 60  # land in the middle of active play


def bench(label, det, frames):
    times = []
    person_counts = []
    smallest_h = []
    ball_count = 0
    for f in frames:
        t0 = time.perf_counter()
        dets = det.detect(f)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        persons = [d for d in dets if d.class_id == 0]
        balls   = [d for d in dets if d.class_id == 32]
        person_counts.append(len(persons))
        if persons:
            smallest_h.append(min(d.height for d in persons))
        if balls:
            ball_count += 1
    t = np.array(times)
    pc = np.array(person_counts)
    sh = np.array(smallest_h) if smallest_h else np.array([0])
    print(f"\n[{label}]")
    print(f"  latency: mean={t.mean():.0f}ms  p90={np.percentile(t,90):.0f}ms")
    print(f"  persons/frame: mean={pc.mean():.1f}  max={pc.max()}")
    print(f"  smallest player height (px): mean={sh.mean():.1f}  min={sh.min():.0f}")
    print(f"  frames with ball: {ball_count}/{len(frames)}")
    return pc.mean(), sh.min(), t.mean()


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "hand3.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * SKIP_S))

    frames = []
    for _ in range(N_FRAMES):
        ret, f = cap.read()
        if not ret: break
        if f.shape[1] > 1920:
            f = cv2.resize(f, (1920, int(f.shape[0] * (1920 / f.shape[1]))))
        frames.append(f)
    cap.release()
    print(f"Loaded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    print("=" * 60)
    print(" base Detector (full-frame YOLO @ imgsz=640)")
    print("=" * 60)
    base_det = Detector(device="cuda:0", conf=0.40, ball_conf=0.05, imgsz=640)
    # warmup
    _ = base_det.detect(frames[0])
    base_pc, base_min, base_t = bench("base Detector", base_det, frames)
    del base_det

    print("\n" + "=" * 60)
    print(" SAHIDetector (sliced 640x640, overlap=0.25, pose@960)")
    print("=" * 60)
    sahi_det = SAHIDetector(device="cuda:0", conf=0.40, ball_conf=0.05, imgsz=640)
    _ = sahi_det.detect(frames[0])
    sahi_pc, sahi_min, sahi_t = bench("SAHIDetector", sahi_det, frames)

    print("\n" + "=" * 60)
    print(" Δ comparison")
    print("=" * 60)
    print(f"  persons/frame:           base={base_pc:.1f}   sahi={sahi_pc:.1f}   "
          f"Δ=+{sahi_pc - base_pc:.1f}")
    print(f"  smallest player (px):    base={base_min:.0f}    sahi={sahi_min:.0f}    "
          f"smaller is better (means catching tinier players)")
    print(f"  latency (ms):            base={base_t:.0f}     sahi={sahi_t:.0f}     "
          f"({sahi_t / max(base_t, 1):.1f}× slower)")


if __name__ == "__main__":
    main()
