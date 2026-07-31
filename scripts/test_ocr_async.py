"""
Demonstrates the async OCR worker:
  - first feed_tracks triggers reader warmup (one-time ~3s)
  - subsequent feed_tracks calls return in milliseconds (just enqueueing)
  - the worker processes crops in the background
"""
import sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector  import Detector
from jersey_ocr import JerseyOCR


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "handball.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for _ in range(int(fps * 12)):
        cap.grab()
    ret, frame = cap.read()
    cap.release()
    if not ret: return 1
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))))

    det = Detector(device="cuda:0", model="yolov8s-pose.pt",
                   conf=0.40, ball_conf=0.05, imgsz=640)
    ocr = JerseyOCR(device="cuda:0", read_every=2, max_concurrent_reads=4, async_worker=True)

    dets = det.detect(frame)
    persons = [d for d in dets if d.class_id == 0]

    class T: pass
    fake_tracks = []
    for i, p in enumerate(persons):
        tk = T()
        tk.x1, tk.y1, tk.x2, tk.y2 = p.x1, p.y1, p.x2, p.y2
        tk.class_id = 0
        tk.track_id = i + 1
        tk.keypoints = p.keypoints
        fake_tracks.append(tk)

    print(f"\n--- 5 successive feed_tracks calls (10 fake-frames) ---")
    for fn in range(1, 11):
        t0 = time.perf_counter()
        ocr.feed_tracks(frame, fake_tracks, frame_n=fn)
        dt = (time.perf_counter() - t0) * 1000
        s = ocr.stats()
        print(f"  frame {fn:2d}: feed={dt:6.1f}ms  enqueued={s['reads']}  locked={s['locks']}  "
              f"tracked={s['tracked']}")
        time.sleep(0.05)   # simulate ~20 fps loop

    # Wait for worker to finish remaining jobs
    print("\n--- waiting for background worker to drain... ---")
    time.sleep(8)
    s = ocr.stats()
    print(f"final stats: {s}")
    print("\nLocked jersey numbers:")
    for tid, num in s["locked"].items():
        print(f"  tid={tid}  → #{num}")

    ocr.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
