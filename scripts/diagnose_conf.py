"""Check the ball model's actual confidence distribution on a real video."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "hand3.mp4"
    weights = "handball_ball.pt"

    if not Path(weights).exists():
        print(f"[ERR] {weights} not found")
        return 1
    if not Path(video).exists():
        print(f"[ERR] {video} not found")
        return 1

    from ultralytics import YOLO
    model = YOLO(weights, task="detect")
    print(f"Loaded {weights}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 60))   # 60s in
    print(f"Video {video}  fps={fps}")
    print()

    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    n_frames = 60
    all_confs = []
    ball_in_n = [0] * len(thresholds)

    for i in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        r = model(f, conf=0.01, verbose=False)[0]
        if len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy().tolist()
            all_confs.extend(confs)
            top = max(confs)
            for j, th in enumerate(thresholds):
                if top >= th:
                    ball_in_n[j] += 1
    cap.release()

    print(f"Tested {n_frames} frames at 60s mark")
    print()

    if not all_confs:
        print("  NO detections AT ALL even at conf=0.01")
        print("  → model is broken (class mismatch or completely untrained)")
        return 1

    print(f"  Total raw detections (conf>=0.01): {len(all_confs)}")
    print()
    print(f"  Confidence stats:")
    print(f"    min:    {min(all_confs):.3f}")
    print(f"    median: {np.median(all_confs):.3f}")
    print(f"    mean:   {np.mean(all_confs):.3f}")
    print(f"    max:    {max(all_confs):.3f}")
    print()
    print(f"  Frames with at least 1 ball per threshold:")
    for th, cnt in zip(thresholds, ball_in_n):
        bar = "#" * int(cnt / n_frames * 40)
        print(f"    conf>={th:.2f}: {cnt:2d}/{n_frames}  [{bar}]")
    print()

    median = float(np.median(all_confs))
    p25 = float(np.percentile(all_confs, 25))
    rec = max(0.05, round(p25 - 0.02, 2))
    print(f"  >>> RECOMMENDATION:")
    print(f"      Lower ball_conf auto-bump to {rec:.2f}")
    print(f"      (current code raises it to 0.20 which loses most detections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
