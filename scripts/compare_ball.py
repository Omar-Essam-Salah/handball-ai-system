"""
Head-to-head: OLD per-frame YOLO ball (Detector+Tracker) vs NEW temporal TrackNet
ball, on the SAME frames. Draws both balls per frame + flags YOLO false positives
that sit on skin (head/marker). Saves annotated frames and prints a summary.

    .venv\\Scripts\\python.exe scripts\\compare_ball.py --source handball.mp4 --frac 0.4 --n 8

RED  = YOLO+tracker ball   (dashed = phantom/guess)
GREEN= TrackNet ball
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import cv2

from detector import Detector
from tracker import Tracker
from ball_appearance import skin_ratio
from training.tracknet.infer import TrackNetBall

BALL = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="handball.mp4")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--every", type=int, default=18)
    ap.add_argument("--weights", default="training/tracknet/runs/ball_tracknet/best.pt")
    ap.add_argument("--out", default="reports/compare")
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    det = Detector(device="cuda:0", model="yolov8s-pose.pt", conf=0.40, imgsz=640, ball_conf=0.05)
    trk = Tracker(device="cuda:0", backend="botsort")
    tnb = TrackNetBall(a.weights, device="cuda:0", threshold=0.5)

    cap = cv2.VideoCapture(a.source)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.frac * tot))
    stem = Path(a.source).stem

    n = saved = 0
    stats = {"yolo_shown": 0, "yolo_on_skin": 0, "tn_shown": 0, "frames": 0}
    while saved < a.n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])))
        n += 1
        stats["frames"] += 1

        # OLD path
        tracks = trk.update(det.detect(f), f)
        yolo_ball = next((t for t in tracks if getattr(t, "class_id", 0) == BALL), None)
        yolo_guess = bool(getattr(trk, "ball_was_hallucinated", False))
        # NEW path (must run every frame to keep its 3-frame buffer continuous)
        tn = tnb.locate(f)

        if yolo_ball is not None:
            stats["yolo_shown"] += 1
            sr = skin_ratio(f, yolo_ball.x1, yolo_ball.y1, yolo_ball.x2, yolo_ball.y2)
            if sr >= 0.45:
                stats["yolo_on_skin"] += 1
        if tn is not None:
            stats["tn_shown"] += 1

        if n % a.every != 0:
            continue
        vis = f.copy()
        for t in tracks:                      # players for context
            if getattr(t, "class_id", 0) != BALL:
                cv2.rectangle(vis, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), (90, 90, 90), 1)
        if yolo_ball is not None:
            c = (0, 0, 255)
            cv2.circle(vis, (int(yolo_ball.cx), int(yolo_ball.cy)), 13, c, 2 if not yolo_guess else 1)
            cv2.putText(vis, "YOLO" + (" guess" if yolo_guess else ""),
                        (int(yolo_ball.cx) + 15, int(yolo_ball.cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        if tn is not None:
            x, y, conf = tn
            cv2.circle(vis, (int(x), int(y)), 10, (0, 220, 0), 2)
            cv2.putText(vis, f"TrackNet {conf:.2f}", (int(x) + 14, int(y) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(vis, "RED=YOLO ball   GREEN=TrackNet ball", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        p = outdir / f"{stem}_{a.frac:.2f}_{saved:02d}.jpg"
        cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("saved", p.name)
        saved += 1
    cap.release()

    fr = max(stats["frames"], 1)
    print(f"\n=== {a.source} ({stats['frames']} frames) ===")
    print(f"  YOLO ball shown   : {stats['yolo_shown']} ({100*stats['yolo_shown']/fr:.0f}%)"
          f"   of which ON SKIN/head: {stats['yolo_on_skin']}")
    print(f"  TrackNet ball shown: {stats['tn_shown']} ({100*stats['tn_shown']/fr:.0f}%)")


if __name__ == "__main__":
    main()
