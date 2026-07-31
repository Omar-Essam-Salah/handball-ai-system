"""Test CourtMask against several frames of a broadcast video."""
import sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from court_mask import CourtMask


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "hand3.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    cm = CourtMask(1280, 720)

    for skip_s in [0, 10, 30, 60, 90, 120, 180]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(skip_s * fps))
        ret, f = cap.read()
        if not ret:
            continue
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(f.shape[0] * (1280 / f.shape[1]))))
        t0 = time.perf_counter()
        for j in range(5):
            cm.update(f, j)
        dt = (time.perf_counter() - t0) * 1000
        s = cm.stats()
        label = "PLAY" if s["is_play"] else "GRAPHIC/REPLAY"
        print(f"t={skip_s:3d}s  visibility={s['visibility']:.2f}  "
              f"locked_hue={s.get('locked_hue')}  "
              f"good={s.get('good_streak')}  bad={s.get('bad_streak')}  "
              f"{label}  ({dt:.0f}ms / 5 calls)")

        # Save annotated frame
        out = f.copy()
        cm.draw_overlay(out)
        cv2.imwrite(f"court_mask_t{skip_s}.png", out)

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
