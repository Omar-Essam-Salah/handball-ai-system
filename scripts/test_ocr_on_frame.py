"""
Smoke-test: run JerseyOCR on a single frame from the video and report what it
managed to read for each detected player.
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
    if not ret:
        print("could not read frame"); return 1
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))))

    print(f"frame: {frame.shape[1]}x{frame.shape[0]}")

    det = Detector(device="cuda:0", model="yolov8s-pose.pt",
                   conf=0.40, ball_conf=0.05, imgsz=640)
    ocr = JerseyOCR(device="cuda:0", read_every=1, max_concurrent_reads=20)

    dets = det.detect(frame)
    persons = [d for d in dets if d.class_id == 0]
    print(f"persons detected: {len(persons)}")

    # Wrap detections as faux tracks (the ocr expects .x1/.y1/.x2/.y2/.track_id/.keypoints/.class_id)
    class T:
        pass
    fake_tracks = []
    for i, p in enumerate(persons):
        tk = T()
        tk.x1, tk.y1, tk.x2, tk.y2 = p.x1, p.y1, p.x2, p.y2
        tk.class_id = 0
        tk.track_id = i + 1
        tk.keypoints = p.keypoints
        fake_tracks.append(tk)

    t0 = time.perf_counter()
    ocr.feed_tracks(frame, fake_tracks, frame_n=1)
    print(f"\nOCR feed_tracks took {(time.perf_counter() - t0)*1000:.0f} ms")

    print(f"\nOCR stats: {ocr.stats()}")
    print("\nPer-track candidate votes:")
    for tk in fake_tracks:
        v = ocr.candidate_votes(tk.track_id)
        if v:
            top = sorted(v.items(), key=lambda kv: -kv[1])
            best = top[0]
            print(f"  tid={tk.track_id}  bbox={int(tk.x1)},{int(tk.y1)}-{int(tk.x2)},{int(tk.y2)}"
                  f"  votes={dict(top)}  top→#{best[0]} ({best[1]:.2f})")

    # Annotate result
    out = frame.copy()
    for tk in fake_tracks:
        v = ocr.candidate_votes(tk.track_id)
        col = (50, 200, 50) if v else (160, 160, 160)
        cv2.rectangle(out, (int(tk.x1), int(tk.y1)), (int(tk.x2), int(tk.y2)), col, 2)
        if v:
            top = sorted(v.items(), key=lambda kv: -kv[1])[0]
            cv2.putText(out, f"#{top[0]}", (int(tk.x1), int(tk.y1) - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 2)
    cv2.imwrite("ocr_test_output.png", out)
    print("\nAnnotated frame saved to ocr_test_output.png")
    return 0

if __name__ == "__main__":
    sys.exit(main())
