"""Render a single mini-court inset onto a real frame to verify visuals."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from goal_replay import GoalReplayRecorder, _Snap, _PendingGoal


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "hand3.mp4"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 30))   # 30s in
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("could not read frame"); return 1
    if frame.shape[1] > 1280:
        frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))))

    # Synthetic players + trajectory simulating a left-side shot
    fake_tracks = [
        {"tid": 1, "cls": 0, "x1": 940, "y1": 360, "x2": 1000, "y2": 480, "cx": 970, "cy": 420, "team": "team_a", "kpts": None},
        {"tid": 2, "cls": 0, "x1": 600, "y1": 380, "x2": 660, "y2": 500, "cx": 630, "cy": 440, "team": "team_a", "kpts": None},
        {"tid": 3, "cls": 0, "x1": 400, "y1": 360, "x2": 460, "y2": 480, "cx": 430, "cy": 420, "team": "team_b", "kpts": None},
        {"tid": 4, "cls": 0, "x1": 500, "y1": 400, "x2": 560, "y2": 520, "cx": 530, "cy": 460, "team": "team_b", "kpts": None},
        {"tid": 5, "cls": 0, "x1":  90, "y1": 320, "x2": 140, "y2": 460, "cx": 115, "cy": 390, "team": "goalkeeper", "kpts": None},
    ]
    # Trajectory from shooter (970, 420) curving toward goal (100, 360)
    traj_pts = []
    for i in range(20):
        t = i / 19.0
        x = 970 + (100 - 970) * t
        y = 420 + (360 - 420) * t - 30 * np.sin(np.pi * t)   # slight arc
        traj_pts.append((x, y, True))

    snap = _Snap(frame=frame, fn=750, ts=0.0, ball_xy=(970, 420), ball_real=True,
                  tracks=fake_tracks)

    pg = _PendingGoal(
        event={"event": "goal", "team": "team_a", "side": "left"},
        scorer_label="A#11",
        team="team_a",
        team_color=(255, 80, 80),
        opponent_color=(50, 180, 255),
        trigger_fn=750,
        pre_snaps=[snap], post_needed=0,
    )

    rec = GoalReplayRecorder(out_dir="reports/_test_mini",
                              fps=25.0, downscale=1.0,
                              write_mp4=False, write_png=False, write_json=False)
    rec.downscale = 1.0  # ensure mapping is straightforward

    out = rec._render_overlay(snap, traj_pts, [(970, 420), (940, 410)],
                                shooter_tid=1, pg=pg, is_key_frame=True)
    cv2.imwrite("mini_court_demo.png", out)
    print(f"Saved demo to: mini_court_demo.png  ({out.shape[1]}x{out.shape[0]})")

    # Also save just the mini-court for clean inspection
    mini = rec._build_mini_court(snap, traj_pts, 1, pg, frame.shape[1], frame.shape[0])
    cv2.imwrite("mini_court_only.png", mini)
    print(f"Saved mini-court alone to: mini_court_only.png  ({mini.shape[1]}x{mini.shape[0]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
