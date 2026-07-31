"""
Home Test Suite — validates every component before going to the real court.

Run:
    python test_home.py

Protocol (do these in order):
  1. Stand in front of camera alone        → person detection + tracking
  2. Wear 2 different colored shirts       → auto color calibration
  3. Hold a ball and move it fast          → ball detection + speed
  4. Walk toward/away from ball            → possession detection
  5. Throw ball fast toward camera         → shot detection trigger
  6. Leave frame and re-enter              → re-identification (track ID stable?)

Results are printed live and saved to  test_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# ── test thresholds ────────────────────────────────────────────────────────────
TARGET_FPS          = 20.0    # minimum acceptable FPS for real court
TARGET_CALIB_S      = 15.0    # color calibration should complete within N seconds
TARGET_BALL_DETECTS = 5       # ball must be detected at least this many frames
TARGET_TRACK_STABLE = 10      # same track ID held for at least N consecutive frames


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_url() -> str:
    cfg_path = ROOT / "config" / "camera.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}
    ip   = cfg.get("ip",       "172.16.5.174")
    user = cfg.get("user",     "admin")
    pw   = cfg.get("password", "")
    port = cfg.get("rtsp_port", 554)
    ch   = cfg.get("channel",  102)
    return f"rtsp://{user}:{pw}@{ip}:{port}/Streaming/Channels/{ch}"


def _bar(label: str, val, target, unit: str = "", width: int = 30) -> str:
    if target > 0:
        fill = min(1.0, val / target)
    else:
        fill = 1.0
    blocks = int(fill * width)
    bar    = "█" * blocks + "░" * (width - blocks)
    ok     = "✓" if fill >= 1.0 else "…"
    return f"  {ok} {label:28s} [{bar}] {val:.1f}{unit} / {target}{unit}"


# ── test runner ───────────────────────────────────────────────────────────────

def run_test():
    from capture    import CaptureThread
    from detector   import Detector
    from tracker    import Tracker
    from team_color import AutoColorDetector
    from handball_rules import HandballEngine

    print("\n" + "═" * 62)
    print("  Handball AI — Home Test Suite")
    print("  Follow the on-screen instructions step by step.")
    print("═" * 62 + "\n")

    url = _load_url()
    print(f"  Camera : {url.replace(url.split('@')[0].split('//')[1], '****:****')}\n")

    # ── init ──────────────────────────────────────────────────────────────────
    print("  Loading AI models…")
    detector  = Detector(device="cuda:0", model="yolov8n.pt", conf=0.25)
    tracker   = Tracker(device="cuda:0")
    color_det = AutoColorDetector(min_crops=25)
    capture   = CaptureThread(url)
    capture.start()

    # Wait for first frame
    print("  Waiting for camera stream…")
    for _ in range(50):
        pkt = capture.get_frame(timeout=0.5)
        if pkt is not None:
            break
    else:
        print("  ✗  No frames received — check camera connection")
        return

    h, w = pkt.frame.shape[:2]
    rules = HandballEngine(frame_w=w, frame_h=h, fps=25.0)
    print(f"  Frame size: {w}×{h}\n")

    # ── results dict ─────────────────────────────────────────────────────────
    results = {
        "fps_avg":          0.0,
        "fps_min":          999.0,
        "person_detected":  False,
        "max_persons":      0,
        "ball_detected":    False,
        "ball_frames":      0,
        "ball_max_speed":   0.0,
        "color_calibrated": False,
        "color_time_s":     0.0,
        "teams_found":      [],
        "tracking_stable":  False,
        "max_stable_id":    0,
        "shot_triggered":   False,
        "possession_works": False,
        "passed":           False,
    }

    # ── test state ────────────────────────────────────────────────────────────
    frame_n       = 0
    fps_ema       = 0.0
    fps_samples   = []
    t_prev        = time.perf_counter()
    t_start       = t_prev
    calib_start   = t_start
    calib_done_t  = None
    id_streak     = {}      # track_id → consecutive frames seen
    test_phase    = 0
    phase_labels  = [
        "PHASE 1 — Stand in front of camera (person detection)",
        "PHASE 2 — Keep moving (color calibration)",
        "PHASE 3 — Hold a ball and move it fast (ball detection)",
        "PHASE 4 — Walk toward / away from ball (possession)",
        "PHASE 5 — Throw ball FAST toward camera (shot detection)",
        "COMPLETE",
    ]

    print(f"  ► {phase_labels[0]}")
    print("  Press Q to stop early  |  Space to skip to next phase\n")

    PHASE_DURATION = 12.0   # seconds per phase

    while True:
        pkt = capture.get_frame(timeout=0.3)
        if pkt is None:
            continue

        frame   = pkt.frame
        frame_n += 1

        # ── AI ────────────────────────────────────────────────────────────────
        dets   = detector.detect(frame)
        tracks = tracker.update(dets, frame)

        ball_track = next((t for t in tracks if t.is_ball), None)
        persons    = [t for t in tracks if not t.is_ball]

        color_det.feed(frame, tracks)
        for t in persons:
            t.team = color_det.classify(frame, t)

        rules.update(tracks, ball_track, frame_n)

        # ── FPS ───────────────────────────────────────────────────────────────
        now      = time.perf_counter()
        elapsed  = now - t_start
        dt       = max(now - t_prev, 1e-6)
        fps_ema  = 0.85 * fps_ema + 0.15 * (1.0 / dt)
        t_prev   = now
        fps_samples.append(fps_ema)

        # ── collect metrics ───────────────────────────────────────────────────
        # Persons
        if persons:
            results["person_detected"] = True
            results["max_persons"]     = max(results["max_persons"], len(persons))

        # Ball
        if ball_track is not None:
            results["ball_detected"]  = True
            results["ball_frames"]   += 1
            vx, vy, spd = rules._ball_velocity()
            results["ball_max_speed"] = max(results["ball_max_speed"], spd)

        # Color calibration
        if color_det.is_calibrated and calib_done_t is None:
            calib_done_t              = now
            results["color_time_s"]   = calib_done_t - calib_start
            results["color_calibrated"] = True
            results["teams_found"]    = [p.label for p in color_det.profiles]

        # Tracking stability
        cur_ids = {t.track_id for t in persons}
        for tid in cur_ids:
            id_streak[tid] = id_streak.get(tid, 0) + 1
        if id_streak:
            best = max(id_streak.values())
            results["max_stable_id"] = max(results["max_stable_id"], best)
            if best >= TARGET_TRACK_STABLE:
                results["tracking_stable"] = True

        # Shot / possession
        if rules.shots["team_a"] + rules.shots["team_b"] > 0:
            results["shot_triggered"] = True
        if rules.possession is not None:
            results["possession_works"] = True

        # ── phase advance ─────────────────────────────────────────────────────
        phase_elapsed = elapsed - test_phase * PHASE_DURATION
        if phase_elapsed >= PHASE_DURATION and test_phase < len(phase_labels) - 2:
            test_phase += 1
            print(f"\n  ► {phase_labels[test_phase]}")

        # ── draw on frame ─────────────────────────────────────────────────────
        vis = frame.copy()

        for t in tracks:
            if t.is_ball:
                col = (0, 220, 220)
                lbl = f"BALL #{t.track_id}"
            elif t.team == "team_a":
                col = color_det.profiles[0].color_bgr if color_det.is_calibrated else (60, 60, 220)
                lbl = f"A #{t.track_id}"
            elif t.team == "team_b":
                col = color_det.profiles[1].color_bgr if color_det.is_calibrated else (220, 80, 0)
                lbl = f"B #{t.track_id}"
            else:
                col = (130, 130, 130)
                lbl = f"? #{t.track_id}"

            cv2.rectangle(vis, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), col, 2)
            cv2.putText(vis, lbl, (int(t.x1), max(int(t.y1) - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # HUD
        phase_pct = min(1.0, phase_elapsed / PHASE_DURATION)
        bar_w     = int((w - 20) * phase_pct)
        cv2.rectangle(vis, (10, h - 16), (w - 10, h - 6), (40, 40, 40), -1)
        cv2.rectangle(vis, (10, h - 16), (10 + bar_w, h - 6), (0, 180, 80), -1)

        cv2.putText(vis, f"FPS: {fps_ema:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Persons: {len(persons)}  Ball: {'YES' if ball_track else 'no'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        if not color_det.is_calibrated:
            pct = int(color_det.calibration_progress * 100)
            cv2.putText(vis, f"Color calibrating… {pct}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        else:
            labels = " | ".join(p.label for p in color_det.profiles)
            cv2.putText(vis, f"Colors locked: {labels}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

        phase_lbl = phase_labels[min(test_phase, len(phase_labels)-1)]
        cv2.putText(vis, phase_lbl, (10, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        if rules.shots["team_a"] + rules.shots["team_b"] > 0:
            cv2.putText(vis, "SHOT DETECTED!", (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)

        cv2.imshow("Home Test — Q=stop  Space=next phase", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and test_phase < len(phase_labels) - 2:
            test_phase += 1
            print(f"\n  ► {phase_labels[test_phase]}")

        # Auto-end after all phases
        if elapsed >= PHASE_DURATION * (len(phase_labels) - 1):
            break

    capture.stop()
    cv2.destroyAllWindows()

    # ── final FPS stats ───────────────────────────────────────────────────────
    if fps_samples:
        results["fps_avg"] = round(sum(fps_samples) / len(fps_samples), 1)
        results["fps_min"] = round(min(fps_samples), 1)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  TEST RESULTS")
    print("═" * 62)

    checks = [
        ("FPS average",       results["fps_avg"],        TARGET_FPS,          " fps"),
        ("FPS minimum",       results["fps_min"],        TARGET_FPS * 0.6,    " fps"),
        ("Ball frames seen",  results["ball_frames"],    TARGET_BALL_DETECTS, " frames"),
        ("Stable track len",  results["max_stable_id"],  TARGET_TRACK_STABLE, " frames"),
    ]
    for label, val, target, unit in checks:
        print(_bar(label, val, target, unit))

    print()
    bools = [
        ("Person detected",     results["person_detected"]),
        ("Ball detected",       results["ball_detected"]),
        ("Color calibrated",    results["color_calibrated"]),
        ("Tracking stable",     results["tracking_stable"]),
        ("Possession detected", results["possession_works"]),
        ("Shot triggered",      results["shot_triggered"]),
    ]
    all_pass = True
    for label, ok in bools:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {label}")
        if not ok:
            all_pass = False

    if results["color_calibrated"]:
        print(f"\n  Color profiles: {', '.join(results['teams_found'])}")
        print(f"  Calibration time: {results['color_time_s']:.1f}s")

    print()
    if results["fps_avg"] >= TARGET_FPS and all_pass:
        results["passed"] = True
        print("  ✓✓  ALL TESTS PASSED — system ready for real court")
    else:
        print("  ⚠   Some tests incomplete — see issues below:")
        if results["fps_avg"] < TARGET_FPS:
            print(f"      • FPS too low ({results['fps_avg']:.1f}) →"
                  " use --model yolov8n.pt or reduce resolution")
        if not results["ball_detected"]:
            print("      • Ball not detected → hold ball closer to camera"
                  " | lower --conf to 0.20")
        if not results["color_calibrated"]:
            print("      • Color not calibrated → move around more"
                  " | lower --min-crops")
        if not results["tracking_stable"]:
            print("      • Tracking unstable → stay in frame longer")
        if not results["shot_triggered"]:
            print("      • Shot not triggered → throw ball FAST toward camera")

    # Save results
    out_path = ROOT / "test_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    run_test()
