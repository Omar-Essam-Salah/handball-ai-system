# Handball Studio — System Architecture

A production-grade sports match-analysis platform. The heavy AI runs on a GPU
PC; phones/tablets are premium thin clients. Everything below is **real,
running code** in this repo — not aspirational.

```
┌──────────────────────────────┐         WiFi / LAN          ┌─────────────────────────┐
│        PC  (GPU host)        │  ◀───────────────────────▶  │   Phone / Tablet client │
│                              │   HTTP : MJPEG + JSON       │                         │
│  ┌────────────────────────┐  │                             │  ┌───────────────────┐  │
│  │  pipeline.py (engine)  │  │   GET  /video  (MJPEG)      │  │  webapp/  (PWA)    │  │
│  │  • Detector            │  │   GET  /status (JSON 700ms) │  │  • loading screen │  │
│  │    - YOLOv8s-pose      │  │   POST /control (overlays,  │  │  • Match dashboard│  │
│  │    - ball net @960     │  │          AI thresholds)     │  │  • Players/Tactics│  │
│  │  • Tracker (BoT-SORT+  │  │   POST /cmd/<k> (R/F/A/B/D) │  │  • Control center │  │
│  │    Kalman+re-acquire)  │  │   POST /open  /stop         │  │    (toggles+sliders)│ │
│  │  • PlayerReID + MSR    │  │   GET  /scan  (cameras)     │  └───────────────────┘  │
│  │  • HandballEngine      │  │   GET  /goals /goal/<n>      │   served by the PC, so  │
│  │    (events/score)      │  │   GET  /ping  (discovery)    │   ALL UI edits are live │
│  │  • GoalReplay          │  │                             │  ┌───────────────────┐  │
│  └─────────┬──────────────┘  │                             │  │ android/ WebView  │  │
│            │ gui_mode hooks   │                             │  │ shell → the APK   │  │
│  ┌─────────▼──────────────┐  │                             │  │ (auto-finds PC)   │  │
│  │  mobile_server.py      │  │                             │  └───────────────────┘  │
│  │  FastAPI + uvicorn     │  │                             └─────────────────────────┘
│  │  • PipelineService     │
│  │    (runs engine in a   │   camera_discovery.py  ── subnet scan ──▶  IP cameras (RTSP)
│  │     thread)            │
│  │  • shared controls{}   │
│  └────────────────────────┘
└──────────────────────────────┘
```

## 1. How the three layers communicate

**Engine ↔ Server (in-process, zero-copy):** `pipeline.run(gui_mode=True, …)`
takes four hooks so it never needs its own window:
- `frame_callback(bgr)` — every annotated frame → server JPEG-encodes the latest.
- `status_callback(dict)` — every ~5 frames → score, possession, players, events, controls.
- `command_queue` — discrete actions (recalibrate/flip/goal/debug) as 1-char codes.
- `controls` — a **shared mutable dict** read each frame for overlay toggles + live
  AI thresholds. The server mutates it in place from `POST /control`, so changes
  take effect on the very next frame with no restart.

**Server ↔ Client (HTTP over LAN):** the phone uses only **relative** URLs, so it
talks to whatever address served it → no hardcoded IP (dynamic by construction).
Video is `multipart/x-mixed-replace` MJPEG (`<img src="/video">`); state is polled
JSON; controls/commands are POSTs. `/ping` returns `{"app":"handball"}` so the
native shell can auto-discover the PC by scanning the WiFi subnet.

**Client packaging:** `webapp/` is an installable PWA. `android/` is a ~150-line
Kotlin WebView shell that finds the PC and loads the live UI — so **feature/UI
changes only touch `webapp/`; the APK almost never rebuilds.**

## 2. The vision pipeline (per frame)

```
capture → Detector(pose@640 + ball@960) → StaticMarkerFilter → Tracker
   → [Kalman gate + re-acquisition + coast] → PlayerReID/MSR → HandballEngine
   → overlays(skeleton/boxes/trajectory) + HUD → frame_callback / status_callback
```
- **Pose** (YOLOv8s-pose) gives 17 COCO keypoints/player → drives overlays AND the
  biomechanics layer (see §4).
- **Ball** runs at higher res (960) because the ball is tiny; tracking adds a Kalman
  filter, a re-acquisition gate, and bounded coast (continuity ~72%).
- **Events** (possession/pass/shot/goal/fast-break) come from `HandballEngine`,
  attributed to ReID identities; goals trigger `GoalReplay` freezes.

## 3. Control center (live, no restart)

`POST /control` body keys (shared `controls` dict):
| key | type | effect |
|---|---|---|
| `skeleton` | bool | COCO-17 pose wireframe overlay |
| `boxes` | bool | bounding boxes |
| `trajectory` | bool | fading ball-path trail |
| `ids` / `markers` | bool | player labels / foot ellipses |
| `player_conf` | 0–1 | YOLO person threshold (live) |
| `ball_conf` | 0–1 | ball-net threshold (live) |

## 4. Deep analysis roadmap (built on the pose stream)

The keypoint stream already flowing per player is the substrate for:
- **Biomechanics:** jump height (hip-y vs standing baseline), throwing-arm angle
  (shoulder–elbow–wrist), release height, trunk torque (shoulder-line vs hip-line).
- **Spatial/tactics:** team centroid + spread, defensive line depth, offensive
  spacing — from homography-projected foot positions.
- **Shot map:** ball trajectory + goal events → shot clusters on a virtual court.

These are additive modules consuming the same `status_callback`/keypoints — no
change to the transport or client needed.

## 5. Run it
- PC server: `Handball_Mobile.bat` (prints the LAN URL).
- Phone: open that URL (PWA) **or** install `HandballStudio.apk` and Auto-find.
- Standalone desktop window: `test_video.bat`. Rebuild APK: `build_apk.bat`.
