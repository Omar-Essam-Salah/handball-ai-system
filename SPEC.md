# Handball AI Analysis System — Technical Specification

**Version:** 1.0
**Sport:** Handball
**Owner:** Goalkeeper Coach
**Target Hardware:** Dell Precision 7740 · Core i9 · 32GB RAM · Quadro RTX 3000 Mobile (6GB VRAM)
**Camera:** Hikvision DS-2CD2686G2-LZS · 8MP · Varifocal 2.8–12mm · RTSP
**Constraint:** 100% Offline — local hotspot only, no cloud, no internet during match

---

## Team Convention

| Color | Team |
|-------|------|
| Red | Home team (coach's team) |
| Blue | Opponent |

---

## Module Breakdown

```
handball-system/
├── Phase 1  Manual Tagger (MVP)            ← DONE
├── Phase 2  Live AI Detection              ← next
├── Phase 3  Court Analysis + GK Module    ← after 2
└── Phase 4  Dashboard + Reports + Notify  ← last
```

---

## Phase 1 — Manual Tagger (MVP)

**Goal:** Working tool today. Coach tags events by hotkey while watching recorded match. Auto-clips around each tag.

**Stack:**
- Python 3.11+
- PyQt6 — desktop GUI, video widget
- ffmpeg — clip export (subprocess)
- pandas — summary CSV
- PyYAML — config

**Features:**
- Open any MP4/MKV match file
- Play / pause / scrub (keyboard)
- Hotkey per event type + team (fully editable in `config/hotkeys.yaml`)
- Tags saved to JSON sidecar next to video (auto-reload on reopen)
- Undo last tag (Ctrl+Z)
- Export clips: `clips/<match>/<event_team_001>.mp4` (±5s/3s around tag, configurable)
- Summary CSV: count per event × team

**Event types:**
`shot` · `goal` · `pass` · `turnover` · `block` · `seven_m` · `counter` · `failed_att` · `save`

**Files:**
```
src/main.py       entry point
src/player.py     main window + hotkey binding
src/tagger.py     Event dataclass + TagStore (JSON)
src/clipper.py    ffmpeg clip export
src/summary.py    pandas CSV report
config/hotkeys.yaml
```

**Run:**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\main.py
```

---

## Phase 2 — Live AI Detection

**Goal:** Real-time RTSP stream with player + ball detection, team color split, player tracking.
**Approach:** AI suggests detections → coach confirms/corrects (semi-automatic).

**Stack additions:**
- `ultralytics` — YOLOv8m (persons=class 0, ball=class 32 COCO, or fine-tuned)
- `boxmot` — BoT-SORT tracker (best for fast small ball)
- `opencv-python` — frame processing, HSV color
- `torch` + CUDA 12.x — GPU inference on RTX 3000
- Thread: capture thread + frame queue (maxsize=2) + inference thread

**Pipeline:**
```
RTSP (/102 sub-stream 1080p H.264)
  → Capture Thread (cv2.VideoCapture)
  → Frame Queue (thread-safe, maxsize=2)
  → YOLOv8m inference (CUDA, ~25-30fps on RTX 3000)
  → BoT-SORT (stable track IDs)
  → HSV classifier on torso crop → red / blue / unknown
  → Annotated preview window
  → Write detections to detection_log.jsonl (one line per frame)
```

**Camera settings (Hikvision web UI):**
- Stream type: Sub (channel 102)
- Resolution: 1280×720 (hardware max for sub-stream on this camera)
- Frame rate: 25fps
- Codec: H.264 Medium profile
- Bitrate: 1024 Kbps VBR (hardware max — recommend CBR, I-Frame=25)
- RTSP URL: `rtsp://<user>:<pass>@172.16.5.174:554/Streaming/Channels/102`

> Note: 720p is sufficient — YOLOv8 resizes input to 640×640 internally.
> Main stream (channel 101) available at 8MP if needed for court homography calibration.

**HSV team ranges (tune per venue lighting):**
```python
# Red team
RED_LOW1  = (0,   100, 80)
RED_HIGH1 = (10,  255, 255)
RED_LOW2  = (170, 100, 80)
RED_HIGH2 = (180, 255, 255)

# Blue team
BLUE_LOW  = (100, 80, 60)
BLUE_HIGH = (130, 255, 255)
```

**Performance target:** 25fps live on RTX 3000 · YOLOv8m · 1080p
**Optional speedup:** Export to TensorRT `.engine` → ~2× inference, ~15fps headroom left

**New files:**
```
src/capture.py    RTSP reader thread + frame queue
src/detector.py   YOLOv8m wrapper (CUDA)
src/tracker.py    BoT-SORT wrapper
src/team_color.py HSV classifier on bbox crop
src/pipeline.py   orchestrate capture→detect→track→color
```

---

## Phase 3 — Court Analysis + GK Module

**Goal:** Map pixel positions to real court coordinates. Detect events automatically. Goalkeeper-specific analytics.

### 3a — Court Homography

**How:** Coach clicks 4 known court corners once at setup. System computes homography matrix H.
**Output:** Every tracked player/ball gets `(x_m, y_m)` in court meters (court = 40m × 20m for handball).

**Court zones (handball):**
```
Left Wing | Left Back | Center | Right Back | Right Wing
          |     6m goal area (goalkeeper zone)
          |     9m free-throw line
                Goal mouth (3m × 2m)
```

**Files:**
```
src/court.py          Homography compute + pixel→meters transform
config/court_zones.yaml  zone polygon definitions in court meters
data/calibration/     stored H matrix per venue
```

### 3b — Event Detection (Rule Engine)

Events detected by rules (no ML needed here — coach defines rules):

| Event | Detection Rule |
|-------|---------------|
| `shot` | Ball velocity spike toward goal direction |
| `goal` | Ball enters goal-mouth polygon |
| `seven_m` | Manual flag OR ball spotted at 7m mark + foul context |
| `turnover` | Ball possession switches team within 3 seconds |
| `counter` | Possession switch + ball moves >15m in <4s |
| `failed_att` | Red team possession ends without shot in attacking half |
| `block` | Ball trajectory reversed within 1m of player (non-GK) |
| `save` | Ball enters goal zone + GK bbox intersects ball, no goal |

**Files:**
```
src/events.py     rule engine, consumes tracker + court output
src/state.py      possession state machine (red/blue/contest/dead)
```

### 3c — Goalkeeper Analysis Module

**Inputs:** Ball position (court coords), GK position, goal-mouth polygon, team context

**Outputs per shot:**
- Shooting angle (°) — `atan2` from ball position to near/far post
- Shot zone — Left / Center / Right (goal split into 3×2 grid)
- GK position at shot moment — was GK on correct side?
- Result — `save` or `goal`

**Aggregate stats:**
- Save % overall
- Save % per shot zone
- Save % per angle range
- Weak zone heatmap (goal-mouth grid)

**Files:**
```
src/gk_analyzer.py    per-shot geometry, aggregate stats
src/goal_mouth.py     goal polygon, zone grid definitions
```

---

## Phase 4 — Dashboard + Reports + Notifications

**Goal:** Live dashboard on phone browser (LAN), post-match PDF report, push notifications to phone/watch.

**Stack additions:**
- `fastapi` + `uvicorn` — local API server
- `websockets` — push live events to browser
- `jinja2` + `weasyprint` — PDF report generation
- HTML5 dashboard (plain JS, no framework needed)

**Architecture:**
```
FastAPI server (port 8000, LAN only)
├── GET  /                    → dashboard HTML
├── GET  /api/stats           → current match stats JSON
├── GET  /api/events          → event log JSON
├── WS   /ws/live             → push new events in real-time
├── GET  /report/{match_id}   → download PDF report
└── POST /api/tag             → manual tag from phone
```

**Notifications:**
- Phone opens `http://192.168.x.x:8000` in browser → receives WebSocket pushes
- Smartwatch browser (Samsung/Apple) same URL
- No app install needed. No cloud. Pure LAN WebSocket.

**Report contents:**
- Match summary (score, shots, saves, turnovers per team)
- GK stats (save %, shot zones, weak area heatmap)
- Per-player stats (ball losses, passes, shots)
- Timeline of key events with timestamps
- Exported as PDF

**New files:**
```
src/api/
  main.py        FastAPI app
  ws.py          WebSocket manager
  models.py      Pydantic schemas
templates/
  dashboard.html live dashboard
  report.html    PDF template
```

---

## Full Data Flow (Phase 4 complete system)

```
Camera
  └─RTSP──► Capture Thread ──► Frame Queue
                                    │
                               Inference Engine
                               (YOLO + BoT-SORT + HSV)
                                    │
                         ┌──────────┼──────────┐
                    Court Map   Event Rules   GK Module
                         └──────────┼──────────┘
                                    │
                              Stats Aggregator
                                    │
                    ┌───────────────┼───────────────┐
               FastAPI/WS      JSONL log         PDF Report
                    │
          ┌─────────┼──────────┐
      Browser    Phone       Smartwatch
     Dashboard  Notify        Notify
     (LAN)      (LAN)          (LAN)
```

---

## Tech Stack Summary

| Layer | Technology | Phase |
|-------|-----------|-------|
| GUI / Tagger | PyQt6 | 1 |
| Clip export | ffmpeg (subprocess) | 1 |
| RTSP capture | OpenCV VideoCapture | 2 |
| Object detection | YOLOv8m (Ultralytics) | 2 |
| Tracking | BoT-SORT (boxmot) | 2 |
| Team color | OpenCV HSV | 2 |
| GPU inference | PyTorch + CUDA 12 | 2 |
| Court mapping | OpenCV Homography | 3 |
| Event detection | Rule engine (Python) | 3 |
| GK analytics | Geometry (math/numpy) | 3 |
| Heatmaps | matplotlib / numpy | 3 |
| API server | FastAPI + Uvicorn | 4 |
| Live push | WebSocket | 4 |
| Reports | Jinja2 + WeasyPrint | 4 |
| Tag storage | JSON sidecar files | 1–4 |
| Stats DB | SQLite (Phase 3+) | 3–4 |

---

## Hardware Requirements (confirmed)

| Resource | Available | Required |
|----------|-----------|---------|
| GPU VRAM | 6GB | ~3GB (YOLOv8m fp16) |
| RAM | 32GB | ~4GB runtime |
| CPU | Core i9 | multi-thread capture + API |
| Storage | — | ~5GB models + clips |
| Network | Local hotspot | phone + watch on same LAN |

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|-----------|
| HSV color fails under IR / bad lighting | High | Always tag team manually at match start; use pre-calibrated HSV ranges |
| Ball too small / fast for YOLO COCO | Medium | Fine-tune YOLOv8m on handball ball images (50–100 labeled frames) |
| RTSP stream drops | Medium | Auto-reconnect loop in capture thread (retry every 2s) |
| RTX 3000 thermal throttle during long match | Low | Monitor GPU temp, reduce resolution to 720p if >85°C |
| 7m throw detection (needs referee cue) | High | Manual hotkey override always available (Phase 1 covers this) |

---

## Development Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1 | Done | Manual tagger + clip export |
| 2 | 3–4 weeks | Live AI detection preview window |
| 3 | 4–6 weeks | Court zones + events + GK stats |
| 4 | 3–4 weeks | Dashboard + PDF report + phone notify |

**Total estimate:** ~3 months to full system
**Sellable after:** Phase 3 complete (GK analysis = unique commercial value)
