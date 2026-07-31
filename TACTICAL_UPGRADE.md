# Tactical Analysis Upgrade — Architecture & Integration Guide

Five new modules in `src/` implement the assist freeze-frames, IHF notation,
transition analysis, and goalkeeper KPI. All are **self-contained and already
smoke-tested** (`reports/ihf_notation_demo.png`, `reports/transition_demo.png`,
`reports/assists_test/`). Nothing in the live pipeline changes until the hooks
below are added.

## 1. System architecture & logic flow

```
CaptureThread ─► Detector/Tracker ─► HandballEngine (rules.update → MatchEvents)
                      │                        │
                      │ tracks + ball          │ PASS / SHOT / GOAL / SAVE /
                      ▼                        │ MISSED_OUT / POSSESSION_CHANGE
        ┌─────────────┴───────────┐            ▼
        │ AssistFreezeAnalyzer     │◄── on_event(GOAL|SAVE|MISSED_OUT)
        │  (rolling 6 s snapshot   │      rewinds to the assist PASS frame,
        │   buffer, like           │      slices frame-by-frame, renders IHF
        │   GoalReplayRecorder)    │      overlays → reports/assists/…
        ├──────────────────────────┤
        │ ActionClassifier         │──► ActionSegments (pass/dribble/shot/move)
        │  (per-frame ball+tracks) │      → ihf_notation renderer (live overlay)
        ├──────────────────────────┤
        │ TransitionAnalyzer       │◄── on_turnover(POSSESSION_CHANGE)
        │  (recovery, laggard,     │      per-frame update() + draw()
        │   Voronoi/grid space)    │      episode summary → logs/reports
        ├──────────────────────────┤
        │ GoalkeeperKPI            │◄── register_event(GOAL|SAVE|MISSED_OUT)
        │  (persistent overlay)    │      draw_overlay() every frame
        └──────────────────────────┘
```

Key design decision: the freeze-frame module **never pauses the live feed** —
it snapshots into a ring buffer (same pattern as `GoalReplayRecorder`) and
exports the paused/sliced tactical sequence to disk asynchronously of the
match clock, so the operator can open it seconds later from the web UI.

## 2. Classification logic (mathematics)

All at processing resolution (1280 px wide), fps-normalisable to m/s via the
court homography (`config/court_calibration.json`).

| Action | Rule |
|---|---|
| **Possessed** | `‖ball − player‖ < 110 px` |
| **Pass** | ball exits holder radius with `v ≥ 6 px/f`, reaches another player's radius within 30 frames, per-step direction change `< 35°` (ballistic), **no** vertical oscillation. Same team → completed; other team → interception. |
| **Dribble** | ball stays `< 130 px` of the **same** player ≥ 8 frames **and** `y(t)` has ≥ 2 sign flips of `dy/dt` with peak-to-peak ≥ 9 px **and** the player displaced ≥ 40 px. |
| **Shot** | ball exits holder radius with `v ≥ 18 px/f` (matches `SHOT_SPEED_MIN_PX`) **and** direction within **22°** of a goal centre. |
| **Move (off-ball)** | non-possessor displaced ≥ 40 px in a 25-frame window. |
| **Assist** | last completed PASS by the finishing team whose receiver == shooter, ≤ 4 s before the finish (falls back to shooter-agnostic last pass). |
| **Recovery** | defender's court-metre `x` crosses the halfway line toward their own goal. Laggard = the last one (or farthest while not back). |
| **Vulnerable space** | 1 m grid over the defensive half; a cell is *open* if `min_dist(cell, recovered defenders) > 3.5 m`. Area = Σ open cells (m²). Rendered by warping the metre-grid mask through the inverse homography (perspective-correct, holes preserved). Voronoi cells (scipy, border-mirrored) available via `voronoi_cells_m()` for playbook figures. |
| **GK Save%** | `saves / (saves + goals_conceded) × 100`; `missed_out` increments off-target only. |

## 3. Integration hooks (pipeline.py)

**a) Init — after `court_det = CourtDetector()` (~line 359):**
```python
from assist_freeze import AssistFreezeAnalyzer
from transition_analysis import TransitionAnalyzer
from gk_kpi import GoalkeeperKPI

assist_fz = AssistFreezeAnalyzer(REPORTS_DIR / "assists", fps=25.0)
trans     = TransitionAnalyzer(1280, 720, fps=25.0)
gk_kpi    = GoalkeeperKPI(team_names)
# feed homography now and inside the reload_calib block (~line 556):
#   trans.set_homography(court_det._H)
```

**b) Per-frame push — right next to `goal_replay.push_frame(...)` (~line 865),
reusing the same track-dict construction:**
```python
snap_dicts = [{"tid": int(t.track_id), "cls": 0,
               "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
               "cx": t.cx, "cy": t.cy, "team": getattr(t, "team", "unknown"),
               "kpts": None} for t in tracks]
ball_xy = (ball_track.cx, ball_track.cy) if ball_track else None
assist_fz.push(frame, snap_dicts, ball_xy, frame_n)
trans.update(frame_n, snap_dicts)
```

**c) Event fan-out — inside the existing `for ev in new_events:` (~line 878):**
```python
gk_kpi.register_event(ev.event_type.value, ev.team, ev.data)
assist_fz.on_event(ev.event_type.value, ev.team, ev.frame_n, ev.data,
                   all_events=rules.events)
if ev.event_type == EventType.POSSESSION_CHANGE:
    trans.on_turnover(ev.team, frame_n)
```

**d) Overlay rendering — in the display block (~line 990, after overlays):**
```python
vis = trans.draw(vis)
vis = gk_kpi.draw_overlay(vis)         # persistent Save% panel, top-right
```

**e) Episode logging — anywhere in the loop:**
```python
if (ep := trans.pop_completed()):
    api.push_match_event({"event": "transition", **ep.summary()})
```

**f) Half-time:** call `trans.flip_sides()` where the pipeline handles the
`F` (flip sides) command.

Live IHF overlays for pass/dribble/shot arrows (optional, CPU-cheap): run
`ActionClassifier.update()` alongside (b) and render closed segments for ~1.5 s
with `ihf_notation.draw_pass_arrow / draw_dribble_path / draw_shot_arrow /
draw_move_arrow`.

## 4. Tuning table

| Constant | File | Default | Meaning |
|---|---|---|---|
| `R_POSSESS_PX` | action_classifier.py | 110 | possession radius |
| `V_SHOT_MIN` | action_classifier.py | 18 px/f | shot speed gate |
| `THETA_GOAL_DEG` | action_classifier.py | 22° | shot aiming cone |
| `R_COVER_M` | transition_analysis.py | 3.5 m | defender coverage disc |
| `EPISODE_MAX_S` | transition_analysis.py | 8 s | transition timeout |
| `buffer_seconds` | assist_freeze.py | 6 s | rewind window |
| `slice_every` | assist_freeze.py | 2 | export every Nth frame |
