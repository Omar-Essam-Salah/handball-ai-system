"""
batch_match_analyzer.py — end-user "upload a clip, get a full tactical
breakdown" engine. Runs the same detection/tracking/rules stack as the live
pipeline, but OFFLINE over an arbitrary uploaded video file: no camera, no
court calibration, no roster required.

For every finishing attempt (GOAL / SAVE / MISSED_OUT) it produces a
"critical moment": the freeze-frame annotated with the defensive-gap
analysis (tactical_annotator), the shooter's real geometric shot OPTIONS at
release (shot_prediction) alongside the shot that actually happened, and a
structured Arabic report entry (arabic_report) — all traceable to real
tracked coordinates, nothing invented.

At the end it aggregates a per-team strengths/weaknesses summary from the
same event stream (shots, saves, passes, fouls, and how often each specific
player got flagged for a positioning error across the whole clip).

    from batch_match_analyzer import BatchMatchAnalyzer

    analyzer = BatchMatchAnalyzer()
    result = analyzer.analyze("uploaded_clip.mp4", progress_cb=lambda f, t: ...)
    for moment in result.moments:
        moment.annotated_frame   # np.ndarray BGR
        moment.report            # arabic_report.PlayReportEntry
    result.summary               # {"team_a": {...}, "team_b": {...}}

Known v1 limitations (disclosed, not hidden):
  - No jersey-number OCR here (keeps batch processing fast) — players are
    identified by their internal tracking ID, noted in each report's
    `note` field.
  - Goal side / attack direction comes from GoalPostDetector's real-time
    geometric detection, which needs a few frames to lock on; moments in
    the first ~1s of the clip may skip the gap/shot-option overlay if no
    goal box was found yet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from team_color import AutoColorDetector
from goal_post_detector import GoalPostDetector, GoalBox
from handball_rules import HandballEngine, EventType, MatchEvent

from defensive_gap_detector import analyze_defensive_shape
from tactical_annotator import render_defensive_analysis
from shot_prediction import compute_shot_options, render_shot_options
from ihf_notation import draw_shot_arrow
from arabic_text import draw_text
from arabic_report import ScorerInfo, build_play_report, format_clock
from gk_kpi import GoalkeeperKPI

FINISHING_EVENTS = {"goal", "save", "missed_out"}
BUFFER_SECONDS   = 5.0        # rolling window kept for freeze-frame export


@dataclass
class _Snap:
    fn:      int
    frame:   np.ndarray
    tracks:  list[dict]
    ball_xy: Optional[tuple]
    goal_box: Optional[GoalBox]


@dataclass
class CriticalMoment:
    event_type:       str
    frame_n:           int
    time_s:            float
    team:              str
    annotated_frame:   np.ndarray
    report:            "PlayReportEntry"           # noqa: F821  (arabic_report.PlayReportEntry)
    gap_flagged_tids:  list[int] = field(default_factory=list)


@dataclass
class MatchAnalysisResult:
    moments: list[CriticalMoment]
    summary: dict
    fps: float
    total_frames: int


def _track_to_dict(t) -> dict:
    return {"tid": int(getattr(t, "track_id", -1)),
           "cx": float(t.cx), "cy": float(t.cy),
           "x1": float(t.x1), "y1": float(t.y1),
           "x2": float(t.x2), "y2": float(t.y2),
           "team": getattr(t, "team", "unknown")}


class BatchMatchAnalyzer:
    def __init__(self, device: str = "cuda:0", model: str = "yolov8s-pose.pt",
                conf: float = 0.30, imgsz: int = 640,
                team_names: Optional[dict] = None):
        self.device, self.model_name, self.conf, self.imgsz = device, model, conf, imgsz
        self.team_names = team_names or {"team_a": "الفريق الأول", "team_b": "الفريق الثاني"}

    def analyze(self, video_path: str | Path,
               progress_cb: Optional[Callable[[int, int], None]] = None,
               max_seconds: Optional[float] = None) -> MatchAnalysisResult:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_seconds:
            total = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)

        det       = Detector(device=self.device, model=self.model_name,
                             conf=self.conf, imgsz=self.imgsz)
        tracker   = Tracker(device=self.device, backend="botsort")
        color_det = AutoColorDetector(min_crops=15)
        goal_det  = GoalPostDetector(frame_w=w, frame_h=h)
        rules     = HandballEngine(frame_w=w, frame_h=h, fps=fps)
        gk_kpi    = GoalkeeperKPI(self.team_names)

        buf: list[_Snap] = []
        buf_len = max(1, int(BUFFER_SECONDS * fps))
        moments: list[CriticalMoment] = []
        gap_tally: dict[str, dict[int, int]] = {"team_a": {}, "team_b": {}}
        attack_counter = {"team_a": 0, "team_b": 0}
        pending_shot: Optional[dict] = None   # awaiting GOAL/SAVE/MISSED_OUT resolution

        frame_n = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or (total and frame_n >= total):
                break

            dets = det.detect(frame)
            tracks = tracker.update(dets, frame)
            color_det.feed(frame, tracks)
            for t in tracks:
                if getattr(t, "class_id", 0) != 32:
                    t.team = (color_det.classify(frame, t)
                             if color_det.is_calibrated else "unknown")

            goal_box = goal_det.update(frame, frame_n)
            rules._dynamic_goal_box = goal_box

            ball_track = next((t for t in tracks if getattr(t, "class_id", 0) == 32), None)
            ball_xy = (float(ball_track.cx), float(ball_track.cy)) if ball_track else None
            ball_hallucinated = bool(getattr(tracker, "ball_was_hallucinated", False))

            score_before = dict(rules.score)
            try:
                events = rules.update(tracks, ball_track, frame_n,
                                      ball_hallucinated=ball_hallucinated)
            except Exception:
                events = []

            buf.append(_Snap(frame_n, frame.copy(),
                             [_track_to_dict(t) for t in tracks if getattr(t, "class_id", 0) != 32],
                             ball_xy, goal_box))
            if len(buf) > buf_len:
                buf.pop(0)

            for ev in events:
                et = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                if et == "shot":
                    pending_shot = {"team": ev.team, "frame_n": ev.frame_n,
                                    "shooter_id": ev.data.get("shooter_id")}
                elif et in FINISHING_EVENTS:
                    gk_kpi.register_event(et, ev.team, ev.data)
                    attack_counter[ev.team] = attack_counter.get(ev.team, 0) + 1
                    moment = self._build_moment(
                        et, ev, pending_shot, buf, fps, score_before, rules.score,
                        attack_counter[ev.team])
                    if moment is not None:
                        moments.append(moment)
                        for tid in moment.gap_flagged_tids:
                            gap_tally.setdefault(ev.team, {})
                            d = gap_tally.setdefault(("team_b" if ev.team == "team_a" else "team_a"), {})
                            d[tid] = d.get(tid, 0) + 1
                    pending_shot = None

            frame_n += 1
            if progress_cb and frame_n % 10 == 0:
                progress_cb(frame_n, total or frame_n)

        cap.release()
        elapsed = time.perf_counter() - t0
        print(f"[BatchAnalyzer] processed {frame_n} frames in {elapsed:.1f}s "
             f"({frame_n / max(elapsed, 1e-6):.1f} fps) — {len(moments)} critical moments")

        summary = self._build_summary(rules, gk_kpi, gap_tally)
        return MatchAnalysisResult(moments=moments, summary=summary, fps=fps,
                                   total_frames=frame_n)

    # ── critical-moment builder ──────────────────────────────────────────────
    def _build_moment(self, et: str, ev: MatchEvent, pending_shot: Optional[dict],
                      buf: list[_Snap], fps: float, score_before: dict,
                      score_after: dict, attack_number: int) -> Optional[CriticalMoment]:
        if not buf:
            return None
        snap = buf[-1]                      # frame at/near the resolving event
        shot_snap = None
        if pending_shot is not None:
            shot_snap = next((s for s in buf if s.fn == pending_shot["frame_n"]), buf[0])

        team = ev.team
        defending_team = "team_b" if team == "team_a" else "team_a"
        # "goalkeeper" is its own AutoColorDetector cluster, distinct from the
        # outfield team_a/team_b labels — excluded from the outfield block
        # (matches how a 6-0 defensive wall is analyzed in practice) but
        # needed separately as the real aim-point for shot options below.
        defenders = [p for p in snap.tracks if p["team"] == defending_team]
        attackers = [p for p in snap.tracks if p["team"] == team]
        keepers   = [p for p in snap.tracks if p["team"] == "goalkeeper"]

        out = snap.frame.copy()
        gap_flagged: list[int] = []
        finding = None
        gb = snap.goal_box or (shot_snap.goal_box if shot_snap else None)
        goal_x, attack_dir = None, None
        if gb is not None and len(defenders) >= 3:
            attack_dir = -1 if gb.side == "left" else +1
            goal_x = gb.x_left if gb.side == "left" else gb.x_right
            finding = analyze_defensive_shape(defenders, attackers, goal_x, attack_dir)
            render_defensive_analysis(out, defenders, attackers, goal_x, attack_dir)
            if finding is not None:
                gap_flagged = [f.tid for f in finding.flagged]

            # shot options at the release frame (real GK + defender geometry)
            if shot_snap is not None and pending_shot.get("shooter_id") is not None:
                shooter = next((p for p in shot_snap.tracks
                               if p["tid"] == pending_shot["shooter_id"]), None)
                gk = (min(keepers, key=lambda k: abs(k["cx"] - goal_x)) if keepers
                     else (min(defenders, key=lambda d: abs(d["cx"] - goal_x))
                           if defenders else None))
                if shooter is not None and gk is not None:
                    options = compute_shot_options(
                        (shooter["cx"], shooter["cy"]), goal_x, gb.y_top, gb.y_bottom,
                        goalkeeper_point=(gk["cx"], gk["cy"]),
                        defender_points=[(d["cx"], d["cy"]) for d in defenders])
                    render_shot_options(out, (shooter["cx"], shooter["cy"]), options)
                    if snap.ball_xy:
                        draw_shot_arrow(out, (shooter["cx"], shooter["cy"]), snap.ball_xy)

        outcome_ar = {"goal": "هدف", "save": "تصدي الحارس", "missed_out": "تسديدة خارج المرمى"}[et]
        draw_text(out, f"لحظة حرجة #{attack_number} — {outcome_ar}", (16, 30),
                  size=22, color=(0, 235, 255))

        shooter_id = ev.data.get("shooter_id")
        report = build_play_report(
            attack_number=attack_number,
            attacking_team_name=self.team_names.get(team, team),
            attack_kind=outcome_ar,
            start_frame=(shot_snap.fn if shot_snap else snap.fn), end_frame=snap.fn, fps=fps,
            team_a_name=self.team_names.get("team_a", "team_a"),
            team_b_name=self.team_names.get("team_b", "team_b"),
            score_a_before=score_before.get("team_a", 0), score_b_before=score_before.get("team_b", 0),
            score_a_after=score_after.get("team_a", 0), score_b_after=score_after.get("team_b", 0),
            scorer=ScorerInfo(team=self.team_names.get(team, team),
                             jersey_number=None, tracking_id=shooter_id, foot=""),
            gap_finding=finding,
            jersey_by_tid=None,
        )
        report.note = ("تعريف اللاعبين برقم التتبع الداخلي — تحليل سريع بدون قراءة "
                       "أرقام القمصان (OCR).")

        return CriticalMoment(event_type=et, frame_n=snap.fn, time_s=snap.fn / fps,
                              team=team, annotated_frame=out, report=report,
                              gap_flagged_tids=gap_flagged)

    # ── strengths/weaknesses aggregate ───────────────────────────────────────
    def _build_summary(self, rules: HandballEngine, gk_kpi: GoalkeeperKPI,
                       gap_tally: dict) -> dict:
        out = {}
        for team in ("team_a", "team_b"):
            name = self.team_names.get(team, team)
            shots  = rules.shots.get(team, 0)
            goals  = rules.score.get(team, 0)
            saves  = rules.saves.get(team, 0)
            passes = rules.passes.get(team, 0)
            fouls  = rules.fouls.get(team, 0)
            fb     = rules.fast_breaks.get(team, 0)
            acc = (100.0 * goals / shots) if shots else 0.0
            gk_stats = gk_kpi.stats[team]

            strengths, weaknesses = [], []
            if acc >= 55 and shots >= 3:
                strengths.append(f"كفاءة تسديد عالية ({acc:.0f}٪ من {shots} محاولة).")
            if fb >= 2:
                strengths.append(f"استغلال جيد للهجمات السريعة ({fb} هجمة مرتدة).")
            if gk_stats.save_pct >= 35 and gk_stats.on_target >= 3:
                strengths.append(f"أداء حارس مرمى قوي (نسبة تصدي {gk_stats.save_pct:.0f}٪).")
            if not strengths:
                strengths.append("لم تُرصد نقطة قوة واضحة الإحصاء في هذه اللقطات — العدد قليل للحكم.")

            if acc < 35 and shots >= 3:
                weaknesses.append(f"كفاءة تسديد منخفضة ({acc:.0f}٪ من {shots} محاولة).")
            if gk_stats.save_pct < 20 and gk_stats.on_target >= 3:
                weaknesses.append(f"حراسة مرمى متذبذبة (نسبة تصدي {gk_stats.save_pct:.0f}٪ فقط).")
            if fouls >= 3:
                weaknesses.append(f"عدد أخطاء مرتفع نسبياً ({fouls}).")
            opp_gaps = gap_tally.get(team, {})
            if opp_gaps:
                worst_tid = max(opp_gaps, key=opp_gaps.get)
                weaknesses.append(
                    f"تكرار خطأ تمركز دفاعي على اللاعب صاحب رقم التتبع #{worst_tid} "
                    f"({opp_gaps[worst_tid]} مرة عبر اللقطات المرصودة).")
            if not weaknesses:
                weaknesses.append("لم تُرصد نقطة ضعف متكررة واضحة في هذه اللقطات.")

            out[team] = {
                "team_name": name, "goals": goals, "shots": shots, "saves": saves,
                "passes": passes, "fouls": fouls, "fast_breaks": fb,
                "shot_accuracy_pct": round(acc, 1),
                "goalkeeper": gk_kpi.snapshot()[team],
                "strengths": strengths, "weaknesses": weaknesses,
            }
        return out
