"""
gemini_match_analyzer.py — "send Gemini the whole clip" critical-moment
analysis, matching the workflow the user already likes from chatting with
Gemini directly: upload the full video ONCE, let Gemini itself find the
significant attacking moments and write the tactical breakdown, no local
YOLO/tracking pipeline involved at all.

Trade-off vs. batch_match_analyzer.py (the local CV pipeline), stated
plainly so the choice is informed, not hidden:
  + Fast: one video upload + one generation call, not per-frame inference.
    A 45s clip that took batch_match_analyzer ~10 minutes takes well under
    a minute here.
  + Handles panning/broadcast camera footage fine — no ball-speed pixel
    heuristics to confuse.
  + Richer natural-language tactical judgment (the whole reason to use an
    LLM here).
  - Player/box positions are NOT pixel-exact — Gemini describes players by
    visible jersey number / shirt color / position, not tracked (x, y)
    coordinates, so there are no precise defender boxes or gap ellipses
    like tactical_annotator.py draws from real tracking data. The frame
    shown per moment is the REAL video frame at Gemini's reported
    timestamp (grabbed locally, not generated), just without that overlay.
  - Costs real Gemini API quota per upload (this account hit hard limits
    on gemini-2.5-pro/flash yesterday — gemini-flash-latest is what
    actually works, so that's the default here). Keep clips reasonably
    short; every analysis re-uploads and re-processes the whole video.

    from gemini_match_analyzer import analyze_match_with_gemini
    result = analyze_match_with_gemini("clip.mp4", api_key="...")
    result["moments"]   # same shape mobile_server.py already serves
    result["summary"]
"""
from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import cv2
from pydantic import BaseModel

MODEL_NAME = "gemini-flash-latest"   # verified working on this account, July 2026
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


class GKeyPlayer(BaseModel):
    identifier: str
    role: str
    action: str


class GCriticalMoment(BaseModel):
    time_s: float
    event_type: str            # "goal" | "save" | "missed_out" | "foul" | "other"
    side_description: str      # e.g. "الفريق الأصفر" — Gemini doesn't know team_a/b labels
    scoreboard_reading: str     # read from the on-screen broadcast graphic if visible
    attack_title: str
    key_players: list[GKeyPlayer]
    defensive_error: str
    shooting_or_pass_options: str


class GTeamSummary(BaseModel):
    side_description: str
    strengths: list[str]
    weaknesses: list[str]


class GMatchReport(BaseModel):
    moments: list[GCriticalMoment]
    team_summaries: list[GTeamSummary]


class GeminiMatchAnalysisError(RuntimeError):
    pass


SYSTEM_INSTRUCTION = """\
You are a world-class handball tactical analyst reviewing a match clip.
Watch the ENTIRE clip and identify the most tactically significant
attacking moments (goals, goalkeeper saves, and clear missed shots) — at
most {max_moments}, prioritising the clearest / most instructive ones.

For each moment report, in ARABIC:
  - time_s: your best estimate of the moment (seconds from clip start) —
    this is used to grab the real video frame, so be as precise as you can.
  - event_type: "goal" | "save" | "missed_out" | "foul" | "other"
  - side_description: which team (by shirt color, since you don't know
    internal team names), e.g. "الفريق الأصفر" أو "الفريق الأزرق"
  - scoreboard_reading: if an on-screen scoreboard graphic is visible in
    the video at that moment, read the ACTUAL numbers/text off it. If none
    is visible, say "غير مرئي في الفيديو" — never invent a score.
  - attack_title: short Arabic title for the play
  - key_players: each player involved, identified by visible jersey number
    if legible, else shirt color + position — with their role and action
  - defensive_error: what went wrong defensively (or "لا يوجد خطأ واضح")
  - shooting_or_pass_options: the realistic alternative options the player
    had (other passing lanes, other shot corners) based on what's visible
    in the frame, and why the chosen option worked/failed

Then give an overall team_summaries: one entry per side (by shirt color),
each with 2-4 short Arabic strengths and 2-4 short Arabic weaknesses, based
ONLY on patterns you actually observed across the clip.

Ground every claim in what is visibly happening. Never invent a jersey
number, score, or event that isn't actually visible.
"""


def _http_code(exc: Exception) -> Optional[int]:
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _call_with_retries(fn, max_retries: int, label: str):
    from google.genai import errors as genai_errors
    attempt = 0
    while True:
        try:
            return fn()
        except genai_errors.APIError as e:
            code = _http_code(e)
            attempt += 1
            if code not in _TRANSIENT_HTTP_CODES or attempt > max_retries:
                raise GeminiMatchAnalysisError(f"{label} failed (HTTP {code}): {e}") from e
            wait = min(30.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"[GeminiMatch] {label}: HTTP {code} — retry {attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)


def _upload_and_wait_active(client, video_path: Path, timeout_s: float, poll_s: float = 3.0):
    uploaded = _call_with_retries(lambda: client.files.upload(file=str(video_path)),
                                  4, "file upload")
    deadline = time.monotonic() + timeout_s
    while True:
        state = getattr(getattr(uploaded, "state", None), "name", None)
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise GeminiMatchAnalysisError(f"Video processing FAILED: {getattr(uploaded, 'error', '')}")
        if time.monotonic() > deadline:
            raise GeminiMatchAnalysisError(f"Video not ACTIVE within {timeout_s:.0f}s (state={state})")
        time.sleep(poll_s)
        uploaded = _call_with_retries(lambda: client.files.get(name=uploaded.name),
                                      4, "file status poll")


def _grab_frame_b64(cap: cv2.VideoCapture, time_s: float, quality: int = 82) -> str:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return ""
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok2 else ""


def analyze_match_with_gemini(
    video_path: str | Path,
    *,
    api_key: Optional[str] = None,
    model_name: str = MODEL_NAME,
    max_moments: int = 8,
    processing_timeout_s: float = 240.0,
) -> dict:
    """Returns {"moments": [...], "summary": {...}} in the SAME shape
    mobile_server.py's /analyze/result already serves, so the existing
    analyze.html mobile page works unchanged regardless of which engine
    (this one, or batch_match_analyzer.py) produced the analysis."""
    from google import genai
    from google.genai import types

    path = Path(video_path)
    if not path.is_file():
        raise GeminiMatchAnalysisError(f"Video file not found: {path}")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiMatchAnalysisError("No API key: set GEMINI_API_KEY or pass api_key=")

    client = genai.Client(api_key=key)
    uploaded = _upload_and_wait_active(client, path, processing_timeout_s)

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION.format(max_moments=max_moments),
            response_mime_type="application/json",
            response_schema=GMatchReport,
            temperature=0.3,
        )
        response = _call_with_retries(
            lambda: client.models.generate_content(
                model=model_name, contents=[uploaded, "Analyze this handball match clip."],
                config=config),
            5, f"generate_content({model_name})")
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    parsed = getattr(response, "parsed", None)
    report = parsed if isinstance(parsed, GMatchReport) else GMatchReport.model_validate(
        json.loads(response.text))

    cap = cv2.VideoCapture(str(path))
    moments = []
    for i, m in enumerate(report.moments, start=1):
        moments.append({
            "time_s": round(m.time_s, 1),
            "event_type": m.event_type,
            "team": m.side_description,
            "frame_b64": _grab_frame_b64(cap, m.time_s),
            "report": {
                "attack_number": i,
                "attack_title": m.attack_title,
                "video_time_range": f"[{int(m.time_s)//60:02d}:{int(m.time_s)%60:02d}]",
                "score_at_execution": m.scoreboard_reading,
                "scorer": {"team": m.side_description, "jersey_number": None,
                          "tracking_id": None, "foot": ""},
                "defensive_error": m.defensive_error,
                "shooting_angle": m.shooting_or_pass_options,
                "score_after": "",
                "note": "تحليل بالكامل عبر Gemini (فهم فيديو) — بدون تتبع محلي، "
                        "المواقع والأرقام كما وصفها النموذج من الفيديو مباشرة.",
            },
        })
    cap.release()

    summary = {}
    for i, t in enumerate(report.team_summaries, start=1):
        summary[f"side_{i}"] = {
            "team_name": t.side_description,
            "strengths": t.strengths,
            "weaknesses": t.weaknesses,
        }
    return {"moments": moments, "summary": summary}
