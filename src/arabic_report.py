"""
arabic_report.py — structured Arabic play-by-play report (the "image-1"
style): attack number, video time range, score at execution, scorer
identity, defensive-error description, shooting-angle description, score
after.

All the OBJECTIVE fields (time, score, jersey number) are filled from real
pipeline data — never guessed. The two free-text tactical fields
(defensive_error, shooting_angle) have a deterministic geometric description
by default (angle math over real tracked coordinates — no LLM, no cost, no
hallucination risk), with an OPTIONAL text-only Gemini polish pass that
rewrites the PROSE into more natural coaching language while the underlying
facts (numbers, sides, names) are given to it as constraints, not invented.

Text-only Gemini calls are far cheaper than the video-upload path in
gemini_tactical_analyst.py — no file upload, no ACTIVE-state wait, just a
few hundred tokens of context in, a paragraph out.
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Optional

import numpy as np
from pydantic import BaseModel

from defensive_gap_detector import GapFinding

_ORDINALS_AR = [
    "", "الأولى", "الثانية", "الثالثة", "الرابعة", "الخامسة", "السادسة",
    "السابعة", "الثامنة", "التاسعة", "العاشرة", "الحادية عشرة",
    "الثانية عشرة", "الثالثة عشرة", "الرابعة عشرة", "الخامسة عشرة",
]


def ordinal_ar(n: int) -> str:
    return _ORDINALS_AR[n] if 0 < n < len(_ORDINALS_AR) else f"رقم {n}"


def format_clock(frame_n: int, fps: float) -> str:
    s = frame_n / fps if fps else 0.0
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# ── Structured schema ─────────────────────────────────────────────────────────
class ScorerInfo(BaseModel):
    team:          str
    jersey_number: Optional[int] = None   # real shirt number — only set when OCR read it
    tracking_id:   Optional[int] = None   # internal tracker ID — NOT a jersey number,
                                          # shown only as a fallback when OCR wasn't run
    foot:          str = ""          # "اليسرى" | "اليمنى" | ""


class PlayReportEntry(BaseModel):
    attack_number:      int
    attack_title:       str
    video_time_range:   str
    score_at_execution: str
    scorer:             ScorerInfo
    defensive_error:    str
    shooting_angle:     str
    score_after:        str
    note:               str = ""     # e.g. disclosing an identification shortcut


# ── Deterministic geometric descriptions ─────────────────────────────────────
def describe_defensive_error(finding: Optional[GapFinding],
                             jersey_by_tid: dict[int, int] | None = None) -> str:
    """Real-geometry sentence from the same GapFinding used to draw the
    overlay — the report text and the video annotation always agree."""
    jersey_by_tid = jersey_by_tid or {}
    if finding is None or not finding.flagged:
        return "لم يتم رصد خطأ تمركز دفاعي واضح في هذه الهجمة."

    def name(tid: int) -> str:
        num = jersey_by_tid.get(tid)
        return f"المدافع رقم {num}" if num else f"المدافع #{tid}"

    gap_flags = [f for f in finding.flagged if f.reason == "gap"]
    beaten = [f for f in finding.flagged if f.reason == "beaten"]

    parts = []
    if len(gap_flags) >= 2:
        parts.append(
            f"عدم التفاهم والانضمام الدفاعي بين {name(gap_flags[0].tid)} و"
            f"{name(gap_flags[1].tid)} ترك ثغرة دفاعية بعرض "
            f"{finding.gap_px:.0f} بكسل في منتصف الحائط الدفاعي.")
    if beaten:
        who = "، ".join(name(f.tid) for f in beaten)
        parts.append(f"{who} تأخر في تتبع اللاعب المهاجم فأتاح مساحة اختراق حرة.")
    return " ".join(parts) or "ثغرة دفاعية طفيفة لم تصل لحد الخطورة."


def describe_shooting_angle(shot_from: tuple, shot_to: tuple,
                            goal_center: tuple, goal_half_height_px: float,
                            deception_move: str = "") -> str:
    """Classify the shot's target corner from real tracked coordinates
    (ball position at release vs. ball position at the goal line) and state
    the angle in degrees off the direct line — grounded, not invented."""
    fx, fy = shot_from
    tx, ty = shot_to
    gx, gy = goal_center

    v_shot = np.array([tx - fx, ty - fy], np.float64)
    v_direct = np.array([gx - fx, gy - fy], np.float64)
    n1, n2 = np.linalg.norm(v_shot), np.linalg.norm(v_direct)
    angle_deg = 0.0
    if n1 > 1e-6 and n2 > 1e-6:
        cosang = float(np.clip(np.dot(v_shot, v_direct) / (n1 * n2), -1, 1))
        angle_deg = math.degrees(math.acos(cosang))

    vertical_frac = (ty - gy) / goal_half_height_px if goal_half_height_px else 0.0
    vertical = "العليا" if vertical_frac < -0.25 else ("السفلى" if vertical_frac > 0.25 else "الوسطى")
    horizontal = "اليمنى" if tx > gx else "اليسرى"

    deception = f"بخدعة {deception_move} " if deception_move else ""
    return (f"سدد اللاعب الكرة {deception}نحو الزاوية {vertical} {horizontal} للمرمى، "
           f"منحرفة {angle_deg:.0f}° عن الخط المباشر لحارس المرمى، وسكنت الشباك بنجاح.")


def format_score(team_a_name: str, score_a: int, team_b_name: str, score_b: int,
                 order: tuple[str, str] = ("b", "a")) -> str:
    vals = {"a": (team_a_name, score_a), "b": (team_b_name, score_b)}
    first, second = vals[order[0]], vals[order[1]]
    return f"{first[0]} {first[1]} - {second[1]} {second[0]}"


def build_play_report(
    *,
    attack_number: int,
    attacking_team_name: str,
    attack_kind: str,                    # e.g. "رمية جزائية - 7 أمتار" / "اختراق من الجهة اليمنى"
    start_frame: int, end_frame: int, fps: float,
    team_a_name: str, team_b_name: str,
    score_a_before: int, score_b_before: int,
    score_a_after: int, score_b_after: int,
    scorer: ScorerInfo,
    gap_finding: Optional[GapFinding] = None,
    jersey_by_tid: dict[int, int] | None = None,
    shot_geometry: Optional[dict] = None,   # {from, to, goal_center, goal_half_height_px, deception_move}
    score_order: tuple[str, str] = ("b", "a"),
) -> PlayReportEntry:
    """Deterministic builder — every field traceable to real pipeline data."""
    defensive_error = describe_defensive_error(gap_finding, jersey_by_tid)
    if shot_geometry:
        shooting_angle = describe_shooting_angle(**shot_geometry)
    else:
        shooting_angle = "لم تُحسب هندسة زاوية التسديد لهذه الهجمة."

    return PlayReportEntry(
        attack_number=attack_number,
        attack_title=f"هجمة {attacking_team_name} {ordinal_ar(attack_number)} ({attack_kind})",
        video_time_range=f"[{format_clock(start_frame, fps)} - {format_clock(end_frame, fps)}]",
        score_at_execution=format_score(team_a_name, score_a_before, team_b_name,
                                        score_b_before, score_order),
        scorer=scorer,
        defensive_error=defensive_error,
        shooting_angle=shooting_angle,
        score_after=format_score(team_a_name, score_a_after, team_b_name,
                                 score_b_after, score_order),
    )


# ── Optional prose polish — TEXT ONLY, no video upload ───────────────────────
_POLISH_SYSTEM = """\
You are an Arabic-language handball broadcast analyst editor. You receive a
tactical report where the FACTS (numbers, jersey numbers, times, sides) are
already verified and MUST NOT change. Your only job is to rewrite the two
free-text fields (defensive_error, shooting_angle) into more natural,
professional Arabic sports-commentary prose, in the same tone as elite
handball broadcast analysis. Do not invent any fact not present in the
input (no new player names, no new numbers, no new events). Keep each field
to 1-2 sentences. Respond with ONLY the two rewritten fields as JSON:
{"defensive_error": "...", "shooting_angle": "..."}
"""


def polish_with_gemini(entry: PlayReportEntry, api_key: Optional[str] = None,
                       model_name: str = "gemini-flash-latest",
                       max_retries: int = 3) -> PlayReportEntry:
    """Text-only rewrite of the two prose fields. Cheap (no video), optional
    — call this only if you want the more polished phrasing; the entry
    returned by build_play_report() is already complete and usable as-is."""
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No API key: set GEMINI_API_KEY or pass api_key=")

    client = genai.Client(api_key=key)
    context = (
        f"defensive_error: {entry.defensive_error}\n"
        f"shooting_angle: {entry.shooting_angle}\n"
    )

    attempt = 0
    while True:
        try:
            resp = client.models.generate_content(
                model=model_name, contents=[context],
                config=types.GenerateContentConfig(
                    system_instruction=_POLISH_SYSTEM,
                    response_mime_type="application/json",
                    temperature=0.4))
            break
        except genai_errors.APIError as e:
            attempt += 1
            code = getattr(e, "code", None)
            if code not in (429, 500, 503) or attempt > max_retries:
                raise
            wait = min(30.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"[ArabicReport] polish retry {attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)

    import json
    data = json.loads(resp.text)
    return entry.model_copy(update={
        "defensive_error": data.get("defensive_error", entry.defensive_error),
        "shooting_angle":  data.get("shooting_angle",  entry.shooting_angle),
    })
