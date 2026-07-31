"""
gemini_tactical_analyst.py — Cloud-based handball clip analysis via the
Gemini API (Google Gen AI SDK).

    pip install google-genai pydantic

Auth: set the GEMINI_API_KEY environment variable (get one from
https://aistudio.google.com/apikey), or pass api_key= explicitly.

────────────────────────────────────────────────────────────────────────────
MODEL NOTE — verified July 2026, please read before you rely on this:

    "gemini-1.5-pro" is RETIRED. Google's own lifecycle docs list
    gemini-1.5-pro-001 retired 2025-05-24 and gemini-1.5-pro-002 retired
    2025-09-24. Calling that model id today returns a 404 NOT_FOUND from
    the API — this is a Google-side removal, not a bug in this file.

    MODEL_NAME below is kept literally as "gemini-1.5-pro" per spec. A
    working fallback is wired in: analyze_handball_clip() tries MODEL_NAME
    first and, only on a 404 model-not-found, retries once with
    fallback_model. Set fallback_model=None to disable that and see the raw
    retirement error, or just change MODEL_NAME to skip the extra call.

    Empirically verified end-to-end (real API key, real video, July 2026):
      - "gemini-2.5-pro"   → HTTP 429, quota limit=0 (needs billing enabled
                             on the project — free tier doesn't cover it)
      - "gemini-2.5-flash" → HTTP 404 "no longer available to new users"
                             (Google gates some models by account age)
      - "gemini-flash-latest" → WORKS. This alias always points at whatever
                             model Google currently routes new free-tier
                             projects to, so it survives the next rename
                             better than a pinned version string. That's why
                             it's the FALLBACK_MODEL default below, not a
                             pinned "gemini-2.5-*" id.
────────────────────────────────────────────────────────────────────────────

This module is standalone and NOT wired into the live pipeline — invoking it
sends a video file to Google's servers, a data flow you should opt into
per-clip (e.g. on an exported reports/assists/*/sequence.mp4), not something
that should happen silently on every play.

Usage
─────
    from gemini_tactical_analyst import analyze_handball_clip

    result = analyze_handball_clip("reports/assists/1234_goal/sequence.mp4")
    print(result["play_description"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_NAME     = "gemini-1.5-pro"      # kept verbatim per spec — see note above
FALLBACK_MODEL = "gemini-flash-latest" # verified working July 2026 — see note below

# ── Processing / retry tunables ──────────────────────────────────────────────
UPLOAD_MAX_RETRIES     = 4
GENERATE_MAX_RETRIES   = 5
BASE_BACKOFF_S         = 2.0
MAX_BACKOFF_S          = 60.0
PROCESSING_TIMEOUT_S   = 180.0     # give a 10 s clip generous transcode time
PROCESSING_POLL_S      = 3.0
_TRANSIENT_HTTP_CODES  = {429, 500, 502, 503, 504}

SYSTEM_INSTRUCTION = """\
You are a world-class handball tactical analyst with decades of experience
coaching at the international level (IHF, EHF Champions League). You watch
match clips the way a head coach's video analyst would: identifying the
attacking system, the defensive block's reaction, the decisive action
(pass, screen, dribble, shot), and the players who made it happen.

When given a short video clip, analyze it and respond with:
  - A precise, coach-level description of the play (what actually happened,
    in the order it happened — not a generic summary).
  - The key players involved, identified by any visible jersey number,
    approximate position, or team color, with their specific role and
    action in the play (e.g. "left back, screened the defender at 9m").
  - A tactical analysis: WHY the play worked or failed — spacing, timing,
    defensive rotation errors, mismatch exploitation, etc.
  - One concrete tactical recommendation a coach could act on next time
    this situation arises.

Be specific and grounded in what is visibly happening in the video. If a
detail (e.g. a jersey number) isn't legible, describe the player by
position/color instead of guessing a number.
"""

DEFAULT_PROMPT = (
    "Analyze this handball play. Identify the attacking pattern, the "
    "decisive action, and every player who touched the ball or created "
    "space for it."
)


# ── Structured output schema ─────────────────────────────────────────────────
class KeyPlayer(BaseModel):
    identifier: str = Field(..., description=(
        "Jersey number if legible, else team color + position, "
        "e.g. '#7' or 'white team, left wing'"))
    role: str = Field(..., description="Their role in this specific play, "
                                       "e.g. 'passer', 'screener', 'shooter'")
    action: str = Field(..., description="What they specifically did")


class PlayAnalysis(BaseModel):
    play_description:        str
    key_players:              list[KeyPlayer]
    tactical_analysis:        str
    tactical_recommendation:  str


class GeminiAnalysisError(RuntimeError):
    """Raised for any non-retriable failure: auth, bad file, retired model
    with no working fallback, processing failure, or retries exhausted."""


# ── Retry helper ─────────────────────────────────────────────────────────────
def _http_code(exc: Exception) -> Optional[int]:
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _call_with_retries(fn, *, max_retries: int, label: str):
    attempt = 0
    while True:
        try:
            return fn()
        except genai_errors.APIError as e:
            code = _http_code(e)
            attempt += 1
            transient = code in _TRANSIENT_HTTP_CODES
            if not transient or attempt > max_retries:
                raise GeminiAnalysisError(
                    f"{label} failed (HTTP {code}, attempt {attempt}): {e}"
                ) from e
            wait = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
            wait += random.uniform(0, wait * 0.25)          # jitter
            print(f"[GeminiAnalyst] {label}: HTTP {code} — retry "
                  f"{attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:                              # network/SDK errors
            attempt += 1
            if attempt > max_retries:
                raise GeminiAnalysisError(
                    f"{label} failed (attempt {attempt}): {type(e).__name__}: {e}"
                ) from e
            wait = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
            print(f"[GeminiAnalyst] {label}: {type(e).__name__} — retry "
                  f"{attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)


def _is_model_not_found(exc: Exception) -> bool:
    code = _http_code(exc)
    msg = str(exc).lower()
    return code == 404 or "not found" in msg or "not_found" in msg


# ── Upload + wait-for-ACTIVE ─────────────────────────────────────────────────
def _upload_and_wait_active(client: "genai.Client", video_path: Path,
                            timeout_s: float, poll_s: float):
    uploaded = _call_with_retries(
        lambda: client.files.upload(file=str(video_path)),
        max_retries=UPLOAD_MAX_RETRIES, label="file upload")

    deadline = time.monotonic() + timeout_s
    while True:
        state_name = getattr(getattr(uploaded, "state", None), "name", None)
        if state_name == "ACTIVE":
            return uploaded
        if state_name == "FAILED":
            err = getattr(uploaded, "error", None)
            raise GeminiAnalysisError(
                f"Video processing FAILED on Google's side: {err or 'no detail'}")
        if time.monotonic() > deadline:
            raise GeminiAnalysisError(
                f"Video did not reach ACTIVE within {timeout_s:.0f}s "
                f"(last state: {state_name})")
        time.sleep(poll_s)
        uploaded = _call_with_retries(
            lambda: client.files.get(name=uploaded.name),
            max_retries=UPLOAD_MAX_RETRIES, label="file status poll")


def _generate(client: "genai.Client", model_name: str, uploaded_file,
             prompt: str) -> "types.GenerateContentResponse":
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=PlayAnalysis,
        temperature=0.3,
    )
    return _call_with_retries(
        lambda: client.models.generate_content(
            model=model_name, contents=[uploaded_file, prompt], config=config),
        max_retries=GENERATE_MAX_RETRIES, label=f"generate_content({model_name})")


def _cleanup(client: "genai.Client", uploaded_file) -> None:
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception as e:
        print(f"[GeminiAnalyst] cleanup warning (non-fatal): {e}")


# ── Public entry point ───────────────────────────────────────────────────────
def analyze_handball_clip(
    video_path: str | Path,
    *,
    api_key: Optional[str] = None,
    model_name: str = MODEL_NAME,
    fallback_model: Optional[str] = FALLBACK_MODEL,
    prompt: str = DEFAULT_PROMPT,
    processing_timeout_s: float = PROCESSING_TIMEOUT_S,
    poll_interval_s: float = PROCESSING_POLL_S,
    delete_after: bool = True,
) -> dict:
    """
    Upload a short handball clip to Gemini and return a structured tactical
    breakdown as a JSON-safe dict:

        {
          "play_description": str,
          "key_players": [{"identifier", "role", "action"}, ...],
          "tactical_analysis": str,
          "tactical_recommendation": str
        }

    Raises GeminiAnalysisError on any non-retriable failure (missing file,
    bad/missing API key, processing failure, retries exhausted, or a
    retired model with no usable fallback). Transient errors (HTTP 429
    rate limits, 5xx) are retried internally with exponential backoff.
    """
    path = Path(video_path)
    if not path.is_file():
        raise GeminiAnalysisError(f"Video file not found: {path}")
    if path.suffix.lower() not in (".mp4", ".mov", ".webm", ".mpeg", ".avi"):
        raise GeminiAnalysisError(f"Unsupported video extension: {path.suffix}")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiAnalysisError(
            "No API key. Set GEMINI_API_KEY (https://aistudio.google.com/apikey) "
            "or pass api_key=...")

    try:
        client = genai.Client(api_key=key)
    except Exception as e:
        raise GeminiAnalysisError(f"Failed to create Gemini client: {e}") from e

    uploaded = _upload_and_wait_active(client, path, processing_timeout_s,
                                       poll_interval_s)
    try:
        try:
            response = _generate(client, model_name, uploaded, prompt)
        except GeminiAnalysisError as e:
            cause = e.__cause__
            if fallback_model and cause is not None and _is_model_not_found(cause):
                print(f"[GeminiAnalyst] '{model_name}' unavailable "
                      f"(retired/not found) — retrying with '{fallback_model}'")
                response = _generate(client, fallback_model, uploaded, prompt)
            else:
                raise
    finally:
        if delete_after:
            _cleanup(client, uploaded)

    # Prefer the SDK's auto-parsed pydantic object; fall back to raw JSON
    # text if parsing didn't populate (defensive — schema drift, etc.).
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PlayAnalysis):
        return parsed.model_dump(mode="json")

    text = getattr(response, "text", None)
    if not text:
        raise GeminiAnalysisError("Empty response from Gemini (no .parsed, no .text)")
    try:
        return PlayAnalysis.model_validate(json.loads(text)).model_dump(mode="json")
    except Exception as e:
        raise GeminiAnalysisError(
            f"Could not parse structured JSON from response: {e}\nRaw text: {text[:500]}"
        ) from e


# ── CLI smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="Path to a short (~10s) mp4 clip")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--no-fallback", action="store_true",
                    help="Disable the gemini-2.5-pro fallback (see retired-model note)")
    args = ap.parse_args()

    try:
        result = analyze_handball_clip(
            args.video, api_key=args.api_key, model_name=args.model,
            fallback_model=None if args.no_fallback else FALLBACK_MODEL)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except GeminiAnalysisError as e:
        print(f"[GeminiAnalyst] FAILED: {e}")
        raise SystemExit(1)
