"""
AI Event Verifier — the "LLM behind YOLO".

YOLO + the tracker are the fast EYES (ball position every frame). They can't
*reason* about a play. A vision-LLM is the slow BRAIN: it looks at a few key
frames of a moment and judges the OUTCOME — goal / save / miss / pass — and can
catch mistakes the geometric rules engine makes.

It only runs on EVENTS (a goal fires a few times per match), in a background
thread, so it never slows the real-time pipeline. Fully optional: if Ollama or a
vision model isn't available it silently disables and the system runs normally.

Backend: local Ollama (http://localhost:11434), vision model (gemma3/4, llava…).
Activate a model with e.g.  `ollama pull llava`  — or use the gemma vision model
already installed. No API key, no cloud, free.
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests

OLLAMA_BASE = "http://localhost:11434"
# Models we know can see images, best-first. Auto-picked from what's installed.
VISION_CANDIDATES = ["llava", "llama3.2-vision", "gemma4:e4b", "gemma3", "bakllava",
                     "moondream", "minicpm-v"]


class EventVerifier:
    def __init__(self, model: Optional[str] = None,
                 base: str = OLLAMA_BASE, max_frames: int = 3,
                 timeout_s: float = 120.0):
        self.base = base.rstrip("/")
        self.max_frames = int(max_frames)
        self.timeout_s = float(timeout_s)
        self.model = model or self._auto_pick_model()
        self.available = self.model is not None
        self.latest: Optional[dict] = None        # most recent verdict
        self.history: list[dict] = []
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        if self.available:
            threading.Thread(target=self._run, daemon=True, name="EventVerifier").start()
            print(f"[AI-Verify] enabled — vision model = {self.model}")
        else:
            print("[AI-Verify] disabled — no Ollama vision model "
                  "(install one with:  ollama pull llava)")

    # ── model discovery ──
    def _auto_pick_model(self) -> Optional[str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return None
        for cand in VISION_CANDIDATES:
            for n in names:
                if n == cand or n.startswith(cand) or n.split(":")[0] == cand:
                    return n
        return None

    # ── public API (called from the pipeline, non-blocking) ──
    def verify_async(self, frames: list, event_type: str, team: str, frame_n: int) -> None:
        """Queue a verification. Drops the request if busy (events are rare)."""
        if not self.available or not frames:
            return
        try:
            self._q.put_nowait((list(frames), event_type, team, frame_n))
        except queue.Full:
            pass

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return dict(self.latest) if self.latest else None

    # ── worker ──
    def _run(self) -> None:
        while True:
            frames, et, team, fn = self._q.get()
            try:
                verdict = self._verify(frames, et, team)
                verdict.update({"event": et, "team": team, "frame_n": fn,
                                "ts": time.time()})
                with self._lock:
                    self.latest = verdict
                    self.history.append(verdict)
                    self.history = self.history[-20:]
                print(f"[AI-Verify] {et.upper()} ({team}) → "
                      f"{verdict.get('outcome','?').upper()} "
                      f"({verdict.get('confidence',0):.0%})  "
                      f"{verdict.get('reason','')[:70]}")
            except Exception as e:
                print(f"[AI-Verify] failed: {type(e).__name__}: {e}")

    def _encode(self, frame: np.ndarray) -> str:
        if frame.shape[1] > 768:                  # shrink for speed
            s = 768 / frame.shape[1]
            frame = cv2.resize(frame, (768, int(frame.shape[0] * s)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()

    def _verify(self, frames: list, et: str, team: str) -> dict:
        # sample up to max_frames evenly across the moment
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
            frames = [frames[i] for i in idx]
        images = [self._encode(f) for f in frames]
        side = "Home" if team == "team_a" else "Away"
        prompt = (
            f"These are consecutive frames from a HANDBALL match, around a shot by the "
            f"{side} team. Judge the OUTCOME of the shot. Choose exactly one:\n"
            f"- GOAL  (ball entered the net)\n"
            f"- SAVE  (goalkeeper stopped it)\n"
            f"- MISS  (went wide/over/out, no goal)\n"
            f"- PASS  (it was a pass, not a shot)\n"
            f"Reply ONLY as compact JSON: "
            f'{{"outcome":"GOAL|SAVE|MISS|PASS","confidence":0.0-1.0,"reason":"<6 words"}}'
        )
        payload = {"model": self.model, "prompt": prompt, "images": images,
                   "stream": False, "options": {"temperature": 0.1, "num_predict": 80}}
        r = requests.post(f"{self.base}/api/generate", json=payload, timeout=self.timeout_s)
        txt = r.json().get("response", "").strip()
        return self._parse(txt)

    @staticmethod
    def _parse(txt: str) -> dict:
        # find the JSON blob even if the model wrapped it in prose
        try:
            a, b = txt.index("{"), txt.rindex("}") + 1
            obj = json.loads(txt[a:b])
            return {
                "outcome": str(obj.get("outcome", "?")).upper(),
                "confidence": float(obj.get("confidence", 0.0)),
                "reason": str(obj.get("reason", ""))[:120],
            }
        except Exception:
            up = txt.upper()
            for k in ("GOAL", "SAVE", "MISS", "PASS"):
                if k in up:
                    return {"outcome": k, "confidence": 0.5, "reason": txt[:80]}
            return {"outcome": "?", "confidence": 0.0, "reason": txt[:80]}
