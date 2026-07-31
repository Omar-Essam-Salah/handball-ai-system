"""
OllamaAnalyst — async tactical evaluator running on the CPU.

Daemon thread that wakes every CHECK_INTERVAL_S (default 5 min) — or on
manual trigger — reads match_log.json + coach_rules.txt, asks Ollama for a
tactical evaluation, and writes the result to tactical_insight.json.

Isolation guarantees from the video pipeline
────────────────────────────────────────────
1. SEPARATE OS THREAD     — the analyst runs in its own threading.Thread.
                            The main vision loop never touches it and
                            never waits on it.
2. CPU-ONLY OLLAMA        — the payload includes "num_gpu": 0.  Ollama will
                            not try to load any model layers into VRAM.
                            The RTX 3000 stays fully reserved for YOLO + ReID.
3. HARD HTTP TIMEOUT      — requests.post(..., timeout=HTTP_TIMEOUT_S).  A
                            hung Ollama process cannot pile up open sockets
                            or block this thread forever.
4. EVERY EXCEPTION SWALLOWED — the analyst can fail silently, but it cannot
                            crash the engine.
5. READS THE JSON FILE    — no in-memory coupling to MatchLogger required.
                            Works even if the analyst runs in a separate
                            process from the pipeline.

Public API
──────────
    analyst = OllamaAnalyst(model="llama3.2")
    analyst.start()                # begin background loop
    analyst.request_now()          # force one immediate evaluation
    analyst.latest_insight()       # → dict | None
    analyst.stop()                 # graceful shutdown
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import requests


PROJECT_ROOT         = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH     = PROJECT_ROOT / "match_log.json"
DEFAULT_RULES_PATH   = PROJECT_ROOT / "coach_rules.txt"
DEFAULT_INSIGHT_PATH = PROJECT_ROOT / "tactical_insight.json"

# Cadence and budgets
CHECK_INTERVAL_S = 5 * 60      # 5 minutes between automatic evaluations
EVENT_WINDOW_S   = 5 * 60      # only events from the last 5 min go to the LLM
MAX_EVENTS       = 80          # hard cap on events per prompt
HTTP_TIMEOUT_S   = 60.0        # generous CPU inference budget for llama3.2
INITIAL_DELAY_S  = 30.0        # let some events accumulate before first call

DEFAULT_RULES = """\
You are a handball tactical analyst for the head coach.
Evaluate the recent live match events and output 1 to 3 short coaching
commands.  Focus on:
  - Defensive line drift / numerical superiority breaks
  - Goalkeeper positioning and reactions
  - Attacking set-play execution and patience
  - Player fatigue and substitution timing
  - Foul management (2-min suspensions)

Format each command on its own line, no preamble:
    TAG: text
where TAG is one of  WARN | INFO | SUB | TIMEOUT.

Be terse — every line must be actionable in under 5 seconds.
"""


class OllamaAnalyst:
    """Background tactical-evaluation thread.  Strictly CPU-only."""

    def __init__(
        self,
        log_path:    Path  = DEFAULT_LOG_PATH,
        rules_path:  Path  = DEFAULT_RULES_PATH,
        insight_path: Path = DEFAULT_INSIGHT_PATH,
        url:         str   = "http://localhost:11434/api/generate",
        model:       str   = "llama3.2",
        interval_s:  float = CHECK_INTERVAL_S,
        timeout_s:   float = HTTP_TIMEOUT_S,
    ):
        self.log_path     = Path(log_path)
        self.rules_path   = Path(rules_path)
        self.insight_path = Path(insight_path)
        self.url          = url
        self.model        = model
        self.interval_s   = float(interval_s)
        self.timeout_s    = float(timeout_s)

        # Make sure a coach_rules.txt exists so users can edit it without
        # remembering the format.  Default content is a sane starting point.
        if not self.rules_path.exists():
            try:
                self.rules_path.write_text(DEFAULT_RULES, encoding="utf-8")
                print(f"[OllamaAnalyst] Wrote default rules → "
                      f"{self.rules_path.name}  (edit this file to tune the LLM)")
            except Exception as exc:
                print(f"[OllamaAnalyst] Could not create rules file: {exc}")

        self._stop    = threading.Event()
        self._trigger = threading.Event()
        self._lock    = threading.RLock()
        self._latest_insight: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    # ── public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="OllamaAnalyst", daemon=True,
        )
        self._thread.start()
        print(f"[OllamaAnalyst] Started  model={self.model}  "
              f"every {self.interval_s/60:.1f} min  "
              f"timeout={self.timeout_s:.0f} s  url={self.url}")

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        self._trigger.set()   # release any pending wait
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def request_now(self) -> None:
        """Wake the thread immediately for an out-of-cycle evaluation."""
        self._trigger.set()

    def latest_insight(self) -> Optional[dict]:
        """Return a copy of the most recent insight (or None)."""
        with self._lock:
            return None if self._latest_insight is None else dict(self._latest_insight)

    def evaluate_now_blocking(self, timeout_s: Optional[float] = None) -> dict:
        """
        Fire one evaluation synchronously on the CALLING thread.  Useful for
        a dashboard "Ask the analyst" button that wants to surface a result
        in the same HTTP response.  This DOES block the caller for as long
        as Ollama takes — never call it from the video loop.
        """
        return self._tick(blocking_timeout=timeout_s)

    # ── thread loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        # Initial delay — let events accumulate before the first analysis.
        # Returns early if a manual trigger or stop arrives.
        self._trigger.wait(timeout=INITIAL_DELAY_S)

        while not self._stop.is_set():
            # Clear BEFORE the tick so that a manual trigger arriving
            # during the tick is preserved for the next iteration.
            self._trigger.clear()
            try:
                self._tick()
            except Exception as exc:
                print(f"[OllamaAnalyst] tick failed (continuing): {exc}")

            # Wait until interval expires OR a manual trigger fires OR stop().
            self._trigger.wait(timeout=self.interval_s)

        print("[OllamaAnalyst] Stopped.")

    # ── single-evaluation tick ──────────────────────────────────────────────

    def _tick(self, blocking_timeout: Optional[float] = None) -> dict:
        rules  = self._read_rules()
        events = self._read_events()
        if not events:
            empty = {
                "ts":       round(time.time(), 3),
                "model":    self.model,
                "n_events": 0,
                "response": "(no events in window — no analysis run)",
                "skipped":  True,
            }
            with self._lock:
                self._latest_insight = empty
            return empty

        prompt = self._build_prompt(rules, events)
        text   = self._call_ollama(prompt, timeout_s=blocking_timeout or self.timeout_s)

        insight = {
            "ts":          round(time.time(), 3),
            "model":       self.model,
            "n_events":    len(events),
            "rules_chars": len(rules),
            "prompt_chars": len(prompt),
            "response":    text,
        }
        with self._lock:
            self._latest_insight = insight
        self._persist(insight)

        first_line = (text.splitlines()[0] if text else "(empty)")[:120]
        print(f"[OllamaAnalyst] Insight ready ({len(events)} events): {first_line}")
        return insight

    # ── helpers ─────────────────────────────────────────────────────────────

    def _read_rules(self) -> str:
        try:
            return self.rules_path.read_text(encoding="utf-8")
        except Exception:
            return DEFAULT_RULES

    def _read_events(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        try:
            data = json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[OllamaAnalyst] Could not read {self.log_path.name}: {exc}")
            return []
        if not isinstance(data, list):
            return []

        # Filter to recent window and cap event count
        cutoff = time.time() - EVENT_WINDOW_S
        recent = [e for e in data
                  if isinstance(e, dict) and e.get("ts", 0) >= cutoff]
        return recent[-MAX_EVENTS:]

    @staticmethod
    def _build_prompt(rules: str, events: list[dict]) -> str:
        # Compact JSON: drop empty fields to keep the prompt small.
        compact = []
        for e in events:
            row = {k: v for k, v in e.items()
                   if v not in ("", None, {}, [])}
            compact.append(row)
        events_str = json.dumps(compact, separators=(",", ":"),
                                ensure_ascii=False)

        return (
            f"COACH RULES:\n{rules}\n\n"
            f"LAST {len(events)} EVENTS (most recent last):\n"
            f"{events_str}\n\n"
            f"Output your tactical commands now.  No preamble."
        )

    def _call_ollama(self, prompt: str, timeout_s: float) -> str:
        """
        POST to /api/generate with num_gpu=0.  Returns the response text or
        a short error string starting with '['; never raises.
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                # ── CRITICAL: keep the analyst off the GPU ──────────────
                # Without this, Ollama would attempt to load llama3.2 into
                # VRAM and compete with YOLO/OSNet.  num_gpu=0 forces 100%
                # CPU inference inside the ollama serve daemon, which runs
                # in its own OS process.
                "num_gpu":     0,
                "num_predict": 240,
                "temperature": 0.30,
                "top_p":       0.90,
                "repeat_penalty": 1.15,
                "stop":        ["\n\n\n"],
            },
        }
        try:
            r = requests.post(self.url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except requests.exceptions.Timeout:
            return f"[OllamaAnalyst] timed out after {timeout_s:.0f}s"
        except requests.exceptions.ConnectionError:
            return f"[OllamaAnalyst] Ollama not reachable at {self.url} " \
                   f"(is `ollama serve` running?)"
        except Exception as exc:
            return f"[OllamaAnalyst] error: {exc}"

    def _persist(self, insight: dict) -> None:
        try:
            self.insight_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.insight_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(insight, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.insight_path)
        except Exception as exc:
            print(f"[OllamaAnalyst] Could not persist insight: {exc}")


# ── ad-hoc CLI for testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OllamaAnalyst once.")
    parser.add_argument("--model",      default="llama3.2")
    parser.add_argument("--url",        default="http://localhost:11434/api/generate")
    parser.add_argument("--timeout",    type=float, default=60.0)
    parser.add_argument("--once",       action="store_true",
                        help="Evaluate once synchronously and exit")
    args = parser.parse_args()

    analyst = OllamaAnalyst(
        url=args.url, model=args.model, timeout_s=args.timeout,
    )
    if args.once:
        result = analyst.evaluate_now_blocking()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        analyst.start()
        try:
            print("[CLI] Analyst running. Ctrl+C to stop.")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[CLI] stopping…")
        finally:
            analyst.stop()