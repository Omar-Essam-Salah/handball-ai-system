"""
Tactical Coach Agent — asynchronous observer of the analytics snapshot,
augmented with In-Context Learning + RAG over a local knowledge base.

Completely decoupled from pipeline.py: polls logs/analytics_snapshot.npz on
its own interval, extracts a compact match-state JSON, and asks a local
Ollama model for one short tactical insight. Zero impact on the 60 FPS
main pipeline — runs as a separate Python process.
"""

import argparse
import ast
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    os.system("")

RESET     = "\033[0m"
DIM       = "\033[2m"        
BRIGHT    = "\033[1;36m"     
WARN      = "\033[33m"       

ROOT               = Path(__file__).parent.parent
SNAPSHOT_PATH      = ROOT / "logs" / "analytics_snapshot.npz"
KNOWLEDGE_DIR      = ROOT / "knowledge"
RULES_PATH         = KNOWLEDGE_DIR / "handball_rules.txt"
MEMORY_PATH        = KNOWLEDGE_DIR / "coach_memory.json"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL      = "llama3"
POLL_INTERVAL_S    = 10.0

RULES_MAX_CHARS    = 3000
MEMORY_MAX_ENTRIES = 20

BASE_SYSTEM_PROMPT = """You are a handball tactical coach analyzing live match statistics.

INPUT: A JSON match snapshot with these per-team fields:
  - total_distance_m, top_speed_m_s            (physical output)
  - skills: {jump_shot, pivot_turn, block}     (technical actions)
  - fatigue_avg                                (0.0 fresh → 1.0 fully fatigued, mean across players)
  - fatigue_high                               (count of players whose fatigue exceeds 0.5)
  - injury_alerts                              (count of players whose top speed dropped suddenly)
  - team_a_heatmap / team_b_heatmap            ({left, center, right}, sums to 1.0)

Plus league-wide:
  - skill_totals                               (totals across both teams)
  - tactic_totals: {wing_crossing, fast_throw_off}   (composite tactic detections)
  - recent_tactics                             (latest 10, with frame_n, name, team)

OUTPUT FORMAT (STRICT):
Respond with a single valid JSON object — nothing before it, nothing after, no
markdown fences. The object MUST have exactly these two keys:

  "internal_reasoning": Step-by-step thinking.
  "tactical_insight": ONE short tactical insight (1-2 sentences max). Final coaching advice.
"""

class _Knowledge:
    def __init__(self):
        self.rules_text:  str   = ""
        self.memory:      list  = []
        self._rules_mtime: float = -1.0
        self._memory_mtime: float = -1.0

    def refresh(self) -> bool:
        changed = False
        try:
            if RULES_PATH.exists():
                m = RULES_PATH.stat().st_mtime
                if m > self._rules_mtime:
                    text = RULES_PATH.read_text(encoding="utf-8", errors="replace").strip()
                    if len(text) > RULES_MAX_CHARS:
                        text = text[:RULES_MAX_CHARS].rsplit("\n", 1)[0] + "\n[...truncated]"
                    self.rules_text   = text
                    self._rules_mtime = m
                    changed = True
            else:
                if self._rules_mtime != -1.0:
                    self.rules_text   = ""
                    self._rules_mtime = -1.0
                    changed = True
        except Exception as e:
            pass

        try:
            if MEMORY_PATH.exists():
                m = MEMORY_PATH.stat().st_mtime
                if m > self._memory_mtime:
                    with MEMORY_PATH.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.memory = data
                    self._memory_mtime = m
                    changed = True
        except Exception as e:
            pass

        return changed

def ensure_memory_file() -> None:
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        if not MEMORY_PATH.exists():
            tmp = MEMORY_PATH.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump([], f)
            tmp.replace(MEMORY_PATH)
    except OSError as e:
        pass

def build_system_prompt(rules_text: str, memory: list) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    if rules_text:
        parts.append("\n\nHANDBALL RULES REFERENCE:\n" + rules_text)
    if memory:
        recent = memory[-MEMORY_MAX_ENTRIES:]
        lines = ["\n\nYOUR PAST MISTAKES & CORRECTIONS:",
                 "(Study these. Do NOT repeat the same mistake.)"]
        for i, entry in enumerate(recent, 1):
            if not isinstance(entry, dict): continue
            ctx   = str(entry.get("context", "")).strip()
            wrong = str(entry.get("wrong_ai_insight", "")).strip()
            corr  = str(entry.get("coach_correction", "")).strip()
            if not (ctx or wrong or corr): continue
            lines.append(f"{i}. Context: {ctx}")
            lines.append(f"   Wrong AI insight:  {wrong}")
            lines.append(f"   Coach correction:  {corr}")
        parts.append("\n".join(lines))
    return "".join(parts)

def _safe_literal(blob, default):
    if blob is None: return default
    try:
        if hasattr(blob, "shape") and blob.shape == (1,): blob = blob[0]
        if isinstance(blob, (bytes, np.bytes_)): blob = blob.decode("utf-8", errors="replace")
        return ast.literal_eval(str(blob))
    except (ValueError, SyntaxError, TypeError):
        return default

def load_snapshot(path: Path) -> dict | None:
    if not path.exists(): return None
    try:
        with np.load(str(path), allow_pickle=True) as npz:
            files = set(npz.files)
            return {
                "team_a_heat":   np.asarray(npz["team_a_heat"])  if "team_a_heat"  in files else None,
                "team_b_heat":   np.asarray(npz["team_b_heat"])  if "team_b_heat"  in files else None,
                "ball_heat":     np.asarray(npz["ball_heat"])    if "ball_heat"    in files else None,
                "stats":         _safe_literal(npz["stats"]         if "stats"         in files else None, {}),
                "skill_totals":  _safe_literal(npz["skill_totals"]  if "skill_totals"  in files else None, {}),
                "tactic_totals": _safe_literal(npz["tactic_totals"] if "tactic_totals" in files else None, {}),
                "tactic_events": _safe_literal(npz["tactic_events"] if "tactic_events" in files else None, []),
            }
    except Exception as e:
        return None

def _heatmap_thirds(heat) -> dict:
    if heat is None or heat.size == 0: return {"left": 0.0, "center": 0.0, "right": 0.0}
    total = float(heat.sum())
    if total <= 0.0: return {"left": 0.0, "center": 0.0, "right": 0.0}
    w = heat.shape[1]
    a, b = w // 3, 2 * w // 3
    return {
        "left":   round(float(heat[:, :a].sum())  / total, 3),
        "center": round(float(heat[:, a:b].sum()) / total, 3),
        "right":  round(float(heat[:, b:].sum())  / total, 3),
    }

def _aggregate_team(stats: dict, team: str) -> dict:
    out = {
        "players_seen":       0,
        "total_distance_m":   0.0,
        "top_speed_m_s":      0.0,
        "skills":             {"jump_shot": 0, "pivot_turn": 0, "block": 0},
        # NEW: fatigue & injury risk rollups
        "fatigue_avg":        0.0,
        "fatigue_high":       0,    # players with score > 0.5
        "injury_alerts":      0,    # players with injury_alert_frame >= 0
    }
    if not isinstance(stats, dict): return out
    fatigue_sum = 0.0
    fatigue_n   = 0
    for _tid, s in stats.items():
        if not isinstance(s, dict) or s.get("team") != team: continue
        out["players_seen"] += 1
        d = s.get("distance_m") or s.get("distance_px", 0.0)
        out["total_distance_m"] += float(d or 0.0)
        v = s.get("max_speed_m_s") or s.get("max_speed_px_s", 0.0)
        out["top_speed_m_s"] = max(out["top_speed_m_s"], float(v or 0.0))
        sk = s.get("skills_count") or {}
        for k, n in sk.items(): out["skills"][k] = out["skills"].get(k, 0) + int(n or 0)
        # Fatigue rollup
        f = float(s.get("fatigue_score", 0.0) or 0.0)
        if f > 0.0:
            fatigue_sum += f
            fatigue_n   += 1
            if f > 0.5:
                out["fatigue_high"] += 1
        if int(s.get("injury_alert_frame", -1)) >= 0:
            out["injury_alerts"] += 1
    out["total_distance_m"] = round(out["total_distance_m"], 1)
    out["top_speed_m_s"]    = round(out["top_speed_m_s"], 2)
    out["fatigue_avg"]      = round(fatigue_sum / fatigue_n, 3) if fatigue_n else 0.0
    return out

def build_prompt_json(snap: dict) -> dict:
    stats = snap.get("stats") or {}
    # Recent tactic events (last 10) — gives the LLM a sense of momentum
    events = snap.get("tactic_events") or []
    recent_tactics = []
    for ev in events[-10:]:
        try:
            recent_tactics.append({"frame_n": int(ev[0]), "name": str(ev[1]), "team": str(ev[2])})
        except Exception:
            pass
    return {
        "team_a":          _aggregate_team(stats, "team_a"),
        "team_b":          _aggregate_team(stats, "team_b"),
        "team_a_heatmap":  _heatmap_thirds(snap.get("team_a_heat")),
        "team_b_heatmap":  _heatmap_thirds(snap.get("team_b_heat")),
        "ball_heatmap":    _heatmap_thirds(snap.get("ball_heat")),
        "skill_totals":    snap.get("skill_totals") or {},
        # NEW
        "tactic_totals":   snap.get("tactic_totals") or {},
        "recent_tactics":  recent_tactics,
    }

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def parse_insight_response(raw: str | None) -> tuple[str | None, str | None]:
    """Pull (internal_reasoning, tactical_insight) out of an LLM response.
       Tolerant of markdown fences, preamble, trailing prose, minor JSON drift."""
    if not raw:
        return None, None
    s = _strip_code_fence(raw)
    i = s.find("{")
    j = s.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(s[i:j + 1])
            if isinstance(obj, dict):
                r = obj.get("internal_reasoning")
                t = obj.get("tactical_insight")
                r = str(r).strip() if r is not None else None
                t = str(t).strip() if t is not None else None
                return r, t
        except json.JSONDecodeError:
            pass
    return None, s.strip() or None


def query_ollama(prompt_json: dict, model: str, url: str,
                 timeout: float = 30.0, system_prompt: str | None = None):
    sys_p = system_prompt if system_prompt is not None else BASE_SYSTEM_PROMPT
    body = json.dumps({
        "model":   model,
        "prompt":  sys_p + "\n\nMATCH JSON:\n" + json.dumps(prompt_json, indent=2),
        "stream":  False,
        "options": {"temperature": 0.3, "num_predict": 280},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = (data.get("response") or "").strip()
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, OSError, TimeoutError):
        return None


def _print_cycle(prompt_json: dict, reasoning: str | None, insight: str | None,
                 llm_disabled: bool):
    ts = time.strftime("%H:%M:%S")
    a, b = prompt_json["team_a"], prompt_json["team_b"]
    ha, hb = prompt_json["team_a_heatmap"], prompt_json["team_b_heatmap"]
    print("─" * 64)
    print(f"[Coach] {ts}  snapshot updated")
    print(f"  Team A  dist={a['total_distance_m']}m  top={a['top_speed_m_s']} m/s  "
          f"fatigue_avg={a.get('fatigue_avg', 0)}  injured={a.get('injury_alerts', 0)}  skills={a['skills']}")
    print(f"  Team B  dist={b['total_distance_m']}m  top={b['top_speed_m_s']} m/s  "
          f"fatigue_avg={b.get('fatigue_avg', 0)}  injured={b.get('injury_alerts', 0)}  skills={b['skills']}")
    print(f"  Heat A  L/C/R = {ha['left']} / {ha['center']} / {ha['right']}")
    print(f"  Heat B  L/C/R = {hb['left']} / {hb['center']} / {hb['right']}")
    tt = prompt_json.get("tactic_totals") or {}
    if tt:
        print(f"  Tactics: wing_crossing={tt.get('wing_crossing', 0)}  "
              f"fast_throw_off={tt.get('fast_throw_off', 0)}")
    if reasoning:
        print(f"  {DIM}[Thinking...]{RESET}")
        for line in reasoning.splitlines() or [reasoning]:
            print(f"  {DIM}  {line}{RESET}")
    if insight:
        print(f"  {BRIGHT}Insight:{RESET} {insight}")
    elif llm_disabled:
        print("  Insight: (LLM disabled by --no-llm)")
    else:
        print(f"  {WARN}Insight: (LLM offline — start `ollama serve` and `ollama pull <model>`){RESET}")


def main():
    p = argparse.ArgumentParser(description="Async tactical coach for the Handball pipeline.")
    p.add_argument("--snapshot",   default=str(SNAPSHOT_PATH))
    p.add_argument("--interval",   type=float, default=POLL_INTERVAL_S)
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--once",       action="store_true")
    p.add_argument("--no-llm",     action="store_true")
    p.add_argument("--timeout",    type=float, default=30.0)
    args = p.parse_args()

    snap_path = Path(args.snapshot)
    last_mtime = -1.0
    waiting_logged = False

    ensure_memory_file()
    knowledge = _Knowledge()
    knowledge.refresh()

    print(f"[Coach] watching {snap_path}")
    print(f"[Coach] knowledge dir: {KNOWLEDGE_DIR}  "
          f"(rules={'yes' if knowledge.rules_text else 'no'}, "
          f"memory_entries={len(knowledge.memory)})")
    print(f"[Coach] interval={args.interval}s  model={args.model}  llm={'OFF' if args.no_llm else 'ON'}")

    try:
        while True:
            knowledge.refresh()
            try:
                cur_mtime = snap_path.stat().st_mtime if snap_path.exists() else -1.0
            except OSError:
                cur_mtime = -1.0

            if cur_mtime <= 0:
                if not waiting_logged:
                    print(f"[Coach] waiting for snapshot at {snap_path} ...")
                    waiting_logged = True
            elif cur_mtime > last_mtime:
                waiting_logged = False
                snap = load_snapshot(snap_path)
                if snap is not None:
                    last_mtime = cur_mtime
                    prompt_json = build_prompt_json(snap)
                    reasoning, insight = None, None
                    if not args.no_llm:
                        sys_prompt = build_system_prompt(knowledge.rules_text, knowledge.memory)
                        raw = query_ollama(prompt_json, args.model, args.ollama_url,
                                           args.timeout, system_prompt=sys_prompt)
                        reasoning, insight = parse_insight_response(raw)
                    _print_cycle(prompt_json, reasoning, insight, llm_disabled=args.no_llm)

            if args.once:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[Coach] stopped.")


if __name__ == "__main__":
    main()