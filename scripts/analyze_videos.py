"""
Batch match analyzer — CV pipeline (eyes) + Ollama (brain) over ALL videos.

For each video it runs the CV pipeline HEADLESS (no window) to extract structured
match facts (score, shots, passes, fast breaks, goal timeline), then asks Ollama
(CPU-only, num_gpu=0 — never touches the YOLO GPU) to write an English match
report. One report per video.

    .venv\\Scripts\\python.exe scripts\\analyze_videos.py                 # all default videos
    .venv\\Scripts\\python.exe scripts\\analyze_videos.py --videos 22.mp4 --minutes 10
    .venv\\Scripts\\python.exe scripts\\analyze_videos.py --model llama3.2 --minutes 0

Prereqs:  `ollama serve` running + a text model pulled (`ollama pull llama3.2`).
Honest note: the CV extraction is the slow part (~real-time). Use --minutes to
cap per video while testing; --minutes 0 = full match. Report quality is only as
good as the CV facts — better ball/player accuracy → better analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import requests
import pipeline as pl

OUT = ROOT / "reports" / "analysis"
DEFAULT_VIDEOS = ["22.mp4", "33.mp4", "handball.mp4", "hand3.mp4",
                  "handi.mp4", "videoplayback.mp4"]


def extract(video: str, max_frames: int) -> dict:
    """Run the CV pipeline headless and collect structured match facts."""
    facts = {"hud": {}, "events": set(), "goals": []}
    prev = {"team_a": 0, "team_b": 0}

    def status_cb(s):
        hud = s.get("hud", {})
        facts["hud"] = hud
        for line in s.get("events", []):
            facts["events"].add(line)
        sc = hud.get("score", {}) or {}
        for t in ("team_a", "team_b"):
            if int(sc.get(t, 0)) > prev[t]:
                prev[t] = int(sc.get(t, 0))
                facts["goals"].append({
                    "team": t, "frame": s.get("frame_n"),
                    "by": s.get("last_goal", "")})

    pl.run(source=str(ROOT / video), gui_mode=True,
           frame_callback=lambda f: None, status_callback=status_cb,
           analytics=False, no_ocr=False, max_frames=max_frames)
    facts["events"] = sorted(facts["events"])
    return facts


def summarize(video: str, facts: dict) -> dict:
    hud = facts.get("hud", {})
    sc = hud.get("score", {}) or {}
    sh = hud.get("shots", {}) or {}
    ps = hud.get("passes", {}) or {}
    fb = hud.get("fast_breaks", {}) or {}
    return {
        "video": video,
        "final_score": {"team_a": sc.get("team_a", 0), "team_b": sc.get("team_b", 0)},
        "shots": sh, "passes": ps, "fast_breaks": fb,
        "goals_timeline": facts.get("goals", []),
        "event_sample": facts.get("events", [])[-25:],
    }


def report(summary: dict, model: str, url: str, timeout: float) -> str:
    prompt = (
        "You are a professional handball analyst. Write a concise ENGLISH match "
        "report from the structured facts below. Use these sections:\n"
        "  • Result — final score and who won\n"
        "  • Flow — shots, passes, fast breaks, what the numbers imply\n"
        "  • Key moments — from the goal timeline\n"
        "  • Tactical read — 2-3 observations\n"
        "Be factual; do NOT invent player names or events not in the data.\n\n"
        f"MATCH FACTS (JSON):\n{json.dumps(summary, ensure_ascii=False)}\n\nReport:")
    try:
        r = requests.post(url, timeout=timeout, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_gpu": 0, "temperature": 0.3, "num_predict": 500}})
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        return f"[Ollama unavailable: {e}]  (start `ollama serve` + pull {model})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    ap.add_argument("--minutes", type=float, default=0, help="cap per video (0=full)")
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--url", default="http://localhost:11434/api/generate")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        src = ROOT / v
        if not src.exists():
            print(f"[skip] {v} not found")
            continue
        print(f"\n===== analyzing {v} =====")
        t0 = time.time()
        # frames cap: assume ~25-50 fps; minutes*60*50 is a safe upper bound
        max_frames = int(a.minutes * 60 * 50) if a.minutes else 0
        facts = extract(v, max_frames)
        summary = summarize(v, facts)
        text = report(summary, a.model, a.url, a.timeout)
        stem = Path(v).stem
        (OUT / f"{stem}.json").write_text(json.dumps(summary, indent=2,
                                          ensure_ascii=False), encoding="utf-8")
        md = (f"# Match Report — {v}\n\n"
              f"**Final score:** {summary['final_score']['team_a']} – "
              f"{summary['final_score']['team_b']}\n\n{text}\n")
        (OUT / f"{stem}.md").write_text(md, encoding="utf-8")
        print(f"  score {summary['final_score']}  goals={len(summary['goals_timeline'])}"
              f"  ({time.time()-t0:.0f}s)  -> reports/analysis/{stem}.md")
    print(f"\nReports in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
