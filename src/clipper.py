"""Export a clip per tagged event using ffmpeg."""
import subprocess
from pathlib import Path
from typing import List

from tagger import Event


def export_clips(
    video: Path,
    events: List[Event],
    out_dir: Path,
    pre_s: int = 5,
    post_s: int = 3,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    n = 0
    for ev in events:
        start = max(0.0, ev.timestamp_ms / 1000.0 - pre_s)
        dur = pre_s + post_s
        key = f"{ev.event_type}_{ev.team}"
        counts[key] = counts.get(key, 0) + 1
        name = f"{key}_{counts[key]:03d}.mp4"
        out_path = out_dir / name
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video),
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {name}:\n{r.stderr[-600:]}"
            )
        n += 1
    return n
