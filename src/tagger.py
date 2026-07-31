"""Tag store: load/save events as JSON sidecar next to the video."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Event:
    timestamp_ms: int
    event_type: str
    team: str
    note: str = ""


class TagStore:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.sidecar = self.video_path.with_suffix(self.video_path.suffix + ".tags.json")
        self.events: List[Event] = []
        self.load()

    def load(self) -> None:
        if self.sidecar.exists():
            data = json.loads(self.sidecar.read_text(encoding="utf-8"))
            self.events = [Event(**e) for e in data.get("events", [])]

    def save(self) -> None:
        payload = {
            "video": self.video_path.name,
            "events": [asdict(e) for e in self.events],
        }
        self.sidecar.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, ev: Event) -> None:
        self.events.append(ev)
        self.events.sort(key=lambda e: e.timestamp_ms)

    def pop(self) -> bool:
        if not self.events:
            return False
        self.events.pop()
        return True
