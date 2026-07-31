"""Build a summary CSV: counts per event_type x team."""
from pathlib import Path
from typing import List

import pandas as pd

from tagger import Event


def build_summary(events: List[Event], csv_path: Path) -> None:
    if not events:
        df = pd.DataFrame(columns=["event_type", "team", "count"])
    else:
        rows = [{"event_type": e.event_type, "team": e.team} for e in events]
        df = (
            pd.DataFrame(rows)
            .groupby(["event_type", "team"])
            .size()
            .reset_index(name="count")
            .sort_values(["team", "event_type"])
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
