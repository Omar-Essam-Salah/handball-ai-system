"""
Match identity — turns generic on-court IDs into real team + player names.

The pipeline tracks players as colour-clustered teams ("team_a" / "team_b") and
reads jersey numbers via OCR. This module bridges that to the seeded rosters so
the HUD shows "France" / "Poland" instead of "Home" / "Away", and a tracked
player with jersey #13 on the France side shows "Karabatic #13" instead of
"B#13".

Design notes
------------
* Colour clustering can't know WHICH cluster is France vs Poland, so the
  team_a→real-team mapping is just a default that the user can flip at runtime
  (swap()) — the choice persists to config/match.json so it survives a restart.
* Jersey numbers can collide across teams (both squads have a #3, #13...), so a
  name lookup ALWAYS needs the team key — never look up by number alone.
* Everything degrades gracefully: if the league DB is empty or unreadable, this
  falls back to "Home"/"Away" and number-only labels, exactly like before.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "match.json"


def _surname(full_name: str) -> str:
    """'Nikola Karabatic' -> 'Karabatic'. Keeps the part most useful on a HUD."""
    parts = str(full_name).split()
    return parts[-1] if parts else str(full_name)


class MatchIdentity:
    def __init__(self, home="France", away="Poland", db_path=None):
        # team_a is the "home" colour cluster by convention; swap() flips it.
        self._names = {"team_a": home, "team_b": away}
        # real-team-name -> {jersey_number: surname}
        self._rosters: dict[str, dict[int, str]] = {}
        self._load_config()
        self._load_rosters(db_path)

    # ── rosters ────────────────────────────────────────────────────────────
    def _load_rosters(self, db_path=None) -> None:
        try:
            from league_db import LeagueDB
        except Exception:
            try:
                from src.league_db import LeagueDB
            except Exception:
                return
        try:
            db = LeagueDB() if db_path is None else LeagueDB(db_path)
            for row in db.con.execute("SELECT id, name FROM clubs").fetchall():
                club_name = row["name"]
                num_to_name: dict[int, str] = {}
                for p in db.roster(row["id"]):
                    num = p.get("number")
                    if isinstance(num, int) and num > 0:
                        num_to_name[num] = _surname(p.get("name", ""))
                if num_to_name:
                    self._rosters[club_name] = num_to_name
        except Exception:
            pass

    # ── config persistence ─────────────────────────────────────────────────
    def _load_config(self) -> None:
        try:
            if CONFIG.exists():
                d = json.loads(CONFIG.read_text(encoding="utf-8"))
                a, b = d.get("team_a"), d.get("team_b")
                if a and b:
                    self._names = {"team_a": a, "team_b": b}
        except Exception:
            pass

    def _save_config(self) -> None:
        try:
            CONFIG.parent.mkdir(parents=True, exist_ok=True)
            CONFIG.write_text(json.dumps(self._names, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        except Exception:
            pass

    # ── public API ──────────────────────────────────────────────────────────
    def swap(self) -> None:
        """Flip which colour cluster is which real team (user pressed the swap
        key because the auto colour-assignment mapped them backwards)."""
        self._names = {"team_a": self._names["team_b"],
                       "team_b": self._names["team_a"]}
        self._save_config()

    def team_names(self) -> dict:
        """For the HUD: {'team_a': 'France', 'team_b': 'Poland'}."""
        return dict(self._names)

    def team_label(self, team_key: str) -> str:
        if team_key in ("goalkeeper", "gk"):
            return "GK"
        return self._names.get(team_key, "?")

    def player_label(self, team_key: str, number) -> str | None:
        """team_key in {'team_a','team_b'} + jersey number -> 'Karabatic #13'.
        Returns None when there is no roster hit, so the caller can fall back."""
        if number is None:
            return None
        real = self._names.get(team_key)
        if not real:
            return None
        name = self._rosters.get(real, {}).get(int(number))
        if not name:
            return None
        return f"{name} #{int(number)}"

    @property
    def has_rosters(self) -> bool:
        return bool(self._rosters)


if __name__ == "__main__":
    mi = MatchIdentity()
    print("team names:", mi.team_names())
    print("has rosters:", mi.has_rosters)
    print("team_a #13:", mi.player_label("team_a", 13))
    print("team_b #12:", mi.player_label("team_b", 12))
