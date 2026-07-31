"""
Egyptian Handball League — local relational store (offline, zero-config SQLite).

Schema (3NF, fully relational):

    clubs(id, name, name_ar, city, logo_path)
        └─< players(id, club_id→clubs, name, name_ar, number, position,
                    height_cm, dominant_hand, photo_path, reid_fingerprint BLOB)
    matches(id, home_club_id→clubs, away_club_id→clubs, date, competition,
            home_score, away_score, video_path)
        └─< player_match_stats(id, player_id→players, match_id→matches,
                    minutes, goals, shots, assists, turnovers, saves, ...)
    shots(id, player_id→players, match_id→matches, gk_id→players,
          zone, cell, outcome, ts)          -- feeds the goal-mouth Shot/Save report

Why SQLite: the system is fully OFFLINE on one machine → SQLite gives ACID,
joins, indexes and a single-file DB with no server. Swap the connection string
for PostgreSQL (psycopg2 / SQLAlchemy) only if you later need multi-analyst
concurrent writes; the schema below is unchanged.

The `reid_fingerprint` BLOB is the bridge to Module 4: the visual Re-ID embedding
(OSNet/HSV) is stored per player, so a face/jersey match on court resolves to a
real DB profile + that player's historical analytics.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "league.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS clubs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    name_ar     TEXT,
    city        TEXT,
    logo_path   TEXT
);
CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    club_id         INTEGER REFERENCES clubs(id) ON DELETE SET NULL,
    name            TEXT NOT NULL,
    name_ar         TEXT,
    number          INTEGER,
    position        TEXT,                 -- LW/RW/LB/RB/CB/Pivot/GK
    height_cm       INTEGER,
    dominant_hand   TEXT,                 -- L / R
    photo_path      TEXT,
    reid_fingerprint BLOB,                -- float32 embedding (Module 4 link)
    UNIQUE(club_id, number)
);
CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    home_club_id  INTEGER REFERENCES clubs(id),
    away_club_id  INTEGER REFERENCES clubs(id),
    date          TEXT,
    competition   TEXT,
    home_score    INTEGER DEFAULT 0,
    away_score    INTEGER DEFAULT 0,
    video_path    TEXT
);
CREATE TABLE IF NOT EXISTS player_match_stats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER REFERENCES players(id) ON DELETE CASCADE,
    match_id    INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    minutes     REAL DEFAULT 0,
    goals       INTEGER DEFAULT 0,
    shots       INTEGER DEFAULT 0,
    assists     INTEGER DEFAULT 0,
    turnovers   INTEGER DEFAULT 0,
    saves       INTEGER DEFAULT 0,
    UNIQUE(player_id, match_id)
);
CREATE TABLE IF NOT EXISTS shots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER REFERENCES players(id),
    match_id    INTEGER REFERENCES matches(id),
    gk_id       INTEGER REFERENCES players(id),
    zone        TEXT,                     -- one of the 12 court zones
    cell        TEXT,                     -- LU CU RU LH CH RH LD CD RD / MISS POST BLOCK
    outcome     TEXT,                     -- GOAL SAVE MISS POST BLOCK
    ts          REAL
);
CREATE INDEX IF NOT EXISTS ix_pms_match  ON player_match_stats(match_id);
CREATE INDEX IF NOT EXISTS ix_shots_match ON shots(match_id);
CREATE INDEX IF NOT EXISTS ix_players_num ON players(number);
"""


class LeagueDB:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False → safe to share between the pipeline thread and
        # the server thread (writes are serialised by SQLite's own lock anyway).
        self.con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self.con.row_factory = sqlite3.Row
        # ── performance + crash-safety pragmas (this is the whole trick) ──
        # WAL: writers don't block readers, ~10x faster small writes, and the DB
        #      cannot be corrupted by a crash mid-write (the journal replays).
        # synchronous=NORMAL: with WAL this is fully durable across app crashes
        #      and still fast (only fsyncs at checkpoints, not every write).
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON",
                       "temp_store=MEMORY", "cache_size=-4000", "busy_timeout=4000"):
            self.con.execute(f"PRAGMA {pragma}")
        self._lock = __import__("threading").Lock()
        self.con.executescript(SCHEMA)
        self.con.commit()

    # ── writes ──────────────────────────────────────────────────────────
    def add_club(self, name, name_ar=None, city=None, logo_path=None) -> int:
        cur = self.con.execute(
            "INSERT OR IGNORE INTO clubs(name,name_ar,city,logo_path) VALUES(?,?,?,?)",
            (name, name_ar, city, logo_path))
        self.con.commit()
        return cur.lastrowid or self.con.execute(
            "SELECT id FROM clubs WHERE name=?", (name,)).fetchone()["id"]

    def add_player(self, club_id, name, number=None, position=None, **kw) -> int:
        cur = self.con.execute(
            "INSERT INTO players(club_id,name,name_ar,number,position,height_cm,"
            "dominant_hand,photo_path) VALUES(?,?,?,?,?,?,?,?)",
            (club_id, name, kw.get("name_ar"), number, position, kw.get("height_cm"),
             kw.get("dominant_hand"), kw.get("photo_path")))
        self.con.commit()
        return cur.lastrowid

    def add_match(self, home_club_id, away_club_id, date=None, competition=None,
                  video_path=None) -> int:
        cur = self.con.execute(
            "INSERT INTO matches(home_club_id,away_club_id,date,competition,video_path)"
            " VALUES(?,?,?,?,?)", (home_club_id, away_club_id, date, competition, video_path))
        self.con.commit()
        return cur.lastrowid

    def record_shot(self, match_id, player_id, *, gk_id=None, zone="Center Back",
                    cell="CH", outcome="GOAL", ts=0.0) -> int:
        cur = self.con.execute(
            "INSERT INTO shots(player_id,match_id,gk_id,zone,cell,outcome,ts)"
            " VALUES(?,?,?,?,?,?,?)", (player_id, match_id, gk_id, zone, cell, outcome, ts))
        # roll-up into player_match_stats
        self.con.execute(
            "INSERT INTO player_match_stats(player_id,match_id,shots,goals) VALUES(?,?,1,?)"
            " ON CONFLICT(player_id,match_id) DO UPDATE SET shots=shots+1, "
            " goals=goals+?", (player_id, match_id, int(outcome == "GOAL"),
                               int(outcome == "GOAL")))
        if gk_id and outcome == "SAVE":
            self.con.execute(
                "INSERT INTO player_match_stats(player_id,match_id,saves) VALUES(?,?,1)"
                " ON CONFLICT(player_id,match_id) DO UPDATE SET saves=saves+1",
                (gk_id, match_id))
        self.con.commit()
        return cur.lastrowid

    # ── Re-ID linkage (Module 4 ⇄ DB) ───────────────────────────────────
    def link_fingerprint(self, player_id: int, emb: np.ndarray) -> None:
        self.con.execute("UPDATE players SET reid_fingerprint=? WHERE id=?",
                         (emb.astype(np.float32).tobytes(), player_id))
        self.con.commit()

    def match_fingerprint(self, emb: np.ndarray, club_id: Optional[int] = None,
                          thresh: float = 0.6) -> Optional[dict]:
        """Cosine-match an on-court embedding to a stored player profile."""
        q = emb.astype(np.float32); q /= (np.linalg.norm(q) + 1e-6)
        sql = "SELECT * FROM players WHERE reid_fingerprint IS NOT NULL"
        rows = self.con.execute(sql + (" AND club_id=?" if club_id else ""),
                                ((club_id,) if club_id else ())).fetchall()
        best, best_s = None, thresh
        for r in rows:
            g = np.frombuffer(r["reid_fingerprint"], dtype=np.float32)
            if g.shape != q.shape:
                continue
            s = float(q @ (g / (np.linalg.norm(g) + 1e-6)))
            if s > best_s:
                best_s, best = s, dict(r)
        return best

    def find_by_number(self, number: int, club_id: Optional[int] = None) -> Optional[dict]:
        sql = "SELECT * FROM players WHERE number=?"
        args = [number]
        if club_id:
            sql += " AND club_id=?"; args.append(club_id)
        r = self.con.execute(sql, args).fetchone()
        return dict(r) if r else None

    # ── reads ───────────────────────────────────────────────────────────
    def player_profile(self, player_id: int) -> dict:
        p = self.con.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        if not p:
            return {}
        career = self.con.execute(
            "SELECT SUM(goals) g, SUM(shots) s, SUM(assists) a, COUNT(*) matches "
            "FROM player_match_stats WHERE player_id=?", (player_id,)).fetchone()
        return {**dict(p), "career": dict(career)}

    def roster(self, club_id: int) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM players WHERE club_id=? ORDER BY number", (club_id,))]

    def close(self):
        self.con.close()


if __name__ == "__main__":
    db = LeagueDB(ROOT / "data" / "league_demo.db")
    ah = db.add_club("Al Ahly", "الأهلي", "Cairo")
    zm = db.add_club("Zamalek", "الزمالك", "Giza")
    p1 = db.add_player(ah, "Yahia Khaled", number=30, position="LB", dominant_hand="R")
    gk = db.add_player(zm, "Karim Hendawy", number=16, position="GK")
    m = db.add_match(ah, zm, "2026-06-05", "Egyptian League")
    db.record_shot(m, p1, gk_id=gk, zone="Left Back", cell="LD", outcome="GOAL")
    db.record_shot(m, p1, gk_id=gk, zone="Left Back", cell="CH", outcome="SAVE")
    db.link_fingerprint(p1, np.random.rand(512).astype(np.float32))
    print("profile:", db.player_profile(p1))
    print("roster Ahly:", [r["name"] for r in db.roster(ah)])
