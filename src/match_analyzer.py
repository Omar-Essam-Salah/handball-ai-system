"""
Match Analyzer - Advanced Statistical Engine for Handball
Aggregates raw match events into per-player and per-team analytics.
Outputs structured data ideal for RAG/LLM ingestion and PDF reports.
"""

from collections import defaultdict
import math

class MatchAnalyzer:
    def __init__(self, court_width=40.0, court_height=20.0):
        self.cw = court_width
        self.ch = court_height
        
        # Per-player statistics schema
        self.player_stats = defaultdict(lambda: {
            "name": "Unknown",
            "team": "Unknown",
            "goals": 0,
            "shots": 0,
            "saves": 0,
            "assists": 0,
            "efficiency": 0.0,
            "zones": {"wing_left": 0, "wing_right": 0, "center_6m": 0, "center_9m": 0, "penalty_7m": 0},
            "outcomes": {"goal": 0, "saved": 0, "missed": 0, "blocked": 0, "post": 0}
        })
        
        # Team statistics schema
        self.team_stats = defaultdict(lambda: {
            "goals": 0,
            "shots": 0,
            "efficiency": 0.0,
            "fast_breaks": 0,
            "saves": 0,
            "turnovers": 0
        })

    def process_events(self, events, analytics_profiles=None):
        """
        Process a list of events (dicts with keys: event, team, data, etc.)
        analytics_profiles: optional object with all_profiles() method (e.g., PlayerAnalytics)
        """
        # Pre-fill player names if analytics profiles are provided
        if analytics_profiles and hasattr(analytics_profiles, 'all_profiles'):
            try:
                for prof in analytics_profiles.all_profiles():
                    pid = getattr(prof, 'player_id', None)
                    if pid is not None:
                        self.player_stats[pid]["name"] = getattr(prof, 'name', 'Unknown')
                        self.player_stats[pid]["team"] = getattr(prof, 'team', 'Unknown')
            except Exception:
                pass

        for ev in events:
            # Handle both dict and MatchEvent objects
            if hasattr(ev, 'to_dict'):
                ev = ev.to_dict()
            
            team = ev.get("team")
            ev_type = ev.get("event") or ev.get("type")
            data = ev.get("data", {})
            
            if not team or not ev_type:
                continue
            
            if ev_type == "shot":
                self.team_stats[team]["shots"] += 1
                shooter = data.get("shooter_id")
                
                if shooter is not None:
                    self.player_stats[shooter]["shots"] += 1
                    self.player_stats[shooter]["team"] = team
                    
                    x, y = data.get("x"), data.get("y")
                    if x is not None and y is not None:
                        zone = self._determine_shot_zone(x, y, team)
                        self.player_stats[shooter]["zones"][zone] += 1
                        
            elif ev_type == "goal":
                self.team_stats[team]["goals"] += 1
                shooter = data.get("scored_by_player_id") or data.get("shooter_id")
                
                if shooter is not None:
                    self.player_stats[shooter]["goals"] += 1
                    self.player_stats[shooter]["outcomes"]["goal"] += 1
                    
            elif ev_type == "save":
                self.team_stats[team]["saves"] += 1
                gk_id = data.get("gk_id")
                if gk_id is not None:
                    self.player_stats[gk_id]["saves"] += 1
                    
            elif ev_type == "fast_break":
                self.team_stats[team]["fast_breaks"] += 1
                
            elif ev_type == "pass":
                # تمريرة ناجحة - لا تؤثر على التسديدات
                pass
                
            elif ev_type == "possession_change":
                # خسارة كرة أو استحواذ جديد
                if data.get("from_team") and data.get("from_team") != "none":
                    self.team_stats[team]["turnovers"] += 1

        # Calculate efficiencies
        for pid, stats in self.player_stats.items():
            if stats["shots"] > 0:
                stats["efficiency"] = round((stats["goals"] / stats["shots"]) * 100, 1)
                
        for t, stats in self.team_stats.items():
            if stats["shots"] > 0:
                stats["efficiency"] = round((stats["goals"] / stats["shots"]) * 100, 1)

    def _determine_shot_zone(self, x, y, team):
        """Classify shot zone based on coordinates (meters)."""
        # Normalize for attacking direction (assuming team_a attacks right)
        norm_x = x if team == "team_a" else self.cw - x
        
        # Wing shots (extreme Y values, near goal line)
        if norm_x > 32:
            if y < 4.5:
                return "wing_right"
            if y > 15.5:
                return "wing_left"
            
        # Penalty 7m line
        if 32.5 <= norm_x <= 33.5 and 9.0 <= y <= 11.0:
            return "penalty_7m"
            
        # Center 6m (inside 6m line, near the center)
        if norm_x >= 31 and 5.0 <= y <= 15.0:
            return "center_6m"
            
        # Default to 9m backcourt
        return "center_9m"

    def get_summary(self):
        """Return compiled dictionary for report generator."""
        return {
            "teams": dict(self.team_stats),
            "players": dict(self.player_stats)
        }
    
    def get_all_players(self):
        """Return list of player dicts (compatible with report_generator)."""
        return [{"player_id": pid, **stats} for pid, stats in self.player_stats.items()]
    
    def reset(self):
        """Clear all accumulated stats."""
        self.player_stats.clear()
        self.team_stats.clear()