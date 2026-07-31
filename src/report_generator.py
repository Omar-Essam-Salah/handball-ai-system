"""
Match report generator — halftime and full-time PDF/HTML exports.
Updated to support MatchAnalyzer directly.
"""

from __future__ import annotations
import io
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from match_analyzer import MatchAnalyzer

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units     import cm, mm
    from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib            import colors
    from reportlab.platypus       import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
    from reportlab.lib.enums     import TA_CENTER, TA_LEFT, TA_RIGHT
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

ROOT = Path(__file__).parent.parent

_DARK_BG   = "#0b0e14"
_ACCENT    = "#00e5ff"
_DANGER    = "#ef4444"
_WARNING   = "#f59e0b"
_SUCCESS   = "#10b981"
_TEXT_MAIN = "#e0e6ed"
_TEXT_DIM  = "#8b949e"

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

if _HAS_REPORTLAB:
    from reportlab.lib.colors import HexColor
    C_BG      = HexColor(_DARK_BG)
    C_PANEL   = HexColor("#161b22")
    C_ACCENT  = HexColor(_ACCENT)
    C_DANGER  = HexColor(_DANGER)
    C_WARNING = HexColor(_WARNING)
    C_SUCCESS = HexColor(_SUCCESS)
    C_WHITE   = colors.white
    C_TEXT    = HexColor(_TEXT_MAIN)
    C_DIM     = HexColor(_TEXT_DIM)
    C_BORDER  = HexColor("#30363d")

def _rules_snapshot(rules) -> dict:
    if rules is None: return {}
    period_elapsed = time.perf_counter() - rules._period_start_ts if hasattr(rules, "_period_start_ts") and rules._period_start_ts else 0.0
    events_out = [{"event": ev.event_type.value, "team": ev.team, "ts": round(ev.timestamp, 3), "data": ev.data} for ev in getattr(rules, "events", [])]
    return {
        "period": getattr(rules, "period", 1),
        "period_elapsed_s": round(period_elapsed, 0),
        "score": dict(getattr(rules, "score", {"team_a": 0, "team_b": 0})),
        "shots": dict(getattr(rules, "shots", {"team_a": 0, "team_b": 0})),
        "saves": dict(getattr(rules, "saves", {"team_a": 0, "team_b": 0})),
        "fast_breaks": dict(getattr(rules, "fast_breaks", {"team_a": 0, "team_b": 0})),
        "shot_accuracy": dict(getattr(rules, "shot_accuracy", {"team_a": 0.0, "team_b": 0.0})),
        "possession": getattr(rules, "possession", None),
        "events": events_out,
    }

def _state_snapshot(state) -> dict:
    if state is None: return {}
    snap = state.snapshot() if hasattr(state, 'snapshot') else {}
    teams = snap.get("teams", {})
    return {
        "team_a_name": teams.get("team_a", {}).get("name", "Home"),
        "team_b_name": teams.get("team_b", {}).get("name", "Away"),
        "team_a_color": teams.get("team_a", {}).get("color_bgr", [60, 60, 220]),
        "team_b_color": teams.get("team_b", {}).get("color_bgr", [220, 60, 60]),
    }

def _analytics_snapshot(analytics) -> list[dict]:
    """
    Convert either PlayerAnalytics or MatchAnalyzer to list of player dicts.
    """
    if analytics is None:
        return []
    
    # Case 1: MatchAnalyzer
    if hasattr(analytics, 'get_all_players'):
        players = analytics.get_all_players()
        # Convert internal stats to expected performance format
        result = []
        for p in players:
            result.append({
                "player_id": p.get("player_id", 0),
                "name": p.get("name", "Unknown"),
                "team": p.get("team", "Unknown"),
                "performance": p.get("efficiency", 0),
                "goals": p.get("goals", 0),
                "shots": p.get("shots", 0),
                "saves": p.get("saves", 0),
                "assists": p.get("assists", 0),
                "efficiency": p.get("efficiency", 0),
                "shot_zones": p.get("zones", {}),
                "shot_outcomes": p.get("outcomes", {}),
            })
        return sorted(result, key=lambda x: (-x.get("performance", 0), x.get("team", "")))
    
    # Case 2: PlayerAnalytics (has all_profiles)
    if hasattr(analytics, 'all_profiles'):
        rows = []
        try:
            for prof in analytics.all_profiles():
                d = analytics.profile_dict(prof.player_id) if hasattr(analytics, 'profile_dict') else None
                if d:
                    rows.append(d)
        except Exception:
            pass
        rows.sort(key=lambda p: (-p.get("performance", 0), p.get("team", "")))
        return rows
    
    return []

def _fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"

def _bgr_to_hex(bgr: list) -> str:
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}" if bgr and len(bgr) >= 3 else "#8b949e"

def _generate_pdf(phase: str, snap_r: dict, snap_s: dict, players: list[dict], analyzer_data: dict, out_path: Path) -> Path:
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    W, H = A4
    content_w = W - 3 * cm
    ss = getSampleStyleSheet()

    def _sty(name: str, **kw) -> ParagraphStyle: return ParagraphStyle(name, parent=ss["Normal"], **kw)

    sty_h1 = _sty("H1", fontSize=22, fontName="Helvetica-Bold", textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=4)
    sty_h2 = _sty("H2", fontSize=14, fontName="Helvetica-Bold", textColor=C_ACCENT, alignment=TA_CENTER, spaceAfter=6)
    sty_h3 = _sty("H3", fontSize=11, fontName="Helvetica-Bold", textColor=C_ACCENT, spaceAfter=4)
    sty_sub = _sty("SUB", fontSize=9, fontName="Helvetica", textColor=C_DIM, alignment=TA_CENTER, spaceAfter=10)
    sty_body = _sty("BODY", fontSize=9, fontName="Helvetica", textColor=C_TEXT, spaceAfter=4)
    sty_note = _sty("NOTE", fontSize=8, fontName="Helvetica-Oblique", textColor=C_DIM, spaceAfter=4)

    na, nb = snap_s.get("team_a_name", "Home"), snap_s.get("team_b_name", "Away")
    sa, sb = snap_r.get("score", {}).get("team_a", 0), snap_r.get("score", {}).get("team_b", 0)
    hex_a, hex_b = _bgr_to_hex(snap_s.get("team_a_color", [60,60,220])), _bgr_to_hex(snap_s.get("team_b_color", [220,60,60]))
    ca, cb = HexColor(hex_a), HexColor(hex_b)
    
    elements = []
    title_data = [[Paragraph(f"<font color='{hex_a}'><b>{na}</b></font>", sty_h1), Paragraph(f"<font color='white'><b>{sa} – {sb}</b></font>", sty_h1), Paragraph(f"<font color='{hex_b}'><b>{nb}</b></font>", sty_h1)]]
    title_tbl = Table(title_data, colWidths=[content_w * 0.35, content_w * 0.30, content_w * 0.35])
    title_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), C_PANEL), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 10), ("BOTTOMPADDING", (0,0), (-1,-1), 10), ("BOX", (0,0), (-1,-1), 1, C_BORDER), ("LINEBELOW", (0,0), (-1,0), 2, C_ACCENT)]))
    elements.extend([title_tbl, Spacer(1, 4), Paragraph("MATCH REPORT", sty_h2), Spacer(1, 10)])

    # Match summary stats
    elements.append(Paragraph("Match Statistics", sty_h3))
    stat_rows = [
        ["", na, nb],
        ["Goals", str(sa), str(sb)],
        ["Shots", str(snap_r.get("shots", {}).get("team_a",0)), str(snap_r.get("shots", {}).get("team_b",0))],
        ["Eff%", f"{snap_r.get('shot_accuracy',{}).get('team_a',0):.1f}%", f"{snap_r.get('shot_accuracy',{}).get('team_b',0):.1f}%"],
        ["Saves", str(snap_r.get("saves", {}).get("team_a",0)), str(snap_r.get("saves", {}).get("team_b",0))],
        ["Fast Breaks", str(snap_r.get("fast_breaks", {}).get("team_a",0)), str(snap_r.get("fast_breaks", {}).get("team_b",0))],
    ]
    stat_tbl = Table(stat_rows, colWidths=[content_w*0.3, content_w*0.35, content_w*0.35])
    stat_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), C_PANEL), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BOX", (0,0), (-1,-1), 1, C_BORDER), ("INNERGRID", (0,0), (-1,-1), 0.5, C_BORDER)]))
    elements.append(stat_tbl)
    elements.append(Spacer(1, 10))

    # Advanced Player Shooting Stats from MatchAnalyzer
    elements.append(Paragraph("Advanced Player Shooting Stats", sty_h3))
    p_rows = [[Paragraph("<b>PLAYER</b>", sty_body), Paragraph("<b>TEAM</b>", sty_body), Paragraph("<b>GOALS</b>", sty_body), Paragraph("<b>SHOTS</b>", sty_body), Paragraph("<b>EFF %</b>", sty_body), Paragraph("<b>WING</b>", sty_body), Paragraph("<b>9M</b>", sty_body), Paragraph("<b>6M</b>", sty_body), Paragraph("<b>7M</b>", sty_body)]]
    
    for pid, stats in analyzer_data.get("players", {}).items():
        if stats["shots"] == 0: continue
        t_col = hex_a if stats["team"] == "team_a" else hex_b
        p_rows.append([
            Paragraph(f"<font color='{t_col}'><b>{stats['name']}</b></font>", sty_body),
            Paragraph(f"<font color='{t_col}'>{stats['team'].replace('team_','')}</font>", sty_body),
            str(stats["goals"]),
            str(stats["shots"]),
            f"{stats['efficiency']}%",
            str(stats["zones"].get("wing_left", 0) + stats["zones"].get("wing_right", 0)),
            str(stats["zones"].get("center_9m", 0)),
            str(stats["zones"].get("center_6m", 0)),
            str(stats["zones"].get("penalty_7m", 0)),
        ])
        
    if len(p_rows) > 1:
        p_tbl = Table(p_rows, colWidths=[content_w * 0.18, content_w * 0.08, content_w * 0.08, content_w * 0.08, content_w * 0.1, content_w * 0.09, content_w * 0.09, content_w * 0.09, content_w * 0.09])
        p_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), C_PANEL), ("TEXTCOLOR", (0,0), (-1,-1), C_TEXT), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("FONTSIZE", (0,0), (-1,-1), 8), ("BOX", (0,0), (-1,-1), 1, C_BORDER), ("INNERGRID", (0,0), (-1,-1), 0.5, C_BORDER)]))
        elements.append(p_tbl)
    else:
        elements.append(Paragraph("No shot data available.", sty_body))
    elements.append(Spacer(1, 10))

    # Player detailed cards (per player)
    elements.append(Paragraph("Individual Player Details", sty_h3))
    for pid, stats in analyzer_data.get("players", {}).items():
        if stats["shots"] == 0 and stats["saves"] == 0:
            continue
        t_col = hex_a if stats["team"] == "team_a" else hex_b
        card_data = [
            [Paragraph(f"<b>Player:</b> {stats['name']}", sty_body), Paragraph(f"<b>Team:</b> {stats['team'].replace('team_','')}", sty_body)],
            ["Goals", str(stats["goals"]), "Shots", str(stats["shots"]), "Efficiency", f"{stats['efficiency']}%"],
            ["Zones:", "", "", "", "", ""],
            ["Wing Left", str(stats["zones"].get("wing_left",0)), "Wing Right", str(stats["zones"].get("wing_right",0)), "Center 6m", str(stats["zones"].get("center_6m",0))],
            ["Center 9m", str(stats["zones"].get("center_9m",0)), "Penalty 7m", str(stats["zones"].get("penalty_7m",0)), "", ""],
        ]
        card_tbl = Table(card_data, colWidths=[content_w*0.15, content_w*0.18, content_w*0.15, content_w*0.18, content_w*0.15, content_w*0.19])
        card_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), C_PANEL),
            ("SPAN", (0,2), (5,2)),  # span "Zones:" row
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOX", (0,0), (-1,-1), 1, C_BORDER),
            ("INNERGRID", (0,0), (-1,-1), 0.5, C_BORDER),
        ]))
        elements.append(card_tbl)
        elements.append(Spacer(1, 6))
        elements.append(PageBreak())

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(elements, onFirstPage=_on_page, onLaterPages=_on_page)
    return out_path

def generate_report(phase: str = "fulltime", state: Any = None, rules: Any = None, analytics: Any = None, out_dir: Path | str | None = None, filename: str = "") -> Path:
    snap_r  = _rules_snapshot(rules)
    snap_s  = _state_snapshot(state)
    players = _analytics_snapshot(analytics)
    
    # If analytics is a MatchAnalyzer, get its summary
    analyzer_data = {}
    if isinstance(analytics, MatchAnalyzer):
        analyzer_data = analytics.get_summary()
    elif analytics and hasattr(analytics, 'get_summary'):
        analyzer_data = analytics.get_summary()
    else:
        # Fallback: create temporary MatchAnalyzer and process events
        temp_analyzer = MatchAnalyzer()
        temp_analyzer.process_events(snap_r.get("events", []))
        analyzer_data = temp_analyzer.get_summary()

    out_dir = Path(out_dir) if out_dir else ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if _HAS_REPORTLAB else "html"
    out_path = out_dir / (filename or f"report_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}")

    if _HAS_REPORTLAB:
        _generate_pdf(phase, snap_r, snap_s, players, analyzer_data, out_path)
    else:
        out_path.write_text("HTML generation not implemented.", encoding="utf-8")
    
    return out_path