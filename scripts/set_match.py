"""
Pick the matchup. Tells the whole system which two teams are playing so it shows
their real names + badge logos and loads both full rosters.

    python scripts/set_match.py --list                 # show available teams
    python scripts/set_match.py "France" "Denmark"     # home, away

Writes config/match.json  ->  {"team_a": <home>, "team_b": <away>}
Generates config/logos/<CODE>.png badges for both teams (auto from team colour).
MatchIdentity reads config/match.json on the next run; press K in the viewer to
flip home/away if the colour clustering mapped them backwards.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from teams_data import TEAMS, get_team  # noqa: E402

CONFIG = ROOT / "config" / "match.json"
LOGO_DIR = ROOT / "config" / "logos"


def make_badge(code: str, color_bgr, out: Path) -> None:
    """Draw a simple round national badge (coloured disc + white code)."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return
    S = 128
    img = np.zeros((S, S, 4), dtype=np.uint8)          # BGRA, transparent
    c = (S // 2, S // 2)
    b, g, r = [int(x) for x in color_bgr]
    cv2.circle(img, c, S // 2 - 4, (b, g, r, 255), -1, cv2.LINE_AA)
    cv2.circle(img, c, S // 2 - 4, (255, 255, 255, 255), 4, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(code, font, 1.1, 2)
    cv2.putText(img, code, (c[0] - tw // 2, c[1] + th // 2),
                font, 1.1, (255, 255, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


def main(argv):
    if not argv or argv[0] in ("--list", "-l"):
        print("Available teams:")
        for name, t in TEAMS.items():
            print(f"  {t['code']}  {name:10s} ({len(t['players'])} players)")
        print('\nUsage: python scripts/set_match.py "France" "Denmark"')
        return 0
    if len(argv) < 2:
        print('Need two teams. e.g.  python scripts/set_match.py "France" "Denmark"')
        return 1

    home, away = get_team(argv[0]), get_team(argv[1])
    for raw, t in ((argv[0], home), (argv[1], away)):
        if t is None:
            print(f"[!] Unknown team: {raw!r}. Run with --list to see options.")
            return 1

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(
        {"team_a": home["name"], "team_b": away["name"],
         "team_a_code": home["code"], "team_b_code": away["code"],
         "team_a_logo": f"config/logos/{home['code']}.png",
         "team_b_logo": f"config/logos/{away['code']}.png"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    make_badge(home["code"], home["color"], LOGO_DIR / f"{home['code']}.png")
    make_badge(away["code"], away["color"], LOGO_DIR / f"{away['code']}.png")

    print(f"Match set:  {home['name']} (home/team_a)  vs  {away['name']} (away/team_b)")
    print(f"  rosters:  {len(home['players'])} vs {len(away['players'])} players")
    print(f"  logos  :  {LOGO_DIR / (home['code']+'.png')} , "
          f"{LOGO_DIR / (away['code']+'.png')}")
    print("  -> config/match.json written. Restart the viewer to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
