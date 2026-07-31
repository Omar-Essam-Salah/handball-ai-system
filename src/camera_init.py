"""
Hikvision ISAPI startup configurator for DS-2CD2686G2-IZS.

Run once at pipeline start (or as a standalone script) to force:
  • Shutter speed → 1/500 s   (freezes fast handball motion blur)
  • Focus         → MANUAL    (locks current focus for a fixed-mount camera)

Usage
─────
    # Programmatic (from pipeline.py):
    from src.camera_init import configure_camera
    configure_camera()          # reads config/camera.yaml automatically

    # Standalone:
    python src/camera_init.py

Notes
─────
• ExposureTime is sent in microseconds.  1/500 s = 2 000 µs.
  If your firmware version stores it in 100 ns units (some older builds),
  change SHUTTER_US to 20_000.  You can verify by GETting the endpoint
  before and after and checking the echoed value.
• The focus endpoint sets mode=MANUAL which freezes the current lens
  position.  If the court is not in focus yet, call set_focus_and_lock()
  which issues a one-shot autofocus first, waits, then locks.
• Both operations are idempotent — safe to call on every restart.
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from requests.auth import HTTPDigestAuth
import yaml

# ── Config ────────────────────────────────────────────────────────────────────

CAMERA_YAML = Path(__file__).parent.parent / "config" / "camera.yaml"

# 1/500 s expressed in microseconds (DS-2CD2 G-series firmware unit)
SHUTTER_US: int = 2_000

_XML_CONTENT_TYPE = {"Content-Type": "application/xml; charset=UTF-8"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg(path: Path = CAMERA_YAML) -> dict:
    return yaml.safe_load(path.read_text())


def _session(cfg: dict) -> requests.Session:
    s = requests.Session()
    s.auth    = HTTPDigestAuth(cfg["user"], cfg["password"])
    s.timeout = (5, 10)   # (connect_timeout, read_timeout) in seconds
    return s


def _base_url(cfg: dict) -> str:
    return f"http://{cfg['ip']}/ISAPI"


def _patch_tag(xml: str, tag: str, value: str) -> str:
    """
    Replace the text content of the first matching tag.
    Handles optional namespace prefixes (e.g. <hk:ExposureTime>).

    Raises ValueError if the tag is not found in the document.
    """
    pat  = rf'(<(?:\w+:)?{re.escape(tag)}\s*>)[^<]*(</(?:\w+:)?{re.escape(tag)}\s*>)'
    repl = rf'\g<1>{re.escape(value)}\g<2>'
    # re.escape on the replacement value is wrong — we want literal substitution
    new, n = re.subn(pat, rf'\g<1>{value}\g<2>', xml,
                     count=1, flags=re.IGNORECASE)
    if n == 0:
        raise ValueError(
            f"Tag <{tag}> not found in XML — camera firmware may use a "
            f"different tag name.  Dump the GET response to inspect it."
        )
    return new


def _put_xml(session: requests.Session, url: str, xml: str) -> requests.Response:
    r = session.put(url, data=xml.encode("utf-8"), headers=_XML_CONTENT_TYPE)
    r.raise_for_status()
    return r


# ── Shutter speed ─────────────────────────────────────────────────────────────

def set_shutter(
    session:    requests.Session,
    base:       str,
    channel:    int,
    shutter_us: int = SHUTTER_US,
) -> None:
    """
    Force the imaging channel to manual exposure at `shutter_us` microseconds.

    Steps:
      1. GET current imagingSettings (preserves all other camera settings).
      2. Patch exposureMode → manual and ExposureTime → shutter_us.
      3. PUT the modified document back.
    """
    url = f"{base}/Image/channels/{channel}/imagingSettings"

    r = session.get(url)
    r.raise_for_status()

    xml = r.text
    xml = _patch_tag(xml, "exposureMode", "manual")
    xml = _patch_tag(xml, "ExposureTime",  str(shutter_us))

    _put_xml(session, url, xml)
    fps_equiv = 1_000_000 // shutter_us
    print(f"[CameraInit] Shutter → 1/{fps_equiv} s  ({shutter_us} µs)  ch={channel}")


# ── Focus lock ────────────────────────────────────────────────────────────────

def lock_focus(
    session: requests.Session,
    base:    str,
    channel: int,
) -> None:
    """
    Freeze the current lens focus position by switching to MANUAL mode.
    Use this when the court is already in focus (fixed far-mount installation).
    """
    url = f"{base}/System/Video/inputs/channels/{channel}/focus"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<FocusData>"
        "<focusMode>MANUAL</focusMode>"
        "</FocusData>"
    )
    _put_xml(session, url, xml)
    print(f"[CameraInit] Focus  → MANUAL (locked)  ch={channel}")


def set_focus_and_lock(
    session:      requests.Session,
    base:         str,
    channel:      int,
    settle_s:     float = 3.0,
) -> None:
    """
    Trigger a one-shot continuous-autofocus pass, wait for it to settle,
    then lock.  Use this on first install when focus hasn't been set yet.
    """
    url = f"{base}/System/Video/inputs/channels/{channel}/focus"

    # Trigger continuous AF
    xml_af = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<FocusData>"
        "<focusMode>AUTO</focusMode>"
        "</FocusData>"
    )
    _put_xml(session, url, xml_af)
    print(f"[CameraInit] Focus  → AUTO (seeking…)  ch={channel}")

    time.sleep(settle_s)

    # Lock at current position
    lock_focus(session, base, channel)


# ── Public entry point ────────────────────────────────────────────────────────

def configure_camera(
    cfg:        Optional[dict] = None,
    shutter_us: int  = SHUTTER_US,
    lock_only:  bool = True,
) -> bool:
    """
    Apply shutter + focus settings to the camera described in camera.yaml.

    Parameters
    ──────────
    cfg         : pre-loaded config dict.  If None, camera.yaml is read.
    shutter_us  : exposure time in microseconds (default 2000 = 1/500 s).
    lock_only   : True  → just lock focus at current position (normal restart).
                  False → run one-shot AF then lock (first-time install).

    Returns True if ALL operations succeeded, False if any failed.
    """
    if cfg is None:
        cfg = _load_cfg()

    session = _session(cfg)
    base    = _base_url(cfg)
    channel = cfg.get("isapi_channel", 1)

    ok = True

    try:
        set_shutter(session, base, channel, shutter_us)
    except requests.exceptions.ConnectionError:
        print(f"[CameraInit] ERROR  Cannot reach {cfg['ip']} — "
              "check network / camera power.", file=sys.stderr)
        return False
    except requests.exceptions.HTTPError as e:
        print(f"[CameraInit] Shutter set FAILED — HTTP {e.response.status_code}: "
              f"{e.response.text[:200]}", file=sys.stderr)
        ok = False
    except ValueError as e:
        print(f"[CameraInit] Shutter set FAILED — XML parse: {e}", file=sys.stderr)
        ok = False

    try:
        if lock_only:
            lock_focus(session, base, channel)
        else:
            set_focus_and_lock(session, base, channel)
    except requests.exceptions.HTTPError as e:
        print(f"[CameraInit] Focus lock FAILED — HTTP {e.response.status_code}: "
              f"{e.response.text[:200]}", file=sys.stderr)
        ok = False
    except Exception as e:
        print(f"[CameraInit] Focus lock FAILED — {e}", file=sys.stderr)
        ok = False

    return ok


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Configure Hikvision camera via ISAPI on pipeline startup."
    )
    parser.add_argument(
        "--shutter-us", type=int, default=SHUTTER_US,
        help=f"Exposure time in µs (default {SHUTTER_US} = 1/500 s).",
    )
    parser.add_argument(
        "--first-install", action="store_true",
        help="Run one-shot autofocus before locking (use on first mount).",
    )
    parser.add_argument(
        "--config", type=Path, default=CAMERA_YAML,
        help="Path to camera.yaml.",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    success = configure_camera(
        cfg        = cfg,
        shutter_us = args.shutter_us,
        lock_only  = not args.first_install,
    )
    sys.exit(0 if success else 1)
