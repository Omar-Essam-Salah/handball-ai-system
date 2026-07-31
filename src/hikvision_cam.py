"""
Hikvision Camera Control — ISAPI over HTTP with Digest authentication.

Protocol choice
───────────────
Hikvision cameras expose two control APIs:
  • ISAPI  — Hikvision-proprietary HTTP+XML REST API.  No extra Python
             dependencies (uses `requests`).  Most feature-complete for
             Hikvision hardware.  ← PRIMARY
  • ONVIF  — industry standard, needs onvif-zeep library.  Provided as an
             optional thin wrapper below.

All ISAPI endpoints are documented in:
  "Hikvision ISAPI Reference v2.0 — PTZ, Imaging, System"

Thread safety
─────────────
A single `requests.Session` is shared; all HTTP calls are guarded by
`self._lock` so the pipeline thread and API thread can both issue PTZ
commands without colliding.

Usage
─────
    cam = HikvisionCamera(ip="172.16.5.174", user="admin", password="...")

    cam.ping()                          # → True / False
    cam.device_info()                   # → {"model": ..., "firmware": ...}

    cam.ptz_continuous(pan=0.5, tilt=0) # move right at half speed
    time.sleep(1)
    cam.ptz_stop()

    cam.goto_preset(1)                  # go to saved preset 1
    cam.save_preset(2, name="Court Wide")

    cam.focus_auto()
    cam.focus_manual("near", speed=0.4)

    cam.iris_auto()
    cam.get_status()   # → {"pan": "45.0", "tilt": "-12.0", "zoom": "10"}
"""

import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.auth import HTTPDigestAuth


# ISAPI XML namespace used by most Hikvision firmware versions
_NS = "http://www.hikvision.com/ver20/XMLSchema"
_NS_MAP = {"h": _NS}


def _find(root: ET.Element, tag: str) -> Optional[str]:
    """Try namespaced then bare tag; return text or None."""
    el = root.find(f"h:{tag}", _NS_MAP) or root.find(tag)
    return el.text if el is not None else None


def _xml_root(tag: str, **fields) -> str:
    """Build minimal ISAPI XML payload.
    Some Hikvision firmware variants reject payloads that lack the XML
    declaration with HTTP 400, so we always emit it."""
    inner = "".join(f"<{k}>{v}</{k}>" for k, v in fields.items())
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f'<{tag} version="2.0" xmlns="{_NS}">{inner}</{tag}>')


@dataclass
class PTZStatus:
    pan:  float = 0.0   # degrees, 0-360
    tilt: float = 0.0   # degrees, -90 to 90
    zoom: float = 1.0   # ×, 1-30 typically
    error: str  = ""


@dataclass
class PTZPreset:
    id:   int
    name: str


class HikvisionCamera:
    """
    Full PTZ + focus + imaging control via Hikvision ISAPI.

    All move methods return True on success, False on camera error or
    network failure.  Check `.last_error` for details when False.
    """

    def __init__(
        self,
        ip:       str,
        user:     str,
        password: str,
        channel:  int   = 1,
        timeout:  float = 4.0,
    ):
        self._ip      = ip
        self._channel = channel
        self._timeout = timeout
        self._base    = f"http://{ip}/ISAPI"
        self._lock    = threading.Lock()
        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(user, password)
        self.last_error: str = ""

    # ── Connectivity ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the camera answers with HTTP 200/401."""
        try:
            with self._lock:
                r = self._session.get(
                    f"{self._base}/System/deviceInfo",
                    timeout=self._timeout)
            return r.status_code in (200, 401)
        except Exception as e:
            self.last_error = str(e)
            return False

    def device_info(self) -> dict:
        """Return model, firmware version, serial number, MAC address."""
        ok, text = self._get("/System/deviceInfo")
        if not ok:
            return {"error": self.last_error}
        try:
            root = ET.fromstring(text)
            return {
                "model":    _find(root, "model") or "",
                "firmware": _find(root, "firmwareVersion") or "",
                "serial":   _find(root, "serialNumber") or "",
                "mac":      _find(root, "macAddress") or "",
                "ip":       self._ip,
            }
        except Exception as e:
            return {"error": str(e), "raw": text[:300]}

    # ── PTZ movement ─────────────────────────────────────────────────────────

    def ptz_continuous(
        self,
        pan:   float = 0.0,
        tilt:  float = 0.0,
        zoom:  float = 0.0,
        speed: float = 1.0,
    ) -> bool:
        """
        Start continuous PTZ movement.

        pan / tilt / zoom:  −1.0 … +1.0
          pan  > 0 = right,   pan  < 0 = left
          tilt > 0 = up,      tilt < 0 = down
          zoom > 0 = tele-in, zoom < 0 = wide-out
        speed: 0.0 … 1.0 — multiplied into every axis value.

        Call ptz_stop() (or ptz_continuous(0,0,0)) to halt.
        """
        def scale(v: float) -> int:
            return max(-100, min(100, int(v * speed * 100)))

        # Some Hikvision firmware accepts <PTZData><pan/><tilt/><zoom/></PTZData>
        # but the strictly-validating ones expect zoom in a separate call.
        # We try the combined form first, and fall back to pan+tilt only +
        # an explicit zoom call on HTTP 400.
        xml = _xml_root("PTZData",
                        pan=scale(pan),
                        tilt=scale(tilt),
                        zoom=scale(zoom))
        ok, _ = self._put(f"/PTZCtrl/channels/{self._channel}/continuous", xml)
        if ok:
            return True

        # Fallback path — split pan/tilt and zoom.
        if "400" in (self.last_error or ""):
            pt_xml = _xml_root("PTZData", pan=scale(pan), tilt=scale(tilt))
            ok_pt, _ = self._put(
                f"/PTZCtrl/channels/{self._channel}/continuous", pt_xml)
            ok_z = self.zoom_continuous(zoom, speed)
            return ok_pt and ok_z
        return False

    def zoom_continuous(self, zoom: float, speed: float = 1.0) -> bool:
        """
        Continuous zoom command using the dedicated zoomChannels endpoint
        (some firmware rejects zoom inside <PTZData>).
        zoom: −1.0 (wide) … +1.0 (tele).  Call zoom_continuous(0) to stop.
        """
        z = max(-100, min(100, int(zoom * speed * 100)))
        xml = _xml_root("ZoomData", zoom=z)
        # Try the documented zoomChannels endpoint first, then fall back to PTZData.
        ok, _ = self._put(
            f"/Image/channels/{self._channel}/zoomChannels", xml)
        if ok:
            return True
        # Fallback: PTZData/continuous with only the zoom element
        xml2 = _xml_root("PTZData", pan=0, tilt=0, zoom=z)
        ok2, _ = self._put(
            f"/PTZCtrl/channels/{self._channel}/continuous", xml2)
        return ok2

    def ptz_stop(self) -> bool:
        """Halt all PTZ movement."""
        # Send pan+tilt+zoom=0 — the most universally accepted "stop" payload.
        xml = _xml_root("PTZData", pan=0, tilt=0, zoom=0)
        ok, _ = self._put(f"/PTZCtrl/channels/{self._channel}/continuous", xml)
        return ok

    def ptz_momentary(
        self,
        pan:         float = 0.0,
        tilt:        float = 0.0,
        zoom:        float = 0.0,
        focus:       float = 0.0,
        duration_ms: int   = 500,
    ) -> bool:
        """
        Move for a fixed duration then auto-stop.

        focus: −1.0 (far) … +1.0 (near)
        duration_ms: how long to move (milliseconds)
        """
        def scale(v: float) -> int:
            return max(-100, min(100, int(v * 100)))

        xml = _xml_root("PTZData",
                        pan=scale(pan),
                        tilt=scale(tilt),
                        zoom=scale(zoom),
                        focus=scale(focus),
                        iris=0,
                        duration=duration_ms)
        ok, _ = self._put(f"/PTZCtrl/channels/{self._channel}/momentary", xml)
        return ok

    def ptz_absolute(
        self,
        pan:   float = 0.0,
        tilt:  float = 0.0,
        zoom:  float = 0.0,
        speed: float = 0.7,
    ) -> bool:
        """
        Move to absolute position.
          pan:  0.0 – 360.0  degrees
          tilt: −90.0 – 90.0 degrees (positive = up)
          zoom: 0.0 – 100.0  percent of optical range
          speed: 0.0 – 1.0
        """
        def _scale_speed(v: float) -> int:
            return max(1, min(100, int(v * 100)))

        xml = (
            f'<PTZData version="2.0" xmlns="{_NS}">'
            f'<AbsoluteHigh>'
            f'<elevation>{int(tilt * 10)}</elevation>'
            f'<azimuth>{int(pan  * 10)}</azimuth>'
            f'<absoluteZoom>{max(0, min(1000, int(zoom * 10)))}</absoluteZoom>'
            f'</AbsoluteHigh>'
            f'</PTZData>'
        )
        ok, _ = self._put(
            f"/PTZCtrl/channels/{self._channel}/absolute", xml)
        return ok

    def ptz_relative(
        self,
        pan:   float = 0.0,
        tilt:  float = 0.0,
        zoom:  float = 0.0,
        speed: float = 0.5,
    ) -> bool:
        """
        Relative move by given amount (normalised −1.0 … +1.0).
        Camera moves from current position by delta × max_range.
        """
        def scale(v: float) -> int:
            return max(-100, min(100, int(v * 100)))

        xml = (
            f'<PTZData version="2.0" xmlns="{_NS}">'
            f'<AbsoluteHigh>'
            f'<elevation>{scale(tilt)}</elevation>'
            f'<azimuth>{scale(pan)}</azimuth>'
            f'<absoluteZoom>{max(0, int(zoom * 100))}</absoluteZoom>'
            f'</AbsoluteHigh>'
            f'</PTZData>'
        )
        ok, _ = self._put(f"/PTZCtrl/channels/{self._channel}/relative", xml)
        return ok

    # ── Presets ───────────────────────────────────────────────────────────────

    def goto_preset(self, preset_id: int) -> bool:
        """Move to a previously saved preset position."""
        ok, _ = self._put(
            f"/PTZCtrl/channels/{self._channel}/presets/{preset_id}/goto", "")
        return ok

    def save_preset(self, preset_id: int, name: str = "") -> bool:
        """Save current position as preset `preset_id`."""
        label = name.strip() or f"Preset {preset_id}"
        xml = _xml_root("PTZPreset",
                        presetName=label,
                        presetID=preset_id,
                        enabled="true")
        ok, _ = self._put(
            f"/PTZCtrl/channels/{self._channel}/presets/{preset_id}", xml)
        return ok

    def delete_preset(self, preset_id: int) -> bool:
        try:
            with self._lock:
                r = self._session.delete(
                    f"{self._base}/PTZCtrl/channels/{self._channel}/presets/{preset_id}",
                    timeout=self._timeout)
            return r.status_code < 300
        except Exception as e:
            self.last_error = str(e)
            return False

    def get_presets(self) -> list[PTZPreset]:
        """Return all saved presets."""
        ok, text = self._get(f"/PTZCtrl/channels/{self._channel}/presets")
        if not ok:
            return []
        try:
            root = ET.fromstring(text)
            presets = []
            for node in (root.findall("h:PTZPreset", _NS_MAP)
                         or root.findall("PTZPreset")):
                pid   = _find(node, "presetID")
                pname = _find(node, "presetName")
                if pid is not None:
                    presets.append(PTZPreset(
                        id=int(pid),
                        name=pname or f"Preset {pid}",
                    ))
            return sorted(presets, key=lambda p: p.id)
        except Exception:
            return []

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> PTZStatus:
        """Return current absolute PTZ position."""
        ok, text = self._get(f"/PTZCtrl/channels/{self._channel}/status")
        if not ok:
            return PTZStatus(error=self.last_error or "unreachable")
        try:
            root  = ET.fromstring(text)
            # Try AbsoluteHigh block first (most firmware versions)
            high  = (root.find("h:AbsoluteHigh", _NS_MAP)
                     or root.find("AbsoluteHigh") or root)
            pan   = float(_find(high, "azimuth")  or 0)
            tilt  = float(_find(high, "elevation") or 0)
            zoom  = float(_find(high, "absoluteZoom") or 1)
            return PTZStatus(pan=pan, tilt=tilt, zoom=zoom)
        except Exception as e:
            return PTZStatus(error=str(e))

    # ── Focus ─────────────────────────────────────────────────────────────────

    def focus_auto(self) -> bool:
        """
        Trigger one-push autofocus.

        Different Hikvision firmwares expose AF at different endpoints — we try
        them in order of compatibility and stop at the first that works.
        Returns False only if every path is rejected (e.g. fixed-focus lens).
        """
        ch = self._channel

        # 1. Universal one-touch AF trigger — empty PUT, supported on most
        #    motorised-focus cameras incl. bullet/turret models.
        ok, _ = self._put(f"/PTZCtrl/channels/{ch}/focusOneTouch", "")
        if ok:
            return True

        # 2. FocusConfiguration with full payload (XML declaration + FocusStyle).
        #    Older firmwares require this exact element set.
        xml = _xml_root("FocusConfiguration", FocusStyle="AUTO")
        ok, _ = self._put(f"/Image/channels/{ch}/focusConfiguration", xml)
        if ok:
            return True

        # 3. Last resort: bump the focus axis briefly to wake the AF loop.
        return self._focus_axis(direction="near", duration_ms=200)

    def focus_manual(self, direction: str = "near", speed: float = 0.5,
                     duration_ms: int = 600) -> bool:
        """
        Momentary manual focus move.
        direction: "near" (closer) or "far" (farther)
        speed: 0.0 – 1.0
        """
        return self._focus_axis(direction=direction, speed=speed,
                                duration_ms=duration_ms)

    def focus_stop(self) -> bool:
        return self._focus_axis(direction="near", speed=0.0, duration_ms=1)

    def _focus_axis(self, direction: str = "near", speed: float = 0.5,
                    duration_ms: int = 500) -> bool:
        """
        Move focus only — uses a focus-only PTZData payload first (strict
        firmware accepts that shape) and falls back to the combined PTZ
        momentary call if needed.
        """
        ch = self._channel
        v  = abs(speed) * (1 if direction == "near" else -1)
        f  = max(-100, min(100, int(v * 100)))

        # Focus-only momentary payload — most compatible with strict firmware.
        xml = _xml_root("PTZData", focus=f, duration=duration_ms)
        ok, _ = self._put(f"/PTZCtrl/channels/{ch}/momentary", xml)
        if ok:
            return True

        # Fallback to the combined call (sets every axis explicitly).
        return self.ptz_momentary(pan=0, tilt=0, zoom=0,
                                  focus=v, duration_ms=duration_ms)

    # ── Iris / Exposure ───────────────────────────────────────────────────────

    def iris_auto(self) -> bool:
        """Switch iris to auto exposure mode."""
        xml = (f'<IrisConfiguration xmlns="{_NS}">'
               f'<IrisLevel>AUTO</IrisLevel>'
               f'</IrisConfiguration>')
        ok, _ = self._put(
            f"/Image/channels/{self._channel}/irisConfiguration", xml)
        return ok

    def get_image_params(self) -> dict:
        """Return brightness / contrast / saturation / sharpness as integers (0–100)."""
        ok, text = self._get(f"/Image/channels/{self._channel}")
        if not ok:
            return {"brightness": 50, "contrast": 50,
                    "saturation": 50, "sharpness": 50, "error": self.last_error}
        try:
            root = ET.fromstring(text)
            def _int(tag, default=50):
                v = _find(root, tag)
                try:
                    return max(0, min(100, int(v)))
                except (TypeError, ValueError):
                    return default
            return {
                "brightness": _int("brightnessLevel"),
                "contrast":   _int("contrastLevel"),
                "saturation": _int("saturationLevel"),
                "sharpness":  _int("sharpnessLevel"),
            }
        except Exception as e:
            return {"brightness": 50, "contrast": 50,
                    "saturation": 50, "sharpness": 50, "error": str(e)}

    def set_image_params(
        self,
        brightness:  int | None = None,
        contrast:    int | None = None,
        saturation:  int | None = None,
        sharpness:   int | None = None,
    ) -> bool:
        """
        Adjust basic imaging parameters (0–100 scale).

        Hikvision ISAPI requires a complete PUT body — partial updates are
        rejected by most firmware versions.  We GET the current values first
        and merge, so callers can pass only the fields they want to change.
        """
        cur = self.get_image_params()

        def _clamp(new_val, cur_key):
            v = new_val if new_val is not None else cur.get(cur_key, 50)
            try:
                return max(0, min(100, int(v)))
            except (TypeError, ValueError):
                return 50

        b  = _clamp(brightness,  "brightness")
        c  = _clamp(contrast,    "contrast")
        s  = _clamp(saturation,  "saturation")
        sh = _clamp(sharpness,   "sharpness")

        xml = (
            f'<ImageChannel version="2.0" xmlns="{_NS}">'
            f'<id>{self._channel}</id>'
            f'<brightnessLevel>{b}</brightnessLevel>'
            f'<contrastLevel>{c}</contrastLevel>'
            f'<saturationLevel>{s}</saturationLevel>'
            f'<sharpnessLevel>{sh}</sharpnessLevel>'
            f'</ImageChannel>'
        )
        ok, _ = self._put(f"/Image/channels/{self._channel}", xml)
        return ok

    # ── Day / Night mode ──────────────────────────────────────────────────────

    def get_day_night_mode(self) -> str:
        """
        Return the current IR-cut filter mode: "day", "night", or "auto".
        Returns "unknown" on error.

        Hikvision ISAPI endpoint: GET /Image/channels/{ch}/ircutFilter
        """
        ok, text = self._get(f"/Image/channels/{self._channel}/ircutFilter")
        if not ok:
            return "unknown"
        try:
            root = ET.fromstring(text)
            mode = _find(root, "ircutFilterType") or ""
            return mode.lower() or "unknown"
        except Exception as e:
            self.last_error = str(e)
            return "unknown"

    def set_day_night_mode(self, mode: str) -> bool:
        """
        Force the IR-cut filter to a specific mode.

        mode: "day"   — white light, colour image  (use for sports)
              "night" — IR illumination, B&W image
              "auto"  — camera decides based on light level

        Returns True on success.
        """
        mode = mode.lower().strip()
        if mode not in ("day", "night", "auto"):
            self.last_error = f"Invalid mode '{mode}' — use day/night/auto"
            return False

        xml = (
            f'<IrcutFilter version="2.0" xmlns="{_NS}">'
            f"<ircutFilterType>{mode}</ircutFilterType>"
            f"</IrcutFilter>"
        )
        ok, _ = self._put(f"/Image/channels/{self._channel}/ircutFilter", xml)
        return ok

    # ── Stream control ────────────────────────────────────────────────────────

    def get_stream_info(self) -> dict:
        """Return main stream resolution / codec / bitrate."""
        ok, text = self._get(f"/Streaming/channels/{self._channel}01")
        if not ok:
            return {"error": self.last_error}
        try:
            root = ET.fromstring(text)
            v = root.find("h:Video", _NS_MAP) or root.find("Video") or root
            return {
                "codec":      _find(v, "videoCodecType"),
                "resolution": _find(v, "videoResolutionWidth"),
                "height":     _find(v, "videoResolutionHeight"),
                "fps":        _find(v, "maxFrameRate"),
                "bitrate":    _find(v, "constantBitRate"),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── OSD text overlay ──────────────────────────────────────────────────────

    def set_osd_text(self, text_line: str, enabled: bool = True) -> bool:
        """Set the custom OSD text overlay on the camera stream."""
        xml = (
            f'<OSDTextOverlay xmlns="{_NS}">'
            f'<enabled>{"true" if enabled else "false"}</enabled>'
            f'<type>textString</type>'
            f'<textString>{text_line[:32]}</textString>'
            f'</OSDTextOverlay>'
        )
        ok, _ = self._put(
            f"/System/Video/inputs/channels/{self._channel}/overlays/text/1", xml)
        return ok

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    def _get(self, path: str) -> tuple[bool, str]:
        try:
            with self._lock:
                r = self._session.get(
                    f"{self._base}{path}", timeout=self._timeout)
            if r.status_code >= 300:
                self.last_error = f"HTTP {r.status_code}"
                return False, r.text
            return True, r.text
        except requests.RequestException as e:
            self.last_error = str(e)
            return False, str(e)

    def _put(self, path: str, xml: str) -> tuple[bool, str]:
        try:
            with self._lock:
                r = self._session.put(
                    f"{self._base}{path}",
                    data=xml.encode("utf-8") if xml else b"",
                    headers={"Content-Type": "application/xml;charset=UTF-8"},
                    timeout=self._timeout)
            if r.status_code >= 300:
                self.last_error = f"HTTP {r.status_code}: {r.text[:120]}"
                return False, r.text
            return True, r.text
        except requests.RequestException as e:
            self.last_error = str(e)
            return False, str(e)


# ── Optional ONVIF wrapper ────────────────────────────────────────────────────

class ONVIFCamera:
    """
    Thin ONVIF wrapper using the `onvif-zeep` library (optional dependency).
    Falls back to a stub that raises ImportError if the library is not installed.

        pip install onvif-zeep

    ONVIF is useful when interoperating with non-Hikvision cameras.
    For Hikvision-only setups prefer HikvisionCamera (ISAPI) above.
    """

    def __init__(self, ip: str, user: str, password: str,
                 port: int = 80, wsdl_dir: str | None = None):
        try:
            from onvif import ONVIFCamera as _ONVIFCam
        except ImportError:
            raise ImportError(
                "onvif-zeep is not installed. "
                "Run: pip install onvif-zeep\n"
                "Or use HikvisionCamera (ISAPI) instead.")

        wsdl = wsdl_dir or self._find_wsdl()
        self._cam = _ONVIFCam(ip, port, user, password, wsdl)
        self._ptz     = self._cam.create_ptz_service()
        self._imaging = self._cam.create_imaging_service()
        self._media   = self._cam.create_media_service()
        self._profile = self._media.GetProfiles()[0]
        self._token   = self._profile.token

    def _find_wsdl(self) -> str:
        import onvif
        return str(__import__("pathlib").Path(onvif.__file__).parent / "wsdl")

    def ptz_continuous(self, pan: float, tilt: float, zoom: float = 0,
                       speed: float = 1.0) -> bool:
        try:
            req = self._ptz.create_type("ContinuousMove")
            req.ProfileToken = self._token
            req.Velocity = {
                "PanTilt": {"x": pan * speed, "y": tilt * speed},
                "Zoom":    {"x": zoom * speed},
            }
            self._ptz.ContinuousMove(req)
            return True
        except Exception:
            return False

    def ptz_stop(self) -> bool:
        try:
            req = self._ptz.create_type("Stop")
            req.ProfileToken = self._token
            req.PanTilt = True
            req.Zoom    = True
            self._ptz.Stop(req)
            return True
        except Exception:
            return False

    def focus_auto(self) -> bool:
        try:
            src = self._media.GetVideoSources()[0].token
            settings = self._imaging.GetImagingSettings(VideoSourceToken=src)
            settings.Focus.AutoFocusMode = "AUTO"
            self._imaging.SetImagingSettings(
                VideoSourceToken=src, ImagingSettings=settings)
            return True
        except Exception:
            return False
