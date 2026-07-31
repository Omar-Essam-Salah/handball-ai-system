"""
Camera auto-discovery — find IP cameras on the local network and build a
working RTSP URL from credentials.

No external deps (no ONVIF lib needed): we do a fast threaded TCP port scan of
the local /24 subnet for the well-known camera ports (RTSP 554, HTTP 80/8000),
then build vendor RTSP URLs and verify them by actually opening the stream with
OpenCV.

Public API
----------
    local_ip()                         -> "192.168.8.23"
    subnet_base(ip)                    -> "192.168.8"
    scan_subnet(base, progress=cb)     -> [CameraHit, ...]
    candidate_rtsp_urls(ip, user, pw)  -> [url, ...]   (vendor variants)
    verify_rtsp(url, timeout_s=6.0)    -> (ok, w, h)
    discover(user, pw, progress=cb)    -> [{"ip","url","w","h"}, ...]  (verified)
"""
from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

# Ports that strongly indicate an IP camera / NVR.
RTSP_PORT = 554
CAMERA_PORTS = (554, 80, 8000)


@dataclass
class CameraHit:
    ip: str
    open_ports: tuple


# ──────────────────────────────────────────────────────────────────────────
def local_ip() -> str:
    """Best-effort LAN IP of this machine (the interface used for outbound)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # no packets actually sent for UDP connect
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def subnet_base(ip: Optional[str] = None) -> str:
    """'192.168.8.23' -> '192.168.8'."""
    ip = ip or local_ip()
    parts = ip.split(".")
    return ".".join(parts[:3]) if len(parts) == 4 else "192.168.1"


def _port_open(ip: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((ip, port)) == 0
    except Exception:
        return False


def _probe_host(ip: str, ports=CAMERA_PORTS, timeout: float = 0.35):
    open_ports = tuple(p for p in ports if _port_open(ip, p, timeout))
    # Require RTSP (554) to call it a camera — kills PCs/phones with port 80.
    if RTSP_PORT in open_ports:
        return CameraHit(ip=ip, open_ports=open_ports)
    return None


def scan_subnet(base: Optional[str] = None,
                progress: Optional[Callable[[int, int], None]] = None,
                timeout: float = 0.35,
                max_workers: int = 128) -> list[CameraHit]:
    """Scan base.1 .. base.254 for hosts exposing RTSP. ``progress(done, total)``
    is called as hosts complete (for a GUI progress bar)."""
    base = base or subnet_base()
    hosts = [f"{base}.{i}" for i in range(1, 255)]
    hits: list[CameraHit] = []
    done, total = 0, len(hosts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_probe_host, h, CAMERA_PORTS, timeout): h for h in hosts}
        for fut in as_completed(futs):
            done += 1
            if progress:
                try: progress(done, total)
                except Exception: pass
            hit = fut.result()
            if hit:
                hits.append(hit)
    hits.sort(key=lambda h: tuple(int(p) for p in h.ip.split(".")))
    return hits


def candidate_rtsp_urls(ip: str, user: str, password: str,
                        port: int = RTSP_PORT) -> list[str]:
    """Vendor RTSP path variants, most-likely first. We try the Hikvision
    sub-stream (102) first since that's this project's camera."""
    cred = f"{user}:{password}@" if user else ""
    base = f"rtsp://{cred}{ip}:{port}"
    return [
        f"{base}/Streaming/Channels/102",   # Hikvision sub-stream (lighter)
        f"{base}/Streaming/Channels/101",   # Hikvision main stream
        f"{base}/h264/ch1/sub/av_stream",   # Dahua / generic sub
        f"{base}/cam/realmonitor?channel=1&subtype=1",  # Dahua
        f"{base}/live",                     # generic
        f"{base}/11",                       # some NVRs
    ]


def verify_rtsp(url: str, timeout_s: float = 6.0):
    """Open the stream briefly and grab one frame. Returns (ok, w, h)."""
    import os
    import cv2
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|timeout;%d000000" % int(timeout_s)
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return (False, 0, 0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return (True, int(frame.shape[1]), int(frame.shape[0]))
        return (False, 0, 0)
    finally:
        cap.release()


def discover(user: str, password: str,
             base: Optional[str] = None,
             progress: Optional[Callable[[int, int], None]] = None,
             verify: bool = True) -> list[dict]:
    """Full flow: scan subnet, then (optionally) verify a working RTSP URL for
    each camera found. Returns a list of {ip, url, w, h} (url/w/h may be blank
    if verify=False or no variant worked)."""
    results: list[dict] = []
    for hit in scan_subnet(base, progress=progress):
        entry = {"ip": hit.ip, "url": "", "w": 0, "h": 0,
                 "ports": list(hit.open_ports)}
        if verify:
            for url in candidate_rtsp_urls(hit.ip, user, password):
                ok, w, h = verify_rtsp(url, timeout_s=5.0)
                if ok:
                    entry.update(url=url, w=w, h=h)
                    break
        results.append(entry)
    return results


if __name__ == "__main__":
    import sys
    u = sys.argv[1] if len(sys.argv) > 1 else "admin"
    p = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"Local IP {local_ip()}  subnet {subnet_base()}.0/24")
    print("Scanning… (RTSP 554)")
    for c in discover(u, p, progress=lambda d, t: None):
        tag = f"  {c['url']}  ({c['w']}x{c['h']})" if c["url"] else "  (no RTSP variant verified)"
        print(f"{c['ip']:16s} ports={c['ports']}{tag}")
