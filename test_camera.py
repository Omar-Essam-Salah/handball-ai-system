"""
Camera Environment Check + RTSP Live Test
Hikvision DS-2CD2686G2-LZS  |  Sub-stream channel 102 (1080p H.264)

Camera:  172.16.5.174
  HTTP   → http://172.16.5.174:80      (web config UI)
  RTSP   → rtsp://<user>:<pass>@172.16.5.174:554/Streaming/Channels/102
  SDK    → 172.16.5.174:8000
  HTTPS  → https://172.16.5.174:8443

Auth:    Digest/MD5  (confirmed from camera config)
RTP:     video=8860  audio=8862  (UDP, not used in TCP mode)

Usage:
    python test_camera.py                    # uses default IP, prompts for password
    python test_camera.py admin MyPassword   # pass credentials directly
    python test_camera.py --channel 101      # main stream (4K) instead of sub
    python test_camera.py --file video.mp4   # test with local file
"""

import os
import sys
import socket
import subprocess
import time
import argparse

# ─── Camera defaults (pre-filled for your setup) ───────────────────────────
DEFAULT_IP       = "172.16.5.174"
DEFAULT_USER     = "admin"
DEFAULT_HTTP     = 80
DEFAULT_HTTPS    = 8443
DEFAULT_SDK_PORT = 8000
DEFAULT_RTSP     = 554
# ───────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────
# 1. ENVIRONMENT CHECK
# ──────────────────────────────────────────────

def check_env():
    print("\n=== Environment Check ===")
    print(f"  Python : {sys.version.split()[0]}")

    results = {}

    # OpenCV
    try:
        import cv2
        build = cv2.getBuildInformation()
        ffmpeg_ok = "FFMPEG:                      YES" in build or "GStreamer" in build
        print(f"  OpenCV : {cv2.__version__}  ✓  (FFMPEG backend: {'YES' if ffmpeg_ok else 'check build'})")
        results["cv2"] = cv2
    except ImportError:
        print("  OpenCV : NOT INSTALLED  ✗")
        print("           → pip install opencv-python")
        results["cv2"] = None

    # NumPy
    try:
        import numpy as np
        print(f"  NumPy  : {np.__version__}  ✓")
        results["np"] = np
    except ImportError:
        print("  NumPy  : NOT INSTALLED  ✗  → pip install numpy")
        results["np"] = None

    # PyTorch / CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  PyTorch: {torch.__version__}  ✓")
            print(f"  CUDA   : {torch.version.cuda}  ✓")
            print(f"  GPU    : {gpu}  ({vram:.1f} GB VRAM)  ✓")
        else:
            print(f"  PyTorch: {torch.__version__}  (CUDA NOT available — CPU only)")
        results["torch"] = torch
        results["cuda"]  = cuda_ok
    except ImportError:
        print("  PyTorch: not installed  (needed for Phase 2 AI)")
        results["torch"] = None
        results["cuda"]  = False

    # FFmpeg binary
    try:
        out = subprocess.run(["ffmpeg", "-version"],
                             capture_output=True, text=True, timeout=5)
        line = out.stdout.splitlines()[0].split("Copyright")[0].strip()
        print(f"  FFmpeg : {line}  ✓")
        results["ffmpeg"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  FFmpeg : NOT FOUND in PATH  ✗  → install ffmpeg and add to PATH")
        results["ffmpeg"] = False

    print()
    return results


# ──────────────────────────────────────────────
# 2. NETWORK / PORT CHECK
# ──────────────────────────────────────────────

def check_port(ip, port, label, timeout=2.0):
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        print(f"  ✓  {label:20s} {ip}:{port}  OPEN")
        return True
    except (ConnectionRefusedError, OSError):
        print(f"  ✗  {label:20s} {ip}:{port}  closed / unreachable")
        return False

def ping_camera(ip):
    print(f"  Ping {ip} ...")
    try:
        result = subprocess.run(
            ["ping", "-n", "3", "-w", "1000", ip],
            capture_output=True, text=True, timeout=10
        )
        if "TTL=" in result.stdout or "ttl=" in result.stdout:
            # Extract avg round-trip time
            for line in result.stdout.splitlines():
                if "Average" in line or "average" in line or "Moyenne" in line:
                    print(f"  ✓  Host reachable  ({line.strip()})")
                    break
            else:
                print(f"  ✓  Host reachable")
            return True
        else:
            print(f"  ✗  No reply — check network / hotspot")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗  Ping timed out")
        return False

def full_network_check(ip):
    print(f"\n=== Network Check  ({ip}) ===")
    reachable = ping_camera(ip)
    print()

    ports = [
        (DEFAULT_HTTP,    "HTTP web UI"),
        (DEFAULT_RTSP,    "RTSP stream"),
        (DEFAULT_SDK_PORT,"SDK service"),
        (DEFAULT_HTTPS,   "HTTPS / SDK-SSL"),
    ]
    results = {}
    for port, label in ports:
        results[port] = check_port(ip, port, label)

    rtsp_open = results.get(DEFAULT_RTSP, False)
    http_open = results.get(DEFAULT_HTTP, False)

    print()
    if http_open:
        print(f"  → Open camera web UI:  http://{ip}:{DEFAULT_HTTP}")
        print(f"    Verify: Video→Encoding→Sub-stream = H.264, 1080p, 25fps, 4-6Mbps")
    if not rtsp_open:
        print(f"  ⚠  RTSP port 554 closed — the stream test will fail.")
        print(f"     Check camera firewall / RTSP service in the web UI.")

    return reachable, rtsp_open


# ──────────────────────────────────────────────
# 3. RTSP STREAM TEST
# ──────────────────────────────────────────────

def build_rtsp_url(ip, user, password, channel=102, port=554):
    return f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}"


def test_stream(url, duration_s=15):
    import cv2

    # Mask password in display
    display_url = url
    try:
        at = url.index("@")
        proto_end = url.index("://") + 3
        display_url = url[:proto_end] + "****:****@" + url[at+1:]
    except ValueError:
        pass

    print(f"\n=== RTSP Stream Test ===")
    print(f"  URL : {display_url}")
    print(f"  (Press Q to stop, or auto-stops after {duration_s}s)\n")

    # Force TCP transport — more reliable on LAN, required for Digest/MD5 auth
    # Must be set before VideoCapture is created
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|"
        "timeout;5000000"          # 5 s connect timeout (microseconds)
    )

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("  ✗  Failed to open stream.")
        print()
        print("  Troubleshoot:")
        print("  1. Wrong password?  → open http://172.16.5.174 in browser and confirm login")
        print("  2. Camera auth is Digest/MD5 — make sure you have opencv-python >= 4.5")
        print("     (older builds have a bug with Digest RTSP)")
        print("  3. Try main stream:  python test_camera.py --channel 101")
        print("  4. Test in VLC first:")
        print("     Media → Open Network Stream →")
        print("     rtsp://admin:PASSWORD@172.16.5.174:554/Streaming/Channels/102")
        print("  5. Check camera: Configuration → Network → Network Service → RTSP → enabled")
        return False

    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_prop = cap.get(cv2.CAP_PROP_FPS)

    print(f"  ✓  Stream opened")
    print(f"     Resolution : {width}×{height}")
    print(f"     FPS (prop) : {fps_prop:.1f}")
    print()

    frame_count  = 0
    drop_count   = 0
    start        = time.perf_counter()
    last_report  = start

    while True:
        ret, frame = cap.read()
        if not ret:
            drop_count += 1
            if drop_count > 10:
                print("  ✗  Too many dropped frames — stream lost")
                break
            continue

        drop_count = 0
        frame_count += 1
        now     = time.perf_counter()
        elapsed = now - start
        live_fps = frame_count / elapsed if elapsed > 0 else 0

        # HUD overlay
        cv2.putText(frame, f"FPS: {live_fps:.1f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}  |  {width}x{height}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.putText(frame, f"172.16.5.174 ch{102}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        cv2.imshow("Handball Camera Test  —  Q to quit", frame)

        if now - last_report >= 3.0:
            print(f"  Live: {live_fps:.1f} fps  |  frames: {frame_count}  |  elapsed: {elapsed:.0f}s")
            last_report = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if elapsed >= duration_s:
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed  = time.perf_counter() - start
    avg_fps  = frame_count / elapsed if elapsed > 0 else 0

    print(f"\n  === Result ===")
    print(f"  Frames   : {frame_count}")
    print(f"  Duration : {elapsed:.1f}s")
    print(f"  Avg FPS  : {avg_fps:.1f}")

    if avg_fps >= 23:
        verdict = "✓  GOOD"
    elif avg_fps >= 15:
        verdict = "⚠  ACCEPTABLE (check bitrate / network load)"
    else:
        verdict = "✗  POOR — reduce resolution or check H.265→H.264"

    print(f"  Verdict  : {verdict}")
    return True


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Handball AI — camera environment + RTSP test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_camera.py                      # prompt for password
  python test_camera.py admin Hik12345       # direct credentials
  python test_camera.py --channel 101        # main stream (4K)
  python test_camera.py --duration 30        # longer test
  python test_camera.py --file match.mp4     # test with local video
        """
    )
    parser.add_argument("user",       nargs="?", default=DEFAULT_USER, help=f"RTSP user (default: {DEFAULT_USER})")
    parser.add_argument("password",   nargs="?", default=None,         help="RTSP password")
    parser.add_argument("--ip",       default=DEFAULT_IP,              help=f"Camera IP (default: {DEFAULT_IP})")
    parser.add_argument("--channel",  type=int, default=102,           help="Hikvision channel: 102=sub 1080p, 101=main 4K (default: 102)")
    parser.add_argument("--duration", type=int, default=15,            help="Preview seconds (default: 15)")
    parser.add_argument("--file",     default=None,                    help="Test with local video file")
    parser.add_argument("--net-only", action="store_true",             help="Network check only, skip stream test")
    args = parser.parse_args()

    print("=" * 60)
    print("  Handball AI  —  Camera Check")
    print(f"  Target: {args.ip}  (HTTP:{DEFAULT_HTTP} RTSP:{DEFAULT_RTSP} SDK:{DEFAULT_SDK_PORT})")
    print("=" * 60)

    # ── Environment ──────────────────────────────
    env = check_env()

    if env["cv2"] is None:
        print("  OpenCV missing — install it first:")
        print("  .venv\\Scripts\\activate")
        print("  pip install opencv-python\n")
        sys.exit(1)

    # ── File mode (skip network) ──────────────────
    if args.file:
        print(f"\n=== Local file test: {args.file} ===")
        test_stream(args.file, duration_s=args.duration)
        return

    # ── Network check ─────────────────────────────
    reachable, rtsp_open = full_network_check(args.ip)

    if args.net_only:
        return

    if not reachable:
        print("\n  Camera unreachable — fix network before stream test.")
        sys.exit(1)

    # ── Credentials ──────────────────────────────
    if args.password is None:
        args.password = input(f"\nRTSP password for {args.user}@{args.ip}: ").strip()

    # ── Stream test ───────────────────────────────
    url = build_rtsp_url(args.ip, args.user, args.password, args.channel)
    test_stream(url, duration_s=args.duration)

    print()
    print("─" * 60)
    print("Phase 2 install checklist (when ready):")
    print("  pip install opencv-python ultralytics boxmot")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("─" * 60)


if __name__ == "__main__":
    main()
