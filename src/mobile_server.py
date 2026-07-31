"""
Handball Mobile — phone/tablet monitoring server.

Runs the full analytics pipeline on this PC and serves a live, mobile-friendly
view over WiFi. Open  http://<this-pc-ip>:8000  on any phone on the same
network (or "Add to Home Screen" to use it like an app).

Mirrors the laptop view:
  • live annotated video (MJPEG)         → <img src="/video">
  • score / possession / shots / breaks  → /status (polled JSON)
  • players list + persistent identities
  • recent events
  • tactical pre-goal freezes             → /goals , /goal/<name>
  • all match controls                    → /cmd/<key>
  • open a video file OR scan + connect an IP camera

The heavy lifting stays in pipeline.run(gui_mode=True, …); this is just a thin
web shell, the same pattern as handball_studio.py.
"""
from __future__ import annotations

import argparse
import queue
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
import os
os.chdir(ROOT)

from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import (JSONResponse, StreamingResponse,
                               FileResponse, Response)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import subprocess
import pipeline as pl
import camera_discovery as cd
from shot_report import ShotReport, ZONES, CELLS, OFF_TARGET, OUTCOMES

_PYEXE = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(_PYEXE).exists():
    _PYEXE = sys.executable

REPORT = ShotReport(ROOT / "data" / "shots.json")

WEBAPP_DIR   = ROOT / "webapp"
MOBILE_MAX_W = 1280     # downscale frames for phone bandwidth


# ──────────────────────────────────────────────────────────────────────────
class PipelineService:
    def __init__(self):
        self.cmd_q: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_status: dict = {}
        self.thread: threading.Thread | None = None
        self.running = False
        self.source: str | None = None
        self.error: str = ""
        # Live control center state (overlays + AI thresholds), mutated by /control.
        self.controls: dict = {
            "skeleton": False, "boxes": False, "trajectory": True,
            "ids": True, "markers": True, "heatmap": False,
            "player_conf": 0.40, "ball_conf": 0.03,
            "paused": False, "speed": 1.0, "seek": None, "step": 0,
        }

    def _on_frame(self, frame: np.ndarray):
        try:
            if frame.shape[1] > MOBILE_MAX_W:
                s = MOBILE_MAX_W / frame.shape[1]
                frame = cv2.resize(frame, (MOBILE_MAX_W, int(frame.shape[0] * s)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
        except Exception:
            pass

    def _on_status(self, s: dict):
        with self.lock:
            self.latest_status = s

    def start(self, source: str):
        self.stop()
        self.source = source
        self.error = ""
        self.cmd_q = queue.Queue()
        self.running = True

        def _run():
            try:
                pl.run(
                    source=source, gui_mode=True,
                    frame_callback=self._on_frame,
                    status_callback=self._on_status,
                    command_queue=self.cmd_q,
                    device="cuda:0",
                    model=("yolov8s-pose.engine" if (ROOT / "yolov8s-pose.engine").exists()
                           else "yolov8s-pose.pt"),
                    conf=0.40, ball_conf=0.03, imgsz=1280,
                    infer_every=2, display_every=1, scale=1.0,
                    analytics=False, debug=False,
                    static_marker_enabled=True, no_ocr=False,
                    loop_source=True,   # replay a video file so the live view never ends
                    controls=self.controls,
                )
            except Exception as e:
                import traceback
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                self.running = False

        self.thread = threading.Thread(target=_run, daemon=True, name="PipelineSvc")
        self.thread.start()

    def stop(self):
        if self.running:
            self.cmd_q.put("q")
            if self.thread:
                self.thread.join(timeout=6.0)
        self.running = False
        with self.lock:
            self.latest_jpeg = None

    def cmd(self, c: str):
        if self.running and isinstance(c, str) and len(c) == 1:
            self.cmd_q.put(c)


SVC = PipelineService()
app = FastAPI(title="Handball Mobile")
# Allow the native WebView shell (and any typed cross-origin server) to call us.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _lan_ip() -> str:
    return cd.local_ip()


# ── video stream ──────────────────────────────────────────────────────────
def _mjpeg():
    blank = None
    while True:
        with SVC.lock:
            jpg = SVC.latest_jpeg
        if jpg is None:
            if blank is None:
                img = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(img, "waiting for video...", (130, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2)
                blank = cv2.imencode(".jpg", img)[1].tobytes()
            jpg = blank
            time.sleep(0.1)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    with SVC.lock:
        s = dict(SVC.latest_status)
    s["running"] = SVC.running
    s["source"] = SVC.source or ""
    s["error"] = SVC.error
    s["controls"] = dict(SVC.controls)
    return JSONResponse(s)


@app.post("/cmd/{c}")
def cmd(c: str):
    SVC.cmd(c)
    return {"ok": True}


_BOOL_KEYS  = {"skeleton", "boxes", "trajectory", "ids", "markers", "heatmap", "paused"}
_FLOAT_KEYS = {"player_conf", "ball_conf"}


@app.post("/control")
def control(payload: dict = Body(...)):
    """Live control center: overlays, AI thresholds, and playback (pause/speed/
    seek/step). Mutates the SHARED controls dict in place so the running pipeline
    picks it up on the very next frame."""
    applied = {}
    for k, v in payload.items():
        try:
            if k in _BOOL_KEYS:
                SVC.controls[k] = bool(v)
            elif k in _FLOAT_KEYS:
                SVC.controls[k] = max(0.0, min(1.0, float(v)))
            elif k == "speed":
                SVC.controls["speed"] = max(0.1, min(4.0, float(v)))
            elif k == "seek":
                SVC.controls["seek"] = max(0.0, min(1.0, float(v)))
            elif k == "step":
                SVC.controls["step"] = max(0, int(v))
            else:
                continue
            applied[k] = SVC.controls[k]
        except (TypeError, ValueError):
            pass
    return {"ok": True, "controls": dict(SVC.controls), "applied": applied}


@app.post("/open")
def open_source(payload: dict = Body(...)):
    src = str(payload.get("source", "")).strip()
    if not src:
        return JSONResponse({"ok": False, "error": "empty source"}, status_code=400)
    SVC.start(src)
    return {"ok": True, "source": src}


@app.post("/stop")
def stop():
    SVC.stop()
    return {"ok": True}


@app.get("/scan")
def scan(user: str = "admin", password: str = ""):
    found = []
    for hit in cd.scan_subnet():
        entry = {"ip": hit.ip, "url": "", "w": 0, "h": 0}
        for url in cd.candidate_rtsp_urls(hit.ip, user, password):
            ok, w, h = cd.verify_rtsp(url, timeout_s=4.0)
            if ok:
                entry.update(url=url, w=w, h=h)
                break
        if not entry["url"]:
            entry["url"] = cd.candidate_rtsp_urls(hit.ip, user, password)[0]
        found.append(entry)
    return {"cameras": found, "subnet": cd.subnet_base()}


@app.get("/goals")
def goals():
    with SVC.lock:
        gd = SVC.latest_status.get("goals_dir", "")
    names = []
    if gd and Path(gd).exists():
        names = sorted(p.name for p in Path(gd).glob("goal_*.png"))
    return {"goals": names}


@app.get("/goal/{name}")
def goal(name: str):
    with SVC.lock:
        gd = SVC.latest_status.get("goals_dir", "")
    p = Path(gd) / name if gd else None
    if p and p.exists() and p.suffix == ".png" and ".." not in name:
        return FileResponse(str(p), media_type="image/png")
    return Response(status_code=404)


# ── Pro Shot/Save reports (the IDEA/ reference format) ────────────────────
@app.get("/report/meta")
def report_meta():
    return {"zones": ZONES, "cells": CELLS, "off_target": OFF_TARGET,
            "outcomes": OUTCOMES, "players": REPORT.players(),
            "goalkeepers": REPORT.goalkeepers(), "n_shots": len(REPORT.shots)}


@app.get("/report/shot")
def report_shot(player: str, team: str = "team_a"):
    return REPORT.shot_report(player, team)


@app.get("/report/save")
def report_save(gk: str):
    return REPORT.save_report(gk)


@app.get("/report/shots")
def report_shots():
    return {"shots": REPORT.shots, "team": REPORT.team_summary()}


@app.post("/report/shot/add")
def report_add(payload: dict = Body(...)):
    rec = REPORT.add(
        player=str(payload.get("player", "?")),
        number=payload.get("number"),
        team=str(payload.get("team", "team_a")),
        zone=str(payload.get("zone", "Center Back")),
        cell=str(payload.get("cell", "CH")),
        outcome=str(payload.get("outcome", "GOAL")),
        gk=payload.get("gk"))
    return {"ok": True, "shot": rec}


@app.post("/report/shot/update")
def report_update(payload: dict = Body(...)):
    ok = REPORT.update(int(payload.get("id", 0)), **payload)
    return {"ok": ok}


@app.post("/report/shot/delete")
def report_delete(payload: dict = Body(...)):
    return {"ok": REPORT.delete(int(payload.get("id", 0)))}


# ── Local tools (the server runs on the PC, so it can launch them) ────────
_TOOL_CMDS = {
    "label":     [_PYEXE, "training/annotate_ball.py", "hand3.mp4"],
    "train":     [_PYEXE, "training/train_ball.py", "--base", "handball_ball.pt"],
    "calibrate": [_PYEXE, "calibrate_court.py"],
    "build_apk": ["cmd", "/c", "build_apk.bat"],
}


@app.post("/tools/{name}")
def run_tool(name: str):
    cmd = _TOOL_CMDS.get(name)
    if not cmd:
        return JSONResponse({"ok": False, "error": "unknown tool"}, status_code=400)
    try:
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        subprocess.Popen(cmd, cwd=str(ROOT), creationflags=flags)
        return {"ok": True, "launched": name}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/ping")
def ping():
    """Discovery signature — lets the phone app confirm this is a Handball
    server (used by the WebView shell's network auto-scan)."""
    return {"app": "handball", "running": SVC.running, "source": SVC.source or ""}


# ── Court calibration (fully in the web UI — no desktop windows) ───────────
#
#   GET  /calib/frame    → fresh RAW frame from the current source (base64)
#   POST /calib/corners  → {"corners": [[x,y]×4]} TL,TR,BR,BL in frame pixels;
#                          computes + persists BOTH homographies and returns a
#                          bird's-eye preview. Hot-reloads a running pipeline.
#   POST /calib/clear    → delete saved calibration
#   GET  /calib/status   → is a calibration on disk / active?
#
# Frames here are RAW (no undistortion) because the pipeline's CaptureThread
# is constructed without camera_matrix — clicks map 1:1 onto pipeline frames.

import base64
import json

CALIB_JSON  = ROOT / "config" / "court_calibration.json"
HOMOG_NPYS  = (ROOT / "homography_matrix.npy",          # video_engine.py
               ROOT / "config" / "homography_matrix.npy")  # pipeline analytics
COURT_W_M, COURT_H_M   = 40.0, 20.0    # IHF court, metres
MINI_W_PX, MINI_H_PX   = 400, 200      # bird's-eye minimap, 10 px = 1 m

_calib_lock = threading.Lock()
_calib_frame: np.ndarray | None = None   # last frame served → used for preview


class _CalibGrabber:
    """Fresh RAW frames for the calibration page.

    RTSP: keeps a low-latency reader thread alive while the page polls
    (auto-stops 15 s after the last request) so live preview is smooth.
    Files: one-shot open/read per request.
    """

    IDLE_STOP_S = 15.0

    def __init__(self):
        self.lock = threading.Lock()
        self.src: str | None = None
        self.frame: np.ndarray | None = None
        self.frame_src: str | None = None
        self.last_req = 0.0
        self.thread: threading.Thread | None = None

    def get(self, src: str) -> np.ndarray | None:
        if not src.startswith("rtsp"):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            ok, f = cap.read()
            cap.release()
            return f if ok else None

        with self.lock:
            self.last_req = time.time()
            if self.thread is None or not self.thread.is_alive() or self.src != src:
                self.src, self.frame, self.frame_src = src, None, None
                self.thread = threading.Thread(target=self._run, args=(src,),
                                               daemon=True, name="CalibGrabber")
                self.thread.start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None and self.frame_src == src:
                    return self.frame.copy()
            time.sleep(0.05)
        return None

    def _run(self, src: str) -> None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
            "rtsp_transport;tcp|timeout;5000000|fflags;nobuffer|flags;low_delay"
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try:
            while True:
                with self.lock:
                    expired = (time.time() - self.last_req) > self.IDLE_STOP_S
                    stale   = self.src != src
                if expired or stale:
                    break
                ok, f = cap.read()
                if not ok or f is None:
                    time.sleep(0.05)
                    continue
                with self.lock:
                    self.frame, self.frame_src = f, src
        finally:
            cap.release()


_calib_grabber = _CalibGrabber()


def _grab_fresh_frame() -> np.ndarray | None:
    src = SVC.source or ""
    f = _calib_grabber.get(src) if src else None
    # Match the pipeline's processing size (it caps frames at 1280 wide) so
    # clicked corner coordinates map 1:1 onto pipeline frames.
    if f is not None and f.shape[1] > 1280:
        f = cv2.resize(f, (1280, int(f.shape[0] * (1280 / f.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    return f


def _jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _calib_mjpeg():
    """Full-resolution RAW MJPEG for the calibration page live view.
    Keeps the grabber thread alive as long as the client stays connected."""
    while True:
        f = _grab_fresh_frame()
        if f is None:
            time.sleep(0.25)
            continue
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.06)          # ~16 fps — plenty for aiming the camera


@app.get("/calib/stream")
def calib_stream():
    return StreamingResponse(_calib_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/calib/frame")
def calib_frame():
    global _calib_frame
    frame = _grab_fresh_frame()
    if frame is None:
        return JSONResponse({"ok": False,
                             "error": "no source open — start the camera/video first"},
                            status_code=409)
    with _calib_lock:
        _calib_frame = frame
    return {"ok": True, "w": frame.shape[1], "h": frame.shape[0],
            "frame_b64": _jpeg_b64(frame)}


@app.post("/calib/corners")
def calib_corners(payload: dict = Body(...)):
    corners = payload.get("corners")
    if not corners or len(corners) != 4:
        return JSONResponse({"ok": False, "error": "need exactly 4 corners"},
                            status_code=400)
    try:
        src = np.array(corners, dtype=np.float32).reshape(4, 2)
    except Exception:
        return JSONResponse({"ok": False, "error": "bad corner format"},
                            status_code=400)

    # 1) Minimap homography (pixels → 400×200 bird's-eye, 10 px/m)
    dst_mini = np.array([[0, 0], [MINI_W_PX, 0],
                         [MINI_W_PX, MINI_H_PX], [0, MINI_H_PX]], np.float32)
    H_mini = cv2.getPerspectiveTransform(src, dst_mini)
    for p in HOMOG_NPYS:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, H_mini)

    # 2) Metre homography (pixels → 40×20 m) for CourtDetector + CourtTransformer
    dst_m = np.array([[0, 0], [COURT_W_M, 0],
                      [COURT_W_M, COURT_H_M], [0, COURT_H_M]], np.float32)
    H_m, _  = cv2.findHomography(src, dst_m)
    if H_m is None:
        return JSONResponse({"ok": False, "error": "homography computation failed"},
                            status_code=422)
    iH_m = np.linalg.inv(H_m)
    CALIB_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALIB_JSON.write_text(json.dumps({
        "homography":     H_m.tolist(),
        "inv_homography": iH_m.tolist(),
        "corners_px":     [[float(x), float(y)] for x, y in src],
        "saved_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }))

    # 3) Hot-reload the running pipeline (consumed once per flag in the loop)
    SVC.controls["reload_calib"] = True

    # 4) Bird's-eye preview (2× minimap scale for visibility)
    preview_b64 = ""
    with _calib_lock:
        base = _calib_frame
    if base is not None:
        warped = cv2.warpPerspective(base, H_mini, (MINI_W_PX, MINI_H_PX))
        preview_b64 = _jpeg_b64(cv2.resize(warped, (MINI_W_PX * 2, MINI_H_PX * 2),
                                           interpolation=cv2.INTER_LANCZOS4))

    print(f"[Calib] Saved court calibration from web UI: {[list(map(round, c)) for c in src.tolist()]}")
    return {"ok": True, "preview_b64": preview_b64,
            "saved": [str(CALIB_JSON)] + [str(p) for p in HOMOG_NPYS]}


@app.post("/calib/clear")
def calib_clear():
    removed = []
    for p in (CALIB_JSON, *HOMOG_NPYS):
        try:
            if p.exists():
                p.unlink()
                removed.append(str(p))
        except Exception:
            pass
    SVC.controls["reload_calib"] = True
    return {"ok": True, "removed": removed}


@app.get("/calib/status")
def calib_status():
    saved_at = None
    if CALIB_JSON.exists():
        try:
            saved_at = json.loads(CALIB_JSON.read_text()).get("saved_at")
        except Exception:
            pass
    return {"ok": True, "calibrated": CALIB_JSON.exists(),
            "minimap": all(p.exists() for p in HOMOG_NPYS),
            "saved_at": saved_at, "running": SVC.running}



# ── Upload-and-analyze: end-user critical-moment breakdown ─────────────────
#
#   POST /analyze/upload         → {video: file}  returns {job_id}
#   GET  /analyze/progress/{id}  → {status, processed, total}
#   GET  /analyze/result/{id}    → {moments:[{time_s, event_type, team,
#                                              frame_b64, report}], summary}
#
# Runs entirely separately from the live camera pipeline (SVC above) — its
# own Detector/Tracker/rules instances, in a background thread, so someone
# can upload a clip for analysis while (or without) a live match is running.

import uuid
from batch_match_analyzer import BatchMatchAnalyzer

UPLOADS_DIR = ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Gemini engine needs GEMINI_API_KEY. Rather than requiring a real OS
# environment variable (lost on every restart), also accept a small local
# file — already covered by .gitignore's `*.key` rule, so it can never be
# committed by accident. Create it once: config/gemini_api_key.key
# containing just the key, no quotes/newlines needed either way.
_GEMINI_KEY_FILE = ROOT / "config" / "gemini_api_key.key"
if not os.environ.get("GEMINI_API_KEY") and _GEMINI_KEY_FILE.exists():
    os.environ["GEMINI_API_KEY"] = _GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()

_analysis_jobs: dict[str, dict] = {}
_analysis_lock = threading.Lock()


def _run_analysis_job_cv(job_id: str, video_path: Path, team_a: str, team_b: str) -> None:
    def progress(processed: int, total: int) -> None:
        with _analysis_lock:
            _analysis_jobs[job_id].update(status="running", processed=processed, total=total)

    analyzer = BatchMatchAnalyzer(team_names={"team_a": team_a, "team_b": team_b})
    result = analyzer.analyze(str(video_path), progress_cb=progress)

    moments = []
    for m in result.moments:
        ok, buf = cv2.imencode(".jpg", m.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        moments.append({
            "time_s": round(m.time_s, 1),
            "event_type": m.event_type,
            "team": m.team,
            "frame_b64": base64.b64encode(buf.tobytes()).decode("ascii") if ok else "",
            "report": m.report.model_dump(),
        })
    with _analysis_lock:
        _analysis_jobs[job_id].update(
            status="done", processed=result.total_frames, total=result.total_frames,
            moments=moments, summary=result.summary)


def _run_analysis_job_gemini(job_id: str, video_path: Path) -> None:
    from gemini_match_analyzer import analyze_match_with_gemini
    with _analysis_lock:
        _analysis_jobs[job_id].update(status="running", processed=1, total=3)   # coarse: upload/wait/generate
    result = analyze_match_with_gemini(str(video_path))
    with _analysis_lock:
        _analysis_jobs[job_id].update(
            status="done", processed=3, total=3,
            moments=result["moments"], summary=result["summary"])


def _run_analysis_job(job_id: str, video_path: Path, team_a: str, team_b: str,
                      engine: str) -> None:
    try:
        if engine == "gemini":
            _run_analysis_job_gemini(job_id, video_path)
        else:
            _run_analysis_job_cv(job_id, video_path, team_a, team_b)
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _analysis_lock:
            _analysis_jobs[job_id].update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        try:
            video_path.unlink(missing_ok=True)      # don't keep uploaded clips around
        except Exception:
            pass


@app.post("/analyze/upload")
async def analyze_upload(video: UploadFile = File(...),
                         team_a: str = "الفريق الأول", team_b: str = "الفريق الثاني",
                         engine: str = "gemini"):
    """engine: "gemini" (default — fast, reads the on-screen scoreboard, works
    well on panning broadcast footage; costs Gemini API quota per upload) or
    "cv" (local YOLO+tracking pipeline — free, pixel-exact defender/gap boxes
    and shot-option arrows, but slow and tuned for a fixed installed camera,
    not moving broadcast footage)."""
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    dest = UPLOADS_DIR / f"{job_id}{suffix}"
    try:
        with open(dest, "wb") as f:
            while chunk := await video.read(1 << 20):
                f.write(chunk)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"upload failed: {e}"}, status_code=400)

    with _analysis_lock:
        _analysis_jobs[job_id] = {"status": "queued", "processed": 0, "total": 0}
    threading.Thread(target=_run_analysis_job, args=(job_id, dest, team_a, team_b, engine),
                     daemon=True, name=f"Analyze-{job_id}").start()
    return {"ok": True, "job_id": job_id, "engine": engine}


@app.get("/analyze/progress/{job_id}")
def analyze_progress(job_id: str):
    with _analysis_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "unknown job_id"}, status_code=404)
        return {"ok": True, "status": job["status"], "processed": job.get("processed", 0),
                "total": job.get("total", 0), "error": job.get("error", "")}


@app.get("/analyze/result/{job_id}")
def analyze_result(job_id: str):
    with _analysis_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "unknown job_id"}, status_code=404)
        if job["status"] == "error":
            return JSONResponse({"ok": False, "error": job.get("error", "analysis failed")},
                                status_code=500)
        if job["status"] != "done":
            return JSONResponse({"ok": False, "error": "not ready", "status": job["status"]},
                                status_code=425)
        return {"ok": True, "moments": job["moments"], "summary": job["summary"]}


# -- serve the PWA (static files) — mounted LAST so the API routes win --------
app.mount("/", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="web")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="", help="video file or RTSP URL to start immediately")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ip = _lan_ip()
    print("=" * 60)
    print("  HANDBALL MOBILE  —  open this on your phone (same WiFi):")
    print(f"      http://{ip}:{args.port}")
    print("=" * 60)
    if args.source:
        SVC.start(args.source)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
