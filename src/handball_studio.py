"""
Handball Studio — the unified end-user program.

One window that ties the whole project together:
  • Open a video file OR auto-detect an IP camera (subnet scan + RTSP) and connect
  • Smooth embedded live view (the full analytics pipeline, no separate CV window)
  • All match controls as buttons (recalibrate, flip sides, manual goals, debug…)
  • Live Match Summary (score / possession / shots / fast breaks)
  • Players list (live on-court + persistent ReID identities)
  • Tactical Playback (browse the auto-saved pre-goal freeze frames)

The heavy lifting stays in pipeline.run(); this file is just a friendly shell
around it. pipeline.run(gui_mode=True, …) feeds us annotated frames + a status
dict and takes single-char control commands back through a queue.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)   # so model/weight paths resolve like the .bat launchers

import numpy as np

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
    from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon, QFont
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QGridLayout, QTabWidget, QListWidget, QListWidgetItem,
        QDialog, QLineEdit, QProgressBar, QFileDialog, QFrame, QSizePolicy,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QStatusBar,
        QGroupBox, QSlider, QComboBox,
    )
except Exception as e:   # pragma: no cover
    print("[Handball Studio] PyQt6 is required:  pip install PyQt6")
    print(f"  import error: {e}")
    sys.exit(1)

import camera_discovery as cd
import pipeline as pl
from shot_report import ShotReport

# ── Theme ────────────────────────────────────────────────────────────────
DARK = """
QWidget { background:#15171c; color:#e6e8ec; font-family:'Segoe UI'; font-size:13px; }
QMainWindow, QDialog { background:#0f1115; }
QPushButton { background:#262a33; border:1px solid #353b47; border-radius:6px;
              padding:7px 12px; }
QPushButton:hover { background:#2f3540; }
QPushButton:pressed { background:#3a4250; }
QPushButton#primary { background:#2d6cf6; border:none; font-weight:600; }
QPushButton#primary:hover { background:#3b78ff; }
QPushButton#danger { background:#b3322f; border:none; }
QPushButton#danger:hover { background:#cc3a36; }
QLineEdit { background:#0f1115; border:1px solid #353b47; border-radius:5px; padding:6px; }
QTabWidget::pane { border:1px solid #262a33; border-radius:6px; }
QTabBar::tab { background:#1b1e25; padding:8px 16px; border-top-left-radius:6px;
               border-top-right-radius:6px; }
QTabBar::tab:selected { background:#262a33; color:#7fb0ff; }
QListWidget, QTableWidget { background:#0f1115; border:1px solid #262a33; border-radius:6px; }
QGroupBox { border:1px solid #262a33; border-radius:6px; margin-top:10px; padding-top:10px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; color:#8a93a6; }
QProgressBar { background:#0f1115; border:1px solid #353b47; border-radius:5px; text-align:center; }
QProgressBar::chunk { background:#2d6cf6; border-radius:4px; }
QStatusBar { background:#0f1115; color:#8a93a6; }
"""

TEAM_COLORS = {"team_a": "#ff5050", "team_b": "#32b4ff",
               "goalkeeper": "#32ff32", "unknown": "#9aa0aa"}


def _bgr_to_qpix(img: np.ndarray) -> QPixmap:
    if img is None or img.size == 0:
        return QPixmap()
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


# ── Pipeline worker thread ────────────────────────────────────────────────
class PipelineWorker(QThread):
    frame_ready  = pyqtSignal(object)   # np.ndarray (BGR)
    status_ready = pyqtSignal(dict)
    log          = pyqtSignal(str)
    stopped      = pyqtSignal()

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self.source   = source
        self.cmd_q: "queue.Queue[str]" = queue.Queue()
        # Shared with pipeline.run(): the GUI mutates these live for playback
        # transport (pause / speed / seek / frame-step). The pipeline reads them
        # each loop, so plain dict writes are enough (no lock needed).
        self.controls: dict = {"paused": False, "speed": 1.0, "seek": None, "step": 0}
        self._params = dict(
            device="cuda:0",
            model=("yolov8s-pose.engine" if (ROOT / "yolov8s-pose.engine").exists()
                   else "yolov8s-pose.pt"),
            conf=0.40, ball_conf=0.03, imgsz=640,
            infer_every=2, display_every=1, scale=1.0,
            analytics=False,            # the analytics worker is a no-op; skip mp in GUI
            debug=False, static_marker_enabled=True, no_ocr=False,
        )

    def send(self, ch: str):
        self.cmd_q.put(ch)

    # ── playback transport (mutates the shared controls dict) ──
    def set_paused(self, paused: bool):
        self.controls["paused"] = bool(paused)

    def toggle_pause(self) -> bool:
        self.controls["paused"] = not self.controls.get("paused", False)
        return self.controls["paused"]

    def set_speed(self, spd: float):
        self.controls["speed"] = float(spd)

    def seek_fraction(self, frac: float):
        self.controls["seek"] = max(0.0, min(1.0, float(frac)))

    def step_one(self):
        """Advance a single frame while paused (frame-by-frame analysis)."""
        self.controls["paused"] = True
        self.controls["step"] = int(self.controls.get("step", 0) or 0) + 1

    def run(self):
        try:
            pl.run(
                source=self.source,
                gui_mode=True,
                frame_callback=lambda f: self.frame_ready.emit(f),
                status_callback=lambda s: self.status_ready.emit(s),
                command_queue=self.cmd_q,
                controls=self.controls,
                **self._params,
            )
        except Exception as e:
            import traceback
            self.log.emit(f"[pipeline error] {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            self.stopped.emit()


# ── Camera scan worker ────────────────────────────────────────────────────
class ScanWorker(QThread):
    progress = pyqtSignal(int, int)
    found    = pyqtSignal(dict)
    done     = pyqtSignal()

    def __init__(self, base, user, pw, parent=None):
        super().__init__(parent)
        self.base, self.user, self.pw = base, user, pw

    def run(self):
        try:
            for hit in cd.scan_subnet(self.base,
                                      progress=lambda d, t: self.progress.emit(d, t)):
                entry = {"ip": hit.ip, "url": "", "w": 0, "h": 0,
                         "ports": list(hit.open_ports)}
                for url in cd.candidate_rtsp_urls(hit.ip, self.user, self.pw):
                    ok, w, h = cd.verify_rtsp(url, timeout_s=4.0)
                    if ok:
                        entry.update(url=url, w=w, h=h)
                        break
                self.found.emit(entry)
        finally:
            self.done.emit()


# ── Camera connect dialog ─────────────────────────────────────────────────
class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect Camera")
        self.setMinimumWidth(560)
        self.chosen_url = None
        self._scan = None

        cfg = self._load_cfg()
        lay = QVBoxLayout(self)

        form = QGroupBox("Network")
        g = QGridLayout(form)
        self.ed_subnet = QLineEdit(cd.subnet_base())
        self.ed_user   = QLineEdit(cfg.get("user", "admin"))
        self.ed_pass   = QLineEdit(cfg.get("password", ""))
        self.ed_pass.setEchoMode(QLineEdit.EchoMode.Password)
        g.addWidget(QLabel("Subnet"),   0, 0); g.addWidget(self.ed_subnet, 0, 1)
        g.addWidget(QLabel("Username"), 1, 0); g.addWidget(self.ed_user,   1, 1)
        g.addWidget(QLabel("Password"), 2, 0); g.addWidget(self.ed_pass,   2, 1)
        lay.addWidget(form)

        row = QHBoxLayout()
        self.btn_scan = QPushButton("🔍  Scan network")
        self.btn_scan.clicked.connect(self._start_scan)
        row.addWidget(self.btn_scan)
        self.bar = QProgressBar(); self.bar.setValue(0)
        row.addWidget(self.bar, 1)
        lay.addLayout(row)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda *_: self._accept_selected())
        lay.addWidget(self.list, 1)

        man = QGroupBox("Or enter RTSP URL manually")
        mg = QHBoxLayout(man)
        self.ed_url = QLineEdit()
        self.ed_url.setPlaceholderText("rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102")
        mg.addWidget(self.ed_url, 1)
        lay.addWidget(man)

        btns = QHBoxLayout()
        btns.addStretch(1)
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        b_ok = QPushButton("Connect"); b_ok.setObjectName("primary")
        b_ok.clicked.connect(self._accept_selected)
        btns.addWidget(b_cancel); btns.addWidget(b_ok)
        lay.addLayout(btns)

    def _load_cfg(self):
        try:
            import yaml
            return yaml.safe_load((ROOT / "config" / "camera.yaml").read_text()) or {}
        except Exception:
            return {}

    def _start_scan(self):
        self.list.clear()
        self.bar.setValue(0)
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Scanning…")
        self._scan = ScanWorker(self.ed_subnet.text().strip(),
                                self.ed_user.text().strip(),
                                self.ed_pass.text())
        self._scan.progress.connect(lambda d, t: self.bar.setValue(int(100 * d / max(t, 1))))
        self._scan.found.connect(self._on_found)
        self._scan.done.connect(self._scan_done)
        self._scan.start()

    def _on_found(self, entry: dict):
        if entry.get("url"):
            label = f"✓  {entry['ip']}   {entry['w']}×{entry['h']}   (RTSP verified)"
        else:
            label = f"•  {entry['ip']}   ports={entry['ports']}   (RTSP not verified — check credentials)"
        it = QListWidgetItem(label)
        it.setData(Qt.ItemDataRole.UserRole, entry)
        self.list.addItem(it)

    def _scan_done(self):
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("🔍  Scan network")
        if self.list.count() == 0:
            self.list.addItem("No cameras with open RTSP (554) found on this subnet.")

    def _accept_selected(self):
        manual = self.ed_url.text().strip()
        if manual:
            self.chosen_url = manual
            self.accept(); return
        it = self.list.currentItem()
        if it is not None:
            entry = it.data(Qt.ItemDataRole.UserRole)
            if entry:
                url = entry.get("url")
                if not url:
                    # Build a default Hikvision URL even if not verified.
                    url = cd.candidate_rtsp_urls(entry["ip"], self.ed_user.text().strip(),
                                                 self.ed_pass.text())[0]
                self.chosen_url = url
                self.accept(); return
        QMessageBox.information(self, "Pick a camera",
                                "Select a camera from the list or type an RTSP URL.")


# ── Main window ───────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handball Studio")
        self.resize(1320, 820)
        self.worker: PipelineWorker | None = None
        self.debug_on = False
        self.goals_dir: Path | None = None
        self._goal_files: set[str] = set()
        self._phone_proc: subprocess.Popen | None = None
        self._pyexe = str(ROOT / ".venv" / "Scripts" / "python.exe")
        if not Path(self._pyexe).exists():
            self._pyexe = sys.executable
        # scouting (event tagging) state
        self.report = ShotReport(ROOT / "data" / "shots.json")
        self._scout_player = None          # (label, team) currently selected
        self._match_clock = 0.0            # seconds, from the live status
        # playback transport state (file sources)
        self._pos_frames   = 0
        self._total_frames = 0
        self._src_fps      = 25.0
        self._scrubbing    = False

        self._build_ui()
        self._goal_timer = QTimer(self)
        self._goal_timer.timeout.connect(self._refresh_goals)
        self._goal_timer.start(1500)

    # ---- layout ----
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        # left: video + controls
        left = QVBoxLayout(); left.setSpacing(8)
        self.video = QLabel("Open a video or connect a camera to begin.")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(880, 560)
        self.video.setStyleSheet("background:#000; border:1px solid #262a33; border-radius:8px; color:#5a6172;")
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.video, 1)
        left.addWidget(self._transport_bar())
        left.addWidget(self._control_bar())
        root.addLayout(left, 1)

        # right: tabs
        self.tabs = QTabWidget(); self.tabs.setFixedWidth(380)
        self.tabs.addTab(self._scout_tab(),   "Scout")
        self.tabs.addTab(self._match_tab(),   "Match")
        self.tabs.addTab(self._players_tab(), "Players")
        self.tabs.addTab(self._tactics_tab(), "Tactics")
        root.addWidget(self.tabs)

        # top toolbar — WATCH
        tb = self.addToolBar("main"); tb.setMovable(False)
        a_open = QAction("📂  Open Video", self); a_open.triggered.connect(self.open_video)
        a_cam  = QAction("📡  Connect Camera", self); a_cam.triggered.connect(self.connect_camera)
        a_stop = QAction("⏹  Stop", self); a_stop.triggered.connect(self.stop_pipeline)
        tb.addAction(a_open); tb.addAction(a_cam); tb.addSeparator(); tb.addAction(a_stop)

        # second toolbar — everything the old .bat files did, now buttons
        self.addToolBarBreak()
        tools = self.addToolBar("tools"); tools.setMovable(False)
        a_phone = QAction("📱  Phone Mode", self); a_phone.setToolTip(
            "Start the phone server so your phone can watch (Handball_Mobile.bat)")
        a_phone.triggered.connect(self.toggle_phone_mode)
        a_label = QAction("🎯  Label Ball", self); a_label.setToolTip(
            "Mark correct ball positions to train the model (annotate_ball.py)")
        a_label.triggered.connect(self.launch_label)
        a_train = QAction("🧠  Train Model", self); a_train.setToolTip(
            "Retrain the ball model on your labels (train_ball.py)")
        a_train.triggered.connect(self.launch_train)
        a_cal   = QAction("📐  Calibrate Court", self); a_cal.triggered.connect(self.launch_calibrate)
        a_apk   = QAction("🔨  Build Phone App", self); a_apk.triggered.connect(self.launch_build_apk)
        for a in (a_phone, a_label, a_train, a_cal, a_apk):
            tools.addAction(a)
        self._a_phone = a_phone

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.  Watch a match, or use the tools row to label / train / go mobile.")

    # ── Playback transport (file scrubbing / speed / seek) ──
    def _transport_bar(self) -> QWidget:
        box = QFrame(); box.setFrameShape(QFrame.Shape.NoFrame)
        v = QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)

        # row 1 — scrub slider + time
        r1 = QHBoxLayout()
        self.tp_slider = QSlider(Qt.Orientation.Horizontal)
        self.tp_slider.setRange(0, 1000)
        self.tp_slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.tp_slider.sliderReleased.connect(self._on_scrub_released)
        self.tp_time = QLabel("00:00 / 00:00")
        self.tp_time.setStyleSheet("color:#9aa3b2; font-variant-numeric:tabular-nums;")
        r1.addWidget(self.tp_slider, 1); r1.addWidget(self.tp_time)
        v.addLayout(r1)

        # row 2 — buttons + speed + go-to
        r2 = QHBoxLayout(); r2.setSpacing(6)
        b_back = QPushButton("⏪ 10s"); b_back.setToolTip("Back 10 seconds")
        b_back.clicked.connect(lambda: self._seek_rel(-10))
        self.tp_play = QPushButton("⏸ Pause"); self.tp_play.setObjectName("primary")
        self.tp_play.clicked.connect(self._toggle_play)
        b_fwd = QPushButton("10s ⏩"); b_fwd.setToolTip("Forward 10 seconds")
        b_fwd.clicked.connect(lambda: self._seek_rel(+10))
        b_step = QPushButton("⏭ Frame"); b_step.setToolTip("Advance one frame (pauses)")
        b_step.clicked.connect(self._step_frame)
        for b in (b_back, self.tp_play, b_fwd, b_step):
            r2.addWidget(b)
        r2.addSpacing(10)
        r2.addWidget(QLabel("Speed"))
        self.tp_speed = QComboBox()
        self.tp_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.tp_speed.setCurrentText("1×")
        self.tp_speed.currentTextChanged.connect(self._set_speed)
        r2.addWidget(self.tp_speed)
        r2.addSpacing(10)
        r2.addWidget(QLabel("Go to"))
        self.tp_goto = QLineEdit(); self.tp_goto.setPlaceholderText("3  or  3:30")
        self.tp_goto.setFixedWidth(78)
        self.tp_goto.returnPressed.connect(self._seek_to_text)
        b_go = QPushButton("Go"); b_go.clicked.connect(self._seek_to_text)
        r2.addWidget(self.tp_goto); r2.addWidget(b_go)
        r2.addStretch(1)
        v.addLayout(r2)

        box.setEnabled(False)          # enabled once a FILE source reports a duration
        self.tp_box = box
        return box

    def _worker_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @staticmethod
    def _fmt_time(secs) -> str:
        secs = max(0, int(secs))
        return f"{secs // 60:02d}:{secs % 60:02d}"

    def _toggle_play(self):
        if not self._worker_running():
            return
        paused = self.worker.toggle_pause()
        self.tp_play.setText("▶ Play" if paused else "⏸ Pause")

    def _set_speed(self, text: str):
        if self._worker_running():
            self.worker.set_speed(float(text.replace("×", "")))

    def _seek_rel(self, dsec: float):
        if not (self._worker_running() and self._total_frames and self._src_fps):
            return
        dur = self._total_frames / self._src_fps
        cur = self._pos_frames / self._src_fps
        self.worker.seek_fraction((cur + dsec) / max(dur, 1e-3))

    def _seek_to_text(self):
        if not (self._worker_running() and self._total_frames and self._src_fps):
            return
        txt = self.tp_goto.text().strip()
        if not txt:
            return
        try:
            if ":" in txt:
                mm, ss = txt.split(":", 1)
                secs = int(mm) * 60 + int(ss)
            else:
                secs = int(float(txt)) * 60     # a bare number means MINUTES
        except ValueError:
            self.statusBar().showMessage("Enter minutes (e.g. 3) or mm:ss (e.g. 3:30).", 4000)
            return
        dur = self._total_frames / self._src_fps
        self.worker.seek_fraction(secs / max(dur, 1e-3))
        self.statusBar().showMessage(f"Jumped to {self._fmt_time(secs)}", 3000)

    def _step_frame(self):
        if self._worker_running():
            self.worker.step_one()
            self.tp_play.setText("▶ Play")

    def _on_scrub_released(self):
        self._scrubbing = False
        if self._worker_running():
            self.worker.seek_fraction(self.tp_slider.value() / 1000.0)

    def _control_bar(self) -> QWidget:
        box = QFrame(); box.setFrameShape(QFrame.Shape.NoFrame)
        h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        def mk(text, ch, tip, obj=None):
            b = QPushButton(text); b.setToolTip(tip)
            if obj: b.setObjectName(obj)
            b.clicked.connect(lambda *_: self._cmd(ch))
            h.addWidget(b); return b
        mk("Recalibrate", "r", "Re-detect team colours (R)")
        mk("Flip Sides", "f", "Swap attacking sides at half-time (F)")
        mk("Goal: Home", "a", "Manual goal for Home (A)", "primary")
        mk("Goal: Away", "b", "Manual goal for Away (B)", "primary")
        self.btn_dbg = mk("Debug: OFF", "d", "Toggle debug overlay (D)")
        mk("Resume", " ", "Dismiss a tactical freeze (Space)")
        h.addStretch(1)
        return box

    # ── Scouting (event tagging) — the Once Sport / Data Video workflow ──
    SCOUT_EVENTS = [
        ("Goal", "#2f9e54", "shot"),       ("Shot Saved", "#2d6cf6", "shot"),
        ("Shot Out", "#d98a2b", "shot"),   ("Shot Post", "#8a6d3b", "shot"),
        ("Blocked", "#5b6472", "shot"),    ("7m Penalty", "#7a52c0", "shot"),
        ("Assist", "#2f9e54", "other"),    ("Steal", "#2d8c8c", "other"),
        ("Turnover", "#c2a33b", "other"),  ("Foul Att", "#b3322f", "other"),
        ("Foul Def", "#8c2c2a", "other"),  ("Fast Break", "#2f8fa0", "other"),
    ]

    def _scout_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setSpacing(8)
        v.addWidget(QLabel("Players  —  tap to select the actor"))
        self.scout_players = QListWidget(); self.scout_players.setMaximumHeight(150)
        self.scout_players.itemClicked.connect(self._scout_pick_player)
        v.addWidget(self.scout_players)
        self.lbl_sel = QLabel("selected: —"); self.lbl_sel.setStyleSheet("color:#7fb0ff;font-weight:600")
        v.addWidget(self.lbl_sel)
        v.addWidget(QLabel("Tag event"))
        grid = QGridLayout(); grid.setSpacing(6)
        for i, (name, col, kind) in enumerate(self.SCOUT_EVENTS):
            b = QPushButton(name)
            b.setStyleSheet(f"background:{col};border:none;color:#fff;font-weight:700;"
                            f"padding:13px 6px;border-radius:8px")
            b.clicked.connect(lambda _=0, n=name, k=kind: self._tag_event(n, k))
            grid.addWidget(b, i // 2, i % 2)
        v.addLayout(grid)
        v.addWidget(self._hline())
        row = QHBoxLayout(); row.addWidget(QLabel("Event timeline")); row.addStretch(1)
        bx = QPushButton("Export"); bx.clicked.connect(self._scout_export); row.addWidget(bx)
        bc = QPushButton("Clear");  bc.clicked.connect(lambda: self.scout_timeline.clear()); row.addWidget(bc)
        v.addLayout(row)
        self.scout_timeline = QListWidget()
        v.addWidget(self.scout_timeline, 1)
        return w

    def _scout_pick_player(self, item):
        self._scout_player = item.data(Qt.ItemDataRole.UserRole) or (item.text(), "team_a")
        self.lbl_sel.setText("selected: " + self._scout_player[0])

    def _tag_event(self, name: str, kind: str):
        import re
        mm, ss = int(self._match_clock) // 60, int(self._match_clock) % 60
        label, team = self._scout_player or ("?", "team_a")
        self.scout_timeline.insertItem(0, QListWidgetItem(f"{mm:02d}:{ss:02d}   {name:<11} {label}"))
        if kind == "shot":
            out = {"Goal": "GOAL", "Shot Saved": "SAVE", "Shot Out": "MISS",
                   "Shot Post": "POST", "Blocked": "BLOCK", "7m Penalty": "GOAL"}.get(name, "GOAL")
            m = re.search(r"#(\d+)", label)
            self.report.add(player=label, number=int(m.group(1)) if m else None,
                            team=team, zone=("7m Penalty" if "7m" in name else "Center Back"),
                            cell="CH", outcome=out)
        self.statusBar().showMessage(f"tagged {name} · {label} · {mm:02d}:{ss:02d}", 4000)

    def _scout_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export timeline", str(ROOT / "match_timeline.txt"), "Text (*.txt)")
        if path:
            lines = [self.scout_timeline.item(i).text() for i in range(self.scout_timeline.count())]
            Path(path).write_text("\n".join(reversed(lines)), encoding="utf-8")
            self.statusBar().showMessage(f"exported {len(lines)} events → {path}", 6000)

    def _match_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.lbl_score = QLabel("0  –  0"); self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont(); f.setPointSize(30); f.setBold(True); self.lbl_score.setFont(f)
        v.addWidget(self.lbl_score)
        self.lbl_teams = QLabel("Home   vs   Away"); self.lbl_teams.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_teams.setStyleSheet("color:#8a93a6;")
        v.addWidget(self.lbl_teams)

        grid = QGridLayout()
        self.stat = {}
        rows = [("Period", "period"), ("Possession", "possession"),
                ("Shots", "shots"), ("Fast breaks", "fast_breaks"),
                ("Passes", "passes"), ("Ball", "ball"), ("FPS", "fps")]
        for i, (label, key) in enumerate(rows):
            grid.addWidget(QLabel(label), i, 0)
            val = QLabel("–"); val.setStyleSheet("color:#cdd3dd; font-weight:600;")
            grid.addWidget(val, i, 1); self.stat[key] = val
        v.addLayout(grid)

        v.addWidget(self._hline())
        v.addWidget(QLabel("Recent events"))
        self.events = QListWidget(); v.addWidget(self.events, 1)
        return w

    def _players_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.lbl_reid = QLabel("Persistent identities: 0")
        self.lbl_reid.setStyleSheet("color:#8a93a6;")
        v.addWidget(self.lbl_reid)
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Player", "Team"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.verticalHeader().setVisible(False)
        v.addWidget(self.tbl, 1)
        return w

    def _tactics_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Pre-goal tactical freezes (auto-saved)"))
        self.goal_list = QListWidget()
        self.goal_list.setIconSize(QSize(120, 68))
        self.goal_list.itemClicked.connect(self._show_goal)
        v.addWidget(self.goal_list, 1)
        self.goal_view = QLabel("Select a goal to view its tactical freeze.")
        self.goal_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.goal_view.setMinimumHeight(180)
        self.goal_view.setStyleSheet("background:#000; border:1px solid #262a33; border-radius:6px; color:#5a6172;")
        v.addWidget(self.goal_view)
        return w

    def _hline(self):
        ln = QFrame(); ln.setFrameShape(QFrame.Shape.HLine); ln.setStyleSheet("color:#262a33;")
        return ln

    # ---- source control ----
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open match video", str(ROOT),
            "Videos (*.mp4 *.mkv *.avi *.mov);;All files (*.*)")
        if path:
            self._start(path)

    def connect_camera(self):
        dlg = CameraDialog(self)
        if dlg.exec() and dlg.chosen_url:
            self._start(dlg.chosen_url)

    def _start(self, source: str):
        self.stop_pipeline()
        # reset the playback transport for the new source
        self._pos_frames = self._total_frames = 0
        self._scrubbing = False
        self.tp_box.setEnabled(False)
        self.tp_slider.setValue(0)
        self.tp_time.setText("00:00 / 00:00")
        self.tp_play.setText("⏸ Pause")
        self.tp_speed.setCurrentText("1×")
        self.video.setText("Loading models…  (first start takes a few seconds)")
        self.statusBar().showMessage(f"Source: {source}")
        self.worker = PipelineWorker(source)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_ready.connect(self._on_status)
        self.worker.log.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.worker.stopped.connect(self._on_stopped)
        self.worker.start()

    def stop_pipeline(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.send("q")
            if not self.worker.wait(4000):
                self.worker.terminate(); self.worker.wait(1000)
        self.worker = None

    def _cmd(self, ch: str):
        if self.worker is not None and self.worker.isRunning():
            self.worker.send(ch)
            if ch == "d":
                self.debug_on = not self.debug_on
                self.btn_dbg.setText(f"Debug: {'ON' if self.debug_on else 'OFF'}")
        else:
            self.statusBar().showMessage("Start a video or camera first.", 4000)

    # ---- unified tools (replaces the old .bat files) ----
    def _launch(self, args: list, label: str):
        try:
            flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            subprocess.Popen(args, cwd=str(ROOT), creationflags=flags)
            self.statusBar().showMessage(f"Launched: {label}  (in a new window)", 6000)
        except Exception as e:
            QMessageBox.warning(self, label, f"Could not launch:\n{e}")

    def toggle_phone_mode(self):
        # Stop if already running
        if self._phone_proc is not None and self._phone_proc.poll() is None:
            try: self._phone_proc.terminate()
            except Exception: pass
            self._phone_proc = None
            self._a_phone.setText("📱  Phone Mode")
            self.statusBar().showMessage("Phone server stopped.", 5000)
            return
        # Embedded view and the server both use the GPU — stop the local view first
        self.stop_pipeline()
        self.video.setText("Phone Mode is ON — watch on your phone.")
        try:
            import camera_discovery as cd
            ip = cd.local_ip()
        except Exception:
            ip = "<this-pc-ip>"
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            self._phone_proc = subprocess.Popen(
                [self._pyexe, "src/mobile_server.py", "--source", "handball.mp4", "--port", "8000"],
                cwd=str(ROOT), creationflags=flags)
            self._a_phone.setText("📱  Phone: ON (click to stop)")
            QMessageBox.information(self, "Phone Mode",
                f"Phone server starting…\n\nOn your phone (same WiFi) open:\n"
                f"    http://{ip}:8000\n\n"
                f"or open the installed Handball Studio app and tap “Auto-find”.")
        except Exception as e:
            QMessageBox.warning(self, "Phone Mode", f"Could not start server:\n{e}")

    def launch_label(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a video to label the ball in", str(ROOT),
            "Videos (*.mp4 *.mkv *.avi *.mov);;All files (*.*)")
        if path:
            self._launch([self._pyexe, "training/annotate_ball.py", path], "Ball labeling")

    def launch_train(self):
        if QMessageBox.question(self, "Train Model",
                "Retrain the ball model on your labelled frames?\n\n"
                "Runs in a new window; can take 15–45 min on the GPU. When it "
                "finishes it tells you how to swap the new model in."
                ) == QMessageBox.StandardButton.Yes:
            self._launch([self._pyexe, "training/train_ball.py", "--base", "handball_ball.pt"],
                         "Model training")

    def launch_calibrate(self):
        self._launch([self._pyexe, "calibrate_court.py"], "Court calibration")

    def launch_build_apk(self):
        bat = ROOT / "build_apk.bat"
        if bat.exists():
            self._launch(["cmd", "/c", str(bat)], "Phone app (APK) build")
        else:
            QMessageBox.information(self, "Build Phone App",
                                    "build_apk.bat not found.")

    # ---- signal slots ----
    def _on_frame(self, img: np.ndarray):
        pix = _bgr_to_qpix(img)
        if not pix.isNull():
            self.video.setPixmap(pix.scaled(
                self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def _update_transport(self, s: dict):
        """Drive the scrub slider + time label from the capture's true position.
        Uses pos/total frames (accurate even after a seek), not the processed
        frame counter. A live stream reports total=0 → transport stays disabled."""
        self._pos_frames = int(s.get("pos_frames", 0) or 0)
        total = int(s.get("total_frames", 0) or 0)
        self._src_fps = float(s.get("src_fps", 25.0) or 25.0)
        if total <= 0:
            return
        self._total_frames = total
        if not self.tp_box.isEnabled():
            self.tp_box.setEnabled(True)
        if not self._scrubbing:
            self.tp_slider.blockSignals(True)
            self.tp_slider.setValue(int(self._pos_frames / total * 1000))
            self.tp_slider.blockSignals(False)
        self.tp_time.setText(
            f"{self._fmt_time(self._pos_frames / self._src_fps)} / "
            f"{self._fmt_time(total / self._src_fps)}")

    def _on_status(self, s: dict):
        hud = s.get("hud", {})
        score = hud.get("score", {})
        self.lbl_score.setText(f"{score.get('team_a', 0)}  –  {score.get('team_b', 0)}")
        # keep the scout clock + roster in sync
        self._match_clock = hud.get("period_elapsed_s", self._match_clock)
        self._update_transport(s)
        _pl = s.get("players", [])
        _cur = [self.scout_players.item(i).text() for i in range(self.scout_players.count())]
        if [p.get("label") for p in _pl] != _cur:
            self.scout_players.clear()
            for p in _pl:
                it = QListWidgetItem(p.get("label", "?"))
                it.setData(Qt.ItemDataRole.UserRole, (p.get("label", "?"), p.get("team", "team_a")))
                it.setForeground(_qcolor(p.get("team", "unknown")))
                self.scout_players.addItem(it)
        poss = hud.get("possession")
        poss_txt = {"team_a": "Home", "team_b": "Away"}.get(poss, "—")
        self.stat["period"].setText(f"P{hud.get('period', 1)}  "
                                    f"{int(hud.get('period_elapsed_s', 0))//60:02d}:"
                                    f"{int(hud.get('period_elapsed_s', 0))%60:02d}")
        self.stat["possession"].setText(f"{poss_txt}  ({hud.get('possession_duration_s', 0):.0f}s)")
        sh = hud.get("shots", {}); fb = hud.get("fast_breaks", {}); ps = hud.get("passes", {})
        self.stat["shots"].setText(f"H {sh.get('team_a', 0)}   ·   A {sh.get('team_b', 0)}")
        self.stat["fast_breaks"].setText(f"H {fb.get('team_a', 0)}   ·   A {fb.get('team_b', 0)}")
        self.stat["passes"].setText(f"H {ps.get('team_a', 0)}   ·   A {ps.get('team_b', 0)}")
        self.stat["ball"].setText(str(s.get("ball_state", "—")))
        self.stat["fps"].setText(f"{s.get('fps', 0):.1f}")

        self.events.clear()
        for e in s.get("events", []):
            self.events.addItem(e)

        players = s.get("players", [])
        self.lbl_reid.setText(f"Persistent identities: {s.get('reid_known', 0)}   "
                              f"·   on court: {len(players)}")
        self.tbl.setRowCount(len(players))
        for i, p in enumerate(players):
            it = QTableWidgetItem(str(p.get("label", "?")))
            self.tbl.setItem(i, 0, it)
            team = p.get("team", "unknown")
            ti = QTableWidgetItem({"team_a": "Home", "team_b": "Away",
                                   "goalkeeper": "GK"}.get(team, "—"))
            ti.setForeground(_qcolor(team))
            self.tbl.setItem(i, 1, ti)

        gd = s.get("goals_dir")
        if gd:
            self.goals_dir = Path(gd)

    def _on_stopped(self):
        self.statusBar().showMessage("Pipeline stopped.", 5000)
        self.video.setText("Stopped.  Open a video or connect a camera to begin.")

    # ---- tactical playback ----
    def _refresh_goals(self):
        if not self.goals_dir or not self.goals_dir.exists():
            return
        for png in sorted(self.goals_dir.glob("goal_*.png")):
            key = png.name
            if key in self._goal_files:
                continue
            self._goal_files.add(key)
            it = QListWidgetItem(QIcon(str(png)), key.replace(".png", ""))
            it.setData(Qt.ItemDataRole.UserRole, str(png))
            self.goal_list.addItem(it)

    def _show_goal(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            pix = QPixmap(path)
            self.goal_view.setPixmap(pix.scaled(
                self.goal_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, e):
        self.stop_pipeline()
        if self._phone_proc is not None and self._phone_proc.poll() is None:
            try: self._phone_proc.terminate()
            except Exception: pass
        super().closeEvent(e)


def _qcolor(team: str):
    from PyQt6.QtGui import QColor
    return QColor(TEAM_COLORS.get(team, "#9aa0aa"))


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
