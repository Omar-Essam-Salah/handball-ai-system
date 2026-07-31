"""
Training Studio — one window to make the ball detector better on ANY video.

The whole flywheel as buttons:
    1. Open a video  →  2. Mine hard frames  →  3. Label the ball  →  4. Train

Pick any match you want, mine the frames where the current model struggles, click
the ball on those (the hardest views — that's exactly what makes it "perfect"),
and train TrackNet. Every video you add makes the model stronger across arenas.

Launch:  Train_Studio.bat   (or  .venv\\Scripts\\python.exe src\\train_studio.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYEXE = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYEXE).exists():
    PYEXE = sys.executable

try:
    from PyQt6.QtCore import Qt, QProcess
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QLineEdit, QFileDialog, QTextEdit, QListWidget,
        QListWidgetItem, QSpinBox, QGroupBox, QMessageBox,
    )
except Exception as e:
    print("PyQt6 required:  pip install PyQt6\n", e)
    sys.exit(1)

STYLE = """
QWidget { background:#1b1e24; color:#e6e9ef; font-size:13px; }
QPushButton { background:#262a33; border:1px solid #353b47; border-radius:7px; padding:9px 12px; }
QPushButton:hover { background:#2f3540; }
QPushButton#go { background:#2d6cf6; border:none; font-weight:700; }
QPushButton#go:hover { background:#3b78ff; }
QTextEdit, QListWidget, QLineEdit, QSpinBox { background:#12141a; border:1px solid #262a33; border-radius:6px; }
QGroupBox { border:1px solid #262a33; border-radius:8px; margin-top:10px; padding-top:10px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; color:#8b93a7; }
"""


class TrainStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handball — Training Studio")
        self.resize(1040, 680)
        self.proc: QProcess | None = None
        self.video = ""
        self._build()
        self._refresh_venues()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setSpacing(10); root.setContentsMargins(10, 10, 10, 10)

        left = QVBoxLayout(); left.setSpacing(8)

        # Step 1 — video + venue
        g1 = QGroupBox("1) Pick a video to learn from"); v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.ed_video = QLineEdit(); self.ed_video.setPlaceholderText("no video chosen — click Open Video")
        b_open = QPushButton("📂 Open Video"); b_open.clicked.connect(self.open_video)
        row.addWidget(self.ed_video, 1); row.addWidget(b_open); v1.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Arena/tag:"))
        self.ed_venue = QLineEdit(); self.ed_venue.setPlaceholderText("e.g. sweden, cairo_final …")
        row2.addWidget(self.ed_venue, 1)
        v1.addLayout(row2)
        left.addWidget(g1)

        # Step 2 — mine
        g2 = QGroupBox("2) Mine the hard frames (where the model struggles)"); v2 = QVBoxLayout(g2)
        rowm = QHBoxLayout()
        rowm.addWidget(QLabel("Keep:"))
        self.sp_keep = QSpinBox(); self.sp_keep.setRange(30, 1000); self.sp_keep.setValue(250)
        rowm.addWidget(self.sp_keep)
        _scan_lbl = QLabel("Scan window:")
        _scan_lbl.setToolTip("How many frames to scan from the start. A long match has\n"
                             "plenty of hard frames in the first window — no need to\n"
                             "scan all 77 min. 15000-30000 frames ~= 5-20 min.")
        rowm.addWidget(_scan_lbl)
        self.sp_scan = QSpinBox(); self.sp_scan.setRange(2000, 200000); self.sp_scan.setSingleStep(5000)
        self.sp_scan.setValue(20000); self.sp_scan.setToolTip(_scan_lbl.toolTip())
        rowm.addWidget(self.sp_scan)
        _start_lbl = QLabel("Start %:")
        _start_lbl.setToolTip("Where in the match this scan window begins.\n"
                              "Mine at 0, then 25, 50, 75 — each ACCUMULATES into the\n"
                              "same arena, covering a 2-hour match in fast chunks.")
        rowm.addWidget(_start_lbl)
        self.sp_start = QSpinBox(); self.sp_start.setRange(0, 95); self.sp_start.setValue(0)
        self.sp_start.setSingleStep(5); self.sp_start.setToolTip(_start_lbl.toolTip())
        rowm.addWidget(self.sp_start)
        rowm.addStretch(1)
        self.b_mine = QPushButton("⛏  Mine Hard Frames"); self.b_mine.setObjectName("go")
        self.b_mine.clicked.connect(self.mine)
        rowm.addWidget(self.b_mine)
        v2.addLayout(rowm)
        left.addWidget(g2)

        # Step 3 — label
        g3 = QGroupBox("3) Label the ball (click the ball in the hardest views)"); v3 = QVBoxLayout(g3)
        self.b_label = QPushButton("🎯  Label Selected Arena"); self.b_label.clicked.connect(self.label)
        v3.addWidget(QLabel("Pick an arena on the right, then:"))
        v3.addWidget(self.b_label)
        left.addWidget(g3)

        # Step 4 — train
        g4 = QGroupBox("4) Train the model on everything labelled"); v4 = QVBoxLayout(g4)
        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("Epochs:"))
        self.sp_ep = QSpinBox(); self.sp_ep.setRange(5, 200); self.sp_ep.setValue(30)
        rowt.addWidget(self.sp_ep); rowt.addStretch(1)
        self.b_train = QPushButton("🧠  Train TrackNet"); self.b_train.setObjectName("go")
        self.b_train.clicked.connect(self.train)
        rowt.addWidget(self.b_train)
        v4.addLayout(rowt)
        self.b_stop = QPushButton("⏹ Stop current job"); self.b_stop.clicked.connect(self.stop)
        v4.addWidget(self.b_stop)
        left.addWidget(g4)
        left.addStretch(1)
        root.addLayout(left, 3)

        # Right — venues + log
        right = QVBoxLayout(); right.setSpacing(8)
        gv = QGroupBox("Arenas (checked = used for training)"); vv = QVBoxLayout(gv)
        self.lst = QListWidget(); vv.addWidget(self.lst)
        b_ref = QPushButton("↻ Refresh"); b_ref.clicked.connect(self._refresh_venues); vv.addWidget(b_ref)
        right.addWidget(gv, 1)
        right.addWidget(QLabel("Log"))
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        right.addWidget(self.log, 2)
        root.addLayout(right, 4)

        self.setStyleSheet(STYLE)

    # ── data ────────────────────────────────────────────────────────────
    def _refresh_venues(self):
        self.lst.clear()
        base = ROOT / "training" / "to_label"
        for d in sorted(p for p in base.glob("*") if p.is_dir()):
            if d.name.endswith("_auto"):
                continue                      # auto-label artifacts: no frames to label
            jpgs = len(list(d.glob("*.jpg")))
            if jpgs == 0:
                continue                      # nothing minable/labelable here yet
            labeled = 0
            lf = d / "labels.json"
            if lf.exists():
                try:
                    import json
                    labeled = sum(1 for v in json.loads(lf.read_text()).values()
                                  if isinstance(v, dict))
                except Exception:
                    pass
            it = QListWidgetItem(f"{d.name}   ({labeled}/{jpgs} labelled)")
            it.setData(Qt.ItemDataRole.UserRole, d.name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if labeled > 0 else Qt.CheckState.Unchecked)
            self.lst.addItem(it)
        if self.lst.count() and self.lst.currentRow() < 0:
            self.lst.setCurrentRow(0)        # auto-highlight so Label always has a target

    # ── actions ─────────────────────────────────────────────────────────
    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose a match video", str(ROOT),
                                           "Videos (*.mp4 *.mkv *.avi *.mov)")
        if p:
            self.video = p
            self.ed_video.setText(p)
            if not self.ed_venue.text().strip():
                self.ed_venue.setText(Path(p).stem.replace(" ", "_").lower())

    def _busy(self) -> bool:
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Busy", "A job is already running. Wait or Stop it.")
            return True
        return False

    def _run(self, args, label):
        if self._busy():
            return
        self.log.append(f"\n$ {label}\n")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_out)
        self.proc.finished.connect(lambda *_: (self.log.append("\n[done]\n"),
                                               self._refresh_venues()))
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.start(PYEXE, args)

    def _on_out(self):
        if self.proc:
            data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
            self.log.moveCursor(self.log.textCursor().MoveOperation.End)
            self.log.insertPlainText(data)
            self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def mine(self):
        if not self.video:
            QMessageBox.information(self, "Pick a video", "Open a video first."); return
        venue = self._venue_tag() or Path(self.video).stem.lower()
        self.ed_venue.setText(venue)          # show the cleaned tag back to the user
        self._run([str(ROOT / "training" / "mine_hard_frames.py"),
                   "--source", self.video, "--venue", venue,
                   "--stride", "20", "--max-keep", str(self.sp_keep.value()),
                   "--max-scan", str(self.sp_scan.value()),
                   "--start-frac", f"{self.sp_start.value() / 100.0:.3f}"],
                  f"Mining {Path(self.video).name} @ {self.sp_start.value()}% -> {venue}")

    def _venue_tag(self) -> str:
        """The arena the Mine/Label buttons act on: the tag box (sanitised) for
        the video you opened; otherwise the arena highlighted in the list."""
        import re
        raw = self.ed_venue.text().strip()
        if raw:
            return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        return self._selected_venue() or ""

    def _selected_venue(self):
        it = self.lst.currentItem()
        if it is None:                       # fall back: the only checked, or only, arena
            checked = [self.lst.item(i) for i in range(self.lst.count())
                       if self.lst.item(i).checkState() == Qt.CheckState.Checked]
            if len(checked) == 1:
                it = checked[0]
            elif self.lst.count() == 1:
                it = self.lst.item(0)
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def label(self):
        v = self._venue_tag()
        if not v:
            QMessageBox.information(self, "Nothing to label yet",
                "Open a video + set a tag (then Mine), or pick an arena in the list."); return
        vdir = ROOT / "training" / "to_label" / v
        if not list(vdir.glob("*.jpg")):
            QMessageBox.information(self, "Mine first",
                f"No frames for '{v}' yet.\n\nClick '⛏ Mine Hard Frames' first — that extracts "
                f"the frames from your video. THEN you can label them."); return
        # the labeler opens its own OpenCV window; run it detached-ish via QProcess
        self._run([str(ROOT / "training" / "label_frames.py"),
                   "--dir", str(ROOT / "training" / "to_label" / v)],
                  f"Labelling {v}  (click the ball; SPACE=next, X=no-ball, Q=quit)")

    def train(self):
        venues = []
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                venues.append(it.data(Qt.ItemDataRole.UserRole))
        venues = [v for v in venues if not v.endswith("_auto")] or venues
        if not venues:
            QMessageBox.information(self, "Nothing to train", "Check at least one labelled arena."); return
        self._run(["-m", "training.tracknet.train", "--venues", *venues,
                   "--epochs", str(self.sp_ep.value())],
                  f"Training TrackNet on: {', '.join(venues)}")

    def stop(self):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()
            self.log.append("\n[stopped]\n")


def main():
    app = QApplication(sys.argv)
    w = TrainStudio(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
