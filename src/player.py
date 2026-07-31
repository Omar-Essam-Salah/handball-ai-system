"""Main window: video player + hotkey tagger + clip export."""
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from clipper import export_clips
from summary import build_summary
from tagger import Event, TagStore

HOTKEYS_PATH = Path(__file__).parent.parent / "config" / "hotkeys.yaml"


def fmt_ms(ms: int) -> str:
    if ms is None or ms < 0:
        ms = 0
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Handball Tagger — Phase 1 MVP")
        self.resize(1280, 780)

        # --- media ---
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)

        # --- controls ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.seek)

        self.time_label = QLabel("00:00 / 00:00")
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.open_btn = QPushButton("Open…")
        self.open_btn.clicked.connect(self.open_video)
        self.export_btn = QPushButton("Export Clips")
        self.export_btn.clicked.connect(self.do_export)
        self.summary_btn = QPushButton("Summary CSV")
        self.summary_btn.clicked.connect(self.do_summary)

        controls = QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(self.export_btn)
        controls.addWidget(self.summary_btn)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.addWidget(self.video_widget, 1)
        left_l.addLayout(controls)

        self.tag_list = QListWidget()
        self.tag_list.itemDoubleClicked.connect(self.jump_to_tag)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.tag_list)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # --- signals ---
        self.player.positionChanged.connect(self.on_pos)
        self.player.durationChanged.connect(self.on_dur)

        # --- state ---
        self.tags: TagStore | None = None
        self.video_path: Path | None = None

        self.hotkeys = self._load_hotkeys()
        self._bind_shortcuts()

        self.statusBar().showMessage(
            "Open a video (Ctrl+O).  Space=play/pause  ←/→ = 5s  Shift+←/→ = 1s  Ctrl+Z=undo"
        )

    # ---------- setup ----------
    def _load_hotkeys(self) -> dict:
        if HOTKEYS_PATH.exists():
            return yaml.safe_load(HOTKEYS_PATH.read_text(encoding="utf-8")) or {}
        return {}

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.nudge(-5000))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.nudge(+5000))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self.nudge(-1000))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self.nudge(+1000))
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_video)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_tag)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.do_export)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.do_summary)

        for key, spec in (self.hotkeys.get("events") or {}).items():
            event_type = spec["event"]
            team = spec.get("team", "red")
            note = spec.get("note", "")
            QShortcut(
                QKeySequence(key),
                self,
                activated=lambda e=event_type, t=team, n=note: self.tag(e, t, n),
            )

    # ---------- file ----------
    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open match video", "", "Video (*.mp4 *.mkv *.mov *.avi)"
        )
        if not path:
            return
        self.video_path = Path(path)
        self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))
        self.tags = TagStore(self.video_path)
        self.refresh_tags()
        self.player.play()
        self.play_btn.setText("Pause")
        self.statusBar().showMessage(f"Loaded {self.video_path.name}")

    # ---------- playback ----------
    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setText("Play")
        else:
            self.player.play()
            self.play_btn.setText("Pause")

    def seek(self, ms: int) -> None:
        self.player.setPosition(ms)

    def nudge(self, delta_ms: int) -> None:
        self.player.setPosition(max(0, self.player.position() + delta_ms))

    def on_pos(self, ms: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(False)
        self.time_label.setText(f"{fmt_ms(ms)} / {fmt_ms(self.player.duration())}")

    def on_dur(self, ms: int) -> None:
        self.slider.setRange(0, ms)

    # ---------- tagging ----------
    def tag(self, event_type: str, team: str, note: str = "") -> None:
        if not self.tags:
            self.statusBar().showMessage("Open a video first.")
            return
        ev = Event(
            timestamp_ms=int(self.player.position()),
            event_type=event_type,
            team=team,
            note=note,
        )
        self.tags.add(ev)
        self.tags.save()
        self.refresh_tags()
        self.statusBar().showMessage(
            f"Tagged {event_type} [{team}] @ {fmt_ms(ev.timestamp_ms)}"
        )

    def undo_tag(self) -> None:
        if self.tags and self.tags.pop():
            self.tags.save()
            self.refresh_tags()
            self.statusBar().showMessage("Undid last tag.")

    def refresh_tags(self) -> None:
        self.tag_list.clear()
        if not self.tags:
            return
        for i, ev in enumerate(self.tags.events):
            label = f"{i + 1:03d}  {fmt_ms(ev.timestamp_ms)}  [{ev.team}]  {ev.event_type}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ev.timestamp_ms)
            self.tag_list.addItem(item)

    def jump_to_tag(self, item: QListWidgetItem) -> None:
        ms = item.data(Qt.ItemDataRole.UserRole)
        if ms is not None:
            self.player.setPosition(int(ms))

    # ---------- export ----------
    def do_export(self) -> None:
        if not self.video_path or not self.tags or not self.tags.events:
            QMessageBox.information(self, "Export", "No video or tags.")
            return
        clip_cfg = self.hotkeys.get("clip") or {}
        pre = int(clip_cfg.get("pre_roll_s", 5))
        post = int(clip_cfg.get("post_roll_s", 3))
        out_dir = self.video_path.parent.parent / "clips" / self.video_path.stem
        try:
            n = export_clips(self.video_path, self.tags.events, out_dir, pre, post)
            QMessageBox.information(self, "Export", f"Exported {n} clips to:\n{out_dir}")
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "ffmpeg missing",
                "ffmpeg not found on PATH. Install from https://ffmpeg.org and retry.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def do_summary(self) -> None:
        if not self.video_path or not self.tags:
            return
        out = self.video_path.parent.parent / "reports"
        csv_path = out / f"{self.video_path.stem}_summary.csv"
        build_summary(self.tags.events, csv_path)
        QMessageBox.information(self, "Summary", f"Wrote:\n{csv_path}")
