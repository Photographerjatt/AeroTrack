#!/usr/bin/env python3
"""
AeroTrack - Point Tracking Stabilizer
=====================================

Click on an aircraft, track it forward, and the app keeps that point locked at
the centre of the output frame (the frame moves, the dot / circle / crop box
stay still) - like the point tracker in Premiere / After Effects / DaVinci.

Install:   pip install PySide6 opencv-python numpy
Optional:  pip install imageio-ffmpeg     (H.264 export + original audio)
"""

import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# POINT TRACKER  (template matching + drift correction)
# ============================================================

class PointTracker:
    """
    Tracks ONE point by following the small image patch around it.

    Why this replaces the old optical-flow feature cloud:
      * Feature clouds lose points every frame and are never replenished, so the
        track eventually dies (or latches onto trees / ground).
      * A patch tracker follows the *appearance* of the aircraft itself, has
        sub-pixel precision, and can recover because it keeps the original
        patch as a drift-free anchor.
    """

    ANCHOR_SCORE = 0.80    # original patch still matches this well -> snap to it (no drift)
    UPDATE_SCORE = 0.70    # only learn the new appearance when the match is confident
    UPDATE_RATE = 0.15     # how fast the adaptive patch follows banking / lighting changes

    def __init__(self):
        self.patch_half = 24        # patch is (2*half+1) px square
        self.search_half = 90       # how far (px) the point may move between frames
        self.min_score = 0.60       # below this match score the track is declared lost
        self.ready = False
        self.pos = (0.0, 0.0)
        self.template_init = None
        self.template_adapt = None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _blur(img):
        return cv2.GaussianBlur(img, (3, 3), 0.8)

    def _patch(self, gray, x, y):
        """Sub-pixel patch centred on (x, y), float32, lightly denoised."""
        size = 2 * self.patch_half + 1
        p = cv2.getRectSubPix(
            gray, (size + 2, size + 2), (float(x), float(y)), patchType=cv2.CV_32F
        )
        return self._blur(p)[1:-1, 1:-1].copy()

    @staticmethod
    def _match(gray, template, cx, cy, half_search):
        """Best match of `template` within +-half_search px of (cx, cy)."""
        H, W = gray.shape[:2]
        th, tw = template.shape[:2]
        ph = tw // 2
        reach = half_search + ph + 1
        icx, icy = int(round(cx)), int(round(cy))

        x0, y0 = max(0, icx - reach), max(0, icy - reach)
        x1, y1 = min(W, icx + reach + 1), min(H, icy + reach + 1)
        if x1 - x0 <= tw or y1 - y0 <= th:
            return None

        roi = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (3, 3), 0.8)
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

        _, score, _, (mx, my) = cv2.minMaxLoc(res)

        # parabolic sub-pixel refinement of the correlation peak
        sx = sy = 0.0
        if 0 < mx < res.shape[1] - 1:
            l, c, r = res[my, mx - 1], res[my, mx], res[my, mx + 1]
            d = l - 2.0 * c + r
            if d < -1e-9:
                sx = float(np.clip((l - r) / (2.0 * d), -1.0, 1.0))
        if 0 < my < res.shape[0] - 1:
            u, c, b = res[my - 1, mx], res[my, mx], res[my + 1, mx]
            d = u - 2.0 * c + b
            if d < -1e-9:
                sy = float(np.clip((u - b) / (2.0 * d), -1.0, 1.0))

        return (x0 + mx + ph + sx, y0 + my + ph + sy, float(score))

    # -------------------------------------------------------------------- API

    def init(self, gray, x, y):
        """Start tracking at (x, y). Returns the patch contrast (std-dev)."""
        self.pos = (float(x), float(y))
        t = self._patch(gray, x, y)
        self.template_init = t
        self.template_adapt = t.copy()
        self.ready = True
        return float(t.std())

    def set_position(self, x, y):
        self.pos = (float(x), float(y))

    def update(self, gray):
        """
        Track into the next frame.
        Returns (x, y, score) or (None, None, best_score) when the track is lost.
        """
        x, y = self.pos

        m = self._match(gray, self.template_adapt, x, y, self.search_half)

        # Adaptive patch failed -> try the original patch (recovers from drift
        # or a brief appearance change).
        if m is None or m[2] < self.min_score:
            m2 = self._match(gray, self.template_init, x, y, self.search_half)
            if m2 is not None and (m is None or m2[2] > m[2]):
                m = m2

        if m is None or m[2] < self.min_score:
            return None, None, (m[2] if m is not None else 0.0)

        nx, ny, score = m

        # Drift correction: if the ORIGINAL patch still fits nearby, trust it.
        a = self._match(gray, self.template_init, nx, ny, 4)
        if a is not None and a[2] >= self.ANCHOR_SCORE:
            nx, ny = a[0], a[1]
            score = max(score, a[2])

        # Slowly learn the new appearance (aircraft banking, lighting, scale).
        if score >= self.UPDATE_SCORE:
            p = self._patch(gray, nx, ny)
            self.template_adapt = (
                (1.0 - self.UPDATE_RATE) * self.template_adapt + self.UPDATE_RATE * p
            )

        self.pos = (nx, ny)
        return nx, ny, float(min(1.0, max(0.0, score)))


# ============================================================
# VIDEO SINKS (export)
# ============================================================

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class CvSink:
    """OpenCV VideoWriter fallback (no audio)."""

    name = "OpenCV (no audio)"

    def __init__(self, path, w, h, fps):
        self.writer = None
        for code in ("avc1", "mp4v"):
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, (w, h))
            if wr.isOpened():
                self.writer = wr
                break
            wr.release()

    @property
    def ok(self):
        return self.writer is not None

    def write(self, frame):
        self.writer.write(frame)
        return True

    def close(self):
        if self.writer:
            self.writer.release()
        return True, ""

    def abort(self):
        self.close()


class FFmpegSink:
    """Pipes frames to ffmpeg: H.264 (CRF 17) + original audio for the tracked range."""

    name = "FFmpeg H.264 + audio"

    def __init__(self, exe, out_path, w, h, fps, src_path, start_sec, duration_sec):
        rate = Fraction(fps).limit_denominator(1001)
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-framerate", f"{rate.numerator}/{rate.denominator}",
            "-i", "-",
            "-ss", f"{start_sec:.6f}", "-t", f"{duration_sec:.6f}", "-i", src_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            out_path,
        ]
        self._err = tempfile.TemporaryFile()
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._err, **kwargs
            )
        except Exception:
            self.proc = None

    @property
    def ok(self):
        return self.proc is not None

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        if rc == 0:
            return True, ""
        self._err.seek(0)
        return False, self._err.read().decode("utf-8", "replace")[-800:]

    def abort(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


# ============================================================
# VIDEO CANVAS
# ============================================================

class VideoCanvas(QWidget):
    """Draws a pre-rendered QImage 1:1 (so mouse coordinates are exact)."""

    clicked = Signal(float, float)   # device-pixel coordinates inside the image
    resized = Signal()

    def __init__(self):
        super().__init__()
        self.image = None
        self.setMinimumSize(800, 450)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, qimage):
        self.image = qimage
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101010"))
        if self.image is not None:
            p.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            dpr = self.devicePixelRatioF()
            self.clicked.emit(event.position().x() * dpr, event.position().y() * dpr)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


# ============================================================
# MAIN APPLICATION
# ============================================================

class AeroTrack(QMainWindow):

    OUTPUT_SHORT_SIDE = {"720p": 720, "1080p": 1080, "1440p": 1440, "2160p (4K)": 2160}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroTrack - Point Tracking Stabilizer")
        self.resize(1400, 900)

        # video  (NOTE: never name these self.width / self.height - that would
        # shadow QWidget.width() / height())
        self.cap = None
        self.video_path = None
        self.fps = 30.0
        self.frame_count = 0
        self.video_w = 0
        self.video_h = 0
        self.current_frame = 0
        self.current_image = None
        self._next_read = 0            # index of the frame cap.read() will return next

        # tracking
        self.tracker = PointTracker()
        self.track_history = {}        # frame -> (x, y, confidence)
        self.track_version = 0
        self._path_cache = None
        self.tracking = False
        self.playing = False
        self.selecting = False
        self.play_end = 0

        # view transform  source px -> canvas px : canvas = src * s + (tx, ty)
        self.view = (1.0, 0.0, 0.0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.track_one_frame)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_one_frame)

        self.build_ui()

        for seq, step in (("Left", -1), ("Right", 1), ("Shift+Left", -5), ("Shift+Right", 5)):
            QShortcut(QKeySequence(seq), self).activated.connect(
                lambda s=step: self.seek_relative(s)
            )

    # ========================================================
    # UI
    # ========================================================

    def _button(self, text, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        return b

    def _slider_row(self, layout, title, lo, hi, value, fmt, on_change):
        layout.addWidget(QLabel(title))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(fmt(value))
        label.setMinimumWidth(75)

        def changed(v):
            label.setText(fmt(v))
            on_change(v)

        slider.valueChanged.connect(changed)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        return slider

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # top bar
        top = QHBoxLayout()
        top.addWidget(self._button("OPEN VIDEO", self.open_video))
        self.status_label = QLabel("No video loaded")
        top.addWidget(self.status_label, 1)
        main.addLayout(top)

        # canvas
        self.canvas = VideoCanvas()
        self.canvas.clicked.connect(self.canvas_clicked)
        self.canvas.resized.connect(self.show_frame)
        main.addWidget(self.canvas, 1)

        self.frame_label = QLabel("Frame: -")
        main.addWidget(self.frame_label)

        # timeline
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.setFocusPolicy(Qt.NoFocus)
        self.timeline.sliderPressed.connect(self.stop_all)
        self.timeline.valueChanged.connect(self.timeline_changed)
        main.addWidget(self.timeline)

        # transport / tracking controls
        row = QHBoxLayout()
        row.addWidget(self._button("<< 5", lambda: self.seek_relative(-5)))
        row.addWidget(self._button("< 1", lambda: self.seek_relative(-1)))
        row.addWidget(self._button("1 >", lambda: self.seek_relative(1)))
        row.addWidget(self._button("5 >>", lambda: self.seek_relative(5)))
        row.addWidget(self._button("SELECT TRACKING POINT", self.select_tracking_mode))
        row.addWidget(self._button("▶ TRACK FORWARD", self.start_tracking))
        row.addWidget(self._button("■ STOP", self.stop_all))
        self.play_button = self._button("▶ PLAY PREVIEW", self.toggle_play)
        row.addWidget(self.play_button)
        row.addWidget(self._button("CLEAR TRACK", self.clear_track))
        main.addLayout(row)

        # tracker settings
        trow = QHBoxLayout()
        self.patch_slider = self._slider_row(
            trow, "Patch:", 8, 100, 24, lambda v: f"{2 * v + 1}px", lambda v: None
        )
        self.search_slider = self._slider_row(
            trow, "Search range:", 20, 500, 90, lambda v: f"±{v}px",
            lambda v: setattr(self.tracker, "search_half", v),
        )
        self.score_slider = self._slider_row(
            trow, "Min match:", 20, 90, 60, lambda v: f"{v}%",
            lambda v: setattr(self.tracker, "min_score", v / 100.0),
        )
        main.addLayout(trow)

        # output settings
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["Original", "16:9", "9:16", "4:3", "1:1", "3:2"])
        self.aspect_combo.currentIndexChanged.connect(lambda *_: self.show_frame())
        orow.addWidget(self.aspect_combo)

        self.zoom_slider = self._slider_row(
            orow, "Zoom:", 100, 400, 130, lambda v: f"{v / 100:.2f}x",
            lambda v: self.show_frame(),
        )
        self.smooth_slider = self._slider_row(
            orow, "Smoothing:", 0, 15, 0, lambda v: "off (exact lock)" if v == 0 else f"±{v} fr",
            lambda v: self.show_frame(),
        )

        orow.addWidget(QLabel("Edges:"))
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Exact lock (black if needed)", "Clamp to video (no black)"])
        self.edge_combo.currentIndexChanged.connect(lambda *_: self.show_frame())
        orow.addWidget(self.edge_combo)

        orow.addWidget(QLabel("Output:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(list(self.OUTPUT_SHORT_SIDE) + ["Native crop size"])
        self.res_combo.setCurrentText("1080p")
        orow.addWidget(self.res_combo)

        orow.addWidget(self._button("GENERATE OUTPUT VIDEO", self.export_video))
        main.addLayout(orow)

        self.help_label = QLabel(
            "1) OPEN VIDEO   2) SELECT TRACKING POINT and click the aircraft   "
            "3) TRACK FORWARD.  The preview then shows the stabilised view: the dot, circle and crop box stay "
            "fixed while the frame moves.  If tracking is lost, the app stops on the frame where it failed - "
            "just select the aircraft again and continue (or step back, re-select and re-track).  "
            "Arrow keys step frames (Shift = 5).  Raise Zoom if black borders appear."
        )
        self.help_label.setWordWrap(True)
        main.addWidget(self.help_label)

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.*)")
        if path:
            self.load_video(path)

    def load_video(self, path):
        self.stop_all()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open this video.")
            return False

        ok, frame = cap.read()
        if not ok:
            cap.release()
            QMessageBox.critical(self, "Error", "Video opened but the first frame could not be decoded.")
            return False

        if self.cap:
            self.cap.release()
        self.cap = cap
        self._next_read = 1
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.video_h, self.video_w = frame.shape[:2]   # real (rotation-corrected) size
        self.current_frame = 0
        self.current_image = frame
        self.selecting = False

        self.track_history.clear()
        self.track_version += 1
        self.tracker = PointTracker()
        self.tracker.search_half = self.search_slider.value()
        self.tracker.min_score = self.score_slider.value() / 100.0

        # sensible defaults for this resolution
        self.patch_slider.setValue(int(np.clip(round(self.video_h * 0.025), 12, 64)))
        self.search_slider.setValue(int(np.clip(round(self.video_h * 0.09), 40, 300)))

        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.timeline.setEnabled(True)
        self._set_timeline(0)

        self.status_label.setText(
            f"{Path(path).name} | {self.video_w}x{self.video_h} | "
            f"{self.fps:.2f} FPS | {self.frame_count} frames"
        )
        self.show_frame()
        return True

    # ========================================================
    # FRAME ACCESS
    # ========================================================

    def read_frame(self, n):
        """Sequential reads are fast; we only seek when we have to."""
        if self.cap is None or n < 0 or n >= self.frame_count:
            return None
        if n != self._next_read:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self.cap.read()
        if not ok:
            self._next_read = -1
            return None
        self._next_read = n + 1
        return frame

    def _set_timeline(self, value):
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(value))
        self.timeline.blockSignals(False)

    # ========================================================
    # SELECT TRACKING POINT
    # ========================================================

    def select_tracking_mode(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Tracking", "Open a video first.")
            return
        self.stop_all()
        self.selecting = True
        self.canvas.setCursor(Qt.CrossCursor)
        self.status_label.setText("CLICK THE AIRCRAFT WHERE YOU WANT TO TRACK")
        self.show_frame()

    def canvas_clicked(self, mouse_x, mouse_y):
        if self.current_image is None or not self.selecting:
            return
        s, tx, ty = self.view
        x = (mouse_x - tx) / s
        y = (mouse_y - ty) / s
        if not (0 <= x < self.video_w and 0 <= y < self.video_h):
            return
        self.initialize_tracker(x, y)

    def initialize_tracker(self, x, y):
        self.stop_all()

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.tracker.patch_half = self.patch_slider.value()
        self.tracker.search_half = self.search_slider.value()
        self.tracker.min_score = self.score_slider.value() / 100.0
        contrast = self.tracker.init(gray, x, y)

        # Keep everything tracked BEFORE this frame, drop what comes after
        # (it is about to be re-tracked from this new point).
        self.track_history = {
            f: v for f, v in self.track_history.items() if f < self.current_frame
        }
        self.track_history[self.current_frame] = (float(x), float(y), 1.0)
        self.track_version += 1

        self.selecting = False
        self.canvas.setCursor(Qt.ArrowCursor)

        msg = f"Point selected on frame {self.current_frame} - press TRACK FORWARD"
        if contrast < 3.0:
            msg += "  (WARNING: this patch has very little detail - click on the aircraft itself)"
        self.status_label.setText(msg)
        self.show_frame()

    # ========================================================
    # TRACKING
    # ========================================================

    def start_tracking(self):
        if self.cap is None:
            QMessageBox.warning(self, "Tracking", "Open a video first.")
            return
        if not self.tracker.ready:
            QMessageBox.warning(self, "Tracking", "Select a tracking point first.")
            return
        if self.current_frame not in self.track_history:
            QMessageBox.warning(
                self, "Tracking",
                "There is no tracked point on this frame.\n\n"
                "Click SELECT TRACKING POINT and click the aircraft on this frame, "
                "or move to a frame that already has the tracking circle.",
            )
            return
        if self.current_frame >= self.frame_count - 1:
            return

        self.stop_all()

        # Resume from the stored position of this frame (works after stepping back).
        x, y, _ = self.track_history[self.current_frame]
        self.tracker.set_position(x, y)
        self.tracker.patch_half = self.tracker.template_init.shape[0] // 2
        self.tracker.search_half = self.search_slider.value()
        self.tracker.min_score = self.score_slider.value() / 100.0

        self.track_history = {
            f: v for f, v in self.track_history.items() if f <= self.current_frame
        }
        self.track_version += 1

        self.selecting = False
        self.tracking = True
        self.status_label.setText("TRACKING FORWARD...")
        self.timer.start(1)    # as fast as the machine can go

    def stop_all(self):
        self.tracking = False
        self.playing = False
        self.timer.stop()
        self.play_timer.stop()
        self.play_button.setText("▶ PLAY PREVIEW")

    def track_one_frame(self):
        if not self.tracking or self.cap is None:
            return

        n = self.current_frame + 1
        if n >= self.frame_count:
            self.stop_all()
            self.status_label.setText("Reached end of video.")
            self.show_frame()
            return

        frame = self.read_frame(n)
        if frame is None:
            self.stop_all()
            self.status_label.setText("Could not decode next frame.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, score = self.tracker.update(gray)

        if x is None:
            # Show the frame where it failed so the user can click the aircraft right here.
            self.current_frame = n
            self.current_image = frame
            self._set_timeline(n)
            self.stop_all()
            self.status_label.setText(
                f"TRACKING LOST at frame {n} (match {score * 100:.0f}%). "
                "Click SELECT TRACKING POINT, click the aircraft, then TRACK FORWARD "
                "(or raise 'Search range' / lower 'Min match')."
            )
            self.show_frame()
            return

        self.current_frame = n
        self.current_image = frame
        self.track_history[n] = (x, y, score)
        self.track_version += 1
        self._set_timeline(n)

        self.status_label.setText(
            f"TRACKING | Frame {n} | Confidence: {score * 100:.0f}%"
        )
        self.show_frame()

    def clear_track(self):
        self.stop_all()
        self.track_history.clear()
        self.track_version += 1
        self.tracker.ready = False
        self.selecting = False
        self.canvas.setCursor(Qt.ArrowCursor)
        self.status_label.setText("Tracking data cleared.")
        self.show_frame()

    # ========================================================
    # SEEKING / PLAYBACK  (never deletes tracking data)
    # ========================================================

    def seek_relative(self, amount):
        if self.cap is None:
            return
        self.seek_frame(self.current_frame + amount)

    def timeline_changed(self, value):
        self.seek_frame(value)

    def seek_frame(self, frame_number):
        if self.cap is None:
            return
        frame_number = max(0, min(self.frame_count - 1, int(frame_number)))
        self.stop_all()
        frame = self.read_frame(frame_number)
        if frame is None:
            return
        self.current_frame = frame_number
        self.current_image = frame
        self._set_timeline(frame_number)
        self.show_frame()

    def toggle_play(self):
        if self.playing:
            self.stop_all()
            return
        if self.cap is None:
            return
        self.stop_all()

        if self.track_history:
            first, last = min(self.track_history), max(self.track_history)
            self.play_end = last
            if not (first <= self.current_frame < last):
                self.seek_frame(first)
        else:
            self.play_end = self.frame_count - 1
            if self.current_frame >= self.play_end:
                self.seek_frame(0)

        self.playing = True
        self.play_button.setText("■ PAUSE PREVIEW")
        self.play_timer.start(max(5, int(1000 / self.fps)))

    def play_one_frame(self):
        n = self.current_frame + 1
        if not self.playing or n > self.play_end:
            self.stop_all()
            return
        frame = self.read_frame(n)
        if frame is None:
            self.stop_all()
            return
        self.current_frame = n
        self.current_image = frame
        self._set_timeline(n)
        self.show_frame()

    # ========================================================
    # TRACK PATH / CROP GEOMETRY
    # ========================================================

    def _path(self):
        """Interpolated (and optionally smoothed) track for every frame first..last."""
        key = (self.track_version, self.smooth_slider.value())
        if self._path_cache is not None and self._path_cache[0] == key:
            return self._path_cache[1]

        frames = np.array(sorted(self.track_history), dtype=np.int64)
        xs = np.array([self.track_history[f][0] for f in frames], dtype=np.float64)
        ys = np.array([self.track_history[f][1] for f in frames], dtype=np.float64)
        all_f = np.arange(frames[0], frames[-1] + 1)
        px = np.interp(all_f, frames, xs)      # linear fill across any gaps
        py = np.interp(all_f, frames, ys)

        radius = self.smooth_slider.value()
        if radius > 0:
            def smooth(a):
                padded = np.pad(a, radius, mode="edge")
                k = np.ones(2 * radius + 1) / (2 * radius + 1)
                return np.convolve(padded, k, mode="valid")
            px, py = smooth(px), smooth(py)

        data = (int(frames[0]), px, py)
        self._path_cache = (key, data)
        return data

    def tracked_point(self, frame_number):
        """The point the output is locked to on this frame, or None."""
        if not self.track_history:
            return None
        if self.tracking:                       # live: raw values (cheap)
            v = self.track_history.get(frame_number)
            return (v[0], v[1]) if v else None
        f0, px, py = self._path()
        i = frame_number - f0
        if 0 <= i < len(px):
            return float(px[i]), float(py[i])
        return None

    def get_aspect_ratio(self):
        value = self.aspect_combo.currentText()
        if value == "Original":
            return self.video_w / self.video_h if self.video_h else 16 / 9
        a, b = value.split(":")
        return float(a) / float(b)

    def compute_crop_size(self):
        """Crop size in SOURCE pixels for the current aspect + zoom."""
        aspect = self.get_aspect_ratio()
        zoom = self.zoom_slider.value() / 100.0
        cw = self.video_w / zoom
        ch = cw / aspect
        if ch > self.video_h / zoom:
            ch = self.video_h / zoom
            cw = ch * aspect
        return max(16.0, cw), max(16.0, ch)

    def crop_center(self, px, py, cw, ch):
        """Where the crop box is centred for tracked point (px, py)."""
        if self.edge_combo.currentIndex() == 1:           # clamp: never leave the video
            px = min(max(px, cw / 2.0), self.video_w - cw / 2.0)
            py = min(max(py, ch / 2.0), self.video_h - ch / 2.0)
        return px, py

    # ========================================================
    # DISPLAY  (stabilised preview)
    # ========================================================

    def show_frame(self):
        if self.current_image is None:
            return

        frame = self.current_image
        H, W = frame.shape[:2]
        dpr = self.canvas.devicePixelRatioF()
        cvw = max(16, int(self.canvas.width() * dpr))
        cvh = max(16, int(self.canvas.height() * dpr))

        # scale that fits the whole frame in the canvas
        s = min(cvw / W, cvh / H)
        nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        s = nw / W
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        crop_w, crop_h = self.compute_crop_size()
        f = self.current_frame
        raw = self.track_history.get(f)
        point = None if self.selecting else self.tracked_point(f)

        if point is not None:
            # ---- STABILISED VIEW: crop centre is pinned to the canvas centre,
            # the frame slides underneath it.
            stabilised = True
            marker = point
            cx, cy = self.crop_center(point[0], point[1], crop_w, crop_h)
            tx, ty = cvw / 2.0 - cx * s, cvh / 2.0 - cy * s
        else:
            # ---- PLAIN VIEW (selecting a point / nothing tracked here yet)
            stabilised = False
            marker = raw[:2] if raw else None
            if marker:
                cx, cy = self.crop_center(marker[0], marker[1], crop_w, crop_h)
            else:
                cx, cy = W / 2.0, H / 2.0
            tx, ty = (cvw - nw) / 2.0, (cvh - nh) / 2.0

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(
            small, M, (cvw, cvh), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20),
        )
        self.view = (s, tx, ty)

        # crop box (+ dim everything outside it)
        rcx, rcy = cx * s + tx, cy * s + ty
        hw, hh = crop_w * s / 2.0, crop_h * s / 2.0
        x0, x1 = int(round(rcx - hw)), int(round(rcx + hw))
        y0, y1 = int(round(rcy - hh)), int(round(rcy + hh))
        bx0, bx1 = max(0, x0), min(cvw, x1)
        by0, by1 = max(0, y0), min(cvh, y1)

        dim = cv2.convertScaleAbs(img, alpha=0.35)
        if bx1 > bx0 and by1 > by0:
            dim[by0:by1, bx0:bx1] = img[by0:by1, bx0:bx1]
        img = dim

        thick = max(1, int(round(2 * dpr)))
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), thick)

        # tracking marker: circle + dot (+ confidence)
        if marker is not None:
            mx = int(round(marker[0] * s + tx))
            my = int(round(marker[1] * s + ty))
            cv2.circle(img, (mx, my), int(round(18 * dpr)), (255, 255, 255), thick, cv2.LINE_AA)
            cv2.circle(img, (mx, my), int(round(4 * dpr)), (0, 255, 0), -1, cv2.LINE_AA)
            if raw is not None:
                conf = raw[2]
                color = (0, 255, 0) if conf >= 0.7 else (0, 220, 255) if conf >= 0.5 else (0, 0, 255)
                cv2.putText(
                    img, f"{conf * 100:.0f}%",
                    (mx + int(round(24 * dpr)), my - int(round(14 * dpr))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * dpr, color, thick, cv2.LINE_AA,
                )

        label = f"Frame {f}   " + ("STABILISED VIEW" if stabilised else "SOURCE VIEW")
        cv2.putText(
            img, label, (int(16 * dpr), int(32 * dpr)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * dpr, (255, 255, 255), thick, cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, cvw, cvh, rgb.strides[0], QImage.Format_RGB888).copy()
        qimg.setDevicePixelRatio(dpr)
        self.canvas.set_image(qimg)

        self.frame_label.setText(
            f"Frame: {f}/{max(0, self.frame_count - 1)}   |   {f / self.fps:.3f}s"
            + (f"   |   tracked frames: {len(self.track_history)}" if self.track_history else "")
        )

    # ========================================================
    # EXPORT
    # ========================================================

    def output_size(self):
        aspect = self.get_aspect_ratio()
        choice = self.res_combo.currentText()
        if choice in self.OUTPUT_SHORT_SIDE:
            short = float(self.OUTPUT_SHORT_SIDE[choice])
        else:
            cw, ch = self.compute_crop_size()
            short = min(cw, ch)
        if aspect >= 1.0:
            oh, ow = short, short * aspect
        else:
            ow, oh = short, short / aspect
        ow, oh = int(round(ow)), int(round(oh))
        return max(2, ow - ow % 2), max(2, oh - oh % 2)

    @staticmethod
    def render_crop(frame, cx, cy, cw, ch, out_w, out_h):
        """Sub-pixel crop centred on (cx, cy), then resize to the output size."""
        cwi, chi = int(round(cw)), int(round(ch))
        M = np.float32([[1, 0, -(cx - cw / 2.0)], [0, 1, -(cy - ch / 2.0)]])
        crop = cv2.warpAffine(
            frame, M, (cwi, chi), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        interp = cv2.INTER_AREA if out_w < cwi else cv2.INTER_CUBIC
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)

    def export_video(self):
        if self.video_path is None:
            QMessageBox.warning(self, "Export", "Open a video first.")
            return
        if len(self.track_history) < 2:
            QMessageBox.warning(self, "Export", "Track at least two frames before generating the output.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Generate Output Video", "", "MP4 Video (*.mp4)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        ok, msg = self.run_export(out_path)
        if ok:
            QMessageBox.information(self, "Output Generated", f"Your stabilised video has been generated:\n\n{out_path}\n\n{msg}")
        elif msg:
            QMessageBox.critical(self, "Export", msg)

    def run_export(self, out_path):
        """Render [first tracked frame .. last tracked frame]. Returns (ok, message)."""
        self.stop_all()

        first, last = min(self.track_history), max(self.track_history)
        total = last - first + 1
        f0, px, py = self._path()

        cw, ch = self.compute_crop_size()
        ow, oh = self.output_size()

        src = cv2.VideoCapture(self.video_path)
        if not src.isOpened():
            return False, "Could not reopen the source video."
        src.set(cv2.CAP_PROP_POS_FRAMES, first)

        sink = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            sink = FFmpegSink(ffmpeg, out_path, ow, oh, self.fps, self.video_path,
                              first / self.fps, total / self.fps)
            if not sink.ok:
                sink = None
        if sink is None:
            sink = CvSink(out_path, ow, oh, self.fps)
            if not sink.ok:
                src.release()
                return False, "Could not create the output video.\n\nInstall FFmpeg (pip install imageio-ffmpeg) and try again."

        dlg = QProgressDialog("Rendering stabilised video...", "Cancel", 0, total, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        cancelled = failed = False
        written = 0
        for i, n in enumerate(range(first, last + 1)):
            if dlg.wasCanceled():
                cancelled = True
                break
            ok, frame = src.read()
            if not ok:
                break

            cx, cy = self.crop_center(px[n - f0], py[n - f0], cw, ch)
            out = self.render_crop(frame, cx, cy, cw, ch, ow, oh)
            if not sink.write(out):
                failed = True
                break
            written += 1
            dlg.setValue(i + 1)
            QApplication.processEvents()

        src.release()
        dlg.close()

        if cancelled:
            sink.abort()
            Path(out_path).unlink(missing_ok=True)
            return False, ""
        ok, err = sink.close()
        if failed or not ok:
            return False, f"Export failed.\n\n{err}"
        if written == 0:
            return False, "No frames could be read from the source video."
        return True, f"{written} frames | {ow}x{oh} | encoder: {sink.name}"

    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):
        self.stop_all()
        if self.cap:
            self.cap.release()
        event.accept()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AeroTrack()
    window.show()
    sys.exit(app.exec())