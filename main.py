import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QPlainTextEdit, QFrame, QProgressBar, QSizePolicy,
    QLineEdit, QShortcut,
)

import face_engine as backend

try:
    import config
except ImportError:  # keep the terminal usable even if config.py is missing
    config = None

MIN_FRAMES = getattr(config, "LIVENESS_MIN_FRAMES", 30)
QR_EVERY_N_FRAMES = 3          # QR decoding is high intensive
CAMERA_INDEX = 0

# terminal a real customer can reach. Ctrl+M hides/shows it at runtime.
MANUAL_ENTRY = True

# palette -----------------------------------------------------------------
INK        = "#0E1116"
PANEL      = "#161A22"
LINE       = "#242A35"
TEXT       = "#E6EAF2"
MUTED      = "#828C9F"
MINT       = "#3DDC97"
AMBER      = "#F2B441"
CORAL      = "#F0685F"

STATE_COLORS = {"idle": MUTED, "wait": AMBER, "ok": MINT, "bad": CORAL}

REASON_TEXT = {
    "collecting": "Reading facial motion",
    "static_face": "Static image — no movement",
    "no_blink":    "Blink to confirm",
    "ok":          "Live person confirmed",
    "verified":    "Verified for this transaction",
}


# worker threading -----------------------------------------------------------
class VisionWorker(QThread):
    """Owns the camera. Emits frames and analysis results to the GUI thread."""

    frame_ready = pyqtSignal(object)      # BGR ndarray
    qr_found = pyqtSignal(str)
    liveness = pyqtSignal(object)         # status dict, or None when no face
    failed = pyqtSignal(str)

    def __init__(self, cam_index=CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self._running = True
        self._reset_requested = False
        self._lock = QMutex()

    def request_reset(self):
        self._lock.lock()
        self._reset_requested = True
        self._lock.unlock()

    def stop(self):
        self._running = False

    def _take_reset_flag(self):
        self._lock.lock()
        flag, self._reset_requested = self._reset_requested, False
        self._lock.unlock()
        return flag

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.failed.emit(f"No camera on index {self.cam_index}. "
                             "Connect a device and restart the terminal.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        tracker = backend.LivenessTracker()
        last_qr = None
        counter = 0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(30)
                continue

            if self._take_reset_flag():
                tracker.reset()
                last_qr = None

            counter += 1

            if counter % QR_EVERY_N_FRAMES == 0:
                try:
                    payload = backend.decode_qr(frame)
                except Exception:
                    payload = None
                if payload and payload != last_qr:
                    last_qr = payload
                    self.qr_found.emit(payload)

            # nk
            if not tracker.verified:
                try:
                    metrics = backend.landmarks_metrics(frame)
                except Exception:
                    metrics = None

                if metrics:
                    tracker.add(metrics)
                    self.liveness.emit(tracker.status())
                else:
                    self.liveness.emit(None)

            self.frame_ready.emit(frame)
            self.msleep(1)

        cap.release()


# small widgets 
class StatusRow(QFrame):
    """One line of terminal state: coloured dot, label, current value."""

    def __init__(self, label, placeholder, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        outer.addWidget(self.dot, 0, Qt.AlignTop)

        column = QVBoxLayout()
        column.setSpacing(3)

        caption = QLabel(label.upper())
        caption.setObjectName("caption")
        column.addWidget(caption)

        self.value = QLabel(placeholder)
        self.value.setObjectName("value")
        self.value.setWordWrap(True)
        column.addWidget(self.value)

        outer.addLayout(column, 1)
        self.set_state("idle", placeholder)

    def set_state(self, state, text):
        colour = STATE_COLORS[state]
        self.dot.setStyleSheet(f"background:{colour}; border-radius:5px;")
        self.value.setStyleSheet(f"color:{colour};")
        self.value.setText(text)


def draw_reticle(frame, colour_bgr):
    """Corner brackets marking the scan area — feedback without hiding the face."""
    h, w = frame.shape[:2]
    bw, bh = int(w * 0.52), int(h * 0.78)
    x1, y1 = (w - bw) // 2, (h - bh) // 2
    x2, y2 = x1 + bw, y1 + bh
    arm = int(min(bw, bh) * 0.12) 
    t = 3
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + arm * dx, cy), colour_bgr, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + arm * dy), colour_bgr, t, cv2.LINE_AA)
    return frame


# main window
class CheckoutTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.qr_value = None
        self.is_live = False
        self.camera_ok = True

        self.setWindowTitle("Self-Service Checkout")
        self.resize(1180, 720)
        self.setMinimumSize(980, 620)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        page = QHBoxLayout(root)
        page.setContentsMargins(22, 22, 22, 22)
        page.setSpacing(20)
        page.addLayout(self._build_video_panel(), 3)
        page.addLayout(self._build_side_panel(), 2)

        self.setStyleSheet(STYLESHEET)

        self.worker = VisionWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.qr_found.connect(self.on_qr)
        self.worker.liveness.connect(self.on_liveness)
        self.worker.failed.connect(self.on_camera_failure)
        self.worker.start()

        self.log("Terminal ready. Present a QR code to begin.")

    # layout 
    def _build_video_panel(self):
        column = QVBoxLayout()
        column.setSpacing(14)

        heading = QLabel("Scan area")
        heading.setObjectName("h2")
        column.addWidget(heading)

        shell = QFrame()
        shell.setObjectName("videoShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(10, 10, 10, 10)

        self.video = QLabel("Starting camera")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        shell_layout.addWidget(self.video)

        column.addWidget(shell, 1)

        self.hint = QLabel("Hold the code steady inside the brackets, then look at the camera.")
        self.hint.setObjectName("hint")
        column.addWidget(self.hint)
        return column

    def _build_side_panel(self):
        column = QVBoxLayout()
        column.setSpacing(14)

        title = QLabel("Checkout terminal")
        title.setObjectName("h1")
        column.addWidget(title)
        # 
        self.banner = QLabel("WAITING FOR CUSTOMER")
        self.banner.setObjectName("banner")
        self.banner.setAlignment(Qt.AlignCenter)
        column.addWidget(self.banner)

        self.qr_row = StatusRow("Basket code", "No code scanned")
        self.live_row = StatusRow("Liveness check", "Waiting for a face")
        column.addWidget(self.qr_row)
        column.addWidget(self.live_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, MIN_FRAMES)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixed    bHeight(6)
        column.addWidget(self.progress)

        column.addWidget(self._build_manual_panel())

        log_caption = QLabel("ACTIVITY")
        log_caption.setObjectName("caption")
        column.addWidget(log_caption)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.document().setMaximumBlockCount(300)
        column.addWidget(self.log_box, 1)

        self.btn_finish = QPushButton("Complete checkout")
        self.btn_finish.setObjectName("primary")
        self.btn_finish.setEnabled(False)
        self.btn_finish.setMinimumHeight(48)
        self.btn_finish.clicked.connect(self.complete_checkout)
        column.addWidget(self.btn_finish)

        self.btn_reset = QPushButton("Start over")
        self.btn_reset.setObjectName("ghost")
        self.btn_reset.setMinimumHeight(44)
        self.btn_reset.clicked.connect(lambda: self.reset("Cleared for the next customer."))
        column.addWidget(self.btn_reset)
        return column

    def _build_manual_panel(self):
        """Keyboard entry standing in for a physical code, for testing."""
        panel = QFrame()
        panel.setObjectName("testCard")

        box = QVBoxLayout(panel)
        box.setContentsMargins(16, 12, 16, 14)
        box.setSpacing(9)

        caption = QLabel("TEST OVERRIDE")
        caption.setObjectName("testCaption")
        box.addWidget(caption)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type a basket code")
        self.manual_input.setMaxLength(128)
        self.manual_input.setMinimumHeight(38)
        self.manual_input.returnPressed.connect(self.submit_manual_code)
        row.addWidget(self.manual_input, 1)

        self.btn_manual = QPushButton("Use code")
        self.btn_manual.setObjectName("small")
        self.btn_manual.setMinimumHeight(38)
        self.btn_manual.clicked.connect(self.submit_manual_code)
        row.addWidget(self.btn_manual)

        box.addLayout(row)

        panel.setVisible(MANUAL_ENTRY)
        self.manual_panel = panel

        toggle = QShortcut(QKeySequence("Ctrl+M"), self)
        toggle.activated.connect(self.toggle_manual_entry)
        return panel

    def toggle_manual_entry(self):
        showing = not self.manual_panel.isVisible()
        self.manual_panel.setVisible(showing)
        if showing:
            self.manual_input.setFocus()

    def submit_manual_code(self):
        code = self.manual_input.text().strip()
        if not code:
            self.manual_input.setFocus()
            return
        self.manual_input.clear()
        self.on_qr(code, manual=True)

    # slots 
    def on_frame(self, frame):
        if not self.camera_ok:
            return
        # 

        display = cv2.flip(frame, 1)                       # mirror for the customer
        colour = (151, 220, 61) if self.is_live else (65, 180, 242)  # BGR mint / amber
        draw_reticle(display, colour)

        rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_qr(self, payload, manual=False):
        self.qr_value = payload
        if manual:
            self.qr_row.set_state("ok", f"{payload}   ·   entered manually")
            self.log(f"Code entered manually: {payload}")
        else:
            self.qr_row.set_state("ok", payload)
            self.log(f"Code scanned: {payload}")
        self.refresh_state()

    def on_liveness(self, status):
        if self.is_live:
            return  # verified customers stay verified until Start over

        if status is None:
            self.live_row.set_state("idle", "No face detected")
            self.progress.setValue(0)
        else:
            self.is_live = status["live"]
            reason = REASON_TEXT.get(status["reason"], status["reason"])
            self.progress.setValue(min(status["frames"], MIN_FRAMES))
            if self.is_live:
                self.progress.setValue(MIN_FRAMES)
                self.live_row.set_state("ok", REASON_TEXT["verified"])
                self.log("Liveness verified — held for this transaction.")
            elif status["reason"] == "collecting":
                self.live_row.set_state("wait", reason)
            else:
                self.live_row.set_state("bad", reason)
        self.refresh_state()

    def on_camera_failure(self, message):
        self.camera_ok = False
        self.video.setText("Camera unavailable")
        self.hint.setText(message)
        self.banner.setText("CAMERA OFFLINE")
        self.banner.setProperty("tone", "bad")
        self._repolish(self.banner)
        self.log(message)

    # state
    def refresh_state(self):
        ready = bool(self.qr_value) and self.is_live
        self.btn_finish.setEnabled(ready)

        if ready:
            text, tone = "READY TO PAY", "ok"
        elif self.qr_value:
            text, tone = "VERIFYING CUSTOMER", "wait"
        elif self.is_live:
            text, tone = "SCAN YOUR BASKET CODE", "wait"
        else:
            text, tone = "WAITING FOR CUSTOMER", "idle"

        if self.banner.text() != text:
            self.banner.setText(text)
            self.banner.setProperty("tone", tone)
            self._repolish(self.banner)

    def complete_checkout(self):
        self.log(f"Checkout completed for {self.qr_value}")
        self.reset("Ready for the next customer.")

    def reset(self, message):
        self.qr_value = None
        self.is_live = False
        self.worker.request_reset()
        self.qr_row.set_state("idle", "No code scanned")
        self.live_row.set_state("idle", "Waiting for a face")
        self.progress.setValue(0)
        self.btn_finish.setEnabled(False)
        self.manual_input.clear()
        self.refresh_state()
        self.log(message)

    def log(self, message):
        self.log_box.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()

# Style sheet css for the whole terminal. Qt doesn't support CSS variables, so we use Python f strings to inject our palette colors.
STYLESHEET = f"""
QWidget {{
    font-family: "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    color: {TEXT};
}}
QWidget#root {{ background: {INK}; }}

QLabel#h1 {{ font-size: 24px; font-weight: 600; letter-spacing: -0.3px; }}
QLabel#h2 {{ font-size: 15px; font-weight: 600; color: {MUTED}; }}
QLabel#caption {{ font-size: 11px; font-weight: 600; letter-spacing: 1.4px; color: {MUTED}; }}
QLabel#value {{ font-size: 15px; font-weight: 500; }}
QLabel#hint {{ color: {MUTED}; }}

QFrame#card {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 14px;
}}
QFrame#videoShell {{
    background: #0A0D12;
    border: 1px solid {LINE};
    border-radius: 16px;
}}
QFrame#testCard {{
    background: transparent;
    border: 1px dashed #3A4150;
    border-radius: 14px;
}}
QLabel#testCaption {{
    font-size: 10px; font-weight: 700; letter-spacing: 1.6px; color: #6E7889;
}}
QLineEdit {{
    background: #10141B;
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 0 12px;
    color: {TEXT};
    font-family: "JetBrains Mono", "SF Mono", Consolas, "DejaVu Sans Mono", monospace;
    selection-background-color: {MINT};
    selection-color: #06231A;
}}
QLineEdit:focus {{ border: 1px solid {MINT}; }}
QPushButton#small {{
    background: #222936;
    border: none;
    border-radius: 10px;
    padding: 0 16px;
    color: {TEXT};
    font-weight: 600;
}}
QPushButton#small:hover {{ background: #2C3441; }}
QLabel#video {{ color: {MUTED}; font-size: 15px; }}

QLabel#banner {{
    padding: 16px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1.8px;
    background: {PANEL};
    border: 1px solid {LINE};
    color: {MUTED};
}}
QLabel#banner[tone="wait"] {{ color: {AMBER};  border-color: #4A3A16; background: #221B0E; }}
QLabel#banner[tone="ok"]   {{ color: {MINT};   border-color: #1B4D38; background: #0E2119; }}
QLabel#banner[tone="bad"]  {{ color: {CORAL};  border-color: #4E2320; background: #211110; }}

QProgressBar {{ background: {LINE}; border: none; border-radius: 3px; }}
QProgressBar::chunk {{ background: {MINT}; border-radius: 3px; }}

QPlainTextEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 10px;
    color: #AEB7C6;
    font-family: "JetBrains Mono", "SF Mono", Consolas, "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}

QPushButton#primary {{
    background: {MINT};
    color: #06231A;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton#primary:hover {{ background: #56E5A9; }}
QPushButton#primary:disabled {{ background: #1C222C; color: #55606F; }}

QPushButton#ghost {{
    background: transparent;
    border: 1px solid {LINE};
    border-radius: 12px;
    color: #C2CAD8;
    font-weight: 600;
}}
QPushButton#ghost:hover {{ background: #1B212B; }}
QPushButton:focus {{ outline: none; border: 1px solid {MINT}; }}
"""


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    # font may not exist on all platforms however qt would fallback
    app.setFont(QFont("Segoe UI", 10))

    window = CheckoutTerminal()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()