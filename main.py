import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QPlainTextEdit, QFrame, QProgressBar, QSizePolicy,
    QLineEdit, QShortcut, QInputDialog,
)

import db
import face_engine as backend

try:
    import config
except ImportError:  # keep the terminal usable even if config.py is missing
    config = None

MIN_FRAMES = getattr(config, "LIVENESS_MIN_FRAMES", 30)
QR_EVERY_N_FRAMES = getattr(config, "QR_EVERY_N_FRAMES", 3)
RECOGNISE_EVERY_N_FRAMES = getattr(config, "RECOGNISE_EVERY_N_FRAMES", 8)
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
CAMERA_WIDTH = getattr(config, "CAMERA_WIDTH", 1280)
CAMERA_HEIGHT = getattr(config, "CAMERA_HEIGHT", 720)
CAMERA_RETRY_MS = getattr(config, "CAMERA_RETRY_MS", 30)
WINDOW_WIDTH = getattr(config, "WINDOW_WIDTH", 1180)
WINDOW_HEIGHT = getattr(config, "WINDOW_HEIGHT", 720)
LOG_MAX_LINES = getattr(config, "LOG_MAX_LINES", 300)
WORKER_SHUTDOWN_MS = getattr(config, "WORKER_SHUTDOWN_MS", 2000)

# terminal a real customer can reach. Ctrl+M hides/shows it at runtime.
MANUAL_ENTRY = True

# palette having neutral colouring

INK        = "#F2F2F2"      # window background
PANEL      = "#FFFFFF"      # cards and panels
SHELL      = "#E6E6E6"      # area behind the camera image
LINE       = "#C6C6C6"      # borders
TEXT       = "#1E1E1E"
MUTED      = "#6E6E6E"
GREEN      = "#3F6B4A"      # verified / ready
BROWN      = "#7A6535"      # waiting on the customer
RED        = "#8C4A45"      # something is wrong

STATE_COLORS = {"idle": MUTED, "wait": BROWN, "ok": GREEN, "bad": RED}

REASON_TEXT = {
    "collecting": "Reading facial motion",
    "static_face": "Static image — no movement",
    "no_blink":    "Blink to confirm",
    "ok":          "Live person confirmed",
    "verified":    "Verified for this transaction",
}


# worker threading 
class VisionWorker(QThread):
    # Owns the camera. Emits frames and analysis results to the GUI thread.

    frame_ready = pyqtSignal(object)      # BGR ndarray
    qr_found = pyqtSignal(str)
    liveness = pyqtSignal(object)         # status dict, or None when no face
    identity = pyqtSignal(object)         # match dict once locked in, else None
    enrol_ready = pyqtSignal(object)      # 128-d encoding, or None if unusable
    failed = pyqtSignal(str)

    def __init__(self, known=None, cam_index=CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self._running = True
        self._reset_requested = False
        self._enrol_requested = False
        self._known = dict(known or {})
        self._lock = QMutex()

    def request_reset(self):
        self._lock.lock()
        self._reset_requested = True
        self._lock.unlock()

    def request_enrol(self):
        # Ask for the next frame's face encoding, for signing a new shopper up.
        self._lock.lock()
        self._enrol_requested = True
        self._lock.unlock()

    def set_known(self, known):
        # Swap in a fresh {customer_id: encoding} map after somebody enrols.
        self._lock.lock()
        self._known = dict(known)
        self._lock.unlock()

    def stop(self):
        self._running = False

    def _take_reset_flag(self):
        self._lock.lock()
        flag, self._reset_requested = self._reset_requested, False
        self._lock.unlock()
        return flag

    def _take_enrol_flag(self):
        self._lock.lock()
        flag, self._enrol_requested = self._enrol_requested, False
        self._lock.unlock()
        return flag

    def _get_known(self):
        self._lock.lock()
        known = dict(self._known)
        self._lock.unlock()
        return known

    def run(self):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.failed.emit(f"No camera on index {self.cam_index}. "
                             "Connect a device and restart the terminal.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        tracker = backend.LivenessTracker()
        voter = backend.IdentityVoter()
        last_qr = None
        counter = 0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(CAMERA_RETRY_MS)
                continue

            if self._take_reset_flag():
                tracker.reset()
                voter.reset()
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

            # Match the face against enrolled accounts. Encoding a face is the
            # most expensive thing in this loop, so it only runs every Nth frame
            # and stops entirely once the voter has locked an account in.
            if counter % RECOGNISE_EVERY_N_FRAMES == 0 and voter.locked_id is None:
                known = self._get_known()
                if known:
                    try:
                        match = backend.recognise(frame, known)
                    except Exception:
                        match = None
                    if match:
                        locked = voter.add(match["user_id"])
                    else:
                        locked = voter.add(None)
                    if locked is not None:
                        self.identity.emit(match or {"user_id": locked})

            # Enrolment grabs the encoding here rather than on the GUI thread,
            # so the window never freezes while dlib works.
            if self._take_enrol_flag():
                try:
                    encoding = backend.encode_face(frame)
                except Exception:
                    encoding = None
                self.enrol_ready.emit(encoding)

            self.frame_ready.emit(frame)
            self.msleep(1)

        cap.release()


# small widgets 
class StatusRow(QFrame):
    # One line of terminal state: coloured dot, label, current value.

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
        self.dot.setStyleSheet(f"background:{colour}; border-radius:5px;") # border radius to make borders more rounded.
        self.value.setStyleSheet(f"color:{colour};")
        self.value.setText(text)


def draw_reticle(frame, colour_bgr):
    # Corner brackets marking the scan area feedback without hiding the face.
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
        self.basket_code = None
        self.basket_total = None      # dollars, or None until a total is known
        self.is_live = False
        self.camera_ok = True
        self.account = None           # the logged-in shopper's row from the database

        db.init_db()
        self.known_faces = db.load_known_encodings()

        self.setWindowTitle("Self-Service Checkout")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
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

        self.worker = VisionWorker(known=self.known_faces)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.qr_found.connect(self.on_qr)
        self.worker.liveness.connect(self.on_liveness)
        self.worker.identity.connect(self.on_identity)
        self.worker.enrol_ready.connect(self.on_enrol_ready)
        self.worker.failed.connect(self.on_camera_failure)
        self.worker.start()

        self.log(f"Terminal ready. {len(self.known_faces)} account(s) enrolled.")
        self.log("Look at the camera, then present a basket code.")

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

        self.hint = QLabel("Hold the code steady inside the brackets, then look at the camera. Open mouth to verify liveness.")
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
        self.account_row = StatusRow("Account", "Not recognised")
        column.addWidget(self.qr_row)
        column.addWidget(self.live_row)
        column.addWidget(self.account_row)
        #progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, MIN_FRAMES)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        column.addWidget(self.progress)

        self.btn_enrol = QPushButton("Sign up with this face")
        self.btn_enrol.setObjectName("ghost")
        self.btn_enrol.setMinimumHeight(40)
        self.btn_enrol.setVisible(False)
        self.btn_enrol.clicked.connect(self.start_enrolment)
        column.addWidget(self.btn_enrol)

        column.addWidget(self._build_manual_panel())

        log_caption = QLabel("ACTIVITY")
        log_caption.setObjectName("caption")
        column.addWidget(log_caption)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.document().setMaximumBlockCount(LOG_MAX_LINES)
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
        # Keyboard entry standing in for a physical code, for testing.
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
         # A code on its own carries no price, so the total can be typed here.
        self.amount_input = QLineEdit()
        self.amount_input.setPlaceholderText("Total")
        self.amount_input.setMaxLength(10)
        self.amount_input.setMinimumHeight(38)
        self.amount_input.setFixedWidth(90)
        self.amount_input.returnPressed.connect(self.submit_manual_code)
        self.amount_input.textChanged.connect(self.on_amount_typed)
        row.addWidget(self.amount_input)

        
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

    @staticmethod
    def clean_total(value):
        # Return a usable basket total, or None if it is not money the kiosk takes.
        #
        # the button must never offer to charge
        # an amount the database is going to refuse.
        try:
            total = float(value)
        except (TypeError, ValueError):
            return None
        if total != total:                              # NaN
            return None
        if total <= 0 or total > db.MAX_TRANSACTION_DOLLARS:
            return None
        return total

    def on_amount_typed(self, text):
        # Let a typed total stand in for one the basket code did not carry.
        text = text.strip()
        # A halftyped box is not an error yet, so this only clears the total.
        if text:
            self.basket_total = self.clean_total(text)
        else:
            self.basket_total = None
        self.refresh_basket_row()
        self.refresh_state()

    def refresh_basket_row(self):
        # Redraw the basket line from the current code and total.
        if not self.basket_code:
            self.qr_row.set_state("idle", "No code scanned")
            return

        if self.basket_total is None:
            self.qr_row.set_state("wait", f"{self.basket_code}   ·   total needed")
        else:
            self.qr_row.set_state(
                "ok", f"{self.basket_code}   ·   {db.format_money(int(round(self.basket_total * 100)))}"
            )

    # slots 
    def on_frame(self, frame):
        if not self.camera_ok:
            return
        # 

        display = cv2.flip(frame, 1)                       # mirror for the customer
        # OpenCV wants BGR, so these are GREEN and MUTED with the bytes swapped.
        if self.is_live:
            colour = (74, 107, 63)
        else:
            colour = (110, 110, 110)
        draw_reticle(display, colour)

        rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_qr(self, payload, manual=False):
        # A top up voucher adds credit rather than paying for a basket, so it is
        # handled first and never reaches the basket code path.
        topup = backend.parse_topup(payload)
        if topup is not None:
            self.redeem_topup(topup, manual)
            return

        self.qr_value = payload
        code, total = backend.parse_basket(payload)
        self.basket_code = code or payload

        # A total printed on the code wins; otherwise fall back on the typed one.
        # A code carrying a nonsense total is treated as carrying none at all.
        if total is not None:
            checked = self.clean_total(total)
        else:
            checked = None
        if checked is not None:
            self.basket_total = checked
            self.amount_input.setText(f"{checked:.2f}")
        elif total is not None:
            self.log(f"Ignoring the total printed on {self.basket_code}: {total} is not a valid amount.")

        if manual:
            source = "entered manually"
        else:
            source = "scanned"

        if self.basket_total is not None:
            total_text = f" — total {self.basket_total:.2f}"
        else:
            total_text = " — no total on the code, type one"

        self.log(f"Basket {self.basket_code} {source}{total_text}")
        self.refresh_basket_row()
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

    def on_identity(self, match):
        # A face won enough votes — load that account and log the shopper in.
        if self.account is not None or not match:
            return

        try:
            customer = db.get_customer(match["user_id"])
        except db.AccountError as error:
            # The account was deleted while its encoding was still in memory.
            self.log(f"Account lookup failed: {error}")
            self.known_faces = db.load_known_encodings()
            self.worker.set_known(self.known_faces)
            return

        self.account = customer
        confidence = match.get("confidence")
        self.show_account()
        if confidence is not None:
            confidence_text = f", {confidence:.0%} confidence)"
        else:
            confidence_text = ")"

        self.log(f"Recognised {customer['name']} (account #{customer['id']}{confidence_text}")
        self.refresh_state()

    def show_account(self):
        # Put the logged-in shopper and their remaining credit on screen.
        if self.account is None:
            self.account_row.set_state("idle", "Not recognised")
            return

        balance = db.format_money(self.account["balance_cents"])
        short = self.basket_total is not None and \
            int(round(self.basket_total * 100)) > self.account["balance_cents"]
        if short:
            state = "bad"
            suffix = "   ·   not enough credit"
        else:
            state = "ok"
            suffix = ""
        self.account_row.set_state(state, f"{self.account['name']}   ·   {balance}{suffix}")

    # enrolment
    def start_enrolment(self):
        # Ask the worker for a clean encoding of whoever is at the camera.
        if not self.camera_ok:
            return
        self.btn_enrol.setEnabled(False)
        self.log("Capturing face for sign up — hold still.")
        self.worker.request_enrol()

    def on_enrol_ready(self, encoding):
        self.btn_enrol.setEnabled(True)

        if encoding is None:
            self.log("Sign up failed: need exactly one clear face in frame.")
            return

        # Refuse to create a second account for a face we already know.
        existing = backend.identify(encoding, self.known_faces)
        if existing:
            self.log(f"That face already belongs to account #{existing['user_id']}.")
            self.on_identity(existing)
            return

        name, confirmed = QInputDialog.getText(self, "New account", "Name for this account:")
        if not confirmed:
            self.log("Sign up cancelled.")
            return

        try:
            customer = db.create_customer(name, encoding)
        except db.AccountError as error:
            self.log(f"Sign up rejected: {error}")
            return

        self.known_faces = db.load_known_encodings()
        self.worker.set_known(self.known_faces)
        self.account = customer
        self.show_account()
        self.log(f"Created account #{customer['id']} for {customer['name']} "
                 f"with {db.format_money(customer['balance_cents'])} credit.")
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
        # Work out what is still missing and say so, in the order it is needed.
        self.show_account()   # balance colouring depends on the current basket total

        has_credit = (
            self.account is not None
            and self.basket_total is not None
            and int(round(self.basket_total * 100)) <= self.account["balance_cents"]
        )
        ready = bool(self.basket_code) and self.is_live and has_credit
        self.btn_finish.setEnabled(ready)

        # Offer sign up only to a confirmed live face with no matching account.
        self.btn_enrol.setVisible(self.camera_ok and self.is_live and self.account is None)

        if self.account is not None:
            if self.basket_total is not None:
                self.btn_finish.setText(
                    f"Pay {db.format_money(int(round(self.basket_total * 100)))} from credit")
            else:
                self.btn_finish.setText("Complete checkout")
        else:
            self.btn_finish.setText("Complete checkout")

        if ready:
            text, tone = "READY TO PAY", "ok"
        elif self.account is not None and self.basket_total is not None and not has_credit:
            text, tone = "NOT ENOUGH CREDIT", "bad"
        elif self.is_live and self.account is None:
            text, tone = "FACE NOT RECOGNISED", "wait"
        elif self.account is not None and not self.basket_code:
            text, tone = "SCAN YOUR BASKET CODE", "wait"
        elif self.account is not None and self.basket_total is None:
            text, tone = "BASKET TOTAL NEEDED", "wait"
        elif self.basket_code:
            text, tone = "VERIFYING CUSTOMER", "wait"
        else:
            text, tone = "WAITING FOR CUSTOMER", "idle"

        if self.banner.text() != text:
            self.banner.setText(text)
            self.banner.setProperty("tone", tone)
            self._repolish(self.banner)

    @staticmethod
    def clean_topup(value):
        # Return a usable top up amount, or None if it is not credit the kiosk
        # will add. db.top_up() rejects these too, but a voucher should be turned
        # away with a message rather than raising.
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None
        if amount != amount:                        # NaN
            return None
        limit = getattr(config, "MAX_TOPUP_DOLLARS", 200.00)
        if amount <= 0 or amount > limit:
            return None
        return amount

    def redeem_topup(self, dollars, manual=False):
        # Add credit to the recognised shopper from a scanned top up voucher.
        if self.account is None:

            self.log("Top up voucher scanned, but nobody is recognised yet. " "Look at the camera first.")
            
            return

        amount = self.clean_topup(dollars)
        if amount is None:
            limit = getattr(config, "MAX_TOPUP_DOLLARS", 200.00)
            self.log(f"Top up voucher rejected: {dollars} is not an amount this "
                     f"kiosk adds (limit {db.format_money(int(round(limit * 100)))}).")
            return

        try:
            result = db.top_up(self.account["id"], amount)
        except db.AccountError as error:
            self.log(f"Top up failed: {error}")
            return

        # Re-read the account so the balance on screen is the one in the database.
        self.account = db.get_customer(self.account["id"])

        if manual:
            source = "entered manually"
        else:
            source = "scanned"
        self.log(f"Top up voucher {source}: added "
                 f"{db.format_money(int(round(amount * 100)))} to "
                 f"{self.account['name']}'s credit. New balance: "
                 f"{db.format_money(result['balance_cents'])}")

        self.show_account()
        self.refresh_state()

    def complete_checkout(self):
        # Take the basket total out of the recognised shopper's stored credit.
        if self.account is None or self.basket_total is None:
            return  # the button should already be disabled, but never trust oit

        try:
            result = db.charge(self.account["id"], self.basket_total, self.basket_code)
        except db.InsufficientCredit as error:
            self.log(f"Declined: {error}")
            self.account = db.get_customer(self.account["id"])  # reread the true balance
            self.show_account()
            self.refresh_state()
            return
        except db.AccountError as error:
            self.log(f"Payment failed: {error}")
            return

        self.log(f"Paid {db.format_money(result['charged_cents'])} for basket "
                 f"{self.basket_code} from {self.account['name']}'s credit. "
                 f"Remaining: {db.format_money(result['balance_cents'])}")
        self.reset("Ready for the next customer.")

    def reset(self, message):
        self.qr_value = None
        self.basket_code = None
        self.basket_total = None
        self.is_live = False
        self.account = None
        self.worker.request_reset()
        self.qr_row.set_state("idle", "No code scanned")
        self.live_row.set_state("idle", "Waiting for a face")
        self.account_row.set_state("idle", "Not recognised")
        self.progress.setValue(0)
        self.btn_finish.setEnabled(False)
        self.manual_input.clear()
        self.amount_input.clear()
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
        self.worker.wait(WORKER_SHUTDOWN_MS)
        event.accept()

# Style sheet css for the whole terminal. Qt doesn't support CSS variables, so we use Python f strings to inject our palette colors.
# system fonts may not exist on windows than mac.
STYLESHEET = f"""
QWidget {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; 
    font-size: 14px;
    color: {TEXT};
}}
QWidget#root {{ background: {INK}; }}

QLabel#h1 {{ font-size: 20px; font-weight: bold; }}
QLabel#h2 {{ font-size: 14px; color: {MUTED}; }}
QLabel#caption {{ font-size: 11px; color: {MUTED}; }}
QLabel#testCaption {{ font-size: 11px; color: {MUTED}; }}
QLabel#value {{ font-size: 14px; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#video {{ color: {MUTED}; }}

QFrame#card {{
    background: {PANEL};
    border: 1px solid {LINE};
}}
QFrame#videoShell {{
    background: {SHELL};
    border: 1px solid {LINE};
}}
QFrame#testCard {{
    background: {PANEL};
    border: 1px dashed {LINE};
}}

QLineEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    padding: 0 8px;
    color: {TEXT};
}}

QLabel#banner {{
    padding: 14px;
    font-weight: bold;
    background: {PANEL};
    border: 1px solid {LINE};
    color: {MUTED};
}}
QLabel#banner[tone="wait"] {{ color: {BROWN}; }}
QLabel#banner[tone="ok"]   {{ color: {GREEN}; }}
QLabel#banner[tone="bad"]  {{ color: {RED}; }}

QProgressBar {{ background: #DEDEDE; border: none; }}
QProgressBar::chunk {{ background: #9E9E9E; }}

QPlainTextEdit {{
    background: {PANEL};
    border: 1px solid {LINE};
    padding: 6px;
    color: {TEXT};
    font-size: 12px;
}}

QPushButton {{
    background: #E4E4E4;
    border: 1px solid {LINE};
    padding: 0 14px;
    color: {TEXT};
}}
QPushButton#primary {{ font-weight: bold; }}
QPushButton:disabled {{ background: #ECECEC; color: #A6A6A6; }}
"""


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #to prevent the user interface from looking tiny, blurry, or pixelated, added for my laptop.
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    # font may not exist on all platforms however qt would fallback
    app.setFont(QFont("Segoe UI", 10))

    window = CheckoutTerminal()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()