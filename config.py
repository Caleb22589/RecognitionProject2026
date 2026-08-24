# Central configuration for the self-checkout kiosk.
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kiosk.db")

# 0.6 is the face_recognition default; lower = stricter.
FACE_MATCH_TOLERANCE = 0.5

#  track the Mouth Aspect Ratio (MAR) across a rolling buffer of frames.
# A printed/static photo has a nearly constant MAR; a live person's lips move.
LIVENESS_FRAME_WINDOW = 12        # frames kept per checkout session
LIVENESS_MIN_FRAMES = 6           # need at least this many before deciding
LIVENESS_MAR_VARIANCE = 0.0006    # min variance of MAR to count as "moving"
LIVENESS_BLINK_EAR = 0.21         # eye-aspect-ratio below this = eye closed
LIVENESS_REQUIRE_BLINK = False    # optionally also require a blink

# ---- Money ----
SIGNUP_BONUS = 20.00              # starting wallet balance for new accounts
CURRENCY = "$"

# The kiosk must recognise the SAME person in at least IDENTITY_MIN_VOTES of the
# last IDENTITY_WINDOW frames before locking the account in. This stops the
# name from flickering on/off on a single good/bad frame.
IDENTITY_WINDOW = 6
IDENTITY_MIN_VOTES = 3
# Once locked, keep showing that shopper until their face is absent for this long.
IDENTITY_HOLD_SECONDS = 4

# Session buffers expire after this many seconds of inactivity.
SESSION_TTL_SECONDS = 60

# Encoding a face costs far more CPU than reading a frame, so we only attempt a
# match every Nth frame. Higher = lighter load but slower to recognise someone.
RECOGNISE_EVERY_N_FRAMES = 8

# Decoding a barcode is cheaper than encoding a face but still not free, so the
# scanner also runs on every Nth frame rather than on all of them.
QR_EVERY_N_FRAMES = 3

CAMERA_INDEX = 0                  # 0 is the built in webcam on most laptops
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_RETRY_MS = 30              # pause before retrying after a dropped frame
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 720
LOG_MAX_LINES = 300               # activity log keeps only the most recent lines
WORKER_SHUTDOWN_MS = 2000         # how long to wait for the camera thread to stop

# ---- Top up vouchers ----
# A voucher is a printed code the shop sells, e.g. "TOPUP|20.00". Scanning one
# adds credit to the recognised shopper instead of taking a payment.
TOPUP_PREFIX = "TOPUP"
MAX_TOPUP_DOLLARS = 200.00
