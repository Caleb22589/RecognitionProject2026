"""Central configuration for the self-checkout kiosk."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kiosk.db")

# ---- Face matching ----
# 0.6 is the face_recognition default; lower = stricter.
FACE_MATCH_TOLERANCE = 0.5

# ---- Liveness (anti-spoofing) ----
# We track the Mouth Aspect Ratio (MAR) across a rolling buffer of frames.
# A printed/static photo has a nearly constant MAR; a live person's lips move.
LIVENESS_FRAME_WINDOW = 12        # frames kept per checkout session
LIVENESS_MIN_FRAMES = 6           # need at least this many before deciding
LIVENESS_MAR_VARIANCE = 0.0006    # min variance of MAR to count as "moving"
LIVENESS_BLINK_EAR = 0.21         # eye-aspect-ratio below this = eye closed
LIVENESS_REQUIRE_BLINK = False    # optionally also require a blink

# ---- Money ----
SIGNUP_BONUS = 20.00              # starting wallet balance for new accounts
CURRENCY = "$"

# ---- Identity stabilisation (kiosk) ----
# The kiosk must recognise the SAME person in at least IDENTITY_MIN_VOTES of the
# last IDENTITY_WINDOW frames before locking the account in. This stops the
# name from flickering on/off on a single good/bad frame.
IDENTITY_WINDOW = 6
IDENTITY_MIN_VOTES = 3
# Once locked, keep showing that shopper until their face is absent for this long.
IDENTITY_HOLD_SECONDS = 4

# Session buffers expire after this many seconds of inactivity.
SESSION_TTL_SECONDS = 60
