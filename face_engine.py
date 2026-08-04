import base64
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from pyzbar.pyzbar import decode as pyzbar_decode

import config

# Initialize MediaPipe once globally
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,  # Set to False for smoother video stream processing
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# ------------------------------------------------------------- image decoding
def decode_data_url(data_url: str) -> np.ndarray:
    """'data:image/jpeg;base64,....' -> BGR numpy image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------- QR codes
def decode_qr(bgr_img: np.ndarray) -> Optional[str]:
    """Return the first QR payload found using PyZbar."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    results = pyzbar_decode(gray)
    return results[0].data.decode("utf-8", errors="ignore") if results else None


# --------------------------------------------------------------- face matching
def encode_face(bgr_img: np.ndarray) -> Optional[np.ndarray]:
    """Return a single 128-d face encoding, or None if not exactly one face."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb, model="hog")
    return encs[0] if len(encs) == 1 else None


def identify(encoding: np.ndarray, known_dict: dict) -> Optional[dict]:
    if not known_dict:
        return None

    ids = list(known_dict.keys())
    known_encodings = list(known_dict.values())

    # `face_recognition.face_distance` replaces custom np.linalg.norm math previously used. It returns a list of distances between the input encoding and each known encoding.
    dists = face_recognition.face_distance(known_encodings, encoding)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])

    if best_dist <= config.FACE_MATCH_TOLERANCE:
        conf = max(0.0, 1.0 - (best_dist / config.FACE_MATCH_TOLERANCE))
        return {"user_id": ids[best_idx], "distance": best_dist, "confidence": round(conf, 3)}
    return None


# ------------------------------------------------------------------- liveness
def landmarks_metrics(bgr_img: np.ndarray) -> Optional[dict]:
    """Return {'mar':..., 'ear':...} from a single frame, or None if no face."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None

    h, w = bgr_img.shape[:2]
    lm = res.multi_face_landmarks[0].landmark

    # Helper to calculate euclidean distance between two landmark indices
    def dist(p1, p2):
        return np.hypot((lm[p1].x - lm[p2].x) * w, (lm[p1].y - lm[p2].y) * h)

    mouth_h = dist(13, 14)          # Inner lip top to bottom
    mouth_w = dist(61, 291) or 1e-6 # Mouth corners left to right
    mar = mouth_h / mouth_w

    eye_h = dist(159, 145)          # Left eye top to bottom
    eye_w = dist(33, 133) or 1e-6   # Left eye corners left to right
    ear = eye_h / eye_w

    return {"mar": mar, "ear": ear}


@dataclass
class LivenessTracker:
    # 

    # deque(maxlen=X) automatically pops old frames when full - no manual list management!
    mars: deque = field(default_factory=lambda: deque(maxlen=config.LIVENESS_FRAME_WINDOW))
    ears: deque = field(default_factory=lambda: deque(maxlen=config.LIVENESS_FRAME_WINDOW))
    blinked: bool = False
    verified: bool = False

    def add(self, metrics: dict):
        if self.verified:
            return  # nothing left to measure this transaction
        self.mars.append(metrics["mar"])
        self.ears.append(metrics["ear"])
        if metrics["ear"] < config.LIVENESS_BLINK_EAR:
            self.blinked = True

    def reset(self):
        """Clear everything for the next customer."""
        self.mars.clear()
        self.ears.clear()
        self.blinked = False
        self.verified = False

    def status(self) -> dict:
        n = len(self.mars)

        # Already passed: hold the result rather than re-testing a stationary face.
        if self.verified:
            return {
                "live": True,
                "reason": "verified",
                "frames": n,
                "blinked": self.blinked,
                "latched": True,
            }

        if n < config.LIVENESS_MIN_FRAMES:
            return {"live": False, "reason": "collecting", "frames": n, "latched": False}

        var = statistics.pvariance(self.mars)
        moving = var >= config.LIVENESS_MAR_VARIANCE
        blink_ok = self.blinked or not config.LIVENESS_REQUIRE_BLINK

        live = moving and blink_ok
        reason = "ok" if live else ("static_face" if not moving else "no_blink")

        if live:
            self.verified = True  # latch the pass

        return {
            "live": live,
            "reason": reason,
            "frames": n,
            "mar_variance": round(var, 6),
            "blinked": self.blinked,
            "latched": False,
        }