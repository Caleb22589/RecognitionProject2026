import base64
import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from pyzbar.pyzbar import decode as pyzbar_decode

import config

# Initialize MediaPipe once globally
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,  # Set to False for smoother video stream processing
    max_num_faces=1, # We only expect one face in the selfcheckout kiosk
    refine_landmarks=True, #landmarks for iris and lips are more accurate with this enabled
    min_detection_confidence=0.5,#0.5 is the default, but can be adjusted based on lighting conditions and camera quality
)

# Image decoding
def decode_data_url(data_url: str) -> np.ndarray: 
    #data:image/jpeg;base64,....' -> BGR numpy image.
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# QR codes
def decode_qr(bgr_img: np.ndarray) -> Optional[str]:
    # Decode a qR code from a BGR image. Returns the decoded string or none if no qr code is found.
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    #pzbar decode would give results, only consider the first one
    results = pyzbar_decode(gray)
    if results:
        return results[0].data.decode("utf-8", errors="ignore")
    else:
        return None

# face recognition and liveness detection
def encode_face(bgr_img: np.ndarray) -> Optional[np.ndarray]:
    # face_recognition expects RGB, but OpenCV loads images as BGR, so the swap here is required
    # Return a 128d face encoding for the first face found, or None if no face or multiple faces are found.
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    #Face_recognition's face_encoding function.
    encs = face_recognition.face_encodings(rgb, model="hog")
    #if no encodings found, return None. Even with multiple faces return None.
    if len(encs) == 1:
        return encs[0]
    else:
        return None

# identify definition
#if the face encoding is found, it returns a 128d dimensional numpy array 
#representing the facial features of the detected face. If no face or 
# multiple faces are found, it returns None.
def identify(encoding: np.ndarray, known_dict: dict) -> Optional[dict]:
    if not known_dict:
        return None

    ids = list(known_dict.keys())
    known_encodings = list(known_dict.values())
    
    # `face_recognition.face_distance` replaces custom np.linalg.norm math previously used. It returns a list of distances between the input encoding and each known encoding.
    dists = face_recognition.face_distance(known_encodings, encoding)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    # If the best distance is within the tolerance, return the corresponding user ID and confidence score. Otherwise, return None.
    if best_dist <= config.FACE_MATCH_TOLERANCE:
        # Confidence is a simple linear mapping: 0 distance = 100% confidence, tolerance distance = 0% confidence. clamping to [0,1] to avoid negative confidence if the distance is slightly above tolerance.
        conf = max(0.0, 1.0 - (best_dist / config.FACE_MATCH_TOLERANCE))
        return {"user_id": ids[best_idx], "distance": best_dist, "confidence": round(conf, 3)}
    return None


# liveness
# we use MediaPipe's face mesh model to detect facial landmarks 
# in the input BGR image. We then calculate the Mouth Aspect Ratio 
# (MAR) and Eye Aspect Ratio (EAR) based on specific landmark points. 
# The MAR is calculated as the ratio of the vertical distance between 
# the inner lip landmarks to the horizontal distance between the mouth 
# corner landmarks. The EAR is calculated similarly for the left eye. 
# These metrics are returned in a dictionary for further processing 
# in liveness detection.
def landmarks_metrics(bgr_img: np.ndarray) -> Optional[dict]:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        # to make sure it doesn't use empty landmarks from previous frame, return None if no face is detected.
        return None

    h, w = bgr_img.shape[:2]
    lm = res.multi_face_landmarks[0].landmark

    # Helper to calculate euclidean distance between two landmark indices
    # This is just enough for a comparision when subject opens mouth or blinks. don't need to calculate it fully.
    # Right now only works with mouth.
    def dist(p1, p2):
        return np.hypot((lm[p1].x - lm[p2].x) * w, (lm[p1].y - lm[p2].y) * h)
    # fixed distance calculation for mouth and eye aspect ratios. 
    # The original code had a bug where it was using the wrong 
    # landmark indices for the mouth width calculation, 
    # which could lead to incorrect MAR values. 
    # The corrected indices now are accurately represent 
    # the inner lip top and bottom points for height, 
    # and the left and right mouth corners for width. 
    # This ensures that the MAR reflects the actual mouth opening, 
    # which is crucial for liveness detection.
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

    # deque(maxlen=X) automatically removes old frames when full so no manual list management!
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
        # Clear everything for the next customer.
        self.mars.clear()
        self.ears.clear()
        self.blinked = False
        self.verified = False

    def status(self) -> dict:
        n = len(self.mars)

        # Already passed: hold the result rather than re testing same face.
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

        #People have different shapes of lips and mouths, 
        # so the MAR can vary significantly between individuals.
        #Thats why we use variance instead of mean to determine if the mouth is moving.
        # Because mean could be misleading if the mouth is naturally wide or narrow, or if the person is speaking or making facial expressions.
        # while variance captures the actual movement over time. Deviation of their own mouth.
        var = statistics.pvariance(self.mars)
        moving = var >= config.LIVENESS_MAR_VARIANCE
        blink_ok = self.blinked or not config.LIVENESS_REQUIRE_BLINK

        live = moving and blink_ok
        if live:
            reason = "ok"
        elif not moving:
            reason = "static_face"
        else:
            reason = "no_blink"

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


# identity stabilisation
# One frame is not enough to log somebody in. A bad angle or a moment of motion
# blur can push an encoding just over FACE_MATCH_TOLERANCE and drop the match, or
# (worse) nudge it towards the wrong account. IdentityVoter only accepts an
# account once it has won IDENTITY_MIN_VOTES of the last IDENTITY_WINDOW frames,
# and then holds it for IDENTITY_HOLD_SECONDS so the name does not flicker while
# the shopper looks down at their basket.
@dataclass
class IdentityVoter:
    votes: deque = field(default_factory=lambda: deque(maxlen=config.IDENTITY_WINDOW))
    locked_id: Optional[int] = None
    last_seen: float = 0.0

    def add(self, user_id: Optional[int]) -> Optional[int]:
        # Record one frame's result and return the locked-in account, if any.
        self.votes.append(user_id)

        if user_id is not None:
            self.last_seen = time.time()

        if self.locked_id is not None:
            # Hold the current shopper until their face has been gone for a while.
            if time.time() - self.last_seen > config.IDENTITY_HOLD_SECONDS:
                self.reset()
            return self.locked_id

        # Count votes per candidate; ignore the None (no match) frames.
        tally = {}
        for vote in self.votes:
            if vote is not None:
                tally[vote] = tally.get(vote, 0) + 1

        for candidate, count in tally.items():
            if count >= config.IDENTITY_MIN_VOTES:
                self.locked_id = candidate
                return candidate
        return None

    def reset(self):
        self.votes.clear()
        self.locked_id = None
        self.last_seen = 0.0


def recognise(bgr_img: np.ndarray, known_dict: dict) -> Optional[dict]:
    # Encode the face in this frame and match it against the enrolled accounts.
    #
    # Returns identify()'s dict, or None when there is no usable face or no match
    if not known_dict:
        return None
    encoding = encode_face(bgr_img)
    if encoding is None:
        return None
    return identify(encoding, known_dict)


# top up vouchers
def parse_topup(payload: str) -> Optional[float]:
    # Recognise a top up voucher and return what it is worth in dollars, or None
    # if this code is not a voucher at all.
    #
    # A voucher is the basket code format with a reserved prefix instead of a
    # basket number, so the same scanner reads both:
    #     TOPUP|20.00
    #     TOPUP-20.00
    #     TOPUP 20.00
    text = (payload or "").strip()
    if not text:
        return None

    if not text.upper().startswith(config.TOPUP_PREFIX):
        return None

    # Whatever follows the prefix and its separator is the amount. Exactly one
    # separator is removed, not a run of them: "-" is both a valid separator and
    # the minus sign, so stripping every leading "-" would silently turn the
    # voucher "TOPUP|-5.00" into a five dollar credit.
    amount_text = text[len(config.TOPUP_PREFIX):]
    if amount_text and amount_text[0] in "|-: ":
        amount_text = amount_text[1:]
    amount_text = amount_text.strip()
    try:
        return float(amount_text)
    except ValueError:
        return None


# basket codes
def parse_basket(payload: str) -> Tuple[str, Optional[float]]:
    # Split a scanned code into (basket code, total in dollars).
    #
    # Three formats are accepted so the kiosk works with whatever the shop's label
    # printer produces:
    #     {"basket": "B-1042", "total': 24.50}   jSON
    #     B-1042|24.50                            pipe separated
    #     B-1042                                  code only, total unknown./
    text = (payload or "").strip()
    if not text:
        return "", None

    # JSON first, since a JSON payload could otherwise contain a "|".
    if text.startswith("{"):
        # Parsing the JSON and reading the total are two separate failures and
        # must be caught separately. If they share one try block, a well formed
        # code carrying a junk total falls into the same handler as unreadable
        # JSON, and the basket code is thrown away with it.
        try:
            data = json.loads(text)
        except ValueError:
            return text, None

        code = str(data.get("basket") or data.get("code") or "").strip()
        if not code:
            code = text

        total = data.get("total", data.get("amount"))
        if total is None:
            return code, None

        try:
            return code, float(total)
        except (ValueError, TypeError):
            # A code we can read carrying a price we cannot. Keep the code and
            # let the customer type the total, exactly as the pipe format does.
            return code, None

    for separator in ("|", ";"):
        if separator in text:
            code, _, amount = text.partition(separator)
            try:
                return code.strip(), float(amount.strip())
            except ValueError:
                return code.strip(), None

    return text, None