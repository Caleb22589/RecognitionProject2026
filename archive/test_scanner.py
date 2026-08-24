# Tests for the basket code scanner.
#
# Run with:  python3 test_scanner.py
# No camera and no printed code are needed
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import face_engine as backend

PASSED = 0
FAILED = 0

# The camera in VisionWorker is asked for 1280x720, so tests use the same size.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BACKGROUND_GREY = 130

PLAIN_CODE = "B-4000"
PIPE_CODE = "B-1042|24.50"
JSON_CODE = '{"basket": "B-3050", "total": 12.99}'


def check(description, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


def make_qr(payload, scale=8, border=3):
    # Encode a payload as a real QR code and return it as a BGR image.
    #
    # The quiet zone border and the nearest-neighbour resize both matter: a QR
    # code with no margin, or one scaled with smoothing, stops being readable.
    encoder = cv2.QRCodeEncoder_create()
    code = encoder.encode(payload)
    code = cv2.copyMakeBorder(code, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=255)
    code = cv2.resize(code, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)


def in_camera_frame(code, fraction=0.35):
    # Paste a code into a blank camera-sized frame, filling `fraction` of it.
    #
    # The fraction is measured against the SHORTER side of the frame. A QR code is
    # square, so scaling it off the 1280 width would make a large one taller than
    # the 720 height and it would not fit.
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND_GREY, dtype=np.uint8)
    target = int(min(FRAME_WIDTH, FRAME_HEIGHT) * fraction)
    ratio = target / max(code.shape[0], code.shape[1])
    resized = cv2.resize(code, (int(code.shape[1] * ratio), int(code.shape[0] * ratio)))
    top = (FRAME_HEIGHT - resized.shape[0]) // 2
    left = (FRAME_WIDTH - resized.shape[1]) // 2
    frame[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return frame


def rotate(frame, degrees):
    # Turn a frame, as if the customer held the code at an angle.
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(frame, matrix, (width, height),borderValue=(BACKGROUND_GREY,) * 3)


def blank_frame():
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND_GREY, dtype=np.uint8)


def run_tests():
    print("\nExpected cases")

    check("a code on its own is decoded",
          backend.decode_qr(make_qr(PLAIN_CODE)) == PLAIN_CODE)
    check("a pipe separated code and total is decoded",
          backend.decode_qr(make_qr(PIPE_CODE)) == PIPE_CODE)
    check("a JSON payload is decoded",
          backend.decode_qr(make_qr(JSON_CODE)) == JSON_CODE)
    check("a code held in the middle of a camera frame is decoded",
          backend.decode_qr(in_camera_frame(make_qr(PIPE_CODE))) == PIPE_CODE)

    print("\nBoundary cases")

    # A shopper will not hold the code square on, centred and close, so the
    # decoder has to cope with the awkward versions of the same code.
    held_close = in_camera_frame(make_qr(PIPE_CODE), fraction=0.60)
    check("held close to the camera (60% of the frame)",
          backend.decode_qr(held_close) == PIPE_CODE)

    held_far = in_camera_frame(make_qr(PIPE_CODE), fraction=0.12)
    check("held far from the camera (12% of the frame)",
          backend.decode_qr(held_far) == PIPE_CODE)

    check("held at a 15 degree angle",
          backend.decode_qr(rotate(in_camera_frame(make_qr(PIPE_CODE)), 15)) == PIPE_CODE)
    check("held upside down",
          backend.decode_qr(rotate(in_camera_frame(make_qr(PIPE_CODE)), 180)) == PIPE_CODE)

    out_of_focus = cv2.GaussianBlur(in_camera_frame(make_qr(PIPE_CODE)), (7, 7), 0)
    check("slightly out of focus", backend.decode_qr(out_of_focus) == PIPE_CODE)

    # A dim shop is the realistic version of this, not a pitch black one.
    dim = (in_camera_frame(make_qr(PIPE_CODE)).astype(np.float32) * 0.35).astype(np.uint8)
    check("dim lighting (35% brightness)", backend.decode_qr(dim) == PIPE_CODE)

    # A long payload pushes the QR code to a denser version of itself.
    long_payload = "B-9999|" + "9" * 60
    check("a long payload still decodes",
          backend.decode_qr(make_qr(long_payload)) == long_payload)

    print("\nInvalid cases")

    check("an empty frame decodes to nothing",
          backend.decode_qr(blank_frame()) is None)

    noise = np.random.default_rng(0).integers(
        0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    check("random camera noise decodes to nothing",
          backend.decode_qr(noise) is None)

    # Half a code is not a code. Blanking the finder patterns should kill it.
    torn = in_camera_frame(make_qr(PIPE_CODE))
    torn[:, :FRAME_WIDTH // 2] = BACKGROUND_GREY
    check("a half covered code decodes to nothing",
          backend.decode_qr(torn) is None)

    print("\nFull pipeline: scan then parse")

    scanned = backend.decode_qr(in_camera_frame(make_qr(PIPE_CODE)))
    check("a scanned pipe code parses into code and total",
          backend.parse_basket(scanned) == ("B-1042", 24.50))

    scanned = backend.decode_qr(in_camera_frame(make_qr(JSON_CODE)))
    check("a scanned JSON code parses into code and total",
          backend.parse_basket(scanned) == ("B-3050", 12.99))

    scanned = backend.decode_qr(in_camera_frame(make_qr(PLAIN_CODE)))
    check("a scanned plain code parses with no total",
          backend.parse_basket(scanned) == ("B-4000", None))

    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
