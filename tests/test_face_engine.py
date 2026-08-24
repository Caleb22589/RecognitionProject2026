# Tests for the vision helpers: basket codes, liveness and identity voting.
#
# Run with:  python3 tests/test_face_engine.py
# No camera and no database are neded.
import math
import os
import sys
import time

# The tests live in tests/, one level below the modules they import, so the
# project root has to go on the import path before config, db, face_engine or
# main can be found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import face_engine as backend

PASSED = 0
FAILED = 0

# An eye-aspect-ratio comfortably above LIVENESS_BLINK_EAR, i.e. eyes open.
EYES_OPEN = 0.30
EYES_SHUT = config.LIVENESS_BLINK_EAR - 0.05


def check(description, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


def feed(tracker, mars, ear=EYES_OPEN):
    # Push a list of mouth-aspect-ratios through a tracker, one frame each.
    for mar in mars:
        tracker.add({"mar": mar, "ear": ear})
    return tracker


def mars_with_variance(count, variance, centre=0.05):
    # An even-length MAR run whose population variance is `variance`.
    #
    # Alternating evenly between centre-h and centre+h gives a mean of centre and
    # a pvariance of h**2, so h = sqrt(variance) puts the run exactly where the
    # test wants it relative to LIVENESS_MAR_VARIANCE.
    half = math.sqrt(variance)
    mars = []
    for i in range(count):
        if i % 2 == 0:
            mars.append(centre - half)
        else:
            mars.append(centre + half)
    return mars


def flat_mars(count, value=0.05):
    # A printed photo: the mouth never changes shape, so variance is zero.
    return [value] * count


def run_tests():
    print("\nExpected cases")

    # -- basket codes --
    check("a pipe separated code splits into code and total",
          backend.parse_basket("B-1042|24.50") == ("B-1042", 24.50))
    check("a JSON code splits into code and total",
          backend.parse_basket('{"basket": "B-1042", "total": 24.50}') == ("B-1042", 24.50))
    check("a code with no total returns the code alone",
          backend.parse_basket("B-1042") == ("B-1042", None))
    check("surrounding whitespace is stripped",
          backend.parse_basket("   B-1042   ") == ("B-1042", None))

    # -- liveness --
    live = feed(backend.LivenessTracker(),
                mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                   config.LIVENESS_MAR_VARIANCE * 4)).status()
    check("a moving mouth is accepted as live", live["live"] and live["reason"] == "ok")

    static = feed(backend.LivenessTracker(),
                  flat_mars(config.LIVENESS_MIN_FRAMES)).status()
    check("a static face is rejected",
          not static["live"] and static["reason"] == "static_face")

    # -- identity voting --
    voter = backend.IdentityVoter()
    for _ in range(config.IDENTITY_MIN_VOTES - 1):
        voter.add(7)
    check("one vote short of the threshold locks nobody in", voter.locked_id is None)
    check("the deciding vote locks the account in", voter.add(7) == 7)

    print("\nBoundary cases")

    # -- liveness: the frame count needed before a decision is made --
    nearly = feed(backend.LivenessTracker(),
                  mars_with_variance(config.LIVENESS_MIN_FRAMES - 1,
                                     config.LIVENESS_MAR_VARIANCE * 4)).status()
    check(f"{config.LIVENESS_MIN_FRAMES - 1} frames is still collecting",
          not nearly["live"] and nearly["reason"] == "collecting")

    enough = feed(backend.LivenessTracker(),
                  mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                     config.LIVENESS_MAR_VARIANCE * 4)).status()
    check(f"{config.LIVENESS_MIN_FRAMES} frames is enough to decide",
          enough["reason"] != "collecting")

    # -- liveness: either side of the movement threshold --
    # has to be 10% either side rather than exactly on it using pvariance of a
    # float  lands a rounding error away from the target, so an exact
    # test would pass or fail on the last bit of the mantissa.
    under = feed(backend.LivenessTracker(),
                 mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                    config.LIVENESS_MAR_VARIANCE * 0.9)).status()
    check("movement just under the threshold is rejected", not under["live"])

    over = feed(backend.LivenessTracker(),
                mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                   config.LIVENESS_MAR_VARIANCE * 1.1)).status()
    check("movement just over the threshold is accepted", over["live"])

    # -- liveness: the rolling window must not grow without limit --
    long_run = feed(backend.LivenessTracker(),
                    flat_mars(config.LIVENESS_FRAME_WINDOW * 3))
    check(f"the window holds at most {config.LIVENESS_FRAME_WINDOW} frames",
          len(long_run.mars) == config.LIVENESS_FRAME_WINDOW)

    # -- liveness: a pass is latched, so looking away cannot un-verify a customer --
    latched = feed(backend.LivenessTracker(),
                   mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                      config.LIVENESS_MAR_VARIANCE * 4))
    latched.status()                                  # first pass sets verified
    feed(latched, flat_mars(config.LIVENESS_FRAME_WINDOW))   # now hold perfectly still
    held = latched.status()
    check("a verified customer stays verified when they stop moving",
          held["live"] and held["latched"])

    # -- liveness: a blink is noticed even though it is not currently required --
    blinker = backend.LivenessTracker()
    feed(blinker, flat_mars(3), ear=EYES_OPEN)
    feed(blinker, flat_mars(1), ear=EYES_SHUT)
    check("a blink is recorded", blinker.blinked)

    # -- identity: votes split between two people must not lock either in --
    split = backend.IdentityVoter()
    for user_id in (1, 2, 1, 2):
        split.add(user_id)
    check("votes split between two faces lock nobody in", split.locked_id is None)
    check("the first face to reach the threshold wins", split.add(1) == 1)

    # -- identity: the lock is released once the face has been gone long enough --
    leaving = backend.IdentityVoter()
    for _ in range(config.IDENTITY_MIN_VOTES):
        leaving.add(3)
    check("the shopper is locked in before they leave", leaving.locked_id == 3)
    leaving.add(None)
    check("a single missed frame does not drop the shopper", leaving.locked_id == 3)
    # Wind the clock back rather than sleeping through IDENTITY_HOLD_SECONDS.
    leaving.last_seen = time.time() - (config.IDENTITY_HOLD_SECONDS + 1)
    check("an absent face is released after the hold expires",
          leaving.add(None) is None and leaving.locked_id is None)

    print("\nInvalid cases")

    check("an empty payload yields no code and no total",
          backend.parse_basket("") == ("", None))
    check("a missing payload yields no code and no total",
          backend.parse_basket(None) == ("", None))
    check("whitespace only yields no code and no total",
          backend.parse_basket("     ") == ("", None))

    code, total = backend.parse_basket('{"basket": "B-9", "total":')
    check("malformed JSON is kept as a plain code with no total", total is None and code)

    check("a non-numeric total after the separator is dropped",
          backend.parse_basket("B-1042|free") == ("B-1042", None))
    check("a second separator makes the total unreadable, so it is dropped",
          backend.parse_basket("B-1042|24.50|extra") == ("B-1042", None))
    check("a JSON total that is not a number is dropped",
          backend.parse_basket('{"basket": "B-9", "total": "free"}') == ("B-9", None))

    empty_tracker = backend.LivenessTracker().status()
    check("a tracker that has seen no frames is not live",
          not empty_tracker["live"] and empty_tracker["frames"] == 0)

    reset_me = feed(backend.LivenessTracker(),
                    mars_with_variance(config.LIVENESS_MIN_FRAMES,
                                       config.LIVENESS_MAR_VARIANCE * 4))
    reset_me.status()
    reset_me.reset()
    check("reset clears a verified pass for the next customer",
          not reset_me.verified and len(reset_me.mars) == 0)

    ignores_none = backend.IdentityVoter()
    for _ in range(config.IDENTITY_WINDOW):
        ignores_none.add(None)
    check("frames with no match never lock an account in",
          ignores_none.add(None) is None and ignores_none.locked_id is None)

    check("matching against an empty account list returns nothing",
          backend.identify([0.0] * 128, {}) is None)

    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
