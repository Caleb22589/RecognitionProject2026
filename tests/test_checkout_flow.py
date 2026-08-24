# Tests for the checkout flow: recognise a face, log in, pay from stored credit.
#
# Run with:  python3 test_checkout_flow.py
# No camera is needed. 
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import config

# Redirect the database to a throwaway file before db reads config.DB_PATH.
_handle, config.DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_handle)
os.remove(config.DB_PATH)

import db      # noqa: E402
import main    # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

PASSED = 0
FAILED = 0


def check(description, condition):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


def fake_encoding(seed):
    return np.random.default_rng(seed).random(db.ENCODING_LENGTH)


def look_at_camera(terminal, customer_id, confidence=0.9):
    # Stand in for the worker: a live face that matched an account.
    terminal.on_liveness({"live": True, "reason": "ok", "frames": config.LIVENESS_MIN_FRAMES})
    terminal.on_identity({"user_id": customer_id, "confidence": confidence})


def run_tests():
    db.init_db()
    main.VisionWorker.run = lambda self: None   # the camera never opens in a test

    app = QApplication([])                      # noqa: F841  (Qt needs one alive)
    terminal = main.CheckoutTerminal()
    terminal.show()

    print("\nExpected cases")
    terminal.on_liveness({"live": True, "reason": "ok", "frames": 6})
    check("an unknown live face is offered sign up", terminal.btn_enrol.isVisible())
    check("banner asks the shopper to sign up", terminal.banner.text() == "FACE NOT RECOGNISED")

    alice = db.create_customer("Alice Smith", fake_encoding(1), opening_balance=20.00)
    terminal.worker.set_known(db.load_known_encodings())

    terminal.on_identity({"user_id": alice["id"], "confidence": 0.91})
    check("recognised shopper is logged in", terminal.account["id"] == alice["id"])
    check("their credit is shown", "$20.00" in terminal.account_row.value.text())
    check("sign up is hidden once logged in", not terminal.btn_enrol.isVisible())

    terminal.on_qr("B-1042|12.35")
    check("basket total is read off the code", terminal.basket_total == 12.35)
    check("terminal is ready to pay", terminal.banner.text() == "READY TO PAY")
    check("pay button shows the amount", "12.35" in terminal.btn_finish.text())

    terminal.complete_checkout()
    check("credit is debited in the database",
          db.get_customer(alice["id"])["balance_cents"] == 765)
    check("terminal clears for the next customer", terminal.account is None)

    print("\nBoundary cases")
    look_at_camera(terminal, alice["id"])
    terminal.on_qr("EXACT|7.65")
    check("a basket equal to the balance is allowed", terminal.btn_finish.isEnabled())
    terminal.complete_checkout()
    check("balance lands exactly on zero",
          db.get_customer(alice["id"])["balance_cents"] == 0)

    db.top_up(alice["id"], 5.00)
    look_at_camera(terminal, alice["id"])
    terminal.on_qr("OVER|5.01")
    check("one cent over the balance is refused", not terminal.btn_finish.isEnabled())
    check("banner explains why", terminal.banner.text() == "NOT ENOUGH CREDIT")

    terminal.reset("next")
    look_at_camera(terminal, alice["id"])
    terminal.on_qr("PLAIN-CODE")
    check("a code with no total blocks payment", not terminal.btn_finish.isEnabled())
    check("banner asks for the total", terminal.banner.text() == "BASKET TOTAL NEEDED")
    terminal.amount_input.setText("2.00")
    check("a typed total unblocks payment", terminal.btn_finish.isEnabled())

    print("\nInvalid cases")
    terminal.amount_input.setText("abc")
    check("a non-numeric total is not accepted", not terminal.btn_finish.isEnabled())
    terminal.amount_input.setText("-5.00")
    check("a negative total is not accepted", not terminal.btn_finish.isEnabled())

    terminal.reset("next")
    balance_before = db.get_customer(alice["id"])["balance_cents"]
    terminal.complete_checkout()
    check("paying with nobody logged in does nothing",
          db.get_customer(alice["id"])["balance_cents"] == balance_before)

    terminal.on_enrol_ready(None)
    check("sign up with no clear face is refused", terminal.account is None)

    terminal.on_identity({"user_id": 9999})
    check("a match on a deleted account does not log anyone in", terminal.account is None)

    terminal.worker.stop()
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        return 1
    else:
        return 0


if __name__ == "__main__":
    try:
        code = run_tests()
    finally:
        if os.path.exists(config.DB_PATH):
            os.remove(config.DB_PATH)
    sys.exit(code)
