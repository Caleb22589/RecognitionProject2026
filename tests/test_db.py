"""Tests for the account and credit database.

Run with:  python3 test_db.py
Uses a throwaway database file so the real kiosk.db is never touched.
"""
import os
import tempfile

import numpy as np

import config

# Point config at a temporary file BEFORE db reads it.
_handle, config.DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_handle)
os.remove(config.DB_PATH)

import db  # noqa: E402  (must come after DB_PATH is redirected)

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


def expect_error(description, func, *args, **kwargs):
    """Assert that a call is rejected rather than quietly accepted."""
    try:
        func(*args, **kwargs)
    except db.AccountError as error:
        check(f"{description} -> rejected ({error})", True)
    else:
        check(f"{description} -> should have been rejected", False)


def fake_encoding(seed):
    """A deterministic stand-in for a 128-d face encoding."""
    return np.random.default_rng(seed).random(db.ENCODING_LENGTH)


def main():
    db.init_db()

    print("\nExpected cases")
    alice = db.create_customer("Alice Smith", fake_encoding(1), opening_balance=20.00)
    check("new account starts on the signup balance", alice["balance_cents"] == 2000)

    result = db.charge(alice["id"], 12.35, "B-1042")
    check("charging 12.35 leaves 7.65", result["balance_cents"] == 765)
    check("money formats back to dollars", db.format_money(765) == "$7.65")

    db.top_up(alice["id"], 5.00)
    check("top up adds credit", db.get_customer(alice["id"])["balance_cents"] == 1265)

    check("transactions are recorded", len(db.recent_transactions(alice["id"])) == 3)

    bob = db.create_customer("Bob Jones", fake_encoding(2))
    known = db.load_known_encodings()
    check("both accounts load for matching", set(known) == {alice["id"], bob["id"]})
    check("encoding survives the round trip",
          np.allclose(known[alice["id"]], fake_encoding(1)))

    print("\nBoundary cases")
    carol = db.create_customer("Carol", fake_encoding(3), opening_balance=10.00)
    spent = db.charge(carol["id"], 10.00, "EXACT")
    check("spending the exact balance is allowed", spent["balance_cents"] == 0)
    expect_error("spending 1c more than the balance", db.charge, carol["id"], 0.01)

    dave = db.create_customer("Dave", fake_encoding(4), opening_balance=1.00)
    db.charge(dave["id"], 0.01)
    check("one cent is charged exactly", db.get_customer(dave["id"])["balance_cents"] == 99)

    # 24.35 * 100 is 2434.9999... in binary floating point.
    erin = db.create_customer("Erin", fake_encoding(5), opening_balance=100.00)
    db.charge(erin["id"], 24.35)
    check("float rounding does not lose a cent",
          db.get_customer(erin["id"])["balance_cents"] == 7565)

    check("shortest allowed name is accepted",
          db.create_customer("Al", fake_encoding(6))["name"] == "Al")
    long_name = "x" * db.MAX_NAME_LENGTH
    check("longest allowed name is accepted",
          db.create_customer(long_name, fake_encoding(7))["name"] == long_name)
    expect_error("a transaction over the kiosk limit",
                 db.charge, erin["id"], db.MAX_TRANSACTION_DOLLARS + 0.01)

    print("\nInvalid cases")
    expect_error("a name that is too short", db.create_customer, "A", fake_encoding(8))
    expect_error("a name that is too long",
                 db.create_customer, "x" * (db.MAX_NAME_LENGTH + 1), fake_encoding(9))
    expect_error("an encoding of the wrong length",
                 db.create_customer, "Wrong Size", np.zeros(64))
    expect_error("a negative amount", db.charge, alice["id"], -5.00)
    expect_error("a zero amount", db.charge, alice["id"], 0.00)
    expect_error("an amount that is not a number", db.charge, alice["id"], "free")
    expect_error("an unknown account id", db.charge, 9999, 1.00)
    expect_error("looking up an account that does not exist", db.get_customer, 9999)

    balance_before = db.get_customer(alice["id"])["balance_cents"]
    try:
        db.charge(alice["id"], 999999.00)
    except db.AccountError:
        pass
    check("a rejected charge leaves the balance untouched",
          db.get_customer(alice["id"])["balance_cents"] == balance_before)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if os.path.exists(config.DB_PATH):
            os.remove(config.DB_PATH)
