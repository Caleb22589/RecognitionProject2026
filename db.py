"""SQLite storage for the kiosk: customer accounts, face encodings and store credit.

Money is stored as whole CENTS in an INTEGER column, never as a float. Floats
cannot represent values like 0.10 exactly, so repeatedly adding and subtracting
them slowly drifts a balance away from the true amount. Dollars are only used at
the edges of this module, where a human reads them.
"""
import sqlite3
import time

import numpy as np

import config

# A face_recognition encoding is always 128 doubles. Anything else is corrupt.
ENCODING_LENGTH = 128
ENCODING_DTYPE = np.float64

# Boundaries for user supplied values.
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 64
MAX_TRANSACTION_DOLLARS = 500.00

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    balance_cents INTEGER NOT NULL DEFAULT 0 CHECK (balance_cents >= 0),
    encoding      BLOB    NOT NULL,
    created_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL,
    amount_cents INTEGER NOT NULL,
    basket_code  TEXT,
    kind         TEXT    NOT NULL,
    created_at   REAL    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions (customer_id);
"""


# errors -------------------------------------------------------------------
class AccountError(Exception):
    """Anything the kiosk should show the customer rather than crash on."""


class InsufficientCredit(AccountError):
    pass


# connection ---------------------------------------------------------------
def connect() -> sqlite3.Connection:
    """Open the kiosk database with foreign keys on and rows accessible by name."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the tables if they do not exist yet. Safe to call on every start."""
    with connect() as conn:
        conn.executescript(SCHEMA)


# encoding <-> blob --------------------------------------------------------
def encoding_to_blob(encoding: np.ndarray) -> bytes:
    """Pack a 128-d encoding into raw bytes for the BLOB column."""
    arr = np.asarray(encoding, dtype=ENCODING_DTYPE)
    if arr.shape != (ENCODING_LENGTH,):
        raise AccountError(f"Face encoding must be {ENCODING_LENGTH} values, got {arr.shape}")
    return arr.tobytes()


def blob_to_encoding(blob: bytes) -> np.ndarray:
    """Unpack a stored BLOB back into a 128-d encoding."""
    arr = np.frombuffer(blob, dtype=ENCODING_DTYPE)
    if arr.shape != (ENCODING_LENGTH,):
        raise AccountError("Stored face encoding is corrupt")
    return arr


# validation ---------------------------------------------------------------
def clean_name(name: str) -> str:
    """Trim and length-check a customer name before it reaches the database."""
    cleaned = " ".join(str(name).split())  # collapse runs of whitespace
    if len(cleaned) < MIN_NAME_LENGTH:
        raise AccountError(f"Name must be at least {MIN_NAME_LENGTH} characters")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise AccountError(f"Name must be at most {MAX_NAME_LENGTH} characters")
    return cleaned


def to_cents(dollars: float) -> int:
    """Convert dollars to whole cents, rejecting anything that isn't sane money.

    round() before int() matters: 24.35 * 100 is 2434.9999... in binary floating
    point, so int() alone would silently charge a cent less.
    """
    try:
        value = float(dollars)
    except (TypeError, ValueError):
        raise AccountError("Amount is not a number")
    if value != value or value in (float("inf"), float("-inf")):  # NaN / infinity
        raise AccountError("Amount is not a number")
    if value <= 0:
        raise AccountError("Amount must be greater than zero")
    if value > MAX_TRANSACTION_DOLLARS:
        raise AccountError(f"Amount is above the {config.CURRENCY}{MAX_TRANSACTION_DOLLARS:.2f} kiosk limit")
    return int(round(value * 100))


def format_money(cents: int) -> str:
    """Render whole cents as a display string, e.g. 2035 -> '$20.35'."""
    return f"{config.CURRENCY}{cents / 100:.2f}"


# customers ----------------------------------------------------------------
def create_customer(name: str, encoding: np.ndarray, opening_balance: float = None) -> dict:
    """Enrol a new shopper with their face encoding and a starting balance."""
    name = clean_name(name)
    blob = encoding_to_blob(encoding)
    dollars = config.SIGNUP_BONUS if opening_balance is None else opening_balance
    cents = to_cents(dollars) if dollars > 0 else 0

    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO customers (name, balance_cents, encoding, created_at) VALUES (?, ?, ?, ?)",
            (name, cents, blob, time.time()),
        )
        customer_id = cur.lastrowid
        if cents:
            conn.execute(
                "INSERT INTO transactions (customer_id, amount_cents, basket_code, kind, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (customer_id, cents, None, "signup_bonus", time.time()),
            )
    return get_customer(customer_id)


def get_customer(customer_id: int) -> dict:
    """Look up one shopper. Raises if the id is not in the database."""
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, balance_cents, created_at FROM customers WHERE id = ?",
            (customer_id,),
        ).fetchone()
    if row is None:
        raise AccountError(f"No account with id {customer_id}")
    return dict(row)


def load_known_encodings() -> dict:
    """Return {customer_id: encoding} for every enrolled shopper.

    face_engine.identify() takes exactly this shape, so the kiosk loads it once
    at start up and refreshes it whenever somebody new enrols.
    """
    known = {}
    with connect() as conn:
        for row in conn.execute("SELECT id, encoding FROM customers"):
            try:
                known[row["id"]] = blob_to_encoding(row["encoding"])
            except AccountError:
                continue  # skip a corrupt row rather than take the whole kiosk down
    return known


def count_customers() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


# money --------------------------------------------------------------------
def charge(customer_id: int, dollars: float, basket_code: str = None) -> dict:
    """Take money off a shopper's credit and record the transaction.

    The read of the balance and the write of the new balance happen inside one
    IMMEDIATE transaction, so two tills cannot both read "$20 left" and each
    approve a $15 basket.
    """
    cents = to_cents(dollars)

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT balance_cents FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            raise AccountError(f"No account with id {customer_id}")

        balance = row["balance_cents"]
        if cents > balance:
            raise InsufficientCredit(
                f"Basket is {format_money(cents)} but the account only holds {format_money(balance)}"
            )

        remaining = balance - cents
        conn.execute("UPDATE customers SET balance_cents = ? WHERE id = ?", (remaining, customer_id))
        conn.execute(
            "INSERT INTO transactions (customer_id, amount_cents, basket_code, kind, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (customer_id, -cents, basket_code, "purchase", time.time()),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {"customer_id": customer_id, "charged_cents": cents, "balance_cents": remaining}


def top_up(customer_id: int, dollars: float) -> dict:
    """Add credit to an account (used for testing and for a future top-up screen)."""
    cents = to_cents(dollars)

    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT balance_cents FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            raise AccountError(f"No account with id {customer_id}")

        remaining = row["balance_cents"] + cents
        conn.execute("UPDATE customers SET balance_cents = ? WHERE id = ?", (remaining, customer_id))
        conn.execute(
            "INSERT INTO transactions (customer_id, amount_cents, basket_code, kind, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (customer_id, cents, None, "top_up", time.time()),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {"customer_id": customer_id, "balance_cents": remaining}


def recent_transactions(customer_id: int, limit: int = 10) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT amount_cents, basket_code, kind, created_at FROM transactions"
            " WHERE customer_id = ? ORDER BY id DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {config.DB_PATH} ({count_customers()} accounts)")
