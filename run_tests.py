# Runs every test suite in tests/ and reports one combined result.
# All Testing for my project

# Run with:  python3 run_tests.py


# single process would let one suite's setup leak into another's.
import os
import subprocess
import sys

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")

SUITES = [
    "test_db.py",              # accounts, credit and money handling
    "test_checkout_flow.py",   # the whole flow, from a face to a payment
]


def run_suite(name):
    # Run one suite and return (passed, failed). Its own output is printed as it goes.
    path = os.path.join(TESTS_DIR, name)
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    print(result.stdout.rstrip())

    # Every suite ends with a line like "23 passed, 0 failed".
    for line in reversed(result.stdout.strip().split("\n")):
        if "passed," in line:
            parts = line.replace(",", "").split()
            return int(parts[0]), int(parts[2])

    print(f"  (no result line — suite crashed)")
    print(result.stderr.rstrip()[-500:])
    return 0, 1


def main():
    total_failed = 0
    total_passed = 0
    

    for name in SUITES:
        passed, failed = run_suite(name)
        total_passed += passed
        total_failed += failed

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed:
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
