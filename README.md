# Self-checkout terminal with facial recognition

Making a self checkout terminal that uses facial recognition in order to detect the user and just by the user in frame, match to database and log in automatically using their own credits.

This is mainly made for small stands or shops that don't have good internet access. All face analysing is done locally — nothing in this program makes a network request.

So user can theoretically just show face until it verifies its real, present a basket code, and pay straight from their stored credit. Removing the absolute friction of a low value self checkout process (like foods). High value purchases should require more verification in practice.

Problems are mainly security, so i added a liveness detector that if the face is moving in a realistic pattern it would mark it as verified.

I like this idea as i want to experiment if truely just cameras on a phone or laptop are enough to replace dedicated self checkouts all together, and make things more seamless.

## Running it

```
pip install -r requirements.txt
python3 main.py          # the kiosk
python3 run_tests.py     # all four test suites
```

## Project layout

| Path | What it does |
|------|--------------|
| `main.py` | PyQt5 interface and the camera thread. Entry point. |
| `face_engine.py` | Face encoding, matching, liveness, identity voting, barcode decoding. |
| `db.py` | SQLite accounts, stored credit and transaction history. |
| `config.py` | Every tunable value — thresholds, window sizes, limits. |
| `tests/` | Four suites, 93 checks, no camera required. |
| `archive/` | Early prototypes, kept for reference. Not part of the program. |

## How it works

It takes a frame from the webcam and converts it from OpenCV's BGR colour order to RGB, which face_recognition and MediaPipe require.

For identification, face_recognition detects the face and produces a 128-number encoding of it. That encoding is compared against every stored encoding, and the closest match within the tolerance is returned as the user. If more than one face is in frame the encoding is thrown away, so two people cannot confuse the match.

One good frame is not enough to log somebody in. `IdentityVoter` requires the same account to win at least 3 of the last 6 frames before it locks in, then holds that account for 4 seconds so the name does not flicker while the shopper looks down at their basket.

For liveness, MediaPipe Face Mesh returns landmark points on the face. From fixed landmark indices I calculate two ratios: the mouth aspect ratio (mouth height ÷ width) and the eye aspect ratio (eye height ÷ width). These are collected over a rolling window of webcam frames.

The check uses the variance of the MAR rather than its average, because resting mouth shape differs between people — variance measures whether the value is changing, not what it is. A live face constantly moves, so its variance is above the threshold. A photo held to the camera gives an identical value every frame, so its variance is near zero and it fails.

A blink is detected when the EAR drops below a threshold, and it is recorded, but it is **not currently required** — `LIVENESS_REQUIRE_BLINK` is set to `False` in config.py, so mouth movement alone decides liveness. Turning it on is a one line change. Once a face passes, the result is latched and not re-tested for the rest of that transaction.

In practice it usually only passes if the mouth is moving noticeably.

Money is stored as whole cents in an INTEGER column, never as a float, because floats cannot represent values like 0.10 exactly and a balance would slowly drift.

## Improvements

Using mouth to verify transaction is kinda weird so calculate both EAR first and MAR as backup or altogether to make it more seamless.

The liveness indicator is not accurate enough to trust on its own, so ideally it would only gate high value transactions rather than every payment as it does now.

## Possible extensions

These are ideas for the project, **not built yet**:

- **Barcode item lookup.** The scanner already decodes real barcodes (EAN-13, Code128, QR), but it only reads a basket code and total. It could instead search an international barcode database to retrieve the exact item.
- **Local price prediction.** To reduce the use of the internet, a price algorithm or AI that would predict how much an item should cost in theory. So if it was a bag of doritos it would think a fair price to pay is $3.50, and maybe when stock is low increase the price. If AI controlled the price, it would be way smarter in balancing user experience vs actual profits in a real situation.

## Modules used

I have mainly used PyQt5 for the GUI, as its faster than other alternatives like tkinter, however does have disadvantages of its very inefficient way of creating elements (takes atleast 5 lines to create one button), so I made a `StatusRow` class that makes it simpler to add UI elements to my interface.

Pyzbar is a library primarly used to find barcodes and scan them. It decodes far more than QR codes — EAN-13 and Code128 both work, which is what a real product barcode uses.

face_recognition (dlib) does the 128-number face encoding, and MediaPipe Face Mesh provides the landmark points used for liveness.
