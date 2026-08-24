# Archive

Earlier versions and experiments kept for reference. **None of these are part of
the working program** — the kiosk is `main.py` in the folder above.

| File | What it was |
|------|-------------|
| `OldGUI.py` | First attempt at the interface in tkinter, before moving to PyQt5. Does not run — left mid-edit. |
| `ChoppyTkintercameraview.py` | Getting a webcam feed into a tkinter window. Named for how the video looked. |
| `FaceRecognitiontest.py` | Proving `face_recognition` could encode and match a face at all, before any GUI existed. |
| `test_face_engine.py` | Extra tests for the liveness and barcode parsing helpers. Cut back — the two suites in `tests/` cover the program. |
| `test_scanner.py` | Extra tests that generated QR codes to check the decoder. Cut for the same reason. |
| `pricehullucinator.py` | An attempt at predicting item prices without an internet lookup. Emptied out — the idea is described under "possible extensions" in the main README. |
