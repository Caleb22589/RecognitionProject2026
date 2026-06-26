import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,QVBoxLayout, QHBoxLayout, QWidget,QPushButton, QFrame)
from PyQt5.QtCore import Qt


class StatCard(QFrame):
    def __init__(self, title: str, initial: str = "—"):
        super().__init__()
        self.setFixedWidth(160)
        self.setStyleSheet("""
            QFrame {background: #1e1e2e; border: 1px solid #313244; border-radius: 10px;}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        lbl_t = QLabel(title.upper())
        lbl_t.setStyleSheet("color: #6c7086; font-size: 10px; letter-spacing: 1px; border: none;")

        self._val = QLabel(initial)
        self._val.setStyleSheet("color: #cdd6f4; font-size: 26px; font-weight: bold; border: none;")

        lay.addWidget(lbl_t)
        lay.addWidget(self._val)
# GUI
class MainWindow(QMainWindow):

    # Colour palette
    BG_BASE   = "#FFFFFF"
    BG_MANTLE = "#11111b"
    BG_SURFACE= "#1e1e2e"
    FG        = "#cdd6f4"
    OVERLAY   = "#313244"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face detection")
        self.setMinimumSize(1000, 680)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"QMainWindow, QWidget {{ background: {self.BG_BASE}; color: {self.FG}; }}")

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(22, 16, 22, 16)
        lay.setSpacing(14)

        # Header
        hdr = QLabel("Face Detection")
        hdr.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {self.FG};")
        lay.addWidget(hdr)
        def 
        # Video Placeholder Area would change after creating logic
        self._video = QLabel("Camera Feed Mockup")
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setStyleSheet(f"""background: {self.BG_MANTLE}; border-radius: 14px; color: #6c7086; font-size: 16px;""")
        lay.addWidget(self._video, stretch=1)

        # Bottom UI bar
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self._fps_card    = StatCard("FPS",            "")
        self._faces_card  = StatCard("Faces",          "0")
        self._status_card = StatCard("Status",         "Ready")

        bar.addWidget(self._fps_card)
        bar.addWidget(self._faces_card)
        bar.addWidget(self._status_card)
        bar.addStretch()

        # Action Button
        self._btn = QPushButton("▶  Start")
        self._btn.setFixedHeight(46)
        self._btn.setStyleSheet(f"""
        QPushButton {{
                background: {self.OVERLAY}; color: {self.FG};
                border: none; border-radius: 8px;
                padding: 0 28px; font-size: 13px;
            }}
            QPushButton:hover   {{ background: #45475a; }}
            QPushButton:pressed {{ background: #585b70; }}
        """)
        bar.addWidget(self._btn)

        lay.addLayout(bar)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())