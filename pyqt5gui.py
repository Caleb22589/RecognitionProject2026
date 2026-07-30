import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# backend script with all the face recognition and QR code decoding logic
import face_engine as backend

class SelfCheckoutGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QR & Biometric Self Service Checkout")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize Liveness Tracker from backend
        self.liveness_tracker = backend.LivenessTracker()
        
        self.initUI()
        self.initCamera()

    def initUI(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # camera feed
        left_panel = QVBoxLayout()
        self.camera_label = QLabel("Camera Feed Initializing...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.camera_label.setFixedSize(640, 480)
        left_panel.addWidget(self.camera_label)
        
        # Info panel
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignTop)
        
        title_font = QFont("Arial", 16, QFont.Bold)
        info_font = QFont("Arial", 12)

        title = QLabel("Checkout Terminal")
        title.setFont(title_font)
        right_panel.addWidget(title)
        
        # QR Code Status
        self.qr_status_label = QLabel("Waiting for QR Code...")
        self.qr_status_label.setFont(info_font)
        self.qr_status_label.setStyleSheet("color: blue; padding-top: 20px;")
        right_panel.addWidget(self.qr_status_label)

        # Liveness Status - check if image is not static and update accordingly
        self.liveness_label = QLabel("Liveness: Checking...")
        self.liveness_label.setFont(info_font)
        self.liveness_label.setStyleSheet("color: orange;")
        right_panel.addWidget(self.liveness_label)

        # Transaction Log - see transactions and QR scans
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        right_panel.addWidget(QLabel("Transaction Logs:"))
        right_panel.addWidget(self.log_box)

        # Action Buttons
        self.btn_reset = QPushButton("Reset Transaction")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self.reset_transaction)
        right_panel.addWidget(self.btn_reset)

        # Combine Layouts
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_box.append("Error: Could not open camera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Timer to fetch frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30) # ~33 fps

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 1. Check for QR Codes
        qr_data = backend.decode_qr(frame)
        if qr_data:
            self.qr_status_label.setText(f"QR Scanned: {qr_data}")
            self.qr_status_label.setStyleSheet("color: green; font-weight: bold;")
            # To avoid spamming the log, only log if it's new data
            if qr_data not in self.log_box.toPlainText():
                self.log_box.append(f"QR Detected: {qr_data}")

        # 2. Check Facial Liveness
        metrics = backend.landmarks_metrics(frame)
        if metrics:
            self.liveness_tracker.add(metrics)
            status = self.liveness_tracker.status()
            
            if status['live']:
                self.liveness_label.setText("Liveness: Verified (Live)")
                self.liveness_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.liveness_label.setText(f"Liveness: {status['reason']} ({status['frames']} frames)")
                self.liveness_label.setStyleSheet("color: red;")
        else:
            self.liveness_label.setText("Liveness: No face detected")
            self.liveness_label.setStyleSheet("color: orange;")

        # Convert Opencv BGR frame to pyqt QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))

    def reset_transaction(self):
        """Clears the current state for the next customer."""
        self.qr_status_label.setText("Waiting for QR Code...")
        self.qr_status_label.setStyleSheet("color: blue;")
        self.liveness_tracker = backend.LivenessTracker() # Reset liveness buffer
        self.log_box.append("Transaction Reset")

    def closeEvent(self, event):
        """Clean up camera on exit."""
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SelfCheckoutGUI()
    window.show()
    sys.exit(app.exec_())f