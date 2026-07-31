import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# backend script with all the face recognition and QR code decoding logic
import face_engine as backend

def main():
    app = QApplication(sys.argv)
    
    state = {
        "liveness_tracker": backend.LivenessTracker(),
        "cap": cv2.VideoCapture(0)
    }

    #  Setup
    window = QMainWindow()
    window.setWindowTitle("QR & Biometric Self Service Checkout")
    window.setGeometry(100, 100, 1000, 600)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    main_layout = QHBoxLayout(central_widget)

    # Camera Feed 
    left_panel = QVBoxLayout()
    camera_label = QLabel("Camera Feed Initializing...")
    camera_label.setAlignment(Qt.AlignCenter)
    camera_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
    camera_label.setFixedSize(640, 480)
    left_panel.addWidget(camera_label)
    
    # Info and Logs
    right_panel = QVBoxLayout()
    right_panel.setAlignment(Qt.AlignTop)
    
    title_font = QFont("Arial", 16, QFont.Bold)
    info_font = QFont("Arial", 12)

    title = QLabel("Checkout Terminal")
    title.setFont(title_font)
    right_panel.addWidget(title)
    
    qr_status_label = QLabel("Waiting for QR Code...")
    qr_status_label.setFont(info_font)
    qr_status_label.setStyleSheet("color: blue; padding-top: 20px;")
    right_panel.addWidget(qr_status_label)

    liveness_label = QLabel("Liveness: Checking...")
    liveness_label.setFont(info_font)
    liveness_label.setStyleSheet("color: orange;")
    right_panel.addWidget(liveness_label)

    log_box = QTextEdit()
    log_box.setReadOnly(True)
    right_panel.addWidget(QLabel("Transaction Logs:"))
    right_panel.addWidget(log_box)

    btn_reset = QPushButton("Reset Transaction")
    btn_reset.setMinimumHeight(40)
    right_panel.addWidget(btn_reset)

    # combinethe Layouts
    main_layout.addLayout(left_panel)
    main_layout.addLayout(right_panel)

    if not state["cap"].isOpened():
        log_box.append("Error: Could not open camera.")
    else:
        state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    
    def reset_transaction():
        """Clears the current state for the next customer."""
        qr_status_label.setText("Waiting for QR Code...")
        qr_status_label.setStyleSheet("color: blue;")
        state["liveness_tracker"] = backend.LivenessTracker() # Reset liveness buffer
        log_box.append("Transaction Reset")

    # Connect the button
    btn_reset.clicked.connect(reset_transaction)

    def process_frame():
        ret, frame = state["cap"].read()
        if not ret:
            return

        # 1. Check for QR Codes
        qr_data = backend.decode_qr(frame)
        if qr_data:
            qr_status_label.setText(f"QR Scanned: {qr_data}")
            qr_status_label.setStyleSheet("color: green; font-weight: bold;")
            if qr_data not in log_box.toPlainText():
                log_box.append(f"QR Detected: {qr_data}")

        # 2. Check Liveness
        metrics = backend.landmarks_metrics(frame)
        if metrics:
            state["liveness_tracker"].add(metrics)
            status = state["liveness_tracker"].status()
            
            if status['live']:
                liveness_label.setText("Liveness: Verified (Live)")
                liveness_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                liveness_label.setText(f"Liveness: {status['reason']} ({status['frames']} frames)")
                liveness_label.setStyleSheet("color: red;")
        else:
            liveness_label.setText("Liveness: No face detected")
            liveness_label.setStyleSheet("color: orange;")

        # 3. Update GUI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Optionally scale the image to fit the label perfectly
        camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(camera_label.width(), camera_label.height(), Qt.KeepAspectRatio))
    def custom_close_event(event):
        timer.stop()
        if state["cap"].isOpened():
            state["cap"].release()
        event.accept()
        
    window.closeEvent = custom_close_event

    timer = QTimer()
    timer.timeout.connect(process_frame)
    timer.start(30) # ~33 fps

    # Execute Application
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
