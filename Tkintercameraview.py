import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
## Choppy tkinter camera view, maybe use pyqt as it needs to be converted to PIL, which is slow
class ChoppyCameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Tkinter Camera")
        
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
         
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Start the update loop
        self.delay = 1  # Milliseconds between frames (Roughly aiming for 60fps, but won't hit it)
        self.update_frame()

        self.window.mainloop()

    def update_frame(self):
        # 1. Read frame from webcam
        ret, frame = self.vid.read()
        
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. Convert to PIL Image
            pil_img = Image.fromarray(rgb_frame)
            
            # 4. Converting to ImageTk PhotoImage
            self.photo = ImageTk.PhotoImage(image=pil_img)
            
            # 5. Update canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
        self.window.after(self.delay, self.update_frame)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Run the app
if __name__ == "__main__":
    ChoppyCameraApp(tk.Tk())