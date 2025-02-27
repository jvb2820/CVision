import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
#make sure to install all the imports before starting the app.

# Just copy the models in the folder and put it in the file you want.
prototxt_path = r'C:\Users\jeuzv\MobileNetSSD_deploy.prototxt.txt'
model_path = r'C:\Users\jeuzv\MobileNetSSD_deploy.caffemodel'

min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


class RealTimeObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Real-time Object Detection")
        self.master.geometry("900x700")
        self.master.configure(bg="#2C2F33")

        self.video_capture = cv2.VideoCapture(0)
        self.detecting = False

        # Top Frame (for centering buttons)
        self.top_frame = tk.Frame(master, bg="#2C2F33")
        self.top_frame.pack(fill="x", pady=5)

        # Centering the buttons using pack with expand=True
        self.buttons_frame = tk.Frame(self.top_frame, bg="#2C2F33")
        self.buttons_frame.pack(pady=5)

        # Start/Stop Detection Button
        self.start_stop_button = ttk.Button(self.buttons_frame, text="Start Detection", command=self.toggle_detection)
        self.start_stop_button.pack(side="left", padx=10)

        # Exit Button
        self.exit_button = ttk.Button(self.buttons_frame, text="Exit", command=self.close)
        self.exit_button.pack(side="left", padx=10)

        # Camera Feed Label
        self.canvas = tk.Label(master)
        self.canvas.pack(fill="both", expand=True)

        self.update()

    def toggle_detection(self):
        self.detecting = not self.detecting
        self.start_stop_button.config(text="Stop Detection" if self.detecting else "Start Detection")

    def update(self):
        ret, frame = self.video_capture.read()

        if ret:
            if self.detecting:
                height, width = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (500, 500)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > min_confidence:
                        class_id = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (startX, startY, endX, endY) = box.astype("int")

                        label = f"{classes[class_id]}: {confidence:.2f}%"
                        color = colors[class_id]
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert frame to display in Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            self.canvas.config(image=image)
            self.canvas.image = image

        self.master.after(10, self.update)

    def close(self):
        self.video_capture.release()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = RealTimeObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
