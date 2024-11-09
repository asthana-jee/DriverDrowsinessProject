import numpy as np
import argparse
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
import cv2
import time

def apply_clahe_rgb(image):
    # Split the image into R, G, and B channels
    r, g, b = cv2.split(image)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))

    # Apply CLAHE to each channel
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)

    # Merge the channels back
    return cv2.merge((r, g, b))

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# construct the argument parser and parse the arguments
print('Loading the Detector...')
detector = load_model("Drowsy_Driver_Detection.h5")

vs = VideoStream(src=args['webcam']).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    display_frame = frame.copy()
    frame = cv2.resize(frame, (32, 32))  # Resize to model's input shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    frame_rgb = apply_clahe_rgb(frame_rgb)

    # Preprocessing the frame as done during model training
    frame_rgb = np.array(frame_rgb, dtype='float32') / 255.0
    frame_rgb = np.expand_dims(frame_rgb, axis=0)

    # Predict the frame
    prediction = detector.predict(frame_rgb)

    if np.argmax(prediction) in [0, 2]:
        cv2.putText(display_frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()