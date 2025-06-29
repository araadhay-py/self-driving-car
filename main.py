import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

def preprocess(image):
    image = cv2.resize(image, (200, 66))  # Input shape expected by Nvidia model
    image = image / 255.0
    return image

cap = cv2.VideoCapture("test_video.mp4")  # Replace with 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess(frame)
    prediction = model.predict(np.expand_dims(processed, axis=0))
    steering_angle = prediction[0][0]

    # Display
    cv2.putText(frame, f"Steering: {steering_angle:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Self-Driving Simulator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
