import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]

# Load both models
smile_model = tf.keras.models.load_model("models/smile_model.keras")
age_model = tf.keras.models.load_model("models/age_model.keras")

IMG_SIZE = (128, 128)
CAPTURE_DIR = "capture"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📸 Webcam started — smile to capture and predict your age...")

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, IMG_SIZE)
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        # Predict smile
        smile_pred = smile_model.predict(face_array, verbose=0)[0][0]
        label = "Smile" if smile_pred > 0.5 else "Not Smile"
        color = (0, 255, 0) if label == "Smile" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Capture & Predict Age only once on smile
        if label == "Smile" and not captured:
            captured = True
            image_path = os.path.join(CAPTURE_DIR, "captured.jpg")
            cv2.imwrite(image_path, face)

            # Predict Age
            img = image.load_img(image_path, target_size=IMG_SIZE)
            img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
            age_pred = age_model.predict(img_array, verbose=0)[0][0]

            # Determine Age Group
            if age_pred < 20:
                age_range = "6–20"
            elif age_pred < 30:
                age_range = "21–30"
            elif age_pred < 45:
                age_range = "31–45"
            elif age_pred < 60:
                age_range = "46–60"
            else:
                age_range = "61–98"

            print(f"🎯 Age Prediction: {age_pred:.2f} years → {age_range}")
            cv2.putText(frame, f"Age Range: {age_range}", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Smile & Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")
