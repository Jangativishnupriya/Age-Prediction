import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image

# ====== CONFIG ======
MODEL_PATH = "best_age_model.keras"
IMG_SIZE = (128, 128)  # must match MobileNetV2 input
AGE_LABELS = ["0_10", "11_20", "21_30", "31_40", "41_50", "51_100"]

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("✅ Model loaded successfully (MobileNetV2)")

# ====== UI ======
st.title("🎥 Real-Time Age Range Prediction (Smile Detection Enabled)")
st.markdown("Upload a photo or use your webcam — prediction happens **only when you smile! 😄**")

# ====== Helper Functions ======
def predict_age(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_label = AGE_LABELS[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_label, confidence, img_rgb

def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return True
    return False

# ====== IMAGE UPLOAD SECTION ======
option = st.radio("Choose Input Mode:", ("📸 Use Webcam (Smile to Capture)", "🖼️ Upload Image"))

# ====== WEBCAM MODE ======
if option == "📸 Use Webcam (Smile to Capture)":
    st.info("👉 The app will capture and predict automatically **only when you smile!**")

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        smiling = detect_smile(frame)

        # Draw smile detection box
        display_text = "😄 Smile Detected! Capturing..." if smiling else "🙂 Please Smile!"
        color = (0, 255, 0) if smiling else (0, 0, 255)
        cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Show live video feed
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if smiling:
            label, conf, img_rgb = predict_age(frame)
            st.success(f"🧠 Predicted Age Range: **{label}** ({conf:.2f}%)")
            st.image(img_rgb, caption=f"Predicted: {label} ({conf:.1f}%)", use_column_width=True)
            break

    cap.release()

# ====== UPLOAD MODE ======
else:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        label, conf, img_rgb = predict_age(img)

        st.image(img_rgb, caption=f"Predicted: {label} ({conf:.1f}%)", use_column_width=True)
        st.success(f"🧠 Predicted Age Range: **{label}** ({conf:.2f}%)")

st.caption("Model: MobileNetV2 | Input: 128×128 | Smile-triggered Real-Time Prediction using Streamlit 🎥")
