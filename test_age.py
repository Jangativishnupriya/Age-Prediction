import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image # type: ignore

# ====== CONFIG ======
MODEL_PATH = "best_age_model.keras"
DATASET_PATH = "balanced_age_dataset"
IMG_SIZE = (128, 128)

# ====== LOAD MODEL ======
print("🔹 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ====== CLASS LABELS ======
class_labels = sorted(os.listdir(DATASET_PATH))
print("🧾 Class labels:", class_labels)

# ====== TEST FUNCTION ======
def predict_age_range(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100
    
    print(f"\n🖼️ Image: {img_path}")
    print(f"➡️ Predicted Age Range: {class_labels[predicted_class]} ({confidence:.2f}% confidence)")

# ====== TEST SAMPLE ======
test_image_path = r"Datasets/age_dataset/utkface_aligned_cropped/UTKFace/32_0_0_20170117133448148.jpg.chip.jpg"  # 🔹 Change this path to your image
predict_age_range(test_image_path)
