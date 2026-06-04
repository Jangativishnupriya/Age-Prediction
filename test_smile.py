import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("models/smile_model.keras")
print("✅ Model loaded successfully!")

def load_images_in_batch(folder_path, target_size=(128,128)):
    images = []
    paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                img_path = os.path.join(root, f)
                try:
                    img = Image.open(img_path).convert('RGB').resize(target_size)
                    images.append(np.array(img)/255.0)
                    paths.append(img_path)
                except:
                    continue
    return np.array(images, dtype=np.float32), paths

def batch_predict(images, paths, model, batch_size=32):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        preds = model.predict(batch)
        for j, p in enumerate(preds):
            class_idx = 1 if p[0] >= 0.5 else 0
            classes = {0:"NotSmile", 1:"Smile"}
            confidence = p[0] if class_idx==1 else 1-p[0]
            results.append((paths[i+j], classes[class_idx], confidence))
            print(f"{paths[i+j]} -> {classes[class_idx]} (Confidence: {confidence:.2f})")
    return results

if __name__ == "__main__":
    test_folder_path = "Datasets/SmileSplit/SmileSplit/Test/NotSmile"
    X_test, paths = load_images_in_batch(test_folder_path)
    print(f"Loaded {len(X_test)} images for testing.")
    test_results = batch_predict(X_test, paths, model, batch_size=64)
    print("Batch prediction completed successfully!")