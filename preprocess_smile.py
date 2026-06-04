import os
import numpy as np
from PIL import Image

def preprocess_smile(dataset_path, target_size=(128,128)):
    images, labels = [], []
    classes = sorted(os.listdir(dataset_path))  # ['notsmile', 'smile']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    print("Classes detected:", class_to_idx)

    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                try:
                    img = Image.open(os.path.join(folder,f)).convert('RGB').resize(target_size)
                    images.append(np.array(img)/255.0)
                    labels.append(class_to_idx[cls])
                except:
                    continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    np.save("Datasets/smile_images.npy", images)
    np.save("Datasets/smile_labels.npy", labels)
    print(f"✅ Smile dataset saved: images={images.shape}, labels={labels.shape}")
    return images, labels

if __name__ == "__main__":
    train_path = "Datasets/SmileSplit/SmileSplit/Train"
    preprocess_smile(train_path)
    print("Preprocessing finished successfully!")
