import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

# Load preprocessed data
X_smile = np.load("Datasets/smile_images.npy")
y_smile = np.load("Datasets/smile_labels.npy")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_smile, y_smile, test_size=0.2, random_state=42, stratify=y_smile
)

# Compute class weights to handle imbalance
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weights)

# Optional: Data augmentation for smile images
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training with class weights and data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    class_weight=class_weights
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/smile_model.keras")
print("✅ Smile Detection Model saved!")
print("Training completed successfully!")