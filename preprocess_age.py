import os
import random
import shutil
from tqdm import tqdm

# Define paths
DATASET_DIR = "Datasets/age_dataset/UTKFace"
OUTPUT_DIR = "balanced_age_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define age categories (ranges)
AGE_RANGES = {
    "0_10": range(0, 11),
    "11_20": range(11, 21),
    "21_30": range(21, 31),
    "31_40": range(31, 41),
    "41_50": range(41, 51),
    "51_100": range(51, 101)
}

# Collect all image paths
def get_all_images(base_folder):
    image_paths = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths

all_images = get_all_images(DATASET_DIR)
print(f"Total images found: {len(all_images)}")

# Sort images into bins based on age from filename (UTKFace format: age_gender_race.jpg)
age_bins = {k: [] for k in AGE_RANGES.keys()}

for img_path in all_images:
    try:
        age = int(os.path.basename(img_path).split("_")[0])
        for label, age_range in AGE_RANGES.items():
            if age in age_range:
                age_bins[label].append(img_path)
                break
    except Exception as e:
        continue

# Balance dataset (equal number of samples per class)
min_samples = min(len(imgs) for imgs in age_bins.values()) if all(age_bins.values()) else 0
print(f"Balancing each age range to {min_samples} images.")

for label, images in age_bins.items():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
    if len(images) >= min_samples and min_samples > 0:
        selected = random.sample(images, min_samples)
        for img_path in tqdm(selected, desc=f"Copying {label}"):
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, label, os.path.basename(img_path)))

print("✅ Preprocessing complete! Balanced dataset ready in:", OUTPUT_DIR)
