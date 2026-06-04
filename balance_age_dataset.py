import os
import shutil
import random

# --------------------------
# CONFIGURATION
# --------------------------
SOURCE_DIRS = [
    "age_dataset/UTKFace",
    "age_dataset/crop_part1",
    "age_dataset/utkface_aligned_cropped/UTKFace",
    "age_dataset/utkface_aligned_cropped/crop_part1"
]
DEST_DIR = "balanced_age_dataset"

# Define age ranges (change as needed)
AGE_RANGES = {
    "0_10": range(0, 11),
    "11_20": range(11, 21),
    "21_30": range(21, 31),
    "31_40": range(31, 41),
    "41_50": range(41, 51),
    "51_100": range(51, 101)
}

os.makedirs(DEST_DIR, exist_ok=True)
for folder in AGE_RANGES.keys():
    os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

def get_age_from_filename(filename):
    """Extract age from UTKFace-style filenames."""
    try:
        return int(filename.split("_")[0])
    except:
        return None

# --------------------------
# Step 1: Collect images by range
# --------------------------
age_group_images = {k: [] for k in AGE_RANGES}

for src_dir in SOURCE_DIRS:
    if not os.path.exists(src_dir):
        continue

    for img_name in os.listdir(src_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        age = get_age_from_filename(img_name)
        if age is None:
            continue

        # Find which range this age belongs to
        for label, age_range in AGE_RANGES.items():
            if age in age_range:
                age_group_images[label].append(os.path.join(src_dir, img_name))
                break

# --------------------------
# Step 2: Balance dataset
# --------------------------
min_count = min(len(imgs) for imgs in age_group_images.values() if len(imgs) > 0)
print(f"\n✅ Minimum count found: {min_count} images per range.\n")

for label, img_list in age_group_images.items():
    print(f"{label}: {len(img_list)} images before balancing")

    random.shuffle(img_list)
    selected_images = img_list[:min_count]
    dest_path = os.path.join(DEST_DIR, label)

    for img_path in selected_images:
        shutil.copy(img_path, dest_path)

print("\n🎉 Dataset successfully balanced and saved to:", DEST_DIR)
