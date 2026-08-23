import os
import shutil
import random
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/PetImages"
OUT_DIR = "data/processed"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

random.seed(42)  # makes the split reproducible every time we re-run this

for category in ["Cat", "Dog"]:
    src_folder = os.path.join(RAW_DIR, category)
    files = os.listdir(src_folder)

    # First split off the training set, then split the remainder in half
    # to get validation and test (this gives us the 80/10/10 ratio).
    train_files, remaining = train_test_split(
        files, train_size=SPLITS["train"], random_state=42
    )
    val_files, test_files = train_test_split(
        remaining, train_size=0.5, random_state=42
    )

    split_map = {"train": train_files, "val": val_files, "test": test_files}

    for split_name, split_files in split_map.items():
        dest_folder = os.path.join(OUT_DIR, split_name, category)
        os.makedirs(dest_folder, exist_ok=True)
        for fname in split_files:
            shutil.copy(
                os.path.join(src_folder, fname),
                os.path.join(dest_folder, fname)
            )

    print(f"{category}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("\nDone splitting data.")