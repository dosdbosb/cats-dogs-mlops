import os

DATA_DIR = "data/raw/PetImages"
removed = 0

for category in ["Cat", "Dog"]:
    folder = os.path.join(DATA_DIR, category)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "rb") as f:
                # Real JPEG files contain the marker "JFIF" near the start
                # of the file. Files that don't have it are the broken/fake
                # ones known to be mixed into this dataset.
                is_jfif = b"JFIF" in f.peek(10)
        except Exception:
            is_jfif = False

        if not is_jfif:
            os.remove(fpath)
            removed += 1
            print(f"Removed corrupt file: {fpath}")

print(f"\nDone. Removed {removed} corrupt files.")