import os
import random
import requests
import csv
from datetime import datetime

API_URL = "http://localhost:8000/predict"
TEST_DIR = "data/processed/test"
SAMPLES_PER_CLASS = 10
OUTPUT_CSV = "post_deployment_results.csv"

random.seed(42)  # reproducible sample each run


def sample_images(class_name, n):
    folder = os.path.join(TEST_DIR, class_name)
    files = os.listdir(folder)
    chosen = random.sample(files, min(n, len(files)))
    return [os.path.join(folder, f) for f in chosen]


def send_prediction(filepath):
    with open(filepath, "rb") as f:
        response = requests.post(API_URL, files={"file": (os.path.basename(filepath), f, "image/jpeg")}, timeout=15)
    response.raise_for_status()
    return response.json()


def main():
    samples = []
    for class_name in ["Cat", "Dog"]:
        for path in sample_images(class_name, SAMPLES_PER_CLASS):
            samples.append((path, class_name))

    random.shuffle(samples)  # avoid sending all cats then all dogs in a row

    results = []
    correct = 0

    for filepath, true_label in samples:
        try:
            result = send_prediction(filepath)
            predicted = result["prediction"]
            is_correct = predicted == true_label
            correct += int(is_correct)
            results.append({
                "filepath": filepath,
                "true_label": true_label,
                "predicted": predicted,
                "confidence": result["confidence"],
                "correct": is_correct,
            })
            print(f"{'✓' if is_correct else '✗'} {os.path.basename(filepath)} | "
                  f"true={true_label} predicted={predicted} conf={result['confidence']}")
        except Exception as e:
            print(f"ERROR on {filepath}: {e}")

    accuracy = correct / len(results) if results else 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "true_label", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n--- Post-Deployment Performance Check ---")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Samples tested: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
