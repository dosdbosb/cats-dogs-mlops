import sys
import time
import io
import requests
from PIL import Image

BASE_URL = "http://localhost:8000"


def make_test_image():
    img = Image.new("RGB", (224, 224), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def check_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    assert data.get("status") == "ok", f"Unexpected health response: {data}"
    assert data.get("model_loaded") is True, "Model not loaded"
    print("Health check passed:", data)


def check_predict():
    files = {"file": ("test.jpg", make_test_image(), "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files, timeout=10)
    r.raise_for_status()
    data = r.json()
    assert "prediction" in data, f"Missing 'prediction' key: {data}"
    assert data["prediction"] in ["Cat", "Dog"], f"Unexpected class: {data}"
    print("Predict check passed:", data)


if __name__ == "__main__":
    # Give the freshly restarted container a moment to finish loading
    # the model before we start hammering it with requests.
    time.sleep(5)
    try:
        check_health()
        check_predict()
        print("\nSmoke tests PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\nSmoke tests FAILED: {e}")
        sys.exit(1)
